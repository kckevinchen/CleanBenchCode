"""Tests for ``src.core.llm_client.LLMClient``.

Pinned by Pass 6c HIGH coverage gap. Covers:

* endpoint validation (env override, empty list rejected)
* round-robin client rotation (thread-safe)
* responses-API model routing (e.g. ``gpt-5.2``) vs chat completions
* successful query returns text, endpoint, attempts
* retry loop on transient exceptions, eventually succeeding
* retry loop exhaustion → returns LLMResponse with ``error`` set
* system_prompt routed via ``instructions=`` for Responses API
* reasoning_summary extraction from Responses payload

All Azure SDK surfaces are mocked. ``time.sleep`` is patched out so
the retry-backoff path doesn't actually wait.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from src.core import llm_client as llm_module
from src.core.llm_client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Fixtures: patch out AzureCliCredential + AzureOpenAI before construction
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_azure(monkeypatch):
    """Replace AzureCliCredential / get_bearer_token_provider / AzureOpenAI
    so :class:`LLMClient` constructs without touching Azure or env tokens."""
    monkeypatch.setattr(llm_module, "AzureCliCredential", MagicMock())
    monkeypatch.setattr(llm_module, "get_bearer_token_provider", MagicMock(return_value=MagicMock()))
    # Each call to AzureOpenAI(...) returns a fresh MagicMock so endpoints
    # get distinct client instances.
    monkeypatch.setattr(
        llm_module, "AzureOpenAI",
        MagicMock(side_effect=lambda **kwargs: MagicMock(name="AzureOpenAIClient")),
    )
    # Suppress the actual sleep in retry loop so tests don't take 10s+.
    monkeypatch.setattr(llm_module.time, "sleep", lambda *a, **kw: None)
    # Clear any AZURE_OPENAI_ENDPOINTS that might be present in the
    # CI environment — we want full control over .endpoints.
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINTS", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLLMClientConstruction:
    def test_missing_endpoints_raises(self) -> None:
        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINTS"):
            LLMClient()

    def test_explicit_endpoints_override_default(self) -> None:
        c = LLMClient(endpoints=["http://a", "http://b"])
        assert c.endpoints == ["http://a", "http://b"]
        assert len(c._clients) == 2

    def test_env_var_overrides_endpoints(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINTS",
            "http://e1, http://e2 , http://e3",
        )
        c = LLMClient()
        assert c.endpoints == ["http://e1", "http://e2", "http://e3"]

    def test_singular_env_var_supplies_endpoint(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://single")
        c = LLMClient()
        assert c.endpoints == ["http://single"]

    def test_empty_endpoints_via_env_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINTS", "  ,  ")
        with pytest.raises(ValueError, match="at least one endpoint"):
            LLMClient()

    def test_empty_endpoints_explicit_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one endpoint"):
            LLMClient(endpoints=[])


# ---------------------------------------------------------------------------
# Round-robin
# ---------------------------------------------------------------------------


class TestRoundRobin:
    def test_clients_rotated_in_order(self) -> None:
        c = LLMClient(endpoints=["http://a", "http://b", "http://c"])
        cycle = [c._next_client() for _ in range(7)]
        endpoints = [pair[1] for pair in cycle]
        assert endpoints == [
            "http://a", "http://b", "http://c",
            "http://a", "http://b", "http://c",
            "http://a",
        ]


# ---------------------------------------------------------------------------
# Query — Chat Completions path
# ---------------------------------------------------------------------------


def _stub_chat_response(content: str) -> Any:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    result = MagicMock()
    result.choices = [choice]
    return result


class TestQueryChatCompletions:
    def test_successful_call_returns_text(self) -> None:
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].chat.completions.create.return_value = _stub_chat_response(
            "hello world"
        )
        resp = c.query("gpt-4o", "say hi")
        assert resp.ok
        assert resp.text == "hello world"
        assert resp.endpoint == "http://a"
        assert resp.attempts == 1
        assert resp.error is None

    def test_system_prompt_passed_through(self) -> None:
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].chat.completions.create.return_value = _stub_chat_response("ok")
        c.query("gpt-4o", "user-text", system_prompt="be terse")
        kwargs = c._clients[0].chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        msgs = kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "be terse"}
        assert msgs[1] == {"role": "user", "content": "user-text"}

    def test_retry_recovers_after_transient_failure(self) -> None:
        c = LLMClient(
            endpoints=["http://a", "http://b"], max_retries=3,
            backoff_base_seconds=0.0,
        )
        # Round-robin advances on every call, so a failure on ``a``
        # naturally rolls the next attempt to ``b`` -- which is
        # exactly what we want under transient failure (give a
        # different endpoint a chance to recover).
        #   * endpoint a: first call raises
        #   * endpoint b: succeeds on the second attempt
        c._clients[0].chat.completions.create.side_effect = (
            RuntimeError("boom")
        )
        c._clients[1].chat.completions.create.side_effect = [
            _stub_chat_response("recovered"),
        ]
        resp = c.query("gpt-4o", "x")
        assert resp.ok
        assert resp.text == "recovered"
        assert resp.attempts == 2
        assert resp.endpoint == "http://b"
        c._clients[1].chat.completions.create.assert_called_once()

    def test_retry_exhausted_returns_error_response(self) -> None:
        c = LLMClient(
            endpoints=["http://a"], max_retries=2, backoff_base_seconds=0.0,
        )
        c._clients[0].chat.completions.create.side_effect = RuntimeError("dead")
        resp = c.query("gpt-4o", "x")
        assert not resp.ok
        assert resp.text == ""
        assert resp.error and "RuntimeError" in resp.error
        assert resp.attempts == 2  # capped by max_retries


# ---------------------------------------------------------------------------
# Query — Responses API path
# ---------------------------------------------------------------------------


def _stub_responses_payload(text: str, summary_lines: List[str] | None = None) -> Any:
    """Construct a mock Responses API result that looks like the SDK's."""
    result = MagicMock()
    result.output_text = text
    payload = {
        "output": [
            {
                "type": "reasoning",
                "summary": [{"text": s} for s in (summary_lines or [])],
            }
        ],
        "reasoning": {"summary": "auto"},  # ignored sentinel value
    }
    # Mock _safe_model_dump path: ``model_dump()`` returns the dict.
    result.model_dump = MagicMock(return_value=payload)
    return result


class TestQueryResponsesApi:
    def test_responses_api_triggered_for_known_models(self) -> None:
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].responses.create.return_value = _stub_responses_payload(
            "the answer", summary_lines=["thinking..."]
        )
        resp = c.query("gpt-5.2", "x")
        assert resp.ok
        assert resp.text == "the answer"
        assert resp.reasoning_summary == "thinking..."
        # Chat completions endpoint MUST NOT have been called.
        c._clients[0].chat.completions.create.assert_not_called()

    def test_system_prompt_becomes_instructions(self) -> None:
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].responses.create.return_value = _stub_responses_payload("ok")
        c.query("gpt-5.2", "user-text", system_prompt="be precise")
        kwargs = c._clients[0].responses.create.call_args.kwargs
        assert kwargs["model"] == "gpt-5.2"
        assert kwargs["input"] == "user-text"
        assert kwargs["instructions"] == "be precise"
        # reasoning kwargs forwarded.
        assert "reasoning" in kwargs

    def test_responses_api_case_insensitive_match(self) -> None:
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].responses.create.return_value = _stub_responses_payload("ok")
        c.query("GPT-5.2", "x")
        c._clients[0].responses.create.assert_called_once()
        c._clients[0].chat.completions.create.assert_not_called()

    def test_gpt_5_4_and_5_5_route_to_responses_api(self) -> None:
        # Newer GPT-5 family members must also flow through the
        # Responses API; otherwise Chat Completions would silently
        # drop ``reasoning_effort`` / ``instructions``.
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].responses.create.return_value = _stub_responses_payload("ok")
        for model in ("gpt-5.4", "gpt-5.5"):
            c._clients[0].responses.create.reset_mock()
            c._clients[0].chat.completions.create.reset_mock()
            c.query(model, "x")
            c._clients[0].responses.create.assert_called_once()
            c._clients[0].chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# 404 handling: per-(endpoint, model) dead-combo tracking
# ---------------------------------------------------------------------------


def _stub_404_error(message: str = "DeploymentNotFound") -> Exception:
    """Build an exception that the 404 sniffer will recognise.

    Mirrors what the OpenAI SDK raises for missing deployments: an
    exception with ``status_code == 404`` (the wrapper class name
    happens to be ``NotFoundError`` in newer SDKs, but the sniffer
    primarily keys off the status code so older / wrapped versions
    still work).
    """
    err = RuntimeError(message)
    err.status_code = 404  # type: ignore[attr-defined]
    return err


class TestFourOhFourSkip:
    def test_404_marks_combo_dead_and_falls_through(self) -> None:
        c = LLMClient(
            endpoints=["http://a", "http://b"],
            max_retries=2, backoff_base_seconds=0.0,
        )
        # Endpoint a doesn't carry the model; b does.
        c._clients[0].chat.completions.create.side_effect = _stub_404_error()
        c._clients[1].chat.completions.create.return_value = _stub_chat_response(
            "answered by b"
        )
        resp = c.query("gpt-4o", "x")
        assert resp.ok
        assert resp.endpoint == "http://b"
        assert resp.text == "answered by b"
        # The dead set must record (a, gpt-4o) so subsequent calls skip
        # endpoint a entirely.
        assert c.is_dead("http://a", "gpt-4o")
        assert not c.is_dead("http://b", "gpt-4o")

    def test_404_does_not_consume_retry_budget(self) -> None:
        # max_retries=1: a single transient failure would exhaust the
        # budget. But a 404 is a routing decision, not a transient
        # error - it must NOT count against the budget, so we still
        # have all our transient retries available against the
        # surviving endpoints.
        c = LLMClient(
            endpoints=["http://a", "http://b"],
            max_retries=1, backoff_base_seconds=0.0,
        )
        c._clients[0].chat.completions.create.side_effect = _stub_404_error()
        c._clients[1].chat.completions.create.return_value = _stub_chat_response(
            "ok via b"
        )
        resp = c.query("gpt-4o", "x")
        assert resp.ok
        assert resp.text == "ok via b"
        # Dead combo recorded; budget never decremented because no
        # transient error occurred.
        assert c.is_dead("http://a", "gpt-4o")

    def test_all_endpoints_404_returns_error_response(self) -> None:
        c = LLMClient(
            endpoints=["http://a", "http://b"],
            max_retries=2, backoff_base_seconds=0.0,
        )
        c._clients[0].chat.completions.create.side_effect = _stub_404_error()
        c._clients[1].chat.completions.create.side_effect = _stub_404_error()
        resp = c.query("phantom-model", "x")
        assert not resp.ok
        assert resp.error is not None
        assert "404" in resp.error
        # Both combos marked dead.
        assert c.is_dead("http://a", "phantom-model")
        assert c.is_dead("http://b", "phantom-model")

    def test_dead_combo_skipped_on_subsequent_calls(self) -> None:
        # First call 404s on a, succeeds on b. Second call must NOT
        # re-probe a - it should go straight to b.
        c = LLMClient(
            endpoints=["http://a", "http://b"],
            max_retries=2, backoff_base_seconds=0.0,
        )
        c._clients[0].chat.completions.create.side_effect = _stub_404_error()
        c._clients[1].chat.completions.create.return_value = _stub_chat_response(
            "b"
        )
        c.query("gpt-4o", "first")
        # Reset call counts so we can observe the second call cleanly.
        c._clients[0].chat.completions.create.reset_mock()
        c._clients[1].chat.completions.create.reset_mock()
        c._clients[1].chat.completions.create.return_value = _stub_chat_response(
            "b again"
        )
        resp = c.query("gpt-4o", "second")
        assert resp.ok
        assert resp.endpoint == "http://b"
        # Endpoint a was skipped without a probe call.
        c._clients[0].chat.completions.create.assert_not_called()
        c._clients[1].chat.completions.create.assert_called_once()

    def test_dead_combo_is_per_model_not_per_endpoint(self) -> None:
        # ``gpt-foo`` 404s on a, but a still carries other models -
        # so ``gpt-bar`` should still try a.
        c = LLMClient(
            endpoints=["http://a"],
            max_retries=2, backoff_base_seconds=0.0,
        )
        c._clients[0].chat.completions.create.side_effect = [
            _stub_404_error(),                     # gpt-foo 404
            _stub_chat_response("bar works"),      # gpt-bar succeeds
        ]
        c.query("gpt-foo", "x")
        assert c.is_dead("http://a", "gpt-foo")
        # gpt-bar must NOT inherit the dead marker.
        assert not c.is_dead("http://a", "gpt-bar")
        resp = c.query("gpt-bar", "x")
        assert resp.ok
        assert resp.text == "bar works"

    def test_transient_then_404_then_success(self) -> None:
        # First a transient fails on a (consumes 1 retry).
        # Then b 404s (no budget consumed).
        # Then c succeeds.
        c = LLMClient(
            endpoints=["http://a", "http://b", "http://c"],
            max_retries=3, backoff_base_seconds=0.0,
        )
        c._clients[0].chat.completions.create.side_effect = RuntimeError("boom")
        c._clients[1].chat.completions.create.side_effect = _stub_404_error()
        c._clients[2].chat.completions.create.return_value = _stub_chat_response(
            "c wins"
        )
        resp = c.query("gpt-4o", "x")
        assert resp.ok
        assert resp.text == "c wins"
        assert resp.endpoint == "http://c"
        # Only the RuntimeError counts toward attempts; the 404 is
        # routing.
        assert resp.attempts == 2
        assert c.is_dead("http://b", "gpt-4o")
        assert not c.is_dead("http://a", "gpt-4o")


# ---------------------------------------------------------------------------
# probe_availability matrix
# ---------------------------------------------------------------------------


class TestReasoningEffortCoercion:
    """``"minimal"`` is documented in some places but rejected by the
    deployed GPT-5 family with HTTP 400. The client coerces it to
    ``"low"`` at the boundary so a stale config doesn't burn the
    retry budget on every call."""

    @pytest.fixture(autouse=True)
    def _clear_warning_state(self) -> None:
        # The warn-once dedupe set is a module-global. Clear it
        # before each test so the warning behaviour is observable
        # independently per case.
        llm_module._warned_effort_rewrites.clear()

    def test_minimal_rewritten_to_low_for_responses_api(self) -> None:
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].responses.create.return_value = _stub_responses_payload("ok")
        c.query("gpt-5.2", "x", reasoning_effort="minimal")
        kwargs = c._clients[0].responses.create.call_args.kwargs
        assert kwargs["reasoning"]["effort"] == "low"

    def test_supported_efforts_pass_through_unchanged(self) -> None:
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].responses.create.return_value = _stub_responses_payload("ok")
        for effort in ("none", "low", "medium", "high", "xhigh"):
            c._clients[0].responses.create.reset_mock()
            c._clients[0].responses.create.return_value = _stub_responses_payload(
                "ok"
            )
            c.query("gpt-5.2", "x", reasoning_effort=effort)
            kwargs = c._clients[0].responses.create.call_args.kwargs
            assert kwargs["reasoning"]["effort"] == effort

    def test_warning_logged_once_per_model_value_pair(self, caplog) -> None:
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].responses.create.return_value = _stub_responses_payload("ok")
        with caplog.at_level("WARNING", logger=llm_module.logger.name):
            c.query("gpt-5.2", "x", reasoning_effort="minimal")
            c.query("gpt-5.2", "y", reasoning_effort="minimal")
            c.query("gpt-5.2", "z", reasoning_effort="minimal")
        rewrite_warnings = [
            r for r in caplog.records
            if "Rewriting reasoning_effort" in r.getMessage()
        ]
        # Three calls, same (model, effort) pair -> exactly one warning.
        assert len(rewrite_warnings) == 1

    def test_chat_completions_path_does_not_touch_effort(self) -> None:
        # Chat Completions doesn't accept ``reasoning_effort`` at all
        # (it's a Responses-API parameter); the coercion only lives on
        # the Responses path. Sanity-check that chat completions
        # ignores ``reasoning_effort`` entirely - if the kwarg ever
        # leaks into the chat call, this test will catch it.
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].chat.completions.create.return_value = _stub_chat_response(
            "ok"
        )
        c.query("gpt-4o", "x", reasoning_effort="minimal")
        kwargs = c._clients[0].chat.completions.create.call_args.kwargs
        assert "reasoning" not in kwargs
        assert "reasoning_effort" not in kwargs


class TestProbeAvailability:
    def test_probe_returns_status_per_combo(self) -> None:
        c = LLMClient(endpoints=["http://a", "http://b"])
        # a: gpt-foo OK, gpt-bar 404
        # b: gpt-foo 404, gpt-bar OK
        def a_side(model: str, **kwargs):
            if model == "gpt-foo":
                return _stub_chat_response("a-foo")
            raise _stub_404_error()

        def b_side(model: str, **kwargs):
            if model == "gpt-bar":
                return _stub_chat_response("b-bar")
            raise _stub_404_error()

        c._clients[0].chat.completions.create.side_effect = lambda **kw: a_side(**kw)
        c._clients[1].chat.completions.create.side_effect = lambda **kw: b_side(**kw)

        matrix = c.probe_availability(["gpt-foo", "gpt-bar"], prompt="ping")
        assert matrix == {
            "http://a": {"gpt-foo": "ok", "gpt-bar": "404"},
            "http://b": {"gpt-foo": "404", "gpt-bar": "ok"},
        }
        # Probe-time 404s should populate the dead set so a follow-up
        # query() doesn't re-probe them.
        assert c.is_dead("http://a", "gpt-bar")
        assert c.is_dead("http://b", "gpt-foo")

    def test_probe_records_non_404_errors_by_class_name(self) -> None:
        c = LLMClient(endpoints=["http://a"])
        c._clients[0].chat.completions.create.side_effect = TimeoutError("slow")
        matrix = c.probe_availability(["gpt-foo"])
        assert matrix == {"http://a": {"gpt-foo": "TimeoutError"}}
        # Timeout is transient, NOT a 404 - the combo must remain
        # available for subsequent retries.
        assert not c.is_dead("http://a", "gpt-foo")

    def test_probe_no_retries_one_call_per_combo(self) -> None:
        # Probe is a snapshot, not a recovery attempt - exactly one
        # call per (endpoint, model) cell.
        c = LLMClient(endpoints=["http://a"], max_retries=5)
        c._clients[0].chat.completions.create.side_effect = RuntimeError("flaky")
        c.probe_availability(["gpt-foo", "gpt-bar"])
        assert c._clients[0].chat.completions.create.call_count == 2


def test_extract_output_text_skips_reasoning_item():
    """Regression: Responses output[0] is often a reasoning item with no
    .content; _extract_output_text must find the message item's text
    rather than IndexError or return the output list's repr."""
    from types import SimpleNamespace
    reasoning = SimpleNamespace(content=None)
    message = SimpleNamespace(content=[SimpleNamespace(text="the answer")])
    result = SimpleNamespace(output_text="", output=[reasoning, message])
    assert LLMClient._extract_output_text(result) == "the answer"


def test_extract_output_text_empty_on_reasoning_only():
    from types import SimpleNamespace
    reasoning = SimpleNamespace(content=None)
    result = SimpleNamespace(output_text="", output=[reasoning])
    assert LLMClient._extract_output_text(result) == ""
