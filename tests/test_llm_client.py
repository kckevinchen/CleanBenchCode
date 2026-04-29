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


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLLMClientConstruction:
    def test_default_endpoints_used(self) -> None:
        c = LLMClient()
        assert c.endpoints == llm_module.DEFAULT_ENDPOINTS
        assert len(c._clients) == len(llm_module.DEFAULT_ENDPOINTS)

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
        # Both client mocks: first 2 calls (across the round-robin)
        # raise; 3rd call succeeds.
        c._clients[0].chat.completions.create.side_effect = [
            RuntimeError("boom"),
            _stub_chat_response("recovered"),
        ]
        c._clients[1].chat.completions.create.side_effect = [
            RuntimeError("also boom"),
        ]
        resp = c.query("gpt-4o", "x")
        assert resp.ok
        assert resp.text == "recovered"
        assert resp.attempts == 3
        # Endpoint reflects the one that finally succeeded — index 0
        # again on the third round-robin step.
        assert resp.endpoint == "http://a"

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
