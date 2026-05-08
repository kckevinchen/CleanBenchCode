"""Tests for the GPT-family resolver + per-call rotation.

The detection script accepts a comma-separated ``--model`` (or
``llm.model`` config value). The resolver normalises both inputs into
a canonical comma-list spec; the LLM client then rotates the pool
per call so a single unified run distributes its LLM traffic across
the family rather than hammering one deployment.

These tests cover:

1. ``_resolve_model_spec`` - CLI/config precedence, splitting,
   deduping, and the GPT-5 family fallback.
2. ``_run_id_tag_for_model_spec`` - the comma-list -> ``gpt5-family``
   rewrite that keeps run-dir names shell-friendly.
3. ``LLMClient`` rotation - the per-call counter that picks the
   primary model for each query, plus the fallback path that rotates
   to the next family member when the primary is globally 404.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_cleaning_detection import (  # noqa: E402
    _resolve_model_spec,
    _run_id_tag_for_model_spec,
)
from src.core.llm_client import GPT5_FAMILY, LLMClient, LLMResponse  # noqa: E402


# ---------------------------------------------------------------------------
# _resolve_model_spec
# ---------------------------------------------------------------------------


class TestResolveModelSpec:
    def test_explicit_single_cli_wins(self) -> None:
        assert _resolve_model_spec("gpt-5.5", "gpt-5.2") == "gpt-5.5"

    def test_cli_comma_list_normalised(self) -> None:
        # Whitespace trimmed, duplicates removed in first-seen order.
        assert _resolve_model_spec(
            " gpt-5.2 , gpt-5.4 , gpt-5.2 ", None
        ) == "gpt-5.2,gpt-5.4"

    def test_config_string_used_when_cli_blank(self) -> None:
        assert _resolve_model_spec("", "gpt-5.4") == "gpt-5.4"
        assert _resolve_model_spec(None, "gpt-5.4") == "gpt-5.4"

    def test_config_list_value_supported(self) -> None:
        # YAML may load the field as a list rather than a comma string.
        assert _resolve_model_spec(
            None, ["gpt-5.2", "gpt-5.4"]
        ) == "gpt-5.2,gpt-5.4"

    def test_default_is_full_gpt5_family(self) -> None:
        assert _resolve_model_spec(None, None) == ",".join(GPT5_FAMILY)
        assert _resolve_model_spec("", None) == ",".join(GPT5_FAMILY)
        assert _resolve_model_spec("   ", "") == ",".join(GPT5_FAMILY)

    def test_all_empty_tokens_falls_back_to_family(self) -> None:
        assert _resolve_model_spec(", ,  ,", None) == ",".join(GPT5_FAMILY)


# ---------------------------------------------------------------------------
# _run_id_tag_for_model_spec
# ---------------------------------------------------------------------------


class TestRunIdTagForModelSpec:
    def test_single_model_passes_through(self) -> None:
        assert _run_id_tag_for_model_spec("gpt-5.2") == "gpt-5.2"

    def test_family_collapses_to_tag(self) -> None:
        # A comma-list must NOT appear verbatim in run-dir names
        # (commas in paths are awkward) - fold to a stable label.
        assert _run_id_tag_for_model_spec(
            "gpt-5.2,gpt-5.4,gpt-5.5"
        ) == "gpt5-family"

    def test_two_models_also_tagged(self) -> None:
        assert _run_id_tag_for_model_spec("gpt-5.2,gpt-5.4") == "gpt5-family"

    def test_empty_returns_empty(self) -> None:
        assert _run_id_tag_for_model_spec("") == ""


# ---------------------------------------------------------------------------
# LLMClient rotation
# ---------------------------------------------------------------------------


def _make_client_with_stub_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seen: List[Tuple[str, str]],
    failing_model: Optional[str] = None,
) -> LLMClient:
    """Build a real LLMClient but bypass __post_init__ + _do_call.

    ``seen`` records ``(endpoint, model)`` tuples in call order so
    tests can assert on the rotation. Setting ``failing_model``
    makes any call against that model raise a 404-shaped exception
    so we can exercise the cross-model fallback path. Endpoint
    state (clients list, locks, dead-combo set) is hand-wired to
    avoid touching Azure SDK or env vars.
    """
    client = LLMClient.__new__(LLMClient)
    client.endpoints = ["ep0", "ep1"]
    client.max_retries = 1
    client.backoff_base_seconds = 0.0
    client.backoff_cap_seconds = 0.0
    client.request_timeout_seconds = 1.0
    client.api_version = "test"
    client.token_scope = "test"
    client._clients = ["client0", "client1"]  # type: ignore[assignment]
    from threading import Lock
    client._index = 0
    client._index_lock = Lock()
    client._model_index = 0
    client._model_index_lock = Lock()
    client._dead_combos = set()
    client._dead_lock = Lock()

    def fake_do_call(
        self,  # bound-method first arg when patched on the class
        _client_obj,
        endpoint: str,
        called_model: str,
        _prompt: str,
        **_kwargs,
    ) -> LLMResponse:
        seen.append((endpoint, called_model))
        if failing_model is not None and called_model == failing_model:
            class FakeNotFound(Exception):
                status_code = 404
            raise FakeNotFound("DeploymentNotFound")
        return LLMResponse(
            text="ok", endpoint=endpoint, model=called_model, attempts=1,
        )

    monkeypatch.setattr(LLMClient, "_do_call", fake_do_call)
    return client


class TestLLMClientRotation:
    def test_single_model_string_calls_that_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: List[Tuple[str, str]] = []
        client = _make_client_with_stub_call(monkeypatch, seen=seen)
        for _ in range(3):
            client.query("gpt-5.2", "hi")
        # Only one model in the pool -> all three calls go there.
        assert all(m == "gpt-5.2" for _, m in seen)

    def test_comma_list_rotates_round_robin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: List[Tuple[str, str]] = []
        client = _make_client_with_stub_call(monkeypatch, seen=seen)
        for _ in range(6):
            client.query("gpt-5.2,gpt-5.4,gpt-5.5", "hi")
        # Six calls / three models -> exactly two per model.
        per_model: dict[str, int] = {}
        for _, m in seen:
            per_model[m] = per_model.get(m, 0) + 1
        assert per_model == {"gpt-5.2": 2, "gpt-5.4": 2, "gpt-5.5": 2}
        # And the order must actually rotate, not arbitrarily group.
        # First three calls hit three distinct models.
        assert len({m for _, m in seen[:3]}) == 3

    def test_pool_dedupes_repeats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``"gpt-5.2,gpt-5.2"`` should behave like a single-model pool.
        seen: List[Tuple[str, str]] = []
        client = _make_client_with_stub_call(monkeypatch, seen=seen)
        for _ in range(2):
            client.query("gpt-5.2,gpt-5.2", "hi")
        assert all(m == "gpt-5.2" for _, m in seen)
        assert len(seen) == 2

    def test_empty_spec_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Silently defaulting an empty spec would mask config bugs in
        # callers that compute the model string dynamically.
        client = _make_client_with_stub_call(monkeypatch, seen=[])
        with pytest.raises(ValueError, match="empty"):
            client.query("  , , ", "hi")

    def test_404_on_primary_falls_through_to_next_family_member(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``gpt-5.4`` is globally unavailable. With pool gpt-5.2 first,
        # then gpt-5.4, then gpt-5.5: when rotation lands on gpt-5.4
        # all endpoints 404 -> we should NOT return an error response;
        # instead, fall through to gpt-5.5.
        seen: List[Tuple[str, str]] = []
        client = _make_client_with_stub_call(
            monkeypatch, seen=seen, failing_model="gpt-5.4",
        )
        # Force the rotation to start at gpt-5.4 by pre-advancing the
        # model counter (so this test isn't flaky on counter init).
        client._model_index = 1  # type: ignore[attr-defined]
        resp = client.query("gpt-5.2,gpt-5.4,gpt-5.5", "hi")
        assert resp.error is None
        # The successful response was served by gpt-5.5 (next in the
        # rotated pool after gpt-5.4 fully 404'd).
        assert resp.model == "gpt-5.5"
        # The 404 attempts on gpt-5.4 are recorded (one per endpoint),
        # then the successful gpt-5.5 call.
        attempted_models = [m for _, m in seen]
        assert "gpt-5.4" in attempted_models  # tried it
        assert attempted_models[-1] == "gpt-5.5"  # succeeded after fallback

    def test_concurrent_calls_spread_across_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate concurrent calls (sequential is fine - the lock
        # guarantees a monotonic sequence regardless of thread mix).
        # 9 calls into a 3-model pool should give 3 per model.
        seen: List[Tuple[str, str]] = []
        client = _make_client_with_stub_call(monkeypatch, seen=seen)
        for _ in range(9):
            client.query(",".join(GPT5_FAMILY), "hi")
        per_model: dict[str, int] = {}
        for _, m in seen:
            per_model[m] = per_model.get(m, 0) + 1
        for m in GPT5_FAMILY:
            assert per_model[m] == 3, f"unbalanced rotation: {per_model}"
