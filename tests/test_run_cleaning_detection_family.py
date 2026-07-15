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
    DEDICATED_HANDLER_TYPES,
    _resolve_model_spec,
    _run_id_tag_for_model_spec,
    _scope_of,
    main,
)
from src.core.error_taxonomy import load_taxonomy  # noqa: E402
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


# ---------------------------------------------------------------------------
# inter_column_pipeline routing
# ---------------------------------------------------------------------------


def _routed_away(type_keys: List[str]) -> List[str]:
    """Replicate main()'s inter_column routing predicate for a selection.

    Returns the subset of ``type_keys`` that the routing strips from the
    active error-type set (delegating them to the relation-discovery
    pipeline). The remainder stays in ``error_types`` so this runner's own
    detectors can handle them.
    """
    tax = load_taxonomy(PROJECT_ROOT / "prompts" / "error_types.yaml")
    selected = [et for et in tax.types if et.key in set(type_keys)]
    return [
        et.key
        for et in selected
        if _scope_of(et.category) != "single-column"
        and et.key not in DEDICATED_HANDLER_TYPES
    ]


class TestInterColumnRouting:
    """Guards the fix for the row_duplication-only crash: the dedicated
    in-process handler types must survive the inter_column routing so their
    dispatch (``wants_row_dup`` / ``multi_table_enabled``) still fires and
    ``error_types`` never empties out underneath the adaptive-mode detector.
    """

    def test_row_duplication_not_routed_away(self) -> None:
        # The regression: selecting only row_duplication must NOT empty the
        # active error-type set (which crashed PerTypeCleaningDetector).
        assert _routed_away(["row_duplication"]) == []

    def test_inter_table_inconsistency_not_routed_away(self) -> None:
        assert _routed_away(["inter_table_inconsistency"]) == []

    def test_multi_column_scope_row_types_still_routed(self) -> None:
        # The three scope:row types have no in-process handler here; they
        # must still be delegated to run_relation_discovery.
        relation_types = [
            "fd_violation",
            "row_calculation_inconsistency",
            "string_calculation_inconsistency",
        ]
        assert set(_routed_away(relation_types)) == set(relation_types)

    def test_single_column_types_never_routed(self) -> None:
        assert _routed_away(
            ["typo", "inconsistency", "semantic_incompatible", "numeric_outlier"]
        ) == []

    def test_dedicated_handler_types_are_non_single_column(self) -> None:
        # Each dedicated type must actually be a non-single-column type --
        # otherwise the exclusion is meaningless (it would never be routed
        # anyway) and a refactor that renames scopes would slip past.
        tax = load_taxonomy(PROJECT_ROOT / "prompts" / "error_types.yaml")
        by_key = {et.key: et for et in tax.types}
        for key in DEDICATED_HANDLER_TYPES:
            assert key in by_key, f"{key} missing from taxonomy"
            assert _scope_of(by_key[key].category) != "single-column"


# ---------------------------------------------------------------------------
# Adaptive mode: relation-only selection must not crash (regression)
# ---------------------------------------------------------------------------


class TestAdaptiveRelationOnlyNoCrash:
    """Selecting only relation-pipeline types (e.g. ``fd_violation``) in a
    detection run routes every type to the relation pipeline, emptying
    ``error_types``. Adaptive mode then runs zero passes and previously
    crashed building a throwaway ``PerTypeCleaningDetector`` solely to sniff
    its concurrency ("needs at least one error type"). The run must instead
    complete, write the standard (empty) artifacts, and rely on the
    operator-facing warning that points at ``run_relation_discovery``. Makes
    zero LLM calls (zero passes), so it needs no Azure credentials.
    """

    def test_fd_violation_only_adaptive_completes(self, tmp_path: Path) -> None:
        csv = tmp_path / "fd.csv"
        csv.write_text(
            "id,city,country\n1,Paris,France\n2,Paris,Germany\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        rc = main(
            [
                "--csv", str(csv),
                "--error-types", "fd_violation",
                "--scopes", "all",
                "--mode", "adaptive",
                "--max-chunks", "1",
                "--output-dir", str(out_dir),
            ]
        )
        assert rc == 0
        # Exactly one run dir with the standard summary artifact.
        run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "summary.json").is_file()
