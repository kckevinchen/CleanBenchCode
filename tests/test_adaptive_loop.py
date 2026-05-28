"""Integration tests for ``experiments.run_cleaning_detection._run_adaptive_loop``.

Pinned by Pass 6c as the last HIGH coverage gap. Uses a stub
:class:`LLMClient` (returns canned JSON) so the loop's full
perturbation → detection → Good-Turing → stop-policy chain runs
end-to-end without network. Coverage:

* fixed mode runs exactly ``num_runs`` passes
* adaptive overall mode early-stops once GT missing mass <= target
  (and respects the ``min_runs`` floor)
* per-type policy saturates types individually and prunes them from
  later passes (``runs_probed_per_type`` reflects pruning)
* skip_identity_run shifts run_id offset (no run_id=0 baseline)
* prompt-ensemble path: variant generator invoked only on non-identity
  passes, variant_records_per_run populated, save_variant_records is
  called for each pass
* loop returns artefacts shaped as expected
* ``_resolve_adaptive_settings`` validates input ranges
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from experiments.run_cleaning_detection import (
    _auto_stop_scope_for_types,
    _resolve_adaptive_settings,
    _run_adaptive_loop,
)
from src.adaptive.scoped_gt import ArmConfig
from src.core.error_taxonomy import ErrorTaxonomy, ErrorType
from src.core.llm_client import LLMResponse


# ---------------------------------------------------------------------------
# Stubs / fixtures
# ---------------------------------------------------------------------------


class _CannedLLM:
    """Deterministic stub mirroring the public surface used by the
    detector + variant generator: ``.endpoints`` and ``.query(...)``."""

    def __init__(self, *, payload: str = '[{"row":0,"columns":["c0"],"observed":"x","explanation":"e"}]'):
        self.endpoints = ["http://stub-1"]
        self._payload = payload
        self.detector_calls = 0
        self.variant_calls = 0

    def query(self, model, prompt, *, reasoning_effort, reasoning_summary,
              system_prompt):
        # Variant-generator system prompt contains the substring
        # "rewrite" in our test template; detector's contains "SYS-".
        if system_prompt and "rewrite" in system_prompt.lower():
            self.variant_calls += 1
            body = json.dumps({
                "description": "rewritten",
                "notes": "rewritten-notes",
                "example": "rewritten-ex",
            })
            return LLMResponse(
                text=body, endpoint=self.endpoints[0],
                model=model, attempts=1,
            )
        self.detector_calls += 1
        return LLMResponse(
            text=self._payload, endpoint=self.endpoints[0],
            model=model, attempts=1,
        )


def _seed_types() -> List[ErrorType]:
    return [
        ErrorType(key="typo", name="Typo", category="Cat-A",
                   description="d-typo", example="ex-typo", notes="n-typo"),
        ErrorType(key="casing", name="Casing", category="Cat-A",
                   description="d-cas", example="ex-cas", notes="n-cas"),
    ]


def _taxonomy(types: List[ErrorType]) -> ErrorTaxonomy:
    return ErrorTaxonomy(
        system_prompt="SYS-prompt",
        user_template=(
            "TYPE={type_key} NAME={type_name} CAT={type_category}\n"
            "DESC={type_description}\nEX={type_example}\nNOTES={type_notes}\n"
            "HEADERS={headers}\nROWS:\n{rows}\nRANGE={start_row}..{end_row}"
        ),
        types=tuple(types),
    )


def _sources() -> Dict[str, pd.DataFrame]:
    return {
        "src_a.csv": pd.DataFrame({"c0": [f"a{i}" for i in range(4)]}),
    }


def _settings(**overrides) -> Dict[str, Any]:
    target = overrides.get("target_missing_mass", 1.0)
    base: Dict[str, Any] = {
        "gt_mode": "fixed",
        "num_runs": 2,
        "min_runs": 1,
        "target_missing_mass": 1.0,
        # Unified per-arm stop. ``stop_scope=dataset`` gives one arm
        # per error_type (the old "per_type" semantics). The legacy
        # ``target_policy`` alias is kept on the payload only for
        # back-compat consumers (report_template etc.).
        "stop_scope": "dataset",
        "target_policy": "per_type",  # back-compat alias
        # Permissive per-arm config so the legacy
        # ``target_missing_mass`` knob alone drives saturation in
        # tests: no two-consecutive requirement, K floor at 1, and
        # epsilon plumbed in from target_missing_mass so any p_miss
        # at/below the target instantly saturates the arm.
        "per_arm": (
            ArmConfig(
                epsilon=target,
                k_min=1,
                k_max=10000,
                require_two_consecutive=False,
            ),
            {},
            None,
        ),
        "base_seed": 42,
        "shuffle_rows": True,
        "shuffle_columns": False,
        "skip_identity_run": False,
        "prompt_ensemble": False,
    }
    base.update(overrides)
    # Re-derive per_arm.epsilon if target_missing_mass overridden but
    # per_arm not explicitly supplied.
    if (
        "target_missing_mass" in overrides
        and "per_arm" not in overrides
    ):
        base["per_arm"] = (
            ArmConfig(
                epsilon=base["target_missing_mass"],
                k_min=1, k_max=10000,
                require_two_consecutive=False,
            ),
            {},
            None,
        )
    return base


def _run(settings, *, llm=None, taxonomy=None, types=None, frames=None,
          variant_generator=None, run_dir=None):
    llm = llm or _CannedLLM()
    types = types or _seed_types()
    taxonomy = taxonomy or _taxonomy(types)
    frames = frames or _sources()
    return _run_adaptive_loop(
        llm=llm,
        model="stub-model",
        base_taxonomy=taxonomy,
        seed_error_types=types,
        reasoning_effort="low",
        reasoning_summary="detailed",
        concurrency=1,
        source_frames=frames,
        chunk_size=4,
        max_chunks=None,
        start_row=0,
        column_mode="all",
        active_columns=None,
        enable_synthesis=False,
        settings=settings,
        variant_generator=variant_generator,
        run_dir=run_dir or Path("/tmp/_unused_adaptive_run_dir"),
    )


# ---------------------------------------------------------------------------
# _resolve_adaptive_settings
# ---------------------------------------------------------------------------


def _ns(**kwargs) -> argparse.Namespace:
    """argparse.Namespace with all gt_* attributes nullable by default."""
    base = dict(
        gt_mode=None, gt_num_runs=None, gt_min_runs=None,
        gt_target_missing_mass=None, gt_target_policy=None,
        gt_stop_scope=None,
        gt_base_seed=None, gt_shuffle_rows=None, gt_shuffle_columns=None,
        gt_skip_identity_run=None, gt_prompt_ensemble=None,
        gt_variant_template=None, gt_variant_effort=None,
        gt_per_arm_epsilon=None,
        gt_per_arm_k_max=None,
        gt_per_arm_two_consecutive=None,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


class TestResolveAdaptiveSettings:
    def test_defaults_when_no_overrides(self) -> None:
        s = _resolve_adaptive_settings(_ns(), config={})
        assert s["gt_mode"] == "adaptive"
        assert s["num_runs"] == 5
        assert s["min_runs"] == 3
        assert s["target_missing_mass"] == 0.05
        # New canonical knob; default is now "auto" (resolved to a
        # concrete scope in ``main`` once the active error-type set is
        # known).
        assert s["stop_scope"] == "auto"
        # Back-compat alias is preserved on the payload.
        assert "target_policy" in s

    def test_cli_wins_over_yaml(self) -> None:
        s = _resolve_adaptive_settings(
            _ns(gt_mode="fixed", gt_num_runs=10),
            config={"good_turing": {"mode": "adaptive", "num_runs": 3}},
        )
        assert s["gt_mode"] == "fixed"
        assert s["num_runs"] == 10

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 'fixed' or 'adaptive'"):
            _resolve_adaptive_settings(_ns(gt_mode="bogus"), config={})

    def test_target_policy_overall_now_rejected(self) -> None:
        with pytest.raises(ValueError, match="overall.*removed"):
            _resolve_adaptive_settings(
                _ns(gt_target_policy="overall"), config={},
            )

    def test_target_policy_per_type_maps_to_dataset(self) -> None:
        s = _resolve_adaptive_settings(
            _ns(gt_target_policy="per_type"), config={},
        )
        assert s["stop_scope"] == "dataset"

    def test_invalid_stop_scope_rejected(self) -> None:
        with pytest.raises(ValueError, match="stop_scope"):
            _resolve_adaptive_settings(
                _ns(gt_stop_scope="bogus"), config={},
            )

    def test_stop_scope_auto_passes_through(self) -> None:
        # ``auto`` is a sentinel resolved later in ``main`` once the
        # active error-type set is known; ``_resolve_adaptive_settings``
        # must accept it verbatim.
        s = _resolve_adaptive_settings(_ns(gt_stop_scope="auto"), config={})
        assert s["stop_scope"] == "auto"

    def test_negative_num_runs_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_runs"):
            _resolve_adaptive_settings(_ns(gt_num_runs=0), config={})

    def test_target_mass_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_missing_mass"):
            _resolve_adaptive_settings(
                _ns(gt_target_missing_mass=1.5), config={},
            )

    def test_min_runs_clamped_to_num_runs(self) -> None:
        s = _resolve_adaptive_settings(
            _ns(gt_num_runs=2, gt_min_runs=10), config={},
        )
        assert s["num_runs"] == 2
        assert s["min_runs"] == 2  # clamped


# ---------------------------------------------------------------------------
# _auto_stop_scope_for_types
# ---------------------------------------------------------------------------


def _et(category: str) -> ErrorType:
    return ErrorType(
        key=f"k_{category.replace(' ', '_').replace('/', '_')}",
        name="n", category=category, description="d", example="e", notes="n",
    )


class TestAutoStopScope:
    def test_single_column_only_resolves_to_column(self) -> None:
        types = [_et("Single-Column / Typo"), _et("Single-Column / Inconsistency")]
        assert _auto_stop_scope_for_types(types) == "column"

    def test_multi_column_in_mix_resolves_to_column_pair(self) -> None:
        types = [_et("Single-Column / Typo"), _et("Multi-Column / Intra-Record")]
        assert _auto_stop_scope_for_types(types) == "column_pair"

    def test_multi_row_in_mix_resolves_to_csv(self) -> None:
        types = [
            _et("Single-Column / Typo"),
            _et("Multi-Column / Intra-Record"),
            _et("Multi-Row / Cross-Row"),
        ]
        assert _auto_stop_scope_for_types(types) == "csv"

    def test_multi_table_wins_over_all_others(self) -> None:
        types = [
            _et("Single-Column / Typo"),
            _et("Multi-Column / Intra-Record"),
            _et("Multi-Row / Cross-Row"),
            _et("Multi-Table / Cross-Source"),
        ]
        assert _auto_stop_scope_for_types(types) == "multi_table"

    def test_empty_falls_back_to_column(self) -> None:
        assert _auto_stop_scope_for_types([]) == "column"

    def test_unknown_category_falls_back_to_column(self) -> None:
        assert _auto_stop_scope_for_types([_et("Unknown-Scope")]) == "column"


# ---------------------------------------------------------------------------
# _run_adaptive_loop — fixed mode
# ---------------------------------------------------------------------------


class TestFixedMode:
    def test_runs_exactly_num_runs(self) -> None:
        llm = _CannedLLM()
        settings = _settings(gt_mode="fixed", num_runs=3, min_runs=1)
        all_reports, all_flat, gt_artifacts = _run(settings, llm=llm)
        # 3 passes × 1 chunk × 2 types = 6 detector calls.
        assert llm.detector_calls == 6
        assert llm.variant_calls == 0
        assert gt_artifacts["num_runs_completed"] == 3
        assert "fixed: completed 3" in gt_artifacts["stop_reason"]

    def test_returns_well_formed_artifacts(self) -> None:
        settings = _settings(gt_mode="fixed", num_runs=2)
        all_reports, all_flat, gt = _run(settings)
        # Top-level keys the caller persists.
        for key in (
            "num_runs_completed", "stop_reason", "species_records",
            "progression", "estimates_payload", "markdown_report",
            "overall_estimate", "per_run_dumps", "variant_records_per_run",
            "per_error_type", "per_source", "run_info",
        ):
            assert key in gt
        # progression has one entry per pass.
        assert len(gt["progression"]) == 2
        # per_run_dumps too.
        assert [d["run_id"] for d in gt["per_run_dumps"]] == [0, 1]


# ---------------------------------------------------------------------------
# _run_adaptive_loop — adaptive mode (unified per-arm stop)
# ---------------------------------------------------------------------------


class TestAdaptiveOverall:
    def test_early_stop_after_min_runs_when_mass_meets_target(self) -> None:
        # The canned LLM always emits the SAME singleton (row=0, c0).
        # With permissive per-arm config (k_min=1, no two-consecutive),
        # the arm's missing mass <= target after pass 2 → loop stops
        # at min_runs=2.
        llm = _CannedLLM()
        settings = _settings(
            gt_mode="adaptive", num_runs=5, min_runs=2,
            target_missing_mass=0.5,
        )
        _, _, gt = _run(settings, llm=llm)
        assert gt["num_runs_completed"] == 2
        assert "saturated" in gt["stop_reason"]
        assert "after 2 run" in gt["stop_reason"]

    def test_min_runs_floor_honoured(self) -> None:
        llm = _CannedLLM()
        # min_runs=4 — even though arms saturate from pass 1, the loop
        # is FORBIDDEN from stopping until 4 passes complete.
        settings = _settings(
            gt_mode="adaptive", num_runs=6, min_runs=4,
            target_missing_mass=1.0,
        )
        _, _, gt = _run(settings, llm=llm)
        assert gt["num_runs_completed"] == 4
        assert llm.detector_calls == 4 * 2  # 4 passes × 2 types

    def test_ceiling_reached_without_target_attainment(self) -> None:
        # Singletons every pass under target=0.0 keep arms active
        # forever; loop must hit the num_runs ceiling.
        llm = _CannedLLM(payload=(
            '[{"row":0,"columns":["c0"],"observed":"x","explanation":"e"},'
            '{"row":1,"columns":["c0"],"observed":"y","explanation":"e"}]'
        ))
        settings = _settings(
            gt_mode="adaptive", num_runs=2, min_runs=1,
            target_missing_mass=0.0,
        )
        _, _, gt = _run(settings, llm=llm)
        assert gt["stop_reason"]
        assert gt["num_runs_completed"] in (1, 2)


# ---------------------------------------------------------------------------
# _run_adaptive_loop — dataset-scope (one arm per error_type)
# ---------------------------------------------------------------------------


class TestAdaptivePerType:
    def test_per_type_saturation_history_populated(self) -> None:
        llm = _CannedLLM()
        settings = _settings(
            gt_mode="adaptive", num_runs=4, min_runs=1,
            target_missing_mass=0.5, stop_scope="dataset",
        )
        _, _, gt = _run(settings, llm=llm)
        payload = gt["estimates_payload"]
        # When all types saturate, loop stops early with both keys
        # listed in saturated_types (sorted).
        assert sorted(payload["saturated_types"]) == ["casing", "typo"]
        assert payload["stop_scope"] == "dataset"
        # saturation_history has one entry per completed pass.
        assert len(payload["saturation_history"]) == gt["num_runs_completed"]
        assert "all 2 seed type(s) saturated" in gt["stop_reason"]
        # runs_probed_per_type reflects pruning: each type was probed
        # for ALL completed passes (saturation only prunes the NEXT
        # pass, so the count stays at runs_completed for both types
        # when they saturate together on the same pass).
        assert payload["runs_probed_per_type"]["typo"] == gt["num_runs_completed"]
        assert payload["runs_probed_per_type"]["casing"] == gt["num_runs_completed"]

    def test_zero_observation_type_saturates_after_min_runs(self) -> None:
        # Detector returns ZERO detections on every pass. After
        # ``min_runs`` passes with no evidence, every seed type should
        # be saturated and the loop should early-stop. Pre-fix
        # behaviour: zero-observation types were treated as "still
        # active" indefinitely, so the loop ran to ``num_runs`` even
        # when nothing was being detected.
        llm = _CannedLLM(payload="[]")
        settings = _settings(
            gt_mode="adaptive", num_runs=10, min_runs=2,
            target_missing_mass=0.5, stop_scope="dataset",
        )
        _, _, gt = _run(settings, llm=llm)
        payload = gt["estimates_payload"]
        # Loop must stop the pass min_runs is hit (or shortly after).
        assert gt["num_runs_completed"] == 2
        assert sorted(payload["saturated_types"]) == ["casing", "typo"]
        # Stop reason mentions the per-arm rule, not "hit ceiling".
        assert "saturated" in gt["stop_reason"]
        assert "hit ceiling" not in gt["stop_reason"]


# ---------------------------------------------------------------------------
# Per-arm LLM-call pruning
# ---------------------------------------------------------------------------


class TestPerArmPruning:
    """Once an arm saturates, the (chunk, error_type) pairs that map
    to that arm must be removed from the next pass's job list - i.e.
    no LLM ``query`` call is made for them.
    """

    def _two_source_frames(self) -> Dict[str, pd.DataFrame]:
        # Distinct column names so the SourceAwareLLM stub can tell
        # which source produced any given prompt.
        return {
            "src_sat.csv": pd.DataFrame({"col_a": [f"x{i}" for i in range(4)]}),
            "src_active.csv": pd.DataFrame({"col_b": [f"y{i}" for i in range(4)]}),
        }

    def _two_types(self) -> List[ErrorType]:
        return [
            ErrorType(key="alpha", name="Alpha", category="A",
                      description="d", example="ex", notes="n"),
            ErrorType(key="beta", name="Beta", category="A",
                      description="d", example="ex", notes="n"),
        ]

    def test_arm_pruning_skips_llm_calls_for_saturated_arm(self) -> None:
        # Two sources, two types, stop_scope=csv (so each (source,
        # type) is its own arm).
        #
        # Stub layout:
        #   * type "alpha" + src_sat  -> singleton  → arm saturates
        #     after pass 1 (k_min=1, target=1.0).
        #   * type "alpha" + src_active -> []      → arm absent from
        #     the table, so type "alpha" stays "active" (its
        #     observed-arm set is fully saturated, but the loop's
        #     type-level rule treats unobserved arms conservatively
        #     - 1 arm saturated of 1 observed → type alpha saturates).
        #
        # To keep the loop running long enough to observe arm-level
        # pruning of (src_sat, alpha) on pass 2, we add type "beta"
        # whose arms NEVER saturate (never saturating via by_error_type).
        types = self._two_types()
        taxonomy = _taxonomy(types)
        frames = self._two_source_frames()

        class _StubLLM:
            def __init__(self) -> None:
                self.endpoints = ["http://stub-1"]
                self.detector_calls = 0
                self.calls: List[Dict[str, Any]] = []

            def query(self, model, prompt, *, reasoning_effort,
                      reasoning_summary, system_prompt):
                self.detector_calls += 1
                # Sniff (source, type) from the rendered prompt.
                src_token = (
                    "src_sat" if "HEADERS=col_a" in prompt else "src_active"
                )
                type_token = (
                    "alpha" if "TYPE=alpha" in prompt else "beta"
                )
                self.calls.append({"source": src_token, "type": type_token})
                # Saturating arm: alpha + src_sat → 1 singleton at row 0.
                # Beta arms: 4 singletons each (epsilon=0.0, never saturates).
                # Alpha + src_active → empty (so its alpha arm is absent).
                if type_token == "alpha" and src_token == "src_sat":
                    payload = ('[{"row":0,"columns":["col_a"],'
                               '"observed":"x","explanation":"e"}]')
                elif type_token == "alpha":
                    payload = "[]"
                else:  # beta on either source
                    col = "col_a" if src_token == "src_sat" else "col_b"
                    payload = json.dumps([
                        {"row": r, "columns": [col],
                         "observed": "x", "explanation": "e"}
                        for r in range(4)
                    ])
                return LLMResponse(
                    text=payload, endpoint=self.endpoints[0],
                    model=model, attempts=1,
                )

        llm = _StubLLM()
        # Build per-arm config: alpha saturates fast (epsilon=1.0,
        # k_min=1), beta never saturates (k_min unreachable) so the
        # loop keeps running and we can observe arm-level pruning of
        # alpha:src_sat while alpha:src_sat's chunk is NOT skipped
        # for type beta.
        alpha_cfg = ArmConfig(
            epsilon=1.0, k_min=1, k_max=10000,
            require_two_consecutive=False,
        )
        beta_cfg = ArmConfig(
            epsilon=1.0, k_min=10_000, k_max=1_000_000,
            require_two_consecutive=False,
        )
        settings = _settings(
            gt_mode="adaptive", num_runs=3, min_runs=1,
            target_missing_mass=1.0, stop_scope="csv",
            per_arm=(alpha_cfg, {"beta": beta_cfg}, None),
        )
        _, _, gt = _run(
            settings, llm=llm, types=types, taxonomy=taxonomy, frames=frames,
        )

        # Loop must complete num_runs passes (beta never saturates).
        assert gt["num_runs_completed"] == 3, (
            f"loop should run all 3 passes; stopped at "
            f"{gt['num_runs_completed']} (stop_reason={gt['stop_reason']!r})"
        )

        # Per-arm LLM-call counting.
        def n_calls(*, source: str, type_: str) -> int:
            return sum(
                1 for c in llm.calls
                if c["source"] == source and c["type"] == type_
            )

        # Pass 1 (no pruning): all 4 (source,type) pairs called once → 4 calls.
        # Pass 2: alpha:src_sat arm saturated → that pair must be SKIPPED.
        #         alpha:src_active still probed (alpha not yet type-saturated).
        # Pass 3: same as pass 2.
        #
        # ⇒ alpha:src_sat called exactly ONCE across all 3 passes.
        assert n_calls(source="src_sat", type_="alpha") == 1, (
            f"alpha:src_sat should be probed once then pruned; calls={llm.calls}"
        )
        # alpha:src_active never observed in the arm table → not pruned by
        # arm logic. But alpha as a TYPE saturates after pass 1 (its only
        # observed arm is saturated), so type-level pruning kicks in → 1 call.
        assert n_calls(source="src_active", type_="alpha") == 1
        # Beta arms never saturate → probed every pass on both sources.
        assert n_calls(source="src_sat", type_="beta") == 3
        assert n_calls(source="src_active", type_="beta") == 3
        # Total: 1 + 1 + 3 + 3 = 8. Without any pruning: 2*2*3 = 12.
        assert llm.detector_calls == 8

        # Per-arm transition recorded in saturation_history.
        history = gt["estimates_payload"]["saturation_history"]
        sat_arms_pass_1 = history[0]["newly_saturated_arms"]
        labels = {(e["scope_label"], e["type_key"]) for e in sat_arms_pass_1}
        assert ("src_sat.csv", "alpha") in labels
        assert history[0]["saturated_arms_total"] >= 1



# ---------------------------------------------------------------------------
# skip_identity_run
# ---------------------------------------------------------------------------


class TestSkipIdentityRun:
    def test_skip_identity_shifts_seed(self) -> None:
        # When skip_identity_run=True, run 0 internally uses
        # effective_run_id=1 → frame perturbation actually fires.
        llm = _CannedLLM()
        settings = _settings(
            gt_mode="fixed", num_runs=1, skip_identity_run=True,
            shuffle_rows=True,
        )
        _, _, gt = _run(settings, llm=llm)
        # The single completed run's view has shuffled_rows=True
        # because run_id is shifted to 1, not 0.
        run_meta = gt["run_info"][0]
        # description string contains "shuffle" rather than "identity".
        desc = list(run_meta["descriptions"].values())[0]
        assert "shuffle" in desc

    def test_keep_identity_includes_baseline(self) -> None:
        llm = _CannedLLM()
        settings = _settings(
            gt_mode="fixed", num_runs=1, skip_identity_run=False,
            shuffle_rows=True,
        )
        _, _, gt = _run(settings, llm=llm)
        run_meta = gt["run_info"][0]
        desc = list(run_meta["descriptions"].values())[0]
        # Run 0 with include_identity → identity baseline.
        assert desc == "identity"


# ---------------------------------------------------------------------------
# Prompt-ensemble path
# ---------------------------------------------------------------------------


@pytest.fixture
def variant_template_path(tmp_path: Path) -> Path:
    p = tmp_path / "variant.yaml"
    p.write_text(
        "base:\n"
        "  system: rewrite the error type so it reads differently\n"
        "  user_template: |\n"
        "    key={type_key} name={type_name} cat={type_category}\n"
        "    desc={description}\n"
        "    notes={notes}\n"
        "    ex={example}\n"
        "    pass={pass_index}\n"
    )
    return p


class TestPromptEnsemble:
    def test_variants_only_on_non_identity_passes(
        self, variant_template_path: Path, tmp_path: Path
    ) -> None:
        from src.adaptive.prompt_variants import PromptVariantGenerator

        llm = _CannedLLM()
        gen = PromptVariantGenerator(
            llm=llm, model="stub-model",
            template_path=variant_template_path,
            concurrency=1,
        )
        # 2 passes total; pass 0 is identity (no variants), pass 1 fires
        # one variant call per active type (2 types).
        settings = _settings(
            gt_mode="fixed", num_runs=2, prompt_ensemble=True,
            shuffle_rows=False, skip_identity_run=False,
        )
        run_dir = tmp_path / "run_dir"
        _, _, gt = _run(
            settings, llm=llm,
            variant_generator=gen, run_dir=run_dir,
        )
        # Variant generator invoked exactly 2 times (1 non-identity pass × 2 types).
        assert llm.variant_calls == 2
        # variant_records_per_run lists ONLY the non-identity pass.
        assert len(gt["variant_records_per_run"]) == 1
        rec = gt["variant_records_per_run"][0]
        assert rec["run_id"] == 1
        assert rec["pass_index"] == 1
        # Persisted variant records on disk under prompt_variants/run_01/.
        run_01_dir = run_dir / "prompt_variants" / "run_01"
        assert run_01_dir.is_dir()
        assert (run_01_dir / "typo.json").is_file()
        assert (run_01_dir / "casing.json").is_file()
        # No record dir for run_00 (identity pass — variants disabled).
        assert not (run_dir / "prompt_variants" / "run_00").exists()

    def test_saturated_type_excluded_from_variant_generation(
        self, variant_template_path: Path, tmp_path: Path
    ) -> None:
        """Once all of a type's arms saturate, the variant generator
        must not be invoked for that type on subsequent passes.

        We force ``typo`` to saturate after pass 1 (k_min=1,
        target=1.0) and keep ``casing`` permanently active (never saturating).
        With ``skip_identity_run=True`` every pass is a variant pass,
        so the variant call count is the cleanest signal that prompt
        generation actually skipped the saturated type.
        """
        from src.adaptive.prompt_variants import PromptVariantGenerator

        llm = _CannedLLM()
        gen = PromptVariantGenerator(
            llm=llm, model="stub-model",
            template_path=variant_template_path,
            concurrency=1,
        )
        typo_cfg = ArmConfig(
            epsilon=1.0, k_min=1, k_max=10000,
            require_two_consecutive=False,
        )
        casing_cfg = ArmConfig(
            epsilon=1.0, k_min=10_000, k_max=1_000_000,
            require_two_consecutive=False,
        )
        # 3 passes, every pass perturbed (skip_identity_run). Without
        # type-level pruning of variant gen: 3 passes × 2 types = 6
        # variant calls. With pruning (typo saturates after pass 1):
        # pass 1 (typo + casing) + pass 2 (casing) + pass 3 (casing)
        # = 4 variant calls.
        settings = _settings(
            gt_mode="adaptive", num_runs=3, min_runs=1,
            target_missing_mass=1.0, stop_scope="dataset",
            prompt_ensemble=True, skip_identity_run=True,
            shuffle_rows=False,
            per_arm=(typo_cfg, {"casing": casing_cfg}, None),
        )
        run_dir = tmp_path / "run_dir_pruned"
        _, _, gt = _run(
            settings, llm=llm,
            variant_generator=gen, run_dir=run_dir,
        )

        # Loop runs all 3 passes (casing never saturates).
        assert gt["num_runs_completed"] == 3
        # typo enters saturated_types after pass 1 → variant gen skips
        # it on passes 2 and 3.
        assert llm.variant_calls == 4, (
            f"variant_calls={llm.variant_calls}; expected 2+1+1=4 after "
            f"typo's variant generation is pruned"
        )
        # On disk: pass 0 (run_00) has variants for both types; passes
        # 1 & 2 have variants for casing only.
        run_00 = run_dir / "prompt_variants" / "run_00"
        assert (run_00 / "typo.json").is_file()
        assert (run_00 / "casing.json").is_file()
        for pass_dir in ("run_01", "run_02"):
            d = run_dir / "prompt_variants" / pass_dir
            assert d.is_dir(), f"{d} should exist"
            assert (d / "casing.json").is_file()
            assert not (d / "typo.json").exists(), (
                f"typo variant should be pruned in {pass_dir}"
            )
        # variant_records_per_run records the slim 2/1/1 layout.
        per_run_counts = [
            len(rec["records"]) for rec in gt["variant_records_per_run"]
        ]
        assert per_run_counts == [2, 1, 1]


# ---------------------------------------------------------------------------
# Per-arm scope-aware estimator (additive artifact)
# ---------------------------------------------------------------------------


class TestPerArmArtifacts:
    """The adaptive loop should now produce a per-(column, error_type)
    estimator artifact alongside the legacy global block. Purely
    additive: never breaks the run, never affects stopping.
    """

    def test_gt_artifacts_carries_per_arm_payload(self) -> None:
        # The artifact's ``scope_kind`` follows the loop's
        # ``stop_scope`` so the carried-over status is consistent with
        # what the loop tracked. Default ``_settings`` uses
        # ``stop_scope="dataset"``.
        settings = _settings(gt_mode="fixed", num_runs=2)
        _, _, gt = _run(settings)
        payload = gt.get("per_arm_payload")
        assert payload is not None
        assert payload["scope_kind"] == "dataset"
        assert payload["num_runs_completed"] == 2
        # Two seed types ("typo", "casing") under stop_scope=dataset
        # collapse to two arms (one per type, one shared scope_id=()).
        arms = payload["arms"]
        assert isinstance(arms, list)
        assert all("scope_kind" in a and a["scope_kind"] == "dataset"
                   for a in arms)
        # Each arm carries the new status field.
        assert all(a["status"] in {
            "active", "saturated", "exhausted",
            "insufficient", "not_applicable",
        } for a in arms)

    def test_per_arm_artifact_follows_stop_scope_column(self) -> None:
        # Explicit column scope -> artifact reports per (source, column).
        settings = _settings(
            gt_mode="fixed", num_runs=2, stop_scope="column",
        )
        _, _, gt = _run(settings)
        payload = gt["per_arm_payload"]
        assert payload["scope_kind"] == "column"
        for arm in payload["arms"]:
            assert arm["scope_kind"] == "column"

    def test_per_arm_markdown_rendered(self) -> None:
        # Force column scope so the markdown header asserts below stay
        # stable regardless of default.
        settings = _settings(
            gt_mode="fixed", num_runs=2, stop_scope="column",
        )
        _, _, gt = _run(settings)
        md = gt.get("per_arm_markdown")
        assert md is not None
        # Section header for the column scope_kind, table header rows.
        assert "scope_kind = `column`" in md
        assert "p_miss" in md and "FN_hat" in md

    def test_per_arm_artifacts_are_optional_under_failure(
        self, monkeypatch
    ) -> None:
        """If the per-arm computation blows up, the loop must still
        return successfully with payload/markdown set to None."""
        from experiments import run_cleaning_detection as rcd

        def _boom(*a, **kw):
            raise RuntimeError("simulated estimator failure")

        monkeypatch.setattr(rcd, "aggregate_runs_by_scope", _boom)

        settings = _settings(gt_mode="fixed", num_runs=2)
        # Loop completes fine; artifact keys exist but are None.
        _, _, gt = _run(settings)
        assert gt["per_arm_payload"] is None
        assert gt["per_arm_markdown"] is None


# ---------------------------------------------------------------------------
# Regression: streak persists across passes (require_two_consecutive)
# ---------------------------------------------------------------------------


class TestStreakAcrossPasses:
    """``evaluate_stop`` is stateless. The adaptive loop is responsible
    for threading each arm's previous ``(status, streak)`` into the
    next pass's call so ``require_two_consecutive`` can fire across
    passes. Pre-fix bug: the streak reset to 0 every pass, so with
    two-consecutive enabled NO arm could ever saturate mid-loop and
    the loop ran to its ``num_runs`` ceiling.
    """

    def test_two_consecutive_actually_saturates(self) -> None:
        # min_runs=1 so saturation is reachable from pass 2; epsilon=1.0
        # makes the GT mass condition trivially true; k_min=1 so a
        # single pass qualifies. With streak persisted, the per-arm
        # predicate flips arms to "saturated" on pass 2
        # (streak: 1 -> 2) and the loop short-circuits.
        cfg = ArmConfig(
            epsilon=1.0, k_min=1, k_max=10000,
            require_two_consecutive=True,
        )
        settings = _settings(
            gt_mode="adaptive", num_runs=10, min_runs=1,
            target_missing_mass=1.0, stop_scope="dataset",
            per_arm=(cfg, {}, None),
        )
        _, _, gt = _run(settings)
        # Pre-fix: ran all 10 passes. Post-fix: stops by pass 3 once
        # both seed-type arms have a streak of 2.
        assert gt["num_runs_completed"] <= 3, (
            f"loop ran {gt['num_runs_completed']} pass(es); streak "
            f"is not persisting across passes"
        )
        payload = gt["estimates_payload"]
        assert sorted(payload["saturated_types"]) == ["casing", "typo"]
        # Saturation history: at least one entry must show
        # ``newly_saturated_arms`` populated, proving the in-loop arm
        # status transitions actually fired.
        any_saturation = any(
            entry.get("newly_saturated_arms")
            for entry in payload["saturation_history"]
        )
        assert any_saturation, (
            "expected newly_saturated_arms to be populated in at "
            "least one saturation_history entry"
        )

    def test_per_arm_artifact_status_reflects_mid_run_saturation(
        self,
    ) -> None:
        # Same setup as above; the post-loop per-arm artifact must
        # report ``saturated`` (carried over from in-loop bookkeeping)
        # rather than ``exhausted`` (which is what the bug produced
        # because num_runs == k_max on the final pass).
        cfg = ArmConfig(
            epsilon=1.0, k_min=1, k_max=10000,
            require_two_consecutive=True,
        )
        settings = _settings(
            gt_mode="adaptive", num_runs=10, min_runs=1,
            target_missing_mass=1.0, stop_scope="dataset",
            per_arm=(cfg, {}, None),
        )
        _, _, gt = _run(settings)
        statuses = {a["status"] for a in gt["per_arm_payload"]["arms"]}
        # At least one arm should be ``saturated``, not all ``exhausted``.
        assert "saturated" in statuses, (
            f"expected at least one saturated arm, got statuses={statuses}"
        )

    def test_artifact_does_not_double_count_streak(self) -> None:
        """Regression: the final per-arm artifact must NOT re-run
        ``evaluate_stop`` with the post-pass ``arm_prev_streak``.

        Pre-fix bug: in-loop bookkeeping stored the *after-pass* status
        and streak; the final artifact then called ``evaluate_stop``
        again on the same cumulative species table, threading those
        post-pass values in as ``previous_*``. With
        ``require_two_consecutive=True``, an arm that ended the loop
        ``active`` with streak=1 would be incremented to streak=2 and
        spuriously promoted to ``saturated`` even though no new pass
        had been observed.

        Setup: ``num_runs=2`` (loop forced to terminate on pass 2 due
        to the ``num_runs`` cap), ``epsilon=1.0`` so the missing-mass
        condition is trivially satisfied every pass, and
        ``require_two_consecutive=True``. Pass 1: streak 0->1
        (active). Pass 2: streak 1->2 (saturated in-loop). The bug
        would still trigger if the loop terminated WITHOUT the
        in-loop saturation having fired (i.e. exactly when the
        artifact's redundant call could change the answer); to force
        that regime we cap at ``num_runs=1``.
        """
        cfg = ArmConfig(
            epsilon=1.0, k_min=1, k_max=10000,
            require_two_consecutive=True,
        )
        # num_runs=1: only one pass, so streak can reach 1 in-loop
        # but never 2. Pre-fix: artifact bumps to 2 -> "saturated".
        # Post-fix: artifact trusts the in-loop "active" decision.
        settings = _settings(
            gt_mode="adaptive", num_runs=1, min_runs=1,
            target_missing_mass=1.0, stop_scope="dataset",
            per_arm=(cfg, {}, None),
        )
        _, _, gt = _run(settings)
        statuses = [a["status"] for a in gt["per_arm_payload"]["arms"]]
        streaks = [a["saturated_streak"] for a in gt["per_arm_payload"]["arms"]]
        assert "saturated" not in statuses, (
            f"artifact double-counted streak: arms={statuses}, "
            f"streaks={streaks}; expected 'active' for all (pass 1 "
            f"alone cannot satisfy require_two_consecutive)"
        )
        assert all(s <= 1 for s in streaks), (
            f"streak exceeded 1 after a single pass: {streaks}"
        )
