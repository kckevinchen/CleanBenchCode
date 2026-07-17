"""Tests for adaptive (Good-Turing) multi-table group discovery
(``src.core.multi_table.discovery_adaptive``).

The adaptive loop repeats the per-workbook union/join confirm under
perturbation, accumulates the recall-first union of groups keyed by the
coarse species ``(kind, frozenset(sheets))``, and stops per workbook
when the Good-Turing / Chao1 estimate saturates.
"""

from __future__ import annotations

import json
from typing import Callable, List, Optional

import pandas as pd

from src.core.llm_client import LLMResponse
from src.core.multi_table.discovery_adaptive import (
    AdaptiveDiscoveryConfig,
    group_species_key,
    resolve_workbook_adaptive,
    summarize_discovery,
)
from src.core.multi_table.resolver import (
    ColumnMappingEntry,
    ConfirmPromptConfig,
    ResolvedGroup,
)
from src.core.table_digest import build_table_digest


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------

_PROMPT = ConfirmPromptConfig(
    model="m", reasoning_effort="medium", reasoning_summary="detailed",
    system_prompt="BASE SYSTEM",
    user_template=(
        "{workbook} {sheet_headers_block} {unionable_groups_block} "
        "{joinable_pairs_block} {row_samples_block}"
    ),
)


def _frames():
    return {
        "a.csv": pd.DataFrame({"K": ["k1", "k2", "k3"], "v": [1, 2, 3]}),
        "b.csv": pd.DataFrame({"K": ["k1", "k2", "k3"], "v": [1, 2, 9]}),
        "c.csv": pd.DataFrame({"K": ["k1", "k9"], "v": [1, 2]}),
    }


def _digests(frames):
    return {
        s: build_table_digest(df, source=s, sample_k_values=8)
        for s, df in frames.items()
    }


def _union(group_id, sheets, cols, pk=("K",)):
    return {
        "group_id": group_id, "kind": "union", "confirmed": True,
        "rationale": "r", "sheets": list(sheets),
        "column_mapping": {
            c: [{"sheet": s, "column": c} for s in sheets] for c in cols
        },
        "primary_key": list(pk),
    }


class _ScriptLLM:
    """Returns a scripted list of group-dicts per call; records prompts."""

    def __init__(self, script: Callable[[int], list]) -> None:
        self.endpoints = ["http://stub"]
        self.script = script
        self.n = 0
        self.system_prompts: List[str] = []

    def query(self, model, prompt, *, reasoning_effort,
              reasoning_summary, system_prompt):
        idx = self.n
        self.n += 1
        self.system_prompts.append(system_prompt)
        groups = self.script(idx)
        return LLMResponse(
            text=json.dumps({"groups": groups}),
            endpoint="http://stub", model=model, attempts=1,
        )


def _run(script, **cfg_kw):
    frames = _frames()
    cfg = AdaptiveDiscoveryConfig(enabled=True, **cfg_kw)
    llm = _ScriptLLM(script)
    rw, stats = resolve_workbook_adaptive(
        workbook="wb", sheets=["a.csv", "b.csv", "c.csv"],
        frames=frames, digests=_digests(frames),
        llm=llm, model="m", prompt_cfg=_PROMPT, cfg=cfg,
    )
    return rw, stats, llm


# ---------------------------------------------------------------------------
# species key
# ---------------------------------------------------------------------------


class TestSpeciesKey:
    def test_sheet_order_invariant(self) -> None:
        g1 = ResolvedGroup(
            workbook="wb", group_id="u1", kind="union", confirmed=True,
            rationale="", sheets=("a.csv", "b.csv"), column_mapping={},
            primary_key=(),
        )
        g2 = ResolvedGroup(
            workbook="wb", group_id="uX", kind="union", confirmed=True,
            rationale="", sheets=("b.csv", "a.csv"), column_mapping={},
            primary_key=(),
        )
        assert group_species_key(g1) == group_species_key(g2)

    def test_kind_distinguishes(self) -> None:
        g1 = ResolvedGroup(
            workbook="wb", group_id="u1", kind="union", confirmed=True,
            rationale="", sheets=("a.csv", "b.csv"), column_mapping={},
            primary_key=(),
        )
        g2 = ResolvedGroup(
            workbook="wb", group_id="j1", kind="join", confirmed=True,
            rationale="", sheets=("a.csv", "b.csv"), column_mapping={},
            primary_key=(),
        )
        assert group_species_key(g1) != group_species_key(g2)


class TestArmConfig:
    def test_maps_fields(self) -> None:
        cfg = AdaptiveDiscoveryConfig(
            k_min=3, k_max=9, stop_when="both", epsilon=0.02,
        )
        arm = cfg.to_arm_config()
        assert arm.k_min == 3 and arm.k_max == 9
        assert arm.stop_when == "both" and arm.epsilon == 0.02


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------


class TestAdaptiveLoop:
    def test_stable_group_stops_two_consecutive(self) -> None:
        # Same single union every pass -> no new species after run 0.
        script = lambda i: [_union("u1", ["a.csv", "b.csv"], ["K"])]  # noqa: E731
        rw, stats, llm = _run(
            script, k_min=2, k_max=8, require_two_consecutive=True,
            shuffle_rows=False, prompt_ensemble=False,
        )
        # run0 new, run1 no-new, run2 no-new -> stop at 3 passes.
        assert llm.n == 3
        assert stats["stopped_reason"] == "two_consecutive"
        assert len(rw.groups) == 1

    def test_stops_at_k_min_without_two_consecutive(self) -> None:
        script = lambda i: [_union("u1", ["a.csv", "b.csv"], ["K"])]  # noqa: E731
        rw, stats, llm = _run(
            script, k_min=2, k_max=8, require_two_consecutive=False,
            shuffle_rows=False, prompt_ensemble=False,
        )
        # GT missing-mass is 0 once the only species repeats -> stop at k_min.
        assert llm.n == 2
        assert len(rw.groups) == 1

    def test_churning_runs_to_k_max(self) -> None:
        # A brand-new species every pass -> never saturates -> k_max.
        # Use only REAL sheets (a/b/c); validation strips unknown sheets.
        def script(i):
            specs = [
                _union("u_ab", ["a.csv", "b.csv"], ["K"]),
                _union("u_ac", ["a.csv", "c.csv"], ["K"]),
                _union("u_bc", ["b.csv", "c.csv"], ["K"]),
                _union("u_abc", ["a.csv", "b.csv", "c.csv"], ["K"]),
            ]
            return specs[: i + 1]   # one new species per pass
        rw, stats, llm = _run(
            script, k_min=2, k_max=4, require_two_consecutive=True,
            shuffle_rows=False, prompt_ensemble=False,
        )
        assert llm.n == 4
        assert stats["stopped_reason"] == "k_max"
        assert stats["good_turing_missing_mass"] > 0.0

    def test_accumulates_union_including_new_group(self) -> None:
        # run0 finds u1; run1+ also surface an ADDED group (a,c).
        def script(i):
            gs = [_union("u1", ["a.csv", "b.csv"], ["K"])]
            if i >= 1:
                gs.append(_union("u_new1", ["a.csv", "c.csv"], ["K"]))
            return gs
        rw, stats, llm = _run(
            script, k_min=2, k_max=6, require_two_consecutive=True,
            shuffle_rows=False, prompt_ensemble=False,
        )
        species = {group_species_key(g) for g in rw.groups}
        assert ("union", frozenset({"a.csv", "b.csv"})) in species
        assert ("union", frozenset({"a.csv", "c.csv"})) in species

    def test_keeps_most_complete_mapping_representative(self) -> None:
        # run0 maps only K; run1+ map K and v for the same species.
        def script(i):
            cols = ["K"] if i == 0 else ["K", "v"]
            return [_union("u1", ["a.csv", "b.csv"], cols)]
        rw, _stats, _llm = _run(
            script, k_min=2, k_max=4, require_two_consecutive=False,
            shuffle_rows=False, prompt_ensemble=False,
        )
        u1 = next(iter(rw.groups))
        assert sorted(u1.column_mapping.keys()) == ["K", "v"]


class TestPerturbation:
    def test_prompt_ensemble_appends_lens(self) -> None:
        script = lambda i: [_union("u1", ["a.csv", "b.csv"], ["K"])]  # noqa: E731
        _rw, _stats, llm = _run(
            script, k_min=3, k_max=3, require_two_consecutive=False,
            shuffle_rows=False, prompt_ensemble=True,
        )
        # run 0 = base system; runs 1+ carry an appended lens.
        assert llm.system_prompts[0] == "BASE SYSTEM"
        assert any("ADDITIONAL LENS" in sp for sp in llm.system_prompts[1:])

    def test_prompt_ensemble_off_keeps_base_system(self) -> None:
        script = lambda i: [_union("u1", ["a.csv", "b.csv"], ["K"])]  # noqa: E731
        _rw, _stats, llm = _run(
            script, k_min=3, k_max=3, require_two_consecutive=False,
            shuffle_rows=False, prompt_ensemble=False,
        )
        assert all(sp == "BASE SYSTEM" for sp in llm.system_prompts)


class TestSummarize:
    def test_rollup_shape(self) -> None:
        per_wb = [
            {"n_passes": 3, "good_turing_missing_mass": 0.0,
             "stopped_reason": "two_consecutive"},
            {"n_passes": 6, "good_turing_missing_mass": 0.2,
             "stopped_reason": "k_max"},
        ]
        s = summarize_discovery(per_wb)
        assert s["n_workbooks"] == 2
        assert s["total_passes"] == 9
        assert s["stopped_reasons"] == {"two_consecutive": 1, "k_max": 1}
        assert s["max_missing_mass"] == 0.2

    def test_empty(self) -> None:
        s = summarize_discovery([])
        assert s["n_workbooks"] == 0 and s["total_passes"] == 0


# ---------------------------------------------------------------------------
# Global mode=adaptive drives discovery (no separate flag)
# ---------------------------------------------------------------------------


class TestFromAdaptiveSettings:
    def test_adaptive_settings_enable_discovery(self) -> None:
        cfg = AdaptiveDiscoveryConfig.from_adaptive_settings({
            "gt_mode": "adaptive",
            "min_runs": 3,
            "num_runs": 8,
            "target_missing_mass": 0.1,
            "shuffle_rows": False,
            "prompt_ensemble": True,
            "base_seed": 99,
        })
        assert cfg.enabled is True
        # k_min/k_max inherit the global min_runs/num_runs.
        assert cfg.k_min == 3 and cfg.k_max == 8
        assert cfg.epsilon == 0.1
        assert cfg.shuffle_rows is False
        assert cfg.prompt_ensemble is True
        assert cfg.base_seed == 99

    def test_single_mode_disables(self) -> None:
        assert (
            AdaptiveDiscoveryConfig.from_adaptive_settings(None).enabled is False
        )
        assert AdaptiveDiscoveryConfig.from_adaptive_settings(
            {"gt_mode": "single"}
        ).enabled is False

    def test_per_arm_overrides_stop_and_epsilon(self) -> None:
        from src.adaptive.scoped_gt import ArmConfig
        arm = ArmConfig(stop_when="both", epsilon=0.02, epsilon_chao1=0.03)
        cfg = AdaptiveDiscoveryConfig.from_adaptive_settings({
            "gt_mode": "adaptive", "min_runs": 2, "num_runs": 6,
            "per_arm": (arm, {}, None),
        })
        assert cfg.stop_when == "both"
        assert cfg.epsilon == 0.02 and cfg.epsilon_chao1 == 0.03

    def test_no_discovery_flags_in_cli(self) -> None:
        from src.labeling.cli_builders import build_detection_argv
        argv = build_detection_argv({
            "error_types": "inter_table_inconsistency",
            "multi_table_strategy": "semantic",
        })
        # The dedicated discovery knobs are gone; ``mode`` drives it.
        assert not any("discovery" in a for a in argv)
