"""Unit tests for the offline stop-threshold replay.

Pins replay semantics against hand-constructed per-pass per-arm
traces. The replay must match :func:`src.adaptive.scoped_gt.evaluate_stop`
exactly so that "what would this threshold have done?" computed
offline equals "what this threshold did online" -- otherwise the
hyperparameter search picks a winner that doesn't reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from experiments.analyze_stop_thresholds import (
    ArmKey,
    ThresholdConfig,
    expand_grid,
    load_per_pass_traces,
    replay_arm,
    replay_sweep,
)
from src.adaptive.scoped_gt import (
    ArmConfig,
    ArmEstimate,
    evaluate_stop,
)


# ---------------------------------------------------------------------------
# Trace fixtures
# ---------------------------------------------------------------------------


def _row(
    pass_idx: int,
    *,
    n_observed: int = 5,
    n1: int = 1,
    total_hits: int = 10,
    p_miss: float = 0.1,
    chao1_high: float = 6.0,
    arm_etype: str = "typo",
) -> Dict[str, Any]:
    return {
        "pass_idx": pass_idx,
        "runs_so_far": pass_idx + 1,
        "scope_kind": "column",
        "scope_id": ["a.csv", "c"],
        "error_type": arm_etype,
        "k_arm": pass_idx + 1,
        "n_observed": n_observed,
        "n1": n1,
        "n2": 1,
        "total_hits": total_hits,
        "good_turing_missing_mass": p_miss,
        "chao1_estimated_missed": 1.0,
        "chao1_estimated_total": float(n_observed) + 1.0,
        "chao1_ci_95_low": float(n_observed),
        "chao1_ci_95_high": chao1_high,
        "fn_rate_point": 0.0,
        "status": "active",
        "saturated_streak": 0,
    }


# ---------------------------------------------------------------------------
# replay_arm: GT criterion
# ---------------------------------------------------------------------------


class TestReplayArmGT:
    def test_saturates_when_GT_below_epsilon_with_kmin_met(self) -> None:
        # 5 passes, GT mass starts at 0.5 and drops below 0.05 at pass 4.
        rows = [
            _row(0, p_miss=0.50),
            _row(1, p_miss=0.30),
            _row(2, p_miss=0.10),
            _row(3, p_miss=0.04),  # below threshold; pass 4 (1-based)
            _row(4, p_miss=0.04),
        ]
        cfg = ThresholdConfig(
            stop_when="gt",
            epsilon=0.05,
            epsilon_chao1=0.05,
            k_min=3,
            k_max=5,
            require_two_consecutive=False,
        )
        stop, status = replay_arm(rows, cfg)
        assert stop == 4
        assert status == "saturated"

    def test_two_consec_gate_requires_two_passes(self) -> None:
        # Same trace but require_two_consecutive=True -> stops at pass 5,
        # not pass 4 (the first qualifying pass alone isn't enough).
        rows = [
            _row(0, p_miss=0.50),
            _row(1, p_miss=0.30),
            _row(2, p_miss=0.10),
            _row(3, p_miss=0.04),
            _row(4, p_miss=0.04),
        ]
        cfg = ThresholdConfig(
            stop_when="gt", epsilon=0.05, epsilon_chao1=0.05,
            k_min=3, k_max=5, require_two_consecutive=True,
        )
        stop, status = replay_arm(rows, cfg)
        assert stop == 5
        assert status == "saturated"

    def test_streak_resets_when_predicate_breaks(self) -> None:
        # Predicate satisfied at pass 4, broken at pass 5, satisfied
        # again at pass 6 and 7. Two-consec rule should NOT fire at
        # pass 5 (streak reset), but should at pass 7.
        rows = [
            _row(0, p_miss=0.50),
            _row(1, p_miss=0.30),
            _row(2, p_miss=0.10),
            _row(3, p_miss=0.04),  # pass 4: streak=1
            _row(4, p_miss=0.20),  # pass 5: streak resets to 0
            _row(5, p_miss=0.04),  # pass 6: streak=1
            _row(6, p_miss=0.04),  # pass 7: streak=2 -> saturated
        ]
        cfg = ThresholdConfig(
            stop_when="gt", epsilon=0.05, epsilon_chao1=0.05,
            k_min=3, k_max=10, require_two_consecutive=True,
        )
        stop, status = replay_arm(rows, cfg)
        assert stop == 7
        assert status == "saturated"

    def test_kmin_floor_blocks_early_saturation(self) -> None:
        # Predicate satisfied from pass 1, but k_min=4 -> can't
        # saturate until pass 4 even though the criterion has held
        # from the start.
        rows = [
            _row(0, p_miss=0.001),
            _row(1, p_miss=0.001),
            _row(2, p_miss=0.001),
            _row(3, p_miss=0.001),
            _row(4, p_miss=0.001),
        ]
        cfg = ThresholdConfig(
            stop_when="gt", epsilon=0.05, epsilon_chao1=0.05,
            k_min=4, k_max=5, require_two_consecutive=False,
        )
        stop, status = replay_arm(rows, cfg)
        assert stop == 4
        assert status == "saturated"

    def test_exhausted_when_predicate_never_fires(self) -> None:
        # GT mass stays high; arm runs all the way to k_max.
        rows = [_row(i, p_miss=0.5) for i in range(5)]
        cfg = ThresholdConfig(
            stop_when="gt", epsilon=0.05, epsilon_chao1=0.05,
            k_min=3, k_max=5, require_two_consecutive=False,
        )
        stop, status = replay_arm(rows, cfg)
        assert stop is None
        assert status == "exhausted"


# ---------------------------------------------------------------------------
# replay_arm: Chao1 criterion
# ---------------------------------------------------------------------------


class TestReplayArmChao1:
    def test_chao1_uses_upper_ci_fraction(self) -> None:
        # n_observed=10, chao1_ci_95_high=10.4 -> f0_upper=0.4 -> 4%.
        # epsilon_chao1=0.05 -> satisfied.
        rows = [
            _row(0, n_observed=10, chao1_high=10.4, p_miss=0.5),
            _row(1, n_observed=10, chao1_high=10.4, p_miss=0.5),
            _row(2, n_observed=10, chao1_high=10.4, p_miss=0.5),
            _row(3, n_observed=10, chao1_high=10.4, p_miss=0.5),
        ]
        cfg = ThresholdConfig(
            stop_when="chao1", epsilon=0.5, epsilon_chao1=0.05,
            k_min=3, k_max=5, require_two_consecutive=False,
        )
        stop, status = replay_arm(rows, cfg)
        assert status == "saturated"
        assert stop == 3

    def test_chao1_unsatisfied_when_upper_ci_too_high(self) -> None:
        # Same n_observed but chao1_high=15 -> f0_upper=5 -> 50%.
        rows = [
            _row(i, n_observed=10, chao1_high=15.0, p_miss=0.001)
            for i in range(5)
        ]
        cfg = ThresholdConfig(
            stop_when="chao1", epsilon=0.001, epsilon_chao1=0.05,
            k_min=3, k_max=5, require_two_consecutive=False,
        )
        stop, status = replay_arm(rows, cfg)
        assert status == "exhausted"

    def test_chao1_zero_observations_fails_safe(self) -> None:
        # n_observed=0 -> degenerate; predicate must NOT fire.
        rows = [_row(i, n_observed=0, chao1_high=0.0) for i in range(5)]
        cfg = ThresholdConfig(
            stop_when="chao1", epsilon=0.5, epsilon_chao1=0.5,
            k_min=3, k_max=5, require_two_consecutive=False,
        )
        stop, status = replay_arm(rows, cfg)
        assert status == "exhausted"


# ---------------------------------------------------------------------------
# replay_arm: both criteria
# ---------------------------------------------------------------------------


class TestReplayArmBoth:
    def test_both_requires_both_to_hold(self) -> None:
        # Trace where GT is happy from pass 3 but Chao1 is happy from pass 4.
        # Under "both" the conjunction first holds at pass 4.
        rows = [
            _row(0, p_miss=0.50, n_observed=10, chao1_high=15.0),
            _row(1, p_miss=0.50, n_observed=10, chao1_high=15.0),
            _row(2, p_miss=0.04, n_observed=10, chao1_high=15.0),  # GT yes, chao1 no
            _row(3, p_miss=0.04, n_observed=10, chao1_high=10.4),  # both yes (pass 4)
            _row(4, p_miss=0.04, n_observed=10, chao1_high=10.4),
        ]
        cfg = ThresholdConfig(
            stop_when="both", epsilon=0.05, epsilon_chao1=0.05,
            k_min=3, k_max=5, require_two_consecutive=False,
        )
        stop, status = replay_arm(rows, cfg)
        assert stop == 4
        assert status == "saturated"


# ---------------------------------------------------------------------------
# Replay matches the production evaluate_stop, pass-by-pass
# ---------------------------------------------------------------------------


def _evaluate_online(
    rows: List[Dict[str, Any]], cfg: ThresholdConfig
) -> int:
    """Run the in-process ``evaluate_stop`` against the trace, threading
    streak state pass-by-pass exactly like the real loop. Returns the
    pass index (1-based) at which the arm becomes saturated, or
    ``k_max + 1`` if it never does.
    """
    arm_cfg = ArmConfig(
        stop_when=cfg.stop_when,
        epsilon=cfg.epsilon,
        epsilon_chao1=cfg.epsilon_chao1,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        require_two_consecutive=cfg.require_two_consecutive,
    )
    prev_status: Any = "active"
    prev_streak = 0
    for row in rows:
        est = ArmEstimate(
            arm=(("column", ("a.csv", "c")), str(row["error_type"])),
            num_runs=int(row["runs_so_far"]),
            n_observed=int(row["n_observed"]),
            frequency_counts={1: int(row["n1"]), 2: int(row["n2"])},
            total_hits=int(row["total_hits"]),
            good_turing_missing_mass=float(row["good_turing_missing_mass"]),
            chao1_estimated_missed=float(row["chao1_estimated_missed"]),
            chao1_estimated_total=float(row["chao1_estimated_total"]),
            chao1_variance=0.0,
            chao1_ci_95_low=float(row["chao1_ci_95_low"]),
            chao1_ci_95_high=float(row["chao1_ci_95_high"]),
            fn_rate_point=0.0,
            fn_rate_ci_95_low=0.0,
            fn_rate_ci_95_high=0.0,
        )
        status, streak = evaluate_stop(
            est, arm_cfg,
            previous_status=prev_status,
            previous_streak=prev_streak,
        )
        if status == "saturated":
            return int(row["runs_so_far"])
        prev_status = status
        prev_streak = streak
    return cfg.k_max + 1  # never saturated


@pytest.mark.parametrize("stop_when", ["gt", "chao1", "both"])
@pytest.mark.parametrize("two", [True, False])
def test_offline_replay_matches_online_evaluate_stop(stop_when, two) -> None:
    # Mixed trace where the predicate becomes satisfied at different
    # passes under different criteria.
    rows = [
        _row(0, p_miss=0.50, n_observed=5, chao1_high=20.0),
        _row(1, p_miss=0.30, n_observed=8, chao1_high=15.0),
        _row(2, p_miss=0.04, n_observed=10, chao1_high=12.0),
        _row(3, p_miss=0.04, n_observed=12, chao1_high=12.4),
        _row(4, p_miss=0.04, n_observed=14, chao1_high=14.4),
        _row(5, p_miss=0.03, n_observed=15, chao1_high=15.5),
    ]
    cfg = ThresholdConfig(
        stop_when=stop_when, epsilon=0.05, epsilon_chao1=0.05,
        k_min=3, k_max=6, require_two_consecutive=two,
    )
    online_pass = _evaluate_online(rows, cfg)
    offline_pass, offline_status = replay_arm(rows, cfg)
    if offline_status == "saturated":
        assert online_pass == offline_pass
    else:
        assert online_pass > cfg.k_max  # production also wouldn't stop


# ---------------------------------------------------------------------------
# Sweep summary metrics
# ---------------------------------------------------------------------------


def _trace_for(arm_etype: str, p_misses: List[float]) -> List[Dict[str, Any]]:
    return [
        _row(i, p_miss=p, arm_etype=arm_etype, n_observed=5 + i,
             total_hits=10 + i)
        for i, p in enumerate(p_misses)
    ]


class TestReplaySweep:
    def test_sweep_reports_saturation_rate(self) -> None:
        # Two arms: arm A saturates fast, arm B never does.
        traces = {
            ("column", ("a.csv", "c1"), "typo"):
                _trace_for("typo", [0.5, 0.5, 0.04, 0.04, 0.04]),
            ("column", ("a.csv", "c2"), "typo"):
                _trace_for("typo", [0.9, 0.8, 0.7, 0.6, 0.5]),
        }
        cfg = ThresholdConfig(
            stop_when="gt", epsilon=0.05, epsilon_chao1=0.05,
            k_min=3, k_max=5, require_two_consecutive=False,
        )
        out = replay_sweep(traces, cfg, k_max_global=5)
        assert out["num_arms"] == 2
        assert out["num_saturated"] == 1
        assert out["saturation_rate"] == 0.5

    def test_passes_saved_zero_when_nothing_saturates(self) -> None:
        traces = {
            ("column", ("a.csv", "c"), "typo"):
                _trace_for("typo", [0.5, 0.5, 0.5, 0.5, 0.5]),
        }
        cfg = ThresholdConfig(
            stop_when="gt", epsilon=0.05, epsilon_chao1=0.05,
            k_min=3, k_max=5, require_two_consecutive=False,
        )
        out = replay_sweep(traces, cfg, k_max_global=5)
        assert out["passes_saved_total"] == 0

    def test_recall_is_one_when_arm_observes_nothing(self) -> None:
        # Edge: an arm that observed 0 species at K_max should
        # contribute neutrally to recall, not skew it down.
        traces = {
            ("column", ("a.csv", "c"), "typo"):
                [_row(i, n_observed=0, total_hits=0, p_miss=0.0)
                 for i in range(5)],
        }
        cfg = ThresholdConfig(
            stop_when="gt", epsilon=0.05, epsilon_chao1=0.05,
            k_min=3, k_max=5, require_two_consecutive=False,
        )
        out = replay_sweep(traces, cfg, k_max_global=5)
        assert out["species_recall_mean"] == 1.0


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


class TestExpandGrid:
    def test_collapses_irrelevant_axes(self) -> None:
        # In gt mode, epsilon_chao1 is irrelevant -> grid collapses
        # along that axis (we don't want 6x duplicate rows).
        cfgs = expand_grid(
            stop_whens=["gt"],
            epsilons=[0.01, 0.05],
            epsilon_chao1s=[0.02, 0.05, 0.10],
            k_mins=[3],
            two_consecs=[True],
            k_max=5,
        )
        # 2 epsilons * 1 epsilon_chao1 (collapsed) * 1 k_min * 1 two = 2
        assert len(cfgs) == 2

    def test_chao1_collapses_epsilon(self) -> None:
        cfgs = expand_grid(
            stop_whens=["chao1"],
            epsilons=[0.01, 0.05, 0.10],
            epsilon_chao1s=[0.05, 0.10],
            k_mins=[3],
            two_consecs=[True],
            k_max=5,
        )
        # 1 epsilon (collapsed) * 2 epsilon_chao1 * 1 * 1 = 2
        assert len(cfgs) == 2

    def test_both_uses_full_product(self) -> None:
        cfgs = expand_grid(
            stop_whens=["both"],
            epsilons=[0.01, 0.05],
            epsilon_chao1s=[0.05, 0.10],
            k_mins=[3, 4],
            two_consecs=[True, False],
            k_max=5,
        )
        # 2 * 2 * 2 * 2 = 16
        assert len(cfgs) == 16


# ---------------------------------------------------------------------------
# Trace loader
# ---------------------------------------------------------------------------


class TestLoadPerPassTraces:
    def test_groups_by_arm_and_sorts_by_pass(self, tmp_path: Path) -> None:
        # Two arms, rows shuffled out of order on disk.
        rows = [
            _row(2, arm_etype="typo"),
            _row(0, arm_etype="typo"),
            _row(1, arm_etype="typo"),
            _row(0, arm_etype="casing"),
            _row(1, arm_etype="casing"),
        ]
        path = tmp_path / "per_pass_per_arm.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        traces, k_max = load_per_pass_traces(path)
        assert k_max == 3
        assert len(traces) == 2
        for arm, recs in traces.items():
            ks = [r["runs_so_far"] for r in recs]
            assert ks == sorted(ks)

    def test_missing_file_raises_friendly_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="--gt-no-stop"):
            load_per_pass_traces(tmp_path / "missing.jsonl")
