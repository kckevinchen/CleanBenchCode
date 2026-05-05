"""Tests for :mod:`src.adaptive.inner_loop`.

The inner loop now consumes the unified :class:`ArmConfig` from
``scoped_gt`` so the row-dup detector and the outer per-arm loop
share one FN-control surface. Tests cover:

* Each ``stop_when`` mode (``gt`` / ``chao1`` / ``both``).
* The two-consecutive-no-new-species safeguard.
* ``k_max`` cap when species keep arriving.
* ``k_min`` floor.
* Edge cases (empty species, ``k_max=1``, missing config fields).
* The ``merge_partitions`` union-find helper.
"""

from __future__ import annotations

import pytest

from src.adaptive.inner_loop import (
    InnerLoopResult,
    merge_partitions,
    run_adaptive,
)
from src.adaptive.scoped_gt import ArmConfig


def _cfg(**overrides) -> ArmConfig:
    """Build a fully-populated ArmConfig for tests."""
    return ArmConfig(**overrides).merge_into(ArmConfig.default())


# ---------------------------------------------------------------------------
# run_adaptive - basic stop rules
# ---------------------------------------------------------------------------


class TestRunAdaptiveStopRules:
    def test_two_consecutive_no_new_triggers_stop(self) -> None:
        """Identical species across runs => loop converges quickly."""
        calls = []

        def sample(idx: int):
            calls.append(idx)
            return frozenset({"a", "b", "c"})

        cfg = _cfg(
            k_max=5, k_min=2, epsilon=0.05, epsilon_chao1=0.05,
            stop_when="gt", require_two_consecutive=True,
        )
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)

        # k_min=2 + two-consecutive => stops after exactly 3 runs.
        assert result.runs_completed == 3
        assert calls == [0, 1, 2]
        assert result.stopped_reason == "two_consecutive"
        assert result.union_species() == frozenset({"a", "b", "c"})

    def test_stop_when_gt_uses_good_turing_only(self) -> None:
        """With ``stop_when='gt'`` the Chao1 ratio is ignored."""

        def sample(idx: int):
            return frozenset({"a"})

        cfg = _cfg(
            k_max=4, k_min=2,
            epsilon=0.99, epsilon_chao1=1e-6,  # impossible chao1 threshold
            stop_when="gt", require_two_consecutive=False,
        )
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)
        # Chao1 would never satisfy, but stop_when=gt only checks GT
        # => epsilon=0.99 trivially met after k_min runs.
        assert result.runs_completed == 2
        assert result.stopped_reason == "gt"

    def test_stop_when_chao1_ignores_good_turing(self) -> None:
        """With ``stop_when='chao1'`` the GT mass is ignored."""

        def sample(idx: int):
            return frozenset({"a"})

        cfg = _cfg(
            k_max=4, k_min=2,
            epsilon=1e-6, epsilon_chao1=0.99,  # impossible GT threshold
            stop_when="chao1", require_two_consecutive=False,
        )
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)
        assert result.runs_completed == 2
        assert result.stopped_reason == "chao1"

    def test_stop_when_both_requires_both_to_hold(self) -> None:
        """``stop_when='both'`` only saturates when GT *and* Chao1 do."""

        # Each run yields a fresh species => n1 grows unboundedly,
        # GT missing mass = n1 / N stays near 1.0 forever. Chao1
        # estimator is also large but with epsilon_chao1=0.99 it
        # could in principle be met. With epsilon=0.05 GT is never
        # met, so 'both' must keep running.
        def sample(idx: int):
            return frozenset({f"sp_{idx}"})

        cfg = _cfg(
            k_max=4, k_min=2,
            epsilon=0.05, epsilon_chao1=0.99,
            stop_when="both", require_two_consecutive=False,
        )
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)
        assert result.runs_completed == 4
        assert result.stopped_reason == "k_max"

    def test_stop_when_both_saturates_when_both_met(self) -> None:
        """Sanity: with very loose thresholds, 'both' saturates."""

        def sample(idx: int):
            return frozenset({"a"})

        cfg = _cfg(
            k_max=5, k_min=2,
            epsilon=0.99, epsilon_chao1=0.99,
            stop_when="both", require_two_consecutive=False,
        )
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)
        assert result.runs_completed == 2
        assert result.stopped_reason == "both"

    def test_unknown_stop_when_raises(self) -> None:
        cfg = _cfg(
            k_max=2, k_min=2,
            epsilon=0.05, epsilon_chao1=0.05,
            stop_when="bogus",
        )

        def sample(idx: int):
            return frozenset({"a"})

        with pytest.raises(ValueError, match="unknown stop_when"):
            run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)

    def test_missing_config_field_raises(self) -> None:
        # ArmConfig() has all fields None until merged with default().
        # Pass it raw to the loop and verify the guardrail fires.
        bad_cfg = ArmConfig()
        with pytest.raises(ValueError, match="missing required fields"):
            run_adaptive(
                arm_id="t",
                sample_fn=lambda i: frozenset({"a"}),
                cfg=bad_cfg,
            )


# ---------------------------------------------------------------------------
# run_adaptive - cap / floor / edge cases
# ---------------------------------------------------------------------------


class TestRunAdaptiveBoundaries:
    def test_runs_to_k_max_when_new_species_keep_arriving(self) -> None:
        def sample(idx: int):
            return frozenset({f"sp_{idx}"})

        cfg = _cfg(k_max=3, k_min=2, stop_when="gt")
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)
        assert result.runs_completed == 3
        assert result.stopped_reason == "k_max"
        assert result.union_species() == frozenset({"sp_0", "sp_1", "sp_2"})

    def test_respects_k_min(self) -> None:
        """Run 0 alone cannot trigger stop, even with permissive epsilon."""
        calls = []

        def sample(idx: int):
            calls.append(idx)
            return frozenset({"a"})

        cfg = _cfg(
            k_max=3, k_min=2,
            epsilon=0.99, epsilon_chao1=0.99,
            stop_when="gt", require_two_consecutive=False,
        )
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)
        assert result.runs_completed >= 2

    def test_empty_species_does_not_crash(self) -> None:
        def sample(idx: int):
            return frozenset()

        cfg = _cfg(
            k_max=3, k_min=2, stop_when="gt", require_two_consecutive=True,
        )
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)
        # Empty species: every run counts as "no new" => after 2 runs
        # consecutive_no_new=2 satisfies the two-consecutive predicate.
        assert result.runs_completed == 2
        assert result.union_species() == frozenset()
        assert result.species_to_runs == {}

    def test_k_max_one_runs_exactly_once(self) -> None:
        def sample(idx: int):
            return frozenset({"a"})

        cfg = _cfg(k_max=1, k_min=1, stop_when="gt")
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)
        assert result.runs_completed == 1

    def test_records_per_species_run_indices(self) -> None:
        seen = [
            frozenset({"a", "b"}),
            frozenset({"a", "c"}),
            frozenset({"a", "b", "c"}),
        ]

        def sample(idx: int):
            return seen[idx]

        cfg = _cfg(
            k_max=3, k_min=3, stop_when="gt", require_two_consecutive=False,
        )
        result = run_adaptive(arm_id="t", sample_fn=sample, cfg=cfg)
        assert result.species_to_runs["a"] == [0, 1, 2]
        assert result.species_to_runs["b"] == [0, 2]
        assert result.species_to_runs["c"] == [1, 2]


# ---------------------------------------------------------------------------
# merge_partitions
# ---------------------------------------------------------------------------


class TestMergePartitions:
    def test_unions_overlapping_classes(self) -> None:
        partitions = [
            [frozenset({"A", "B"})],
            [frozenset({"B", "C"})],
        ]
        merged = merge_partitions(partitions)
        assert len(merged) == 1
        assert merged[0] == frozenset({"A", "B", "C"})

    def test_keeps_disjoint_classes_separate(self) -> None:
        partitions = [
            [frozenset({"A", "B"}), frozenset({"X", "Y"})],
            [frozenset({"A", "B"}), frozenset({"X", "Y"})],
        ]
        merged = merge_partitions(partitions)
        assert len(merged) == 2
        classes = sorted([sorted(c) for c in merged])
        assert classes == [["A", "B"], ["X", "Y"]]

    def test_recall_first_single_run_merge_wins(self) -> None:
        partitions = [
            [frozenset({"A"}), frozenset({"B"})],
            [frozenset({"A", "B"})],
        ]
        merged = merge_partitions(partitions)
        assert len(merged) == 1
        assert merged[0] == frozenset({"A", "B"})

    def test_handles_singletons(self) -> None:
        partitions = [[frozenset({"A"})], [frozenset({"B"})]]
        merged = merge_partitions(partitions)
        assert sorted(sorted(c) for c in merged) == [["A"], ["B"]]

    def test_empty_input(self) -> None:
        assert merge_partitions([]) == []
