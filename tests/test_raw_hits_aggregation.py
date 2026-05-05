"""Focused unit tests for the raw_hits counter in both aggregators.

The Good-Turing identity ``N = sum_x R_x = sum_r r * n_r`` only holds
when ``raw_hits`` is the true emission count: every yield from
``iter_species_keys`` must increment it, with NO within-pass dedup.
A stale or deduped raw_hits silently breaks the missing-mass
denominator and prevents adaptive arms from saturating correctly.

These tests pin the raw_hits semantics for both:

* :func:`src.adaptive.cell_gt.aggregate_runs`
* :func:`src.adaptive.scoped_gt.aggregate_runs_by_scope`

against carefully constructed multi-run / multi-emission scenarios.
"""

from __future__ import annotations

from src.adaptive.cell_gt import aggregate_runs as cell_aggregate_runs
from src.adaptive.scoped_gt import aggregate_runs_by_scope


# ---------------------------------------------------------------------------
# Cell-level aggregator
# ---------------------------------------------------------------------------


def _det_cell(source, etype, row, *, col="c", **kw):
    d = {"source": source, "error_type": etype, "row": row, "columns": [col]}
    d.update(kw)
    return d


class TestCellLevelRawHits:
    def test_single_emission_per_run_raw_hits_equals_runs_found(self) -> None:
        # Standard case: each run emits each species at most once.
        # Under Interp B this yields raw_hits == len(runs_found_in).
        runs = [
            [_det_cell("a.csv", "typo", 1)],
            [_det_cell("a.csv", "typo", 1)],
            [_det_cell("a.csv", "typo", 1)],
        ]
        table = cell_aggregate_runs(runs)
        rec = next(iter(table.values()))
        assert rec.raw_hits == 3
        assert len(rec.runs_found_in) == 3

    def test_multiple_emissions_within_a_pass_count_separately(self) -> None:
        # Same cell flagged twice in pass 0 (e.g., once as typo,
        # once as casing_inconsistency). Under Interp B this counts
        # as TWO raw emissions but ONE entry in runs_found_in.
        runs = [
            [
                _det_cell("a.csv", "typo", 1),
                _det_cell("a.csv", "casing_inconsistency", 1),
            ],
            [],
        ]
        table = cell_aggregate_runs(runs)
        rec = next(iter(table.values()))
        assert rec.raw_hits == 2
        assert rec.runs_found_in == [0]
        # And the per-error-type breakdown is also raw.
        assert rec.raw_hits_by_error_type == {
            "typo": 1,
            "casing_inconsistency": 1,
        }

    def test_affected_rows_expansion_increments_raw_hits_per_row(self) -> None:
        # A single detection with affected_rows=[0,1,2] yields three
        # species; each must contribute 1 raw_hit.
        runs = [
            [_det_cell("a.csv", "row_duplication", 0, col="id",
                       affected_rows=[0, 1, 2])],
        ]
        table = cell_aggregate_runs(runs)
        # Three distinct cells.
        assert len(table) == 3
        for rec in table.values():
            assert rec.raw_hits == 1
            assert rec.runs_found_in == [0]

    def test_per_error_type_raw_hits_sum_to_total(self) -> None:
        # Invariant: sum over error_types of raw_hits_by_error_type
        # must equal raw_hits, for every species. Otherwise the
        # per-type GT denominator drifts away from the cell-level one.
        runs = [
            [
                _det_cell("a.csv", "typo", 1),
                _det_cell("a.csv", "typo", 1),  # duplicate within pass 0
                _det_cell("a.csv", "casing_inconsistency", 1),
            ],
            [_det_cell("a.csv", "typo", 1)],
        ]
        table = cell_aggregate_runs(runs)
        rec = next(iter(table.values()))
        # 3 typo + 1 casing = 4 emissions total.
        assert rec.raw_hits == 4
        assert sum(rec.raw_hits_by_error_type.values()) == rec.raw_hits

    def test_textbook_identity_holds_across_full_population(self) -> None:
        # The most important macro-invariant: across the entire
        # species table, sum(raw_hits) must equal the total number
        # of (cell, error_type) emission events flattened from runs.
        runs = [
            [
                _det_cell("a.csv", "typo", 1),
                _det_cell("a.csv", "typo", 2),
                _det_cell("b.csv", "casing_inconsistency", 5, col="x"),
            ],
            [
                _det_cell("a.csv", "typo", 1),
                _det_cell("a.csv", "casing_inconsistency", 1),
            ],
        ]
        table = cell_aggregate_runs(runs)
        # Hand-count: 3 + 2 = 5 emissions.
        total_raw = sum(r.raw_hits for r in table.values())
        assert total_raw == 5


# ---------------------------------------------------------------------------
# Scope-aware aggregator (per-arm)
# ---------------------------------------------------------------------------


def _det_scope(source, etype, row, *, columns=("c",), **kw):
    d = {
        "source": source,
        "error_type": etype,
        "row": row,
        "columns": list(columns),
    }
    d.update(kw)
    return d


class TestScopedRawHits:
    def test_column_scope_one_emission_per_pass(self) -> None:
        runs = [
            [_det_scope("a.csv", "typo", 1)],
            [_det_scope("a.csv", "typo", 1)],
        ]
        table = aggregate_runs_by_scope(runs, scope_kind="column")
        arm = (("column", ("a.csv", "c")), "typo")
        assert arm in table
        rec = table[arm][(1,)]
        assert rec.raw_hits == 2
        assert rec.runs_found_in == [0, 1]

    def test_column_scope_multi_emission_within_pass(self) -> None:
        # Same locator (row 1 in column c) flagged twice in pass 0.
        # Different error_types -> different arms; each arm should
        # see raw_hits=1 for its own emission.
        runs = [
            [
                _det_scope("a.csv", "typo", 1),
                _det_scope("a.csv", "casing_inconsistency", 1),
            ],
        ]
        table = aggregate_runs_by_scope(runs, scope_kind="column")
        arm_typo = (("column", ("a.csv", "c")), "typo")
        arm_cas = (("column", ("a.csv", "c")), "casing_inconsistency")
        assert table[arm_typo][(1,)].raw_hits == 1
        assert table[arm_cas][(1,)].raw_hits == 1

    def test_column_scope_same_etype_repeated_in_pass(self) -> None:
        # Two emissions of the SAME (arm, locator) in one pass.
        # raw_hits must be 2 (no within-pass dedup); runs_found_in
        # stays deduped at [0].
        runs = [
            [
                _det_scope("a.csv", "typo", 1),
                _det_scope("a.csv", "typo", 1),
            ],
        ]
        table = aggregate_runs_by_scope(runs, scope_kind="column")
        arm = (("column", ("a.csv", "c")), "typo")
        rec = table[arm][(1,)]
        assert rec.raw_hits == 2
        assert rec.runs_found_in == [0]

    def test_column_scope_multi_column_detection_fans_out(self) -> None:
        # One detection listing two columns yields two species; each
        # contributes 1 raw_hit to its respective column-arm.
        runs = [
            [_det_scope("a.csv", "pattern_inconsistency", 1,
                        columns=("c1", "c2"))],
        ]
        table = aggregate_runs_by_scope(runs, scope_kind="column")
        arm_c1 = (("column", ("a.csv", "c1")), "pattern_inconsistency")
        arm_c2 = (("column", ("a.csv", "c2")), "pattern_inconsistency")
        assert table[arm_c1][(1,)].raw_hits == 1
        assert table[arm_c2][(1,)].raw_hits == 1

    def test_column_scope_affected_rows_fan_out(self) -> None:
        # One detection with affected_rows=[1,2] yields two species
        # in the same arm; each gets raw_hits=1.
        runs = [
            [_det_scope("a.csv", "typo", 1, affected_rows=[1, 2])],
        ]
        table = aggregate_runs_by_scope(runs, scope_kind="column")
        arm = (("column", ("a.csv", "c")), "typo")
        assert {loc[0] for loc in table[arm]} == {1, 2}
        for rec in table[arm].values():
            assert rec.raw_hits == 1

    def test_dataset_scope_pools_all_sources(self) -> None:
        # scope_kind="dataset": every (source, row) emission lands in
        # the SAME arm per error_type. raw_hits accumulates across
        # sources without dedup.
        runs = [
            [
                _det_scope("a.csv", "typo", 1),
                _det_scope("b.csv", "typo", 1),
            ],
        ]
        table = aggregate_runs_by_scope(runs, scope_kind="dataset")
        arm = (("dataset", ()), "typo")
        # Two distinct locators (a.csv:1 vs. b.csv:1).
        assert len(table[arm]) == 2
        for rec in table[arm].values():
            assert rec.raw_hits == 1

    def test_textbook_identity_per_arm(self) -> None:
        # Macro invariant: in every arm, sum(raw_hits) over species
        # must equal the total number of yielded emissions for that
        # arm. We verify by re-aggregating with a hand counter.
        runs = [
            [
                _det_scope("a.csv", "typo", 1),
                _det_scope("a.csv", "typo", 2),
                _det_scope("a.csv", "typo", 1),  # repeat in pass 0
            ],
            [
                _det_scope("a.csv", "typo", 1),
                _det_scope("a.csv", "casing_inconsistency", 1),
            ],
        ]
        table = aggregate_runs_by_scope(runs, scope_kind="column")
        arm_typo = (("column", ("a.csv", "c")), "typo")
        # Hand count: 3 emissions in pass 0 + 1 in pass 1 = 4.
        total_typo = sum(r.raw_hits for r in table[arm_typo].values())
        assert total_typo == 4
        arm_cas = (("column", ("a.csv", "c")), "casing_inconsistency")
        total_cas = sum(r.raw_hits for r in table[arm_cas].values())
        assert total_cas == 1

    def test_estimate_arm_total_hits_equals_sum_raw_hits(self) -> None:
        # Critical Interp B invariant: estimate_arm.total_hits MUST
        # equal sum(species.raw_hits) for the arm. If they drift the
        # GT denominator no longer matches the textbook N.
        from src.adaptive.scoped_gt import estimate_by_arm

        runs = [
            [
                _det_scope("a.csv", "typo", 1),
                _det_scope("a.csv", "typo", 1),  # repeat
                _det_scope("a.csv", "typo", 2),
            ],
            [_det_scope("a.csv", "typo", 1)],
        ]
        table = aggregate_runs_by_scope(runs, scope_kind="column")
        arm = (("column", ("a.csv", "c")), "typo")
        # Hand: row 1 raw_hits = 3 (twice in pass 0, once in pass 1),
        # row 2 raw_hits = 1. Total = 4.
        sum_raw = sum(r.raw_hits for r in table[arm].values())
        assert sum_raw == 4

        ests = estimate_by_arm(table, num_runs_by_arm={arm: 2})
        assert ests[arm].total_hits == sum_raw == 4
        # And the textbook identity holds at the estimator level too.
        identity = sum(
            r * n for r, n in ests[arm].frequency_counts.items()
        )
        assert identity == sum_raw

    def test_runs_found_in_deduped_even_when_raw_hits_inflate(self) -> None:
        # Defensive: runs_found_in must NOT count duplicates within a
        # pass, otherwise its diagnostic display is misleading.
        runs = [
            [
                _det_scope("a.csv", "typo", 1),
                _det_scope("a.csv", "typo", 1),
                _det_scope("a.csv", "typo", 1),
            ],
        ]
        table = aggregate_runs_by_scope(runs, scope_kind="column")
        arm = (("column", ("a.csv", "c")), "typo")
        rec = table[arm][(1,)]
        assert rec.raw_hits == 3
        assert rec.runs_found_in == [0]  # NOT [0, 0, 0]
