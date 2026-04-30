"""Tests for the cell-level Good-Turing aggregator.

Key invariant under test
------------------------

Singletons are CELLS detected in only one run. The same cell flagged
with different error_types across runs counts as a single species
(detected in N runs), not as N independent singletons.
"""

from __future__ import annotations

import pytest

from src.adaptive.cell_gt import (
    aggregate_runs,
    estimate,
    estimate_by_group,
    iter_species_keys,
)


def _det(source: str, etype: str, row: int, col: str = "name", **kw):
    d = {"source": source, "error_type": etype, "row": row, "columns": [col]}
    d.update(kw)
    return d


def test_iter_species_keys_excludes_error_type_from_cell_id() -> None:
    keys = list(iter_species_keys(_det("a.csv", "typo", 7, col="x")))
    assert keys == [(("a.csv", "x", 7), "typo")]


def test_same_cell_different_error_types_is_one_species() -> None:
    runs = [
        [_det("a.csv", "typo", 5)],
        [_det("a.csv", "casing_inconsistency", 5)],
    ]
    species = list(aggregate_runs(runs).values())
    est = estimate(species, num_runs=2)

    assert est.n_observed == 1
    assert est.frequency_counts == {1: 0, 2: 1}
    assert est.good_turing_missing_mass == 0.0

    only = species[0]
    assert sorted(only.error_types) == ["casing_inconsistency", "typo"]
    assert only.runs_by_error_type == {"typo": [0], "casing_inconsistency": [1]}


def test_singleton_survives_when_only_one_run_finds_a_cell() -> None:
    runs = [
        [_det("a.csv", "typo", 1), _det("a.csv", "typo", 2)],
        [_det("a.csv", "typo", 1)],
    ]
    species = list(aggregate_runs(runs).values())
    est = estimate(species, num_runs=2)
    assert est.n_observed == 2
    # Row 1 was found by both runs; row 2 only by run 0 -> exactly one
    # singleton.
    assert est.frequency_counts == {1: 1, 2: 1}
    assert est.good_turing_missing_mass == pytest.approx(1 / 3)


def test_dedup_within_a_single_run() -> None:
    runs = [
        [
            _det("a.csv", "typo", 3, col="c"),
            _det("a.csv", "casing_inconsistency", 3, col="c"),
        ],
        [],
    ]
    species = list(aggregate_runs(runs).values())
    est = estimate(species, num_runs=2)
    # One cell is still ONE species (the species table is keyed by
    # cell), but Good-Turing now operates on RAW emission counts:
    # the same cell flagged twice within a single pass contributes
    # ``R_x = 2``, so it lands in n_2, not n_1.
    assert est.n_observed == 1
    assert species[0].raw_hits == 2
    assert est.frequency_counts.get(1, 0) == 0
    assert est.frequency_counts.get(2, 0) == 1
    # Per-pass dedup is preserved on ``runs_found_in`` for diagnostics.
    assert species[0].runs_found_in == [0]


def test_per_error_type_breakdown() -> None:
    runs = [
        [_det("a.csv", "typo", 5)],
        [_det("a.csv", "casing_inconsistency", 5)],
    ]
    species = list(aggregate_runs(runs).values())
    by_type = estimate_by_group(species, num_runs=2, key="error_type")

    # Each type sees the cell in exactly one run -> per-type singleton.
    assert by_type["typo"].n_observed == 1
    assert by_type["typo"].frequency_counts[1] == 1
    assert by_type["casing_inconsistency"].frequency_counts[1] == 1


def test_per_source_breakdown_partitions_cells() -> None:
    runs = [
        [_det("a.csv", "typo", 1), _det("b.csv", "typo", 1)],
        [_det("a.csv", "typo", 1)],
    ]
    species = list(aggregate_runs(runs).values())
    by_src = estimate_by_group(species, num_runs=2, key="source")
    assert set(by_src) == {"a.csv", "b.csv"}
    assert by_src["a.csv"].n_observed == 1
    assert by_src["b.csv"].n_observed == 1


def test_group_level_finding_expands_rows() -> None:
    runs = [
        [
            _det("a.csv", "row_duplication", 0, col="id", affected_rows=[0, 1, 2]),
        ],
        [_det("a.csv", "row_duplication", 0, col="id", affected_rows=[0, 1])],
    ]
    species = list(aggregate_runs(runs).values())
    est = estimate(species, num_runs=2)
    # Three cells observed (rows 0/1/2). Rows 0 and 1 in both runs;
    # row 2 only in run 0 -> n1=1, n2=2.
    assert est.n_observed == 3
    assert est.frequency_counts == {1: 1, 2: 2}
