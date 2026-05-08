"""Tests for the scope-aware Good-Turing estimator
(:mod:`src.adaptive.scoped_gt`).

Each scope_kind gets a focused test that:
1. Pins down the species shape produced by :func:`iter_species_keys`.
2. Verifies aggregation across runs bins species correctly.
3. Sanity-checks the per-arm estimator and stop predicate.

We also test the config hierarchy and the scheduler.
"""

from __future__ import annotations

import pytest

from src.adaptive.scoped_gt import (
    ArmConfig,
    ArmKey,
    ScopeKey,
    aggregate_runs_by_scope,
    estimate_arm,
    estimate_by_arm,
    evaluate_stop,
    is_applicable,
    iter_species_keys,
    next_pass_targets,
    resolve_arm_config,
    run_estimator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _det(**kw):
    """Detection record helper. Caller supplies whatever fields the
    target scope_kind needs."""
    base = {"source": "a.csv", "error_type": "typo"}
    base.update(kw)
    return base


def _arm(scope_kind, scope_id, error_type) -> ArmKey:
    return ((scope_kind, tuple(scope_id)), error_type)


# ---------------------------------------------------------------------------
# iter_species_keys: per-scope_kind dispatch
# ---------------------------------------------------------------------------


def test_column_species_one_arm_per_column() -> None:
    keys = list(
        iter_species_keys(
            _det(row=3, columns=["name", "addr"]),
            scope_kind="column",
        )
    )
    # Two columns -> two arms, one species each.
    arms = {k[0] for k in keys}
    assert arms == {
        ("column", ("a.csv", "name")),
        ("column", ("a.csv", "addr")),
    }
    # Locator is (row,) and error_type is preserved.
    assert all(k[1] == (3,) for k in keys)
    assert all(k[2] == "typo" for k in keys)


def test_column_species_skips_row_only_detection() -> None:
    # No columns -> a column-scope species is undefined.
    assert list(
        iter_species_keys(_det(row=3, columns=[]), scope_kind="column")
    ) == []


def test_column_species_expands_affected_rows() -> None:
    keys = list(
        iter_species_keys(
            _det(row=3, columns=["name"], affected_rows=[3, 5, 7]),
            scope_kind="column",
        )
    )
    # 1 column x 3 rows = 3 species (the primary row is in affected_rows).
    assert len(keys) == 3
    assert sorted(k[1] for k in keys) == [(3,), (5,), (7,)]


# ---------------------------------------------------------------------------
# Dataset scope: one arm per error_type, locator preserves cell identity
# ---------------------------------------------------------------------------


def test_dataset_species_singleton_scope_id() -> None:
    keys = list(
        iter_species_keys(
            _det(row=3, columns=["name"]),
            scope_kind="dataset",
        )
    )
    # Exactly one arm; scope_id is the empty tuple (no partitioning).
    assert len(keys) == 1
    assert keys[0][0] == ("dataset", ())
    # Locator preserves source + column + row.
    assert keys[0][1] == ("a.csv", "name", 3)
    assert keys[0][2] == "typo"


def test_dataset_species_fanout_across_columns_and_rows() -> None:
    keys = list(
        iter_species_keys(
            _det(row=3, columns=["a", "b"], affected_rows=[3, 4]),
            scope_kind="dataset",
        )
    )
    # 2 columns x 2 rows = 4 species, all under the same scope_id.
    assert len(keys) == 4
    assert all(k[0] == ("dataset", ()) for k in keys)
    locs = sorted(k[1] for k in keys)
    assert locs == [
        ("a.csv", "a", 3), ("a.csv", "a", 4),
        ("a.csv", "b", 3), ("a.csv", "b", 4),
    ]


def test_dataset_species_row_only_falls_back_to_row_locator() -> None:
    keys = list(
        iter_species_keys(
            _det(row=3, columns=[]), scope_kind="dataset"
        )
    )
    # No columns -> locator is (source, row).
    assert len(keys) == 1
    assert keys[0][0] == ("dataset", ())
    assert keys[0][1] == ("a.csv", 3)


def test_dataset_species_skips_when_source_or_row_missing() -> None:
    assert list(
        iter_species_keys({"error_type": "typo", "row": 1}, scope_kind="dataset")
    ) == []
    assert list(
        iter_species_keys({"source": "a.csv", "error_type": "typo"},
                          scope_kind="dataset")
    ) == []


def test_dataset_aggregation_one_arm_per_error_type() -> None:
    runs = [
        [_det(row=1, columns=["c0"]),
         _det(row=2, columns=["c0"], error_type="casing")],
        [_det(row=1, columns=["c0"]),  # duplicate -> bumps freq
         _det(row=3, columns=["c1"])],
    ]
    table = aggregate_runs_by_scope(runs, scope_kind="dataset")
    # Two arms total: typo and casing, both under scope_id=().
    arms = sorted(table.keys())
    assert arms == [
        (("dataset", ()), "casing"),
        (("dataset", ()), "typo"),
    ]
    typo_arm = table[(("dataset", ()), "typo")]
    # 3 distinct species: (a.csv,c0,1), (a.csv,c0,1) again, (a.csv,c1,3).
    # Distinct: 2 species; (a.csv,c0,1) seen in both runs => freq=2.
    species_freqs = sorted(s.frequency for s in typo_arm.values())
    assert species_freqs == [1, 2]


def test_column_pair_species_canonical_sort_and_fanout() -> None:
    keys = list(
        iter_species_keys(
            _det(row=3, columns=["b", "a", "c"]),
            scope_kind="column_pair",
        )
    )
    # 3 columns -> C(3,2) = 3 pairs, sorted alphabetically inside each arm.
    arms = sorted(k[0] for k in keys)
    assert arms == [
        ("column_pair", ("a.csv", "a", "b")),
        ("column_pair", ("a.csv", "a", "c")),
        ("column_pair", ("a.csv", "b", "c")),
    ]
    # Single-column detection produces no column_pair species.
    assert list(
        iter_species_keys(
            _det(row=3, columns=["a"]), scope_kind="column_pair"
        )
    ) == []


def test_row_species_one_per_row() -> None:
    keys = list(
        iter_species_keys(
            _det(row=3, columns=[], affected_rows=[3, 8]),
            scope_kind="row",
        )
    )
    arms = {k[0] for k in keys}
    assert arms == {("row", ("a.csv",))}
    assert sorted(k[1] for k in keys) == [(3,), (8,)]


def test_csv_species_pair_locator_for_dup() -> None:
    keys = list(
        iter_species_keys(
            _det(row=3, other_row=7, error_type="exact_duplicate"),
            scope_kind="csv",
        )
    )
    assert len(keys) == 1
    scope, locator, etype = keys[0]
    assert scope == ("csv", ("a.csv",))
    # Pair is sorted -> stable across run-to-run swap of "primary".
    assert locator == (3, 7)
    assert etype == "exact_duplicate"

    # Reversing the pair across runs must produce the same locator.
    keys2 = list(
        iter_species_keys(
            _det(row=7, other_row=3, error_type="exact_duplicate"),
            scope_kind="csv",
        )
    )
    assert keys2[0][1] == (3, 7)


def test_csv_species_column_locator_for_schema() -> None:
    keys = list(
        iter_species_keys(
            _det(
                row=None,
                columns=["zip"],
                error_type="schema_violation",
            ),
            scope_kind="csv",
        )
    )
    assert keys == [(("csv", ("a.csv",)), ("zip",), "schema_violation")]


def test_cross_csv_species_canonical_pair_and_locator() -> None:
    keys = list(
        iter_species_keys(
            _det(
                source="orders.csv",
                other_source="customers.csv",
                row=10,
                other_row=42,
                error_type="fk_violation",
            ),
            scope_kind="cross_csv",
        )
    )
    assert len(keys) == 1
    scope, locator, _ = keys[0]
    # Sources sorted -> "customers.csv" first.
    assert scope == ("cross_csv", ("customers.csv", "orders.csv"))
    # Locator must mirror that sorted order: customer row first,
    # order row second, regardless of which side the detection
    # called primary.
    assert locator == (42, 10)

    keys2 = list(
        iter_species_keys(
            _det(
                source="customers.csv",
                other_source="orders.csv",
                row=42,
                other_row=10,
                error_type="fk_violation",
            ),
            scope_kind="cross_csv",
        )
    )
    assert keys2[0][1] == (42, 10), "locator must be canonical across runs"


def test_workbook_species_cross_sheet_pair() -> None:
    keys = list(
        iter_species_keys(
            _det(
                workbook="Q1.xlsx",
                sheet="b",
                other_sheet="a",
                row=None,
                error_type="cross_sheet_duplicate",
            ),
            scope_kind="workbook",
        )
    )
    assert keys == [
        (
            ("workbook", ("Q1.xlsx",)),
            ("a", "b"),
            "cross_sheet_duplicate",
        )
    ]


def test_multi_table_species_canonical_tuple() -> None:
    keys = list(
        iter_species_keys(
            _det(
                sources=["c.csv", "a.csv", "b.csv"],
                source_rows=[["b.csv", 5], ["a.csv", 1], ["c.csv", 9]],
                error_type="three_way_join_violation",
            ),
            scope_kind="multi_table",
        )
    )
    assert len(keys) == 1
    scope, locator, _ = keys[0]
    assert scope == ("multi_table", ("a.csv", "b.csv", "c.csv"))
    assert locator == (("a.csv", 1), ("b.csv", 5), ("c.csv", 9))


def test_iter_species_keys_unknown_scope_kind_raises() -> None:
    with pytest.raises(ValueError):
        list(
            iter_species_keys(
                _det(row=1, columns=["x"]),
                scope_kind="bogus",  # type: ignore[arg-type]
            )
        )


# ---------------------------------------------------------------------------
# Aggregation: K runs -> per-arm species table
# ---------------------------------------------------------------------------


def test_aggregate_runs_independent_arms_per_column() -> None:
    runs = [
        # Run 0: typo on (a.csv, name) row 1; casing on (a.csv, addr) row 1.
        [
            _det(row=1, columns=["name"], error_type="typo"),
            _det(row=1, columns=["addr"], error_type="casing"),
        ],
        # Run 1: typo on (a.csv, name) row 1 (stable); typo new at row 2.
        [
            _det(row=1, columns=["name"], error_type="typo"),
            _det(row=2, columns=["name"], error_type="typo"),
        ],
    ]
    table = aggregate_runs_by_scope(runs, scope_kind="column")

    name_typo = _arm("column", ("a.csv", "name"), "typo")
    addr_casing = _arm("column", ("a.csv", "addr"), "casing")

    # Three arms total: name/typo, name/casing? No - addr only got casing.
    assert set(table.keys()) == {name_typo, addr_casing}

    # Within name_typo arm, two species: row 1 (in both runs), row 2 (run 1).
    assert {rec.locator for rec in table[name_typo].values()} == {(1,), (2,)}
    assert table[name_typo][(1,)].frequency == 2
    assert table[name_typo][(2,)].frequency == 1


def test_aggregate_dedups_within_a_run() -> None:
    # Same arm + locator detected twice in the same run -> count once.
    runs = [
        [
            _det(row=1, columns=["x"], error_type="typo"),
            _det(row=1, columns=["x"], error_type="typo"),
        ]
    ]
    table = aggregate_runs_by_scope(runs, scope_kind="column")
    arm = _arm("column", ("a.csv", "x"), "typo")
    assert table[arm][(1,)].frequency == 1


# ---------------------------------------------------------------------------
# Estimator: per-arm Chao1 + GT
# ---------------------------------------------------------------------------


def _make_arm_with_freqs(num_runs: int, freq_pattern):
    """Build a (table, arm) pair where the arm has species with the
    given frequency pattern (list of run-counts per species)."""
    runs = [[] for _ in range(num_runs)]
    arm_key = _arm("column", ("a.csv", "x"), "typo")
    for sp_idx, freq in enumerate(freq_pattern):
        for r in range(freq):
            runs[r].append(
                {
                    "source": "a.csv",
                    "error_type": "typo",
                    "row": sp_idx,
                    "columns": ["x"],
                }
            )
    table = aggregate_runs_by_scope(runs, scope_kind="column")
    return table, arm_key


def test_estimate_arm_no_singletons_no_missed() -> None:
    # 5 species each found in all 3 runs -> n1 = 0 -> missed = 0.
    table, arm = _make_arm_with_freqs(3, [3, 3, 3, 3, 3])
    est = estimate_arm(arm, table[arm].values(), num_runs=3)
    assert est.n_observed == 5
    assert est.frequency_counts == {1: 0, 2: 0, 3: 5}
    assert est.good_turing_missing_mass == 0.0
    assert est.chao1_estimated_missed == 0.0
    assert est.fn_rate_point == 0.0


def test_estimate_arm_classic_chao1() -> None:
    # n1 = 4, n2 = 2 -> missed = 16 / 4 = 4.
    table, arm = _make_arm_with_freqs(
        3, [1, 1, 1, 1, 2, 2, 3]
    )
    est = estimate_arm(arm, table[arm].values(), num_runs=3)
    assert est.n_observed == 7
    assert est.frequency_counts == {1: 4, 2: 2, 3: 1}
    assert est.chao1_estimated_missed == pytest.approx(4.0)
    # FN rate = 4 / (7 + 4) = 4/11.
    assert est.fn_rate_point == pytest.approx(4.0 / 11.0)


def test_estimate_arm_bias_corrected_when_n2_zero() -> None:
    # n1 = 3, n2 = 0 -> missed = 3 * 2 / 2 = 3.
    table, arm = _make_arm_with_freqs(3, [1, 1, 1, 3])
    est = estimate_arm(arm, table[arm].values(), num_runs=3)
    assert est.frequency_counts == {1: 3, 2: 0, 3: 1}
    assert est.chao1_estimated_missed == pytest.approx(3.0)


def test_estimate_arm_empty_returns_zeros() -> None:
    arm = _arm("column", ("a.csv", "x"), "typo")
    est = estimate_arm(arm, [], num_runs=3)
    assert est.n_observed == 0
    assert est.fn_rate_point == 0.0
    assert est.status == "active"


def test_estimate_by_arm_does_not_pool() -> None:
    # Two arms with different K. Per-arm K must drive each estimate.
    runs = [
        # Run 0: arm_x detects row 1; arm_y detects row 100.
        [
            _det(row=1, columns=["x"]),
            _det(row=100, columns=["y"]),
        ],
        # Run 1: only arm_x is probed (arm_y was pruned).
        [_det(row=2, columns=["x"])],
    ]
    table = aggregate_runs_by_scope(runs, scope_kind="column")
    arm_x = _arm("column", ("a.csv", "x"), "typo")
    arm_y = _arm("column", ("a.csv", "y"), "typo")

    estimates = estimate_by_arm(
        table, num_runs_by_arm={arm_x: 2, arm_y: 1}
    )
    # Arm Y had one run, one species -> n1 = 1 with K=1. The
    # frequency-of-frequencies vector must be bounded by K_a, not by
    # max(K_a) across arms.
    assert max(estimates[arm_y].frequency_counts.keys()) == 1
    assert max(estimates[arm_x].frequency_counts.keys()) == 2


# ---------------------------------------------------------------------------
# ArmConfig hierarchy
# ---------------------------------------------------------------------------


def test_resolve_arm_config_per_arm_overrides_all_layers() -> None:
    arm = _arm("column", ("a.csv", "x"), "typo")
    cfg = resolve_arm_config(
        arm,
        global_default=ArmConfig(
            epsilon=0.1, k_min=3, k_max=10, require_two_consecutive=True
        ),
        by_scope_kind={"column": ArmConfig(epsilon=0.08)},
        by_error_type={"typo": ArmConfig(epsilon=0.06)},
        by_arm={arm: ArmConfig(epsilon=0.01, k_max=20)},
    )
    # Per-arm wins for epsilon and k_max; per-error_type and per-scope
    # are overridden by per-arm where set; lower-priority defaults
    # propagate where the higher layers said None.
    assert cfg.epsilon == pytest.approx(0.01)
    assert cfg.k_max == 20
    assert cfg.k_min == 3
    assert cfg.require_two_consecutive is True


def test_resolve_arm_config_falls_through_to_lower_layers() -> None:
    arm = _arm("column", ("a.csv", "x"), "typo")
    cfg = resolve_arm_config(
        arm,
        global_default=ArmConfig(
            epsilon=0.1, k_min=3, k_max=10, require_two_consecutive=True
        ),
        by_error_type={"typo": ArmConfig(k_min=2)},
    )
    # Only per-error-type overrides k_min; everything else from default.
    assert cfg.epsilon == pytest.approx(0.1)
    assert cfg.k_min == 2


def test_default_armconfig_is_complete() -> None:
    cfg = ArmConfig.default()
    assert cfg.epsilon is not None
    assert cfg.k_min is not None
    assert cfg.k_max is not None
    assert cfg.require_two_consecutive is not None


# ---------------------------------------------------------------------------
# Stop predicate
# ---------------------------------------------------------------------------


def _est(num_runs, n_obs, p_miss, **kw) -> "ArmEstimate":
    """Minimal ArmEstimate stub for stop-predicate tests."""
    from src.adaptive.scoped_gt import ArmEstimate
    arm = _arm("column", ("a.csv", "x"), "typo")
    base = dict(
        arm=arm,
        num_runs=num_runs,
        n_observed=n_obs,
        frequency_counts={r: 0 for r in range(1, num_runs + 1)},
        total_hits=0,
        good_turing_missing_mass=p_miss,
        chao1_estimated_missed=0.0,
        chao1_estimated_total=0.0,
        chao1_variance=0.0,
        chao1_ci_95_low=0.0,
        chao1_ci_95_high=0.0,
        fn_rate_point=0.0,
        fn_rate_ci_95_low=0.0,
        fn_rate_ci_95_high=0.0,
    )
    base.update(kw)
    return ArmEstimate(**base)


def _cfg(**kw) -> ArmConfig:
    return ArmConfig(**kw).merge_into(ArmConfig.default())


def test_stop_active_when_predicate_not_met() -> None:
    # p_miss high enough to fail the threshold -> still active.
    status, streak = evaluate_stop(
        _est(5, 10, p_miss=0.5), _cfg()
    )
    assert status == "active"
    assert streak == 0


def test_stop_requires_two_consecutive_when_enabled() -> None:
    cfg = _cfg(require_two_consecutive=True)
    # First saturated pass -> still active, streak = 1.
    s1, st1 = evaluate_stop(_est(3, 10, p_miss=0.01), cfg)
    assert s1 == "active"
    assert st1 == 1
    # Second consecutive saturated pass -> saturated.
    s2, st2 = evaluate_stop(
        _est(4, 10, p_miss=0.01),
        cfg,
        previous_status=s1,
        previous_streak=st1,
    )
    assert s2 == "saturated"
    assert st2 == 2


def test_stop_one_consecutive_when_two_disabled() -> None:
    cfg = _cfg(require_two_consecutive=False)
    s, _ = evaluate_stop(_est(3, 10, p_miss=0.01), cfg)
    assert s == "saturated"


def test_stop_streak_resets_when_predicate_breaks() -> None:
    cfg = _cfg(require_two_consecutive=True)
    # Saturated -> streak 1.
    s1, st1 = evaluate_stop(_est(3, 10, p_miss=0.01), cfg)
    # Non-saturated next pass (e.g. a new defect surfaced) -> reset.
    s2, st2 = evaluate_stop(
        _est(4, 12, p_miss=0.20),
        cfg,
        previous_status=s1,
        previous_streak=st1,
    )
    assert s2 == "active"
    assert st2 == 0


def test_stop_exhausted_when_k_max_with_high_p_miss() -> None:
    # K_max reached but missing-mass still above epsilon -> exhausted.
    cfg = _cfg(k_max=3, epsilon=0.05)
    s, _ = evaluate_stop(_est(3, 10, p_miss=0.5), cfg)
    assert s == "exhausted"


def test_stop_saturated_is_sticky() -> None:
    cfg = _cfg()
    s, st = evaluate_stop(
        _est(5, 100, p_miss=0.5),
        cfg,
        previous_status="saturated",
        previous_streak=2,
    )
    assert s == "saturated"
    assert st == 2  # untouched


# ---------------------------------------------------------------------------
# stop_when modes (gt / chao1 / both)
# ---------------------------------------------------------------------------


def test_stop_when_gt_only_ignores_chao1_count() -> None:
    """`stop_when="gt"` only consults the GT mass criterion.

    Scenario: GT mass is tiny (so the *next emission* is unlikely to
    be a never-seen species) but Chao1 says many rare species are
    still missing. Pure-GT mode stops; pure-Chao1 mode does not.
    """
    # n_observed=100, chao1 upper-95 = 200 -> missing fraction = 1.0
    # (way above any reasonable epsilon_chao1).
    est = _est(
        4, 100, p_miss=0.01,
        chao1_ci_95_high=200.0,
    )
    cfg_gt = _cfg(stop_when="gt", require_two_consecutive=False)
    cfg_chao = _cfg(stop_when="chao1", require_two_consecutive=False)
    cfg_both = _cfg(stop_when="both", require_two_consecutive=False)

    assert evaluate_stop(est, cfg_gt)[0] == "saturated"
    assert evaluate_stop(est, cfg_chao)[0] == "active"
    assert evaluate_stop(est, cfg_both)[0] == "active"


def test_stop_when_chao1_only_ignores_gt_mass() -> None:
    """`stop_when="chao1"` only consults the Chao1 count criterion.

    Scenario: Chao1 says nothing material is missing (count-wise) but
    GT mass is high because hot singletons keep showing up. Pure-Chao1
    mode stops; pure-GT mode does not.
    """
    # n_observed=100, chao1 upper-95 = 100 -> missing fraction = 0.
    est = _est(
        4, 100, p_miss=0.5,
        chao1_ci_95_high=100.0,
    )
    cfg_gt = _cfg(stop_when="gt", require_two_consecutive=False)
    cfg_chao = _cfg(stop_when="chao1", require_two_consecutive=False)
    cfg_both = _cfg(stop_when="both", require_two_consecutive=False)

    assert evaluate_stop(est, cfg_gt)[0] == "active"
    assert evaluate_stop(est, cfg_chao)[0] == "saturated"
    assert evaluate_stop(est, cfg_both)[0] == "active"


def test_stop_when_both_requires_both_criteria() -> None:
    """`stop_when="both"` is the conjunction of GT and Chao1."""
    # Both criteria pass: low p_miss AND tight chao1 upper bound.
    est_both_pass = _est(
        4, 100, p_miss=0.01,
        chao1_ci_95_high=102.0,  # missing fraction = 2%
    )
    cfg = _cfg(
        stop_when="both",
        epsilon=0.05,
        epsilon_chao1=0.05,
        require_two_consecutive=False,
    )
    assert evaluate_stop(est_both_pass, cfg)[0] == "saturated"

    # Either alone failing keeps it active.
    est_only_gt_pass = _est(
        4, 100, p_miss=0.01, chao1_ci_95_high=200.0,
    )
    est_only_chao_pass = _est(
        4, 100, p_miss=0.5, chao1_ci_95_high=102.0,
    )
    assert evaluate_stop(est_only_gt_pass, cfg)[0] == "active"
    assert evaluate_stop(est_only_chao_pass, cfg)[0] == "active"


def test_stop_when_chao1_uses_upper_ci_not_point() -> None:
    """The Chao1 criterion is on the *upper-95* CI of the missing count,
    not the point estimate; this is what makes it conservative."""
    # Point estimate says 0 missing, but upper-95 = 110 -> 10% missing.
    est = _est(
        4, 100, p_miss=0.5,
        chao1_estimated_missed=0.0,
        chao1_ci_95_high=110.0,
    )
    cfg_strict = _cfg(
        stop_when="chao1",
        epsilon_chao1=0.05,
        require_two_consecutive=False,
    )
    cfg_loose = _cfg(
        stop_when="chao1",
        epsilon_chao1=0.20,
        require_two_consecutive=False,
    )
    assert evaluate_stop(est, cfg_strict)[0] == "active"
    assert evaluate_stop(est, cfg_loose)[0] == "saturated"


def test_stop_when_both_saturates_when_S_obs_zero_after_k_min() -> None:
    """A column with no errors should saturate, not exhaust.

    Regression: when S_obs=0 the Chao1 missing-fraction is degenerate.
    We define the ratio to be 0 (no species observed -> no species
    inferred missing) so that K consecutive empty passes saturate
    under stop_when=both. The k_min and two-consecutive guards on
    the caller prevent a premature stop at K=1.
    """
    cfg = _cfg(
        stop_when="both",
        epsilon=0.05,
        epsilon_chao1=0.05,
        k_min=3,
        k_max=15,
        require_two_consecutive=True,
    )

    # Pass 1 (K=1): below k_min, criterion holds vacuously but the
    # k_min gate keeps us active. Streak does not advance.
    s1, st1 = evaluate_stop(
        _est(1, 0, p_miss=0.0, chao1_ci_95_high=0.0), cfg
    )
    assert s1 == "active"
    assert st1 == 0

    # Pass 2 (K=2): still below k_min.
    s2, st2 = evaluate_stop(
        _est(2, 0, p_miss=0.0, chao1_ci_95_high=0.0),
        cfg,
        previous_status=s1,
        previous_streak=st1,
    )
    assert s2 == "active"
    assert st2 == 0

    # Pass 3 (K=3): k_min reached, criterion holds -> streak=1.
    # Two-consecutive still requires another pass.
    s3, st3 = evaluate_stop(
        _est(3, 0, p_miss=0.0, chao1_ci_95_high=0.0),
        cfg,
        previous_status=s2,
        previous_streak=st2,
    )
    assert s3 == "active"
    assert st3 == 1

    # Pass 4 (K=4): two consecutive empties -> saturated.
    s4, st4 = evaluate_stop(
        _est(4, 0, p_miss=0.0, chao1_ci_95_high=0.0),
        cfg,
        previous_status=s3,
        previous_streak=st3,
    )
    assert s4 == "saturated"
    assert st4 == 2


def test_stop_when_both_does_not_exhaust_clean_column_under_k_max() -> None:
    """Before this fix, a clean column would burn all k_max passes
    under stop_when=both because the Chao1 ratio at S_obs=0 was inf.
    Pin the new behaviour: it saturates well before k_max."""
    cfg = _cfg(
        stop_when="both",
        epsilon=0.05,
        epsilon_chao1=0.05,
        k_min=3,
        k_max=15,
        require_two_consecutive=True,
    )
    status = "active"
    streak = 0
    saturated_at: int | None = None
    for k in range(1, cfg.k_max + 1):
        status, streak = evaluate_stop(
            _est(k, 0, p_miss=0.0, chao1_ci_95_high=0.0),
            cfg,
            previous_status=status,
            previous_streak=streak,
        )
        if status == "saturated":
            saturated_at = k
            break
    assert saturated_at is not None
    assert saturated_at < cfg.k_max
    assert status == "saturated"


def test_stop_when_default_is_gt_for_back_compat() -> None:
    """Omitting `stop_when` from configs falls through to ``"gt"``,
    preserving pre-existing behaviour."""
    cfg = _cfg(require_two_consecutive=False)
    assert cfg.stop_when == "gt"
    # Hot Chao1, low GT -> saturated under default (gt).
    est = _est(4, 100, p_miss=0.01, chao1_ci_95_high=10_000.0)
    assert evaluate_stop(est, cfg)[0] == "saturated"


def test_stop_when_unknown_raises() -> None:
    cfg = _cfg(stop_when="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stop_when"):
        evaluate_stop(_est(4, 10, p_miss=0.01), cfg)


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


def test_is_applicable_no_whitelist_default_allows() -> None:
    arm = _arm("csv", ("a.csv",), "typo")
    assert is_applicable(arm) is True


def test_is_applicable_unknown_scope_kind_default_allows() -> None:
    # Whitelist mentions only "column"; "csv" is not mentioned ->
    # default-allow.
    arm = _arm("csv", ("a.csv",), "typo")
    assert (
        is_applicable(arm, whitelist={"column": ["typo"]}) is True
    )


def test_is_applicable_explicit_scope_blocks_unlisted_types() -> None:
    arm = _arm("csv", ("a.csv",), "typo")
    assert (
        is_applicable(
            arm, whitelist={"csv": ["exact_duplicate"]}
        )
        is False
    )


def test_is_applicable_empty_list_blocks_everything_in_scope() -> None:
    arm = _arm("csv", ("a.csv",), "typo")
    assert is_applicable(arm, whitelist={"csv": []}) is False


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_next_pass_targets_groups_active_arms_by_scope() -> None:
    arm_a = _arm("column", ("a.csv", "x"), "typo")
    arm_b = _arm("column", ("a.csv", "x"), "casing")
    arm_c = _arm("column", ("a.csv", "y"), "typo")
    arm_done = _arm("column", ("a.csv", "z"), "typo")
    estimates = {
        arm_a: _est(1, 1, 0.5),
        arm_b: _est(1, 1, 0.5),
        arm_c: _est(1, 1, 0.5),
        arm_done: _est(1, 1, 0.0),
    }
    estimates[arm_done].status = "saturated"
    targets = next_pass_targets(estimates)
    # "z" has no active arms -> not present.
    assert ("column", ("a.csv", "x")) in targets
    assert ("column", ("a.csv", "y")) in targets
    assert ("column", ("a.csv", "z")) not in targets
    assert targets[("column", ("a.csv", "x"))] == ["casing", "typo"]


def test_next_pass_targets_empty_when_all_done() -> None:
    arm = _arm("column", ("a.csv", "x"), "typo")
    e = _est(1, 1, 0.0)
    e.status = "saturated"
    assert next_pass_targets({arm: e}) == {}


# ---------------------------------------------------------------------------
# End-to-end smoke through run_estimator
# ---------------------------------------------------------------------------


def test_run_estimator_column_scope_end_to_end() -> None:
    # Two columns, very different stability.
    runs = []
    for _ in range(5):
        # Column "x": same 4 defects every run -> all saturated.
        x_run = [
            _det(row=r, columns=["x"], error_type="typo")
            for r in range(4)
        ]
        runs.append(x_run)
    # Plus column "y" finds 1 defect per run, none repeated.
    for i in range(5):
        runs[i].append(
            _det(row=100 + i, columns=["y"], error_type="typo")
        )

    out = run_estimator(
        runs,
        scope_kind="column",
        global_default=ArmConfig(
            epsilon=0.05, k_min=2, k_max=10,
            require_two_consecutive=False,
        ),
    )
    arm_x = _arm("column", ("a.csv", "x"), "typo")
    arm_y = _arm("column", ("a.csv", "y"), "typo")

    assert out[arm_x].status == "saturated"
    # All-singletons arm: high p_miss, so still active (or exhausted
    # if K_max exceeded).
    assert out[arm_y].status in {"active", "exhausted", "insufficient"}


def test_run_estimator_applicability_marks_not_applicable() -> None:
    runs = [[_det(row=1, columns=["x"], error_type="typo")]]
    out = run_estimator(
        runs,
        scope_kind="column",
        applicability={"column": ["casing"]},  # typo NOT applicable
    )
    arm = _arm("column", ("a.csv", "x"), "typo")
    assert out[arm].status == "not_applicable"
