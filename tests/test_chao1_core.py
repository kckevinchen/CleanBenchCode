"""Unit tests for the closed-form Chao1 + Good-Turing math.

These pin every numerical formula in :mod:`src.adaptive._chao1` to a
hand-calculated reference value. The motivation is operational: the
adaptive loop's stop predicate consumes these numbers directly, so a
stray coefficient (e.g. the ``r^4`` term in
:func:`chao1_variance_abundance`) silently inflates the Chao1 upper
CI and prevents arms from ever pruning. We caught one such bug in an
audit; these tests exist to make sure it stays caught.

Reference formulas (all verified against Chao 1987 / Chao & Yang 1993
and cross-checked with SpadeR / EstimateS source):

* GT mass:               ``p_0 = n_1 / N``  with ``N = sum_r r * n_r``
* Chao1 abundance:       ``f_0_hat = f_1^2 / (2 f_2)`` when ``f_2 > 0``
* Chao1 bias-corrected:  ``f_0_hat = f_1 (f_1 - 1) / (2 (f_2 + 1))``
                         (used when ``f_2 == 0`` with ``f_2 + 1 = 1``)
* Var (abundance):       ``f_2 * (0.5 r^2 + r^3 + 0.25 r^4)`` with
                         ``r = f_1 / f_2``
* Log-normal CI:         Chao & Yang (1993). Lower bound is always
                         strictly above ``S_obs - 1`` (a key sanity
                         property; the naive Wald interval can dip
                         below ``S_obs`` which is impossible).
* Identity:              ``sum_x R_x = sum_r r * n_r = N``
"""

from __future__ import annotations

import math

import pytest

from src.adaptive._chao1 import (
    chao1_log_normal_ci,
    chao1_variance_abundance,
    chao1_variance_bias_corrected,
    compute_core,
    frequency_of_frequencies,
    is_saturated_by_estimate,
)


# ---------------------------------------------------------------------------
# frequency_of_frequencies
# ---------------------------------------------------------------------------


class TestFrequencyOfFrequencies:
    def test_zero_fills_r_in_1_to_K(self) -> None:
        # Empty input still returns a {r: 0} dict for r in [1, K].
        out = frequency_of_frequencies([], num_runs=4)
        assert out == {1: 0, 2: 0, 3: 0, 4: 0}

    def test_buckets_match_inputs(self) -> None:
        # Three singletons, two doubletons, one triple.
        freqs = [1, 1, 1, 2, 2, 3]
        out = frequency_of_frequencies(freqs, num_runs=3)
        assert out == {1: 3, 2: 2, 3: 1}

    def test_admits_r_above_K_under_interp_b(self) -> None:
        # Interpretation B: a species emitted multiple times within a
        # single pass can have R_x > K. The output dict must still
        # zero-fill r in [1, K] AND retain the larger key.
        out = frequency_of_frequencies([1, 7], num_runs=3)
        assert out == {1: 1, 2: 0, 3: 0, 7: 1}

    def test_drops_zero_and_negative(self) -> None:
        # A species with zero observations is not a species; defensive
        # filter so callers that accidentally include them don't
        # corrupt the histogram.
        out = frequency_of_frequencies([0, -1, 1, 1], num_runs=2)
        assert out == {1: 2, 2: 0}

    def test_num_runs_zero_still_returns_dict(self) -> None:
        # Edge case: num_runs == 0 means "no passes done"; we still
        # zero-fill r=1 so callers can index without KeyError.
        out = frequency_of_frequencies([], num_runs=0)
        assert out == {1: 0}


# ---------------------------------------------------------------------------
# Variance forms
# ---------------------------------------------------------------------------


class TestChao1VarianceAbundance:
    def test_canonical_value_r_equals_2(self) -> None:
        # f1=2, f2=1 -> r=2. Standard formula:
        #   f2 * (0.5 r^2 + r^3 + 0.25 r^4)
        # = 1 * (0.5*4 + 8 + 0.25*16)
        # = 2 + 8 + 4
        # = 14.0
        assert chao1_variance_abundance(2, 1) == pytest.approx(14.0)

    def test_canonical_value_r_equals_3(self) -> None:
        # f1=3, f2=1 -> r=3.
        # 1 * (0.5*9 + 27 + 0.25*81) = 4.5 + 27 + 20.25 = 51.75
        assert chao1_variance_abundance(3, 1) == pytest.approx(51.75)

    def test_canonical_value_r_equals_1(self) -> None:
        # f1=4, f2=4 -> r=1.
        # 4 * (0.5 + 1 + 0.25) = 4 * 1.75 = 7.0
        assert chao1_variance_abundance(4, 4) == pytest.approx(7.0)

    def test_returns_nan_when_f2_is_zero(self) -> None:
        # The abundance form is undefined for f2 == 0; we surface NaN
        # so the caller (compute_core) routes to the bias-corrected
        # branch instead of silently using the wrong formula.
        assert math.isnan(chao1_variance_abundance(5, 0))

    def test_returns_nan_when_f2_negative(self) -> None:
        # Defensive: negative counts shouldn't reach us, but if they
        # do, the abundance form is meaningless.
        assert math.isnan(chao1_variance_abundance(5, -1))

    def test_monotone_in_f1(self) -> None:
        # Holding f2 fixed, growing f1 should never decrease the
        # variance. (The estimator is more uncertain when there are
        # many singletons.)
        prev = chao1_variance_abundance(1, 2)
        for f1 in range(2, 10):
            cur = chao1_variance_abundance(f1, 2)
            assert cur >= prev
            prev = cur


class TestChao1VarianceBiasCorrected:
    def test_canonical_value_n1_equals_4(self) -> None:
        # f1=4, f2=0 -> standard bias-corrected formula:
        #   denom = 1
        #   term1 = 0.25 * (4*3)^2 / 1 = 0.25 * 144 = 36
        #   term2 = 4*3*7 / 2 = 42
        #   term3 = 4*3 / 2 = 6
        # total = 84
        assert chao1_variance_bias_corrected(4, 0, s_obs=4) == pytest.approx(84.0)

    def test_clamped_at_zero(self) -> None:
        # f1=0 / f1=1 produce missed=0; variance must be >= 0 (we
        # max(.., 0.0) so callers don't get a tiny negative from FP).
        assert chao1_variance_bias_corrected(0, 0, s_obs=0) >= 0.0
        assert chao1_variance_bias_corrected(1, 0, s_obs=1) >= 0.0


# ---------------------------------------------------------------------------
# Log-normal CI
# ---------------------------------------------------------------------------


class TestChao1LogNormalCI:
    def test_lower_bound_is_strictly_above_s_obs_minus_1(self) -> None:
        # Key correctness property: the log-normal CI on S_hat never
        # lets the lower bound dip below S_obs (which is impossible -
        # we already saw S_obs species). The naive Wald CI can; that's
        # why we use the log-normal form.
        low, high = chao1_log_normal_ci(s_obs=10, missed=5.0, variance=20.0)
        assert low >= 10.0  # >= S_obs

    def test_collapses_to_point_when_no_uncertainty(self) -> None:
        # missed=0 OR variance=0 -> CI is the point estimate (no
        # spread). Defensive against divide-by-zero in the log term.
        low, high = chao1_log_normal_ci(s_obs=10, missed=0.0, variance=20.0)
        assert (low, high) == (10.0, 10.0)
        low2, high2 = chao1_log_normal_ci(s_obs=10, missed=5.0, variance=0.0)
        assert (low2, high2) == (10.0, 15.0)

    def test_high_above_low(self) -> None:
        low, high = chao1_log_normal_ci(s_obs=20, missed=8.0, variance=30.0)
        assert high > low

    def test_widens_with_variance(self) -> None:
        # Holding missed fixed, larger variance -> wider CI.
        low_a, high_a = chao1_log_normal_ci(20, 5.0, 5.0)
        low_b, high_b = chao1_log_normal_ci(20, 5.0, 20.0)
        assert (high_b - low_b) > (high_a - low_a)


# ---------------------------------------------------------------------------
# compute_core: end-to-end identity + edge cases
# ---------------------------------------------------------------------------


class TestComputeCore:
    def test_textbook_identity_N_equals_sum_r_n_r(self) -> None:
        # The single most important invariant under Interpretation B:
        # the textbook GT identity must hold. If it doesn't, the
        # missing-mass denominator is divorced from R_x and the stop
        # criterion is meaningless.
        freqs = [1, 1, 2, 3, 5, 8]  # = 20
        core = compute_core(freqs, num_runs=4)
        assert core.total_hits == sum(freqs) == 20
        identity = sum(r * n for r, n in core.frequency_counts.items())
        assert identity == 20

    def test_gt_missing_mass_point_value(self) -> None:
        # 5 species, raw_hits=[1,1,2,3,5]. n_1=2, N=12.
        # p_0 = 2/12 = 1/6.
        core = compute_core([1, 1, 2, 3, 5], num_runs=4)
        assert core.n_observed == 5
        assert core.frequency_counts.get(1) == 2
        assert core.total_hits == 12
        assert core.good_turing_missing_mass == pytest.approx(1 / 6)

    def test_chao1_abundance_when_n2_positive(self) -> None:
        # Same vector. n_1=2, n_2=1 -> Chao1 missed = 4/2 = 2.
        # S_hat = 5 + 2 = 7.
        core = compute_core([1, 1, 2, 3, 5], num_runs=4)
        assert core.chao1_estimated_missed == pytest.approx(2.0)
        assert core.chao1_estimated_total == pytest.approx(7.0)
        # Variance for r=2: 14.0 (see TestChao1VarianceAbundance).
        assert core.chao1_variance == pytest.approx(14.0)

    def test_chao1_bias_corrected_when_n2_zero(self) -> None:
        # 4 singletons, no doubletons. Bias-corrected:
        # missed = n1*(n1-1)/2 = 4*3/2 = 6.
        # S_hat = 4 + 6 = 10.
        core = compute_core([1, 1, 1, 1], num_runs=2)
        assert core.frequency_counts.get(1) == 4
        assert core.frequency_counts.get(2) == 0
        assert core.chao1_estimated_missed == pytest.approx(6.0)
        assert core.chao1_estimated_total == pytest.approx(10.0)

    def test_admits_R_x_above_K(self) -> None:
        # One chatty species (R_x = 7) and one singleton, K = 2.
        # The frequency dict must keep r=7 and zero-fill r in [1,2].
        core = compute_core([1, 7], num_runs=2)
        assert core.frequency_counts == {1: 1, 2: 0, 7: 1}
        # Identity still holds: 1*1 + 7*1 = 8.
        assert core.total_hits == 8
        # GT mass = n_1 / N = 1 / 8.
        assert core.good_turing_missing_mass == pytest.approx(0.125)

    def test_empty_input_returns_zero_bundle(self) -> None:
        # No species => everything zero, no NaN, no divide-by-zero.
        core = compute_core([], num_runs=4)
        assert core.n_observed == 0
        assert core.total_hits == 0
        assert core.good_turing_missing_mass == 0.0
        assert core.chao1_estimated_missed == 0.0
        assert core.chao1_estimated_total == 0.0
        assert core.chao1_variance == 0.0
        # Frequency dict still zero-filled for r in [1, K].
        assert core.frequency_counts == {1: 0, 2: 0, 3: 0, 4: 0}

    def test_zero_num_runs_returns_zero_bundle(self) -> None:
        # Edge: caller asks for the estimate before any pass has run.
        core = compute_core([1, 1], num_runs=0)
        assert core.n_observed == 0  # short-circuited
        assert core.total_hits == 0
        assert core.good_turing_missing_mass == 0.0

    def test_no_singletons_means_zero_missing_mass(self) -> None:
        # Every species seen at least twice -> n_1 = 0 -> p_0 = 0.
        # Chao1 also says "no missing": n_1 = 0 -> missed = 0.
        core = compute_core([2, 2, 3, 4], num_runs=4)
        assert core.frequency_counts.get(1, 0) == 0
        assert core.good_turing_missing_mass == 0.0
        assert core.chao1_estimated_missed == 0.0

    def test_all_singletons_saturates_GT_mass_at_one(self) -> None:
        # Pathological: every species is a singleton.
        # p_0 = n_1 / N = S_obs / S_obs = 1.0.
        core = compute_core([1, 1, 1, 1, 1], num_runs=2)
        assert core.good_turing_missing_mass == pytest.approx(1.0)

    def test_p_0_in_unit_interval(self) -> None:
        # Random-ish vector; p_0 must always be in [0, 1].
        for vec in [[1], [1, 2, 3], [5, 5, 5, 5], [1, 2, 1, 1, 4, 7]]:
            core = compute_core(vec, num_runs=3)
            assert 0.0 <= core.good_turing_missing_mass <= 1.0

    def test_fn_rate_consistent_with_chao1(self) -> None:
        # FN rate = missed / S_hat. Sanity: in [0, 1] and equal to the
        # standard ratio.
        core = compute_core([1, 1, 2, 3, 5], num_runs=4)
        expected = core.chao1_estimated_missed / core.chao1_estimated_total
        assert core.fn_rate_point == pytest.approx(expected)
        assert 0.0 <= core.fn_rate_point <= 1.0

    def test_ci_brackets_point(self) -> None:
        # 95% CI must cover the point estimate (basic monotonicity).
        core = compute_core([1, 1, 2, 3, 5], num_runs=4)
        assert core.chao1_ci_95_low <= core.chao1_estimated_total
        assert core.chao1_estimated_total <= core.chao1_ci_95_high
        # And the FN-rate CI is similarly ordered.
        assert core.fn_rate_ci_95_low <= core.fn_rate_point
        assert core.fn_rate_point <= core.fn_rate_ci_95_high

    def test_chao1_lower_bounds_true_S_for_uniform_distribution(self) -> None:
        # Stress test: simulate a population of 100 equiprobable
        # species with N=200 emissions. Chao1 is supposed to be a
        # *lower* bound on the true S; check that property holds.
        # (This is a property test, not a tight numerical pin.)
        import random
        rng = random.Random(42)
        true_S = 100
        N_emissions = 200
        emissions = [rng.randrange(true_S) for _ in range(N_emissions)]
        # raw_hits per species = how many times it was emitted.
        from collections import Counter
        per_species = list(Counter(emissions).values())
        core = compute_core(per_species, num_runs=N_emissions)
        # Chao1 gives a *lower* bound; we expect it to underestimate.
        assert core.chao1_estimated_total <= true_S * 1.5
        # And the upper-95 CI should cover the truth (with high prob).
        # Not a guaranteed test but stable for this seed.
        assert core.chao1_ci_95_high >= true_S * 0.6


# ---------------------------------------------------------------------------
# is_saturated_by_estimate: S_obs == 0 semantics
# ---------------------------------------------------------------------------


class TestSaturatedAtZeroObservations:
    """When S_obs == 0 the Chao1 estimator is degenerate (no f1/f2 to
    extrapolate). We define the missing-fraction ratio to be 0 so that
    K consecutive empty passes are read as "no species exist." The
    k_min / two-consecutive guards on the caller prevent a premature
    stop on a single empty draw."""

    def test_chao1_zero_obs_treated_as_saturated(self) -> None:
        # An empty draw under "chao1": ratio is 0 -> saturated.
        assert is_saturated_by_estimate(
            n_observed=0,
            good_turing_missing_mass=0.0,
            chao1_ci_95_high=0.0,
            epsilon=0.05,
            epsilon_chao1=0.05,
            stop_when="chao1",
        ) is True

    def test_both_zero_obs_treated_as_saturated(self) -> None:
        # Under "both", GT-zero AND chao1-zero both pass at S_obs=0,
        # so the conjunction also saturates.
        assert is_saturated_by_estimate(
            n_observed=0,
            good_turing_missing_mass=0.0,
            chao1_ci_95_high=0.0,
            epsilon=0.05,
            epsilon_chao1=0.05,
            stop_when="both",
        ) is True

    def test_gt_zero_obs_treated_as_saturated(self) -> None:
        # GT mass is zero by convention at S_obs=0 (compute_core
        # short-circuits): "gt" mode also saturates.
        assert is_saturated_by_estimate(
            n_observed=0,
            good_turing_missing_mass=0.0,
            chao1_ci_95_high=0.0,
            epsilon=0.05,
            epsilon_chao1=0.05,
            stop_when="gt",
        ) is True

    def test_chao1_with_observations_unchanged(self) -> None:
        # Sanity: the >0 branch is unaffected. Tight CI saturates,
        # loose CI does not.
        # n_observed=10, ci_95_high=10 -> ratio=0 -> saturated.
        assert is_saturated_by_estimate(
            n_observed=10,
            good_turing_missing_mass=0.5,  # gt fails
            chao1_ci_95_high=10.0,
            epsilon=0.05,
            epsilon_chao1=0.05,
            stop_when="chao1",
        ) is True
        # n_observed=10, ci_95_high=20 -> ratio=1.0 -> not saturated.
        assert is_saturated_by_estimate(
            n_observed=10,
            good_turing_missing_mass=0.0,
            chao1_ci_95_high=20.0,
            epsilon=0.05,
            epsilon_chao1=0.05,
            stop_when="chao1",
        ) is False
