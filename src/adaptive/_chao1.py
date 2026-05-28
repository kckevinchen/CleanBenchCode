"""Shared Chao1 variance + log-normal CI helpers.

Both :mod:`src.adaptive.cell_gt` (cell-level dashboard headlines) and
:mod:`src.adaptive.scoped_gt` (per-arm stop predicate) need the same
closed-form Chao 1987 / Chao-Yang 1993 math. This module hosts the
single canonical implementation; importing from here keeps the two
estimators numerically identical and avoids drift if a coefficient is
ever corrected.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, NamedTuple, Tuple

__all__ = [
    "chao1_variance_abundance",
    "chao1_variance_bias_corrected",
    "chao1_log_normal_ci",
    "frequency_of_frequencies",
    "ChaoCore",
    "compute_core",
    "is_saturated_by_estimate",
]


def chao1_variance_abundance(f1: float, f2: float) -> float:
    """Abundance-form Chao1 variance (Chao 1987, Eq. 5a).

    Valid when ``f2 > 0``. When ``f2 == 0`` the caller should fall back
    to the bias-corrected variance; we return ``nan`` here so hitting
    this branch with ``f2=0`` is loud.
    """
    if f2 <= 0:
        return float("nan")
    r = f1 / f2
    # Chao 1987 / Chao & Shen 2010, Theorem 1:
    #   Var(f_0_hat) = f2 * ( 0.5 r^2 + r^3 + 0.25 r^4 )
    # where r = f1/f2. The r^4 coefficient is 1/4 (NOT 1/2 - that's
    # a recurring transcription error in some open-source copies);
    # see SpadeR / EstimateS source for the canonical form.
    return f2 * (0.5 * r ** 2 + r ** 3 + 0.25 * r ** 4)


def chao1_variance_bias_corrected(f1: float, f2: float, s_obs: float) -> float:
    """Variance for the bias-corrected Chao1 (used when ``f2 == 0``).

    Bias-corrected estimator: ``f0_hat = f1 (f1 - 1) / (2 (f2 + 1))``.
    Chao (1987) gives the corresponding variance; the closed form
    below is the standard reproduction.

    ``s_obs`` is accepted for API symmetry with prior call-sites; it
    is not currently used (the variance depends only on f1, f2).
    """
    del s_obs  # unused; preserved for back-compat call shape
    denom = f2 + 1.0
    term1 = 0.25 * (f1 * (f1 - 1)) ** 2 / denom ** 3
    term2 = f1 * (f1 - 1) * (2 * f1 - 1) / (2.0 * denom ** 2)
    term3 = f1 * (f1 - 1) / (2.0 * denom)
    var = term1 + term2 + term3
    return max(var, 0.0)


def chao1_log_normal_ci(
    s_obs: int,
    missed: float,
    variance: float,
    z: float = 1.96,
) -> Tuple[float, float]:
    """Chao & Yang (1993) log-normal 95% CI on ``S_hat``.

    Used in preference to a naive Wald interval because the latter
    can return a lower bound below ``S_obs`` (impossible).
    """
    if missed <= 0 or variance <= 0:
        return (float(s_obs), float(s_obs + missed))
    c = math.exp(
        z * math.sqrt(math.log(1.0 + variance / (missed ** 2)))
    )
    low = s_obs + missed / c
    high = s_obs + missed * c
    return (low, high)


# ---------------------------------------------------------------------------
# Shared estimator core
# ---------------------------------------------------------------------------


def frequency_of_frequencies(
    frequencies: Iterable[int],
    *,
    num_runs: int,
) -> Dict[int, int]:
    """Bucket per-species emission counts into ``{r: n_r}``.

    The function is agnostic to which Good-Turing interpretation the
    caller is using:

    * **Interpretation A (Bernoulli per pass)** -- ``r`` is the number
      of passes (out of ``num_runs``) that observed the species. Used
      by :mod:`src.core.relations.adaptive` and
      :mod:`src.adaptive.inner_loop` where a "pass" is one LLM call
      and the question is "did the species appear at least once in
      this pass?". Under this interpretation ``r`` is bounded by
      ``num_runs`` and ``p_0 = n_1 / N`` estimates the next-pass
      missing mass.
    * **Interpretation B (raw emission count)** -- ``r`` is the total
      number of raw detection emissions for the species (no
      within-pass de-duplication). Used by
      :mod:`src.adaptive.scoped_gt` (``raw_hits``) where ``r`` may
      exceed ``num_runs`` and ``p_0`` estimates next-emission missing
      mass.

    The closed-form math (Chao 1987 / Chao-Yang 1993) is identical
    under both interpretations -- only the units of ``p_0`` differ,
    so keep callers consistent within a single dashboard tile.

    The returned dict zero-fills ``r in [1, num_runs]`` for shape
    stability on empty / sparse arms; observed counts above
    ``num_runs`` are added on top (only relevant under Interpretation
    B). Frequencies ``< 1`` are silently dropped (a species with zero
    observations is not a species).
    """
    counts: Dict[int, int] = {r: 0 for r in range(1, max(num_runs, 1) + 1)}
    for r in frequencies:
        if r < 1:
            continue
        counts[r] = counts.get(r, 0) + 1
    return counts


class ChaoCore(NamedTuple):
    """Closed-form Chao1 + Good-Turing bundle.

    All callers post-process this into their domain dataclass
    (``cell_gt.Estimate`` / ``scoped_gt.ArmEstimate``).
    """

    n_observed: int
    frequency_counts: Dict[int, int]
    total_hits: int
    good_turing_missing_mass: float
    chao1_estimated_missed: float
    chao1_estimated_total: float
    chao1_variance: float
    chao1_ci_95_low: float
    chao1_ci_95_high: float
    fn_rate_point: float
    fn_rate_ci_95_low: float
    fn_rate_ci_95_high: float


def compute_core(
    frequencies: Iterable[int],
    *,
    num_runs: int,
) -> ChaoCore:
    """Compute Good-Turing missing mass + Chao1 from raw frequencies.

    Each value in ``frequencies`` is the per-species emission count
    ``R_x`` (Wikipedia's notation): total raw detection emissions
    for that species across the entire experiment, with NO within-
    pass de-duplication. The textbook GT identity
    ``N = sum_r r * n_r = sum_x R_x`` then holds, and
    ``p_0 = n_1 / N`` recovers the classical missing-mass
    interpretation: probability that the next emission belongs to a
    species not yet observed.

    Edge cases:
    * ``S_obs == 0`` or ``num_runs <= 0`` -> all-zeros bundle (FN rate
      reported as 0 because we have no evidence either way).
    * ``n_2 == 0`` -> bias-corrected Chao1 ``n1 (n1-1) / 2`` plus the
      matching bias-corrected variance.
    * ``n_1 == 0`` -> missed = 0, FN rate 0.

    The returned ``frequency_counts`` always covers
    ``r in [1, num_runs]`` (zero-filled) plus any larger ``r`` values
    actually observed; this keeps downstream payload shapes stable
    for empty arms while still admitting raw counts above K.
    """
    freqs = list(frequencies)
    s_obs = len(freqs)
    if s_obs == 0 or num_runs <= 0:
        empty = {r: 0 for r in range(1, max(num_runs, 1) + 1)}
        return ChaoCore(
            n_observed=0,
            frequency_counts=empty,
            total_hits=0,
            good_turing_missing_mass=0.0,
            chao1_estimated_missed=0.0,
            chao1_estimated_total=0.0,
            chao1_variance=0.0,
            chao1_ci_95_low=0.0,
            chao1_ci_95_high=0.0,
            fn_rate_point=0.0,
            fn_rate_ci_95_low=0.0,
            fn_rate_ci_95_high=0.0,
        )

    freq = frequency_of_frequencies(freqs, num_runs=num_runs)
    n1 = freq.get(1, 0)
    n2 = freq.get(2, 0)
    n_total = sum(r * count for r, count in freq.items())
    gt_missing_mass = (n1 / n_total) if n_total > 0 else 0.0

    if n2 > 0:
        missed = (n1 * n1) / (2.0 * n2)
        var = chao1_variance_abundance(n1, n2)
    elif n1 > 0:
        missed = (n1 * (n1 - 1)) / 2.0
        var = chao1_variance_bias_corrected(n1, n2, s_obs)
    else:
        missed = 0.0
        var = 0.0
    if math.isnan(var):
        var = 0.0

    s_hat = s_obs + missed
    ci_low, ci_high = chao1_log_normal_ci(s_obs, missed, var)
    fn_point = (missed / s_hat) if s_hat > 0 else 0.0
    fn_low = ((ci_low - s_obs) / ci_low) if ci_low > 0 else 0.0
    fn_high = ((ci_high - s_obs) / ci_high) if ci_high > 0 else 0.0
    fn_low = max(fn_low, 0.0)
    fn_high = max(fn_high, fn_point)

    return ChaoCore(
        n_observed=s_obs,
        frequency_counts=freq,
        total_hits=n_total,
        good_turing_missing_mass=gt_missing_mass,
        chao1_estimated_missed=missed,
        chao1_estimated_total=s_hat,
        chao1_variance=var,
        chao1_ci_95_low=ci_low,
        chao1_ci_95_high=ci_high,
        fn_rate_point=fn_point,
        fn_rate_ci_95_low=fn_low,
        fn_rate_ci_95_high=fn_high,
    )


# ---------------------------------------------------------------------------
# Shared saturation predicate
# ---------------------------------------------------------------------------


def is_saturated_by_estimate(
    *,
    n_observed: int,
    good_turing_missing_mass: float,
    chao1_ci_95_high: float,
    epsilon: float,
    epsilon_chao1: float,
    stop_when: str,
) -> bool:
    """Shared saturation criterion for adaptive loops.

    Used by both the outer per-arm loop (``scoped_gt.evaluate_arm_stop``)
    and the inner per-LLM-call loops (``adaptive.inner_loop.run_adaptive``)
    so the FN-control semantics are identical at every layer.

    Criteria:
    * ``"gt"``    -- Good-Turing missing-mass point estimate
                     ``p_0 <= epsilon``.
    * ``"chao1"`` -- Chao1 upper-95 CI on missing count, normalised
                     by ``S_obs``, ``<=  epsilon_chao1``. When
                     ``S_obs == 0`` the Chao1 estimator is degenerate
                     (no f1/f2 to extrapolate from); the ratio is
                     defined as 0.0, so K consecutive empty passes
                     are read as "no species exist." The k_min /
                     two-consecutive guards on the caller prevent a
                     premature stop on a single empty draw.
    * ``"both"``  -- conjunction of the above.

    Returns ``True`` if the configured criterion is met, else
    ``False``. Callers combine this with ``k_min`` / ``k_max`` /
    two-consecutive bookkeeping.
    """
    gt_ok = good_turing_missing_mass <= epsilon
    if n_observed > 0:
        f0_upper = max(0.0, chao1_ci_95_high - n_observed)
        chao1_ratio = f0_upper / n_observed
    else:
        chao1_ratio = 0.0
    chao1_ok = chao1_ratio <= epsilon_chao1

    if stop_when == "gt":
        return gt_ok
    if stop_when == "chao1":
        return chao1_ok
    if stop_when == "both":
        return gt_ok and chao1_ok
    raise ValueError(
        f"is_saturated_by_estimate: unknown stop_when={stop_when!r}; "
        "expected one of 'gt', 'chao1', 'both'."
    )
