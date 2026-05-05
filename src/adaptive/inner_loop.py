"""Adaptive inner-loop driver for per-LLM-call recall boosting.

The outer adaptive layer (``src.adaptive.scoped_gt``) re-runs the
*entire detection pipeline* K times with perturbations and uses
Good-Turing / Chao1 to estimate the false-negative mass. This module
provides the same primitive at a much smaller granularity: drive a
single LLM call ("arm") up to ``k_max`` repeats with caller-supplied
perturbations and stop early once the missing-mass estimator says we
have converged.

Both layers consume the **same** :class:`~src.adaptive.scoped_gt.ArmConfig`
type and the **same** saturation predicate
(:func:`src.adaptive._chao1.is_saturated_by_estimate`), so the
operator has one place to express FN-control intent and every LLM
call -- chunk scan, row-dup adjudication, value canonicalisation --
inherits it.

Used by the row-duplication detector for two stages:

* **Cluster adjudication.** Each candidate cluster is sampled K
  times with shuffled row order. Species = ``(verdict, frozenset(affected_rows))``.
  The recall-first union of species across runs becomes the final
  per-cluster issue set.
* **Per-column canonicalisation.** Each column's distinct values are
  bucketed K times with shuffled value order. Species = the canonical
  partition seen in one run. Recall-first merging of partitions via
  union-find produces the final equivalence classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Hashable, Iterable, List

from ._chao1 import compute_core, is_saturated_by_estimate
from .scoped_gt import ArmConfig

logger = logging.getLogger(__name__)


@dataclass
class InnerLoopResult:
    """Outcome of an adaptive inner loop.

    ``species_per_run`` preserves the per-run species sets in order so
    callers that care about *which* run first surfaced a species can
    reconstruct that. ``species_to_runs`` is the inverted index used
    by callers to compute "k of K runs agreed" agreement scores.
    """

    runs_completed: int
    species_per_run: List[FrozenSet[Hashable]] = field(default_factory=list)
    species_to_runs: Dict[Hashable, List[int]] = field(default_factory=dict)
    gt_missing_mass: float = 0.0
    chao1_missed: float = 0.0
    stopped_reason: str = "k_max"

    def union_species(self) -> FrozenSet[Hashable]:
        """All species seen across all runs (recall-first union)."""
        out: set = set()
        for s in self.species_per_run:
            out.update(s)
        return frozenset(out)


def run_adaptive(
    *,
    arm_id: str,
    sample_fn: Callable[[int], Iterable[Hashable]],
    cfg: ArmConfig,
) -> InnerLoopResult:
    """Drive ``sample_fn`` up to ``cfg.k_max`` times with the configured stop rule.

    ``sample_fn(run_idx)`` should perform one perturbed sample (e.g.
    one LLM call with shuffled inputs) and return the iterable of
    species observed on that run. Species must be hashable; duplicates
    within one call are collapsed (one species can be observed at most
    once per run, by definition).

    The stop predicate is shared with the outer adaptive layer:

    * Run at least ``cfg.k_min`` times.
    * Stop when the configured criterion (``cfg.stop_when``) holds:
        - ``"gt"``    : ``p_0 <= cfg.epsilon``
        - ``"chao1"`` : Chao1 upper-95 missing fraction ``<= cfg.epsilon_chao1``
        - ``"both"``  : conjunction
    * If ``cfg.require_two_consecutive`` is True, additionally require
      two consecutive runs with no new species.
    * Hard cap: stop after ``cfg.k_max`` runs.

    The function never raises on estimator edge cases (empty species
    sets, ``S_obs < 2`` ...) -- it simply returns once the loop
    completes.
    """
    if (
        cfg.k_max is None
        or cfg.k_min is None
        or cfg.epsilon is None
        or cfg.epsilon_chao1 is None
        or cfg.stop_when is None
    ):
        raise ValueError(
            "run_adaptive: ArmConfig is missing required fields after "
            "resolution. Pass cfg = your_cfg.merge_into(ArmConfig.default())."
        )

    species_per_run: List[FrozenSet[Hashable]] = []
    species_to_runs: Dict[Hashable, List[int]] = {}
    consecutive_no_new = 0
    stopped_reason = "k_max"

    k_max = max(1, int(cfg.k_max))
    k_min_raw = max(1, int(cfg.k_min))
    if k_min_raw > k_max:
        # Misconfigured arm: warn once so the operator can fix it.
        # Silent clamping has bitten us before -- a stale yaml with
        # k_min=10, k_max=5 ends up running exactly k_max passes
        # because k_min collapses to 5, with no log line to indicate
        # that the intended floor was raised above the ceiling.
        logger.warning(
            "inner_loop[%s]: k_min (%d) > k_max (%d); clamping k_min to k_max. "
            "Check your ArmConfig / good_turing.per_arm settings.",
            arm_id, k_min_raw, k_max,
        )
    k_min = min(k_min_raw, k_max)
    two_consec = bool(cfg.require_two_consecutive)

    for run_idx in range(k_max):
        observed = frozenset(sample_fn(run_idx))
        species_per_run.append(observed)

        new_seen = 0
        for sp in observed:
            if sp not in species_to_runs:
                species_to_runs[sp] = []
                new_seen += 1
            species_to_runs[sp].append(run_idx)

        if new_seen == 0:
            consecutive_no_new += 1
        else:
            consecutive_no_new = 0

        completed = run_idx + 1
        if completed < k_min:
            continue

        # Compute Chao1 / GT on per-species emission counts.
        frequencies = [len(runs) for runs in species_to_runs.values()]
        core = compute_core(frequencies, num_runs=completed)

        criterion_ok = is_saturated_by_estimate(
            n_observed=core.n_observed,
            good_turing_missing_mass=core.good_turing_missing_mass,
            chao1_ci_95_high=core.chao1_ci_95_high,
            epsilon=cfg.epsilon,
            epsilon_chao1=cfg.epsilon_chao1,
            stop_when=cfg.stop_when,
        )
        two_consec_ok = (not two_consec) or consecutive_no_new >= 2

        if criterion_ok and two_consec_ok:
            stopped_reason = (
                "two_consecutive"
                if (two_consec and consecutive_no_new >= 2)
                else cfg.stop_when
            )
            logger.debug(
                "inner_loop[%s]: stop after run %d/%d "
                "(gt=%.4f chao1_missed=%.4f s_obs=%d reason=%s)",
                arm_id, completed, k_max,
                core.good_turing_missing_mass, core.chao1_estimated_missed,
                core.n_observed, stopped_reason,
            )
            return InnerLoopResult(
                runs_completed=completed,
                species_per_run=species_per_run,
                species_to_runs=species_to_runs,
                gt_missing_mass=core.good_turing_missing_mass,
                chao1_missed=core.chao1_estimated_missed,
                stopped_reason=stopped_reason,
            )

    # Hit k_max without converging.
    frequencies = [len(runs) for runs in species_to_runs.values()]
    core = compute_core(frequencies, num_runs=k_max)
    logger.debug(
        "inner_loop[%s]: ran to k_max=%d (gt=%.4f chao1_missed=%.4f s_obs=%d)",
        arm_id, k_max, core.good_turing_missing_mass,
        core.chao1_estimated_missed, core.n_observed,
    )
    return InnerLoopResult(
        runs_completed=k_max,
        species_per_run=species_per_run,
        species_to_runs=species_to_runs,
        gt_missing_mass=core.good_turing_missing_mass,
        chao1_missed=core.chao1_estimated_missed,
        stopped_reason=stopped_reason,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def merge_partitions(
    partitions: Iterable[Iterable[FrozenSet[str]]],
) -> List[FrozenSet[str]]:
    """Union-find merge of multiple partitions of the same value set.

    Each input partition is an iterable of value-sets (frozensets).
    Two values that appear together in *any* partition end up in the
    same merged class - i.e. the merged partition is the transitive
    closure of "co-occurred in some run". Recall-first by design: a
    single run merging two values is enough.

    Used by the canonicalisation inner loop to combine K per-call
    bucketings into one final equivalence relation.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for partition in partitions:
        for cluster in partition:
            members = [m for m in cluster if isinstance(m, str)]
            if len(members) < 2:
                # Still register singletons so they appear in output.
                for m in members:
                    find(m)
                continue
            anchor = members[0]
            for m in members[1:]:
                union(anchor, m)

    classes: Dict[str, List[str]] = {}
    for v in list(parent):
        root = find(v)
        classes.setdefault(root, []).append(v)
    return [frozenset(c) for c in classes.values()]
