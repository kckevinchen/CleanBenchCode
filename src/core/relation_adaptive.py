"""Adaptive multi-pass relation discovery with per-arm Good-Turing stopping.

This is the relation-level analogue of :mod:`src.adaptive.scoped_gt`,
specialised for the relation-discovery loop:

* **Species**: a discovered :class:`src.core.relations.Relation`,
  identified by its :meth:`~src.core.relations.Relation.species_key`.
  Two relations with the same species key (same error_type, kind,
  column set, predicate fold) collapse to one species.

* **Arm**: a ``(source, error_type)`` pair. Each arm has its own
  Bernoulli-style sampling experiment - a relation is "discovered in
  pass *i*" iff that pass's LLM response for that ``(source,
  error_type)`` contained the species. We keep arms separate because
  the marginal cost of a pass is per-call, and one source's FD pool
  saturating tells us nothing about another source's calculation
  pool.

* **Stopping rule** (per arm): retire the arm once it has run at
  least ``min_passes`` passes AND its Good-Turing missing-mass
  estimate falls below ``gt_threshold``. Saturated arms are skipped
  in subsequent passes; the loop ends when every arm has either
  saturated or hit ``max_passes``.

* **Perturbation**: the only per-pass variation is the row sample
  seed. Each pass picks a different ``base_seed + pass_index``
  pseudo-random sample of rows from each table, so the LLM sees
  different evidence per pass. The schema digest itself is
  whole-table and so is unchanged.

The math is delegated to :func:`src.adaptive._chao1.compute_core` -
the same closed form that the cell-level estimator uses, so the
numbers reported here match the cell estimator's reported numbers
for any given frequency vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from src.adaptive._chao1 import compute_core as _compute_core

from .relations import DiscoveryReport, Relation, SchemaDiscoverer

logger = logging.getLogger(__name__)

ArmKey = Tuple[str, str]  # (source, error_type)

StopWhen = Literal["gt", "chao1", "both"]
"""Per-arm saturation criterion.

Mirrors :data:`src.adaptive.scoped_gt.StopWhen`:

* ``"gt"``    -- Good-Turing point estimate ``p_0 = n_1 / N`` is
  below ``gt_threshold``. Bounds the *probability mass* on
  undetected species.
* ``"chao1"`` -- upper-95 CI on the Chao1 missing count, normalised
  by ``S_obs``, is below ``epsilon_chao1``. Bounds the *count* of
  undetected species (using the conservative upper end of the CI
  because Chao1 is itself a lower bound on S).
* ``"both"``  -- both criteria simultaneously; strictly stronger.
"""


# ---------------------------------------------------------------------------
# Per-arm bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class ArmStats:
    """Per-arm species-frequency record across passes.

    ``frequencies`` maps a relation species_key to the number of
    passes (out of ``passes_run``) that surfaced it. ``passes_run``
    is the actual number of LLM calls executed for the arm; once
    ``saturated`` flips to True no further calls are issued and
    ``passes_run`` stops growing.
    """

    source: str
    error_type: str
    frequencies: Dict[str, int] = field(default_factory=dict)
    passes_run: int = 0
    saturated: bool = False
    saturated_at_pass: Optional[int] = None
    # Most recent Chao1/GT estimate for the arm.
    last_estimate: Optional[Dict[str, Any]] = None
    # Why the most recent saturation check returned its verdict.
    # One of:
    #   ``None``              -- no check has run yet (passes_run < min_passes)
    #   ``"below-min-passes"`` -- still warming up
    #   ``"empty"``           -- no relations across any pass; arm retired
    #   ``"gt"``              -- GT criterion satisfied (stop_when="gt")
    #   ``"chao1"``           -- Chao1 criterion satisfied (stop_when="chao1")
    #   ``"both"``            -- both criteria satisfied (stop_when="both")
    #   ``"gt-only"``         -- GT ok but Chao1 not (stop_when="both", continuing)
    #   ``"chao1-only"``      -- Chao1 ok but GT not (stop_when="both", continuing)
    #   ``"none"``            -- neither criterion satisfied yet
    last_reason: Optional[str] = None


@dataclass
class AdaptiveDiscoveryResult:
    """Output of :meth:`AdaptiveSchemaDiscoverer.run`."""

    pass_reports: List[List[DiscoveryReport]]
    arm_stats: Dict[ArmKey, ArmStats]
    union_relations: List[Relation]
    union_metadata: Dict[str, Dict[str, Any]]  # species_key -> {seen_in_passes, frequency, source, error_type}
    config: Dict[str, Any]

    @property
    def n_passes(self) -> int:
        return len(self.pass_reports)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class AdaptiveSchemaDiscoverer:
    """Run :class:`SchemaDiscoverer` for multiple passes with per-arm GT stopping."""

    def __init__(
        self,
        discoverer: SchemaDiscoverer,
        *,
        max_passes: int = 5,
        min_passes: int = 2,
        gt_threshold: float = 0.05,
        base_seed: int = 1000,
        stop_when: StopWhen = "both",
        epsilon_chao1: float = 0.05,
    ) -> None:
        if max_passes < 1:
            raise ValueError("max_passes must be >= 1")
        if min_passes < 1:
            raise ValueError("min_passes must be >= 1")
        if min_passes > max_passes:
            raise ValueError("min_passes must be <= max_passes")
        if not (0.0 <= gt_threshold <= 1.0):
            raise ValueError("gt_threshold must be in [0, 1]")
        if not (0.0 <= epsilon_chao1):
            raise ValueError("epsilon_chao1 must be >= 0")
        if stop_when not in ("gt", "chao1", "both"):
            raise ValueError(
                f"stop_when must be one of 'gt', 'chao1', 'both'; got {stop_when!r}"
            )
        self.discoverer = discoverer
        self.max_passes = int(max_passes)
        self.min_passes = int(min_passes)
        self.gt_threshold = float(gt_threshold)
        self.base_seed = int(base_seed)
        self.stop_when: StopWhen = stop_when
        self.epsilon_chao1 = float(epsilon_chao1)

    # ------------------------------------------------------------------

    def _seed_for_pass(self, pass_index: int) -> int:
        # pass_index is 0-based; offset by base_seed so two adaptive
        # runs with different base seeds don't share row samples.
        return self.base_seed + pass_index

    def _arm_estimate(self, arm: ArmStats) -> Dict[str, Any]:
        """Compute the Chao1/GT estimate dict for one arm."""
        core = _compute_core(arm.frequencies.values(), num_runs=arm.passes_run)
        return {
            "source": arm.source,
            "error_type": arm.error_type,
            "passes_run": arm.passes_run,
            "n_observed": core.n_observed,
            "frequency_counts": {str(k): v for k, v in core.frequency_counts.items()},
            "total_hits": core.total_hits,
            "good_turing_missing_mass": core.good_turing_missing_mass,
            "chao1": {
                "estimated_missed": core.chao1_estimated_missed,
                "estimated_total": core.chao1_estimated_total,
                "variance": core.chao1_variance,
                "ci_95": [core.chao1_ci_95_low, core.chao1_ci_95_high],
            },
            "estimated_fn_rate": {
                "point": core.fn_rate_point,
                "ci_95": [core.fn_rate_ci_95_low, core.fn_rate_ci_95_high],
            },
            "saturated": arm.saturated,
            "saturated_at_pass": arm.saturated_at_pass,
            "last_reason": arm.last_reason,
        }

    def _check_saturation(
        self,
        arm: ArmStats,
        *,
        pass_index: int,
    ) -> bool:
        """Decide whether this arm has saturated.

        Saturation requires:

        * at least ``min_passes`` passes have run, AND
        * the configured ``stop_when`` criterion holds:

          - ``gt``    -- ``p_0 = n_1 / N <= gt_threshold``. The
            point estimate is well-defined for any ``N > 0``; no
            ``n_2 > 0`` precondition (see Good 1953; McAllester &
            Schapire 2000 for concentration without that guard).
          - ``chao1`` -- ``(chao1_ci_95_high - S_obs) / S_obs <= epsilon_chao1``.
            The bias-corrected Chao1 variant in ``_chao1`` already
            handles ``n_2 == 0`` (it is unstable for very small
            ``n_2``, but defined).
          - ``both``  -- both of the above hold simultaneously.

        Special case: empty arm (no relations across all passes) is
        treated as saturated under any criterion -- nothing more to
        discover.
        """
        if arm.passes_run < self.min_passes:
            arm.last_reason = "below-min-passes"
            return False
        core = _compute_core(arm.frequencies.values(), num_runs=arm.passes_run)
        # Empty arm: model returned nothing in every pass. Treat as saturated.
        if core.n_observed == 0:
            arm.saturated = True
            arm.saturated_at_pass = pass_index
            arm.last_reason = "empty"
            return True

        # Use ``<=`` to match the docstring above and the shared
        # ``_chao1.is_saturated_by_estimate`` predicate. Using strict
        # ``<`` would let an arm hover at exactly ``gt_threshold``
        # without ever saturating.
        gt_ok = core.good_turing_missing_mass <= self.gt_threshold

        if core.n_observed > 0:
            f0_upper = max(0.0, core.chao1_ci_95_high - core.n_observed)
            chao1_ratio = f0_upper / core.n_observed
        else:
            chao1_ratio = float("inf")
        chao1_ok = chao1_ratio <= self.epsilon_chao1

        if self.stop_when == "gt":
            criterion_ok = gt_ok
            reason = "gt" if gt_ok else "none"
        elif self.stop_when == "chao1":
            criterion_ok = chao1_ok
            reason = "chao1" if chao1_ok else "none"
        else:  # "both"
            criterion_ok = gt_ok and chao1_ok
            if criterion_ok:
                reason = "both"
            elif gt_ok:
                reason = "gt-only"
            elif chao1_ok:
                reason = "chao1-only"
            else:
                reason = "none"

        arm.last_reason = reason
        if criterion_ok:
            arm.saturated = True
            arm.saturated_at_pass = pass_index
            return True
        return False

    # ------------------------------------------------------------------

    def run(self) -> AdaptiveDiscoveryResult:
        # Initialise one arm per (source, error_type).
        arms: Dict[ArmKey, ArmStats] = {}
        for src in self.discoverer.source_frames:
            if src not in self.discoverer.digests:
                continue
            for et in self.discoverer.error_types:
                arms[(src, et.key)] = ArmStats(source=src, error_type=et.key)

        if not arms:
            return AdaptiveDiscoveryResult(
                pass_reports=[],
                arm_stats={},
                union_relations=[],
                union_metadata={},
                config=self._config_dict(),
            )

        # species_key -> (representative Relation, count, set of pass indices)
        union_rel: Dict[str, Relation] = {}
        union_freq: Dict[str, int] = {}
        union_passes: Dict[str, Set[int]] = {}

        pass_reports: List[List[DiscoveryReport]] = []

        for pass_index in range(self.max_passes):
            # Build skip predicate from current saturated arms.
            saturated_keys = {k for k, a in arms.items() if a.saturated}

            def _skip(src: str, et: str) -> bool:
                return (src, et) in saturated_keys

            active_count = len(arms) - len(saturated_keys)
            if active_count == 0:
                logger.info(
                    "Adaptive discovery: all %d arm(s) saturated before pass %d",
                    len(arms), pass_index + 1,
                )
                break

            seed = self._seed_for_pass(pass_index)
            # Per-pass title is shown by the inner SchemaDiscoverer's
            # tqdm bar (which then becomes the only line on screen for
            # the duration of the pass). Demoted to debug to keep the
            # workbench job tail uncluttered.
            logger.debug(
                "Adaptive pass %d/%d: %d active arm(s), seed=%d",
                pass_index + 1, self.max_passes, active_count, seed,
            )
            reports = self.discoverer.run(
                seed=seed,
                skip=_skip,
                desc=f"Discover pass {pass_index + 1}/{self.max_passes}"
                     f" ({active_count} arm{'s' if active_count != 1 else ''})",
            )
            pass_reports.append(reports)

            # Per-arm tally for THIS pass.
            for report in reports:
                key = (report.source, report.error_type)
                arm = arms.get(key)
                if arm is None:
                    continue
                arm.passes_run += 1
                seen_this_pass: Set[str] = set()
                for rel in report.relations:
                    sk = rel.species_key()
                    # Each pass contributes at most ONE hit per
                    # species (Bernoulli per pass, not per relation
                    # listing). The single-pass _discover already
                    # dedups within a call, so this is mostly
                    # belt-and-braces.
                    if sk in seen_this_pass:
                        continue
                    seen_this_pass.add(sk)
                    arm.frequencies[sk] = arm.frequencies.get(sk, 0) + 1
                    union_rel.setdefault(sk, rel)
                    union_freq[sk] = union_freq.get(sk, 0) + 1
                    union_passes.setdefault(sk, set()).add(pass_index)

            # Update saturation.
            for key, arm in arms.items():
                if arm.saturated:
                    continue
                self._check_saturation(arm, pass_index=pass_index + 1)
                arm.last_estimate = self._arm_estimate(arm)

        # Final: ensure last_estimate is filled for every arm.
        for arm in arms.values():
            if arm.last_estimate is None:
                arm.last_estimate = self._arm_estimate(arm)

        union_relations = [union_rel[k] for k in union_rel]
        union_metadata: Dict[str, Dict[str, Any]] = {
            sk: {
                "frequency": union_freq[sk],
                "seen_in_passes": sorted(union_passes[sk]),
                "source": union_rel[sk].source,
                "error_type": union_rel[sk].error_type,
            }
            for sk in union_rel
        }

        return AdaptiveDiscoveryResult(
            pass_reports=pass_reports,
            arm_stats=arms,
            union_relations=union_relations,
            union_metadata=union_metadata,
            config=self._config_dict(),
        )

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "max_passes": self.max_passes,
            "min_passes": self.min_passes,
            "gt_threshold": self.gt_threshold,
            "base_seed": self.base_seed,
            "stop_when": self.stop_when,
            "epsilon_chao1": self.epsilon_chao1,
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def adaptive_summary_dict(result: AdaptiveDiscoveryResult) -> Dict[str, Any]:
    """Build a JSON-serialisable summary of an adaptive run."""
    arms = result.arm_stats
    n_arms = len(arms)
    n_saturated = sum(1 for a in arms.values() if a.saturated)
    saturated_at: Dict[int, int] = {}
    for a in arms.values():
        if a.saturated and a.saturated_at_pass is not None:
            saturated_at[a.saturated_at_pass] = saturated_at.get(a.saturated_at_pass, 0) + 1

    # Tally why each arm finished where it did - useful for spotting
    # arms that hit max_passes for "gt-only"/"chao1-only" reasons
    # (i.e. one criterion was happy but the other wasn't).
    reason_counts: Dict[str, int] = {}
    for a in arms.values():
        key = a.last_reason or "unchecked"
        reason_counts[key] = reason_counts.get(key, 0) + 1

    # Per-error-type aggregates.
    by_error_type: Dict[str, Dict[str, Any]] = {}
    for (src, et), arm in arms.items():
        bucket = by_error_type.setdefault(et, {
            "n_arms": 0,
            "n_saturated": 0,
            "total_passes_run": 0,
            "total_relations": 0,
        })
        bucket["n_arms"] += 1
        bucket["n_saturated"] += int(arm.saturated)
        bucket["total_passes_run"] += arm.passes_run
        bucket["total_relations"] += len(arm.frequencies)

    return {
        "config": result.config,
        "n_passes": result.n_passes,
        "n_arms": n_arms,
        "n_saturated_arms": n_saturated,
        "saturated_arms_by_pass": {str(k): v for k, v in sorted(saturated_at.items())},
        "last_reason_counts": reason_counts,
        "n_union_relations": len(result.union_relations),
        "by_error_type": by_error_type,
        "arms": [a.last_estimate for a in arms.values() if a.last_estimate is not None],
    }
