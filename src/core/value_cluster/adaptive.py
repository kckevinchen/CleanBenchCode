"""Multi-sample adaptive driver for column standardization clustering.

Wraps :class:`src.core.value_cluster.proposer.ValueClusterProposer` in
the shared adaptive inner loop (``src.adaptive.inner_loop.run_adaptive``)
so the LLM is queried multiple times with shuffled value ordering and
the stop rule is driven by Good-Turing / Chao1 saturation. Recall-first
union of proposed clusters across runs becomes the draft offered to the
human annotator in the column-board UI.

Output schema (one record per (source, column), JSON-serialisable):

    {
      "source": "<csv basename>",
      "column": "<column header>",
      "runs_completed":      <int>,
      "stopped_reason":      "gt" | "chao1" | "both" | "two_consecutive" | "k_max",
      "gt_missing_mass":     <float>,
      "chao1_missed":        <float>,
      "draft_clusters": [
        {
          "values":        ["<v1>", "<v2>", ...],   # sorted
          "kind":          "<ClusterKind value>",   # most-common across runs
          "canonical":     "<v>" | null,            # most-common across runs
          "target_style":  "<short phrase>" | null, # most-common across runs;
                                                    # describes the column's
                                                    # competing-majority style
                                                    # for type-2 groups (null
                                                    # for type-1 aliases)
          "support_runs":  [<run_idx>, ...],        # 0-based
          "agreement":     <float>                  # |runs| / runs_completed
        },
        ...
      ]
    }

The aggregator IS NOT a union-find merger: each proposed cluster is
kept as-is (a recall-first list of candidate standardization groups).
Resolving overlaps between runs (e.g. run 1 proposes {A,B}, run 2
proposes {A,C}) is the human annotator's job in the labelling UI.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from src.adaptive._chao1 import compute_core, is_saturated_by_estimate
from src.adaptive.inner_loop import InnerLoopResult, run_adaptive
from src.adaptive.scoped_gt import ArmConfig
from src.core.value_cluster.proposer import (
    ProposedCluster,
    ValueClusterProposer,
)
from src.labeling.schema import ClusterKind

logger = logging.getLogger(__name__)


# Stable, reproducible RNG per (source, column) so a re-run on the
# same input produces the same shuffle sequence (same seed every
# call; the run index is folded into the shuffle independently).
def _stable_rng(*parts: str) -> random.Random:
    seed = 0
    for p in parts:
        for ch in p:
            seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    return random.Random(seed)


@dataclass
class DraftCluster:
    """Aggregated cluster across all adaptive-loop runs."""

    values: Tuple[str, ...]
    kind: ClusterKind = ClusterKind.UNTAGGED
    canonical: Optional[str] = None
    target_style: Optional[str] = None
    support_runs: List[int] = field(default_factory=list)
    agreement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": list(self.values),
            "kind": self.kind.value,
            "canonical": self.canonical,
            "target_style": self.target_style,
            "support_runs": list(self.support_runs),
            "agreement": self.agreement,
        }


@dataclass
class ColumnClusteringResult:
    """Outcome of the adaptive loop on one (source, column)."""

    source: str
    column: str
    runs_completed: int
    stopped_reason: str
    gt_missing_mass: float
    chao1_missed: float
    draft_clusters: List[DraftCluster]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "column": self.column,
            "runs_completed": self.runs_completed,
            "stopped_reason": self.stopped_reason,
            "gt_missing_mass": self.gt_missing_mass,
            "chao1_missed": self.chao1_missed,
            "draft_clusters": [c.to_dict() for c in self.draft_clusters],
        }


@dataclass
class ColumnClusteringAdaptive:
    """Adaptive-loop driver for one column.

    The species hashed by the inner loop is the sorted-values tuple of
    each :class:`ProposedCluster`. Across runs we keep the per-run
    proposals so we can reconstruct each species' kind/canonical
    consensus.
    """

    proposer: ValueClusterProposer
    arm_config: ArmConfig

    def run(
        self,
        *,
        source: str,
        column: str,
        value_counts: Sequence[Tuple[str, int]],
    ) -> ColumnClusteringResult:
        """Drive the adaptive loop for one (source, column).

        ``value_counts`` is ``[(value, count), ...]``. Empty / blank
        values should be removed by the caller. When fewer than two
        distinct values are present the loop is skipped and an empty
        result is returned (saves an LLM call when the column is
        trivially clean).
        """
        distinct = [(v, c) for v, c in value_counts if v != ""]
        if len({v for v, _ in distinct}) < 2:
            return ColumnClusteringResult(
                source=source, column=column,
                runs_completed=0, stopped_reason="trivial",
                gt_missing_mass=0.0, chao1_missed=0.0,
                draft_clusters=[],
            )

        rng = _stable_rng("value_cluster", source, column)
        # ``proposals_by_run[i]`` is the list of ProposedCluster
        # returned by the i-th LLM call. We keep them around so the
        # aggregator can vote on each species' kind/canonical.
        proposals_by_run: List[List[ProposedCluster]] = []

        def sample_fn(run_idx: int):
            order = [v for v, _ in distinct]
            if run_idx > 0 and len(order) > 1:
                rng.shuffle(order)
            # ``variant_idx=run_idx`` threads prompt-ensemble through
            # the proposer: run 0 uses the seed prompt, runs 1..K-1
            # use paraphrased variants when the caller installed any
            # (via ``ValueClusterProposer.variants``). Falls back to
            # the seed cleanly when ``variants`` is None.
            clusters = self.proposer.propose_once(
                source=source, column=column,
                value_counts=distinct, value_order=order,
                variant_idx=run_idx,
            )
            proposals_by_run.append(clusters)
            # Species = sorted-values tuple.
            return [c.as_species_key() for c in clusters]

        result: InnerLoopResult = run_adaptive(
            arm_id=f"value_cluster:{source}:{column}",
            sample_fn=sample_fn,
            cfg=self.arm_config,
        )

        draft = _aggregate_drafts(
            species_to_runs=result.species_to_runs,
            proposals_by_run=proposals_by_run,
            runs_completed=result.runs_completed,
        )

        return ColumnClusteringResult(
            source=source, column=column,
            runs_completed=result.runs_completed,
            stopped_reason=result.stopped_reason,
            gt_missing_mass=result.gt_missing_mass,
            chao1_missed=result.chao1_missed,
            draft_clusters=draft,
        )


# ---------------------------------------------------------------------------
# Aggregator (separable, testable without an LLM)
# ---------------------------------------------------------------------------


def _aggregate_drafts(
    *,
    species_to_runs: Dict[Any, List[int]],
    proposals_by_run: List[List[ProposedCluster]],
    runs_completed: int,
) -> List[DraftCluster]:
    """Build per-species DraftCluster records.

    For each species (sorted-values tuple) we collect every
    :class:`ProposedCluster` across all runs that surfaced it, then
    vote on its kind (most common; ties break alphabetical) and on
    its canonical (most common non-null; ``None`` wins when no run
    set one). Agreement is the fraction of runs that emitted this
    species.
    """
    # Group proposals by species key.
    by_species: Dict[Tuple[str, ...], List[ProposedCluster]] = {}
    for run_props in proposals_by_run:
        for p in run_props:
            by_species.setdefault(p.values, []).append(p)

    drafts: List[DraftCluster] = []
    for species_key, support_runs in species_to_runs.items():
        # ``species_key`` is the same tuple we stored above.
        props = by_species.get(species_key, [])
        if not props:
            # Shouldn't happen unless the inner loop and the proposer
            # log disagreed, but be defensive.
            continue
        kind_counts: Counter = Counter(p.kind.value for p in props)
        # ``props`` is non-empty by the guard above, so kind_counts is
        # always non-empty and ``most_common(1)[0]`` is safe. Use the
        # ``next(iter(...))`` form anyway so a future refactor that
        # removes the guard fails loudly with a ValueError on an empty
        # Counter rather than IndexError-ing on an empty list.
        top_kind = kind_counts.most_common(1)
        kind_value = (
            top_kind[0][0] if top_kind else ClusterKind.UNTAGGED.value
        )
        canonical_counts: Counter = Counter(
            p.canonical for p in props if p.canonical is not None
        )
        canonical: Optional[str] = None
        if canonical_counts:
            canonical, _ = canonical_counts.most_common(1)[0]
        # Target-style hint: most-common non-null wins; ties resolve
        # alphabetically via Counter's stable insertion ordering plus
        # an explicit min() over the top tier. ``None`` wins when no
        # run set one (typical for type-1 alias groups).
        target_style_counts: Counter = Counter(
            p.target_style for p in props if p.target_style is not None
        )
        target_style: Optional[str] = None
        if target_style_counts:
            top_n = target_style_counts.most_common(1)[0][1]
            tied = [t for t, n in target_style_counts.items() if n == top_n]
            target_style = min(tied)
        agreement = (
            len(support_runs) / runs_completed if runs_completed else 0.0
        )
        drafts.append(DraftCluster(
            values=species_key,
            kind=ClusterKind.from_string(kind_value),
            canonical=canonical,
            target_style=target_style,
            support_runs=sorted(support_runs),
            agreement=agreement,
        ))
    # Stable order: highest agreement first, ties by cluster size desc,
    # then alphabetical.
    drafts.sort(key=lambda d: (
        -d.agreement, -len(d.values), d.values,
    ))
    return drafts


# ---------------------------------------------------------------------------
# Per-run outer-loop sweep (mirrors the cleaning-detection adaptive layer)
# ---------------------------------------------------------------------------


# One column's mutable state as the sweep iterates over passes. The
# fields mirror the per-arm bookkeeping the chunk-scan adaptive layer
# (``experiments/run_cleaning_detection.py``) maintains across passes,
# scoped down to a single (source, column) clustering arm. We
# accumulate species across runs, recompute the GT/Chao1 estimators
# after each pass, and flip ``stopped`` once the configured stop
# predicate fires -- exactly the same gating logic
# :func:`src.adaptive.inner_loop.run_adaptive` uses per arm, just
# threaded out into an outer per-run loop so the seed prompt sweeps
# all columns in pass 0, variant 1 sweeps all (still-active) columns
# in pass 1, and so on.
@dataclass
class _ColumnArm:
    source: str
    column: str
    distinct: List[Tuple[str, int]]
    rng: random.Random
    proposals_by_run: List[List[ProposedCluster]] = field(default_factory=list)
    species_per_run: List[FrozenSet[Tuple[str, ...]]] = field(default_factory=list)
    species_to_runs: Dict[Tuple[str, ...], List[int]] = field(default_factory=dict)
    consecutive_no_new: int = 0
    runs_completed: int = 0
    stopped: bool = False
    stopped_reason: str = "k_max"
    gt_missing_mass: float = 0.0
    chao1_missed: float = 0.0


def _ingest_run(
    arm: _ColumnArm,
    run_idx: int,
    clusters: Sequence[ProposedCluster],
) -> int:
    """Fold one run's proposals into the arm's species index.

    Mirrors the inner-loop bookkeeping in
    :func:`src.adaptive.inner_loop.run_adaptive`: collapse duplicates
    within the same call, append to ``proposals_by_run`` /
    ``species_per_run`` / ``species_to_runs``, and update the
    consecutive-no-new streak. Returns the number of newly observed
    species this run (callers may log it for visibility).
    """
    props = list(clusters)
    arm.proposals_by_run.append(props)
    observed = frozenset(p.as_species_key() for p in props)
    arm.species_per_run.append(observed)

    new_seen = 0
    for sp in observed:
        if sp not in arm.species_to_runs:
            arm.species_to_runs[sp] = []
            new_seen += 1
        arm.species_to_runs[sp].append(run_idx)

    if new_seen == 0:
        arm.consecutive_no_new += 1
    else:
        arm.consecutive_no_new = 0
    return new_seen


def _evaluate_stop(arm: _ColumnArm, *, k_min: int, two_consec: bool, cfg: ArmConfig) -> None:
    """Apply the shared GT/Chao1 stop predicate to one arm.

    Idempotent and side-effect-only: when the predicate fires the arm
    flips to ``stopped=True`` and records the final estimator values
    + stop reason. Otherwise the arm is left unchanged so it stays in
    the active set for the next pass. This matches the per-arm
    ``saturated`` transition in the chunk-scan loop -- we just
    recompute the cumulative estimator from
    ``species_to_runs`` rather than going through
    ``estimate_by_arm`` because the species set is already in arm-
    local memory.
    """
    if arm.stopped or arm.runs_completed < k_min:
        return
    frequencies = [len(rs) for rs in arm.species_to_runs.values()]
    core = compute_core(frequencies, num_runs=arm.runs_completed)
    criterion_ok = is_saturated_by_estimate(
        n_observed=core.n_observed,
        good_turing_missing_mass=core.good_turing_missing_mass,
        chao1_ci_95_high=core.chao1_ci_95_high,
        epsilon=cfg.epsilon,
        epsilon_chao1=cfg.epsilon_chao1,
        stop_when=cfg.stop_when,
    )
    two_consec_ok = (not two_consec) or arm.consecutive_no_new >= 2
    if criterion_ok and two_consec_ok:
        arm.stopped = True
        arm.stopped_reason = (
            "two_consecutive"
            if (two_consec and arm.consecutive_no_new >= 2)
            else cfg.stop_when  # type: ignore[assignment]
        )
        arm.gt_missing_mass = core.good_turing_missing_mass
        arm.chao1_missed = core.chao1_estimated_missed


def _finalize_arm(arm: _ColumnArm, *, k_max: int) -> None:
    """Compute the final estimator values for an arm that ran to k_max.

    Arms that flipped ``stopped`` mid-loop already have their values
    set. Arms that consumed the full budget land here at the end so
    the on-disk artifact (``stopped_reason="k_max"``) carries the
    last-pass GT/Chao1 numbers rather than zeros.
    """
    if arm.stopped:
        return
    arm.stopped_reason = "k_max"
    if arm.runs_completed == 0:
        return
    frequencies = [len(rs) for rs in arm.species_to_runs.values()]
    core = compute_core(frequencies, num_runs=arm.runs_completed or k_max)
    arm.gt_missing_mass = core.good_turing_missing_mass
    arm.chao1_missed = core.chao1_estimated_missed


def run_sweep(
    *,
    columns: Sequence[Tuple[str, str, Sequence[Tuple[str, int]]]],
    proposer: ValueClusterProposer,
    arm_config: ArmConfig,
    max_workers: int = 1,
    on_pass_start: Optional[Callable[[int, int, int], None]] = None,
    on_arm_completed: Optional[Callable[[int, _ColumnArm, int], None]] = None,
    on_pass_end: Optional[Callable[[int, int, int, int], None]] = None,
) -> List[ColumnClusteringResult]:
    """Drive the column-clustering loop in **per-run** (outer-loop)
    order across all candidate columns.

    Pass 0 calls ``proposer.propose_once(..., variant_idx=0)`` (seed
    prompt) once per non-trivial column. Pass 1 calls it with
    ``variant_idx=1`` (first paraphrase) for every column whose
    Good-Turing / Chao1 estimator has not yet declared the arm
    saturated. The loop continues until either every arm has
    saturated or the configured ``k_max`` is reached -- the same
    contract :func:`src.adaptive.inner_loop.run_adaptive` provides
    per-arm.

    Reuses the saturation predicate from :mod:`src.adaptive._chao1`
    so the on-disk per-arm-estimates artifact stays comparable with
    the chunk-scan adaptive layer.

    Columns with fewer than two distinct values are returned without
    making any LLM call (``runs_completed=0``,
    ``stopped_reason="trivial"``); they show up in the final list in
    input order, alongside the actively-probed arms.

    The optional callbacks let CLI drivers attach progress reporting
    (tqdm / log lines) without leaking presentation concerns into
    this module.

    Returns one :class:`ColumnClusteringResult` per input candidate,
    sorted by ``(source, column)`` for stable on-disk output.
    """
    if (
        arm_config.k_max is None
        or arm_config.k_min is None
        or arm_config.epsilon is None
        or arm_config.epsilon_chao1 is None
        or arm_config.stop_when is None
    ):
        raise ValueError(
            "run_sweep: ArmConfig is missing required fields after "
            "resolution. Pass cfg = your_cfg.merge_into(ArmConfig.default())."
        )

    k_max = max(1, int(arm_config.k_max))
    k_min_raw = max(1, int(arm_config.k_min))
    if k_min_raw > k_max:
        logger.warning(
            "run_sweep: k_min (%d) > k_max (%d); clamping k_min to k_max.",
            k_min_raw, k_max,
        )
    k_min = min(k_min_raw, k_max)
    two_consec = bool(arm_config.require_two_consecutive)

    arms: List[_ColumnArm] = []
    trivial: List[ColumnClusteringResult] = []
    for source, column, value_counts in columns:
        distinct = [(v, c) for v, c in value_counts if v != ""]
        if len({v for v, _ in distinct}) < 2:
            trivial.append(ColumnClusteringResult(
                source=source, column=column,
                runs_completed=0, stopped_reason="trivial",
                gt_missing_mass=0.0, chao1_missed=0.0,
                draft_clusters=[],
            ))
            continue
        arms.append(_ColumnArm(
            source=source, column=column,
            distinct=distinct,
            rng=_stable_rng("value_cluster", source, column),
        ))

    if not arms:
        return trivial

    def _probe(arm: _ColumnArm, run_idx: int) -> Tuple[_ColumnArm, List[ProposedCluster]]:
        order = [v for v, _ in arm.distinct]
        if run_idx > 0 and len(order) > 1:
            arm.rng.shuffle(order)
        # ``variant_idx=run_idx`` threads prompt-ensemble through the
        # proposer (run 0 = seed prompt, run i = paraphrase i). The
        # proposer treats ``variant_idx`` >= len(variants)+1 as a
        # cyclic index, so passing the raw run index is safe even
        # when the ensemble is smaller than k_max.
        try:
            clusters = proposer.propose_once(
                source=arm.source, column=arm.column,
                value_counts=arm.distinct, value_order=order,
                variant_idx=run_idx,
            )
        except Exception as exc:  # noqa: BLE001
            # Mirror the inner-loop behaviour: a failed LLM call is
            # treated as "no species observed this run" rather than
            # tearing down the whole sweep. Operator can still see
            # what happened via the per-arm GT artifact (it will show
            # an arm that ran K passes with very few species).
            logger.warning(
                "run_sweep: propose_once failed for %s::%s on run %d: %s",
                arm.source, arm.column, run_idx, exc,
            )
            clusters = []
        return arm, clusters

    for run_idx in range(k_max):
        active = [a for a in arms if not a.stopped]
        if not active:
            break
        if on_pass_start is not None:
            on_pass_start(run_idx, len(active), len(arms))

        completed_this_pass = 0
        if max_workers <= 1:
            for arm in active:
                _, clusters = _probe(arm, run_idx)
                arm.runs_completed = run_idx + 1
                _ingest_run(arm, run_idx, clusters)
                _evaluate_stop(
                    arm, k_min=k_min, two_consec=two_consec, cfg=arm_config,
                )
                completed_this_pass += 1
                if on_arm_completed is not None:
                    on_arm_completed(run_idx, arm, completed_this_pass)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_probe, arm, run_idx): arm for arm in active}
                for fut in as_completed(futs):
                    arm, clusters = fut.result()
                    arm.runs_completed = run_idx + 1
                    _ingest_run(arm, run_idx, clusters)
                    _evaluate_stop(
                        arm, k_min=k_min, two_consec=two_consec, cfg=arm_config,
                    )
                    completed_this_pass += 1
                    if on_arm_completed is not None:
                        on_arm_completed(run_idx, arm, completed_this_pass)

        n_saturated_total = sum(1 for a in arms if a.stopped)
        n_active_next = sum(1 for a in arms if not a.stopped)
        if on_pass_end is not None:
            on_pass_end(
                run_idx, completed_this_pass, n_saturated_total, n_active_next,
            )

    for arm in arms:
        _finalize_arm(arm, k_max=k_max)

    out: List[ColumnClusteringResult] = list(trivial)
    for arm in arms:
        draft = _aggregate_drafts(
            species_to_runs=dict(arm.species_to_runs),
            proposals_by_run=arm.proposals_by_run,
            runs_completed=arm.runs_completed,
        )
        out.append(ColumnClusteringResult(
            source=arm.source, column=arm.column,
            runs_completed=arm.runs_completed,
            stopped_reason=arm.stopped_reason,
            gt_missing_mass=arm.gt_missing_mass,
            chao1_missed=arm.chao1_missed,
            draft_clusters=draft,
        ))
    out.sort(key=lambda r: (r.source, r.column))
    return out
