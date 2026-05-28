"""Slice 5: benchmark export for column-cluster labels.

Wraps :func:`experiments.score_column_clusters.materialize_inconsistency_benchmark`
with two responsibilities the raw scorer intentionally does not handle:

1. **Filename convention** — emit
   ``label_result/benchmark_inconsistency_<timestamp>Z_<tag>.jsonl``
   to match the existing pattern used by
   ``benchmark_typo_<ts>Z_<tag>.jsonl`` and
   ``benchmark_semantic_incompatible_<ts>Z_<tag>.jsonl``.

2. **Optional threshold-driven direction flip** — by default we emit
   bi-directional cells (``corrected_value`` is the sorted list of
   acceptable surface forms; the downstream cleaner may map any cell
   to any value). When the caller passes ``threshold T \\in (0, 1]``,
   every cluster whose ``count(canonical) / sum(counts)`` is ``>= T``
   flips to one-directional: non-canonical cells get scalar
   ``corrected_value = canonical_value`` (the polymorphic shape that
   signals a single ground-truth target); canonical cells are *not*
   emitted (they are already correct). Clusters that fail the
   threshold stay bi-directional. The threshold flip is independent
   of the operator's per-cluster ``confirmed_canonical`` (which
   already drives a one-directional fix at synthesis time); the two
   mechanisms compose without conflict.

This module deliberately depends on the Slice 2 scorer for the cell
enumeration; it does not re-implement the CSV-row matching.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.labeling.schema import (
    BenchmarkCell,
    ClusterKind,
    ClusterValue,
    ColumnClusters,
    ColumnLabelStatus,
    StdCluster,
    column_cluster_target_id,
)

# Re-use the scorer so we have a single source of truth for the
# CSV-row enumeration. The scorer lives in ``experiments`` because it
# was authored as a CLI first, but we treat it as a library here.
from experiments.score_column_clusters import (
    materialize_inconsistency_benchmark,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterFlipDecision:
    """Result of the threshold check for one (source, column, cluster).

    The exporter keeps these around so a downstream report can show
    "we flipped 7 of 15 clusters to one_directional at threshold 0.6".
    """

    source: str
    column: str
    cluster_id: str
    canonical_value: Optional[str]
    canonical_count: int
    total_count: int
    canonical_fraction: float
    flipped: bool


def _flip_cells_for_cluster(
    cells: List[BenchmarkCell],
    canonical: str,
) -> List[BenchmarkCell]:
    """Convert bi-directional cells of one cluster to one-directional.

    ``cells`` must all belong to the same global cluster_id.
    ``canonical`` is the operator-confirmed (or threshold-promoted)
    target value the cluster should be normalised to. Cells whose
    ``original_value`` equals the canonical are dropped (no error to
    correct); other cells get scalar ``corrected_value=canonical``
    (the polymorphic shape that signals one-directional in the
    unified ``BenchmarkCell`` schema).
    """
    out: List[BenchmarkCell] = []
    for cell in cells:
        if cell.original_value == canonical:
            # Already correct: a one_directional benchmark only
            # contains rows that need correcting.
            continue
        out.append(cell.model_copy(update={
            "corrected_value": canonical,
        }))
    return out


def _decide_flip(
    rec: ColumnClusters,
    threshold: Optional[float],
) -> Dict[str, ClusterFlipDecision]:
    """Compute a per-cluster flip decision for one column record.

    Returns ``cluster_id -> ClusterFlipDecision``. ``flipped=False``
    everywhere when ``threshold`` is None.

    The flip is driven by ``llm_proposed_canonical`` (the cluster
    proposer's hint), NOT ``canonical_value`` (which is reserved for
    operator-confirmed canonicals and already drives one-directional
    output via the scorer). The threshold path is therefore an
    opt-in power-user override that auto-promotes an unconfirmed
    LLM hint when the cluster's distribution is overwhelmingly
    dominated by that proposed canonical.

    Operator-confirmed clusters always emit one-directional from
    the scorer regardless of threshold; for those clusters the flip
    decision recorded here is informational only (the export's flip
    branch is gated on ``cluster.canonical_value is None`` so we
    never overwrite an operator-confirmed canonical).
    """
    decisions: Dict[str, ClusterFlipDecision] = {}
    for cluster in rec.clusters:
        total = sum(v.count for v in cluster.values)
        # Prefer the LLM hint (the unconfirmed proposal) for the
        # threshold check. Fall back to the operator-confirmed
        # canonical so the legacy CLI still surfaces a non-zero
        # ``canonical_fraction`` for confirmed clusters in the
        # sidecar report.
        candidate = cluster.llm_proposed_canonical or cluster.canonical_value
        canonical_count = 0
        for v in cluster.values:
            if v.value == candidate:
                canonical_count = v.count
                break
        fraction = (canonical_count / total) if total > 0 else 0.0
        flip = (
            threshold is not None
            and candidate is not None
            and total > 0
            and fraction >= threshold
        )
        decisions[cluster.cluster_id] = ClusterFlipDecision(
            source=rec.source,
            column=rec.column,
            cluster_id=cluster.cluster_id,
            canonical_value=candidate,
            canonical_count=canonical_count,
            total_count=total,
            canonical_fraction=fraction,
            flipped=flip,
        )
    return decisions


def export_inconsistency_benchmark(
    records: Iterable[ColumnClusters],
    data_dir: Path,
    *,
    threshold: Optional[float] = None,
    source_filter: Optional[str] = None,
    column_filter: Optional[str] = None,
) -> Tuple[List[BenchmarkCell], List[ClusterFlipDecision]]:
    """Materialize per-cell benchmark with optional one-directional flip.

    Returns ``(cells, decisions)``. ``decisions`` is one entry per
    (source, column, cluster) considered (not per cell) so callers
    can produce a flip-rate summary.

    The function loads each source CSV exactly once via the scorer's
    internal ``csv_cache``.

    Cells are emitted as the unified :class:`BenchmarkCell` (same
    schema as typo / semantic_incompatible exports). Direction is
    encoded by the polymorphic ``corrected_value``:
      * ``List[str]`` (sorted acceptable set) for bi-directional
        clusters -- the operator did not endorse a single canonical.
      * ``str`` (the canonical) for one-directional clusters -- the
        operator confirmed a canonical or the threshold flip
        promoted one.
    """
    if threshold is not None and not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"threshold must be in (0, 1] when provided, got {threshold!r}",
        )

    record_list = [
        r for r in records
        if r.status == ColumnLabelStatus.LABELED
        and (source_filter is None or r.source == source_filter)
        and (column_filter is None or r.column == column_filter)
    ]

    # 1. Enumerate cells via the bi-directional scorer.
    csv_cache: Dict[str, "object"] = {}  # type: ignore[type-arg]
    bi_cells = materialize_inconsistency_benchmark(
        record_list, data_dir,
        source_filter=source_filter,
        column_filter=column_filter,
        csv_cache=csv_cache,  # type: ignore[arg-type]
    )

    # 2. Compute per-cluster flip decisions.
    all_decisions: List[ClusterFlipDecision] = []
    decisions_by_gid: Dict[str, ClusterFlipDecision] = {}
    canonical_by_gid: Dict[str, Optional[str]] = {}
    for rec in record_list:
        rec_decisions = _decide_flip(rec, threshold)
        for cluster_id, dec in rec_decisions.items():
            gid = f"{rec.source}::{rec.column}::{cluster_id}"
            decisions_by_gid[gid] = dec
            canonical_by_gid[gid] = dec.canonical_value
            all_decisions.append(dec)

    # 3. Group cells by global cluster_id and flip the ones whose
    #    decision says so. Cells emitted by the scorer already carry
    #    the bi-directional shape (``corrected_value`` is a list);
    #    a flipped cluster overwrites that with the scalar canonical.
    cells_by_gid: Dict[str, List[BenchmarkCell]] = {}
    for cell in bi_cells:
        # ``cluster_ids`` is a list now; the synthesised records emit
        # exactly one cluster per cell so [0] is safe.
        gid = cell.cluster_ids[0] if cell.cluster_ids else ""
        cells_by_gid.setdefault(gid, []).append(cell)

    out_cells: List[BenchmarkCell] = []
    for gid, group in cells_by_gid.items():
        dec = decisions_by_gid.get(gid)
        canon = canonical_by_gid.get(gid)
        if dec is not None and dec.flipped and canon is not None:
            out_cells.extend(_flip_cells_for_cluster(group, canon))
        else:
            out_cells.extend(group)

    n_flipped = sum(1 for d in all_decisions if d.flipped)
    logger.info(
        "export: clusters_considered=%d flipped=%d threshold=%s "
        "cells_in=%d cells_out=%d",
        len(all_decisions), n_flipped, threshold,
        len(bi_cells), len(out_cells),
    )
    return out_cells, all_decisions


def benchmark_output_filename(
    timestamp: str,
    tag: Optional[str] = None,
) -> str:
    """Return ``benchmark_inconsistency_<ts>Z_<tag>.jsonl``.

    Matches the existing label_result/ convention. ``timestamp`` should
    already be the UTC ISO8601-basic form ``20260507T215150``; the
    trailing ``Z`` is appended automatically.
    """
    if tag:
        return f"benchmark_inconsistency_{timestamp}Z_{tag}.jsonl"
    return f"benchmark_inconsistency_{timestamp}Z.jsonl"


# ---------------------------------------------------------------------
# Per-cluster-label synthesis (drafts + labels.jsonl)
# ---------------------------------------------------------------------
#
# Inconsistency clusters are voted on individually in the labelling
# UI: each draft cluster gets its own TP/FP/Partial verdict, recorded
# in ``labels.jsonl`` keyed by ``column_cluster_target_id``. No
# ``ColumnClusters`` record is created by a verdict (only the column-
# board UI builds those, when the operator finalises a disjoint
# partition for a column).
#
# When the operator hasn't gone to the column board, we still want to
# export a benchmark from the per-cluster verdicts. The functions
# below synthesize :class:`ColumnClusters` records from drafts that
# carry a TP or Partial label, so the existing scorer/exporter pipeline
# can process them unchanged. Each labelled draft becomes its own
# single-cluster record (the ``ColumnClusters`` schema enforces
# disjointness within a record but not across records, so overlapping
# LLM proposals translate to multiple records on the same (source,
# column) pair).


# Verdict statuses that contribute to the benchmark. ``tp`` means
# the operator confirmed the cluster as a real inconsistency;
# ``partial`` means the cluster is mostly right but the operator
# noted a caveat in the label note. ``fp``/``unsure`` are excluded
# (the cluster's cells simply don't appear in the benchmark).
#
# ``skip`` is special — it follows the same exclusion rule as the
# cell-level export: a single skip on ANY draft for a (source,
# column) drops the WHOLE column from the benchmark, including
# sibling TP/Partial verdicts on that same column. Skip means
# "this column is unusable as ground truth" (e.g. data is too
# messy / spans languages / the operator wants to revisit it
# later), so emitting any of its cells would pollute the
# benchmark even if some clusters look right.
_BENCHMARK_VERDICT_STATUSES: frozenset = frozenset({"tp", "partial"})
_SKIP_STATUS = "skip"


def _latest_label_status_by_target(
    labels_path: Path,
) -> Dict[str, str]:
    """Return ``target_id -> latest status`` for column-cluster labels.

    Reads ``labels.jsonl`` (the workbench's label store) and keeps
    the most recent ``column_cluster``-target verdict per target_id.
    Order in the file is append-order so a later line wins.

    Missing or unparseable file returns an empty dict; the exporter
    silently emits zero cells in that case.
    """
    out: Dict[str, str] = {}
    if not labels_path.exists():
        return out
    try:
        with labels_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("target") != "column_cluster":
                    continue
                tid = rec.get("target_id") or ""
                status = rec.get("status") or ""
                if tid and status:
                    out[tid] = status
    except OSError as exc:
        logger.warning(
            "_latest_label_status_by_target: read failed: %s", exc,
        )
    return out


def _latest_confirmed_canonical_by_target(
    labels_path: Path,
) -> Dict[str, Optional[str]]:
    """Return ``target_id -> latest confirmed_canonical`` (may be None).

    Mirrors :func:`_latest_label_status_by_target` for the operator's
    confirmed-canonical pick. ``None`` (or absent key) means the
    operator did not endorse any value, so the cluster stays
    bi-directional in the export. Missing-on-record is treated
    identically to missing-from-file.
    """
    out: Dict[str, Optional[str]] = {}
    if not labels_path.exists():
        return out
    try:
        with labels_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("target") != "column_cluster":
                    continue
                tid = rec.get("target_id") or ""
                if not tid:
                    continue
                # Latest-line-wins: read the field even when null so
                # an explicit clearing (operator removed the canonical)
                # overrides an earlier confirmation.
                out[tid] = rec.get("confirmed_canonical")
    except OSError as exc:
        logger.warning(
            "_latest_confirmed_canonical_by_target: read failed: %s", exc,
        )
    return out


def synthesize_records_from_drafts_and_labels(
    drafts_path: Path,
    labels_path: Path,
    *,
    statuses: Iterable[str] = _BENCHMARK_VERDICT_STATUSES,
    annotator_fallback: str = "?",
) -> List[ColumnClusters]:
    """Synthesize one ``ColumnClusters`` record per labelled draft.

    Reads ``column_cluster_drafts.jsonl`` (LLM proposals) and joins
    against ``labels.jsonl`` (operator verdicts). For each draft
    cluster whose target_id (``colclust\\x00src\\x00col\\x00c{idx}``)
    has a verdict in ``statuses``, emits a single-cluster
    ``ColumnClusters`` record.

    Returns possibly multiple records per ``(source, column)`` pair
    when overlapping LLM drafts both received TP/Partial verdicts.
    The schema's disjointness check applies *within* a record (one
    cluster, no overlap with itself), so this is always valid even
    for overlapping drafts.

    **Skip-column rule** (mirror of the cell-level export). If
    *any* draft for a given ``(source, column)`` carries a ``skip``
    verdict, the entire column is excluded from the export — even
    sibling drafts on that column with TP/Partial verdicts are
    dropped. Skip means "this column is unusable as ground truth";
    a single skip cleanly removes the column the same way a single
    skip on a row removes the row from the cell-level benchmark.

    Designed for the export pipeline:
      records = synthesize_records_from_drafts_and_labels(
          run_dir / "column_cluster_drafts.jsonl",
          run_dir / "labels.jsonl",
      )
      cells, decisions = export_inconsistency_benchmark(records, ...)
    """
    keep_statuses = frozenset(statuses)
    if not drafts_path.exists():
        return []
    label_status = _latest_label_status_by_target(labels_path)
    confirmed_canonicals = _latest_confirmed_canonical_by_target(
        labels_path,
    )

    # First pass: identify (source, column) pairs that have at least
    # one ``skip`` verdict on any of their drafts. We need to know
    # this before deciding to keep any TP/Partial verdict on the
    # same column.
    skipped_columns: set = set()
    suppressed_verdicts = 0
    try:
        with drafts_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = doc.get("source") or ""
                column = doc.get("column") or ""
                if not source or not column:
                    continue
                drafts = doc.get("draft_clusters") or []
                for idx, _dr in enumerate(drafts, start=1):
                    target_id = column_cluster_target_id(
                        source, column, f"c{idx}",
                    )
                    if label_status.get(target_id) == _SKIP_STATUS:
                        skipped_columns.add((source, column))
                        break
    except OSError as exc:
        logger.warning(
            "synthesize_records: skip-pre-pass read failed: %s", exc,
        )
        return []

    records: List[ColumnClusters] = []
    try:
        with drafts_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = doc.get("source") or ""
                column = doc.get("column") or ""
                if not source or not column:
                    continue
                if (source, column) in skipped_columns:
                    # Track how many TP/Partial verdicts the skip rule
                    # suppressed so the operator sees it in the log.
                    drafts = doc.get("draft_clusters") or []
                    for idx, _dr in enumerate(drafts, start=1):
                        target_id = column_cluster_target_id(
                            source, column, f"c{idx}",
                        )
                        if label_status.get(target_id) in keep_statuses:
                            suppressed_verdicts += 1
                    continue
                drafts = doc.get("draft_clusters") or []
                for idx, dr in enumerate(drafts, start=1):
                    local_id = f"c{idx}"
                    target_id = column_cluster_target_id(
                        source, column, local_id,
                    )
                    status = label_status.get(target_id)
                    if status not in keep_statuses:
                        continue
                    confirmed_canon = confirmed_canonicals.get(target_id)
                    cluster = _draft_to_std_cluster(
                        local_id, dr,
                        confirmed_canonical=confirmed_canon,
                    )
                    if cluster is None:
                        continue
                    try:
                        records.append(ColumnClusters(
                            source=source,
                            column=column,
                            status=ColumnLabelStatus.LABELED,
                            clusters=[cluster],
                            annotator=annotator_fallback,
                            note=f"per-cluster verdict={status}",
                        ))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "synthesize: skip draft %s/%s/%s: %s",
                            source, column, local_id, exc,
                        )
    except OSError as exc:
        logger.warning(
            "synthesize_records_from_drafts_and_labels: "
            "read failed: %s", exc,
        )
    if skipped_columns:
        logger.info(
            "synthesize_records: skip rule excluded %d (source,column) "
            "pair(s); %d sibling TP/Partial verdict(s) suppressed",
            len(skipped_columns), suppressed_verdicts,
        )
    return records


def _draft_to_std_cluster(
    local_id: str, draft: Dict,
    *,
    confirmed_canonical: Optional[str] = None,
) -> Optional[StdCluster]:
    """Convert one draft dict to a :class:`StdCluster` instance.

    Mirrors the lenient parsing used elsewhere: skip entries where
    ``values`` would yield <2 valid strings; null out a canonical
    that isn't in the value set; default kind to ``untagged``.
    Returns ``None`` for unusable drafts.

    The LLM's ``draft.canonical`` (if any) becomes the cluster's
    ``llm_proposed_canonical`` — a hint surfaced downstream via
    ``BenchmarkCell.llm_suggested_value``. The cluster's
    ``canonical_value`` is set ONLY from ``confirmed_canonical``
    (the operator's explicit endorsement). This split is what
    enforces the "verified by the human reviewer" policy: an
    unconfirmed LLM proposal never flips the cluster to
    one-directional, but is still preserved in the export so
    downstream consumers can see the LLM's suggestion.
    """
    raw_values = draft.get("values") or []
    cluster_values: List[ClusterValue] = []
    for v in raw_values:
        if isinstance(v, dict):
            val = v.get("value")
            cnt = v.get("count")
        else:
            val = v
            cnt = None
        if not isinstance(val, str) or not val:
            continue
        try:
            cluster_values.append(
                ClusterValue(value=val, count=int(cnt or 1))
            )
        except Exception:  # noqa: BLE001
            continue
    if len(cluster_values) < 2:
        return None
    seen = {cv.value for cv in cluster_values}
    # Operator-confirmed canonical wins; only kept if it's a real
    # value of the cluster (defensive — the server already enforces
    # this, but a stale label could survive a draft re-cluster).
    canonical = confirmed_canonical
    if canonical is not None and canonical not in seen:
        canonical = None
    # LLM hint: draft may carry it under either ``canonical`` or
    # ``canonical_value`` (legacy field name); prefer the new one.
    llm_canon = draft.get("canonical") or draft.get("canonical_value")
    if llm_canon is not None and llm_canon not in seen:
        llm_canon = None
    kind_raw = draft.get("kind") or "untagged"
    try:
        kind = ClusterKind(kind_raw)
    except ValueError:
        kind = ClusterKind.UNTAGGED
    try:
        return StdCluster(
            cluster_id=local_id,
            kind=kind,
            canonical_value=canonical,
            llm_proposed_canonical=llm_canon,
            values=cluster_values,
        )
    except Exception:  # noqa: BLE001
        return None


def load_records_for_export(
    run_dir: Path,
    *,
    annotator_fallback: str = "?",
) -> List[ColumnClusters]:
    """Load records for the inconsistency benchmark export.

    Prefers the saved ``column_clusters.jsonl`` (column-board path)
    for any (source, column) it contains. Falls back to per-cluster
    verdicts on drafts (per-cluster labelling path) for columns the
    column board hasn't finalized.

    The two sources can be mixed within one run: some columns may
    have been finalized via the board (disjoint partition saved into
    ``column_clusters.jsonl``) while others were only voted on
    cluster-by-cluster from the LLM drafts.
    """
    from src.labeling.column_cluster_store import ColumnClusterStore

    saved_records: List[ColumnClusters] = []
    saved_columns: set = set()
    saved_path = run_dir / "column_clusters.jsonl"
    if saved_path.exists():
        store = ColumnClusterStore(saved_path)
        for (src, col), rec in store.latest_by_column().items():
            if rec.status == ColumnLabelStatus.LABELED:
                saved_records.append(rec)
                saved_columns.add((src, col))

    synth = synthesize_records_from_drafts_and_labels(
        run_dir / "column_cluster_drafts.jsonl",
        run_dir / "labels.jsonl",
        annotator_fallback=annotator_fallback,
    )
    # Skip synthesised records for columns the board has already
    # finalised: those rows are already represented by the saved
    # disjoint partition, and emitting both would double-count cells
    # in the benchmark.
    synth_filtered = [
        r for r in synth if (r.source, r.column) not in saved_columns
    ]

    logger.info(
        "load_records_for_export: %d saved record(s) + %d synthesized "
        "from drafts (drafts hidden for %d board-finalized columns)",
        len(saved_records), len(synth_filtered),
        len(synth) - len(synth_filtered),
    )
    return saved_records + synth_filtered
