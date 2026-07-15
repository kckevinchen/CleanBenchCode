"""Exclusion endpoints under ``/api/runs/{run_id}/exclusions``.

Owns the per-run exclusion list (CSV / rows / columns scoped) that
feeds both the labelling UI's red-strike overlay and the
``export/sources`` sanitiser.

When an exclusion is added, every cluster that falls inside the new
scope gets an automatic ``skip`` cluster label so the queue advances
past it (matches the frontend's ``_auto_skip_for_exclusion`` contract).
The auto-skip labels are tagged in their ``note`` with a stable
``__exclusion=<id>`` token so the DELETE path can find and revert
them by appending ``UNLABELED`` tombstones.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from ..column_cluster_queue import iter_column_cluster_targets
from ..schema import (
    Exclusion,
    ExclusionIn,
    ExclusionScope,
    Label,
    LabelStatus,
    LabelTarget,
)
from ._context import RouterContext


logger = logging.getLogger(__name__)


# Sentinel embedded in the auto-skip label's ``note`` so the DELETE
# handler can find every cluster label written by ONE exclusion id
# without an extra index. Format: ``__exclusion=<id>``.
_EXCLUSION_TAG_PREFIX = "__exclusion="


def _exclusion_note(exclusion: Exclusion) -> str:
    """Build the ``note`` we attach to every auto-skip label.

    Includes the human-readable reason AND the machine-readable
    ``__exclusion=<id>`` token used by the undo path.
    """
    if exclusion.scope == ExclusionScope.CSV:
        reason = "csv"
    elif exclusion.scope == ExclusionScope.ROWS:
        reason = f"row in {sorted(int(r) for r in exclusion.rows)}"
    elif exclusion.scope == ExclusionScope.COLUMNS:
        reason = f"col in {sorted(exclusion.columns)}"
    else:
        reason = exclusion.scope
    return f"(auto: exclusion -> skip; {reason}) {_EXCLUSION_TAG_PREFIX}{exclusion.id}"


def _cluster_in_scope(cluster: Dict[str, Any], exclusion: Exclusion) -> bool:
    """Return True iff ``cluster`` falls within ``exclusion``'s scope.

    Mirrors the logic in :func:`src.labeling.app._build_exclusion_lookup`
    / :func:`_apply_exclusions_to_cells` so per-cell and per-cluster
    views stay consistent.
    """
    if (cluster.get("source") or "") != exclusion.source:
        return False
    if exclusion.scope == ExclusionScope.CSV:
        return True
    if exclusion.scope == ExclusionScope.ROWS:
        row = cluster.get("row")
        if not isinstance(row, int):
            return False
        try:
            wanted = {int(r) for r in exclusion.rows or []}
        except (TypeError, ValueError):
            return False
        return row in wanted
    if exclusion.scope == ExclusionScope.COLUMNS:
        cols = cluster.get("columns") or []
        wanted = {str(c) for c in exclusion.columns or [] if c is not None}
        return any(str(c) in wanted for c in cols)
    return False


def _cell_in_scope(
    source: str, row: Optional[int], column: Optional[str], exclusion: Exclusion,
) -> bool:
    """Return True iff a CELL-target label at ``(source, row, column)`` falls
    within ``exclusion``'s scope.

    Mirrors :func:`_cluster_in_scope` but for the per-cell FN labels the
    trivial taggers write (and any cell-level verdict the converter stamps).
    A ROWS-scope exclusion matches by row; a COLUMNS-scope by column; a
    CSV-scope matches every cell in the source.
    """
    if source != exclusion.source:
        return False
    if exclusion.scope == ExclusionScope.CSV:
        return True
    if exclusion.scope == ExclusionScope.ROWS:
        if not isinstance(row, int):
            return False
        try:
            wanted = {int(r) for r in exclusion.rows or []}
        except (TypeError, ValueError):
            return False
        return row in wanted
    if exclusion.scope == ExclusionScope.COLUMNS:
        if not column:
            return False
        wanted = {str(c) for c in exclusion.columns or [] if c is not None}
        return str(column) in wanted
    return False


def _column_in_scope(
    source: str, column: str, exclusion: Exclusion,
) -> bool:
    """Return True iff the ``(source, column)`` pair falls within
    ``exclusion``'s scope for the **column-cluster** target type.

    Column-clusters span the entire column, so ``ROWS`` exclusions do
    not apply (a row-scoped exclusion is a per-cell statement, while
    a column-cluster's verdict covers every distinct value in the
    column). ``CSV`` and ``COLUMNS`` scopes both match: a CSV-scope
    drops every column-cluster in the source; a COLUMNS-scope drops
    every column-cluster whose ``column`` is in the excluded set.
    """
    if source != exclusion.source:
        return False
    if exclusion.scope == ExclusionScope.CSV:
        return True
    if exclusion.scope == ExclusionScope.COLUMNS:
        wanted = {str(c) for c in exclusion.columns or [] if c is not None}
        return column in wanted
    # ROWS / anything else: column-clusters are not row-scoped.
    return False


def _cascade_skip_for_exclusion(
    run_ctx: Any, exclusion: Exclusion, *, ann: str,
) -> int:
    """Append ``skip`` labels for every label-page queue item inside
    ``exclusion``.

    Covers **both** target kinds the label page surfaces:

      * :class:`LabelTarget.CLUSTER` (cell-level clusters from
        ``clusters.jsonl``) -- gated by :func:`_cluster_in_scope`.
      * :class:`LabelTarget.COLUMN_CLUSTER` (inconsistency
        column-clusters from ``column_clusters.jsonl`` saved records
        and ``column_cluster_drafts.jsonl`` LLM drafts) -- gated by
        :func:`_column_in_scope`.

    Targets whose latest verdict is already ``skip`` are left alone
    (idempotent). Returns the total number of new labels written.
    """
    latest = run_ctx.store.latest_by_target()
    note = _exclusion_note(exclusion)
    batch: List[Label] = []

    # --- Cell clusters -------------------------------------------------
    for c in run_ctx.clusters:
        if not _cluster_in_scope(c, exclusion):
            continue
        cid = c.get("cluster_id")
        if not cid:
            continue
        existing = latest.get(cid)
        if (
            existing is not None
            and existing.target == LabelTarget.CLUSTER
            and existing.status == LabelStatus.SKIP
        ):
            continue
        cols = list(c.get("columns") or [])
        row = c.get("row")
        batch.append(Label(
            target=LabelTarget.CLUSTER,
            target_id=cid,
            source=c.get("source") or exclusion.source,
            row=int(row) if isinstance(row, int) else None,
            column=cols[0] if cols else (
                next(iter(exclusion.columns or []), None)
            ),
            columns=cols,
            status=LabelStatus.SKIP,
            corrected_value=None,
            note=note,
            annotator=ann,
        ))

    # --- Cell-target labels --------------------------------------------
    # FN cells (and any cell-level verdict the trivial converter stamped) are
    # NOT clusters, so the cluster loop above misses them. Convert every
    # in-scope CELL label that is not already ``skip`` so an excluded cell is
    # consistently ``skip`` regardless of whether it was tagged before or
    # after the exclusion. ``unlabeled`` tombstones are left alone.
    for tid, existing in latest.items():
        if existing.target != LabelTarget.CELL:
            continue
        if existing.status in (LabelStatus.SKIP, LabelStatus.UNLABELED):
            continue
        if not _cell_in_scope(
            existing.source or "", existing.row, existing.column, exclusion,
        ):
            continue
        prev = (existing.note or "").strip()
        batch.append(Label(
            target=LabelTarget.CELL,
            target_id=tid,
            source=existing.source,
            row=existing.row,
            column=existing.column,
            columns=list(existing.columns) if existing.columns else [],
            status=LabelStatus.SKIP,
            corrected_value=existing.corrected_value,
            corrected_values=(
                dict(existing.corrected_values) if existing.corrected_values else {}
            ),
            note=f"{prev} {note}".strip() if prev else note,
            annotator=ann,
        ))

    # --- Column clusters (inconsistency) -------------------------------
    # CSV / COLUMNS scopes cascade to the column-cluster queue too. A
    # ROWS-scope exclusion is a per-cell statement and does not touch
    # column-clusters (which span the whole column).
    if exclusion.scope in (ExclusionScope.CSV, ExclusionScope.COLUMNS):
        try:
            cc_targets = iter_column_cluster_targets(run_ctx.run_dir)
        except Exception as exc:  # noqa: BLE001 - never break the POST on a read error
            logger.warning(
                "exclusion cascade: column-cluster enumeration failed: %s",
                exc,
            )
            cc_targets = []
        for src, col, target_id in cc_targets:
            if not _column_in_scope(src, col, exclusion):
                continue
            existing = latest.get(target_id)
            if (
                existing is not None
                and existing.target == LabelTarget.COLUMN_CLUSTER
                and existing.status == LabelStatus.SKIP
            ):
                continue
            batch.append(Label(
                target=LabelTarget.COLUMN_CLUSTER,
                target_id=target_id,
                source=src,
                row=None,
                column=col,
                columns=[col],
                status=LabelStatus.SKIP,
                corrected_value=None,
                note=note,
                annotator=ann,
            ))

    if batch:
        run_ctx.store.append_many(batch)
    return len(batch)


def _revert_skip_for_exclusion(
    run_ctx: Any, exclusion_id: str, *, ann: str,
) -> int:
    """Append ``UNLABELED`` tombstones for every label this exclusion
    auto-skipped, across both cell-cluster **and** column-cluster
    targets.

    Finds the labels by the ``__exclusion=<id>`` token embedded in
    their note. The tombstone uses ``LabelStatus.UNLABELED`` which
    :meth:`LabelStore.latest_by_target` filters out, so consumers see
    the cluster as unlabeled again. Already-overwritten targets
    (operator manually relabelled after the auto-skip) are skipped:
    we never resurrect a stale verdict.
    """
    tag = f"{_EXCLUSION_TAG_PREFIX}{exclusion_id}"
    # Walk full history so we still find the auto-skip even if the
    # operator later wrote a different label on top. Cover both
    # cluster and column-cluster targets (the cascade above writes
    # to both).
    candidates: Dict[str, Label] = {}
    for rec in run_ctx.store.all():
        if rec.target not in (
            LabelTarget.CLUSTER, LabelTarget.COLUMN_CLUSTER, LabelTarget.CELL,
        ):
            continue
        if rec.status != LabelStatus.SKIP:
            continue
        if tag not in (rec.note or ""):
            continue
        # Keep the latest matching record per target so we never write
        # multiple tombstones for the same cluster on one undo.
        candidates[rec.target_id] = rec
    if not candidates:
        return 0

    # The undo must not overwrite a label the operator manually wrote
    # AFTER the auto-skip. The latest-wins reload guarantees that the
    # current ``_latest`` map points at whatever was last appended, so
    # if it isn't our auto-skip anymore, leave it alone.
    latest_raw = {}
    with run_ctx.store._lock:  # noqa: SLF001 - we own the contract
        latest_raw = dict(run_ctx.store._latest)
    tomb_batch: List[Label] = []
    for cid, src_rec in candidates.items():
        current = latest_raw.get(cid)
        if current is None or current.status != LabelStatus.SKIP:
            continue
        if tag not in (current.note or ""):
            # Operator wrote a fresh skip without our tag; respect it.
            continue
        tomb_batch.append(Label(
            target=src_rec.target,
            target_id=cid,
            source=src_rec.source,
            row=src_rec.row,
            column=src_rec.column,
            columns=list(src_rec.columns),
            status=LabelStatus.UNLABELED,
            corrected_value=None,
            note=f"(auto: exclusion undo) {tag}",
            annotator=ann,
        ))
    if tomb_batch:
        run_ctx.store.append_many(tomb_batch)
    return len(tomb_batch)


def register(app: FastAPI, ctx: RouterContext) -> None:
    cache = ctx.cache
    annotator = ctx.annotator

    @app.get("/api/runs/{run_id}/exclusions")
    def list_exclusions(run_id: str) -> Dict[str, Any]:
        run_ctx = cache.get(run_id)
        actives = run_ctx.exclusions_store.active()
        return {"items": [e.model_dump() for e in actives]}

    @app.post("/api/runs/{run_id}/exclusions")
    def post_exclusion(run_id: str, payload: ExclusionIn) -> Exclusion:
        run_ctx = cache.get(run_id)
        if payload.source not in run_ctx.sources:
            raise HTTPException(404, f"source not found: {payload.source}")
        if payload.scope == ExclusionScope.CSV:
            if payload.rows or payload.columns:
                raise HTTPException(
                    400, "scope=csv must not carry rows / columns",
                )
        elif payload.scope == ExclusionScope.ROWS:
            if not payload.rows:
                raise HTTPException(400, "scope=rows requires non-empty rows")
        elif payload.scope == ExclusionScope.COLUMNS:
            if not payload.columns:
                raise HTTPException(
                    400, "scope=columns requires non-empty columns",
                )
        ann = (payload.annotator or annotator).strip() or annotator
        record = run_ctx.exclusions_store.add(payload, annotator=ann)
        # Cascade: any cluster inside the new scope gets an auto-skip.
        # Errors here are non-fatal: the exclusion is already saved and
        # the operator can re-run the reconcile script offline.
        try:
            _cascade_skip_for_exclusion(run_ctx, record, ann=ann)
        except Exception:  # noqa: BLE001 - never break the POST on a cascade failure
            pass
        return record

    @app.delete("/api/runs/{run_id}/exclusions/{exclusion_id}")
    def delete_exclusion(run_id: str, exclusion_id: str) -> Dict[str, Any]:
        run_ctx = cache.get(run_id)
        tomb = run_ctx.exclusions_store.remove(exclusion_id, annotator=annotator)
        if tomb is None:
            raise HTTPException(404, f"exclusion not found: {exclusion_id}")
        # Revert the auto-skip labels we wrote on POST.
        try:
            _revert_skip_for_exclusion(
                run_ctx, exclusion_id, ann=annotator,
            )
        except Exception:  # noqa: BLE001 - same rationale as the POST cascade
            pass
        return {"ok": True, "id": exclusion_id}
