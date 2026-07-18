"""Bulk-labelling / propagation endpoints.

Same-value FN propagation (cell + cluster level) and the trivial-tagger
endpoints (whitespace, phone-format). These spread one operator verdict
across same-value siblings or auto-label mechanical format variants, so
the human only has to rule once.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from .. import phone_format
from .._helpers import _has_whitespace_anomaly, _normalize_trivial, _scalar
from ..schema import Label, LabelStatus, LabelTarget, cell_target_id
from ._context import RouterContext

# ``_has_whitespace_anomaly`` (the strict whitespace-only rule) and
# ``_normalize_trivial`` (its deterministic fix) live in ``_helpers`` so
# the benchmark exporter can reuse them; they're imported above. Re-exported
# here for callers/tests that import them from this module.

# Note stamped on labels CREATED by the dataset scan (vs the bare
# " TRIVIAL" appended to an existing operator note). Carries the TRIVIAL
# token the benchmark exporter filters on, plus provenance.
_TRIVIAL_CREATE_NOTE = "(auto: spacing anomaly) TRIVIAL"

# Cluster verdicts that mean "this is a real error". Only these get a
# TRIVIAL tag; an fp / unsure / skip verdict is the operator saying it is
# NOT an error, so the scan leaves it alone.
_POSITIVE_STATUSES = (LabelStatus.TP, LabelStatus.PARTIAL, LabelStatus.FN)

# Cap on how many candidate rows the dry-run preview returns. The write
# phase rebuilds the full set server-side, so capping the wire payload
# never drops a write -- it only bounds the preview table.
_CANDIDATE_PREVIEW_CAP = 800


def _scan_dataset_for_trivial(
    ctx: Any, *, source_filter: Optional[str], dry_run: bool, ann: str,
) -> Dict[str, Any]:
    """Walk every cell of every (optionally filtered) source and act on
    each spacing-rule anomaly using the workbench's native propagation rules:

      * cell covered by a cluster
          - cluster already carries a positive verdict (tp/partial/fn)
            -> append ``TRIVIAL`` to its note (unless already tagged);
          - cluster has a NEGATIVE verdict (fp/unsure/skip)
            -> leave it (the operator said it is not an error);
          - cluster unlabelled -> create a TP verdict + ``TRIVIAL``.
      * cell NOT covered by any cluster
          - an existing positive cell label -> tag ``TRIVIAL``;
          - a negative (fp) cell label -> leave it;
          - no label -> create an FN cell label + ``TRIVIAL``.

    Creations carry the mechanically-normalised value as the fix.
    Idempotent: a re-run finds the ``TRIVIAL`` note and skips.
    """
    # (row, column) -> cluster ids covering it. A cluster covers its anchor
    # row's columns plus every detection's affected_rows (same as the
    # propagate-same-value coverage set), so multi-row clusters block their
    # own siblings from being mislabelled as uncovered FNs.
    cluster_by_loc: Dict[tuple, List[str]] = {}
    cluster_by_id: Dict[str, Dict[str, Any]] = {}
    for c in ctx.clusters:
        src = c.get("source")
        cid = c.get("cluster_id")
        if not src or not isinstance(cid, str):
            continue
        cluster_by_id[cid] = c
        cols = [x for x in (c.get("columns") or []) if isinstance(x, str)]
        r = c.get("row")
        if isinstance(r, int):
            for col in cols:
                cluster_by_loc.setdefault((src, r, col), []).append(cid)
        for det in (c.get("detections") or ()):
            det_cols = [
                x for x in (det.get("columns") or cols) if isinstance(x, str)
            ]
            for ar in (det.get("affected_rows") or ()):
                if isinstance(ar, int):
                    for col in det_cols:
                        cluster_by_loc.setdefault((src, ar, col), []).append(cid)

    latest = ctx.store.latest_by_target()
    cell_label_by_loc: Dict[tuple, Any] = {}
    for lab in latest.values():
        if (lab.target == LabelTarget.CELL and lab.source
                and lab.row is not None and lab.column):
            cell_label_by_loc[(lab.source, int(lab.row), lab.column)] = lab

    sources_to_scan = (
        [source_filter] if source_filter is not None else list(ctx.sources)
    )

    candidates: List[Dict[str, Any]] = []
    cells_scanned = 0
    anomaly_cells = 0
    skipped_already_tagged = 0
    skipped_negative_verdict = 0
    seen_cluster_ids: set = set()

    def _add(target, target_id, src, row, col, value, status, note, action,
             corrected=None):
        candidates.append({
            "target_id": target_id,
            "target": target,
            "source": src,
            "row": row,
            "column": col,
            "columns": [col],
            "ws_hits": [{"column": col, "value": value}],
            "status": status,
            "current_note": note,
            "already_tagged": action == "already_tagged",
            "action": action,
            "corrected_value": corrected,
        })

    for src in sources_to_scan:
        try:
            frame = ctx.load_frame(src)
        except (FileNotFoundError, KeyError):
            continue
        n = len(frame)
        for loc, col_obj in enumerate(frame.columns):
            col = str(col_obj)
            series = frame.iloc[:, loc].tolist()
            for row_int in range(n):
                cells_scanned += 1
                v = _scalar(series[row_int])
                if not _has_whitespace_anomaly(v):
                    continue
                anomaly_cells += 1
                fixed = _normalize_trivial(v)
                cids = cluster_by_loc.get((src, row_int, col)) or []
                if cids:
                    cid = cids[0]
                    if cid in seen_cluster_ids:
                        continue
                    seen_cluster_ids.add(cid)
                    existing = latest.get(cid)
                    if (existing is not None
                            and existing.target == LabelTarget.CLUSTER):
                        if existing.status not in _POSITIVE_STATUSES:
                            skipped_negative_verdict += 1
                            continue
                        note = existing.note or ""
                        if "TRIVIAL" in note.upper():
                            skipped_already_tagged += 1
                            _add("cluster", cid, src, row_int, col, v,
                                 existing.status.value, note, "already_tagged")
                            continue
                        _add("cluster", cid, src, row_int, col, v,
                             existing.status.value, note, "tag")
                    else:
                        _add("cluster", cid, src, row_int, col, v,
                             "tp", "", "create_tp", corrected=fixed)
                else:
                    existing = cell_label_by_loc.get((src, row_int, col))
                    if existing is not None:
                        if existing.status not in _POSITIVE_STATUSES:
                            skipped_negative_verdict += 1
                            continue
                        note = existing.note or ""
                        if "TRIVIAL" in note.upper():
                            skipped_already_tagged += 1
                            _add("cell", existing.target_id, src, row_int,
                                 col, v, existing.status.value, note,
                                 "already_tagged")
                            continue
                        _add("cell", existing.target_id, src, row_int, col,
                             v, existing.status.value, note, "tag")
                    else:
                        tid = cell_target_id(src, row_int, col)
                        _add("cell", tid, src, row_int, col, v,
                             "fn", "", "create_fn", corrected=fixed)

    # Stable preview order: by (source, row, column, action).
    candidates.sort(
        key=lambda c: (c["source"], c["row"], c["column"], c["action"])
    )

    to_tag = sum(1 for c in candidates if c["action"] == "tag")
    to_create_tp = sum(1 for c in candidates if c["action"] == "create_tp")
    to_create_fn = sum(1 for c in candidates if c["action"] == "create_fn")

    tagged = 0
    created_tp = 0
    created_fn = 0
    if not dry_run:
        batch: List[Label] = []
        for c in candidates:
            act = c["action"]
            if act == "tag":
                lab = latest[c["target_id"]]
                new_note = (
                    (lab.note + " TRIVIAL").strip() if lab.note else "TRIVIAL"
                )
                batch.append(Label(
                    target=lab.target,
                    target_id=lab.target_id,
                    source=lab.source,
                    row=lab.row,
                    column=lab.column,
                    columns=list(lab.columns) if lab.columns else [],
                    status=lab.status,
                    corrected_value=lab.corrected_value,
                    corrected_values=(
                        dict(lab.corrected_values) if lab.corrected_values else {}
                    ),
                    confirmed_canonical=lab.confirmed_canonical,
                    value_corrections=(
                        dict(lab.value_corrections) if lab.value_corrections else {}
                    ),
                    note=new_note,
                    annotator=ann,
                ))
                tagged += 1
            elif act == "create_tp":
                rec = cluster_by_id.get(c["target_id"], {})
                cols = [
                    x for x in (rec.get("columns") or [c["column"]])
                    if isinstance(x, str)
                ] or [c["column"]]
                rrow = rec.get("row")
                batch.append(Label(
                    target=LabelTarget.CLUSTER,
                    target_id=c["target_id"],
                    source=c["source"],
                    row=int(rrow) if isinstance(rrow, int) else c["row"],
                    column=cols[0],
                    columns=cols,
                    status=LabelStatus.TP,
                    corrected_value=c["corrected_value"],
                    note=_TRIVIAL_CREATE_NOTE,
                    annotator=ann,
                ))
                created_tp += 1
            elif act == "create_fn":
                batch.append(Label(
                    target=LabelTarget.CELL,
                    target_id=c["target_id"],
                    source=c["source"],
                    row=c["row"],
                    column=c["column"],
                    columns=[c["column"]],
                    status=LabelStatus.FN,
                    corrected_value=c["corrected_value"],
                    note=_TRIVIAL_CREATE_NOTE,
                    annotator=ann,
                ))
                created_fn += 1
        if batch:
            ctx.store.append_many(batch)

    preview = candidates[:_CANDIDATE_PREVIEW_CAP]
    return {
        "dry_run": dry_run,
        "scan_dataset": True,
        "cells_scanned": cells_scanned,
        "anomaly_cells": anomaly_cells,
        "scanned": cells_scanned,
        "to_tag": to_tag,
        "to_create_tp": to_create_tp,
        "to_create_fn": to_create_fn,
        "to_write": to_tag + to_create_tp + to_create_fn,
        "tagged": tagged,
        "created_tp": created_tp,
        "created_fn": created_fn,
        "skipped_already_tagged": skipped_already_tagged,
        "skipped_negative_verdict": skipped_negative_verdict,
        "skipped_no_value": 0,
        "candidates": preview,
        "candidates_truncated": len(candidates) > len(preview),
    }


# Note stamped on row-duplication labels CREATED by the exact-duplicate
# scan (vs the bare " TRIVIAL" appended to an existing operator note).
# Carries the TRIVIAL token the benchmark exporter filters on.
_TRIVIAL_DUP_CREATE_NOTE = "(auto: exact-duplicate row) TRIVIAL"

# row_duplication clusters are the only ones whose ``affected_rows`` we
# treat as a duplicate group; other multi-row types (FD / calc / cross-
# table) flag rows for a different reason, so an exact-value match there
# is not a "trivial duplicate".
_ROW_DUP_TYPE = "row_duplication"


def _row_signatures(frame: "pd.DataFrame") -> tuple:
    """Return ``(sig_by_row, sig_counts)`` for a frame.

    ``sig_by_row[r]`` is the full-row signature: a tuple of every
    column's ``_scalar``-coerced value, so two rows share a signature
    iff they are byte-identical across ALL columns (a "completely
    identical" / exact-duplicate row). ``sig_counts[sig]`` is how many
    rows carry that signature; ``>= 2`` means the row is part of an
    exact-duplicate group.
    """
    n = len(frame)
    # Column-major scalarisation: one ``.tolist()`` per column is far
    # cheaper than per-cell ``.iat`` on a wide frame.
    col_values = [frame.iloc[:, i].tolist() for i in range(frame.shape[1])]
    sig_by_row: List[tuple] = []
    sig_counts: Dict[tuple, int] = {}
    for r in range(n):
        sig = tuple(_scalar(col_values[c][r]) for c in range(len(col_values)))
        sig_by_row.append(sig)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
    return sig_by_row, sig_counts


def _scan_dataset_for_row_duplication(
    ctx: Any, *, source_filter: Optional[str], dry_run: bool, ann: str,
) -> Dict[str, Any]:
    """Auto-tag every *completely identical* (exact-duplicate) row group
    ``TRIVIAL``, mirroring :func:`_scan_dataset_for_trivial` but for the
    row-level ``row_duplication`` error type.

    A row is a trivial duplicate when another row in the same source is
    byte-identical across EVERY column. For each such group:

      * the group is covered by a ``row_duplication`` cluster whose
        ``affected_rows`` are themselves all byte-identical (i.e. the
        LLM flagged this exact duplicate)
          - cluster already carries a positive verdict (tp/partial/fn)
            -> append ``TRIVIAL`` to its note (unless already tagged);
          - cluster has a NEGATIVE verdict (fp/unsure/skip)
            -> leave it (the operator said it is not a duplicate);
          - cluster unlabelled -> create a TP verdict + ``TRIVIAL``.
      * a duplicate row NOT covered by any ``row_duplication`` cluster
        (the LLM missed it)
          - an existing positive row-level cell label -> tag ``TRIVIAL``;
          - a negative (fp) row-level cell label -> leave it;
          - no label -> create a row-level FN cell label + ``TRIVIAL``.

    Row-level FN labels carry ``column=""`` (the same sentinel the
    benchmark exporter uses for ``row_duplication`` cluster verdicts).
    Idempotent: a re-run finds the ``TRIVIAL`` note and skips.
    """
    # (source, row) -> covering row_duplication cluster ids. A cluster
    # covers its anchor row plus every detection's affected_rows.
    cluster_by_row: Dict[tuple, List[str]] = {}
    cluster_by_id: Dict[str, Dict[str, Any]] = {}
    # cid -> sorted list of every row the cluster touches.
    cluster_rows: Dict[str, List[int]] = {}
    for c in ctx.clusters:
        src = c.get("source")
        cid = c.get("cluster_id")
        if not src or not isinstance(cid, str):
            continue
        if _ROW_DUP_TYPE not in (c.get("error_types") or ()):
            continue
        cluster_by_id[cid] = c
        rows: set = set()
        r = c.get("row")
        if isinstance(r, int):
            rows.add(r)
        for det in (c.get("detections") or ()):
            for ar in (det.get("affected_rows") or ()):
                if isinstance(ar, int):
                    rows.add(ar)
        cluster_rows[cid] = sorted(rows)
        for rr in rows:
            cluster_by_row.setdefault((src, rr), []).append(cid)

    latest = ctx.store.latest_by_target()
    # Existing ROW-LEVEL cell labels (column empty) keyed by (source,row).
    cell_label_by_loc: Dict[tuple, Any] = {}
    for lab in latest.values():
        if (lab.target == LabelTarget.CELL and lab.source
                and lab.row is not None and not lab.column):
            cell_label_by_loc[(lab.source, int(lab.row))] = lab

    sources_to_scan = (
        [source_filter] if source_filter is not None else list(ctx.sources)
    )

    # Per-source row-signature cache so a source frame is hashed once.
    sig_cache: Dict[str, tuple] = {}

    def _sigs(src: str) -> tuple:
        if src not in sig_cache:
            sig_cache[src] = _row_signatures(ctx.load_frame(src))
        return sig_cache[src]

    candidates: List[Dict[str, Any]] = []
    rows_scanned = 0
    dup_rows = 0
    dup_groups = 0
    skipped_already_tagged = 0
    skipped_negative_verdict = 0

    def _add(target, target_id, src, row, group_rows, status, note, action):
        candidates.append({
            "target_id": target_id,
            "target": target,
            "source": src,
            "row": row,
            "group_rows": list(group_rows),
            "group_size": len(group_rows),
            "status": status,
            "current_note": note,
            "already_tagged": action == "already_tagged",
            "action": action,
        })

    # ---- Pass 1: row_duplication clusters whose affected_rows form an
    # exact-duplicate group (the LLM-found trivial duplicates).
    for cid, rows in cluster_rows.items():
        rec = cluster_by_id[cid]
        src = rec.get("source")
        if source_filter is not None and src != source_filter:
            continue
        if len(rows) < 2:
            continue
        try:
            sig_by_row, _counts = _sigs(src)
        except (FileNotFoundError, KeyError):
            continue
        n = len(sig_by_row)
        if any(not (0 <= r < n) for r in rows):
            continue
        # Trivial iff every affected row shares ONE full-row signature.
        if len({sig_by_row[r] for r in rows}) != 1:
            continue
        anchor = rec.get("row")
        anchor = anchor if isinstance(anchor, int) else rows[0]
        existing = latest.get(cid)
        if existing is not None and existing.target == LabelTarget.CLUSTER:
            if existing.status not in _POSITIVE_STATUSES:
                skipped_negative_verdict += 1
                continue
            note = existing.note or ""
            if "TRIVIAL" in note.upper():
                skipped_already_tagged += 1
                _add("cluster", cid, src, anchor, rows,
                     existing.status.value, note, "already_tagged")
                continue
            _add("cluster", cid, src, anchor, rows,
                 existing.status.value, note, "tag")
        else:
            _add("cluster", cid, src, anchor, rows, "tp", "", "create_tp")

    # ---- Pass 2: exact-duplicate rows NOT covered by any
    # row_duplication cluster (the LLM-missed trivial duplicates -> FN).
    for src in sources_to_scan:
        try:
            sig_by_row, sig_counts = _sigs(src)
        except (FileNotFoundError, KeyError):
            continue
        n = len(sig_by_row)
        rows_scanned += n
        # Group rows by signature so the preview can show the full
        # duplicate set each FN row belongs to.
        rows_by_sig: Dict[tuple, List[int]] = {}
        for r in range(n):
            rows_by_sig.setdefault(sig_by_row[r], []).append(r)
        for sig, group in rows_by_sig.items():
            if len(group) < 2:
                continue
            dup_groups += 1
            dup_rows += len(group)
            for r in group:
                if (src, r) in cluster_by_row:
                    # Covered by a row_duplication cluster: handled in
                    # pass 1 (or deliberately left for manual review on
                    # a conflicting cluster). Never double-label here.
                    continue
                existing = cell_label_by_loc.get((src, r))
                if existing is not None:
                    if existing.status not in _POSITIVE_STATUSES:
                        skipped_negative_verdict += 1
                        continue
                    note = existing.note or ""
                    if "TRIVIAL" in note.upper():
                        skipped_already_tagged += 1
                        _add("cell", existing.target_id, src, r, group,
                             existing.status.value, note, "already_tagged")
                        continue
                    _add("cell", existing.target_id, src, r, group,
                         existing.status.value, note, "tag")
                else:
                    tid = cell_target_id(src, r, "")
                    _add("cell", tid, src, r, group, "fn", "", "create_fn")

    # Stable preview order: by (source, row, action).
    candidates.sort(key=lambda c: (c["source"], c["row"], c["action"]))

    to_tag = sum(1 for c in candidates if c["action"] == "tag")
    to_create_tp = sum(1 for c in candidates if c["action"] == "create_tp")
    to_create_fn = sum(1 for c in candidates if c["action"] == "create_fn")

    tagged = 0
    created_tp = 0
    created_fn = 0
    if not dry_run:
        batch: List[Label] = []
        for c in candidates:
            act = c["action"]
            if act == "tag":
                lab = latest[c["target_id"]]
                new_note = (
                    (lab.note + " TRIVIAL").strip() if lab.note else "TRIVIAL"
                )
                batch.append(Label(
                    target=lab.target,
                    target_id=lab.target_id,
                    source=lab.source,
                    row=lab.row,
                    column=lab.column,
                    columns=list(lab.columns) if lab.columns else [],
                    status=lab.status,
                    corrected_value=lab.corrected_value,
                    corrected_values=(
                        dict(lab.corrected_values) if lab.corrected_values else {}
                    ),
                    confirmed_canonical=lab.confirmed_canonical,
                    value_corrections=(
                        dict(lab.value_corrections) if lab.value_corrections else {}
                    ),
                    note=new_note,
                    annotator=ann,
                ))
                tagged += 1
            elif act == "create_tp":
                rec = cluster_by_id.get(c["target_id"], {})
                rrow = rec.get("row")
                batch.append(Label(
                    target=LabelTarget.CLUSTER,
                    target_id=c["target_id"],
                    source=c["source"],
                    row=int(rrow) if isinstance(rrow, int) else c["row"],
                    column=None,
                    columns=[],
                    status=LabelStatus.TP,
                    note=_TRIVIAL_DUP_CREATE_NOTE,
                    annotator=ann,
                ))
                created_tp += 1
            elif act == "create_fn":
                batch.append(Label(
                    target=LabelTarget.CELL,
                    target_id=c["target_id"],
                    source=c["source"],
                    row=c["row"],
                    column="",
                    columns=[],
                    status=LabelStatus.FN,
                    note=_TRIVIAL_DUP_CREATE_NOTE,
                    annotator=ann,
                ))
                created_fn += 1
        if batch:
            ctx.store.append_many(batch)

    preview = candidates[:_CANDIDATE_PREVIEW_CAP]
    return {
        "dry_run": dry_run,
        "scan_dataset": True,
        "rows_scanned": rows_scanned,
        "scanned": rows_scanned,
        "dup_rows": dup_rows,
        "dup_groups": dup_groups,
        "to_tag": to_tag,
        "to_create_tp": to_create_tp,
        "to_create_fn": to_create_fn,
        "to_write": to_tag + to_create_tp + to_create_fn,
        "tagged": tagged,
        "created_tp": created_tp,
        "created_fn": created_fn,
        "skipped_already_tagged": skipped_already_tagged,
        "skipped_negative_verdict": skipped_negative_verdict,
        "skipped_no_value": 0,
        "candidates": preview,
        "candidates_truncated": len(candidates) > len(preview),
    }


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache
    annotator = router_ctx.annotator

    @app.post("/api/runs/{run_id}/labels/propagate-same-value")
    def propagate_same_value(
        run_id: str, payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """Auto-label every cell in ``column`` of ``source`` whose value
        equals the cell at (``row``, ``column``) as a false negative.

        Use case: the LLM flags a typo / bad value at one row but the
        same erroneous value also appears in other rows of the same
        column that the LLM never reported. After the operator marks
        the original cluster as TP (or PARTIAL), this endpoint walks
        the column and writes an FN cell label - with the SAME
        ``corrected_value`` - for every other row that holds an
        identical value, *unless* that cell is already flagged by ANY
        cluster (the LLM caught it -> not a false negative) or already
        carries an explicit cell label, so we never double-label an
        LLM-found cell or silently overwrite a prior decision.

        Required keys: ``source``, ``row``, ``column``, ``corrected_value``.
        Optional: ``note``, ``annotator``, ``error_types``.
        Response: ``{"propagated": int, "skipped_covered": int,
                     "skipped_existing": int, "matched": int, "cells": [...]}``.
        """
        source = payload.get("source")
        row = payload.get("row")
        column = payload.get("column")
        corrected = payload.get("corrected_value")
        note = (payload.get("note") or "").strip()
        ann = (payload.get("annotator") or annotator).strip() or annotator
        if not source or row is None or not column:
            raise HTTPException(400, "source, row, column are required")
        try:
            row = int(row)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"invalid row: {row!r}") from exc
        if corrected is None or str(corrected).strip() == "":
            raise HTTPException(
                400,
                "corrected_value is required (use the operator's correction so "
                "all FN siblings inherit the same fix)",
            )

        ctx = cache.get(run_id)
        if source not in ctx.sources:
            raise HTTPException(404, f"source not found: {source}")
        try:
            frame = ctx.load_frame(source)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        if column not in frame.columns:
            raise HTTPException(
                404,
                f"column not found in {source}: {column!r}",
            )
        if not (0 <= row < len(frame)):
            raise HTTPException(400, f"row {row} out of range for {source}")

        # Reference value = the cell value the operator just labelled
        # TP. Compare other cells using the same ``_scalar`` coercion
        # the slice/cells endpoints use so the operator's mental model
        # of "this value equals that value" matches what gets written.
        target_value = _scalar(frame.iat[row, frame.columns.get_loc(column)])

        # A same-value cell is an FN candidate only if NO cluster already
        # flagged it. Build the covered set from EVERY cluster on this
        # source (its anchor row + each detection's ``affected_rows``),
        # regardless of whether the operator has verdicted it yet:
        #   * the LLM flagged those cells, so they are NOT false
        #     negatives -- the operator will rule on the owning cluster;
        #   * auto-FN'ing a still-unlabelled sibling cluster's cell would
        #     pre-empt that cluster and double-label a cell the LLM found.
        # (Previously only clusters carrying an active verdict blocked
        # propagation, which wrongly FN'd same-value cells belonging to
        # unlabelled sibling clusters -- and, before detection coverage
        # existed, a cluster's OWN multi-row siblings.) Existing explicit
        # cell labels are honoured separately below so we never clobber a
        # prior annotator decision.
        covered: set = set()
        for c in ctx.clusters:
            if c.get("source") != source:
                continue
            cluster_cols = list(c.get("columns") or [])
            r = c.get("row")
            if isinstance(r, int):
                for col in cluster_cols:
                    covered.add((r, col))
            # Detections often list every row holding the same bad value
            # (the anchor is just one of them); cover them all, falling
            # back to the cluster's columns when a detection omits its own.
            for det in (c.get("detections") or ()):
                det_cols = list(det.get("columns") or cluster_cols)
                for ar in (det.get("affected_rows") or ()):
                    if isinstance(ar, int):
                        for col in det_cols:
                            covered.add((ar, col))

        # Existing explicit cell labels at (source, row, col) - e.g. a
        # prior FN with its own correction - must also be left untouched.
        latest = ctx.store.latest_by_target()
        cell_labels: set = set()
        for tid, lab in latest.items():
            if lab.target != LabelTarget.CELL or lab.source != source:
                continue
            if lab.row is None or not lab.column:
                continue
            cell_labels.add((int(lab.row), lab.column))

        propagated_cells: List[Dict[str, Any]] = []
        batch: List[Label] = []
        skipped_covered = 0
        skipped_existing = 0
        matched = 0
        col_loc = frame.columns.get_loc(column)
        for idx in range(len(frame)):
            if idx == row:
                continue
            v = _scalar(frame.iat[idx, col_loc])
            if v != target_value:
                continue
            matched += 1
            if (idx, column) in covered:
                skipped_covered += 1
                continue
            if (idx, column) in cell_labels:
                skipped_existing += 1
                continue
            batch.append(Label(
                target=LabelTarget.CELL,
                target_id=cell_target_id(source, idx, column),
                source=source,
                row=idx,
                column=column,
                status=LabelStatus.FN,
                corrected_value=str(corrected),
                note=note or "(auto: same-value as labelled TP cell)",
                annotator=ann,
            ))
            propagated_cells.append({"row": idx, "column": column})
        # Single fsync for the whole propagation: avoids the 10ms-per-
        # record cost on /datadrive that turned a 30-row propagation
        # into a >300ms stall.
        ctx.store.append_many(batch)

        return {
            "source": source,
            "column": column,
            "anchor_row": row,
            "anchor_value": target_value,
            "matched": matched,
            "propagated": len(propagated_cells),
            "skipped_covered": skipped_covered,
            "skipped_existing": skipped_existing,
            "cells": propagated_cells,
        }

    @app.post("/api/runs/{run_id}/labels/tag-trivial-whitespace")
    def tag_trivial_whitespace(
        run_id: str, payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Dict[str, Any]:
        """Append the literal token ``TRIVIAL`` to the ``note`` of every
        TP / PARTIAL / FN label whose underlying cell value matches the
        red-underline whitespace-anomaly rule used by the CSV viewer.

        Why this exists: the workbench's whitespace scan finds edge /
        double / non-printable whitespace in cells. Such typos
        are mechanical and uninteresting compared with semantic typos,
        so the operator wants them flagged ``TRIVIAL`` in the note so
        downstream benchmark consumers can choose to weight or filter
        them out without losing the label.

        Idempotent: any label whose latest ``note`` already contains
        ``TRIVIAL`` (case-insensitive substring) is skipped. Re-running
        the endpoint after manual additions is a no-op.

        Payload (all optional):
          * ``source``   - limit the scan to one CSV.
          * ``statuses`` - list of statuses to include (default
            ``["tp", "partial", "fn"]``). The ``fn`` cell-target labels
            from the auto-whitespace pass are the bulk of real
            whitespace typos; ``tp``/``partial`` cluster verdicts pick
            up the LLM-found ones.
          * ``dry_run``  - if true, return the candidate list without
            writing any labels. The UI uses this to power the preview.
          * ``annotator`` - override annotator on the new records.
        Response always includes ``candidates`` (list of dicts with
        ``target_id``, ``source``, ``row``, ``column``, ``value``,
        ``status``, ``current_note``) plus counts:
          * ``tagged``                - records appended (0 in dry_run)
          * ``skipped_already_tagged`` - candidates whose note already
            had ``TRIVIAL``
          * ``skipped_no_value``       - labels whose target cell could
            not be read (deleted source / out-of-range row)
          * ``scanned``                - latest labels considered
        """
        ctx = cache.get(run_id)
        source_filter = payload.get("source") or None
        if source_filter is not None and source_filter not in ctx.sources:
            raise HTTPException(404, f"source not found: {source_filter}")
        dry_run = bool(payload.get("dry_run"))
        raw_statuses = payload.get("statuses")
        if raw_statuses is None:
            wanted_statuses = {
                LabelStatus.TP, LabelStatus.PARTIAL, LabelStatus.FN,
            }
        else:
            if not isinstance(raw_statuses, list) or not raw_statuses:
                raise HTTPException(
                    400, "statuses must be a non-empty list of status names",
                )
            wanted_statuses = set()
            for s in raw_statuses:
                try:
                    wanted_statuses.add(LabelStatus(str(s).lower()))
                except ValueError as exc:
                    raise HTTPException(
                        400, f"invalid status: {s!r}",
                    ) from exc
        ann = (payload.get("annotator") or annotator).strip() or annotator

        # scan_dataset: instead of only re-noting EXISTING labels, walk
        # every cell of every (filtered) source and auto-label each
        # spacing-rule anomaly TRIVIAL -- covered cells become TP,
        # uncovered cells become FN. See _scan_dataset_for_trivial.
        if bool(payload.get("scan_dataset")):
            return _scan_dataset_for_trivial(
                ctx, source_filter=source_filter, dry_run=dry_run, ann=ann,
            )

        latest = ctx.store.latest_by_target()
        # Index clusters by id for O(1) lookup of (row, columns) given
        # a CLUSTER label.
        cluster_by_id: Dict[str, Dict[str, Any]] = {}
        for c in ctx.clusters:
            cid = c.get("cluster_id")
            if isinstance(cid, str):
                cluster_by_id[cid] = c

        # Per-source dataframe cache: ``ctx.load_frame`` already
        # memoises but we capture failures to avoid reloading a missing
        # source for every candidate.
        frame_cache: Dict[str, Optional[pd.DataFrame]] = {}

        def _frame_for(src: str) -> Optional[pd.DataFrame]:
            if src in frame_cache:
                return frame_cache[src]
            try:
                fr = ctx.load_frame(src)
            except (FileNotFoundError, KeyError):
                fr = None
            frame_cache[src] = fr
            return fr

        candidates: List[Dict[str, Any]] = []
        skipped_already_tagged = 0
        skipped_no_value = 0
        scanned = 0
        # Stable order: by (source, row, column, target_id) so the
        # preview is reproducible run-to-run.
        items = sorted(
            latest.items(),
            key=lambda kv: (
                kv[1].source or "",
                kv[1].row if kv[1].row is not None else -1,
                kv[1].column or "",
                kv[0],
            ),
        )
        for tid, lab in items:
            if lab.status not in wanted_statuses:
                continue
            scanned += 1
            # Resolve (source, row, column) for the label. CELL labels
            # carry these directly; CLUSTER labels need a cluster lookup.
            # COLUMN_CLUSTER and ISSUE labels don't fit the spacing-rule
            # model (no single anchor cell), so skip them.
            if lab.target == LabelTarget.CELL:
                src = lab.source
                row_idx = lab.row
                cols_to_check: List[str] = [lab.column] if lab.column else []
            elif lab.target == LabelTarget.CLUSTER:
                rec = cluster_by_id.get(lab.target_id)
                if rec is None:
                    continue
                src = rec.get("source")
                row_idx = rec.get("row")
                cols_to_check = [
                    c for c in (lab.columns or rec.get("columns") or [])
                    if isinstance(c, str)
                ]
            else:
                continue

            if not src or row_idx is None or not cols_to_check:
                continue
            if source_filter is not None and src != source_filter:
                continue
            frame = _frame_for(src)
            if frame is None:
                skipped_no_value += 1
                continue
            try:
                row_int = int(row_idx)
            except (TypeError, ValueError):
                skipped_no_value += 1
                continue
            if not (0 <= row_int < len(frame)):
                skipped_no_value += 1
                continue

            # Find the columns whose value triggers the spacing rule.
            # Even if the operator labelled multiple columns at once,
            # we only flag TRIVIAL when at least one of them is a true
            # whitespace anomaly. Surface the matching cells so the
            # preview can show the operator exactly what's being marked.
            ws_hits: List[Dict[str, Any]] = []
            for col in cols_to_check:
                if col not in frame.columns:
                    continue
                v = _scalar(frame.iat[row_int, frame.columns.get_loc(col)])
                if _has_whitespace_anomaly(v):
                    ws_hits.append({"column": col, "value": v})
            if not ws_hits:
                continue

            current_note = lab.note or ""
            already_tagged = "TRIVIAL" in current_note.upper()
            if already_tagged:
                skipped_already_tagged += 1
                # Surface the entry so the UI can show "X already
                # tagged" without an extra request, but mark it so
                # the preview doesn't double-count it.
                candidates.append({
                    "target_id": tid,
                    "target": lab.target.value,
                    "source": src,
                    "row": row_int,
                    "column": cols_to_check[0],
                    "columns": cols_to_check,
                    "ws_hits": ws_hits,
                    "status": lab.status.value,
                    "current_note": current_note,
                    "already_tagged": True,
                })
                continue

            candidates.append({
                "target_id": tid,
                "target": lab.target.value,
                "source": src,
                "row": row_int,
                "column": cols_to_check[0],
                "columns": cols_to_check,
                "ws_hits": ws_hits,
                "status": lab.status.value,
                "current_note": current_note,
                "already_tagged": False,
            })

        to_tag = [c for c in candidates if not c["already_tagged"]]
        if dry_run:
            return {
                "dry_run": True,
                "scanned": scanned,
                "tagged": 0,
                "to_tag": len(to_tag),
                "skipped_already_tagged": skipped_already_tagged,
                "skipped_no_value": skipped_no_value,
                "candidates": candidates,
            }

        # Build new label records that mirror the latest existing
        # record for each target but with " TRIVIAL" appended to the
        # note. The label store is append-only and ``latest_by_target``
        # already returns the most-recent fields, so we can copy
        # everything verbatim.
        batch: List[Label] = []
        for entry in to_tag:
            lab = latest[entry["target_id"]]
            new_note = (lab.note + " TRIVIAL").strip() if lab.note else "TRIVIAL"
            batch.append(Label(
                target=lab.target,
                target_id=lab.target_id,
                source=lab.source,
                row=lab.row,
                column=lab.column,
                columns=list(lab.columns) if lab.columns else [],
                status=lab.status,
                corrected_value=lab.corrected_value,
                corrected_values=dict(lab.corrected_values) if lab.corrected_values else {},
                confirmed_canonical=lab.confirmed_canonical,
                value_corrections=(
                    dict(lab.value_corrections) if lab.value_corrections else {}
                ),
                note=new_note,
                annotator=ann,
            ))
        if batch:
            ctx.store.append_many(batch)

        return {
            "dry_run": False,
            "scanned": scanned,
            "tagged": len(batch),
            "to_tag": len(to_tag),
            "skipped_already_tagged": skipped_already_tagged,
            "skipped_no_value": skipped_no_value,
            "candidates": candidates,
        }

    @app.post("/api/runs/{run_id}/labels/tag-trivial-row-duplication")
    def tag_trivial_row_duplication(
        run_id: str, payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Dict[str, Any]:
        """Auto-tag every *completely identical* (exact-duplicate) row
        ``TRIVIAL``, the row-level analogue of
        ``tag-trivial-whitespace``'s ``scan_dataset`` mode.

        A row is a trivial duplicate when another row in the same source
        is byte-identical across EVERY column. The scan walks each
        (optionally filtered) source and, per exact-duplicate group:

          * a ``row_duplication`` cluster whose ``affected_rows`` are all
            byte-identical (the LLM flagged this exact duplicate) becomes
            **TP** + ``TRIVIAL`` when unlabelled, has ``TRIVIAL`` appended
            when already tp/partial/fn, and is left untouched when the
            operator ruled it fp / unsure / skip;
          * a duplicate row NOT covered by any ``row_duplication`` cluster
            (the LLM missed it) gets a row-level **FN** cell label
            (``column=""``) + ``TRIVIAL``.

        Conflicting / partial duplicates (same key, differing other
        columns) are NOT touched - only fully identical rows.

        Payload (all optional):
          * ``source``    - limit the scan to one CSV (404 if absent).
          * ``dry_run``   - return candidates without writing (preview).
          * ``annotator`` - override annotator on the new records.

        Response mirrors the whitespace scan: ``candidates`` plus
        ``rows_scanned`` / ``dup_rows`` / ``dup_groups`` /
        ``to_tag`` / ``to_create_tp`` / ``to_create_fn`` / ``to_write``
        and the post-write ``tagged`` / ``created_tp`` / ``created_fn`` /
        ``skipped_already_tagged`` / ``skipped_negative_verdict`` counts.
        Idempotent: rows whose latest note already contains ``TRIVIAL``
        are skipped.
        """
        ctx = cache.get(run_id)
        source_filter = payload.get("source") or None
        if source_filter is not None and source_filter not in ctx.sources:
            raise HTTPException(404, f"source not found: {source_filter}")
        dry_run = bool(payload.get("dry_run"))
        ann = (payload.get("annotator") or annotator).strip() or annotator
        return _scan_dataset_for_row_duplication(
            ctx, source_filter=source_filter, dry_run=dry_run, ann=ann,
        )

    @app.post("/api/runs/{run_id}/labels/tag-trivial-phone-format")
    def tag_trivial_phone_format(
        run_id: str, payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Dict[str, Any]:
        """Auto-detect phone-format columns, bulk-label the format
        variants as FN, and SKIP every cluster touching such a column.

        Phone numbers stored in mixed surface formats (``(575) 433-0200``
        vs ``575-433-0200``) are a mechanical inconsistency, not a
        semantic typo. This endpoint mirrors ``tag-trivial-whitespace``:
        it finds columns whose values are >= ``threshold`` phone-shaped
        (``src.labeling.phone_format``), votes a column-majority canonical
        template, and for every row whose surface form differs (but whose
        digit count matches, so a clean reformat exists) writes a
        cell-level FN label carrying the canonical ``corrected_value``.
        Clusters anchored on a phone column are SKIPped so they drop out
        of the active queue. Both records are tagged ``TRIVIAL
        phone_format`` in the note.

        Idempotent: a cell / cluster whose latest label already carries
        the ``TRIVIAL phone_format`` tag is surfaced as ``already_tagged``
        and never re-written, so re-running is a no-op.

        Payload (all optional):
          * ``source``         - limit the scan to one CSV (404 if absent).
          * ``threshold``      - min phone-shaped fraction for a column to
            count as a phone column (default 0.80).
          * ``min_phone_rows`` - min phone-shaped rows required (default 5).
          * ``dry_run``        - return candidates without writing.
          * ``annotator``      - override annotator on the new records.
        Response: ``columns`` (one ``{source, column, template,
        digit_count, cells_to_correct, clusters_to_skip}`` per detected
        column), ``cell_candidates``, ``cluster_candidates`` (each with
        ``already_tagged``), and the ``cells_written`` / ``clusters_written``
        counts (0 in dry_run / on an idempotent re-run).
        """
        ctx = cache.get(run_id)
        source_filter = payload.get("source") or None
        if source_filter is not None and source_filter not in ctx.sources:
            raise HTTPException(404, f"source not found: {source_filter}")
        dry_run = bool(payload.get("dry_run"))
        try:
            threshold = float(
                payload.get("threshold", phone_format.DEFAULT_COLUMN_THRESHOLD)
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "threshold must be a number") from exc
        try:
            min_phone_rows = int(
                payload.get("min_phone_rows", phone_format.DEFAULT_MIN_PHONE_ROWS)
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "min_phone_rows must be an integer") from exc
        ann = (payload.get("annotator") or annotator).strip() or annotator

        phone_tag = "TRIVIAL phone_format"
        latest = ctx.store.latest_by_target()

        def _already_tagged(target_id: str) -> bool:
            lab = latest.get(target_id)
            return lab is not None and phone_tag.lower() in (lab.note or "").lower()

        sources_to_scan = (
            [source_filter] if source_filter is not None else list(ctx.sources)
        )

        columns_summary: List[Dict[str, Any]] = []
        cell_candidates: List[Dict[str, Any]] = []
        cluster_candidates: List[Dict[str, Any]] = []
        seen_cluster_ids: set[str] = set()

        for src in sources_to_scan:
            try:
                frame = ctx.load_frame(src)
            except (FileNotFoundError, KeyError):
                continue
            for col_obj in frame.columns:
                col = str(col_obj)
                rows = [
                    (i, "" if pd.isna(v) else str(v))
                    for i, v in enumerate(frame[col_obj].tolist())
                ]
                majority, corrections = phone_format.find_phone_corrections(
                    rows, threshold=threshold, min_phone_rows=min_phone_rows,
                )
                if majority is None:
                    continue
                template, digit_count = majority

                col_cells: List[Dict[str, Any]] = []
                for corr in corrections:
                    tid = cell_target_id(src, corr.row, col)
                    col_cells.append({
                        "target_id": tid,
                        "source": src,
                        "row": corr.row,
                        "column": col,
                        "original_value": corr.original,
                        "canonical": corr.canonical,
                        "already_tagged": _already_tagged(tid),
                    })

                col_clusters: List[Dict[str, Any]] = []
                for c in ctx.clusters:
                    if c.get("source") != src:
                        continue
                    cid = c.get("cluster_id")
                    if not isinstance(cid, str) or cid in seen_cluster_ids:
                        continue
                    ccols = [
                        x for x in (c.get("columns") or []) if isinstance(x, str)
                    ]
                    if col not in ccols:
                        continue
                    # Never overwrite a human verdict: if the operator
                    # already ruled this phone-column cluster
                    # tp/fp/partial/unsure, leave it (only unruled /
                    # skip clusters are eligible for the trivial skip).
                    _lab = latest.get(cid)
                    if (_lab is not None
                            and _lab.target == LabelTarget.CLUSTER
                            and _lab.status in (
                                LabelStatus.TP, LabelStatus.FP,
                                LabelStatus.PARTIAL, LabelStatus.UNSURE,
                            )):
                        continue
                    seen_cluster_ids.add(cid)
                    col_clusters.append({
                        "target_id": cid,
                        "source": src,
                        "row": c.get("row"),
                        "columns": ccols,
                        "already_tagged": _already_tagged(cid),
                    })

                columns_summary.append({
                    "source": src,
                    "column": col,
                    "template": template,
                    "digit_count": digit_count,
                    "cells_to_correct": len(col_cells),
                    "clusters_to_skip": len(col_clusters),
                })
                cell_candidates.extend(col_cells)
                cluster_candidates.extend(col_clusters)

        cells_written = 0
        clusters_written = 0
        if not dry_run:
            batch: List[Label] = []
            for cand in cell_candidates:
                if cand["already_tagged"]:
                    continue
                batch.append(Label(
                    target=LabelTarget.CELL,
                    target_id=cand["target_id"],
                    source=cand["source"],
                    row=cand["row"],
                    column=cand["column"],
                    columns=[cand["column"]],
                    status=LabelStatus.FN,
                    corrected_value=cand["canonical"],
                    note=phone_tag,
                    annotator=ann,
                ))
                cells_written += 1
            for cand in cluster_candidates:
                if cand["already_tagged"]:
                    continue
                batch.append(Label(
                    target=LabelTarget.CLUSTER,
                    target_id=cand["target_id"],
                    source=cand["source"],
                    row=cand["row"],
                    columns=cand["columns"],
                    status=LabelStatus.SKIP,
                    note=phone_tag,
                    annotator=ann,
                ))
                clusters_written += 1
            if batch:
                ctx.store.append_many(batch)

        return {
            "dry_run": dry_run,
            "threshold": threshold,
            "min_phone_rows": min_phone_rows,
            "columns": columns_summary,
            "cell_candidates": cell_candidates,
            "cluster_candidates": cluster_candidates,
            "cells_written": cells_written,
            "clusters_written": clusters_written,
        }

    @app.post("/api/runs/{run_id}/clusters/propagate-status")
    def propagate_cluster_status(
        run_id: str, payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """Propagate a cluster verdict (fp / tp / partial / unsure) to
        every OTHER cluster on the same ``(source, column)`` whose
        anchor cell value matches the seed cell at ``(row, column)``.

        Use case: the LLM flags many rows of one column with the same
        value (e.g. "Dan R. Vacek/Mark G. Elworth" repeated across 30
        rows). The operator clicks one cluster and rules ``fp``: all
        the other clusters carrying the same value should also be
        ``fp``, since the LLM was wrong globally about this value.

        Same-value siblings are OVERRIDDEN (latest-wins): the operator
        clicked propagation precisely because this value's verdict
        should be uniform, so a prior (often contradictory) label is
        replaced rather than preserved. The exceptions:
          - the seed cluster itself (the caller already labelled it);
          - clusters whose anchor cell value differs from the seed;
          - a sibling that already carries the IDENTICAL status from the
            SAME annotator -> no-op (avoids a redundant audit record).

        ``status`` accepts tp / fp / partial / unsure / skip; ``skip``
        spreads the skip verdict across sibling clusters on the column
        (distinct from the row-level skip cascade). ``fn`` is rejected --
        only cells carry fn.

        Required keys: ``source``, ``row``, ``column``, ``status``.
        Optional: ``note``, ``annotator``.
        Response: ``{"matched": int, "propagated": int,
                     "overridden_existing": int, "unchanged": int,
                     "skipped_existing": int (back-compat, always 0),
                     "clusters": [...]}``.
        """
        source = payload.get("source")
        row = payload.get("row")
        column = payload.get("column")
        status_raw = payload.get("status")
        note = (payload.get("note") or "").strip()
        ann = (payload.get("annotator") or annotator).strip() or annotator
        if not source or row is None or not column or not status_raw:
            raise HTTPException(
                400, "source, row, column, and status are required",
            )
        try:
            row = int(row)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"invalid row: {row!r}") from exc
        try:
            status = LabelStatus(str(status_raw).lower())
        except ValueError as exc:
            raise HTTPException(400, f"invalid status: {status_raw!r}") from exc
        # FN never applies to cluster verdicts (only cells carry fn).
        # SKIP is allowed: it spreads the skip verdict to sibling
        # clusters on the column (distinct from the row-level skip
        # cascade).
        if status not in (
            LabelStatus.TP, LabelStatus.FP, LabelStatus.PARTIAL,
            LabelStatus.UNSURE, LabelStatus.SKIP,
        ):
            raise HTTPException(
                400,
                "status must be one of tp/fp/partial/unsure/skip for "
                "cluster propagation",
            )

        ctx = cache.get(run_id)
        if source not in ctx.sources:
            raise HTTPException(404, f"source not found: {source}")
        try:
            frame = ctx.load_frame(source)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        if column not in frame.columns:
            raise HTTPException(
                404, f"column not found in {source}: {column!r}",
            )
        if not (0 <= row < len(frame)):
            raise HTTPException(400, f"row {row} out of range for {source}")

        # Anchor value the operator just verdicted - use the SAME
        # ``_scalar`` coercion the cells endpoints use so equality
        # matches the operator's mental model.
        col_loc = frame.columns.get_loc(column)
        target_value = _scalar(frame.iat[row, col_loc])

        latest = ctx.store.latest_by_target()
        propagated_clusters: List[Dict[str, Any]] = []
        batch: List[Label] = []
        # Retained for back-compat with older UI callers; propagation now
        # OVERRIDES prior verdicts (latest-wins) instead of skipping, so
        # this stays 0.
        skipped_existing = 0
        overridden_existing = 0
        unchanged = 0
        matched = 0
        for c in ctx.clusters:
            if c.get("source") != source:
                continue
            cid = c.get("cluster_id", "")
            if not cid:
                continue
            cr = c.get("row")
            if not isinstance(cr, int) or not (0 <= cr < len(frame)):
                continue
            cols = list(c.get("columns") or [])
            # Restrict to clusters that actually flag this column.
            # Multi-column clusters where ``column`` is one of the
            # flagged cols still qualify.
            if column not in cols:
                continue
            # Skip the seed itself (caller already labelled it).
            if cr == row:
                continue
            # Anchor cell value must match.
            v = _scalar(frame.iat[cr, col_loc])
            if v != target_value:
                continue
            matched += 1
            existing = latest.get(cid)
            has_verdict = (
                existing is not None
                and existing.target == LabelTarget.CLUSTER
            )
            # No-op when the sibling already carries the identical verdict
            # from the same annotator: re-appending would only pollute the
            # append-only audit log with a duplicate record.
            if (
                has_verdict
                and existing.status == status
                and (existing.annotator or "") == ann
            ):
                unchanged += 1
                continue
            # Otherwise OVERRIDE the prior verdict (latest-wins).
            if has_verdict:
                overridden_existing += 1
            batch.append(Label(
                target=LabelTarget.CLUSTER,
                target_id=cid,
                source=source,
                row=cr,
                column=cols[0] if cols else column,
                columns=cols,
                status=status,
                corrected_value=None,
                note=note or f"(auto: same-value as labelled {status.value} cluster)",
                annotator=ann,
            ))
            propagated_clusters.append({
                "cluster_id": cid, "row": cr, "columns": cols,
            })
        # Single fsync for the entire cascade.
        ctx.store.append_many(batch)

        return {
            "source": source,
            "column": column,
            "anchor_row": row,
            "anchor_value": target_value,
            "status": status.value,
            "matched": matched,
            "propagated": len(propagated_clusters),
            "overridden_existing": overridden_existing,
            "unchanged": unchanged,
            "skipped_existing": skipped_existing,
            "clusters": propagated_clusters,
        }

    # ---------------------------------------------------------------
    # Exclusion endpoints. The operator can mark a whole CSV (scope=csv)
    # OR a set of row indices (scope=rows) OR a set of column names
    # (scope=columns) inside one source as too messy to label. The
    # records persist in ``<run_dir>/exclusions.jsonl`` so a later
    # cleaning step can rebuild a sanitised dataset; the labelling UI
    # only renders strikethroughs over the affected rows/cols.
    # Soft-delete (undo) is supported via DELETE /exclusions/{id}, which
    # appends a tombstone record sharing the same id.
    # ---------------------------------------------------------------

