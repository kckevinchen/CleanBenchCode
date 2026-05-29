"""Bulk-labelling / propagation endpoints.

Same-value FN propagation (cell + cluster level) and the trivial-tagger
endpoints (whitespace, phone-format). These spread one operator verdict
across same-value siblings or auto-label mechanical format variants, so
the human only has to rule once.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from .. import phone_format
from .._helpers import _scalar
from ..schema import Label, LabelStatus, LabelTarget, cell_target_id
from ._context import RouterContext


# Cells that carry a trivial-cosmetic anomaly. Used by the
# ``tag-trivial-whitespace`` endpoint to decide which existing labels
# should get a ``TRIVIAL`` note tag. Bundles:
#   1. /^[ ]|[ ]$|[ ]{2,}/   -- edge or 2+ internal ASCII spaces.
#   2. /[\t\u00A0\n\r\u2028\u2029\u200B\u200C\u200D\u00AD\uFEFF]/
#      -- tab, NBSP, line breaks, zero-width chars, soft hyphen, BOM.
#   3. /,[A-Za-z]/           -- missing space after a comma
#                               (e.g. ``Gas,Oil``, ``HOME,LLC``).
#                               Bypasses decimals like ``1,234``.
#   4. / [)\]}]/             -- single space before a closing bracket
#                               (e.g. ``Trust )``).
#   5. /'S(?=\W|$)/          -- capital ``S`` immediately after an
#                               apostrophe at word boundary
#                               (e.g. ``Defender'S Office`` ->
#                               ``Defender's``).
#   6. /\s=\S|\S=\s/         -- ``=`` with whitespace on only one
#                               side (e.g. ``1 lb =16``).
# The ``has-ws`` class in ``frontend/labeling/label.js`` still uses
# only (1)+(2) for the CSV-viewer red-underline; this regex is
# specifically the trivial-tagger rule and is mirrored in
# ``label.js`` as ``_WS_MARKER_RE``. Kept narrower in
# ``experiments/auto_label_whitespace_anomalies.py`` (which creates
# new labels rather than tagging existing ones).
_WHITESPACE_ANOMALY_RE = re.compile(
    r"^ | $|  +"
    r"|[\t\u00A0\n\r\u2028\u2029\u200B\u200C\u200D\u00AD\uFEFF]"
    r"|,[A-Za-z]"
    r"| [)\]}]"
    r"|'S(?=\W|$)"
    r"|\s=\S|\S=\s"
)


def _has_whitespace_anomaly(value: str) -> bool:
    if not value:
        return False
    return _WHITESPACE_ANOMALY_RE.search(value) is not None


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
        identical value, *unless* that cell is already covered by an
        annotator action (cluster-level verdict OR an existing cell
        label) so we never silently overwrite prior decisions.

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

        # Skip cells that already have an annotator decision so we
        # never overwrite prior TPs/FPs/FNs from this or any other
        # annotator. Two cover-states:
        #   (a) a cluster verdict (tp/fp/partial/unsure) flags the
        #       cell - the LLM caught it, the operator already ruled.
        #   (b) an explicit cell label exists at (source, row, col) -
        #       could be a prior FN with a different correction.
        latest = ctx.store.latest_by_target()
        covered: set = set()
        for c in ctx.clusters:
            if c.get("source") != source:
                continue
            r = c.get("row")
            if not isinstance(r, int):
                continue
            cid = c.get("cluster_id", "")
            lab = latest.get(cid)
            # Only a cluster carrying an ACTIVE verdict
            # (tp/fp/partial/unsure) counts as coverage that blocks
            # propagation: the operator has ruled on it, so auto-FN'ing
            # its same-value siblings would clobber that decision. An
            # UNLABELLED cluster means the operator hasn't ruled yet,
            # and a SKIP is a deferred (non-)decision -- in both cases
            # we are free to propagate the FN to the sibling cell (the
            # operator can still verdict the cluster itself later).
            if not (
                lab is not None
                and lab.target == LabelTarget.CLUSTER
                and lab.status in (
                    LabelStatus.TP, LabelStatus.FP,
                    LabelStatus.PARTIAL, LabelStatus.UNSURE,
                )
            ):
                continue
            cluster_cols = list(c.get("columns") or [])
            for col in cluster_cols:
                covered.add((r, col))
            # Detections inside the cluster may flag many more rows
            # than the cluster's anchor row (typo clusters often carry
            # a single anchor row=N but a detection whose
            # ``affected_rows`` lists every row holding the same bad
            # value). Those rows ARE covered by the LLM finding -
            # auto-FN'ing them would double-count what the cluster
            # verdict already settles. Treat each detection's
            # ``affected_rows`` x ``columns`` as covered, falling
            # back to the cluster's columns when the detection omits
            # its own column list.
            for det in (c.get("detections") or ()):
                det_cols = list(det.get("columns") or cluster_cols)
                for ar in (det.get("affected_rows") or ()):
                    if not isinstance(ar, int):
                        continue
                    for col in det_cols:
                        covered.add((ar, col))
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

        Why this exists: the auto-whitespace detector (see
        ``experiments/auto_label_whitespace_anomalies.py``) finds
        edge / double / non-printable whitespace in cells. Such typos
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

