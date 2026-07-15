"""Multi-table (cross-source) inconsistency labelling endpoints.

Backed by ``<run_dir>/multi_table_labels.jsonl``. Each finding is keyed
by ``violation_id`` (shared across the 1+ rows the LLM detector emitted
for one inconsistency); a single human verdict per finding is the source
of truth, with full append-only history. Inputs are read from
``multi_table_violations.jsonl``, ``multi_table_groups.jsonl``, and
``multi_table_verifications.jsonl`` written by the relation-discovery
pipeline.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query

from .._helpers import _scalar
from ..multi_table_label_store import MultiTableLabelStore
from ..schema import MultiTableLabel, MultiTableLabelIn
from ._context import RouterContext

if TYPE_CHECKING:
    from ..app import RunContext

logger = logging.getLogger(__name__)


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache
    annotator = router_ctx.annotator

    _mt_label_stores: Dict[str, MultiTableLabelStore] = {}
    _mt_label_lock = threading.Lock()

    def _mt_label_store(run_id: str) -> MultiTableLabelStore:
        rd = cache._safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        with _mt_label_lock:
            store = _mt_label_stores.get(run_id)
            if store is None:
                store = MultiTableLabelStore(rd / "multi_table_labels.jsonl")
                _mt_label_stores[run_id] = store
            return store

    def _read_multi_table_violations(
        run_dir: Path,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """Load ``multi_table_violations.jsonl`` and group per finding.

        Returns ``(by_id, rows_by_id)`` where ``by_id`` is a dict from
        ``violation_id`` to the *first* record seen (used for finding-
        level metadata: kind, workbook, group_id, rationale) and
        ``rows_by_id`` carries every record sharing that id (one per
        affected (sheet, row) location, ordered as they appear on disk).
        """
        path = run_dir / "multi_table_violations.jsonl"
        by_id: Dict[str, Dict[str, Any]] = {}
        rows_by_id: Dict[str, List[Dict[str, Any]]] = {}
        if not path.exists():
            return by_id, rows_by_id
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                vid = rec.get("violation_id")
                if not vid:
                    continue
                rows_by_id.setdefault(vid, []).append(rec)
                by_id.setdefault(vid, rec)
        return by_id, rows_by_id

    def _read_multi_table_verifications(
        run_dir: Path,
    ) -> Dict[str, Dict[str, Any]]:
        """Load LLM verifier verdicts keyed by ``violation_id``. Latest
        record per id wins (the writer already dedupes, but the read
        path is defensive)."""
        path = run_dir / "multi_table_verifications.jsonl"
        out: Dict[str, Dict[str, Any]] = {}
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                vid = rec.get("violation_id")
                if vid:
                    out[vid] = rec
        return out

    def _read_multi_table_groups(run_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Load resolved-group records keyed by ``(workbook, group_id)``.

        Each line in ``multi_table_groups.jsonl`` is one workbook's full
        resolver output (sheets + groups + column_mapping + primary_key).
        We flatten it into a per-group lookup so the detail endpoint can
        attach the group's column mapping + PK to a violation without
        re-scanning the file each call.
        """
        path = run_dir / "multi_table_groups.jsonl"
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                workbook = rec.get("workbook") or ""
                for g in rec.get("groups") or []:
                    gid = g.get("group_id") or ""
                    out[(workbook, gid)] = {**g, "workbook": workbook}
        return out

    @app.get("/api/runs/{run_id}/multi-table-violations")
    def list_multi_table_violations(
        run_id: str,
        workbook: Optional[str] = None,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(500, ge=1, le=5000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        """Paginated index of multi-table findings.

        One entry per ``violation_id`` (so multi-row findings collapse
        to a single queue item). Each entry carries the LLM verifier's
        verdict + the human verdict (if any) so the UI can drive its
        filter dropdowns and queue rendering off this single response.
        """
        ctx = cache.get(run_id)
        by_id, rows_by_id = _read_multi_table_violations(ctx.run_dir)
        verifications = _read_multi_table_verifications(ctx.run_dir)
        store = _mt_label_store(run_id)
        latest = store.latest_by_violation()

        items: List[Dict[str, Any]] = []
        kind_set: Dict[str, int] = {}
        wb_set: Dict[str, int] = {}
        for vid, rec in by_id.items():
            wb_val = rec.get("workbook") or ""
            kn = rec.get("kind") or ""
            wb_set[wb_val] = wb_set.get(wb_val, 0) + 1
            kind_set[kn] = kind_set.get(kn, 0) + 1
            if workbook and wb_val != workbook:
                continue
            if kind and kn != kind:
                continue
            human = latest.get(vid)
            human_status = human.status.value if human else "unlabeled"
            if status and status != human_status:
                continue
            ver = verifications.get(vid) or {}
            sheets = sorted({
                str(r.get("sheet") or "") for r in rows_by_id.get(vid, [])
            })
            items.append({
                "violation_id": vid,
                "workbook": wb_val,
                "group_id": rec.get("group_id") or "",
                "group_kind": rec.get("group_kind") or "",
                "kind": kn,
                "sheets": sheets,
                "n_rows": len(rows_by_id.get(vid, [])),
                "columns": list(rec.get("columns") or []),
                "llm_verdict": ver.get("verdict"),
                "llm_confidence": ver.get("confidence"),
                "label_status": human_status,
                "label_ts": human.ts if human else None,
                "label_annotator": human.annotator if human else None,
                "note": human.note if human else "",
            })

        # Stable order: workbook, group, kind, vid (so the queue is
        # deterministic across reloads).
        items.sort(key=lambda x: (
            x["workbook"], x["group_id"], x["kind"], x["violation_id"],
        ))
        total = len(items)
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": items[offset: offset + limit],
            "facets": {
                "workbooks": sorted(wb_set.keys()),
                "kinds": sorted(kind_set.keys()),
            },
        }

    @app.get("/api/runs/{run_id}/multi-table-groups")
    def list_multi_table_groups(
        run_id: str,
        include_unconfirmed: bool = False,
    ) -> Dict[str, Any]:
        """Per-workbook view of the DISCOVERED RELATIONS.

        Surfaces the resolver's ``multi_table_groups.jsonl`` so the
        labeller can see, for each file, which CSVs we believe are
        *unionable* (union groups = replica sheets sharing a schema/key)
        and which are *joinable* (join groups = a parent->child foreign
        key), independent of whether any row-level finding survived the
        scope filter / guard. Each group carries a count of findings
        still attached to it so a join with zero findings is visible.

        Only **confirmed** relations are returned by default (what we
        believe); discovery proposes many unconfirmed candidates that
        would otherwise swamp the view. Pass ``include_unconfirmed=true``
        to see candidates too. ``totals.unconfirmed`` always reports how
        many were filtered out.
        """
        ctx = cache.get(run_id)
        groups = _read_multi_table_groups(ctx.run_dir)
        by_id, _rows = _read_multi_table_violations(ctx.run_dir)

        n_find: Dict[Tuple[str, str], int] = {}
        for _vid, rec in by_id.items():
            key = (rec.get("workbook") or "", rec.get("group_id") or "")
            n_find[key] = n_find.get(key, 0) + 1

        per_wb: Dict[str, List[Dict[str, Any]]] = {}
        n_union = n_join = n_unconfirmed = 0
        for (wb, gid), g in groups.items():
            confirmed = bool(g.get("confirmed", True))
            if not confirmed:
                n_unconfirmed += 1
                if not include_unconfirmed:
                    continue
            kind = g.get("kind") or ""
            join = g.get("join") or None
            join_out = None
            if isinstance(join, dict):
                join_out = {
                    "parent_sheet": join.get("parent_sheet"),
                    "parent_columns": list(join.get("parent_columns") or []),
                    "child_sheet": join.get("child_sheet"),
                    "child_columns": list(join.get("child_columns") or []),
                }
            per_wb.setdefault(wb, []).append({
                "group_id": gid,
                "kind": kind,
                "confirmed": confirmed,
                "sheets": list(g.get("sheets") or []),
                "primary_key": list(g.get("primary_key") or []),
                "join": join_out,
                "rationale": g.get("rationale") or "",
                "n_findings": n_find.get((wb, gid), 0),
            })
            if kind == "union":
                n_union += 1
            elif kind == "join":
                n_join += 1

        workbooks: List[Dict[str, Any]] = []
        for wb in sorted(per_wb):
            gs = per_wb[wb]
            gs.sort(key=lambda e: (e["kind"], e["group_id"]))
            sheets = sorted({s for e in gs for s in e["sheets"]})
            workbooks.append({
                "workbook": wb,
                "sheets": sheets,
                "groups": gs,
            })

        return {
            "workbooks": workbooks,
            "totals": {
                "workbooks": len(workbooks),
                "union": n_union,
                "join": n_join,
                "groups": n_union + n_join,
                "unconfirmed": n_unconfirmed,
            },
        }

    @app.get("/api/runs/{run_id}/multi-table-violations/{violation_id}")
    def get_multi_table_violation(
        run_id: str, violation_id: str,
    ) -> Dict[str, Any]:
        """Detail endpoint: every linked row + group context + verdict.

        Returns the complete row contents (every column the CSV has,
        not just the flagged ones) so the labeller can audit values
        outside the LLM's narrow snapshot. ``ground_columns`` mirrors
        the union of every row's CSV columns so the frontend can
        align panels even when the two sheets have different schemas.
        """
        ctx = cache.get(run_id)
        by_id, rows_by_id = _read_multi_table_violations(ctx.run_dir)
        rec = by_id.get(violation_id)
        if rec is None:
            raise HTTPException(404, f"violation not found: {violation_id}")
        rows = rows_by_id.get(violation_id, [])

        verifications = _read_multi_table_verifications(ctx.run_dir)
        groups = _read_multi_table_groups(ctx.run_dir)
        group = groups.get(
            (rec.get("workbook") or "", rec.get("group_id") or ""),
        )

        # Resolve full row contents from the CSVs themselves so the
        # labeller can see every column - not just the LLM's narrow
        # snapshot. Falls back to the snapshot when the CSV cannot be
        # loaded (deleted dataset, mismatched workbook, ...).
        full_rows: List[Dict[str, Any]] = []
        for r in rows:
            sheet = r.get("sheet") or ""
            row_idx = r.get("row")
            full: Dict[str, Any] = {}
            cols: List[str] = []
            try:
                if sheet and isinstance(row_idx, int):
                    frame = ctx.load_frame(sheet)
                    cols = list(frame.columns)
                    if 0 <= row_idx < len(frame):
                        for c in cols:
                            full[c] = _scalar(frame.iat[row_idx, frame.columns.get_loc(c)])
            except FileNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001 - defensive
                logger.debug(
                    "failed to load row %s:%s for violation %s: %s",
                    sheet, row_idx, violation_id, exc,
                )
            # ``MultiTableViolation.values`` is the LLM-emitted
            # snapshot of the flagged cells. Two shapes are written by
            # the two strategies:
            #   - compiler  : flat ``{<canonical>: <value>, "primary_key": [...]}``
            #   - semantic  : ``{"evidence": ..., "kind_detail": ...,
            #                    "row_snapshot": {<canonical>: <value>}}``
            # The frontend expects a flat ``{column: value}`` map so it
            # can render the cell when the full CSV row can't be loaded
            # (deleted dataset, relations runs without ``csv_paths``,
            # header_synonym findings with row=-1, ...). Normalise both
            # shapes here, prefer the nested ``row_snapshot`` when
            # present, and strip metadata keys (``primary_key`` is
            # exposed under ``group``; ``evidence`` / ``kind_detail``
            # are diagnostic only and not column values).
            raw_values = dict(r.get("values") or {})
            nested = raw_values.get("row_snapshot")
            if isinstance(nested, dict) and nested:
                snap_raw = dict(nested)
            else:
                snap_raw = {
                    k: v for k, v in raw_values.items()
                    if k not in {"evidence", "kind_detail",
                                 "row_snapshot", "primary_key"}
                }
            full_rows.append({
                "sheet": sheet,
                "row": row_idx,
                "columns": cols,
                "values": full,
                "snapshot": snap_raw,
                "flagged_columns": list(r.get("columns") or []),
                "linked": list(r.get("linked") or []),
            })

        # Union of all columns across rows so the UI can stack panels.
        # Mapping (canonical -> per-sheet raw column name) lets the
        # frontend align the comparison even when the two sheets use
        # different headers for the same logical attribute.
        column_mapping: Dict[str, Dict[str, str]] = {}
        primary_key: List[str] = []
        if group is not None:
            for canonical, entries in (group.get("column_mapping") or {}).items():
                column_mapping[canonical] = {
                    e.get("sheet") or "": e.get("column") or ""
                    for e in entries or []
                }
            primary_key = list(group.get("primary_key") or [])

        store = _mt_label_store(run_id)
        human = store.latest_for(violation_id)
        history = store.history(violation_id)
        ver = verifications.get(violation_id) or {}

        return {
            "violation_id": violation_id,
            "workbook": rec.get("workbook") or "",
            "group_id": rec.get("group_id") or "",
            "group_kind": rec.get("group_kind") or "",
            "kind": rec.get("kind") or "",
            "rationale": rec.get("rationale") or "",
            "rows": full_rows,
            "group": {
                "primary_key": primary_key,
                "column_mapping": column_mapping,
                "rationale": (group or {}).get("rationale") or "",
                "sheets": list((group or {}).get("sheets") or []),
            } if group else None,
            "llm_verdict": {
                "verdict": ver.get("verdict"),
                "confidence": ver.get("confidence"),
                "rationale": ver.get("rationale"),
            } if ver else None,
            "label": human.model_dump(mode="json") if human else None,
            "history": [h.model_dump(mode="json") for h in history],
        }

    @app.post("/api/runs/{run_id}/multi-table-labels")
    def post_multi_table_label(
        run_id: str, payload: MultiTableLabelIn,
    ) -> Dict[str, Any]:
        """Record a human verdict on a multi-table finding.

        The ``violation_id`` MUST exist in the run's
        ``multi_table_violations.jsonl`` - the server validates against
        the file rather than blindly accepting any id, so a stale or
        forged id can't poison the store.
        """
        ctx = cache.get(run_id)
        by_id, _ = _read_multi_table_violations(ctx.run_dir)
        if payload.violation_id not in by_id:
            raise HTTPException(
                404, f"violation not found: {payload.violation_id}",
            )

        store = _mt_label_store(run_id)
        # Echo metadata from the violation record into the label so
        # the JSONL is self-describing (no need to cross-reference
        # violations.jsonl when consuming labels downstream).
        rec = by_id[payload.violation_id]
        record = MultiTableLabel(
            violation_id=payload.violation_id,
            workbook=payload.workbook or rec.get("workbook") or "",
            group_id=payload.group_id or rec.get("group_id") or "",
            kind=payload.kind or rec.get("kind") or "",
            status=payload.status,
            note=payload.note or "",
            annotator=(
                (payload.annotator or annotator).strip() or annotator
            ),
        )
        saved = store.append(record)
        return {
            "label": saved.model_dump(mode="json"),
            "history": [
                h.model_dump(mode="json")
                for h in store.history(payload.violation_id)
            ],
        }

    @app.get("/api/runs/{run_id}/multi-table-labels")
    def list_multi_table_labels(run_id: str) -> Dict[str, Any]:
        """All multi-table labels written for this run (full history +
        latest-per-violation roll-up)."""
        store = _mt_label_store(run_id)
        latest = {
            vid: rec.model_dump(mode="json")
            for vid, rec in store.latest_by_violation().items()
        }
        records = [r.model_dump(mode="json") for r in store.all()]
        return {"records": records, "latest": latest}

