"""CSV source + per-cell-state endpoints under ``/api/runs/{run_id}/sources/*``.

Serves a source CSV slice, the derived per-cell label state (the
``state`` each cell paints in the CSV viewer, with TP-cluster shadowing
of FN cells), and the false-negative cell listing.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query

from .._helpers import _scalar, _validate_source_name
from ..schema import LabelStatus, LabelTarget, cell_target_id
from ._context import RouterContext


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache
    annotator = router_ctx.annotator

    @app.get("/api/runs/{run_id}/sources/{name}")
    def get_source(
        run_id: str,
        name: str,
        start: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=5000),
    ) -> Dict[str, Any]:
        _validate_source_name(name)
        ctx = cache.get(run_id)
        try:
            frame = ctx.load_frame(name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

        total_rows = len(frame)
        end = min(start + limit, total_rows)
        slice_ = frame.iloc[start:end]
        # Sibling CSVs in the same workbook (parent dir). Empty unless
        # the run targets a multi-table corpus.
        wb = ctx.source_workbook.get(name, "")
        siblings: List[str] = []
        if wb:
            siblings = sorted(
                n for n, w in ctx.source_workbook.items()
                if w == wb and n != name
            )
        has_xlsx = ctx.workbook_xlsx_path(wb) is not None if wb else False
        return {
            "source": name,
            "workbook": wb,
            "siblings": siblings,
            "has_xlsx": has_xlsx,
            "total_rows": total_rows,
            "start": start,
            "end": end,
            "columns": list(frame.columns),
            "rows": [
                {"row": int(idx), "values": {k: _scalar(v) for k, v in rec.items()}}
                for idx, rec in zip(slice_.index, slice_.to_dict(orient="records"))
            ],
        }

    @app.get("/api/runs/{run_id}/sources/{name}/cells")
    def get_cell_state(run_id: str, name: str) -> Dict[str, Any]:
        """Per-cell state for one CSV, fused from cluster verdicts and
        ad-hoc cell labels.

        The frontend uses this to colour every cell in the CSV viewer
        (so the operator can see which cells are TP/FP/FN at a glance)
        and to decide whether to offer the "Mark as FN" action: a cell
        already covered by a TP cluster cannot be re-labelled FN.

        Precedence inside a single cell, highest wins:
          tp > partial > fp > fn > unsure
        Cluster labels (tp/fp/partial/unsure) win over a cell-level FN
        flag; this is how a TP verdict on the cluster automatically
        supersedes any earlier FN the operator placed on the same cell.
        """
        _validate_source_name(name)
        ctx = cache.get(run_id)
        latest = ctx.store.latest_by_target()
        # Cell -> {state, cluster_ids:[], note, error_types:[], annotators:[]}
        cells: Dict[str, Dict[str, Any]] = {}

        rank = {"tp": 5, "partial": 4, "fp": 3, "fn": 2, "unsure": 1}

        def _bump(key: str, state: str, **extra: Any) -> None:
            cur = cells.get(key)
            if cur is None or rank.get(state, 0) > rank.get(cur["state"], 0):
                cells[key] = {
                    "state": state,
                    "cluster_ids": list(extra.get("cluster_ids") or []),
                    "error_types": list(extra.get("error_types") or []),
                    "note": extra.get("note", ""),
                    "annotator": extra.get("annotator", ""),
                    "label_source": extra.get("label_source", "cluster"),
                    "corrected_value": extra.get("corrected_value"),
                }
            else:
                # Lower-rank states still attach metadata for context
                # (e.g. an FN annotator under a TP cluster cell).
                for cid in (extra.get("cluster_ids") or []):
                    if cid not in cur["cluster_ids"]:
                        cur["cluster_ids"].append(cid)

        # Cluster-derived states.
        for c in ctx.clusters:
            if c.get("source") != name:
                continue
            row = c.get("row")
            if not isinstance(row, int):
                continue
            cid = c.get("cluster_id", "")
            label = latest.get(cid)
            if label is None or label.target != LabelTarget.CLUSTER:
                continue
            st = label.status.value
            if st in ("skip",):
                continue
            for col in (c.get("columns") or []):
                key = f"{row}\x00{col}"
                _bump(
                    key, st,
                    cluster_ids=[cid],
                    error_types=c.get("error_types") or [],
                    note=label.note,
                    annotator=label.annotator,
                    label_source="cluster",
                    corrected_value=label.corrected_value,
                )

        # Cell-level FN labels.
        for tid, lab in latest.items():
            if lab.target != LabelTarget.CELL or lab.source != name:
                continue
            if lab.row is None or not lab.column:
                continue
            if lab.status != LabelStatus.FN:
                # Cell labels only carry FN today; ignore others.
                continue
            key = f"{int(lab.row)}\x00{lab.column}"
            _bump(
                key, "fn",
                error_types=[],
                note=lab.note,
                annotator=lab.annotator,
                label_source="cell",
                corrected_value=lab.corrected_value,
            )

        # Materialise.
        out: List[Dict[str, Any]] = []
        shadowed = 0
        for key, payload in cells.items():
            row_str, col = key.split("\x00", 1)
            row = int(row_str)
            entry = {"row": row, "column": col, **payload}
            # Detect double-labelling: cell has both an FN cell-label
            # AND a TP/PARTIAL cluster-label. Surface this so the UI
            # can show a "superseded by cluster X" hint.
            cell_lab = latest.get(cell_target_id(name, row, col))
            if cell_lab and cell_lab.status == LabelStatus.FN and entry["state"] in ("tp", "partial"):
                entry["superseded_fn_by"] = entry["cluster_ids"][:1]
                shadowed += 1
            out.append(entry)
        return {
            "source": name,
            "cells": out,
            "shadowed_fn_count": shadowed,
        }

    @app.get("/api/runs/{run_id}/fn-cells")
    def list_fn_cells(run_id: str) -> Dict[str, Any]:
        """Return every active FN cell label in the run.

        Used by the label-queue frontend to surface operator-marked
        false negatives alongside cluster findings (so the operator
        can revisit / clear them without scrolling the CSV grid). A
        cell is ``shadowed`` when a TP / PARTIAL cluster verdict
        covers the same (source, row, column); the queue drops
        shadowed entries because the cluster verdict supersedes the
        FN claim.
        """
        ctx = cache.get(run_id)
        latest = ctx.store.latest_by_target()

        # LLM cluster coverage indexed by (source, row, column). An
        # FN cell label that lands on a cell the LLM ALREADY flagged
        # is duplicative -- the cluster will surface in the queue on
        # its own, so the FN entry would render as a second row for
        # the same cell. We tag those FNs as ``shadowed`` so the
        # frontend (which already drops shadowed FNs) collapses the
        # pair down to the single cluster entry the operator can
        # verdict directly.
        cluster_cover: Dict[tuple, str] = {}
        for c in ctx.clusters:
            src = c.get("source")
            row = c.get("row")
            if not src or not isinstance(row, int):
                continue
            cid = c.get("cluster_id", "")
            lab = latest.get(cid)
            # "any cluster" coverage: a labelled cluster's verdict
            # is more useful to surface than its mere presence, so
            # record the verdict when present; an unlabelled cluster
            # still counts as coverage (``unlabeled`` sentinel).
            if lab is not None and lab.target == LabelTarget.CLUSTER:
                st = lab.status.value
                if st == "skip":
                    continue
            else:
                st = "unlabeled"
            for col in (c.get("columns") or []):
                key = (src, int(row), col)
                rank = {
                    "tp": 5, "partial": 4, "fp": 3,
                    "unsure": 2, "unlabeled": 1,
                }
                cur = cluster_cover.get(key)
                if cur is None or rank.get(st, 0) > rank.get(cur, 0):
                    cluster_cover[key] = st

        items: List[Dict[str, Any]] = []
        # Statuses we surface in the FN list: the original FN claim
        # plus the verdicts the operator can attach to that claim via
        # the FN detail panel (tp/partial/unsure/skip). FP and
        # UNLABELED tombstone the cell and never appear here
        # (``latest_by_target`` already drops unlabeled).
        _fn_visible_statuses = {
            LabelStatus.FN, LabelStatus.TP, LabelStatus.PARTIAL,
            LabelStatus.UNSURE, LabelStatus.SKIP,
        }
        for tid, lab in latest.items():
            if lab.target != LabelTarget.CELL:
                continue
            if lab.status not in _fn_visible_statuses:
                continue
            if lab.row is None or not lab.column or not lab.source:
                continue
            key = (lab.source, int(lab.row), lab.column)
            shadowing = cluster_cover.get(key)
            # Any LLM cluster on the same cell shadows the FN: the
            # queue would otherwise show two rows (the LLM cluster +
            # the operator's FN claim) for the same (row, column).
            shadowed = shadowing is not None
            items.append({
                "target_id": tid,
                "source": lab.source,
                "row": int(lab.row),
                "column": lab.column,
                "status": lab.status.value,
                "corrected_value": lab.corrected_value,
                "note": lab.note or "",
                "annotator": lab.annotator or "",
                "ts": (
                    lab.ts.isoformat() if hasattr(lab.ts, "isoformat") else lab.ts
                ),
                "shadowed": shadowed,
                "shadowed_by": shadowing if shadowed else None,
            })
        return {"items": items}

