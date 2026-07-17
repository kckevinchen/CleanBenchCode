"""Per-row verdict override endpoints under ``/api/runs/{run_id}/row-verdicts``.

A row verdict overrides the enclosing cluster's verdict for a single
``(source, row, column)`` cell -- used to label one row of an
``fd_violation`` conflict group independently of the cluster's verdict
(which otherwise fans out identically to every conflicting row). Stores
are append-only with ``unlabeled`` tombstones (= "revert to the cluster
default"), so the GET filters those out and the POST records them.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from ..schema import RowVerdict, RowVerdictIn
from ._context import RouterContext


def register(app: FastAPI, ctx: RouterContext) -> None:
    cache = ctx.cache
    annotator = ctx.annotator

    @app.get("/api/runs/{run_id}/row-verdicts")
    def list_row_verdicts(run_id: str) -> Dict[str, Any]:
        """Return every active per-row verdict override.

        ``unlabeled`` tombstones are filtered out (they're equivalent to
        "no override"). Each entry is keyed by ``(source, row, column)``
        so the UI can hydrate a conflict group's per-row verdicts without
        an extra round-trip per cell.
        """
        run_ctx = cache.get(run_id)
        return {
            "overrides": [
                {
                    "source": src,
                    "row": row,
                    "column": col,
                    "status": rec.status.value,
                    "corrected_value": rec.corrected_value,
                    "note": rec.note,
                    "annotator": rec.annotator,
                }
                for (src, row, col), rec in (
                    run_ctx.row_verdict_store.active_overrides().items()
                )
            ],
        }

    @app.post("/api/runs/{run_id}/row-verdicts")
    def upsert_row_verdict(
        run_id: str, payload: RowVerdictIn,
    ) -> RowVerdict:
        """Set / clear the per-row verdict override for one cell.

        The override supersedes the enclosing cluster's fanned-out
        verdict at export time and in the viewer for the single
        ``(source, row, column)`` cell. A ``status`` of ``unlabeled`` is
        recorded as a tombstone, reverting the cell to the cluster
        default. Validates that the source exists in the run; ``row`` /
        ``column`` are accepted as opaque keys so callers can override
        peer rows the run has no dedicated cluster for too.
        """
        run_ctx = cache.get(run_id)
        if payload.source not in run_ctx.sources:
            raise HTTPException(404, f"source not found: {payload.source}")
        return run_ctx.row_verdict_store.upsert(
            payload,
            annotator=(payload.annotator or annotator).strip() or annotator,
        )
