"""Per-cell note override endpoints under ``/api/runs/{run_id}/cell-notes``.

A cell note overrides the enclosing cluster's ``note`` at export time
for a single ``(source, row, column)`` cell. Stores are append-only with
empty-note tombstones (equivalent to "no override"), so the GET filters
those out and the POST records them.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from ..schema import CellNote, CellNoteIn
from ._context import RouterContext


def register(app: FastAPI, ctx: RouterContext) -> None:
    cache = ctx.cache
    annotator = ctx.annotator

    @app.get("/api/runs/{run_id}/cell-notes")
    def list_cell_notes(run_id: str) -> Dict[str, Any]:
        """Return every active per-cell note override.

        Empty-note tombstones are filtered out (they're equivalent to
        "no override"). Each entry is keyed by ``(source, row, column)``
        so the UI can hydrate the textarea for a freshly-opened
        cluster without an extra round-trip per cell.
        """
        run_ctx = cache.get(run_id)
        return {
            "overrides": [
                {
                    "source": src, "row": row, "column": col, "note": note,
                }
                for (src, row, col), note in (
                    run_ctx.cell_notes_store.active_overrides().items()
                )
            ],
        }

    @app.post("/api/runs/{run_id}/cell-notes")
    def upsert_cell_note(
        run_id: str, payload: CellNoteIn,
    ) -> CellNote:
        """Set / clear the per-cell note override for one cell.

        The override supersedes the enclosing cluster's ``note`` at
        export time for the single ``(source, row, column)`` cell.
        An empty (or whitespace-only) ``note`` is recorded as a
        tombstone, which makes the exporter fall back to the cluster
        note again. Validates that the source exists in the run; the
        ``row`` / ``column`` are accepted as opaque keys so callers
        can override notes on cells the run has no cluster for too.
        """
        run_ctx = cache.get(run_id)
        if payload.source not in run_ctx.sources:
            raise HTTPException(404, f"source not found: {payload.source}")
        return run_ctx.cell_notes_store.upsert(
            payload,
            annotator=(payload.annotator or annotator).strip() or annotator,
        )
