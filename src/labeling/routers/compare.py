"""Run-comparison endpoints under ``/api/compare``.

Reads cached cell-level diffs computed by
``experiments.compare_runs.diff_runs`` and exposes them in three
shapes: a high-level summary, a paginated cell list, and a single-cell
detail drawer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query

from ._context import RouterContext


def register(app: FastAPI, ctx: RouterContext) -> None:
    @app.get("/api/compare")
    def compare_summary(
        run_a: str = Query(..., description="Run id A"),
        run_b: str = Query(..., description="Run id B"),
    ) -> Dict[str, Any]:
        """Per-bucket overlap table + source alignment for two runs.

        Does not return the per-cell diff list (that endpoint is
        paginated separately) so this stays cheap to call from the
        UI's run-picker preview.
        """
        if run_a == run_b:
            raise HTTPException(400, "run_a and run_b must differ")
        diff = ctx.get_diff(run_a, run_b)
        return {
            "run_a": run_a,
            "run_b": run_b,
            "n_a_issues": diff["n_a_issues"],
            "n_b_issues": diff["n_b_issues"],
            "n_diff_cells": diff["n_diff_cells"],
            "sources": diff["sources"],
            "by_bucket": diff["by_bucket"],
        }

    @app.get("/api/compare/cells")
    def compare_cells(
        run_a: str = Query(...),
        run_b: str = Query(...),
        side: Optional[str] = Query(None, description="a_only | b_only | both"),
        bucket: Optional[str] = Query(None),
        source: Optional[str] = Query(None),
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
    ) -> Dict[str, Any]:
        """Paginated diff cells, optionally filtered by side/bucket/source."""
        if run_a == run_b:
            raise HTTPException(400, "run_a and run_b must differ")
        diff = ctx.get_diff(run_a, run_b)
        cells = diff["diff_cells"]
        if side:
            if side not in ("a_only", "b_only", "both"):
                raise HTTPException(400, f"invalid side: {side!r}")
            cells = [c for c in cells if c["side"] == side]
        if bucket:
            cells = [c for c in cells if bucket in (c.get("buckets") or [c.get("bucket")])]
        if source:
            cells = [c for c in cells if c["source"] == source]
        total = len(cells)
        page = cells[offset:offset + limit]
        return {
            "run_a": run_a,
            "run_b": run_b,
            "total": total,
            "offset": offset,
            "limit": limit,
            "cells": page,
        }

    @app.get("/api/compare/cell")
    def compare_cell(
        run_a: str = Query(...),
        run_b: str = Query(...),
        source: str = Query(...),
        row: int = Query(..., ge=0),
        column: str = Query(...),
    ) -> Dict[str, Any]:
        """All issues from both runs for a single cell (drawer detail)."""
        if run_a == run_b:
            raise HTTPException(400, "run_a and run_b must differ")
        from experiments.compare_runs import diff_cell_detail

        a_dir = ctx.run_dir_or_404(run_a)
        b_dir = ctx.run_dir_or_404(run_b)
        try:
            return diff_cell_detail(a_dir, b_dir,
                                    source=source, row=row, column=column)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
