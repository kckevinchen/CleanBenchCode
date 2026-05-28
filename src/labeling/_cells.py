"""Benchmark-cell helpers shared by the export and labeling-effort endpoints.

Cell-value resolution, exclusion filtering, and cache-only inconsistency
materialisation. All take a per-run ``RunContext`` and are import-safe
from both ``app.py`` and the routers (no circular import: ``RunContext``
is referenced only as a string annotation under ``TYPE_CHECKING``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from ._helpers import _scalar
from .inconsistency_export import materialize_inconsistency_benchmark
from .schema import ExclusionScope

if TYPE_CHECKING:
    from .app import RunContext

logger = logging.getLogger(__name__)


def _load_drafts_for_column(
    run_dir: Path, source: str, column: str,
) -> List[Dict[str, Any]]:
    """Return the latest LLM ``draft_clusters`` for a (source, column).

    Reads ``<run_dir>/column_cluster_drafts.jsonl`` and applies
    latest-wins (the file is append-only, so a later record for the same
    (source, column) supersedes an earlier one). Returns an empty list
    when the run has no draft file or no draft for this pair. Each draft
    dict carries ``{values, canonical, agreement, kind}`` -- the shape
    persisted by ``experiments.run_column_clustering``.

    Shared by the column-cluster router (surface drafts as queue items)
    and the core label endpoint (lazily promote a draft to a real record
    on the operator's first verdict).
    """
    drafts_path = run_dir / "column_cluster_drafts.jsonl"
    if not drafts_path.exists():
        return []
    found: Optional[List[Dict[str, Any]]] = None
    try:
        with drafts_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if rec.get("source") == source and rec.get("column") == column:
                    found = rec.get("draft_clusters") or []
    except OSError as exc:
        logger.warning("_load_drafts_for_column: read failed: %s", exc)
        return []
    return found or []


def _make_value_resolver(ctx: "RunContext"):
    """Build a ``(source, row, column) -> str | None`` resolver
    backed by the run's source CSVs.

    Used by the export endpoints to populate ``original_value``
    on every benchmark row, even on cells whose enclosing cluster
    has a sparse / missing ``row_values`` snapshot (same-value
    propagation, FN-only cells, etc.). The resolver is permissive:
    any failure to read the cell (missing source, out-of-range
    row, unknown column, normalize-pass crash) returns ``None``
    so the materialiser can fall back gracefully without ever
    raising on the export path.
    """
    def _resolve(source: str, row: int, column: str) -> Optional[str]:
        try:
            if source not in ctx.sources:
                return None
            frame = ctx.load_frame(source)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if column not in frame.columns:
            return None
        try:
            row_i = int(row)
        except (TypeError, ValueError):
            return None
        if not (0 <= row_i < len(frame)):
            return None
        try:
            return _scalar(frame.iat[row_i, frame.columns.get_loc(column)])
        except (IndexError, KeyError):
            return None
    return _resolve

def _build_exclusion_lookup(
    ctx: "RunContext",
) -> tuple[set[str], Dict[str, set[int]], Dict[str, set[str]]]:
    """Reduce ``ctx.exclusions_store.active()`` to per-scope lookups.

    Returns ``(excluded_files, rows_by_source, cols_by_source)``
    suitable for fast membership checks. Tombstoned exclusions
    are already filtered out by ``ExclusionStore.active``.
    Mirrors the bucketing used by :func:`export_sources` so
    sources_<run>.zip and the benchmark exports stay in lockstep.
    """
    actives = ctx.exclusions_store.active()
    excluded_files: set[str] = set()
    per_source_rows: Dict[str, set[int]] = {}
    per_source_cols: Dict[str, set[str]] = {}
    for e in actives:
        if e.scope == ExclusionScope.CSV:
            excluded_files.add(e.source)
        elif e.scope == ExclusionScope.ROWS:
            bucket = per_source_rows.setdefault(e.source, set())
            for r in e.rows or []:
                try:
                    bucket.add(int(r))
                except (TypeError, ValueError):
                    continue
        elif e.scope == ExclusionScope.COLUMNS:
            bucket = per_source_cols.setdefault(e.source, set())
            for c in e.columns or []:
                if c is not None:
                    bucket.add(str(c))
    return excluded_files, per_source_rows, per_source_cols

def _apply_exclusions_to_cells(
    ctx: "RunContext", cells: Iterable[Any],
) -> List[Any]:
    """Drop benchmark cells that point at excluded files / rows /
    columns so the exported JSONL aligns with sources_<run>.zip."""
    excluded_files, per_source_rows, per_source_cols = (
        _build_exclusion_lookup(ctx)
    )
    if not (excluded_files or per_source_rows or per_source_cols):
        return list(cells)
    kept: List[Any] = []
    for cell in cells:
        source = getattr(cell, "source", "") or ""
        if source in excluded_files:
            continue
        try:
            row_i = int(getattr(cell, "row", 0))
        except (TypeError, ValueError):
            row_i = None
        if row_i is not None and row_i in per_source_rows.get(source, ()):
            continue
        column = getattr(cell, "column", "") or ""
        if column and column in per_source_cols.get(source, ()):
            continue
        kept.append(cell)
    return kept

def _materialize_inconsistency_cells_for_export(
    ctx: "RunContext",
) -> List[Any]:
    """Cache-only inconsistency benchmark cells for the live
    export endpoints.

    Same materialiser the offline
    ``experiments.export_inconsistency_benchmark_with_llm`` CLI
    uses, but with ``dry_run=True`` and ``llm_client=None`` so
    the request handler never spends LLM dollars: every prompt
    is looked up in ``inconsistency_llm_cache.jsonl`` and any
    groups without a cache hit fall back to either the operator's
    saved ``value_corrections`` (authoritative) or the cluster-
    rescue null-emit path. Returns ``[]`` if the run has no
    inconsistency artifacts (no ``column_cluster_drafts.jsonl``
    or no TP/unsure column-cluster verdicts).

    Failures are swallowed and logged: a partial export (typo
    cells only) is strictly better than failing the download.
    """
    try:
        return materialize_inconsistency_benchmark(
            ctx.run_dir, dry_run=True, llm_client=None,
        )
    except Exception:  # noqa: BLE001 - export must never crash on bad cache.
        logger.exception(
            "[export] inconsistency materialisation failed for "
            "run %s; falling back to typo-only cells",
            ctx.run_dir.name,
        )
        return []

