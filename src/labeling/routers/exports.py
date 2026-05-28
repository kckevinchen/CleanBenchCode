"""Benchmark / sources export endpoints under ``/api/runs/{run_id}/export/*``.

Materialises the per-cell ground-truth benchmark (overall, by-error-type,
and single-type) and the sanitised sources ZIP (source CSVs minus the
operator's exclusions), plus the label-conflict report. Cell
materialisation + exclusion filtering live in :mod:`src.labeling._cells`.
"""

from __future__ import annotations

import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .._cells import (
    _apply_exclusions_to_cells,
    _make_value_resolver,
    _materialize_inconsistency_cells_for_export,
)
from ..export import (
    _slug_for_error_type,
    find_label_conflicts,
    materialize_label_benchmark,
    materialize_label_benchmark_by_error_type,
    write_benchmark,
    write_benchmark_by_error_type,
)
from ._context import RouterContext


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache
    annotator = router_ctx.annotator

    @app.get("/api/runs/{run_id}/export/conflicts")
    def export_conflicts(run_id: str) -> Dict[str, Any]:
        """Pre-export sanity check: groups of (source, column,
        original_value) cells whose ``label`` verdicts disagree.

        Runs against the SAME materialised benchmark the
        :func:`export_benchmark` endpoint would emit (including the
        same exclusion filter), so what the UI shows here is exactly
        the conflict set the operator's downloaded JSONL would carry.

        Empty ``conflicts`` => safe to export. Each conflict group
        carries the (source, column, original_value) triple, the
        distinct labels involved, and a per-cell list (row, label,
        corrected_value, cluster_ids, note) so the UI can let the
        operator jump straight to each offending cell.
        """
        ctx = cache.get(run_id)
        cells = materialize_label_benchmark(
            ctx.clusters, ctx.store,
            value_resolver=_make_value_resolver(ctx),
            cell_note_overrides=ctx.cell_notes_store.active_overrides(),
        )
        # Inconsistency cells share the (source, row, column,
        # original_value) key space so any divergent labels across
        # the typo and inconsistency pipelines surface here too.
        cells.extend(_materialize_inconsistency_cells_for_export(ctx))
        cells = _apply_exclusions_to_cells(ctx, cells)
        conflicts = find_label_conflicts(cells)
        return {"conflicts": conflicts}

    @app.get("/api/runs/{run_id}/export/benchmark")
    def export_benchmark(run_id: str) -> FileResponse:
        ctx = cache.get(run_id)
        # Materialise once, drop excluded slices, then write.
        # Tracks the same exclusion semantics as /export/sources so
        # the benchmark and the sanitised CSV bundle never disagree.
        cells = materialize_label_benchmark(
            ctx.clusters, ctx.store,
            value_resolver=_make_value_resolver(ctx),
            cell_note_overrides=ctx.cell_notes_store.active_overrides(),
        )
        # Append cache-only inconsistency cells so column-cluster-only
        # runs (where ``clusters.json`` is empty) still get a populated
        # benchmark from this single endpoint.
        cells.extend(_materialize_inconsistency_cells_for_export(ctx))
        cells = _apply_exclusions_to_cells(ctx, cells)
        out_path = ctx.run_dir / "benchmark.jsonl"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for cell in cells:
                fh.write(
                    json.dumps(cell.model_dump(), ensure_ascii=False) + "\n"
                )
        os.replace(tmp_path, out_path)
        return FileResponse(
            out_path,
            media_type="application/x-ndjson",
            filename="benchmark.jsonl",
        )

    @app.get("/api/runs/{run_id}/export/benchmark-by-type")
    def export_benchmark_by_type(run_id: str) -> FileResponse:
        """Per-error-type benchmark bundle.

        Materialises the same per-cell ground truth as
        :func:`export_benchmark`, but groups cells by ``error_type``
        and emits one JSONL per type into
        ``<run_dir>/benchmark_by_type/`` before zipping the directory
        for download. Cells with multiple ``error_types`` are
        replicated into each matching bucket so e.g. ``typo.jsonl``
        contains every cell where the annotator confirmed a typo,
        even when the same cell was also labelled with another type.
        Cells with no ``error_types`` (FP cells, FN cells without an
        enclosing cluster, ambiguous row-level findings) land in
        ``_uncategorized.jsonl`` so the bundle is lossless.
        """
        ctx = cache.get(run_id)
        out_dir = ctx.run_dir / "benchmark_by_type"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Apply exclusions before bucketing so every emitted JSONL
        # in the per-type bundle aligns with sources_<run>.zip.
        buckets = materialize_label_benchmark_by_error_type(
            ctx.clusters, ctx.store,
            value_resolver=_make_value_resolver(ctx),
            cell_note_overrides=ctx.cell_notes_store.active_overrides(),
        )
        # Fold inconsistency cells in under the natural bucket key.
        # Their ``error_types=["inconsistency"]`` means they always
        # land in a single slot; cells with empty error_types would
        # land in ``_uncategorized`` but the inconsistency exporter
        # never emits such cells.
        for inc_cell in _materialize_inconsistency_cells_for_export(ctx):
            for et in (inc_cell.error_types or ["_uncategorized"]):
                slug = _slug_for_error_type(et)
                buckets.setdefault(slug, []).append(inc_cell)
        # Reuse cell-level filter on each bucket; cells are object
        # instances shared across buckets but the filter is
        # idempotent so the duplication is harmless.
        filtered: Dict[str, List[Any]] = {}
        for slug, rows in buckets.items():
            kept = _apply_exclusions_to_cells(ctx, rows)
            if kept:
                filtered[slug] = kept
        # Drop stale per-type files we previously wrote so excluded
        # buckets disappear from the bundle, then rewrite the
        # surviving ones atomically.
        for stale in out_dir.glob("*.jsonl"):
            if stale.stem not in filtered:
                try:
                    stale.unlink()
                except OSError:
                    pass
        for slug, rows in filtered.items():
            path = out_dir / f"{slug}.jsonl"
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                for cell in rows:
                    fh.write(
                        json.dumps(cell.model_dump(), ensure_ascii=False)
                        + "\n"
                    )
            os.replace(tmp_path, path)
        # Zip the per-type directory for a single-file download. We
        # rebuild the zip every request so it stays in sync with the
        # latest labels (the operator may have just labelled more
        # clusters before clicking "Per-type benchmark").
        zip_path = ctx.run_dir / "benchmark_by_type.zip"
        tmp_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
        with zipfile.ZipFile(
            tmp_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for path in sorted(out_dir.glob("*.jsonl")):
                zf.write(path, arcname=path.name)
        os.replace(tmp_zip, zip_path)
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"benchmark_by_type_{run_id}.zip",
        )

    @app.get("/api/runs/{run_id}/export/benchmark/{error_type}")
    def export_benchmark_single_type(
        run_id: str, error_type: str
    ) -> FileResponse:
        """Per-(run, error_type) benchmark JSONL.

        Returns the materialised benchmark FILTERED to a single
        ``error_type`` bucket - the same content the per-type ZIP
        would carry under ``<error_type>.jsonl``, but as a plain
        JSONL download so an operator can grab e.g. just the
        ``typo`` cells without unpacking a bundle.

        ``error_type`` is matched against the same slug taxonomy
        used by :func:`write_benchmark_by_error_type`. The synthetic
        ``_uncategorized`` bucket is reachable too (cells with empty
        ``error_types`` - FP cells, FN cells without an enclosing
        cluster).

        Returns 404 when no labelled cell currently belongs to the
        requested bucket. The error message lists the buckets that
        DO exist for this run so the caller can correct the request
        without polling :func:`export_benchmark_by_type`.
        """
        ctx = cache.get(run_id)
        # Path-segment slug-normalise: ``error_type`` arrives URL-decoded
        # but we still want to canonicalise spaces/punct so ``Typo`` /
        # ``typo`` / ``typo.jsonl`` all resolve to the same bucket.
        requested = (error_type or "").strip()
        if requested.endswith(".jsonl"):
            requested = requested[: -len(".jsonl")]
        slug = _slug_for_error_type(requested)
        buckets = materialize_label_benchmark_by_error_type(
            ctx.clusters, ctx.store,
            value_resolver=_make_value_resolver(ctx),
            cell_note_overrides=ctx.cell_notes_store.active_overrides(),
        )
        # Mirror the per-type bundle: inconsistency cells fold in
        # under the bucket key derived from their error_types so the
        # single-type endpoint also serves them.
        for inc_cell in _materialize_inconsistency_cells_for_export(ctx):
            for et in (inc_cell.error_types or ["_uncategorized"]):
                inc_slug = _slug_for_error_type(et)
                buckets.setdefault(inc_slug, []).append(inc_cell)
        rows = buckets.get(slug)
        # Apply exclusions BEFORE the empty-bucket 404, so a bucket
        # that only ever held excluded cells now correctly returns 404
        # instead of a JSONL pointing at files / rows / columns that
        # are missing from sources_<run>.zip.
        if rows:
            rows = _apply_exclusions_to_cells(ctx, rows)
        if not rows:
            available = sorted(
                k for k, v in buckets.items()
                if _apply_exclusions_to_cells(ctx, v)
            )
            raise HTTPException(
                404,
                detail={
                    "error": "no_labelled_cells_for_error_type",
                    "requested": slug,
                    "available": available,
                },
            )
        # Materialise to a real file under the run dir so the
        # FileResponse can stream it. Atomic .tmp + os.replace dance
        # mirrors :func:`write_benchmark` so concurrent readers never
        # see a truncated bucket.
        out_dir = ctx.run_dir / "benchmark_by_type"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}.jsonl"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for cell in rows:
                fh.write(
                    json.dumps(cell.model_dump(), ensure_ascii=False) + "\n"
                )
        os.replace(tmp_path, out_path)
        return FileResponse(
            out_path,
            media_type="application/x-ndjson",
            filename=f"benchmark_{slug}_{run_id}.jsonl",
        )

    @app.get("/api/runs/{run_id}/export/sources")
    def export_sources(run_id: str) -> FileResponse:
        """ZIP of the run's source CSVs, sanitised against exclusions.

        Mirrors what the operator's exclusion decisions imply for a
        downstream cleaning step:

        * ``scope=csv``     - the file is omitted from the bundle.
        * ``scope=rows``    - the listed positional row indices are
                              dropped from the file. Indices match the
                              same 0-based ``row`` field used in
                              clusters / the labelling UI.
        * ``scope=columns`` - the listed column names are dropped.

        Sources with no active exclusion pass through unchanged
        (raw bytes are copied verbatim so quirky encodings / quoting
        survive). Sources with active row/column exclusions are
        re-emitted via pandas' default CSV writer using the cached,
        normalised frame the rest of the labelling backend uses, so
        the cell coordinates inside the sanitised file still align
        with ``original_value`` recorded in the benchmark.

        The ZIP preserves each source's directory structure relative
        to the longest path prefix all sources share - so e.g. a run
        targeting ``data/<corpus>/<workbook>/<sheet>.csv`` for many
        workbooks is bundled as ``<workbook>/<sheet>.csv``. When all
        sources share no ancestor above the basename (single-source
        runs, or sources scattered across unrelated trees), the bundle
        falls back to flat basenames.
        """
        ctx = cache.get(run_id)
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

        # Build the per-source arcname map. We prefer paths relative
        # to the deepest directory all in-bundle sources share, so the
        # zip retains the meaningful corpus layout (e.g.
        # ``<workbook>/<sheet>.csv``) without leaking absolute paths.
        # When the common ancestor would degenerate to "no shared
        # parent" (e.g. mixed corpora) we fall back to the raw basename
        # so the export still succeeds.
        included = [
            (source, src_path)
            for source, src_path in sorted(ctx.sources.items())
            if source not in excluded_files
        ]
        try:
            resolved_paths = [p.resolve() for _, p in included]
        except OSError:
            resolved_paths = [Path(p) for _, p in included]
        common_prefix: Optional[Path] = None
        if len(resolved_paths) >= 2:
            try:
                common_str = os.path.commonpath(
                    [str(p) for p in resolved_paths]
                )
                cp = Path(common_str)
                # Only use the ancestor when it's a proper directory
                # above the files. ``commonpath`` returns a parent dir
                # for distinct files; if it equals one of the paths we
                # have a degenerate single-file case.
                if cp.is_dir() and not any(p == cp for p in resolved_paths):
                    common_prefix = cp
            except ValueError:
                # Mixed drive letters / unrelated trees - fall back.
                common_prefix = None

        def _arcname(src_path: Path, source: str) -> str:
            if common_prefix is not None:
                try:
                    rp = src_path.resolve()
                    rel = rp.relative_to(common_prefix)
                    rel_str = rel.as_posix()
                    # Defence in depth: never let a resolved path
                    # escape the prefix (symlink games, etc.).
                    if not rel_str.startswith("..") and rel_str:
                        return rel_str
                except (OSError, ValueError):
                    pass
            return source

        out_dir = ctx.run_dir / "sanitised_sources"
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = ctx.run_dir / f"sanitised_sources_{run_id}.zip"
        tmp_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
        with zipfile.ZipFile(
            tmp_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for source, src_path in included:
                arcname = _arcname(src_path, source)
                drop_rows = per_source_rows.get(source) or set()
                drop_cols = per_source_cols.get(source) or set()
                if not drop_rows and not drop_cols:
                    # Fast path: copy the raw file verbatim so
                    # encoding / quoting / mixed dtypes survive.
                    if not src_path.exists():
                        continue
                    zf.write(src_path, arcname=arcname)
                    continue
                # Row / column exclusions: re-emit via the loaded frame
                # so the dropped slices line up with the indices the
                # labelling UI used.
                try:
                    frame = ctx.load_frame(source)
                except (FileNotFoundError, OSError, ValueError):
                    continue
                if drop_cols:
                    keep_cols = [c for c in frame.columns if c not in drop_cols]
                    frame = frame.loc[:, keep_cols]
                if drop_rows:
                    keep_mask = [
                        i for i in range(len(frame)) if i not in drop_rows
                    ]
                    frame = frame.iloc[keep_mask]
                # Materialise the sanitised frame to disk first so
                # zipfile can stream its bytes; the directory is
                # reused on every export so it doubles as an
                # inspectable artefact under <run_dir>. Use the same
                # arcname-relative layout on disk as in the zip so an
                # operator browsing the run dir sees the full corpus
                # structure, not flat basenames.
                out_path = out_dir / arcname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                frame.to_csv(tmp_path, index=False)
                os.replace(tmp_path, out_path)
                zf.write(out_path, arcname=arcname)
        os.replace(tmp_zip, zip_path)
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"sources_{run_id}.zip",
        )

