"""FastAPI app factory for the labelling + experiment-launcher UI.

One server instance = one project root. The server manages:

* All detection-run directories under ``<project_root>/outputs/``.
* All label stores (one ``labels.jsonl`` per run dir; loaded lazily and
  cached).
* Child-process jobs launched via ``experiments.run_cleaning_detection``
  and ``experiments.build_report``.

URL layout
==========

``GET  /``                                - redirect to ``/ui/``
``/ui/*``                                 - static frontend (index.html
                                            = runs dashboard + launcher,
                                            label.html = annotation UI).
``/api/runs``                             - list all run dirs.
``/api/runs/{run_id}``                    - run metadata + sources.
``/api/runs/{run_id}/clusters``           - queue, filtered.
``/api/runs/{run_id}/clusters/{cid}``     - detail + label + history.
``/api/runs/{run_id}/sources/{name}``     - CSV slice.
``/api/runs/{run_id}/labels`` (GET/POST)  - label store for this run.
``/api/runs/{run_id}/stats``              - per-status / per-source counts.
``/api/runs/{run_id}/export/benchmark``   - per-cell benchmark JSONL.
``/api/runs/{run_id}/export/sources``     - ZIP of source CSVs minus
                                            exclusions (whole-file,
                                            row, and column scopes).
``/api/runs/{run_id}/report``             - redirect to report.html.
``/api/launcher/config``                  - defaults + taxonomy + datasets.
``/api/launcher/jobs`` (GET/POST)         - list / start jobs.
``/api/launcher/jobs/{job_id}``           - job status.
``/api/launcher/jobs/{job_id}/log``       - tail job log.
``/api/launcher/jobs/{job_id}/cancel``    - SIGTERM (POST).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..core.data_loader import load_csv
from ..core.llm_client import LLMClient
from ..core.text_normalize import NormalizeOptions, normalize_frame
from . import correction_suggester
from . import phone_format
from ._cells import (
    _apply_exclusions_to_cells,
    _build_exclusion_lookup,
    _make_value_resolver,
    _materialize_inconsistency_cells_for_export,
)
from ._helpers import _scalar, _validate_source_name
from .export import (
    _slug_for_error_type,
    find_label_conflicts,
    materialize_label_benchmark,
    materialize_label_benchmark_by_error_type,
    write_benchmark,
    write_benchmark_by_error_type,
)
from .inconsistency_export import materialize_inconsistency_benchmark
from .runs import (
    JobManager,
    RunRegistry,
    list_datasets,
    load_config_defaults,
    load_taxonomy_keys,
    _load_run_metadata,
    _save_run_metadata,
)
from .schema import (
    CellNote,
    CellNoteIn,
    Exclusion,
    ExclusionIn,
    ExclusionScope,
    Label,
    LabelIn,
    LabelStats,
    LabelStatus,
    LabelTarget,
    MultiTableLabel,
    MultiTableLabelIn,
    ColumnClusters,
    ColumnClustersIn,
    ColumnLabelStatus,
    cell_target_id,
    column_cluster_target_id,
)
from .store import LabelStore
from .cell_notes_store import CellNotesStore
from .exclusions_store import ExclusionStore
from .multi_table_label_store import MultiTableLabelStore
from .column_cluster_store import ColumnClusterStore
from .column_cluster_queue import iter_column_cluster_targets
from .routers import cell_notes as cell_notes_router
from .routers import compare as compare_router
from .routers import exclusions as exclusions_router
from .routers import exports as exports_router
from .routers import launcher as launcher_router
from .routers import multi_table as multi_table_router
from .routers import relation_reviews as relation_reviews_router
from .routers import sources as sources_router
from .routers import stats as stats_router
from .routers._context import RouterContext

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Per-run-directory context (lazy, cached)
# ---------------------------------------------------------------------------


# Bing21 deduped corpus stores one xlsx-derived sheet per CSV with a
# filename of the form ``<wb_sha40>.xlsx_<sheet_sha8>.csv``. The xlsx
# prefix uniquely identifies the source workbook even though every CSV
# sits in the same flat parent directory, so we lift it out and use it
# as the workbook key. Anything that doesn't match this pattern (the
# usual case: curated_pbi/<sid>/*.csv) falls back to the parent-dir
# basename, which is the natural workbook unit for hierarchical
# layouts.
_BING21_FLAT_RE = re.compile(
    r"^(?P<wb>[0-9a-f]{8,64}\.xlsx)_[0-9a-f]+\.csv$",
    re.IGNORECASE,
)


def _workbook_for(pp: Path) -> str:
    m = _BING21_FLAT_RE.match(pp.name)
    if m:
        return m.group("wb")
    return pp.parent.name


def _count_multi_table_findings(run_dir: Path) -> int:
    """Count distinct ``violation_id``s in ``multi_table_violations.jsonl``.

    The file holds one record per (sheet, row) per finding; many
    findings span 2+ rows so we deduplicate by ``violation_id`` for
    the headline figure exposed alongside cluster counts in the run
    summary endpoint. Returns 0 when the file is absent or unreadable.
    """
    path = run_dir / "multi_table_violations.jsonl"
    if not path.exists():
        return 0
    seen: set = set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vid = rec.get("violation_id")
                if vid:
                    seen.add(vid)
    except OSError:
        return 0
    return len(seen)


# Soft caps on the openpyxl-driven preview. A pathological xlsx can
# allocate gigabytes of cell-objects, which is bad for an interactive
# server. We bound (rows, cols, sheets) and emit a truncation banner
# rather than refusing to render.
_PREVIEW_MAX_SHEETS = 32
_PREVIEW_MAX_ROWS = 500
_PREVIEW_MAX_COLS = 60


def _render_xlsx_preview_html(
    xlsx_path: Path,
    *,
    workbook_name: str,
    download_url: str,
) -> str:
    """Render every sheet of ``xlsx_path`` as an HTML preview page.

    Uses ``openpyxl`` in read-only mode with ``data_only=True`` so we
    see evaluated formula values rather than the formula text. Output
    is a single self-contained document with one tab per sheet plus
    a download button. All cell text is HTML-escaped.
    """
    from html import escape as h

    try:
        import openpyxl  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (
            "<!doctype html><html><body><pre>openpyxl unavailable: "
            f"{h(str(exc))}</pre></body></html>"
        )

    try:
        wb = openpyxl.load_workbook(
            xlsx_path, read_only=True, data_only=True
        )
    except Exception as exc:  # noqa: BLE001
        return (
            "<!doctype html><html><body><pre>"
            f"failed to open {h(workbook_name)}: {h(str(exc))}"
            "</pre></body></html>"
        )

    sheet_names = list(wb.sheetnames)
    truncated_sheets = False
    if len(sheet_names) > _PREVIEW_MAX_SHEETS:
        sheet_names = sheet_names[:_PREVIEW_MAX_SHEETS]
        truncated_sheets = True

    parts: List[str] = []
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    parts.append(f"<title>{h(workbook_name)} - preview</title>")
    parts.append(
        "<meta http-equiv='Content-Security-Policy' content=\""
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'\">"
    )
    parts.append("<style>")
    parts.append(
        "body{font:13px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "margin:0;background:#0b1020;color:#e2e8f0}"
        "header{position:sticky;top:0;background:#111a3a;border-bottom:1px solid #1e293b;"
        "padding:10px 14px;display:flex;gap:12px;align-items:center;z-index:10}"
        "header h1{margin:0;font-size:14px;font-weight:600}"
        "header .muted{color:#94a3b8;font-size:12px}"
        "header a{color:#60a5fa;text-decoration:none;border:1px solid #1e293b;"
        "padding:4px 10px;border-radius:6px;font-size:12px}"
        "header a:hover{background:#1e293b}"
        "nav.tabs{display:flex;gap:2px;flex-wrap:wrap;background:#0f172a;"
        "padding:6px 10px;border-bottom:1px solid #1e293b;position:sticky;top:42px;z-index:9}"
        "nav.tabs button{background:transparent;color:#cbd5e1;border:1px solid transparent;"
        "padding:4px 10px;border-radius:6px 6px 0 0;cursor:pointer;font-size:12px}"
        "nav.tabs button.active{background:#1e293b;color:#fff;border-color:#334155;border-bottom-color:#1e293b}"
        "section.sheet{display:none;padding:10px 14px}"
        "section.sheet.active{display:block}"
        ".banner{background:#7c2d12;color:#fed7aa;padding:6px 10px;border-radius:6px;"
        "margin:6px 0;font-size:12px}"
        "table{border-collapse:collapse;font-size:12px;font-family:ui-monospace,monospace;"
        "background:#0f172a;border:1px solid #1e293b}"
        "th,td{padding:4px 8px;border:1px solid #1e293b;text-align:left;vertical-align:top;"
        "max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        "th{background:#1e293b;color:#cbd5e1;font-weight:600;position:sticky;top:0}"
        "td:first-child,th:first-child{position:sticky;left:0;background:#1e293b;color:#94a3b8;"
        "min-width:42px;text-align:right;font-weight:400}"
        "tr:nth-child(even) td{background:#111a3a}"
    )
    parts.append("</style></head><body>")

    parts.append("<header>")
    parts.append(f"<h1>{h(workbook_name)}</h1>")
    parts.append(f"<span class='muted'>{len(wb.sheetnames)} sheet(s)</span>")
    parts.append(f"<a href='{h(download_url)}' download>Download .xlsx</a>")
    parts.append("</header>")

    parts.append("<nav class='tabs'>")
    for i, sn in enumerate(sheet_names):
        cls = "active" if i == 0 else ""
        parts.append(
            f"<button type='button' class='{cls}' "
            f"data-sheet='sheet-{i}'>{h(sn)}</button>"
        )
    parts.append("</nav>")
    if truncated_sheets:
        parts.append(
            f"<div class='banner' style='margin:10px 14px'>Showing first "
            f"{_PREVIEW_MAX_SHEETS} of {len(wb.sheetnames)} sheet(s).</div>"
        )

    for i, sn in enumerate(sheet_names):
        ws = wb[sn]
        cls = "sheet active" if i == 0 else "sheet"
        parts.append(f"<section id='sheet-{i}' class='{cls}'>")
        truncated_rows = False
        truncated_cols = False
        rows_iter = ws.iter_rows(values_only=True)
        rendered_rows: List[List[str]] = []
        max_cols_seen = 0
        for r_idx, row in enumerate(rows_iter):
            if r_idx >= _PREVIEW_MAX_ROWS:
                truncated_rows = True
                break
            row_list = list(row)
            if len(row_list) > _PREVIEW_MAX_COLS:
                row_list = row_list[:_PREVIEW_MAX_COLS]
                truncated_cols = True
            max_cols_seen = max(max_cols_seen, len(row_list))
            rendered_rows.append(["" if v is None else str(v) for v in row_list])
        if not rendered_rows:
            parts.append(f"<div class='muted'>(empty sheet: {h(sn)})</div>")
            parts.append("</section>")
            continue
        if truncated_rows or truncated_cols:
            note = []
            if truncated_rows:
                note.append(f"first {_PREVIEW_MAX_ROWS} rows")
            if truncated_cols:
                note.append(f"first {_PREVIEW_MAX_COLS} columns")
            parts.append(f"<div class='banner'>Showing {' / '.join(note)}.</div>")
        # Use the first row as headers; render every subsequent row
        # under it. We don't try to detect "real" headers - the raw
        # first row is good enough for a preview.
        parts.append("<table><thead><tr><th>#</th>")
        header_row = rendered_rows[0]
        # pad header to max width seen so header+body widths align.
        if len(header_row) < max_cols_seen:
            header_row = header_row + [""] * (max_cols_seen - len(header_row))
        for cell in header_row:
            parts.append(f"<th>{h(cell)}</th>")
        parts.append("</tr></thead><tbody>")
        for r_n, body_row in enumerate(rendered_rows[1:], start=2):
            if len(body_row) < max_cols_seen:
                body_row = body_row + [""] * (max_cols_seen - len(body_row))
            parts.append(f"<tr><td>{r_n}</td>")
            for cell in body_row:
                parts.append(f"<td>{h(cell)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
        parts.append("</section>")

    wb.close()
    parts.append(
        "<script>"
        "document.addEventListener('click',function(e){"
        "var b=e.target.closest('nav.tabs button[data-sheet]');"
        "if(!b)return;"
        "var id=b.getAttribute('data-sheet');"
        "document.querySelectorAll('section.sheet').forEach(function(s){s.classList.remove('active')});"
        "document.querySelectorAll('nav.tabs button').forEach(function(x){x.classList.remove('active')});"
        "b.classList.add('active');"
        "var t=document.getElementById(id);if(t)t.classList.add('active');"
        "});"
        "</script>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def _fold_header(s: object) -> str:
    """Backward-compat wrapper around
    :func:`src.core.columns.fold_header`. Kept so callers / tests that
    imported the labeler-internal symbol continue to work.
    """
    from ..core.columns import fold_header
    return fold_header(s)


def _canonicalize_cluster_columns(
    cluster: Mapping[str, Any], frame_headers: List[str]
) -> Dict[str, Any]:
    """Backward-compat wrapper around
    :func:`src.core.columns.canonicalize_cluster_columns`. The shared
    module is the single source of truth for the matching tiers.
    """
    from ..core.columns import canonicalize_cluster_columns
    return canonicalize_cluster_columns(cluster, frame_headers)


class RunContext:
    """Lazy read-only view of one run dir."""

    # Pipeline-written artifacts the context snapshots at construction.
    # ``RunCache.get()`` re-stats these to detect re-runs of the
    # detection/relation pipelines that rewrote the run dir under a
    # live server, and rebuilds the context so the UI doesn't keep
    # serving the pre-rerun in-memory view. Operator-written stores
    # (labels.jsonl / exclusions.jsonl) are mutated through this same
    # process and so don't need mtime tracking here.
    _PIPELINE_ARTIFACTS: Tuple[str, ...] = (
        "summary.json", "clusters.json", "issues.json",
    )

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"run_dir not found: {self.run_dir}")

        self.summary: Dict[str, Any] = self._read_json("summary.json") or {}
        self.clusters: List[Dict[str, Any]] = self._read_json("clusters.json") or []
        self.issues: List[Dict[str, Any]] = self._read_json("issues.json") or []
        # Snapshot the on-disk mtimes the context's clusters/issues/
        # summary state was loaded from, so RunCache can detect when a
        # rerun has invalidated us.
        self._artifact_mtimes: Dict[str, float] = {
            name: self._artifact_mtime(name)
            for name in self._PIPELINE_ARTIFACTS
        }

        cfg = (self.summary.get("config") or {}) if isinstance(self.summary, dict) else {}
        self.sources: Dict[str, Path] = {}
        # Each source's parent-directory basename. For multi-table corpora
        # like ``data/curated_pbi/<workbook_id>/*.csv`` or
        # ``data/Bing21/<workbook_hash>/*.csv`` this is the workbook the
        # CSV belongs to. For flat single-table corpora every source
        # ends up sharing the same parent (e.g. "curated"), in which
        # case the frontend collapses the grouping (see
        # ``multi_workbook`` flag in ``GET /api/runs/{run_id}``).
        self.source_workbook: Dict[str, str] = {}
        for p in cfg.get("csv_paths") or []:
            pp = Path(p)
            self.sources.setdefault(pp.name, pp)
            self.source_workbook.setdefault(pp.name, _workbook_for(pp))

        # Backfill for legacy multi-table runs (relations runs written
        # before ``csv_paths`` landed in summary.json). When the run
        # dir contains a ``multi_table_groups.jsonl`` but the config
        # has no CSV paths, scan ``<project_root>/data/**/<workbook>/
        # <sheet>`` for the workbook+sheet combos referenced by the
        # groups file. The labelling UI's multi-table detail endpoint
        # uses these to surface full row contents next to the LLM
        # snapshot. Failures here are non-fatal: load_frame() falls
        # back to the snapshot anyway.
        if not self.sources:
            self._backfill_multi_table_sources()

        # The parent-dir = workbook heuristic only carries meaning when
        # the run actually targets multiple workbooks AND at least one
        # of them owns 2+ CSVs (e.g. curated_pbi/<sid>/*.csv). For flat
        # corpora like ``data/curated_bing21/*.csv`` every CSV gets its
        # own synthetic workbook (derived from the filename) and is
        # *not* a sibling of the others. Pre-compute the verdict so
        # both the run summary and the per-source detail endpoint stay
        # consistent.
        wb_to_sources: Dict[str, List[str]] = {}
        for name, wb in self.source_workbook.items():
            wb_to_sources.setdefault(wb or "", []).append(name)
        distinct = [wb for wb in wb_to_sources if wb]
        self.multi_workbook: bool = (
            len(distinct) >= 2
            and any(len(v) > 1 for v in wb_to_sources.values())
        )

        self._frame_cache: Dict[str, pd.DataFrame] = {}
        self._frame_lock = threading.Lock()
        self._canonical_cache: Dict[str, Dict[str, Any]] = {}

        # Original-xlsx source directory. Resolved either from explicit
        # config (``cfg["xlsx_source_dir"]``) or auto-detected by looking
        # for a sibling ``<corpus>_xlsx`` directory next to the curated
        # CSV corpus. Each workbook (= ``self.source_workbook[name]``,
        # which for our datasets is already the original ``<hash>.xlsx``
        # filename) is then served as ``<xlsx_source_dir>/<wb>``.
        self.xlsx_source_dir: Optional[Path] = None
        self._xlsx_source_dir_resolved: Optional[Path] = None
        cfg_xlsx = cfg.get("xlsx_source_dir") if isinstance(cfg, dict) else None
        candidates: List[Path] = []
        if cfg_xlsx:
            candidates.append(Path(cfg_xlsx))
        # Auto-detect: every CSV path lives under a corpus root; try
        # ``<corpus_root>_xlsx`` siblings. We probe the parent (single-
        # table flat corpora) and the grandparent (workbook-grouped
        # layouts like ``data/curated_bing21_min/<wb>.xlsx/<table>.csv``).
        seen: set = set()
        for p in self.sources.values():
            for anc in (p.parent, p.parent.parent):
                if anc in seen or anc == anc.parent:
                    continue
                seen.add(anc)
                candidates.append(anc.parent / f"{anc.name}_xlsx")
        for cand in candidates:
            try:
                rp = cand.resolve()
            except OSError:
                continue
            if rp.is_dir():
                self.xlsx_source_dir = cand
                self._xlsx_source_dir_resolved = rp
                break

        # If the run was launched with --normalize-input, the LLM saw a
        # transformed view of the CSV (lowercased / whitespace-folded /
        # punctuation-folded headers and cells). Cluster column names and
        # row_values were recorded against that transformed view, so the
        # labeling UI must render the same transformed frame - otherwise
        # ``flagged.has(col)`` silently fails to highlight cells whose
        # column casing/whitespace differs from the raw CSV header.
        self._normalize_opts: Optional[NormalizeOptions] = None
        ni = cfg.get("normalize_input") if isinstance(cfg, dict) else None
        if isinstance(ni, dict) and ni.get("enabled"):
            try:
                # Filter to only known fields so older configs with
                # extra keys don't blow up the dataclass.
                allowed = {f for f in NormalizeOptions.__dataclass_fields__}
                self._normalize_opts = NormalizeOptions(
                    **{k: v for k, v in ni.items() if k in allowed}
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("normalize_input config invalid: %s", exc)
                self._normalize_opts = None

        self._cluster_by_id = {c["cluster_id"]: c for c in self.clusters if "cluster_id" in c}
        self._issue_by_id = {i["id"]: i for i in self.issues if "id" in i}

        self.store = LabelStore(self.run_dir / "labels.jsonl")
        # Per-cell note overrides (override the cluster's ``note`` for
        # a single (source, row, column) at export time). See
        # :mod:`src.labeling.cell_notes_store` for the storage format.
        self.cell_notes_store = CellNotesStore(
            self.run_dir / "cell_notes.jsonl"
        )
        # Per-source exclusions (whole CSV / specific rows / specific
        # columns the operator marked as too messy to label). Loaded
        # eagerly so the UI can render existing strikethroughs without
        # an extra round-trip.
        self.exclusions_store = ExclusionStore(self.run_dir / "exclusions.jsonl")

        # Web verifications (optional, written by experiments.run_web_verify).
        # Key by cluster_id for O(1) lookup; the file is small enough
        # (one record per gate-eligible cluster) to keep in memory.
        self._verifications: Dict[str, Dict[str, Any]] = {}
        self._verifications_mtime: float = 0.0
        self._reload_verifications()

        # Per-arm Good-Turing artifact (optional). Adaptive runs write
        # ``good_turing/per_arm_estimates.json`` with one record per
        # (scope_kind, scope_id, error_type) arm. We cache it lazily
        # so single-pass runs incur no I/O.
        self._per_arm_payload: Optional[Dict[str, Any]] = None
        self._per_arm_loaded: bool = False

    def per_arm_payload(self) -> Optional[Dict[str, Any]]:
        """Return the parsed ``good_turing/per_arm_estimates.json`` or
        ``None`` if the file is missing/invalid (e.g. single-pass run)."""
        if not self._per_arm_loaded:
            self._per_arm_loaded = True
            path = self.run_dir / "good_turing" / "per_arm_estimates.json"
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        self._per_arm_payload = raw
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("failed to parse %s: %s", path, exc)
        return self._per_arm_payload

    def _verifications_path(self) -> Path:
        return self.run_dir / "web_verifications.json"

    def _reload_verifications(self) -> None:
        path = self._verifications_path()
        if not path.exists():
            self._verifications = {}
            self._verifications_mtime = 0.0
            return
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime == self._verifications_mtime and self._verifications:
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to parse %s: %s", path, exc)
            return
        if not isinstance(raw, list):
            return
        self._verifications = {
            r["cluster_id"]: r
            for r in raw
            if isinstance(r, dict) and r.get("cluster_id")
        }
        self._verifications_mtime = mtime

    def verification(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Return the persisted web-verification record for a cluster,
        or ``None`` if the verifier hasn't run on it yet. Reloads the
        underlying JSON file lazily so a freshly-finished verifier
        run becomes visible without restarting the server."""
        self._reload_verifications()
        return self._verifications.get(cluster_id)

    def _read_json(self, name: str) -> Any:
        path = self.run_dir / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse %s: %s", path, exc)
            return None

    def _artifact_mtime(self, name: str) -> float:
        """Return the current mtime of ``<run_dir>/<name>`` or 0.0 if
        the file is missing. Used by :meth:`is_stale` to detect
        out-of-band rewrites of pipeline artifacts."""
        try:
            return (self.run_dir / name).stat().st_mtime
        except OSError:
            return 0.0

    def is_stale(self) -> bool:
        """Return True if any of the pipeline-written artifacts the
        context snapshotted at construction has a newer mtime on disk.

        :class:`RunCache` uses this to detect reruns of the detection /
        relation-validation pipelines that rewrote ``clusters.json``,
        ``issues.json`` or ``summary.json`` under a live server, and to
        rebuild the context so the UI stops serving the pre-rerun view.
        """
        for name, cached_mtime in self._artifact_mtimes.items():
            if self._artifact_mtime(name) > cached_mtime:
                return True
        return False

    def _backfill_multi_table_sources(self) -> None:
        """Populate ``self.sources`` from a sibling ``data/`` tree when
        the run's ``summary.json`` has no ``csv_paths``.

        Used by legacy relations runs where the summary writer didn't
        record source paths. Reads ``multi_table_groups.jsonl`` to
        collect (workbook, sheet) pairs and looks them up under
        ``<project_root>/data/**/<workbook>/<sheet>``.

        When multiple corpora hold the same hash-named file (e.g. raw
        ``bing21_grouped`` vs curated ``curated_bing21_min_multi_table``)
        we disambiguate by matching against ``sources.json`` row counts:
        the LLM ran against one specific corpus and the row indices in
        the violations file are only meaningful for that frame.
        Failures here are non-fatal: load_frame() falls back to the
        snapshot anyway.
        """
        groups_path = self.run_dir / "multi_table_groups.jsonl"
        if not groups_path.exists():
            return
        # Project root = parent of outputs/. Defensive: only proceed
        # when ``data/`` exists at that level (otherwise we'd glob the
        # whole filesystem).
        project_root = self.run_dir.parent.parent
        data_root = project_root / "data"
        if not data_root.is_dir():
            return
        wanted: Dict[str, str] = {}      # sheet -> workbook
        try:
            with groups_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    wb = (rec.get("workbook") or "").strip()
                    for sh in rec.get("sheets") or []:
                        if isinstance(sh, str) and sh:
                            wanted.setdefault(sh, wb)
        except OSError as exc:
            logger.debug("backfill: failed to read %s: %s", groups_path, exc)
            return
        if not wanted:
            return

        # ``sources.json`` (written by both runners) records ``n_rows``
        # per source. Matching that against each candidate's actual
        # row count lets us pick the curation the LLM actually saw,
        # not just the first directory we glob into.
        expected_rows: Dict[str, int] = {}
        sources_path = self.run_dir / "sources.json"
        if sources_path.exists():
            try:
                payload = json.loads(sources_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    for s in payload:
                        if isinstance(s, dict):
                            name = s.get("source")
                            n = s.get("n_rows")
                            if isinstance(name, str) and isinstance(n, int):
                                expected_rows[name] = n
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("backfill: failed to parse %s: %s", sources_path, exc)

        def _row_count(p: Path) -> Optional[int]:
            # Cheap CSV row count (excluding header). pandas would be
            # accurate but expensive for hundreds of files; counting
            # newlines is good enough as a discriminator.
            try:
                with p.open("rb") as fh:
                    n = sum(1 for _ in fh)
                return max(0, n - 1)
            except OSError:
                return None

        for sheet, workbook in wanted.items():
            candidates: List[Path] = []
            if workbook:
                candidates.extend(
                    p for p in data_root.glob(f"*/{workbook}/{sheet}")
                    if p.is_file()
                )
            if not candidates:
                candidates.extend(
                    p for p in data_root.glob(f"**/{sheet}") if p.is_file()
                )
            if not candidates:
                continue
            chosen: Optional[Path] = None
            target = expected_rows.get(sheet)
            if target is not None:
                for cand in candidates:
                    if _row_count(cand) == target:
                        chosen = cand
                        break
            if chosen is None:
                chosen = candidates[0]
            self.sources.setdefault(sheet, chosen)
            self.source_workbook.setdefault(
                sheet, workbook or _workbook_for(chosen),
            )


    def load_frame(self, source: str) -> pd.DataFrame:
        with self._frame_lock:
            cached = self._frame_cache.get(source)
            if cached is not None:
                return cached
            path = self.sources.get(source)
            if path is None or not path.exists():
                raise FileNotFoundError(f"source not found: {source}")
            # Use the same loader the detector pipeline uses so column
            # names line up exactly with what's recorded in
            # ``clusters.json`` (incl. duplicate-column disambiguation
            # via ``Foo``/``Foo.1``). Otherwise the labeling UI would
            # silently fail to highlight flagged cells whose column
            # name in the cluster differs from the raw CSV header.
            frame = load_csv(path)
            if self._normalize_opts is not None:
                frame = normalize_frame(frame, self._normalize_opts)
            self._frame_cache[source] = frame
            return frame

    def cluster(self, cid: str) -> Optional[Dict[str, Any]]:
        return self._cluster_by_id.get(cid)

    def canonical_cluster(self, cid: str) -> Optional[Dict[str, Any]]:
        """Return the cluster with ``columns`` resolved against the
        actual CSV headers.

        The LLM occasionally returns shortened column names in its issue
        payload (e.g. ``"District"`` when the real header is
        ``"District\\nCode"``). Those mangled names get persisted into
        ``clusters.json``, so cluster ``columns`` cannot be trusted to
        match what's actually in the frame. We resolve once here, using
        the loaded frame's headers as the authoritative set, and cache
        per-run / per-cluster so the labeling UI can do plain exact-set
        membership lookups.
        """
        c = self._cluster_by_id.get(cid)
        if c is None:
            return None
        cached = self._canonical_cache.get(cid)
        if cached is not None:
            return cached
        source = c.get("source") or ""
        try:
            frame = self.load_frame(source) if source else None
        except FileNotFoundError:
            frame = None
        if frame is None:
            return c
        resolved = _canonicalize_cluster_columns(c, list(frame.columns))
        self._canonical_cache[cid] = resolved
        return resolved

    def issue(self, iid: str) -> Optional[Dict[str, Any]]:
        return self._issue_by_id.get(iid)

    @property
    def known_workbooks(self) -> set:
        return {wb for wb in self.source_workbook.values() if wb}

    def workbook_xlsx_path(self, workbook: str) -> Optional[Path]:
        """Resolve a known workbook to its original ``.xlsx`` file.

        Returns ``None`` if no xlsx source directory is configured /
        auto-detected, the workbook is unknown, or the resolved file
        sits outside the configured source dir (defense-in-depth
        against path traversal). The returned path always exists.
        """
        if self._xlsx_source_dir_resolved is None:
            return None
        if workbook not in self.known_workbooks:
            return None
        # Workbook names are file basenames (e.g. ``<hash>.xlsx``); we
        # reject anything with separators or traversal tokens before
        # touching the filesystem.
        if (
            not workbook
            or workbook in (".", "..")
            or "/" in workbook
            or "\\" in workbook
            or "\x00" in workbook
        ):
            return None
        cand = self._xlsx_source_dir_resolved / workbook
        try:
            resolved = cand.resolve()
        except OSError:
            return None
        # Must remain inside the configured xlsx source dir.
        try:
            resolved.relative_to(self._xlsx_source_dir_resolved)
        except ValueError:
            return None
        if not resolved.is_file():
            return None
        return resolved


class RunCache:
    """Cache loaded :class:`RunContext` instances keyed by run_id."""

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = Path(outputs_dir)
        self._outputs_root = self.outputs_dir.resolve()
        self._cache: Dict[str, RunContext] = {}
        self._lock = threading.Lock()

    def _safe_run_dir(self, run_id: str) -> Path:
        if (
            not run_id
            or run_id in (".", "..")
            or "/" in run_id
            or "\\" in run_id
            or "\x00" in run_id
        ):
            raise HTTPException(400, f"invalid run_id: {run_id!r}")
        rd = (self.outputs_dir / run_id).resolve()
        try:
            rd.relative_to(self._outputs_root)
        except ValueError as exc:
            raise HTTPException(400, f"invalid run_id: {run_id!r}") from exc
        if rd == self._outputs_root or rd.parent != self._outputs_root:
            raise HTTPException(400, f"invalid run_id: {run_id!r}")
        return rd

    def get(self, run_id: str) -> RunContext:
        rd = self._safe_run_dir(run_id)
        with self._lock:
            ctx = self._cache.get(run_id)
            if ctx is not None:
                if not ctx.is_stale():
                    return ctx
                # Pipeline artifacts have been rewritten under us
                # (typical cause: an out-of-band re-run of
                # ``experiments.run_relation_discovery`` /
                # ``experiments.revalidate_relations`` /
                # ``experiments.run_detection``). Evict and rebuild so
                # the UI stops serving the pre-rerun view.
                self._cache.pop(run_id, None)
                logger.info(
                    "RunCache: rebuilding stale context for run_id=%s",
                    run_id,
                )
        try:
            ctx = RunContext(rd)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        with self._lock:
            self._cache.setdefault(run_id, ctx)
            return self._cache[run_id]

    def invalidate(self, run_id: str) -> None:
        """Drop a cached :class:`RunContext` so the next access re-reads
        the run dir from disk. Safe to call for non-cached IDs."""
        with self._lock:
            self._cache.pop(run_id, None)


# ---------------------------------------------------------------------------
# In-memory background-job tracker for batch correction-suggestion runs.
#
# Tiny by design: we hold onto the result dict in memory until the
# server restarts. Batch jobs typically finish in tens of seconds (the
# LLM cache covers most of the work) so persisting them across restarts
# is overkill; the operator can always click the batch button again to
# rebuild from cache. Kept module-level so a single ThreadPool is
# shared across requests within one server process.
# ---------------------------------------------------------------------------


@dataclass
class _CorrectionJob:
    """State for one async batch correction job.

    ``status`` transitions: ``queued`` -> ``running`` -> ``done`` |
    ``error``. ``progress`` is updated by the dispatch progress callback
    so the UI can render a "12 / 87 LLM calls done" hint.
    """

    job_id: str
    run_id: str
    targets: List[Tuple[str, str, str]] = field(default_factory=list)
    status: str = "queued"            # queued | running | done | error
    phase: str = ""                    # "phase A" / "phase B" / ""
    done: int = 0
    total: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    # Per-cluster suggestion payload, ready for the UI. Keyed by
    # ``f"{source}\x00{column}\x00{cluster_local_id}"`` for JSON-safe
    # transport (tuple keys can't be serialised).
    results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class _CorrectionJobRegistry:
    """Thread-safe registry + bounded executor for correction jobs.

    Concurrency is intentionally low (max_workers=2): the underlying
    ``dispatch_groups`` already fans out per-LLM-call concurrency
    internally, so two simultaneous batch jobs would otherwise hammer
    the LLM endpoint.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, _CorrectionJob] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="correction-job",
        )

    def create(
        self, run_id: str, targets: List[Tuple[str, str, str]],
    ) -> _CorrectionJob:
        job = _CorrectionJob(
            job_id=uuid.uuid4().hex[:16],
            run_id=run_id,
            targets=list(targets),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[_CorrectionJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(
        self, job: _CorrectionJob, fn,
    ) -> None:
        """Enqueue ``fn(job)`` on the background executor."""
        self._executor.submit(fn, job)


# Single registry per process. ``create_app`` doesn't know about
# multi-tenancy, so a module-global is fine here (mirrors how
# ``RunCache`` and the launcher's ``JobManager`` are scoped).
_correction_jobs = _CorrectionJobRegistry()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    project_root: Path,
    *,
    annotator: str,
    static_dir: Optional[Path] = None,
) -> FastAPI:
    project_root = Path(project_root).resolve()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if static_dir is None:
        static_dir = project_root / "frontend" / "labeling"
    static_dir = Path(static_dir).resolve()
    if not static_dir.is_dir():
        raise FileNotFoundError(f"static dir not found: {static_dir}")

    registry = RunRegistry(outputs_dir)
    jobs = JobManager(project_root)
    cache = RunCache(outputs_dir)

    # Resolve once at app-creation time so every request can compare
    # resolved sub-paths against this root cheaply.
    _outputs_root = outputs_dir.resolve()

    def _validate_run_id(run_id: str) -> None:
        """Reject run_ids that could escape ``outputs_dir`` via
        path-traversal, separator injection, or NUL bytes."""
        if (
            not run_id
            or run_id in (".", "..")
            or "/" in run_id
            or "\\" in run_id
            or "\x00" in run_id
        ):
            raise HTTPException(400, f"invalid run_id: {run_id!r}")

    def _safe_run_dir(run_id: str) -> Path:
        """Return the resolved run directory, asserting it lives
        directly under ``outputs_dir``. Symlinked run dirs whose target
        escapes the outputs root are rejected to prevent
        ``shutil.rmtree`` from following them outside the workbench."""
        _validate_run_id(run_id)
        rd = (outputs_dir / run_id).resolve()
        try:
            rd.relative_to(_outputs_root)
        except ValueError as exc:
            raise HTTPException(400, f"invalid run_id: {run_id!r}") from exc
        if rd == _outputs_root or rd.parent != _outputs_root:
            raise HTTPException(400, f"invalid run_id: {run_id!r}")
        return rd

    def _run_dir_or_404(run_id: str) -> Path:
        rd = _safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        return rd

    app = FastAPI(title="Data-cleaning workbench", docs_url="/api/docs")

    # Defense-in-depth CSP. The labelling UI is self-hosted (modules
    # under /ui/, fetches under /api/), so a strict same-origin policy
    # blocks any future inline-script regression and any third-party
    # asset load. Inline ``style="..."`` attributes are still in use,
    # so style-src keeps 'unsafe-inline'.
    _CSP = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "base-uri 'none'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )

    @app.middleware("http")
    async def _add_csp_header(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", _CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    # =====================================================================
    # Register routers (split out of this file - see src/labeling/routers/)
    # =====================================================================

    router_ctx = RouterContext(
        project_root=project_root,
        outputs_dir=outputs_dir,
        static_dir=static_dir,
        annotator=annotator,
        registry=registry,
        jobs=jobs,
        cache=cache,
    )
    launcher_router.register(app, router_ctx)
    cell_notes_router.register(app, router_ctx)
    exclusions_router.register(app, router_ctx)
    exports_router.register(app, router_ctx)
    multi_table_router.register(app, router_ctx)
    sources_router.register(app, router_ctx)
    stats_router.register(app, router_ctx)
    relation_reviews_router.register(app, router_ctx)
    compare_router.register(app, router_ctx)

    # =====================================================================
    # Runs listing + per-run endpoints
    # =====================================================================

    @app.get("/api/runs")
    def list_runs() -> Dict[str, Any]:
        return {
            "project_root": str(project_root),
            "annotator": annotator,
            "runs": [r.to_dict() for r in registry.list()],
        }

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str, force: bool = False) -> Dict[str, Any]:
        # _safe_run_dir enforces both syntactic checks AND that the
        # resolved path is a direct child of outputs_dir, so a symlink
        # in outputs/<id> can't redirect rmtree outside the workbench.
        run_dir = _safe_run_dir(run_id)
        if not run_dir.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        # Server-side safety net: starred runs are explicitly opted
        # out of accidental deletion. The dashboard re-confirms with
        # the operator before sending ``?force=true``; CLI / scripted
        # deletes must also pass it. Without this guard a careless
        # bulk-delete request could wipe a pinned run.
        if not force:
            meta = _load_run_metadata(run_dir)
            if meta.get("starred"):
                raise HTTPException(
                    409,
                    f"run {run_id!r} is starred; pass ?force=true to confirm",
                )
        cache.invalidate(run_id)
        import shutil
        shutil.rmtree(run_dir)
        return {"ok": True, "run_id": run_id}

    @app.patch("/api/runs/{run_id}")
    def patch_run(run_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Update operator-editable run metadata (display_name, starred,
        notes). Body keys are merged with existing sidecar - unknown
        keys are silently dropped. Returns the run's full updated
        ``RunInfo`` payload so the client can replace the row in place.
        """
        run_dir = _run_dir_or_404(run_id)
        try:
            _save_run_metadata(run_dir, payload or {})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        # Re-parse so the response carries the merged metadata fields
        # already attached to the rest of the run's RunInfo (counts,
        # artifact flags, ...). The dashboard replaces the row in
        # place from this payload, no extra refresh needed.
        info = registry.get(run_id)
        if info is None:
            raise HTTPException(404, f"run not found: {run_id}")
        return info.to_dict()


    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        by_source: Dict[str, int] = {}
        for c in ctx.clusters:
            src = c.get("source", "")
            by_source[src] = by_source.get(src, 0) + 1

        # Per-source label coverage so the UI can flag which CSVs the
        # operator has touched. ``labelled_clusters`` counts clusters
        # that received any verdict (tp/fp/partial/unsure/skip).
        # ``fn_cells`` counts CELL-target labels currently set to FN.
        # ``checked`` is the OR of the two - "this CSV has been seen".
        latest = ctx.store.latest_by_target()
        cluster_to_source: Dict[str, str] = {
            c.get("cluster_id", ""): c.get("source", "") or ""
            for c in ctx.clusters
        }
        labelled_clusters: Dict[str, int] = {}
        fn_cells: Dict[str, int] = {}
        for tid, lab in latest.items():
            if lab.target == LabelTarget.CLUSTER:
                src = cluster_to_source.get(tid, "")
                if src:
                    labelled_clusters[src] = labelled_clusters.get(src, 0) + 1
            elif lab.target == LabelTarget.CELL and lab.status == LabelStatus.FN:
                src = lab.source or ""
                if src:
                    fn_cells[src] = fn_cells.get(src, 0) + 1

        # Group sources by their parent-dir basename (= workbook). We
        # only surface the grouping when there are at least 2 distinct
        # workbooks AND at least one of them owns 2+ CSVs - otherwise
        # the grouping is either redundant (every CSV is its own
        # "workbook") or trivial (single corpus). The verdict is
        # pre-computed on the context so it stays consistent with the
        # per-source ``siblings`` field.
        workbook_to_sources: Dict[str, List[str]] = {}
        for name in sorted(ctx.sources):
            wb = ctx.source_workbook.get(name, "") or ""
            workbook_to_sources.setdefault(wb, []).append(name)
        multi_workbook = ctx.multi_workbook

        workbooks_payload: List[Dict[str, Any]] = []
        if multi_workbook:
            for wb in sorted(workbook_to_sources):
                names = workbook_to_sources[wb]
                workbooks_payload.append({
                    "name": wb,
                    "sources": names,
                    "clusters": sum(by_source.get(n, 0) for n in names),
                    "has_xlsx": ctx.workbook_xlsx_path(wb) is not None,
                })

        # Cache the per-workbook xlsx availability so the per-source
        # entries below are an O(1) lookup rather than a stat-per-row.
        wb_xlsx_avail: Dict[str, bool] = {
            wb: (ctx.workbook_xlsx_path(wb) is not None)
            for wb in workbook_to_sources
        }

        return {
            "run_id": run_id,
            "run_dir": str(ctx.run_dir),
            "annotator": annotator,
            "summary": ctx.summary,
            "sources": [
                {
                    "name": name,
                    "path": str(path),
                    "clusters": by_source.get(name, 0),
                    "labelled_clusters": labelled_clusters.get(name, 0),
                    "fn_cells": fn_cells.get(name, 0),
                    "checked": bool(
                        labelled_clusters.get(name, 0) or fn_cells.get(name, 0)
                    ),
                    "available": path.exists(),
                    "table_group": ctx.source_workbook.get(name, "") or "",
                    "has_xlsx": wb_xlsx_avail.get(
                        ctx.source_workbook.get(name, "") or "", False
                    ),
                }
                for name, path in sorted(ctx.sources.items())
            ],
            "workbooks": workbooks_payload,
            "multi_workbook": multi_workbook,
            "xlsx_source_dir": (
                str(ctx.xlsx_source_dir) if ctx.xlsx_source_dir else None
            ),
            "xlsx_available": any(wb_xlsx_avail.values()),
            "counts": {
                "clusters": len(ctx.clusters),
                "issues": len(ctx.issues),
                # Multi-table findings (one per ``violation_id``) live
                # in ``multi_table_violations.jsonl`` and are surfaced
                # alongside clusters in the unified label queue. Empty
                # for runs that didn't enable the cross-sheet pass.
                "multi_table_findings": _count_multi_table_findings(
                    ctx.run_dir,
                ),
            },
        }

    @app.get("/api/runs/{run_id}/clusters")
    def list_clusters(
        run_id: str,
        source: Optional[str] = None,
        workbook: Optional[str] = None,
        status: Optional[str] = None,
        error_type: Optional[str] = None,
        limit: int = Query(500, ge=1, le=5000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        latest = ctx.store.latest_by_target()
        filtered: List[Dict[str, Any]] = []

        for c in ctx.clusters:
            if source and c.get("source") != source:
                continue
            if workbook:
                src_name = c.get("source", "") or ""
                if ctx.source_workbook.get(src_name, "") != workbook:
                    continue
            if error_type and error_type not in (c.get("error_types") or []):
                continue
            label = latest.get(c.get("cluster_id", ""))
            label_status = label.status.value if label else "unlabeled"
            if status and status != label_status:
                continue

            item = _cluster_summary(ctx.canonical_cluster(c.get("cluster_id", "")) or c)
            item["label_status"] = label_status
            if label is not None:
                item["corrected_value"] = label.corrected_value
                item["note"] = label.note
                item["annotator"] = label.annotator
                item["ts"] = label.ts
            filtered.append(item)

        # Dedup: collapse clusters that resolve to the SAME canonical
        # (source, row, frozenset(columns)) key. This is the historical
        # version of the same fix that ``flatten_issues`` now applies
        # pre-clustering: the LLM occasionally returns differently-
        # mangled forms of the same multi-line header, the aggregator
        # clusters on raw strings (so it doesn't merge them), and the
        # serve-time canonicalizer maps both back to the actual frame
        # column - producing apparent "two clusters on the same cell"
        # in the UI. Picking a primary keeps existing labels visible
        # while the duplicates remain reachable by direct cluster_id
        # lookup (so prior labels on absorbed clusters never disappear).
        primary_by_key: Dict[Any, Dict[str, Any]] = {}
        absorbed_count = 0
        for item in filtered:
            cols = tuple(sorted(item.get("columns") or []))
            row = item.get("row")
            key = (item.get("source") or "", row, cols)
            # Only dedup when the column set is non-empty AND the row
            # is concrete; row-less / column-less defects are dataset-
            # wide flags that legitimately exist as separate clusters.
            if not cols or not isinstance(row, int):
                primary_by_key[id(item)] = item
                continue
            existing = primary_by_key.get(key)
            if existing is None:
                primary_by_key[key] = item
                continue
            absorbed_count += 1
            # Pick the better cluster as the primary. Preference order:
            #   1. labelled > unlabeled (don't hide an annotator's work),
            #   2. higher evidence_count (more corroboration),
            #   3. lower cluster_id (stable, deterministic).
            def _rank(it: Dict[str, Any]) -> tuple:
                labelled = (it.get("label_status") or "unlabeled") != "unlabeled"
                return (
                    not labelled,
                    -int(it.get("evidence_count") or 0),
                    str(it.get("cluster_id") or ""),
                )
            if _rank(item) < _rank(existing):
                primary_by_key[key] = item
                # Track absorbed predecessor so the UI can show "+N more".
                merged = list(existing.get("merged_cluster_ids") or [])
                merged.append(existing.get("cluster_id"))
                merged.extend(item.get("merged_cluster_ids") or [])
                item["merged_cluster_ids"] = [m for m in merged if m]
            else:
                merged = list(existing.get("merged_cluster_ids") or [])
                merged.append(item.get("cluster_id"))
                merged.extend(item.get("merged_cluster_ids") or [])
                existing["merged_cluster_ids"] = [m for m in merged if m]
        # Rebuild the filtered list from the dedup'd values, preserving
        # whatever ordering hash-iteration gives us; the explicit sort
        # below is the authoritative final order.
        filtered = list(primary_by_key.values())
        if absorbed_count:
            logger.debug(
                "queue-dedup: collapsed %d duplicate cluster(s) on run=%s",
                absorbed_count, run_id,
            )

        # Sort by location so the operator sees clusters in the same
        # order they'd read the data: workbook (parent dir / table_group)
        # first so siblings of one logical workbook are adjacent, then
        # CSV name, then row index, then the first flagged column's
        # position in the CSV (so the cursor walks left-to-right within
        # a row instead of jumping around).
        col_pos_cache: Dict[str, Dict[str, int]] = {}

        def _first_col_pos(item_: Dict[str, Any]) -> int:
            cols = item_.get("columns") or []
            if not cols:
                return -1
            src = item_.get("source") or ""
            pos_map = col_pos_cache.get(src)
            if pos_map is None:
                try:
                    headers = list(ctx.load_frame(src).columns) if src else []
                except FileNotFoundError:
                    headers = []
                pos_map = {h: i for i, h in enumerate(headers)}
                col_pos_cache[src] = pos_map
            return min(
                (pos_map.get(c, len(pos_map)) for c in cols),
                default=len(pos_map),
            )

        def _sort_key(item_: Dict[str, Any]):
            src = item_.get("source") or ""
            wb = ctx.source_workbook.get(src, "") or ""
            row = item_.get("row")
            return (
                wb,
                src,
                row if isinstance(row, int) else 1 << 30,
                _first_col_pos(item_),
                item_.get("cluster_id", ""),
            )

        filtered.sort(key=_sort_key)

        return {
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "items": filtered[offset : offset + limit],
        }

    @app.get("/api/runs/{run_id}/clusters/{cluster_id}")
    def get_cluster(run_id: str, cluster_id: str) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        c = ctx.canonical_cluster(cluster_id)
        if c is None:
            raise HTTPException(404, f"cluster not found: {cluster_id}")
        label = ctx.store.latest_for(cluster_id)
        return {
            "cluster": c,
            "label": label.model_dump() if label else None,
            "history": [h.model_dump() for h in ctx.store.history(cluster_id)],
            "web_verification": ctx.verification(cluster_id),
        }

    @app.get("/api/runs/{run_id}/clusters/{cluster_id}/web-verification")
    def get_web_verification(
        run_id: str, cluster_id: str,
    ) -> Dict[str, Any]:
        """Return the persisted web-verification record for one cluster.

        ``404`` when the verifier has never run on this cluster (the
        operator can still label manually). Run ``python -m
        experiments.run_web_verify --run-id <run>`` to populate the
        store.
        """
        ctx = cache.get(run_id)
        if ctx.cluster(cluster_id) is None:
            raise HTTPException(404, f"cluster not found: {cluster_id}")
        ver = ctx.verification(cluster_id)
        if ver is None:
            raise HTTPException(
                404,
                "no web verification on file for this cluster "
                "(run experiments.run_web_verify to populate)",
            )
        return ver

    @app.post("/api/runs/{run_id}/clusters/{cluster_id}/web-verification")
    def create_web_verification(
        run_id: str, cluster_id: str,
        retry: bool = Query(
            False,
            description=(
                "When ``true``, replace an existing **errored** "
                "verification record (e.g. an Azure agent timeout) "
                "instead of returning 409. Has no effect when the "
                "prior record is non-errored - those still 409 to "
                "prevent accidental overwrite."
            ),
        ),
    ) -> Dict[str, Any]:
        """Trigger a single-cluster web-verification on demand.

        Used by the "Verify on web" button in the labeling UI for
        clusters the batch verifier didn't pick up. Synchronous - the
        UI shows a spinner while we make the Foundry agent call
        (typically 5-30 s per cluster, depending on how many web
        searches the agent decides to run). Returns the new
        verification record on success.

        Status codes:
        * ``200`` - verification produced (verdict may be any of
          confirmed_error / false_positive / ambiguous / errored).
        * ``404`` - cluster not found in this run.
        * ``409`` - cluster already has a non-errored verification
          (UI bug or double-click; refuse to clobber). For an
          errored prior record, the UI may pass ``?retry=true`` to
          force a re-run.
        * ``422`` - cluster is gate-rejected (PII column / non-entity
          column with no whitelisted error_type / unsuitable value).
        * ``500`` - missing agent config, prompt file unreadable, etc.
        """
        ctx = cache.get(run_id)
        if ctx.cluster(cluster_id) is None:
            raise HTTPException(404, f"cluster not found: {cluster_id}")
        # Importing inside the handler keeps the labeling-app
        # importable even when the verifier package has unmet
        # dependencies (the only network-side dep is azure-ai-agents,
        # which is required by requirements.txt; be defensive anyway).
        try:
            from experiments.run_web_verify import (  # noqa: WPS433
                VerifyError,
                verify_single_cluster,
            )
        except ImportError as exc:
            raise HTTPException(500, f"verifier unavailable: {exc}")
        try:
            # ``verify_single_cluster`` defaults to
            # ``include_person=True`` so person-name columns flow
            # through with the lenient "valid if anyone has the
            # name" criterion. See ``GateConfig.verify_person`` for
            # the underlying gate.
            record = verify_single_cluster(
                ctx.run_dir, cluster_id,
                retry_errored=retry,
            )
        except VerifyError as exc:
            code_map = {
                "not_found": 404,
                "already_verified": 409,
                "gate_rejected": 422,
                "internal": 500,
            }
            raise HTTPException(
                code_map.get(exc.code, 500),
                exc.message,
            )
        except FileNotFoundError as exc:
            # Most likely a missing prompt file; surface as 500.
            raise HTTPException(500, str(exc))
        # ``verify_single_cluster`` already wrote the record to disk
        # atomically. ctx.verification() re-reads via mtime check and
        # returns the canonical dict shape (matching what the GET
        # endpoint serves).
        result = ctx.verification(cluster_id)
        if result is None:
            # Should never happen - the file was just written - but
            # fall back to converting the in-memory record so the
            # client gets *something*.
            result = {
                "cluster_id": record.cluster_id,
                "verdict": record.verdict,
                "canonical_value": record.canonical_value,
                "confidence": record.confidence,
                "error": record.error,
            }
        return result

    # -------- column-cluster value-corrections (LLM suggest + persist) --
    #
    # Live counterpart to the offline
    # ``experiments.export_inconsistency_benchmark_with_llm`` script.
    # The shared logic lives in :mod:`src.labeling.correction_suggester`
    # so the on-disk LLM cache is binary-compatible between the two
    # entry points - re-running the export after the operator has
    # used these endpoints will pick up cached samples for free.
    #
    # The persisted output is a per-cluster ``value_corrections``
    # mapping on the latest column-cluster Label record (see
    # :class:`LabelIn.value_corrections`). Operators may save partial
    # mappings; an absent key means "drop this value from the
    # benchmark export" rather than "use LLM suggestion".

    def _parse_cluster_target(cluster_id: str) -> Tuple[str, str, str]:
        """Decode the ``colclust\\x00src\\x00col\\x00cN`` target.

        Raises ``HTTPException(400)`` for malformed payloads so the
        endpoint can short-circuit with a friendly error.
        """
        parts = correction_suggester.parse_colclust_target_id(cluster_id)
        if parts is None:
            raise HTTPException(
                400,
                f"invalid column-cluster id: {cluster_id!r} "
                f"(expected 'colclust\\x00src\\x00col\\x00cN')",
            )
        return parts

    def _suggestions_to_payload(
        suggestions: List[correction_suggester.ValueSuggestion],
    ) -> List[Dict[str, Any]]:
        """Trim a list of ``ValueSuggestion`` to the shape the UI wants.

        We hide the per-sample raw lists (heavy, only useful for
        diagnostic export) and keep just the consensus + agreement +
        provenance hints needed to render the table.
        """
        out: List[Dict[str, Any]] = []
        for s in suggestions:
            out.append({
                "value": s.value,
                "suggestion": s.suggestion,
                "agreement": s.agreement,
                "n_valid_samples": s.n_valid_samples,
                "n_total_samples": s.n_total_samples,
                "would_drop_per_export_rule": s.would_drop_per_export_rule,
                "rows": s.rows,
            })
        return out

    def _saved_value_corrections_for(
        ctx: "RunContext", target_id: str,
    ) -> Dict[str, Union[str, List[str]]]:
        """Fetch the latest persisted ``value_corrections`` for a target.

        Empty dict when the cluster has no label record yet, or its
        latest record carries an empty mapping.
        """
        latest = ctx.store.latest_for(target_id)
        if latest is None:
            return {}
        return dict(latest.value_corrections or {})

    @app.post(
        "/api/runs/{run_id}/clusters/{cluster_id}/suggest-corrections"
    )
    def suggest_cluster_corrections(
        run_id: str, cluster_id: str,
        payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Dict[str, Any]:
        """Run the LLM correction suggester for ONE column-cluster.

        Synchronous; typically completes in a few seconds when the
        cache is warm and 30-90 s on a cold cluster (K=3 + retry-4).
        The UI uses this for the per-cluster "Generate corrections"
        button.

        Body fields (all optional):
          * ``samples`` (int, default 3) - K self-consistency samples.
          * ``samples_threshold`` (float, default 0.5) - vote frac.
          * ``retry_samples`` (int, default 4) - extra samples when
            consensus = [original_value].
          * ``model`` (str, default ``"gpt-5.2"``).
          * ``reasoning_effort`` (str, default ``"medium"``).
          * ``row_context_rows`` (int, default 3).

        Returns ``{"suggestions": [...], "saved_value_corrections": {...}}``
        where each suggestion describes one (value, rows) tuple in the
        cluster.
        """
        ctx = cache.get(run_id)
        s, c, lid = _parse_cluster_target(cluster_id)
        if s not in ctx.sources:
            raise HTTPException(404, f"source not found: {s}")

        samples = int(payload.get("samples", 3))
        samples_threshold = float(payload.get("samples_threshold", 0.5))
        retry_samples = int(payload.get("retry_samples", 4))
        model = str(payload.get("model", "gpt-5.2"))
        reasoning_effort = str(payload.get("reasoning_effort", "medium"))
        row_context_rows = int(payload.get("row_context_rows", 3))

        try:
            llm_client = LLMClient()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"LLM client unavailable: {exc}")

        try:
            results = correction_suggester.suggest_corrections_for_clusters(
                ctx.run_dir,
                targets=[(s, c, lid)],
                llm_client=llm_client,
                model=model,
                reasoning_effort=reasoning_effort,
                samples=samples,
                samples_threshold=samples_threshold,
                retry_samples=retry_samples,
                row_context_rows=row_context_rows,
                concurrency=8,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                404,
                f"missing run artifact for correction suggestion: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "suggest_corrections failed for run=%s cluster=%s",
                run_id, cluster_id,
            )
            raise HTTPException(500, f"correction suggestion failed: {exc}")

        suggestions = results.get((s, c, lid), [])
        if not suggestions:
            raise HTTPException(
                404,
                f"no draft cluster matched target {cluster_id!r} "
                f"(check column_cluster_drafts.jsonl in this run)",
            )
        return {
            "cluster_id": cluster_id,
            "source": s,
            "column": c,
            "cluster_local_id": lid,
            "suggestions": _suggestions_to_payload(suggestions),
            "saved_value_corrections": _saved_value_corrections_for(
                ctx, cluster_id,
            ),
        }

    @app.get(
        "/api/runs/{run_id}/clusters/{cluster_id}/corrections"
    )
    def get_cluster_corrections(
        run_id: str, cluster_id: str,
    ) -> Dict[str, Any]:
        """Return the latest persisted ``value_corrections`` for a cluster
        plus any cached LLM suggestions (no LLM calls).

        The UI calls this when opening a cluster: if
        ``saved_value_corrections`` is non-empty, the UI shows the
        operator's curated mapping; if it is empty AND
        ``cached_suggestions`` are available, the UI offers a one-click
        "Use cached suggestions" affordance. Otherwise the operator
        must invoke ``POST /suggest-corrections`` to spend LLM budget.
        """
        ctx = cache.get(run_id)
        s, c, lid = _parse_cluster_target(cluster_id)
        if s not in ctx.sources:
            raise HTTPException(404, f"source not found: {s}")

        saved = _saved_value_corrections_for(ctx, cluster_id)

        # Cache-only path: pass dry_run=True so dispatch_groups returns
        # whatever lives in inconsistency_llm_cache.jsonl without ever
        # constructing an LLM client.
        cached_suggestions: List[Dict[str, Any]] = []
        try:
            results = correction_suggester.suggest_corrections_for_clusters(
                ctx.run_dir,
                targets=[(s, c, lid)],
                llm_client=None,
                samples=3,
                samples_threshold=0.5,
                retry_samples=4,
                row_context_rows=3,
                concurrency=1,
                dry_run=True,
            )
            cached_suggestions = _suggestions_to_payload(
                results.get((s, c, lid), []),
            )
        except FileNotFoundError:
            # Run still missing draft artifact - that's fine; saved
            # corrections (if any) are still returned below.
            cached_suggestions = []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_cluster_corrections cache-replay failed for "
                "run=%s cluster=%s: %s",
                run_id, cluster_id, exc,
            )
            cached_suggestions = []

        return {
            "cluster_id": cluster_id,
            "source": s,
            "column": c,
            "cluster_local_id": lid,
            "saved_value_corrections": saved,
            "cached_suggestions": cached_suggestions,
        }

    @app.post(
        "/api/runs/{run_id}/clusters/{cluster_id}/corrections"
    )
    def save_cluster_corrections(
        run_id: str, cluster_id: str,
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """Persist a ``value_corrections`` mapping for one cluster.

        Body fields:
          * ``value_corrections`` (dict, required) - mapping from each
            original surface form in the cluster to its correction
            (scalar = one-direction, list = multi-direction). An ABSENT
            key means "drop this value from the benchmark export".
          * ``note_tag`` (str, optional, default ``"[manual]"``) -
            short prefix appended to the existing note for audit trail.
            Common values: ``"[llm-suggested]"`` (operator accepted the
            LLM consensus verbatim), ``"[llm-edited]"`` (operator edited
            the LLM consensus), ``"[manual]"`` (operator typed the
            corrections from scratch).

        Builds a fresh Label record from the latest cluster verdict's
        fields (status / source / column / columns / confirmed_canonical /
        note) plus the new ``value_corrections`` and the appended
        ``note_tag``. Requires a prior verdict in {tp, unsure}; saving
        corrections on an FP/skip cluster is rejected.
        """
        ctx = cache.get(run_id)
        s, c, lid = _parse_cluster_target(cluster_id)
        if s not in ctx.sources:
            raise HTTPException(404, f"source not found: {s}")

        raw = payload.get("value_corrections")
        if not isinstance(raw, dict):
            raise HTTPException(
                400, "value_corrections must be a JSON object",
            )
        # Coerce + validate: keys must be strings, values must be
        # str | list[str]; reject any other shape so we don't silently
        # store junk that the export pipeline later mishandles.
        normalized: Dict[str, Union[str, List[str]]] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                raise HTTPException(
                    400, f"value_corrections keys must be strings, got {k!r}",
                )
            if isinstance(v, str):
                normalized[k] = v
            elif isinstance(v, list):
                if not all(isinstance(x, str) for x in v):
                    raise HTTPException(
                        400,
                        f"value_corrections[{k!r}] list elements must be strings",
                    )
                normalized[k] = list(v)
            else:
                raise HTTPException(
                    400,
                    f"value_corrections[{k!r}] must be string or list of strings, "
                    f"got {type(v).__name__}",
                )

        latest = ctx.store.latest_for(cluster_id)
        if latest is None:
            raise HTTPException(
                404,
                f"cluster {cluster_id!r} has no prior verdict; "
                f"label it tp/unsure before saving corrections",
            )
        if latest.status not in (LabelStatus.TP, LabelStatus.UNSURE):
            raise HTTPException(
                400,
                f"cluster verdict is {latest.status!r}; "
                f"corrections are only valid for status in {{tp, unsure}}",
            )

        note_tag = str(payload.get("note_tag") or "[manual]").strip() or "[manual]"
        prior_note = (latest.note or "").strip()
        if note_tag in prior_note:
            new_note = prior_note
        elif prior_note:
            new_note = f"{note_tag}; {prior_note}"
        else:
            new_note = note_tag

        new_label = Label(
            target=LabelTarget.COLUMN_CLUSTER,
            target_id=cluster_id,
            source=latest.source,
            row=latest.row,
            column=latest.column,
            columns=list(latest.columns),
            status=latest.status,
            corrected_value=latest.corrected_value,
            corrected_values=dict(latest.corrected_values),
            confirmed_canonical=latest.confirmed_canonical,
            value_corrections=normalized,
            note=new_note,
            annotator=(payload.get("annotator") or annotator).strip()
                or annotator,
        )
        ctx.store.append(new_label)
        return {
            "cluster_id": cluster_id,
            "saved_value_corrections": normalized,
            "note": new_note,
        }

    # -------- batch correction-suggestion job (async) ------------------
    #
    # Enumerates every TP/unsure column-cluster in the run (optionally
    # filtered to one source), kicks off the same shared dispatcher in
    # a background thread, and returns a job-id the UI can poll. Cache
    # hits make most batches finish in seconds; cold runs may take
    # several minutes.

    @app.post("/api/runs/{run_id}/correction-jobs")
    def start_correction_job(
        run_id: str,
        payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Dict[str, Any]:
        """Start a background correction-suggestion job.

        Body fields (all optional):
          * ``source`` (str) - restrict the job to this source CSV.
            Default: every source in the run.
          * ``samples``, ``samples_threshold``, ``retry_samples``,
            ``model``, ``reasoning_effort``, ``row_context_rows`` -
            same defaults as ``suggest-corrections``.

        Returns ``{"job_id", "total_targets", "status": "queued"}``.
        Poll ``GET /correction-jobs/{job_id}`` for progress + final
        results.
        """
        ctx = cache.get(run_id)
        source_filter = payload.get("source")
        if source_filter is not None:
            if not isinstance(source_filter, str):
                raise HTTPException(400, "source filter must be a string")
            if source_filter not in ctx.sources:
                raise HTTPException(404, f"source not found: {source_filter}")

        # Walk every saved column-cluster Label record and pick out
        # those with a TP/unsure verdict. We iterate the LabelStore
        # rather than re-parsing drafts so we naturally get only the
        # clusters the operator has actually reviewed.
        seen: set = set()
        targets: List[Tuple[str, str, str]] = []
        for lab in ctx.store.all():
            if lab.target != LabelTarget.COLUMN_CLUSTER:
                continue
            if lab.status not in (LabelStatus.TP, LabelStatus.UNSURE):
                continue
            tid = lab.target_id
            if tid in seen:
                continue  # latest-wins; first hit per id is the latest
            seen.add(tid)
            parsed = correction_suggester.parse_colclust_target_id(tid)
            if parsed is None:
                continue
            s, c, lid = parsed
            if source_filter is not None and s != source_filter:
                continue
            # latest_for is order-independent; use it as the source of truth
            latest = ctx.store.latest_for(tid)
            if latest is None or latest.status not in (
                LabelStatus.TP, LabelStatus.UNSURE,
            ):
                continue
            targets.append((s, c, lid))

        if not targets:
            raise HTTPException(
                404,
                "no TP/unsure column-cluster verdicts in this run "
                + (f"for source {source_filter!r}" if source_filter else ""),
            )

        samples = int(payload.get("samples", 3))
        samples_threshold = float(payload.get("samples_threshold", 0.5))
        retry_samples = int(payload.get("retry_samples", 4))
        model = str(payload.get("model", "gpt-5.2"))
        reasoning_effort = str(payload.get("reasoning_effort", "medium"))
        row_context_rows = int(payload.get("row_context_rows", 3))

        job = _correction_jobs.create(run_id, targets)

        def _run(job: _CorrectionJob) -> None:
            job.status = "running"
            job.started_at = time.time()

            def _progress(phase: str, done: int, total: int) -> None:
                # Dispatch fires this once per finished LLM call; the
                # registry is module-global so writes are visible to
                # the GET endpoint without further locking (single
                # writer per job).
                job.phase = phase
                job.done = done
                job.total = total

            try:
                llm_client = LLMClient()
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"LLM client unavailable: {exc}"
                job.finished_at = time.time()
                return

            try:
                results = correction_suggester.suggest_corrections_for_clusters(
                    ctx.run_dir,
                    targets=job.targets,
                    llm_client=llm_client,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    samples=samples,
                    samples_threshold=samples_threshold,
                    retry_samples=retry_samples,
                    row_context_rows=row_context_rows,
                    concurrency=8,
                    progress_cb=_progress,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "correction batch job failed run=%s job=%s",
                    job.run_id, job.job_id,
                )
                job.status = "error"
                job.error = str(exc)
                job.finished_at = time.time()
                return

            for (s, c, lid), suggs in results.items():
                key = f"{s}\x00{c}\x00{lid}"
                job.results[key] = _suggestions_to_payload(suggs)
            job.status = "done"
            job.finished_at = time.time()

        _correction_jobs.submit(job, _run)
        return {
            "job_id": job.job_id,
            "total_targets": len(targets),
            "status": job.status,
        }

    @app.get("/api/runs/{run_id}/correction-jobs/{job_id}")
    def get_correction_job(run_id: str, job_id: str) -> Dict[str, Any]:
        """Return job state for polling. Result payload is included in
        the same response when ``status == "done"``."""
        job = _correction_jobs.get(job_id)
        if job is None or job.run_id != run_id:
            raise HTTPException(404, f"correction job not found: {job_id}")
        return {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "status": job.status,
            "phase": job.phase,
            "done": job.done,
            "total": job.total,
            "total_targets": len(job.targets),
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "results": job.results if job.status == "done" else {},
        }

    @app.get("/api/runs/{run_id}/web-verifications")
    def list_web_verifications(run_id: str) -> Dict[str, Any]:
        """Aggregate counts + the full list. The list is small enough
        (one row per gate-eligible cluster) that the labeling UI can
        cache it client-side and decide whether to render the
        verification block per cluster without a second round-trip."""
        ctx = cache.get(run_id)
        ctx._reload_verifications()  # noqa: SLF001  (intentional)
        records = list(ctx._verifications.values())  # noqa: SLF001
        counts = {"confirmed_error": 0, "false_positive": 0, "ambiguous": 0, "errored": 0}
        for r in records:
            if r.get("error"):
                counts["errored"] += 1
                continue
            v = str(r.get("verdict") or "ambiguous")
            counts[v] = counts.get(v, 0) + 1
        return {
            "available": ctx._verifications_path().exists(),  # noqa: SLF001
            "counts": counts,
            "items": records,
        }

    # -------- column-wide web verification (column-audit) ----------
    #
    # Auxiliary artifacts written by ``experiments.run_web_verify_column``.
    # One JSONL per audited (source, column) under
    # ``outputs/<run_id>/column_verifications/`` plus a ``_index.jsonl``
    # manifest. The endpoints below are read-only; the audit itself is
    # launched via POST /api/launcher/jobs (kind="column_verify") so it
    # benefits from the existing job-runner / log-tail UI.

    def _column_verify_dir(rd: Path) -> Path:
        return rd / "column_verifications"

    def _column_verify_path(rd: Path, source: str, column: str) -> Path:
        # File-name slug must match ``run_web_verify_column._slug`` so
        # the UI can predict the on-disk path before issuing the read.
        from experiments.run_web_verify_column import (  # noqa: WPS433
            column_artifact_path,
        )
        return column_artifact_path(rd, source, column)

    def _read_column_verify_jsonl(path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not path.exists():
            return out
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        out.append(rec)
        except OSError:
            return []
        return out

    @app.get("/api/runs/{run_id}/column-verifications")
    def list_column_verifications(run_id: str) -> Dict[str, Any]:
        """Manifest of every (source, column) audited by the column-wide
        verifier.

        Reads ``outputs/<run_id>/column_verifications/_index.jsonl``
        (rebuilt on every audit run by ``run_web_verify_column``).
        Falls back to scanning the directory when the manifest is
        missing (e.g. someone moved files in by hand).

        Response::

            {
              "available": bool,
              "items": [
                {"source", "column", "path", "counts": {...}, "ts"},
                ...
              ]
            }
        """
        rd = _run_dir_or_404(run_id)
        base = _column_verify_dir(rd)
        if not base.is_dir():
            return {"available": False, "items": []}
        index_path = base / "_index.jsonl"
        items: List[Dict[str, Any]] = []
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(rec, dict):
                            items.append(rec)
            except OSError as exc:
                logger.warning(
                    "column-verifications index unreadable for %s: %s",
                    run_id, exc,
                )
        if not items:
            # Fallback: synthesise the manifest on the fly. Cheap when
            # there are only a handful of files; the per-file read is
            # the same the dedicated endpoint does.
            for p in sorted(base.glob("*.jsonl")):
                if p.name.startswith("_"):
                    continue
                recs = _read_column_verify_jsonl(p)
                if not recs:
                    continue
                first = recs[0]
                counts = {
                    "confirmed_error": 0,
                    "false_positive": 0,
                    "ambiguous": 0,
                    "errored": 0,
                    "fn_candidates": 0,
                    "total": len(recs),
                }
                for r in recs:
                    if r.get("error"):
                        counts["errored"] += 1
                        continue
                    v = str(r.get("verdict") or "ambiguous")
                    counts[v] = counts.get(v, 0) + 1
                    if v == "confirmed_error" and not r.get("was_flagged_by_llm"):
                        counts["fn_candidates"] += 1
                items.append({
                    "source": first.get("source") or "",
                    "column": first.get("column") or "",
                    "path": str(p.relative_to(rd)),
                    "counts": counts,
                    "ts": max(
                        (str(r.get("ts") or "") for r in recs),
                        default="",
                    ),
                })
        return {"available": True, "items": items}

    @app.get("/api/runs/{run_id}/column-verifications/detail")
    def get_column_verification(
        run_id: str,
        source: str = Query(..., min_length=1),
        column: str = Query(..., min_length=1),
    ) -> Dict[str, Any]:
        """Return every audit record for one (source, column).

        ``source`` + ``column`` are query params (not path segments)
        because column headers may legitimately contain ``/`` and other
        characters URL-decoded path templating doesn't handle cleanly.

        Response::

            {
              "available": bool,
              "source": str,
              "column": str,
              "counts": {confirmed_error, false_positive, ambiguous,
                         errored, fn_candidates, total},
              "items": [<column-audit record>, ...]
            }
        """
        _validate_source_name(source)
        rd = _run_dir_or_404(run_id)
        path = _column_verify_path(rd, source, column)
        if not path.exists():
            return {
                "available": False,
                "source": source,
                "column": column,
                "counts": {
                    "confirmed_error": 0, "false_positive": 0,
                    "ambiguous": 0, "errored": 0, "fn_candidates": 0,
                    "total": 0,
                },
                "items": [],
            }
        items = _read_column_verify_jsonl(path)
        counts = {
            "confirmed_error": 0, "false_positive": 0, "ambiguous": 0,
            "errored": 0, "fn_candidates": 0, "total": len(items),
        }
        for r in items:
            if r.get("error"):
                counts["errored"] += 1
                continue
            v = str(r.get("verdict") or "ambiguous")
            counts[v] = counts.get(v, 0) + 1
            if v == "confirmed_error" and not r.get("was_flagged_by_llm"):
                counts["fn_candidates"] += 1
        # Sort: FN candidates first (highest-priority for the
        # operator), then confirmed_error, then ambiguous, then
        # false_positive. Within a bucket, by first-seen row.
        verdict_rank = {"confirmed_error": 0, "ambiguous": 1,
                        "false_positive": 2, "errored": 3}

        def _row0(r: Dict[str, Any]) -> int:
            rs = r.get("rows") or []
            if isinstance(rs, list) and rs:
                try:
                    return int(rs[0])
                except (TypeError, ValueError):
                    return 1 << 30
            return 1 << 30

        def _key(r: Dict[str, Any]):
            v = "errored" if r.get("error") else str(r.get("verdict") or "ambiguous")
            is_fn = v == "confirmed_error" and not r.get("was_flagged_by_llm")
            return (
                0 if is_fn else 1,
                verdict_rank.get(v, 99),
                _row0(r),
                str(r.get("value") or ""),
            )

        items.sort(key=_key)
        return {
            "available": True,
            "source": source,
            "column": column,
            "counts": counts,
            "items": items,
        }

    @app.get("/api/runs/{run_id}/column-verifications/progress")
    def get_column_verification_progress(run_id: str) -> Dict[str, Any]:
        """Live progress JSON for an in-flight column-audit run.

        Returns ``{"available": False}`` when no audit has ever run
        against this run; otherwise the parsed contents of
        ``column_verifications/_progress.json`` (which the script
        atomically rewrites on every column transition + every
        value-batch checkpoint). The labelling UI polls this endpoint
        while a column-audit job is running to drive a progress bar.
        """
        rd = _run_dir_or_404(run_id)
        path = _column_verify_dir(rd) / "_progress.json"
        if not path.exists():
            return {"available": False}
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            # Best-effort: if the file is being rewritten when we
            # read it we may catch an empty file or a half-written
            # one. Surface "available but no payload" so the UI keeps
            # polling rather than giving up.
            return {"available": True, "stale": True}
        if not isinstance(payload, dict):
            return {"available": True, "stale": True}
        payload["available"] = True
        return payload

    @app.get("/api/runs/{run_id}/labels")
    def list_labels(run_id: str) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        latest = ctx.store.latest_by_target()
        return {
            "latest": {k: v.model_dump() for k, v in latest.items()},
            "history_count": len(ctx.store.all()),
        }

    @app.post("/api/runs/{run_id}/labels")
    def post_label(run_id: str, payload: LabelIn) -> Label:
        ctx = cache.get(run_id)
        if payload.target == LabelTarget.CLUSTER:
            if ctx.cluster(payload.target_id) is None:
                raise HTTPException(404, f"cluster not found: {payload.target_id}")
        elif payload.target == LabelTarget.ISSUE:
            if ctx.issue(payload.target_id) is None:
                raise HTTPException(404, f"issue not found: {payload.target_id}")
        elif payload.target == LabelTarget.COLUMN_CLUSTER:
            # Column-cluster verdicts are independent per cluster:
            # labelling c1 does not finalize the column's clustering.
            # We accept the verdict if the cluster_id exists either
            # in a saved ``column_clusters.jsonl`` record (column
            # board has run) OR in the column's draft JSONL (LLM
            # proposal not yet finalized). Drafts use ``c{idx+1}``
            # in proposal order, matching the synthetic target_ids
            # surfaced by ``list_column_clusters``.
            #
            # We deliberately do NOT auto-promote drafts to a
            # ``ColumnClusters`` record: that record represents a
            # disjoint partition (only the column-board UI builds
            # one), whereas LLM drafts may overlap. Persisting a
            # subset of drafts here would mark the column as
            # ``labeled`` and hide the remaining drafts from the
            # queue, blocking the operator from voting on them.
            if not payload.source or not payload.column:
                raise HTTPException(
                    400, "column_cluster labels require source and column",
                )
            if payload.source not in ctx.sources:
                raise HTTPException(404, f"source not found: {payload.source}")
            # ``target_id`` of the form ``colclust\x00src\x00col\x00cN``.
            parts = payload.target_id.split("\x00")
            if len(parts) != 4 or parts[0] != "colclust":
                raise HTTPException(
                    400,
                    f"invalid column_cluster target_id: {payload.target_id!r}",
                )
            local_id = parts[3]
            store = _cc_store(run_id)
            rec = store.latest_for(payload.source, payload.column)
            in_record = (
                rec is not None
                and rec.status == ColumnLabelStatus.LABELED
                and any(cl.cluster_id == local_id for cl in rec.clusters)
            )
            in_drafts = False
            if not in_record:
                drafts = _load_drafts_for_column(
                    run_id, payload.source, payload.column,
                )
                # Mirror the c{idx+1}-in-draft-order convention used
                # by ``list_column_clusters`` when surfacing drafts.
                # A draft is a usable target if it has >= 2 string
                # values (same minimum the listing enforces).
                draft_ids: set = set()
                for idx, dr in enumerate(drafts, start=1):
                    raw = dr.get("values") or []
                    str_count = sum(
                        1 for v in raw
                        if isinstance(v, str) and v
                        or (isinstance(v, dict)
                            and isinstance(v.get("value"), str)
                            and v.get("value"))
                    )
                    if str_count >= 2:
                        draft_ids.add(f"c{idx}")
                in_drafts = local_id in draft_ids
            if not (in_record or in_drafts):
                raise HTTPException(
                    404,
                    f"cluster {local_id!r} not found in "
                    f"{payload.source!r}/{payload.column!r}; "
                    f"no saved record or matching draft",
                )
            # Validate ``confirmed_canonical`` (if set) against the
            # cluster's actual value-set. The operator may only
            # endorse one of the cluster's surface forms; rejecting
            # arbitrary strings here keeps the synthesised
            # ``ColumnClusters`` records valid (their schema enforces
            # the same constraint).
            if payload.confirmed_canonical is not None:
                cluster_values: set = set()
                if in_record:
                    for cl in rec.clusters:
                        if cl.cluster_id == local_id:
                            cluster_values = {cv.value for cv in cl.values}
                            break
                else:
                    drafts = _load_drafts_for_column(
                        run_id, payload.source, payload.column,
                    )
                    idx = int(local_id[1:]) - 1
                    if 0 <= idx < len(drafts):
                        for v in drafts[idx].get("values") or []:
                            if isinstance(v, str) and v:
                                cluster_values.add(v)
                            elif (isinstance(v, dict)
                                  and isinstance(v.get("value"), str)
                                  and v.get("value")):
                                cluster_values.add(v["value"])
                if payload.confirmed_canonical not in cluster_values:
                    raise HTTPException(
                        400,
                        f"confirmed_canonical "
                        f"{payload.confirmed_canonical!r} is not a "
                        f"value of cluster {local_id!r}",
                    )
            payload = payload.model_copy(update={
                "target_id": column_cluster_target_id(
                    payload.source, payload.column, local_id,
                ),
            })
        else:  # CELL
            # Cell labels carry FN (false negative). Require the same
            # (source, row, column) triple in the payload so the
            # benchmark exporter can find them later. We rebuild the
            # canonical target_id here regardless of what the client
            # sent so two clients can't drift out of sync on key format.
            if payload.row is None or not payload.column or not payload.source:
                raise HTTPException(
                    400, "cell labels require source, row, and column",
                )
            if payload.source not in ctx.sources:
                raise HTTPException(404, f"source not found: {payload.source}")
            # Cell labels carry an operator-only claim that this cell
            # is (or was) an error the LLM missed. ``TP`` is deliberately
            # excluded: a true-positive is a verdict on an LLM detection,
            # which only the cluster path owns -- there is no detection
            # at an unflagged cell to affirm. The supported verdicts
            # carry the operator's confidence on the FN claim; FP and
            # UNLABELED are tombstones that clear the FN.
            _allowed_cell_statuses = {
                LabelStatus.FN, LabelStatus.PARTIAL,
                LabelStatus.UNSURE, LabelStatus.SKIP,
                LabelStatus.FP, LabelStatus.UNLABELED,
            }
            if payload.status not in _allowed_cell_statuses:
                raise HTTPException(
                    400,
                    "cell labels support fn/partial/unsure/skip "
                    "(verdict on the operator's claim) or fp / "
                    "unlabeled (clear the cell); tp belongs to the "
                    "cluster path",
                )
            payload = payload.model_copy(update={
                "target_id": cell_target_id(
                    payload.source, int(payload.row), payload.column,
                ),
            })

        record = Label(
            **payload.model_dump(exclude={"annotator"}),
            annotator=(payload.annotator or annotator).strip() or annotator,
        )
        return ctx.store.append(record)

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
            if not isinstance(cr, int):
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

    # ---------------------------------------------------------------
    # Column-clusters labelling (inconsistency error type).
    # Backed by ``<run_dir>/column_clusters.jsonl``. Each labelled
    # column carries an optional set of value-clusters identifying
    # standardization groups (e.g. {"BBB", "ABC HS"} -> same school).
    # Single source of truth: the latest record per (source, column).
    # ---------------------------------------------------------------

    _cc_label_stores: Dict[str, ColumnClusterStore] = {}
    _cc_label_lock = threading.Lock()

    def _cc_store(run_id: str) -> ColumnClusterStore:
        rd = cache._safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        with _cc_label_lock:
            store = _cc_label_stores.get(run_id)
            if store is None:
                store = ColumnClusterStore(rd / "column_clusters.jsonl")
                _cc_label_stores[run_id] = store
            return store

    def _load_drafts_for_column(
        run_id: str, source: str, column: str,
    ) -> List[Dict[str, Any]]:
        """Return the latest LLM ``draft_clusters`` for a (source, column).

        Reads ``<run_dir>/column_cluster_drafts.jsonl`` and applies
        latest-wins (the file is append-only, so a later record for
        the same (source, column) supersedes an earlier one). Returns
        an empty list when the run has no draft file or no draft for
        this pair. Each draft dict carries ``{values, canonical,
        agreement, kind}`` -- the same shape persisted by
        ``experiments.run_column_clustering``.

        Used by:
          * :func:`list_column_clusters` to surface drafts as queue
            items in columns that have no labelled record yet.
          * :func:`post_label` to lazily promote drafts to a real
            record on the operator's first verdict.
        """
        rd = cache._safe_run_dir(run_id)
        drafts_path = rd / "column_cluster_drafts.jsonl"
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
                    if (rec.get("source") == source
                            and rec.get("column") == column):
                        found = rec.get("draft_clusters") or []
        except OSError as exc:
            logger.warning(
                "_load_drafts_for_column: read failed: %s", exc,
            )
            return []
        return found or []

    def _column_value_counts(
        ctx: "RunContext", source: str, column: str,
    ) -> Tuple[Dict[str, int], int]:
        """Return ``(value -> count, n_rows)`` for one CSV column.

        Empty strings (``""``) are excluded from the count map -- a
        standardization cluster over a blank value is meaningless. The
        empty-cell tally is still reflected in the returned ``n_rows``
        for context.
        """
        frame = ctx.load_frame(source)
        if column not in frame.columns:
            raise HTTPException(
                404, f"column not found in {source}: {column}",
            )
        series = frame[column].astype(str)
        n_rows = int(len(series))
        counts: Dict[str, int] = {}
        for val in series:
            if val == "":
                continue
            counts[val] = counts.get(val, 0) + 1
        return counts, n_rows

    @app.get("/api/runs/{run_id}/inconsistency-columns")
    def list_inconsistency_columns(
        run_id: str,
        source: Optional[str] = None,
        min_distinct: int = Query(2, ge=1),
        # A column whose distinct-value ratio (n_distinct / n_rows)
        # exceeds this is hidden. Defaults to ``1.0`` so every
        # column with >= ``min_distinct`` distinct values surfaces
        # for human review (matches the column-clustering driver,
        # which sends them all to the LLM proposer). Lower the
        # value to re-enable the historical PK / free-text guard.
        max_distinct_ratio: float = Query(1.0, gt=0.0, le=1.0),
        # Maximum total candidates returned in one response. Defaults
        # generously; the UI is expected to scope by ``source``.
        limit: int = Query(2000, ge=1, le=20000),
    ) -> Dict[str, Any]:
        """Enumerate candidate columns for standardization labelling.

        For each (source, column) the response carries:
          * ``n_rows`` - total rows in the source.
          * ``n_distinct`` - distinct non-empty values.
          * ``top_values`` - up to 3 most-frequent values + counts
            (so the UI can render a queue hint).
          * ``status`` / ``n_clusters`` / ``annotator`` / ``ts`` from
            the latest ``column_clusters.jsonl`` record, or
            ``"unlabeled"`` when nothing has been written yet.

        Cardinality filter: ``min_distinct`` excludes columns that
        mathematically can't form a cluster (< 2 distinct values).
        ``max_distinct_ratio`` defaults to ``1.0`` -- every other
        column is surfaced for human review. Callers that want the
        old PK / free-text guard can pass a lower value.
        """
        ctx = cache.get(run_id)
        store = _cc_store(run_id)
        latest = store.latest_by_column()

        # Build a (source, column) -> n_draft_clusters index from
        # column_cluster_drafts.jsonl. Only a fraction of candidate
        # columns actually produce non-empty drafts (~7% in practice),
        # so the frontend's "Has draft" filter and the queue sort
        # need this signal up front - otherwise the operator clicks
        # through dozens of empty-draft columns before finding one
        # the LLM had anything to say about. Returns an empty map if
        # the file is missing (single-pass / non-adaptive runs).
        n_draft_index: Dict[Tuple[str, str], int] = {}
        try:
            rd = ctx.run_dir
            drafts_path = rd / "column_cluster_drafts.jsonl"
            if drafts_path.exists():
                with drafts_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        s = d.get("source") or ""
                        c = d.get("column") or ""
                        if not s or not c:
                            continue
                        dc = d.get("draft_clusters") or []
                        n_draft_index[(s, c)] = len(dc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "inconsistency-columns: failed to index drafts: %s", exc,
            )
            n_draft_index = {}

        items: List[Dict[str, Any]] = []
        sources = [source] if source else list(ctx.sources.keys())
        for src in sources:
            if src not in ctx.sources:
                continue
            try:
                frame = ctx.load_frame(src)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "inconsistency-columns: failed to load %s: %s",
                    src, exc,
                )
                continue
            n_rows = int(len(frame))
            if n_rows == 0:
                continue
            for col in frame.columns:
                series = frame[col].astype(str)
                counts: Dict[str, int] = {}
                for val in series:
                    if val == "":
                        continue
                    counts[val] = counts.get(val, 0) + 1
                n_distinct = len(counts)
                if n_distinct < min_distinct:
                    continue
                if n_rows > 0 and n_distinct / n_rows > max_distinct_ratio:
                    continue
                top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                rec = latest.get((src, col))
                # ``n_draft_clusters`` is the count of LLM-proposed
                # candidate clusters for this column (0 means the LLM
                # ran but didn't find anything, distinguishable from
                # "no LLM pass ran at all" only via the run's overall
                # drafts file existence -- which the queue payload
                # doesn't surface).
                n_drafts = n_draft_index.get((src, col), 0)
                items.append({
                    "source": src,
                    "column": col,
                    "n_rows": n_rows,
                    "n_distinct": n_distinct,
                    "top_values": [
                        {"value": v, "count": c} for v, c in top
                    ],
                    "status": rec.status.value if rec else
                        ColumnLabelStatus.UNLABELED.value,
                    "n_clusters": len(rec.clusters) if rec else 0,
                    "n_draft_clusters": n_drafts,
                    "annotator": rec.annotator if rec else None,
                    "ts": rec.ts if rec else None,
                })

        # Stable order: unlabeled first (most actionable), then
        # columns with at least one LLM-proposed cluster (so the
        # operator lands on something the LLM actually flagged
        # instead of scrolling through dozens of empty drafts),
        # then by source / column.
        order = {
            ColumnLabelStatus.UNLABELED.value: 0,
            ColumnLabelStatus.LABELED.value: 1,
            ColumnLabelStatus.NO_INCONSISTENCY.value: 2,
            ColumnLabelStatus.DROPPED_OVERLAPPING.value: 3,
            ColumnLabelStatus.DROPPED_MESSY.value: 4,
        }
        items.sort(key=lambda x: (
            order.get(x["status"], 99),
            0 if x["n_draft_clusters"] > 0 else 1,
            x["source"],
            x["column"],
        ))
        total = len(items)
        # Headline count of how many entries in the full queue
        # carry an LLM-proposed cluster -- the frontend uses this
        # to decide whether to surface a "with LLM draft" filter
        # button.
        total_with_draft = sum(
            1 for it in items if it["n_draft_clusters"] > 0
        )
        return {
            "total": total,
            "total_with_draft": total_with_draft,
            "items": items[:limit],
        }

    @app.get(
        "/api/runs/{run_id}/columns/{source}/{column}/values"
    )
    def get_column_values(
        run_id: str, source: str, column: str,
        # When the column has thousands of distinct values, return
        # only the top-N by frequency. The labelling UI shows them
        # ordered; rare-tail singletons are surfaced via a separate
        # tray in Slice 6.
        limit: int = Query(500, ge=1, le=10000),
    ) -> Dict[str, Any]:
        """Return distinct values + per-value counts for one column.

        Used by the column-board frontend (Slice 6) to render the
        clusterable value list. Empty cells are excluded from the
        value list but counted in the ``n_empty`` summary so the
        operator can tell whether the column has missing data.
        """
        ctx = cache.get(run_id)
        if source not in ctx.sources:
            raise HTTPException(404, f"source not found: {source}")
        counts, n_rows = _column_value_counts(ctx, source, column)
        items = [
            {"value": v, "count": c}
            for v, c in sorted(
                counts.items(), key=lambda kv: (-kv[1], kv[0]),
            )
        ]
        n_empty = n_rows - sum(counts.values())
        return {
            "source": source,
            "column": column,
            "n_rows": n_rows,
            "n_empty": n_empty,
            "n_distinct": len(counts),
            "values": items[:limit],
            "truncated": len(items) > limit,
        }

    @app.get(
        "/api/runs/{run_id}/columns/{source}/{column}/clusters"
    )
    def get_column_clusters(
        run_id: str, source: str, column: str,
    ) -> Dict[str, Any]:
        """Return the latest column-clusters record for (source, column).

        ``record`` is ``null`` when nothing has been written yet.
        ``history`` carries every append-only record for the (source,
        column) pair in write order (oldest first).
        """
        store = _cc_store(run_id)
        latest = store.latest_for(source, column)
        history = store.history(source, column)
        return {
            "record": latest.model_dump(mode="json") if latest else None,
            "history": [h.model_dump(mode="json") for h in history],
        }

    @app.post(
        "/api/runs/{run_id}/columns/{source}/{column}/clusters"
    )
    def post_column_clusters(
        run_id: str, source: str, column: str,
        payload: ColumnClustersIn,
    ) -> Dict[str, Any]:
        """Write a column-clusters record.

        The path's ``source`` / ``column`` win over the payload's --
        the frontend sends them in the body for convenience but the
        URL is the source of truth. Cluster invariants are enforced
        by :class:`ColumnClustersIn`'s validators; this handler
        additionally validates that every clustered value actually
        appears in the column (so a stale draft can't poison the
        record).
        """
        ctx = cache.get(run_id)
        if source not in ctx.sources:
            raise HTTPException(404, f"source not found: {source}")
        # Validate that values referenced by clusters exist in the
        # column. We don't validate the count fields against the
        # actual count -- the human/LLM annotator may quote a stale
        # count from a draft; the scorer joins against the live CSV
        # anyway. But if a value isn't in the column at all the
        # record is unusable downstream.
        counts, _ = _column_value_counts(ctx, source, column)
        unknown: List[str] = []
        for cl in payload.clusters:
            for cv in cl.values:
                if cv.value not in counts:
                    unknown.append(cv.value)
        if unknown:
            raise HTTPException(
                400,
                f"values not present in {source}/{column}: "
                f"{unknown[:5]}{'...' if len(unknown) > 5 else ''}",
            )

        store = _cc_store(run_id)
        record = ColumnClusters(
            source=source,
            column=column,
            status=payload.status,
            drop_reason=payload.drop_reason,
            clusters=payload.clusters,
            note=payload.note,
            annotator=(
                (payload.annotator or annotator).strip() or annotator
            ),
        )
        saved = store.append(record)
        return {
            "record": saved.model_dump(mode="json"),
            "history": [
                h.model_dump(mode="json")
                for h in store.history(source, column)
            ],
        }

    @app.get(
        "/api/runs/{run_id}/columns/{source}/{column}/draft"
    )
    def get_column_draft(
        run_id: str, source: str, column: str,
    ) -> Dict[str, Any]:
        """Return the LLM-proposed draft clusters for (source, column).

        Reads ``<run_dir>/column_cluster_drafts.jsonl`` (written by
        ``experiments.run_column_clustering``). When the run has no
        draft file, or no draft for this column, returns
        ``{"draft": null}``. When multiple drafts exist for the same
        (source, column), latest-wins (the file is append-only).
        """
        rd = cache._safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        drafts_path = rd / "column_cluster_drafts.jsonl"
        if not drafts_path.exists():
            return {"draft": None}
        found: Optional[Dict[str, Any]] = None
        try:
            with drafts_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001 - skip malformed
                        continue
                    if (rec.get("source") == source
                            and rec.get("column") == column):
                        found = rec
        except OSError as exc:
            logger.warning("get_column_draft: read failed: %s", exc)
            return {"draft": None}
        return {"draft": found}

    # ------------------------------------------------------------------
    # Column endpoints with the action *before* the column name + a
    # ``{column:path}`` converter so the column can contain literal
    # slashes (e.g. "Project Name/Description"). Starlette decodes
    # ``%2F`` to ``/`` before path matching, so the original routes
    # of the form ``.../{column}/<action>`` 404 on columns with a
    # slash. These aliases mount the *same* handlers under a path
    # shape that places ``column`` as the trailing path segment.
    # The original routes are kept for backward compatibility.
    # ------------------------------------------------------------------

    @app.get("/api/runs/{run_id}/columns/{source}/values/{column:path}")
    def get_column_values_alias(
        run_id: str, source: str, column: str,
        limit: int = Query(500, ge=1, le=10000),
    ) -> Dict[str, Any]:
        return get_column_values(run_id, source, column, limit)

    @app.get("/api/runs/{run_id}/columns/{source}/clusters/{column:path}")
    def get_column_clusters_alias(
        run_id: str, source: str, column: str,
    ) -> Dict[str, Any]:
        return get_column_clusters(run_id, source, column)

    @app.post("/api/runs/{run_id}/columns/{source}/clusters/{column:path}")
    def post_column_clusters_alias(
        run_id: str, source: str, column: str,
        payload: ColumnClustersIn,
    ) -> Dict[str, Any]:
        return post_column_clusters(run_id, source, column, payload)

    @app.get("/api/runs/{run_id}/columns/{source}/draft/{column:path}")
    def get_column_draft_alias(
        run_id: str, source: str, column: str,
    ) -> Dict[str, Any]:
        return get_column_draft(run_id, source, column)

    @app.get("/api/runs/{run_id}/column-clusters")
    def list_column_clusters(
        run_id: str,
        source: Optional[str] = None,
        status: Optional[str] = None,
        include_drafts: bool = True,
        limit: int = Query(5000, ge=1, le=20000),
    ) -> Dict[str, Any]:
        """Flatten every saved column-cluster into queue-ready items.

        One JSON object per ``StdCluster`` across every (source,
        column) record in ``column_clusters.jsonl``. Each item carries
        the data the cell-level label page needs to:
          * render a queue row (source, column, kind, n_values),
          * paint the CSV (full ``values`` list + canonical),
          * render the right pane (kind, canonical, agreement, llm
            suggestion if drafted), and
          * route a TP/FP/Partial/Unsure/Skip verdict back through
            ``POST /labels`` (synthetic ``target_id``).

        The verdict (``label_status``) is joined from the labels
        store so the queue dot color is consistent with cell-level
        clusters on first paint - no extra fetch needed.

        When ``include_drafts`` is true (the default), LLM-proposed
        draft clusters are *also* surfaced for columns whose
        ``column_clusters.jsonl`` record is missing or not in the
        ``labeled`` state. Synthetic items use ``c{idx+1}`` local
        ids matching the draft order and carry
        ``draft_pending: true``. The first verdict on such an item
        lazy-promotes the column's full draft list to a real
        record (see ``post_label``'s COLUMN_CLUSTER branch) so the
        cluster_id assignment stays deterministic.

        Filters:
          * ``source`` - exact CSV basename match.
          * ``status`` - one of ``unlabeled``/``tp``/``fp``/``partial``
            /``unsure``/``skip`` (matches the joined verdict).
          * ``include_drafts`` - set to ``false`` to hide unsaved
            drafts (matches the legacy behavior).
        """
        ctx = cache.get(run_id)
        store = _cc_store(run_id)
        latest_records = store.latest_by_column()
        label_index = ctx.store.latest_by_target()

        # Pre-load draft clusters so we can attach the agreement /
        # canonical-suggestion to each saved cluster whose value-set
        # matches a draft cluster's value-set. The match key is the
        # ``frozenset`` of cluster values (canonical-agnostic) - the
        # operator may have chosen a different canonical than the
        # LLM did but the cluster's *identity* is the value-set.
        # We deliberately don't surface drafts here that were never
        # saved into a record (see endpoint docstring).
        draft_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        try:
            drafts_path = ctx.run_dir / "column_cluster_drafts.jsonl"
            if drafts_path.exists():
                with drafts_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key = (d.get("source") or "", d.get("column") or "")
                        if not key[0] or not key[1]:
                            continue
                        draft_lookup[key] = d.get("draft_clusters") or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_column_clusters: draft index failed: %s", exc)
            draft_lookup = {}

        items: List[Dict[str, Any]] = []
        for (src, col), rec in latest_records.items():
            if source and src != source:
                continue
            # Only surface clusters from records the operator actively
            # *labelled* (status=labeled). Records flagged
            # no_inconsistency / dropped_* by the operator carry no
            # clusters anyway, so this is mostly defensive.
            if rec.status != ColumnLabelStatus.LABELED:
                continue
            drafts = draft_lookup.get((src, col), [])
            for cl in rec.clusters:
                target_id = column_cluster_target_id(src, col, cl.cluster_id)
                lbl = label_index.get(target_id)
                label_status = (
                    lbl.status.value if lbl is not None else "unlabeled"
                )
                if status and label_status != status:
                    continue
                # Match this cluster's value-set against the LLM draft
                # to surface the LLM's proposed canonical + agreement.
                # Use frozenset so the order doesn't matter.
                value_set = frozenset(cv.value for cv in cl.values)
                llm_canonical: Optional[str] = None
                llm_agreement: Optional[float] = None
                llm_kind: Optional[str] = None
                llm_target_style: Optional[str] = None
                for dc in drafts:
                    dvs = frozenset(dc.get("values") or [])
                    if dvs == value_set:
                        llm_canonical = dc.get("canonical")
                        llm_agreement = dc.get("agreement")
                        llm_kind = dc.get("kind")
                        ts = dc.get("target_style")
                        llm_target_style = ts if isinstance(ts, str) and ts else None
                        break
                items.append({
                    "target_id": target_id,
                    "source": src,
                    "column": col,
                    "cluster_local_id": cl.cluster_id,
                    "kind": cl.kind.value,
                    "canonical_value": cl.canonical_value,
                    "values": [
                        {"value": cv.value, "count": cv.count}
                        for cv in cl.values
                    ],
                    "n_values": len(cl.values),
                    "annotator": rec.annotator,
                    "ts": rec.ts,
                    "label_status": label_status,
                    "label_note": (lbl.note if lbl else ""),
                    "label_confirmed_canonical": (
                        lbl.confirmed_canonical if lbl else None
                    ),
                    "label_value_corrections": (
                        dict(lbl.value_corrections) if lbl and lbl.value_corrections else {}
                    ),
                    "llm_canonical": llm_canonical,
                    "llm_agreement": llm_agreement,
                    "llm_kind": llm_kind,
                    "llm_target_style": llm_target_style,
                    "from_draft": llm_canonical is not None,
                    "draft_pending": False,
                    "is_col_cluster": True,
                    "error_types": ["inconsistency"],
                })
                if len(items) >= limit:
                    break
            if len(items) >= limit:
                break

        # Surface LLM drafts as queue items for columns whose record
        # is missing or not in the ``labeled`` state. The synthetic
        # ``target_id`` uses ``c{idx+1}`` matching the draft order so
        # the lazy-promotion in ``post_label`` lines up. Drafts in
        # columns that ARE labelled are skipped here -- they only
        # participate as a cross-reference for the LLM hint (handled
        # above) since their value-sets are already represented by
        # the saved clusters.
        if include_drafts and len(items) < limit:
            labeled_columns: set = {
                (src, col)
                for (src, col), rec in latest_records.items()
                if rec.status == ColumnLabelStatus.LABELED
            }
            for (src, col), drafts in draft_lookup.items():
                if source and src != source:
                    continue
                if (src, col) in labeled_columns:
                    continue
                for idx, dc in enumerate(drafts, start=1):
                    raw_values = dc.get("values") or []
                    # Normalise to {value, count} dicts. Drafts written
                    # by ``run_column_clustering`` may store plain
                    # strings; backfill count=1 so the frontend's
                    # ``data-raw`` paint path still finds the cell.
                    norm_values: List[Dict[str, Any]] = []
                    for v in raw_values:
                        if isinstance(v, dict):
                            val = v.get("value")
                            cnt = v.get("count") or 1
                        else:
                            val = v
                            cnt = 1
                        if not isinstance(val, str) or not val:
                            continue
                        norm_values.append({"value": val, "count": int(cnt)})
                    if len(norm_values) < 2:
                        continue
                    local_id = f"c{idx}"
                    target_id = column_cluster_target_id(src, col, local_id)
                    lbl = label_index.get(target_id)
                    label_status = (
                        lbl.status.value if lbl is not None else "unlabeled"
                    )
                    if status and label_status != status:
                        continue
                    canonical = dc.get("canonical") or dc.get("canonical_value")
                    seen = {nv["value"] for nv in norm_values}
                    if canonical not in seen:
                        canonical = None
                    ts = dc.get("target_style")
                    target_style = ts if isinstance(ts, str) and ts else None
                    items.append({
                        "target_id": target_id,
                        "source": src,
                        "column": col,
                        "cluster_local_id": local_id,
                        "kind": dc.get("kind") or "untagged",
                        "canonical_value": canonical,
                        "values": norm_values,
                        "n_values": len(norm_values),
                        "annotator": None,
                        "ts": None,
                        "label_status": label_status,
                        "label_note": (lbl.note if lbl else ""),
                        "label_confirmed_canonical": (
                            lbl.confirmed_canonical if lbl else None
                        ),
                        "label_value_corrections": (
                            dict(lbl.value_corrections) if lbl and lbl.value_corrections else {}
                        ),
                        "llm_canonical": canonical,
                        "llm_agreement": dc.get("agreement"),
                        "llm_kind": dc.get("kind") or "untagged",
                        "llm_target_style": target_style,
                        "from_draft": True,
                        "draft_pending": True,
                        "is_col_cluster": True,
                        "error_types": ["inconsistency"],
                    })
                    if len(items) >= limit:
                        break
                if len(items) >= limit:
                    break
        # Stable order: source, column, then cluster_local_id. The
        # frontend re-bins by source for the sidebar groups, but a
        # deterministic order keeps the queue snapshot stable
        # across reloads.
        items.sort(key=lambda it: (it["source"], it["column"], it["cluster_local_id"]))
        return {"items": items, "total": len(items)}

    @app.get("/api/runs/{run_id}/column-cluster-stats")
    def column_cluster_stats(run_id: str) -> Dict[str, Any]:
        """Dashboard summary: per-status counts of (source, column) pairs.

        Only counts pairs that have an explicit record. The frontend
        derives ``inconsistency_columns_total`` (the queue size) from
        the ``/inconsistency-columns`` listing, not this endpoint.
        """
        store = _cc_store(run_id)
        latest = store.latest_by_column()
        by_status: Dict[str, int] = {}
        n_clusters_total = 0
        for rec in latest.values():
            by_status[rec.status.value] = by_status.get(rec.status.value, 0) + 1
            n_clusters_total += len(rec.clusters)
        return {
            "labelled_columns": len(latest),
            "by_status": by_status,
            "n_clusters_total": n_clusters_total,
        }

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str) -> FileResponse:
        path = outputs_dir / run_id / "report.html"
        if not path.exists():
            raise HTTPException(404, "report.html not generated for this run")
        # The relation-discovery dashboard is a self-contained,
        # generated HTML artifact with inline <script>/<style> blocks
        # (no third-party origins, no remote loads). The workbench's
        # default strict CSP would block its inline JS, so we relax
        # to "self + inline" just for this endpoint. The frame-ancestors
        # directive remains 'none' so the report still can't be
        # iframed by a third party.
        report_csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "base-uri 'none'; "
            "form-action 'self'; "
            "frame-ancestors 'none'"
        )
        return FileResponse(
            path,
            media_type="text/html",
            headers={
                "Content-Security-Policy": report_csp,
                # Reports are regenerated in place when the user clicks
                # "Build report"; without this the browser will keep
                # showing the previous (potentially CSP-broken) copy
                # forever even after a regenerate.
                "Cache-Control": "no-store",
            },
        )

    # ---------------------------------------------------------------
    # Original-xlsx access (download + HTML preview).
    # ---------------------------------------------------------------

    @app.get("/api/runs/{run_id}/workbooks/{workbook}/xlsx")
    def get_workbook_xlsx(run_id: str, workbook: str) -> FileResponse:
        ctx = cache.get(run_id)
        path = ctx.workbook_xlsx_path(workbook)
        if path is None:
            raise HTTPException(
                404,
                f"original xlsx not found for workbook {workbook!r} "
                "(no xlsx_source_dir configured or file missing)",
            )
        return FileResponse(
            path,
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            filename=workbook,
        )

    @app.get("/api/runs/{run_id}/workbooks/{workbook}/preview")
    def get_workbook_preview(run_id: str, workbook: str) -> HTMLResponse:
        ctx = cache.get(run_id)
        path = ctx.workbook_xlsx_path(workbook)
        if path is None:
            raise HTTPException(
                404,
                f"original xlsx not found for workbook {workbook!r}",
            )
        download_url = (
            f"/api/runs/{run_id}/workbooks/{workbook}/xlsx"
        )
        html = _render_xlsx_preview_html(
            path,
            workbook_name=workbook,
            download_url=download_url,
        )
        return HTMLResponse(html)

    # =====================================================================
    # Compare runs (cell-level bucket diff between two runs)
    # =====================================================================

    # =====================================================================
    # Static UI + root
    # =====================================================================

    class _NoCacheHTMLStaticFiles(StaticFiles):
        """StaticFiles subclass that forces revalidation on HTML responses.

        HTML files anchor every other static asset on the page (script and
        link tags with cache-bust query strings). If the browser pins a
        stale HTML, it loads stale JS/CSS without the new query strings
        and never picks up new code.

        For ``.html`` (and the bare directory index) we send
        ``Cache-Control: no-cache`` so the browser must revalidate with
        the server on every load. ETag-based 304s mean revalidation is
        cheap when the file hasn't changed. Other assets keep their
        default headers (ETag + Last-Modified, no Cache-Control), which
        is fine because cache-bust query strings invalidate them.
        """

        async def get_response(self, path, scope):  # type: ignore[override]
            response = await super().get_response(path, scope)
            try:
                # Detect HTML by the actual served content-type rather than
                # the request path; that way directory indexes (path="")
                # and any other indirect paths also pick up the header.
                content_type = response.headers.get("content-type", "").lower()
                # Also force revalidation on the JS/CSS bundles. The
                # cache-bust query strings in index.html cover the *first*
                # load of a new build, but operators frequently keep a tab
                # open across deploys; without an explicit Cache-Control
                # the browser may reuse the cached label.js indefinitely
                # (until the tab is hard-refreshed). ETag-based 304s keep
                # the revalidation cost negligible when the file is
                # unchanged.
                if (
                    content_type.startswith("text/html")
                    or content_type.startswith("text/javascript")
                    or content_type.startswith("application/javascript")
                    or content_type.startswith("text/css")
                ):
                    response.headers["Cache-Control"] = "no-cache, must-revalidate"
            except Exception:  # pragma: no cover - defensive
                pass
            return response

    app.mount(
        "/ui",
        _NoCacheHTMLStaticFiles(directory=str(static_dir), html=True),
        name="ui",
    )

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cluster_summary(c: Mapping[str, Any]) -> Dict[str, Any]:
    syn = c.get("synthesis") or {}
    primary = None
    if isinstance(syn, Mapping):
        primary = syn.get("primary_error_type")
    error_types = list(c.get("error_types") or [])
    # Surface the union of every detection's ``affected_rows`` plus
    # the primary row so the queue panel can show "rows: 5, 12, 27"
    # at a glance instead of forcing the operator to open each
    # cluster and scan the context table. Originally this was only
    # populated for ``row_duplication`` clusters; relation rules
    # (string_calculation_inconsistency, FD, etc.) also fire on many
    # rows but collapse to ONE cluster after the per-incident
    # aggregation in ``compile_code_violations``, so the same
    # affected-rows surface is what tells the labeller "this single
    # TP click covers N rows".
    affected_rows: List[int] = []
    seen: set = set()
    primary_row = c.get("row")
    if isinstance(primary_row, (int, float)):
        seen.add(int(primary_row))
    for d in c.get("detections") or []:
        if not isinstance(d, Mapping):
            continue
        for r in d.get("affected_rows") or ():
            try:
                seen.add(int(r))
            except (TypeError, ValueError):
                continue
    # Only surface when there are MORE rows than just the primary;
    # singleton clusters get an empty list so the existing UI guards
    # (`affected_rows.length > 1`) keep their meaning.
    if len(seen) > 1:
        affected_rows = sorted(seen)
    # Verdict / strategy: row_duplication clusters carry these on
    # each detection. We surface the strongest verdict (duplicate
    # beats near_duplicate) and the union of strategies so the queue
    # panel can show a "near-dup" badge alongside the row list.
    verdict = ""
    discovery_strategies: List[str] = []
    cluster_budget_exhausted = False
    max_groups_per_source = 0
    if "row_duplication" in error_types:
        verdict_rank = {"duplicate": 2, "near_duplicate": 1}
        seen_strats: set = set()
        best_rank = -1
        for d in c.get("detections") or []:
            if not isinstance(d, Mapping):
                continue
            v = str(d.get("verdict", "") or "").strip().lower()
            if v in verdict_rank and verdict_rank[v] > best_rank:
                best_rank = verdict_rank[v]
                verdict = v
            s = str(d.get("discovery_strategy", "") or "").strip()
            if s and s not in seen_strats:
                seen_strats.add(s)
                discovery_strategies.append(s)
            if d.get("cluster_budget_exhausted"):
                cluster_budget_exhausted = True
                try:
                    cap = int(d.get("max_groups_per_source") or 0)
                except (TypeError, ValueError):
                    cap = 0
                if cap > max_groups_per_source:
                    max_groups_per_source = cap
    return {
        "cluster_id": c.get("cluster_id"),
        "source": c.get("source"),
        "row": c.get("row"),
        "columns": list(c.get("columns") or []),
        "error_types": error_types,
        "evidence_count": c.get("evidence_count"),
        "primary_error_type": primary,
        "is_multi_type": len(error_types) >= 2,
        "affected_rows": affected_rows,
        "verdict": verdict,
        "discovery_strategies": discovery_strategies,
        "cluster_budget_exhausted": cluster_budget_exhausted,
        "max_groups_per_source": max_groups_per_source,
    }


