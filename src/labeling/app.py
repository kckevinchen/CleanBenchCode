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
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..core.data_loader import load_csv
from ..core.text_normalize import NormalizeOptions, normalize_frame
from .export import write_benchmark
from .runs import (
    JobManager,
    RunRegistry,
    list_datasets,
    load_config_defaults,
    load_taxonomy_keys,
)
from .schema import (
    Label,
    LabelIn,
    LabelStats,
    LabelStatus,
    LabelTarget,
    cell_target_id,
)
from .store import LabelStore

logger = logging.getLogger(__name__)


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

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"run_dir not found: {self.run_dir}")

        self.summary: Dict[str, Any] = self._read_json("summary.json") or {}
        self.clusters: List[Dict[str, Any]] = self._read_json("clusters.json") or []
        self.issues: List[Dict[str, Any]] = self._read_json("issues.json") or []

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
                return ctx
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

    # Compare-runs cache: results of ``diff_runs(a, b)`` keyed by
    # (a, b, mtime_a, mtime_b). diff_runs scans both issues.jsonl
    # files and indexes every cell, so it's worth memoising for
    # repeated UI calls (cells endpoint pages through the same diff).
    _diff_lock = threading.Lock()
    _diff_cache: Dict[tuple, Any] = {}

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

    def _validate_source_name(name: str) -> None:
        """Reject source names that look like paths."""
        if (
            not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            raise HTTPException(400, f"invalid source name: {name!r}")

    def _run_dir_or_404(run_id: str) -> Path:
        rd = _safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        return rd

    def _get_diff(run_a: str, run_b: str) -> Dict[str, Any]:
        from experiments.compare_runs import diff_runs  # lazy import
        a_dir = _run_dir_or_404(run_a)
        b_dir = _run_dir_or_404(run_b)
        # mtime of issues.jsonl is the only thing that matters for
        # the diff result. Missing file -> 404 inside diff_runs.
        try:
            mtime_a = (a_dir / "issues.jsonl").stat().st_mtime
            mtime_b = (b_dir / "issues.jsonl").stat().st_mtime
        except FileNotFoundError as exc:
            raise HTTPException(404, f"missing issues.jsonl: {exc}") from exc
        key = (run_a, run_b, mtime_a, mtime_b)
        with _diff_lock:
            cached = _diff_cache.get(key)
            if cached is not None:
                return cached
            # Bound cache size so we don't accumulate stale pair-results.
            if len(_diff_cache) >= 8:
                _diff_cache.clear()
        result = diff_runs(a_dir, b_dir)
        with _diff_lock:
            _diff_cache[key] = result
        return result

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
    def delete_run(run_id: str) -> Dict[str, Any]:
        # _safe_run_dir enforces both syntactic checks AND that the
        # resolved path is a direct child of outputs_dir, so a symlink
        # in outputs/<id> can't redirect rmtree outside the workbench.
        run_dir = _safe_run_dir(run_id)
        if not run_dir.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        cache.invalidate(run_id)
        import shutil
        shutil.rmtree(run_dir)
        return {"ok": True, "run_id": run_id}

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
    ) -> Dict[str, Any]:
        """Trigger a single-cluster web-verification on demand.

        Used by the "Verify on web" button in the labeling UI for
        clusters the batch verifier didn't pick up. Synchronous - the
        UI shows a spinner while we make the Tavily + LLM calls
        (typically 5-15 s per cluster). Returns the new verification
        record on success.

        Status codes:
        * ``200`` - verification produced (verdict may be any of
          confirmed_error / false_positive / ambiguous / errored).
        * ``404`` - cluster not found in this run.
        * ``409`` - cluster already has a verification (UI bug or
          double-click; refuse to clobber).
        * ``422`` - cluster is gate-rejected (PII column / non-entity
          column with no whitelisted error_type / unsuitable value).
        * ``500`` - missing API key, prompt file unreadable, etc.
        """
        ctx = cache.get(run_id)
        if ctx.cluster(cluster_id) is None:
            raise HTTPException(404, f"cluster not found: {cluster_id}")
        # Importing inside the handler keeps the labeling-app
        # importable even when the verifier package has unmet
        # dependencies (httpx is the only one and is already required
        # transitively by FastAPI, but be defensive).
        try:
            from experiments.run_web_verify import (  # noqa: WPS433
                VerifyError,
                verify_single_cluster,
            )
        except ImportError as exc:
            raise HTTPException(500, f"verifier unavailable: {exc}")
        try:
            record = verify_single_cluster(ctx.run_dir, cluster_id)
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
            if payload.status not in (LabelStatus.FN, LabelStatus.FP):
                # FP-on-cell is allowed for clearing a previous FN.
                raise HTTPException(
                    400,
                    "cell labels only support status=fn (or fp to clear)",
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
            if lab is None or lab.target != LabelTarget.CLUSTER:
                continue
            if lab.status == LabelStatus.SKIP:
                continue
            for col in (c.get("columns") or []):
                covered.add((r, col))
        cell_labels: set = set()
        for tid, lab in latest.items():
            if lab.target != LabelTarget.CELL or lab.source != source:
                continue
            if lab.row is None or not lab.column:
                continue
            cell_labels.add((int(lab.row), lab.column))

        propagated_cells: List[Dict[str, Any]] = []
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
            label = Label(
                target=LabelTarget.CELL,
                target_id=cell_target_id(source, idx, column),
                source=source,
                row=idx,
                column=column,
                status=LabelStatus.FN,
                corrected_value=str(corrected),
                note=note or "(auto: same-value as labelled TP cell)",
                annotator=ann,
            )
            ctx.store.append(label)
            propagated_cells.append({"row": idx, "column": column})

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

    @app.get("/api/runs/{run_id}/stats")
    def stats(run_id: str) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        latest = ctx.store.latest_by_target()

        def _bucket(cluster_ids: List[str]) -> LabelStats:
            counts = {"tp": 0, "fp": 0, "partial": 0, "unsure": 0, "skip": 0, "unlabeled": 0}
            for cid in cluster_ids:
                label = latest.get(cid)
                key = label.status.value if label else "unlabeled"
                counts[key] = counts.get(key, 0) + 1
            return LabelStats(total=len(cluster_ids), **counts)

        by_source: Dict[str, List[str]] = {}
        all_ids: List[str] = []
        for c in ctx.clusters:
            cid = c.get("cluster_id")
            if not cid:
                continue
            all_ids.append(cid)
            by_source.setdefault(c.get("source", ""), []).append(cid)

        # Cell-level FN counts. We tally only the *latest* state per
        # cell key, so re-labelling a cell as FP doesn't double-count.
        # Cells that are simultaneously TP via cluster are reported as
        # ``shadowed`` so the operator sees the double-labelling.
        fn_total = 0
        fn_shadowed = 0
        fn_by_source: Dict[str, int] = {}
        # Build a quick (source,row,col) -> cluster-state lookup so we
        # can detect shadowing without re-scanning every cluster.
        covered: Dict[tuple, str] = {}
        for c in ctx.clusters:
            cid = c.get("cluster_id", "")
            lab = latest.get(cid)
            if lab is None or lab.target != LabelTarget.CLUSTER:
                continue
            st = lab.status.value
            if st not in ("tp", "partial"):
                continue
            row = c.get("row")
            if not isinstance(row, int):
                continue
            for col in (c.get("columns") or []):
                covered[(c.get("source", ""), row, col)] = st
        for lab in latest.values():
            if lab.target != LabelTarget.CELL or lab.status != LabelStatus.FN:
                continue
            if lab.row is None or not lab.column:
                continue
            fn_total += 1
            fn_by_source[lab.source] = fn_by_source.get(lab.source, 0) + 1
            if (lab.source, int(lab.row), lab.column) in covered:
                fn_shadowed += 1

        return {
            "overall": _bucket(all_ids).model_dump(),
            "by_source": {src: _bucket(ids).model_dump() for src, ids in by_source.items()},
            "fn": {
                "total": fn_total,
                "shadowed": fn_shadowed,
                "by_source": fn_by_source,
            },
        }

    @app.get("/api/runs/{run_id}/labeling-effort")
    def labeling_effort(run_id: str) -> Dict[str, Any]:
        """Per-(source, column) labeling effort projection.

        Driven by the per-arm Good-Turing artifact written by adaptive
        runs (``good_turing/per_arm_estimates.json``). Each arm =
        ``(scope_kind=column, (source, column), error_type)`` exposes:

        * ``n_observed``         - distinct defect species seen so far
        * ``chao1.estimated_total`` - Chao1 point estimate of total
          species (observed + projected unseen).
        * ``good_turing_missing_mass`` - probability the next perturbed
          run surfaces a never-before-seen species.
        * ``status``             - saturated / exhausted / active /
          insufficient / not_applicable.

        We aggregate arms over error_types within each (source, column)
        cell to project the labelling-effort the operator should plan
        for. Returns ``{"available": false}`` when the run has no
        per-arm artifact (single-pass / fixed-K runs).
        """
        ctx = cache.get(run_id)
        payload = ctx.per_arm_payload()
        if not payload or not isinstance(payload.get("arms"), list):
            return {
                "available": False,
                "reason": (
                    "Per-arm Good-Turing estimates not available for this "
                    "run (only adaptive multi-pass runs at "
                    "stop_scope=column produce per-(source, column) arms)."
                ),
            }
        scope_kind = payload.get("scope_kind", "column")
        if scope_kind != "column":
            return {
                "available": False,
                "reason": (
                    "Per-arm artifact uses scope_kind="
                    f"{scope_kind!r}; per-column effort projection "
                    "requires stop_scope=column."
                ),
            }

        # Already-labelled counts per (source, column). Used to show
        # progress vs. the projected total. We only count cluster-level
        # verdicts (tp/partial) plus FN cell-level labels - skipped /
        # unsure clusters do not contribute to "labelled".
        latest = ctx.store.latest_by_target()
        labelled_by_cell: Dict[Tuple[str, str], int] = {}
        for c in ctx.clusters:
            cid = c.get("cluster_id")
            lab = latest.get(cid) if cid else None
            if lab is None or lab.target != LabelTarget.CLUSTER:
                continue
            if lab.status.value not in ("tp", "fp", "partial"):
                continue
            src = c.get("source") or ""
            for col in (c.get("columns") or []):
                k = (src, str(col))
                labelled_by_cell[k] = labelled_by_cell.get(k, 0) + 1
        for lab in latest.values():
            if lab.target != LabelTarget.CELL or lab.status != LabelStatus.FN:
                continue
            if lab.column is None:
                continue
            k = (lab.source, str(lab.column))
            labelled_by_cell[k] = labelled_by_cell.get(k, 0) + 1

        # Group arms by (source, column).
        cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for arm in payload["arms"]:
            scope_id = arm.get("scope_id") or []
            if len(scope_id) < 2:
                continue
            source = str(scope_id[0])
            column = str(scope_id[1])
            etype = str(arm.get("error_type", ""))
            n_obs = float(arm.get("n_observed", 0))
            chao1 = arm.get("chao1") or {}
            est_total = float(chao1.get("estimated_total", n_obs))
            est_missed = max(est_total - n_obs, 0.0)
            ci = chao1.get("ci_95") or [est_total, est_total]
            missing_mass = float(arm.get("good_turing_missing_mass", 0.0))
            status = str(arm.get("status", "active"))

            cell = cells.setdefault((source, column), {
                "source": source,
                "column": column,
                "n_observed": 0.0,
                "chao1_estimated_total": 0.0,
                "chao1_estimated_missed": 0.0,
                "chao1_ci_low": 0.0,
                "chao1_ci_high": 0.0,
                # Mass aggregation: weight each arm's missing mass by
                # its hit count so a tiny arm with mass=1.0 doesn't
                # dominate. Falls back to a simple max if no hits yet.
                "_mass_weight_sum": 0.0,
                "_mass_weighted": 0.0,
                "_mass_max": 0.0,
                "arms_total": 0,
                "arms_saturated": 0,
                "arms_active": 0,
                "arms_insufficient": 0,
                "arms_not_applicable": 0,
                "by_error_type": [],
            })
            cell["n_observed"] += n_obs
            cell["chao1_estimated_total"] += est_total
            cell["chao1_estimated_missed"] += est_missed
            cell["chao1_ci_low"] += float(ci[0]) if ci else est_total
            cell["chao1_ci_high"] += float(ci[1]) if ci else est_total
            # Hit-count weighted missing mass (n_observed proxy for
            # active probe count). Falls back to max() when n_obs=0.
            cell["_mass_weight_sum"] += n_obs
            cell["_mass_weighted"] += missing_mass * n_obs
            cell["_mass_max"] = max(cell["_mass_max"], missing_mass)
            cell["arms_total"] += 1
            if status in ("saturated", "exhausted"):
                cell["arms_saturated"] += 1
            elif status == "not_applicable":
                cell["arms_not_applicable"] += 1
            elif status == "insufficient":
                cell["arms_insufficient"] += 1
            else:
                cell["arms_active"] += 1
            cell["by_error_type"].append({
                "error_type": etype,
                "n_observed": int(n_obs),
                "estimated_total": est_total,
                "estimated_missed": est_missed,
                "missing_mass": missing_mass,
                "status": status,
            })

        # Finalise + attach labelled-so-far counts.
        out: List[Dict[str, Any]] = []
        for (src, col), c in cells.items():
            ws = c.pop("_mass_weight_sum")
            mw = c.pop("_mass_weighted")
            mmax = c.pop("_mass_max")
            c["good_turing_missing_mass"] = (mw / ws) if ws > 0 else mmax
            c["labelled_so_far"] = labelled_by_cell.get((src, col), 0)
            # "Effort remaining" projects how many MORE distinct
            # defect species the operator should expect to see if
            # they fully cover this cell. Based on Chao1's missed
            # estimate net of what they've already labelled (clamped
            # to zero - already-labelled can exceed estimated_missed
            # for very-saturated cells).
            still_to_label = (
                c["chao1_estimated_total"] - c["labelled_so_far"]
            )
            c["effort_remaining"] = max(still_to_label, 0.0)
            # Sort the per-error-type list by est_missed desc so the
            # frontend can show the top contributors.
            c["by_error_type"].sort(
                key=lambda e: e["estimated_missed"], reverse=True
            )
            out.append(c)
        # Default sort: largest projected missed first - the cells the
        # operator should focus on if they want to maximise expected
        # FN coverage per minute spent.
        out.sort(
            key=lambda c: c["chao1_estimated_missed"], reverse=True,
        )

        # Run-level totals so the frontend can show a headline pill.
        total_observed = sum(c["n_observed"] for c in out)
        total_estimated = sum(c["chao1_estimated_total"] for c in out)
        total_missed = sum(c["chao1_estimated_missed"] for c in out)
        total_labelled = sum(c["labelled_so_far"] for c in out)

        return {
            "available": True,
            "scope_kind": scope_kind,
            "num_runs_completed": payload.get("num_runs_completed"),
            "totals": {
                "n_observed": total_observed,
                "chao1_estimated_total": total_estimated,
                "chao1_estimated_missed": total_missed,
                "labelled_so_far": total_labelled,
                "effort_remaining": max(total_estimated - total_labelled, 0.0),
                "cells": len(out),
            },
            "cells": out,
        }

    @app.get("/api/runs/{run_id}/export/benchmark")
    def export_benchmark(run_id: str) -> FileResponse:
        ctx = cache.get(run_id)
        out_path = ctx.run_dir / "benchmark.jsonl"
        write_benchmark(out_path, ctx.clusters, ctx.store)
        return FileResponse(
            out_path,
            media_type="application/x-ndjson",
            filename="benchmark.jsonl",
        )

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str) -> FileResponse:
        path = outputs_dir / run_id / "report.html"
        if not path.exists():
            raise HTTPException(404, "report.html not generated for this run")
        return FileResponse(path, media_type="text/html")

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
        diff = _get_diff(run_a, run_b)
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
        diff = _get_diff(run_a, run_b)
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
        a_dir = _run_dir_or_404(run_a)
        b_dir = _run_dir_or_404(run_b)
        try:
            return diff_cell_detail(a_dir, b_dir,
                                    source=source, row=row, column=column)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    # =====================================================================
    # Launcher
    # =====================================================================

    @app.get("/api/launcher/config")
    def launcher_config() -> Dict[str, Any]:
        defaults = load_config_defaults(project_root)
        return {
            "project_root": str(project_root),
            "defaults": defaults,
            "error_types": load_taxonomy_keys(project_root),
            "datasets": list_datasets(project_root / "data"),
            "scopes": ["single-column", "multi-column", "multi-row", "multi-table"],
            "column_modes": ["single", "all"],
            "modes": ["single", "adaptive"],
            "reasoning_efforts": ["low", "medium", "high"],
            "gt_modes": ["fixed", "adaptive"],
        }

    @app.get("/api/launcher/jobs")
    def list_jobs() -> Dict[str, Any]:
        return {"jobs": [j.to_dict() for j in jobs.list()]}

    @app.post("/api/launcher/jobs")
    def launch_job(payload: Dict[str, Any]) -> Dict[str, Any]:
        kind = (payload.get("kind") or "detection").lower()
        params = payload.get("params") or {}
        try:
            if kind == "detection":
                job = jobs.launch_detection(params)
            elif kind == "report":
                run_id = payload.get("run_id")
                if not run_id:
                    raise HTTPException(400, "run_id is required for kind=report")
                try:
                    job = jobs.launch_report(run_id)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
            elif kind in ("curate", "curate_tables", "curate_pbi"):
                # Unified curate launcher; legacy kind aliases preserved
                # so old UI builds and any saved client requests still work.
                job = jobs.launch_curate(params)
            else:
                raise HTTPException(400, f"unknown kind: {kind}")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return job.to_dict()

    @app.get("/api/launcher/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return job.to_dict()

    @app.get("/api/launcher/jobs/{job_id}/log")
    def get_job_log(
        job_id: str,
        offset: int = Query(0, description="Byte offset; use next_offset to tail."),
    ) -> Dict[str, Any]:
        try:
            return jobs.read_log(job_id, offset=offset)
        except KeyError as exc:
            raise HTTPException(404, f"job not found: {job_id}") from exc

    @app.post("/api/launcher/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Dict[str, Any]:
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return job.to_dict()

    @app.delete("/api/launcher/jobs/{job_id}")
    def delete_job(job_id: str) -> Dict[str, Any]:
        try:
            jobs.delete(job_id)
        except KeyError as exc:
            raise HTTPException(404, f"job not found: {job_id}") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True, "job_id": job_id}

    # =====================================================================
    # Static UI + root
    # =====================================================================

    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

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
    return {
        "cluster_id": c.get("cluster_id"),
        "source": c.get("source"),
        "row": c.get("row"),
        "columns": list(c.get("columns") or []),
        "error_types": list(c.get("error_types") or []),
        "evidence_count": c.get("evidence_count"),
        "primary_error_type": primary,
        "is_multi_type": len(c.get("error_types") or []) >= 2,
    }


def _scalar(v: Any) -> str:
    if v is None:
        return ""
    return str(v)
