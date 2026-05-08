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
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..core.data_loader import load_csv
from ..core.text_normalize import NormalizeOptions, normalize_frame
from .export import (
    _slug_for_error_type,
    materialize_label_benchmark_by_error_type,
    write_benchmark,
    write_benchmark_by_error_type,
)
from .runs import (
    JobManager,
    RunRegistry,
    list_datasets,
    load_config_defaults,
    load_taxonomy_keys,
)
from .schema import (
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
    cell_target_id,
)
from .store import LabelStore
from .exclusions_store import ExclusionStore
from .multi_table_label_store import MultiTableLabelStore
from .relation_review_store import (
    RelationReviewStore,
    ReviewIn as RelationReviewIn,
    ReviewDecision as RelationReviewDecision,
)

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

        Skip rules:
          - the seed cluster itself (the caller already labelled it);
          - clusters that already carry a non-skip verdict from any
            annotator (never overwrite prior decisions);
          - clusters whose anchor cell value differs from the seed.

        Required keys: ``source``, ``row``, ``column``, ``status``.
        Optional: ``note``, ``annotator``.
        Response: ``{"propagated": int, "skipped_existing": int,
                     "matched": int, "clusters": [...]}``.
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
        # FN doesn't apply to cluster verdicts; SKIP has its own
        # row-cascade path so we don't replicate it here.
        if status not in (
            LabelStatus.TP, LabelStatus.FP,
            LabelStatus.PARTIAL, LabelStatus.UNSURE,
        ):
            raise HTTPException(
                400,
                "status must be one of tp/fp/partial/unsure for cluster propagation",
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
        skipped_existing = 0
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
            # Don't clobber prior verdicts (any non-skip cluster label
            # from any annotator).
            existing = latest.get(cid)
            if (
                existing is not None
                and existing.target == LabelTarget.CLUSTER
                and existing.status != LabelStatus.SKIP
            ):
                skipped_existing += 1
                continue
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

    @app.get("/api/runs/{run_id}/exclusions")
    def list_exclusions(run_id: str) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        actives = ctx.exclusions_store.active()
        return {"items": [e.model_dump() for e in actives]}

    @app.post("/api/runs/{run_id}/exclusions")
    def post_exclusion(run_id: str, payload: ExclusionIn) -> Exclusion:
        ctx = cache.get(run_id)
        if payload.source not in ctx.sources:
            raise HTTPException(404, f"source not found: {payload.source}")
        if payload.scope == ExclusionScope.CSV:
            if payload.rows or payload.columns:
                raise HTTPException(
                    400, "scope=csv must not carry rows / columns",
                )
        elif payload.scope == ExclusionScope.ROWS:
            if not payload.rows:
                raise HTTPException(400, "scope=rows requires non-empty rows")
        elif payload.scope == ExclusionScope.COLUMNS:
            if not payload.columns:
                raise HTTPException(
                    400, "scope=columns requires non-empty columns",
                )
        return ctx.exclusions_store.add(payload, annotator=annotator)

    @app.delete("/api/runs/{run_id}/exclusions/{exclusion_id}")
    def delete_exclusion(run_id: str, exclusion_id: str) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        tomb = ctx.exclusions_store.remove(exclusion_id, annotator=annotator)
        if tomb is None:
            raise HTTPException(404, f"exclusion not found: {exclusion_id}")
        return {"ok": True, "id": exclusion_id}

    # ---------------------------------------------------------------
    # Relation-review endpoints (approve / reject / add custom).
    # Backed by ``<run_dir>/relation_reviews.jsonl``. Latest decision
    # per ``relation_id`` (= species_key) wins. ``added`` decisions
    # carry a full ``relation`` snapshot so user-authored relations
    # survive when the LLM rerun changes the discovered set.
    # ---------------------------------------------------------------

    _relation_review_stores: Dict[str, RelationReviewStore] = {}
    _relation_review_lock = threading.Lock()

    def _relation_review_store(run_id: str) -> RelationReviewStore:
        rd = cache._safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        with _relation_review_lock:
            store = _relation_review_stores.get(run_id)
            if store is None:
                store = RelationReviewStore(rd / "relation_reviews.jsonl")
                _relation_review_stores[run_id] = store
            return store

    @app.get("/api/runs/{run_id}/relation-reviews")
    def list_relation_reviews(run_id: str) -> Dict[str, Any]:
        store = _relation_review_store(run_id)
        records = [r.model_dump(mode="json") for r in store.all()]
        latest = {
            rid: rec.model_dump(mode="json")
            for rid, rec in store.latest_by_relation().items()
        }
        return {"records": records, "latest": latest}

    @app.post("/api/runs/{run_id}/relation-reviews")
    def post_relation_review(
        run_id: str,
        payload: RelationReviewIn,
    ) -> Dict[str, Any]:
        if payload.decision == RelationReviewDecision.ADDED:
            if payload.relation is None:
                raise HTTPException(
                    400,
                    "decision=added requires a 'relation' snapshot in the body",
                )
            if not payload.relation.kind:
                raise HTTPException(400, "relation.kind is required")
            if not payload.relation.columns:
                raise HTTPException(400, "relation.columns must be non-empty")
        store = _relation_review_store(run_id)
        record = store.append(payload)
        return {
            "record": record.model_dump(mode="json"),
            "latest": {
                rid: rec.model_dump(mode="json")
                for rid, rec in store.latest_by_relation().items()
            },
        }

    # ---------------------------------------------------------------
    # Multi-table inconsistency labelling.
    # Backed by ``<run_dir>/multi_table_labels.jsonl``. Each finding is
    # keyed by ``violation_id`` (shared across the 1+ rows the LLM
    # detector emitted for a single inconsistency); a single human
    # verdict per finding is the source of truth, with full append-only
    # history. Inputs are read from ``multi_table_violations.jsonl``,
    # ``multi_table_groups.jsonl``, and ``multi_table_verifications.jsonl``
    # written by the relation-discovery pipeline.
    # ---------------------------------------------------------------

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

    @app.get("/api/runs/{run_id}/export/benchmark")
    def export_benchmark(run_id: str) -> FileResponse:
        ctx = cache.get(run_id)
        # Materialise once, drop excluded slices, then write.
        # Tracks the same exclusion semantics as /export/sources so
        # the benchmark and the sanitised CSV bundle never disagree.
        cells = materialize_label_benchmark(
            ctx.clusters, ctx.store,
            value_resolver=_make_value_resolver(ctx),
        )
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
        )
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
        )
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

        out_dir = ctx.run_dir / "sanitised_sources"
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = ctx.run_dir / f"sanitised_sources_{run_id}.zip"
        tmp_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
        with zipfile.ZipFile(
            tmp_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for source, src_path in sorted(ctx.sources.items()):
                if source in excluded_files:
                    continue
                drop_rows = per_source_rows.get(source) or set()
                drop_cols = per_source_cols.get(source) or set()
                if not drop_rows and not drop_cols:
                    # Fast path: copy the raw file verbatim so
                    # encoding / quoting / mixed dtypes survive.
                    if not src_path.exists():
                        continue
                    zf.write(src_path, arcname=source)
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
                # inspectable artefact under <run_dir>.
                out_path = out_dir / source
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                frame.to_csv(tmp_path, index=False)
                os.replace(tmp_path, out_path)
                zf.write(out_path, arcname=source)
        os.replace(tmp_zip, zip_path)
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"sources_{run_id}.zip",
        )

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
        # Relation-discovery prompt: surface the subset of error types
        # for which we have a discovery template, so the UI can filter
        # the chip list down to multi-column / multi-row / multi-table
        # types only.
        relations_types: List[str] = []
        relations_type_scopes: Dict[str, str] = {}
        try:
            import yaml
            disc_path = project_root / "prompts" / "discover_relations.yaml"
            if disc_path.is_file():
                raw = yaml.safe_load(disc_path.read_text(encoding="utf-8")) or {}
                types_node = raw.get("types") or {}
                if isinstance(types_node, dict):
                    relations_types = sorted(types_node.keys())
                    # Surface per-type scope so the launcher chip can
                    # show a tooltip explaining that ``table`` types
                    # only fire on multi-sheet workbooks.
                    for et_key, et_node in types_node.items():
                        if isinstance(et_node, dict):
                            scope = str(et_node.get("scope") or "row")
                            relations_type_scopes[et_key] = scope
        except Exception:  # noqa: BLE001
            relations_types = []
            relations_type_scopes = {}
        return {
            "project_root": str(project_root),
            "defaults": defaults,
            "error_types": load_taxonomy_keys(project_root),
            "relations_error_types": relations_types,
            "relations_error_type_scopes": relations_type_scopes,
            "datasets": list_datasets(project_root / "data"),
            "scopes": ["single-column", "multi-column", "multi-row", "multi-table"],
            "column_modes": ["auto", "single", "all"],
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
            elif kind == "relations":
                job = jobs.launch_relations(params)
            elif kind == "relation_report":
                run_id = payload.get("run_id")
                if not run_id:
                    raise HTTPException(400, "run_id is required for kind=relation_report")
                try:
                    job = jobs.launch_relation_report(run_id)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
            elif kind == "relation_validation":
                run_id = payload.get("run_id")
                if not run_id:
                    raise HTTPException(400, "run_id is required for kind=relation_validation")
                try:
                    job = jobs.launch_relation_validation(run_id, params=params)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
            elif kind == "column_verify":
                run_id = payload.get("run_id")
                if not run_id:
                    raise HTTPException(400, "run_id is required for kind=column_verify")
                try:
                    job = jobs.launch_column_verify(run_id, params=params)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
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
    error_types = list(c.get("error_types") or [])
    # For row_duplication clusters, surface the duplicate-group row
    # list (union of every detection's ``affected_rows`` plus the
    # primary row) so the queue panel can show "rows: 5, 12, 27" at
    # a glance instead of forcing the operator to open each cluster
    # and scan the context table.
    affected_rows: List[int] = []
    if "row_duplication" in error_types:
        seen: set = set()
        primary_row = c.get("row")
        if isinstance(primary_row, (int, float)):
            seen.add(int(primary_row))
        for d in c.get("detections") or []:
            for r in (d.get("affected_rows") or []) if isinstance(d, Mapping) else []:
                try:
                    seen.add(int(r))
                except (TypeError, ValueError):
                    continue
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


def _scalar(v: Any) -> str:
    if v is None:
        return ""
    return str(v)
