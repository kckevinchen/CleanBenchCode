"""Run management + report/xlsx endpoints under ``/api/runs`` (+ ``/{run_id}``).

Lists detection runs, returns/edits one run's metadata + source listing,
deletes a run, redirects to its report.html, and serves/previews the
source workbooks. The ``/`` root and the static ``/ui`` mount stay in the
app factory; this owns the run-scoped REST surface.
"""

from __future__ import annotations

import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ..runs import (
    _add_folder,
    _load_folders,
    _load_run_metadata,
    _remove_folder,
    _save_run_metadata,
    _update_folder,
    dataset_label_from_config,
)
from ..schema import LabelStatus, LabelTarget
from ._context import RouterContext

# Mirror the cap enforced in ``..runs._save_run_metadata`` so the
# "Copy of <name>" prefix we build here doesn't get silently truncated
# in a surprising spot.
_MAX_DISPLAY_NAME = 120


def _looks_like_stamp(token: str) -> bool:
    """True if ``token`` is a ``YYYYMMDDTHHMMSSZ`` run-id timestamp."""
    try:
        datetime.strptime(token, "%Y%m%dT%H%M%SZ")
        return True
    except ValueError:
        return False


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



def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache
    annotator = router_ctx.annotator
    safe_run_dir = router_ctx.safe_run_dir
    run_dir_or_404 = router_ctx.run_dir_or_404
    project_root = router_ctx.project_root
    outputs_dir = router_ctx.outputs_dir
    registry = router_ctx.registry

    @app.get("/api/runs")
    def list_runs() -> Dict[str, Any]:
        return {
            "project_root": str(project_root),
            "annotator": annotator,
            "runs": [r.to_dict() for r in registry.list()],
            # Folder definitions ship alongside the runs so the dashboard
            # can render the grouped view from a single fetch. Membership
            # lives on each run's ``folder_id`` (see ``RunInfo``).
            "folders": _load_folders(outputs_dir),
        }

    # ---------------------------------------------------------------
    # Virtual folders: group runs in the dashboard without moving any
    # run directory on disk. Folder *definitions* live in
    # ``outputs/folders.json``; per-run membership is a ``folder_id`` on
    # each run's ``run_metadata.json`` sidecar. Deleting a folder simply
    # orphans its members back into "Unfiled".
    # ---------------------------------------------------------------

    @app.get("/api/folders")
    def list_folders() -> Dict[str, Any]:
        return {"folders": _load_folders(outputs_dir)}

    def _resolve_run_dirs(run_ids: Any) -> List[Path]:
        """Validate every id, returning the dirs that actually exist.

        Path-validates each id via ``safe_run_dir`` (which raises 400 on
        traversal) and drops ids that no longer resolve to a run
        directory. Performs NO writes, so callers validate the whole
        batch up front and a bad (traversal) id fails the request before
        anything is mutated -- the move is all-or-nothing.
        """
        if not isinstance(run_ids, list):
            raise HTTPException(400, "run_ids must be a list")
        dirs: List[Path] = []
        for rid in run_ids:
            if not isinstance(rid, str):
                continue
            run_dir = safe_run_dir(rid)  # raises 400 on traversal
            if run_dir.is_dir():
                dirs.append(run_dir)
        return dirs

    def _assign_runs_to_folder(run_ids: Any, folder_id: str) -> int:
        """Write ``folder_id`` onto each existing run's sidecar.

        Validation happens for the whole batch first (see
        ``_resolve_run_dirs``) so a traversal id can't leave a partial
        write behind. Returns the number of runs updated.
        """
        dirs = _resolve_run_dirs(run_ids)
        for run_dir in dirs:
            _save_run_metadata(run_dir, {"folder_id": folder_id})
        return len(dirs)

    @app.post("/api/folders")
    def create_folder(payload: Dict[str, Any] = Body(default=None)) -> Dict[str, Any]:
        """Create a folder and (optionally) file an initial set of runs.

        Body: ``{"name": str, "run_ids": [str, ...]}`` - ``run_ids`` is
        optional, so this also creates an empty folder. Returns the new
        folder record plus the count of runs filed into it.
        """
        opts = payload or {}
        name = opts.get("name")
        if name is not None and not isinstance(name, str):
            raise HTTPException(400, "name must be a string")
        # Resolve membership BEFORE persisting the folder so a bad run_id
        # (traversal -> 400) can't leave an orphaned empty folder on disk.
        run_dirs = _resolve_run_dirs(opts.get("run_ids", []))
        folder = _add_folder(outputs_dir, name or "")
        for run_dir in run_dirs:
            _save_run_metadata(run_dir, {"folder_id": folder["id"]})
        return {"ok": True, "folder": folder, "moved": len(run_dirs)}

    @app.patch("/api/folders/{folder_id}")
    def update_folder(
        folder_id: str, payload: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        """Rename a folder and/or toggle its collapsed state.

        Body keys: ``name`` (str) and/or ``collapsed`` (bool). Unknown
        keys are ignored. 404 if the folder doesn't exist.
        """
        opts = payload or {}
        name = opts.get("name")
        if name is not None and not isinstance(name, str):
            raise HTTPException(400, "name must be a string")
        collapsed = opts.get("collapsed")
        if collapsed is not None and not isinstance(collapsed, bool):
            raise HTTPException(400, "collapsed must be a boolean")
        folder = _update_folder(
            outputs_dir, folder_id, name=name, collapsed=collapsed
        )
        if folder is None:
            raise HTTPException(404, f"folder not found: {folder_id}")
        return {"ok": True, "folder": folder}

    @app.delete("/api/folders/{folder_id}")
    def delete_folder(folder_id: str) -> Dict[str, Any]:
        """Delete a folder definition; member runs fall back to Unfiled.

        The run directories are untouched - we just clear ``folder_id``
        from every member's sidecar so no run keeps a dangling pointer.
        """
        # Clear membership first so a crash between the two writes leaves
        # runs orphaned (-> Unfiled on read) rather than pointing at a
        # folder that's already gone (which also reads as Unfiled, so the
        # order is belt-and-braces either way).
        members = [r.run_id for r in registry.list() if r.folder_id == folder_id]
        _assign_runs_to_folder(members, "")
        removed = _remove_folder(outputs_dir, folder_id)
        if not removed:
            raise HTTPException(404, f"folder not found: {folder_id}")
        return {"ok": True, "folder_id": folder_id, "unfiled": len(members)}

    @app.post("/api/folders/assign")
    def assign_folder(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Move a set of runs into a folder (or out to Unfiled).

        Body: ``{"run_ids": [str, ...], "folder_id": str|null}``. A null
        or empty ``folder_id`` clears membership (Unfiled). A non-empty
        ``folder_id`` must reference an existing folder (404 otherwise).
        """
        opts = payload or {}
        folder_id = opts.get("folder_id") or ""
        if not isinstance(folder_id, str):
            raise HTTPException(400, "folder_id must be a string or null")
        if folder_id:
            known = {f["id"] for f in _load_folders(outputs_dir)}
            if folder_id not in known:
                raise HTTPException(404, f"folder not found: {folder_id}")
        moved = _assign_runs_to_folder(opts.get("run_ids", []), folder_id)
        return {"ok": True, "moved": moved, "folder_id": folder_id}

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str, force: bool = False) -> Dict[str, Any]:
        # _safe_run_dir enforces both syntactic checks AND that the
        # resolved path is a direct child of outputs_dir, so a symlink
        # in outputs/<id> can't redirect rmtree outside the workbench.
        run_dir = safe_run_dir(run_id)
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

    @app.post("/api/runs/{run_id}/duplicate")
    def duplicate_run(
        run_id: str, payload: Dict[str, Any] = Body(default=None)
    ) -> Dict[str, Any]:
        """Copy an existing run directory into a fresh ``outputs/<id>/``.

        The clone gets a new timestamp-prefixed run-id (so it sorts to
        the top and never collides with the source's ``labels.jsonl``
        key space) while reusing the source's model/dataset suffix. By
        default the operator's existing labels are copied too; pass
        ``{"reset_labels": true}`` to start the copy with a clean
        labelling slate (drops every operator-authored store:
        ``labels.jsonl``, the column-cluster label store
        ``column_clusters.jsonl``, ``exclusions.jsonl`` and
        ``cell_notes.jsonl`` -- but keeps the LLM-side
        ``column_cluster_drafts.jsonl`` proposals, which are detection
        output, not labels). The clone is always unstarred and its
        ``display_name`` is set to ``"Copy of <name>"`` so the two are
        easy to tell apart in the dashboard.
        """
        src_dir = run_dir_or_404(run_id)
        opts = payload or {}
        reset_labels = bool(opts.get("reset_labels", False))

        # New run-id: replace the source's leading ``<timestamp>Z`` token
        # with a fresh one so the clone is unique and sorts first. If the
        # source id has no recognisable timestamp prefix we just prepend
        # one. The suffix is reused verbatim - it is already a valid,
        # path-safe directory-name component (it IS the source dir name).
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        parts = run_id.split("_", 1)
        suffix = parts[1] if len(parts) == 2 and _looks_like_stamp(parts[0]) else run_id
        new_id = f"{stamp}_{suffix}"
        dest_dir = safe_run_dir(new_id)
        # Guard against the (sub-second) chance the target already exists.
        n = 1
        while dest_dir.exists():
            new_id = f"{stamp}_{suffix}_copy{n}"
            dest_dir = safe_run_dir(new_id)
            n += 1

        # Don't copy the per-run cache/tmp sidecars; everything else
        # (clusters, summary, report, raw responses, column-cluster
        # stores, labels) is part of the run's labellable state.
        def _ignore(_dir: str, names: List[str]) -> List[str]:
            return [nm for nm in names if nm.endswith(".tmp")]

        # A mid-copy failure (disk full, I/O error) must not leave a
        # half-written run dir behind for the registry to list as a
        # corrupt run -- clean up the partial copy and surface a 500.
        try:
            shutil.copytree(src_dir, dest_dir, ignore=_ignore)
        except OSError as exc:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(500, f"duplicate copy failed: {exc}") from exc

        if reset_labels:
            # Drop every operator-authored store so the clone starts with
            # a clean labelling slate. ``column_clusters.jsonl`` is the
            # real column-cluster (inconsistency) verdict store; the LLM
            # ``column_cluster_drafts.jsonl`` proposals are kept.
            for rel in (
                "labels.jsonl",
                "column_clusters.jsonl",
                "exclusions.jsonl",
                "cell_notes.jsonl",
            ):
                p = dest_dir / rel
                if p.exists():
                    p.unlink()

        # The clone is a separate working copy: never inherit the
        # source's starred pin or folder membership (copytree carried the
        # source's run_metadata.json over, so we must reset both
        # explicitly), and label it so the two are obvious. The clone
        # starts Unfiled - the operator files it where they want it.
        src_meta = _load_run_metadata(src_dir)
        base_name = src_meta.get("display_name") or run_id
        _save_run_metadata(
            dest_dir,
            {
                "display_name": f"Copy of {base_name}"[:_MAX_DISPLAY_NAME],
                "starred": False,
                "notes": src_meta.get("notes", ""),
                "folder_id": "",
            },
        )

        info = registry.get(new_id)
        if info is None:
            raise HTTPException(500, f"duplicate failed to materialise: {new_id}")
        return {"ok": True, "run_id": new_id, "source_run_id": run_id, "run": info.to_dict()}

    @app.patch("/api/runs/{run_id}")
    def patch_run(run_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Update operator-editable run metadata (display_name, starred,
        notes). Body keys are merged with existing sidecar - unknown
        keys are silently dropped. Returns the run's full updated
        ``RunInfo`` payload so the client can replace the row in place.
        """
        run_dir = run_dir_or_404(run_id)
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
            "dataset": dataset_label_from_config(
                (ctx.summary or {}).get("config") or {}
            ),
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
