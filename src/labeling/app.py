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
    _load_drafts_for_column,
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
from .row_verdict_store import RowVerdictStore
from .exclusions_store import ExclusionStore
from .multi_table_label_store import MultiTableLabelStore
from .column_cluster_store import ColumnClusterStore
from .column_cluster_queue import iter_column_cluster_targets
from .routers import cell_notes as cell_notes_router
from .routers import row_verdicts as row_verdicts_router
from .routers import clusters as clusters_router
from .routers import column_clusters as column_clusters_router
from .routers import correction_jobs as correction_jobs_router
from .routers import compare as compare_router
from .routers import exclusions as exclusions_router
from .routers import exports as exports_router
from .routers import labels as labels_router
from .routers import launcher as launcher_router
from .routers import multi_table as multi_table_router
from .routers import propagation as propagation_router
from .routers import relation_reviews as relation_reviews_router
from .routers import runs as runs_router
from .routers import sources as sources_router
from .routers import stats as stats_router
from .routers import trivial_corrections as trivial_corrections_router
from .routers import verifications as verifications_router
from .routers._context import RouterContext

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

# pbi_1000 stores 1000 workbooks (some multi-sheet) flat in one directory,
# encoding the workbook in the filename as ``<workbook_id>_<sheet>.csv``.
# Like Bing21 the workbook lives in the filename, not a parent dir, so we
# lift the numeric id out when the parent-dir heuristic would otherwise
# collapse the whole corpus into one group (see RunContext.__init__).
_FLAT_PREFIX_WB_RE = re.compile(r"^(?P<wb>\d{6,})_.+\.csv$", re.IGNORECASE)


def _workbook_for(pp: Path) -> str:
    m = _BING21_FLAT_RE.match(pp.name)
    if m:
        return m.group("wb")
    return pp.parent.name


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

        # Flat corpora that encode the workbook in the *filename* rather
        # than a parent directory (pbi_1000: ``<workbook_id>_<sheet>.csv``)
        # would otherwise collapse to a single ``pbi_1000`` workbook and the
        # UI would lump every CSV into one group. When the parent-dir
        # heuristic yields a single workbook for a multi-CSV run, recover
        # per-workbook keys from the filename prefix -- but only adopt the
        # refinement when it actually splits the run into >=2 workbooks, so
        # genuine single-table flat corpora stay untouched.
        distinct_parents = {wb for wb in self.source_workbook.values() if wb}
        if len(distinct_parents) <= 1 and len(self.source_workbook) > 1:
            refined = {
                name: (m.group("wb") if (m := _FLAT_PREFIX_WB_RE.match(name)) else wb)
                for name, wb in self.source_workbook.items()
            }
            if len({wb for wb in refined.values() if wb}) >= 2:
                self.source_workbook = refined

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
        # Per-source file signature (mtime_ns, size) captured when the frame
        # was cached. ``load_frame`` re-stats the CSV on every call and reloads
        # when the signature changes, so an on-disk edit (e.g. the dataset
        # being normalized AFTER a label run) is picked up instead of serving
        # a stale memoized frame.
        self._frame_sig: Dict[str, tuple] = {}
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
        # Per-row verdict overrides: an explicit verdict for a single
        # (source, row, column) cell that supersedes the enclosing
        # cluster's fanned-out verdict. Lets the operator label each row
        # of an fd_violation conflict group independently. See
        # :mod:`src.labeling.row_verdict_store` for the storage format.
        self.row_verdict_store = RowVerdictStore(
            self.run_dir / "row_verdicts.jsonl"
        )
        # Per-source exclusions (whole CSV / specific rows / specific
        # columns the operator marked as too messy to label). Loaded
        # eagerly so the UI can render existing strikethroughs without
        # an extra round-trip.
        self.exclusions_store = ExclusionStore(self.run_dir / "exclusions.jsonl")
        # Column-cluster (inconsistency) verdicts. One store instance per
        # run so the core label endpoint's writes and the column-cluster
        # router's reads share the same in-memory latest-map.
        self.column_cluster_store = ColumnClusterStore(
            self.run_dir / "column_clusters.jsonl"
        )

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
            path = self.sources.get(source)
            if path is None or not path.exists():
                raise FileNotFoundError(f"source not found: {source}")
            # Current on-disk signature. If it matches the cached frame's
            # signature the file is unchanged and we serve the memoised frame;
            # otherwise the CSV was edited on disk (e.g. the dataset was
            # normalized after this label run, removing excess whitespace) and
            # we MUST reload so taggers / highlights read the true cell value
            # rather than a stale snapshot.
            try:
                st = path.stat()
                sig: tuple = (st.st_mtime_ns, st.st_size)
            except OSError:
                sig = None
            cached = self._frame_cache.get(source)
            if cached is not None and sig is not None and self._frame_sig.get(source) == sig:
                return cached
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
            if sig is not None:
                self._frame_sig[source] = sig
            else:
                self._frame_sig.pop(source, None)
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
    row_verdicts_router.register(app, router_ctx)
    clusters_router.register(app, router_ctx)
    column_clusters_router.register(app, router_ctx)
    correction_jobs_router.register(app, router_ctx)
    trivial_corrections_router.register(app, router_ctx)
    labels_router.register(app, router_ctx)
    exclusions_router.register(app, router_ctx)
    exports_router.register(app, router_ctx)
    multi_table_router.register(app, router_ctx)
    propagation_router.register(app, router_ctx)
    runs_router.register(app, router_ctx)
    sources_router.register(app, router_ctx)
    stats_router.register(app, router_ctx)
    verifications_router.register(app, router_ctx)
    relation_reviews_router.register(app, router_ctx)
    compare_router.register(app, router_ctx)

    # =====================================================================
    # Runs listing + per-run endpoints
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


