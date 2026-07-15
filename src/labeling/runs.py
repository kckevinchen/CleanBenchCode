"""Experiment-launcher support: discover past runs + manage subprocess jobs.

Responsibilities
================

* :class:`RunRegistry`  - scan ``outputs/`` for detection-run dirs,
  expose per-run metadata (model, created timestamp, cluster count,
  artifact presence).
* :class:`JobManager`   - spawn ``python -m experiments.run_cleaning_detection``
  (or ``experiments.build_report``) as a child process; capture stdout
  to a job log file; non-blocking tail; cancel via SIGTERM.
* :func:`build_detection_argv` / :func:`build_report_argv` - translate a
  structured request (dict from the UI form) into a safe argv list.

All state is persisted to ``<project_root>/logs/jobs/<job_id>.{log,json}``
so the server can be killed (or the SSH session can drop) without
losing the job. On next startup the manager rehydrates every job it
finds on disk, checks ``pid`` liveness for "running" ones, and
resumes watching them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    """Thin summary of an on-disk run directory."""

    run_id: str                     # directory name
    path: Path
    created: Optional[str]          # ISO timestamp (from mtime)
    model: Optional[str]
    mode: Optional[str]
    has_summary: bool
    has_clusters: bool
    has_report: bool
    has_labels: bool
    clusters: int = 0
    issues: int = 0
    dataset: Optional[str] = None       # short human label for the corpus
    n_sources: int = 0                  # number of CSV files the run targeted
    normalize_input: bool = False       # ablation flag (text_normalize)
    # Run kind: "detection" (cleaning pipeline) | "relations" (schema
    # discovery). Inferred from artifacts on disk - run dirs predate
    # the "kind" concept so we don't rely on metadata.
    kind: str = "detection"
    # Relation-discovery runs only.
    has_relations: bool = False
    n_relations: int = 0
    n_passes: Optional[int] = None
    n_arms: Optional[int] = None
    n_saturated_arms: Optional[int] = None
    # Multi-table cross-sheet pass artifacts (only populated by
    # ``run_relation_discovery`` when --error-types includes
    # inter_table_inconsistency). Exposed so the runs list can show a
    # "Label MT" entry point.
    has_multi_table: bool = False
    n_multi_table_findings: int = 0
    # Web-verification artifacts (output of ``experiments.run_web_verify``).
    # Exposed so the runs list can flag which detection runs already
    # have web verifications and how many cells are covered.
    has_web_verifications: bool = False
    n_web_verifications: int = 0
    # Column-clusters (inconsistency error type) artifacts.
    # ``has_column_cluster_drafts``: did the LLM proposer
    # (``experiments.run_column_clustering``) write
    # ``column_cluster_drafts.jsonl``?  ``has_column_cluster_labels``:
    # did the labelling UI write any ``column_clusters.jsonl``
    # records? The runs dashboard shows a "Column board" link when
    # either is true; the counts let the UI render a badge.
    has_column_cluster_drafts: bool = False
    n_column_cluster_drafts: int = 0
    has_column_cluster_labels: bool = False
    n_column_cluster_labels: int = 0
    # Operator-editable sidecar metadata (``run_metadata.json``).
    # Empty defaults mean "no metadata written yet" - the runs
    # dashboard falls back to displaying just ``run_id`` and treats
    # the run as unstarred / unpinned. See ``_load_run_metadata``.
    display_name: str = ""
    starred: bool = False
    notes: str = ""
    # Virtual-folder membership. ``""`` means "unfiled" (the default).
    # Folders are purely an organisational grouping layer in the runs
    # dashboard - the run directory itself never moves on disk, so
    # run-id-keyed paths (labels, reports) stay valid. The set of folder
    # definitions lives in ``outputs/folders.json``; this field is the
    # per-run pointer into it (see ``_load_folders``).
    folder_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "created": self.created,
            "model": self.model,
            "mode": self.mode,
            "has_summary": self.has_summary,
            "has_clusters": self.has_clusters,
            "has_report": self.has_report,
            "has_labels": self.has_labels,
            "clusters": self.clusters,
            "issues": self.issues,
            "dataset": self.dataset,
            "n_sources": self.n_sources,
            "normalize_input": self.normalize_input,
            "kind": self.kind,
            "has_relations": self.has_relations,
            "n_relations": self.n_relations,
            "n_passes": self.n_passes,
            "n_arms": self.n_arms,
            "n_saturated_arms": self.n_saturated_arms,
            "has_multi_table": self.has_multi_table,
            "n_multi_table_findings": self.n_multi_table_findings,
            "has_web_verifications": self.has_web_verifications,
            "n_web_verifications": self.n_web_verifications,
            "has_column_cluster_drafts": self.has_column_cluster_drafts,
            "n_column_cluster_drafts": self.n_column_cluster_drafts,
            "has_column_cluster_labels": self.has_column_cluster_labels,
            "n_column_cluster_labels": self.n_column_cluster_labels,
            "display_name": self.display_name,
            "starred": self.starred,
            "notes": self.notes,
            "folder_id": self.folder_id,
        }


# ---------------------------------------------------------------------------
# Operator-editable run metadata sidecar (``run_metadata.json``)
# ---------------------------------------------------------------------------
#
# Lives at ``outputs/<run_id>/run_metadata.json``. Holds fields the
# operator can edit from the runs dashboard so they can rename ("pin")
# runs without renaming the immutable timestamp-based directory and
# without losing the run-id used by labels.jsonl. Schema is
# deliberately tiny and forward-compatible: unknown keys are ignored
# on read so future fields don't break older readers, and missing
# files mean "no metadata" rather than an error.

_RUN_METADATA_FILE = "run_metadata.json"
_MAX_DISPLAY_NAME = 120
_MAX_NOTES = 2000
_MAX_FOLDER_ID = 64


def _load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Read the sidecar; return empty dict for missing/corrupt files."""
    path = run_dir / _RUN_METADATA_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("run_metadata.json parse fail for %s: %s", run_dir, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    dn = raw.get("display_name")
    if isinstance(dn, str):
        out["display_name"] = dn.strip()[:_MAX_DISPLAY_NAME]
    starred = raw.get("starred")
    if isinstance(starred, bool):
        out["starred"] = starred
    notes = raw.get("notes")
    if isinstance(notes, str):
        out["notes"] = notes[:_MAX_NOTES]
    folder_id = raw.get("folder_id")
    if isinstance(folder_id, str):
        out["folder_id"] = folder_id.strip()[:_MAX_FOLDER_ID]
    return out


def _save_run_metadata(run_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``meta`` into the sidecar and write atomically.

    Only known keys (``display_name``, ``starred``, ``notes``,
    ``folder_id``) are persisted. Returns the full post-merge metadata
    dict. Writes use a tmp-file + ``os.replace`` so a crashing server
    can't leave a half-written JSON behind.
    """
    current = _load_run_metadata(run_dir)
    merged: Dict[str, Any] = {
        "display_name": current.get("display_name", ""),
        "starred": bool(current.get("starred", False)),
        "notes": current.get("notes", ""),
        "folder_id": current.get("folder_id", ""),
    }
    if "display_name" in meta:
        dn = meta["display_name"] or ""
        if not isinstance(dn, str):
            raise ValueError("display_name must be a string")
        merged["display_name"] = dn.strip()[:_MAX_DISPLAY_NAME]
    if "starred" in meta:
        merged["starred"] = bool(meta["starred"])
    if "notes" in meta:
        nt = meta["notes"] or ""
        if not isinstance(nt, str):
            raise ValueError("notes must be a string")
        merged["notes"] = nt[:_MAX_NOTES]
    if "folder_id" in meta:
        fid = meta["folder_id"] or ""
        if not isinstance(fid, str):
            raise ValueError("folder_id must be a string")
        merged["folder_id"] = fid.strip()[:_MAX_FOLDER_ID]

    path = run_dir / _RUN_METADATA_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return merged


# ---------------------------------------------------------------------------
# Virtual folders registry (``outputs/folders.json``)
# ---------------------------------------------------------------------------
#
# Folders are an organisational grouping layer for the runs dashboard.
# They do NOT move run directories on disk - membership is a ``folder_id``
# string stored in each run's ``run_metadata.json`` sidecar (see above),
# and the folder *definitions* (id -> name + collapse state) live in a
# single ``outputs/folders.json`` registry. A run whose ``folder_id``
# doesn't match any live folder is treated as unfiled, so deleting a
# folder definition is enough to orphan its members back to "Unfiled".
# Schema is deliberately small and forward-compatible: unknown keys are
# dropped on read, a missing/corrupt file means "no folders".

_FOLDERS_FILE = "folders.json"
_MAX_FOLDER_NAME = 120

# Serialises the load->mutate->save of the (single, shared) folders.json
# registry. Sync FastAPI handlers run in a threadpool, so two folder
# mutations can overlap; without this lock the second writer's full-list
# overwrite would drop the first's change (lost update). Mirrors the
# JobManager's threading.Lock convention for shared on-disk state.
_FOLDERS_LOCK = threading.Lock()


def _normalise_folder(raw: Any) -> Optional[Dict[str, Any]]:
    """Coerce one stored folder record into a clean dict, or ``None``."""
    if not isinstance(raw, dict):
        return None
    fid = raw.get("id")
    if not isinstance(fid, str) or not fid.strip():
        return None
    name = raw.get("name")
    name = name.strip()[:_MAX_FOLDER_NAME] if isinstance(name, str) else ""
    created = raw.get("created")
    return {
        "id": fid.strip()[:_MAX_FOLDER_ID],
        "name": name,
        "created": created if isinstance(created, str) else None,
        "collapsed": bool(raw.get("collapsed", False)),
    }


def _load_folders(outputs_dir: Path) -> List[Dict[str, Any]]:
    """Read the folder registry; ``[]`` for a missing/corrupt file.

    Accepts either ``{"folders": [...]}`` or a bare ``[...]`` on disk so
    a hand-edited file in the older shape still loads. Order is preserved
    (newest folders are stored first by :func:`_add_folder`).
    """
    path = Path(outputs_dir) / _FOLDERS_FILE
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("folders.json parse fail for %s: %s", outputs_dir, exc)
        return []
    items = raw.get("folders") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for it in items:
        folder = _normalise_folder(it)
        if folder is None or folder["id"] in seen:
            continue
        seen.add(folder["id"])
        out.append(folder)
    return out


def _save_folders(outputs_dir: Path, folders: List[Dict[str, Any]]) -> None:
    """Atomically persist the folder registry (tmp-file + ``os.replace``)."""
    path = Path(outputs_dir) / _FOLDERS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [f for f in (_normalise_folder(it) for it in folders) if f]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"folders": cleaned}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _add_folder(outputs_dir: Path, name: str) -> Dict[str, Any]:
    """Create a new folder (newest-first) and return its record."""
    with _FOLDERS_LOCK:
        folders = _load_folders(outputs_dir)
        existing = {f["id"] for f in folders}
        fid = uuid.uuid4().hex[:12]
        while fid in existing:
            fid = uuid.uuid4().hex[:12]
        folder = {
            "id": fid,
            "name": (name or "").strip()[:_MAX_FOLDER_NAME],
            "created": datetime.now(timezone.utc).isoformat(),
            "collapsed": False,
        }
        folders.insert(0, folder)
        _save_folders(outputs_dir, folders)
        return folder


def _update_folder(
    outputs_dir: Path,
    folder_id: str,
    *,
    name: Optional[str] = None,
    collapsed: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Rename and/or set the collapse flag of one folder.

    Returns the updated record, or ``None`` if no folder matched.
    """
    with _FOLDERS_LOCK:
        folders = _load_folders(outputs_dir)
        target: Optional[Dict[str, Any]] = None
        for f in folders:
            if f["id"] == folder_id:
                target = f
                break
        if target is None:
            return None
        if name is not None:
            target["name"] = name.strip()[:_MAX_FOLDER_NAME]
        if collapsed is not None:
            target["collapsed"] = bool(collapsed)
        _save_folders(outputs_dir, folders)
        return target


def _remove_folder(outputs_dir: Path, folder_id: str) -> bool:
    """Drop a folder definition. Returns ``True`` if one was removed.

    Member runs are NOT touched here; an orphaned ``folder_id`` is
    treated as unfiled on read. Callers that want member rows to fall
    back into "Unfiled" cleanly should also clear each member's
    ``folder_id`` sidecar (the API layer does this).
    """
    with _FOLDERS_LOCK:
        folders = _load_folders(outputs_dir)
        kept = [f for f in folders if f["id"] != folder_id]
        if len(kept) == len(folders):
            return False
        _save_folders(outputs_dir, kept)
        return True


def dataset_label_from_config(cfg: Dict[str, Any]) -> Optional[str]:
    """Derive a short dataset/corpus label from a run's ``config`` block.

    Prefers the directory shared by *all* CSV paths since that's the
    corpus name (e.g. ``curated_pbi_500_dedup``); falls back to the
    basename for single-file runs. Returns ``None`` when the config
    carries no CSV path. Shared by the runs-list parser and the
    run-detail endpoint so both surfaces show the same label.
    """
    paths = cfg.get("csv_paths") or []
    # Directory-mode runs record the corpus directory directly.
    explicit_dir = cfg.get("csv_path")
    if explicit_dir:
        return Path(explicit_dir).name
    if not paths:
        return None
    if len(paths) > 1:
        # Multi-file runs: label by the directory common to *all* CSVs.
        # ``paths[0].parent`` is wrong for nested corpora that keep each
        # table in its own sub-dir (curated_pbi_500_dedup/<table>/<table>.csv
        # -> we want "curated_pbi_500_dedup", not the per-table "<table>").
        try:
            return Path(os.path.commonpath([str(p) for p in paths])).name
        except ValueError:
            # Mixed drives / relative paths -- fall back to first parent.
            return Path(paths[0]).parent.name
    return Path(paths[0]).name


def _parse_run_dir(path: Path) -> RunInfo:
    summary_path = path / "summary.json"
    clusters_path = path / "clusters.json"
    relations_path = path / "relations.jsonl"
    is_relations = relations_path.exists() or (path / "discovery_reports.json").exists()
    model: Optional[str] = None
    mode: Optional[str] = None
    clusters = 0
    issues = 0
    dataset: Optional[str] = None
    n_sources = 0
    normalize_input = False
    n_relations = 0
    n_passes: Optional[int] = None
    n_arms: Optional[int] = None
    n_saturated_arms: Optional[int] = None
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            cfg = data.get("config") or {}
            model = data.get("model") or cfg.get("model")
            mode = data.get("mode") or cfg.get("mode")
            issues = int(data.get("total_issues") or 0)
            # Authoritative cluster count from the pipeline summary. Using
            # it avoids json.loads-ing the (often multi-MB) clusters.json
            # array on every runs-list request -- that array parse was the
            # entire cost of a slow /api/runs (1.3 GB across ~150 runs).
            clusters = int(data.get("clusters_total") or 0)
            # Discovery summaries store top-level mode/model (single_pass /
            # adaptive) and per-pass / per-arm aggregates.
            if is_relations:
                n_relations = int(data.get("n_relations") or 0)
                n_sources = int(data.get("n_sources") or 0)
                n_passes = data.get("n_passes")
                n_arms = data.get("n_arms")
                n_saturated_arms = data.get("n_saturated_arms")
            # Derive a short dataset label from the run config. Prefer the
            # parent directory of the CSV path(s) since that's the corpus
            # name (e.g. "curated_bing21_grouped"); fall back to the basename.
            paths = cfg.get("csv_paths") or []
            dataset = dataset_label_from_config(cfg)
            if not is_relations:
                n_sources = int(cfg.get("source_count") or len(paths) or 0)
            ni = cfg.get("normalize_input")
            if isinstance(ni, dict):
                normalize_input = bool(ni.get("enabled"))
            else:
                normalize_input = bool(ni)
        except Exception as exc:  # noqa: BLE001
            logger.debug("summary.json parse fail for %s: %s", path, exc)
    # Fallback cluster count when summary.json lacked ``clusters_total``
    # (older runs / relation runs). Prefer a cheap line count of the
    # line-delimited clusters.jsonl; only parse the clusters.json array as
    # a last resort, since that array can be tens of MB.
    if not clusters:
        clusters_jsonl = path / "clusters.jsonl"
        if clusters_jsonl.exists():
            try:
                with clusters_jsonl.open("r", encoding="utf-8") as fh:
                    clusters = sum(1 for line in fh if line.strip())
            except OSError:
                pass
        if not clusters and clusters_path.exists():
            try:
                raw = json.loads(clusters_path.read_text(encoding="utf-8"))
                clusters = len(raw) if isinstance(raw, list) else 0
            except Exception:  # noqa: BLE001
                pass
    if is_relations and not n_relations and relations_path.exists():
        # Fallback: count lines in relations.jsonl when summary is absent.
        try:
            with relations_path.open("r", encoding="utf-8") as fh:
                n_relations = sum(1 for line in fh if line.strip())
        except OSError:
            pass
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        created = mtime.isoformat()
    except OSError:
        created = None

    # Multi-table cross-sheet artifacts. ``multi_table_violations.jsonl``
    # is one line per (sheet, row) per finding; many findings span 2+
    # rows, so we count UNIQUE violation_ids for the headline figure
    # (= number of distinct findings the user has to review).
    mt_path = path / "multi_table_violations.jsonl"
    has_multi_table = mt_path.exists()
    n_multi_table_findings = 0
    if has_multi_table:
        try:
            seen_vids: set = set()
            with mt_path.open("r", encoding="utf-8") as fh:
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
                        seen_vids.add(vid)
            n_multi_table_findings = len(seen_vids)
        except OSError:
            pass

    # Web-verification artifact (``experiments.run_web_verify`` output).
    # JSON array of one record per gate-eligible cluster; we count the
    # length so the runs list can render a "verified N/M" badge.
    wv_path = path / "web_verifications.json"
    has_web_verifications = wv_path.exists()
    n_web_verifications = 0
    if has_web_verifications:
        try:
            wv_raw = json.loads(wv_path.read_text(encoding="utf-8"))
            if isinstance(wv_raw, list):
                n_web_verifications = len(wv_raw)
        except (OSError, ValueError):
            pass

    # Column-cluster labelling artifacts (inconsistency error type).
    # Drafts are written by ``experiments.run_column_clustering``;
    # labels are written by the labelling-UI POST endpoint. Both are
    # append-only JSONL where the latest record per (source, column)
    # wins, so the headline counts come from the de-duplicated set.
    ccd_path = path / "column_cluster_drafts.jsonl"
    has_column_cluster_drafts = ccd_path.exists()
    n_column_cluster_drafts = 0
    if has_column_cluster_drafts:
        try:
            seen: set = set()
            with ccd_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    src = rec.get("source")
                    col = rec.get("column")
                    if src is not None and col is not None:
                        seen.add((src, col))
            n_column_cluster_drafts = len(seen)
        except OSError:
            pass
    ccl_path = path / "column_clusters.jsonl"
    has_column_cluster_labels = ccl_path.exists()
    n_column_cluster_labels = 0
    if has_column_cluster_labels:
        try:
            seen2: set = set()
            with ccl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    src = rec.get("source")
                    col = rec.get("column")
                    if src is not None and col is not None:
                        seen2.add((src, col))
            n_column_cluster_labels = len(seen2)
        except OSError:
            pass

    meta = _load_run_metadata(path)

    return RunInfo(
        run_id=path.name,
        path=path,
        created=created,
        model=model,
        mode=mode,
        has_summary=summary_path.exists(),
        has_clusters=clusters_path.exists(),
        has_report=(path / "report.html").exists(),
        has_labels=(path / "labels.jsonl").exists(),
        clusters=clusters,
        issues=issues,
        dataset=dataset,
        n_sources=n_sources,
        normalize_input=normalize_input,
        kind="relations" if is_relations else "detection",
        has_relations=is_relations,
        n_relations=n_relations,
        n_passes=n_passes,
        n_arms=n_arms,
        n_saturated_arms=n_saturated_arms,
        has_multi_table=has_multi_table,
        n_multi_table_findings=n_multi_table_findings,
        has_web_verifications=has_web_verifications,
        n_web_verifications=n_web_verifications,
        has_column_cluster_drafts=has_column_cluster_drafts,
        n_column_cluster_drafts=n_column_cluster_drafts,
        has_column_cluster_labels=has_column_cluster_labels,
        n_column_cluster_labels=n_column_cluster_labels,
        display_name=meta.get("display_name", ""),
        starred=bool(meta.get("starred", False)),
        notes=meta.get("notes", ""),
        folder_id=meta.get("folder_id", ""),
    )


class RunRegistry:
    """Lists detection-run directories under ``outputs/``."""

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = Path(outputs_dir)

    def list(self) -> List[RunInfo]:
        if not self.outputs_dir.is_dir():
            return []
        runs: List[RunInfo] = []
        for child in sorted(self.outputs_dir.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            # Heuristic: real runs have at least summary.json OR
            # clusters.json OR (for relation-discovery runs) relations.jsonl.
            if (
                not (child / "summary.json").exists()
                and not (child / "clusters.json").exists()
                and not (child / "relations.jsonl").exists()
            ):
                # Still include it if a job is currently writing - surface
                # the dir but mark artifacts absent.
                pass
            runs.append(_parse_run_dir(child))
        return runs

    def get(self, run_id: str) -> Optional[RunInfo]:
        path = self.outputs_dir / run_id
        if not path.is_dir():
            return None
        return _parse_run_dir(path)


# ---------------------------------------------------------------------------
# Launch request → argv
# ---------------------------------------------------------------------------


_BOOL_KEYS_DETECT = {
    "no_synthesis": "--no-synthesis",
    "no_synthesise_singletons": "--no-synthesise-singletons",
    # Good-Turing booleans - handled below with paired --no-* flags.
}


# Path keys that the launcher accepts from the UI / API and forwards to
# child processes via CLI flags. Each entry must be validated against
# the project root before the subprocess is spawned, otherwise an
# attacker who can hit the unauth API could read/write arbitrary
# paths using the server's credentials.
_DETECTION_PATH_KEYS = ("config", "csv", "csv_dir", "output_dir")
_DETECTION_GLOB_KEYS = ("csv_glob",)  # patterns, not paths
_CURATE_PATH_KEYS = ("source_dir", "output_dir")
_RELATIONS_PATH_KEYS = ("config", "csv", "csv_dir", "output_dir",
                        "discovery_prompt", "aggregate_prompt",
                        "validate_prompt")
_RELATIONS_GLOB_KEYS = ("csv_glob",)
# Step 2 (relation_detection) form keys that resolve to filesystem
# paths under the project root. ``discovery_run`` is the Step-1 dir;
# ``output_dir`` is where the new detection run is written.
_RELATION_DETECTION_PATH_KEYS = (
    "config", "discovery_run", "output_dir",
)


# Non-single-column types the cleaning runner detects in-process via a
# dedicated handler rather than delegating to the relation-discovery
# pipeline. Mirrors ``run_cleaning_detection.DEDICATED_HANDLER_TYPES``;
# kept here so the launcher can decide routing without importing the
# heavy CLI module. ``tests/test_relation_only_launch_routing.py`` guards
# both copies against drift.
_DEDICATED_HANDLER_TYPES = frozenset({"row_duplication", "inter_table_inconsistency"})


def _scope_prefix(category: str) -> str:
    """Lower-cased scope prefix of an error-type category (text before
    the first ``/``). E.g. ``"Multi-Column / Intra-Record"`` ->
    ``"multi-column"``. Mirrors ``run_cleaning_detection._scope_of``."""
    head = (category or "").split("/", 1)[0]
    return " ".join(head.lower().split())


def _load_config_for_payload(
    payload: Dict[str, Any], project_root: Path,
) -> Dict[str, Any]:
    """Load the experiment config a launch payload would use.

    Honours an explicit ``payload['config']`` (resolved against the
    project root) and falls back to the default ``config/experiment.yaml``.
    Fails soft to ``{}`` on any read/parse error so routing decisions
    never block a launch.
    """
    cfg = payload.get("config")
    if cfg:
        import yaml  # deferred

        p = Path(str(cfg))
        if not p.is_absolute():
            p = project_root / p
        try:
            if p.is_file():
                return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return {}
    return load_config_defaults(project_root)


def _is_relation_only_detection(
    payload: Dict[str, Any], project_root: Path,
) -> bool:
    """True when a *detection* launch should be handed off to the
    relation-discovery pipeline instead.

    The cleaning runner routes every non-single-column type that lacks a
    dedicated in-process handler (``row_duplication`` /
    ``inter_table_inconsistency``) to ``run_relation_discovery`` when
    ``inter_column_pipeline.enabled``. If *all* selected types are routed
    that way, the per-chunk detection run has nothing to do and produces
    an empty report -- so we launch the relation pipeline (which does
    discover -> validate -> detect in one invocation) instead.

    Returns ``False`` for the default selection (no ``error_types`` -> the
    full taxonomy, which includes single-column types), for any mixed
    selection that still gives the per-chunk detector real work, and when
    the inter-column pipeline is disabled.
    """
    raw = payload.get("error_types")
    if not raw:
        return False  # default = full taxonomy (has single-column work)
    keys = [k.strip() for k in str(raw).split(",") if k.strip()]
    if not keys:
        return False
    # Respect the master switch: when the inter-column pipeline is off the
    # per-chunk detector owns these types, so detection is the right path.
    config = _load_config_for_payload(payload, project_root)
    inter_column_cfg = config.get("inter_column_pipeline") or {}
    if not bool(inter_column_cfg.get("enabled", True)):
        return False
    category_by_key = {
        str(t.get("key")): str(t.get("category") or "")
        for t in load_taxonomy_keys(project_root)
    }

    def _routed_to_relations(key: str) -> bool:
        category = category_by_key.get(key)
        if category is None:
            return False  # unknown key: let the detector report the error
        return (
            _scope_prefix(category) != "single-column"
            and key not in _DEDICATED_HANDLER_TYPES
        )

    return all(_routed_to_relations(k) for k in keys)


def _path_within(value: str, project_root: Path) -> Path:
    """Resolve ``value`` against ``project_root`` and assert containment.

    Returns the resolved path on success. Raises ``ValueError`` if the
    resolved path escapes ``project_root`` (catches ``..`` traversal,
    absolute-path payloads outside the workspace, and symlinks whose
    targets live outside the workspace).
    """
    if value is None or value == "":
        raise ValueError("path is required")
    p = Path(value)
    if not p.is_absolute():
        p = project_root / p
    try:
        resolved = p.resolve()
    except OSError as exc:
        raise ValueError(f"could not resolve path {value!r}: {exc}") from exc
    root = project_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"path {value!r} escapes project root {root}"
        ) from exc
    return resolved


def _glob_within(value: str, project_root: Path) -> str:
    """Reject glob patterns that contain ``..`` segments or absolute paths.

    Globs are kept verbatim (the child CLI expands them) but we refuse
    obvious attempts to escape the workspace via ``..`` or by anchoring
    the pattern at ``/``.
    """
    if value is None or value == "":
        return value
    s = str(value)
    if s.startswith("/") or s.startswith("\\"):
        raise ValueError(f"absolute glob pattern not allowed: {value!r}")
    # Split on both POSIX and Windows separators.
    parts = re.split(r"[\\/]+", s)
    if any(part == ".." for part in parts):
        raise ValueError(f"glob {value!r} contains '..' segment")
    return s


def _validate_payload_paths(
    payload: Dict[str, Any],
    *,
    project_root: Path,
    path_keys: tuple,
    glob_keys: tuple = (),
) -> None:
    """Enforce path/glob containment for every relevant payload key.

    Mutates ``payload`` to canonical (resolved) form for path keys so the
    spawned subprocess receives a stable absolute path that matches the
    same value the launcher validated.
    """
    for key in path_keys:
        v = payload.get(key)
        if v is None or v == "":
            continue
        try:
            payload[key] = str(_path_within(str(v), project_root))
        except ValueError as exc:
            raise ValueError(f"invalid {key}: {exc}") from exc
    for key in glob_keys:
        v = payload.get(key)
        if v is None or v == "":
            continue
        try:
            payload[key] = _glob_within(str(v), project_root)
        except ValueError as exc:
            raise ValueError(f"invalid {key}: {exc}") from exc


# Argv builders for the launcher's subprocess kinds live in a
# dedicated module to keep this file focused on the JobManager +
# RunRegistry. Re-exported below for backwards compatibility with
# tests that import them from runs.
from .cli_builders import (
    _append_bool,
    _append_if,
    _CURATE_FIELDS,
    _CURATE_LEGACY_KEY_ALIASES,
    _normalise_curate_payload,
    build_column_verify_argv,
    build_curate_argv,
    build_detection_argv,
    build_relation_detection_argv,
    build_relation_report_argv,
    build_relation_validation_argv,
    build_relations_argv,
    build_report_argv,
    build_web_verify_argv,
)




# ---------------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------------


@dataclass
class Job:
    job_id: str
    kind: str                       # "detection" | "report"
    argv: List[str]
    log_path: Path
    started_at: str
    run_id: Optional[str] = None    # populated once we detect the output dir
    exit_code: Optional[int] = None
    status: str = "running"         # running | succeeded | failed | cancelled | orphaned
    pid: Optional[int] = None
    ended_at: Optional[str] = None
    # list of run_ids under outputs/ at launch time; used to diff later
    existing_run_ids: List[str] = field(default_factory=list)
    # set True if this Job was recovered on server startup (we weren't the
    # original parent, so exit code cannot be captured - we poll liveness).
    recovered: bool = False
    # not serialised
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "argv": list(self.argv),
            "log_path": str(self.log_path),
            "started_at": self.started_at,
            "run_id": self.run_id,
            "exit_code": self.exit_code,
            "status": self.status,
            "pid": self.pid,
            "ended_at": self.ended_at,
            "existing_run_ids": list(self.existing_run_ids),
            "recovered": self.recovered,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        return cls(
            job_id=d["job_id"],
            kind=d["kind"],
            argv=list(d.get("argv", [])),
            log_path=Path(d["log_path"]),
            started_at=d["started_at"],
            run_id=d.get("run_id"),
            exit_code=d.get("exit_code"),
            status=d.get("status", "running"),
            pid=d.get("pid"),
            ended_at=d.get("ended_at"),
            existing_run_ids=list(d.get("existing_run_ids", [])),
            recovered=bool(d.get("recovered", False)),
        )


def _mint_run_id(model_spec: Optional[str]) -> str:
    """Assign a unique run-id *before* launch so the launcher never has to
    *infer* which ``outputs/`` dir a detection job created.

    The child honours this via ``--run-name`` and writes its artifacts to
    ``outputs/<run_id>/`` verbatim, so the launcher records the correct
    run_id up front. This fixes two bugs in the old post-hoc inference
    (``_resolve_run_id``): under concurrent launches it cross-wired jobs
    to each other's dirs, and -- because the auto-name was only
    second-resolution (``%Y%m%dT%H%M%SZ``) -- two same-model launches in
    the same second would mint the *same* dir name and clobber each other.

    ``model``/``mode``/``created`` in the dashboard come from the run's
    ``summary.json`` (see ``_parse_run_dir``), never from this string, so
    the model tag here is purely cosmetic for on-disk browsing; the random
    suffix is what guarantees uniqueness. The tag mirrors
    ``run_cleaning_detection._run_id_tag_for_model_spec`` (comma-list
    family -> ``gpt5-family``) for readability, but the two paths are
    independent -- the CLI's own tag is used only when no ``--run-name``
    is passed.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [p.strip() for p in str(model_spec or "").split(",") if p.strip()]
    tag = "gpt5-family" if len(parts) > 1 else (parts[0] if parts else "model")
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-_.") or "model"
    return f"{ts}_{tag}_{uuid.uuid4().hex[:8]}"


class JobManager:
    """Spawn + track child processes for detection and report builds."""

    def __init__(
        self,
        project_root: Path,
        python_executable: Optional[str] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.python_executable = python_executable or sys.executable
        self.log_dir = Path(log_dir or self.project_root / "logs" / "jobs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._registry = RunRegistry(self.project_root / "outputs")
        self._recover_from_disk()

    # ------------------------------------------------------------- launch

    def launch_detection(self, payload: Dict[str, Any]) -> Job:
        payload = dict(payload)
        _validate_payload_paths(
            payload,
            project_root=self.project_root,
            path_keys=_DETECTION_PATH_KEYS,
            glob_keys=_DETECTION_GLOB_KEYS,
        )
        # Auto-route relation-only selections to the relation pipeline.
        # When every selected error type is a multi-column relation type
        # (e.g. fd_violation), the per-chunk detector has nothing to run
        # and would write an empty report; run_relation_discovery instead
        # does discover -> validate -> detect in one invocation and emits
        # the same clusters/issues artifacts the labelling UI consumes.
        if _is_relation_only_detection(payload, self.project_root):
            logger.info(
                "launch_detection: every selected type is a relation "
                "type (%s); routing to the relation-discovery pipeline "
                "(discover -> validate -> detect) instead of the empty "
                "per-chunk path.",
                payload.get("error_types"),
            )
            return self.launch_relations(payload)
        # ``multi_table_from`` accepts either a bare run id (resolved
        # against ``outputs/``) or a project-root-relative path. The
        # run-id form must not be auto-converted to an absolute path
        # (the detector's resolver tries both candidates), so we only
        # block ``..`` traversal and absolute paths outside the
        # workspace here rather than running it through ``_path_within``.
        mt_from = payload.get("multi_table_from")
        if mt_from:
            mt_from_str = str(mt_from)
            parts = re.split(r"[\\/]+", mt_from_str)
            if any(p == ".." for p in parts):
                raise ValueError(
                    "invalid multi_table_from: '..' segment not allowed"
                )
            mt_path = Path(mt_from_str)
            if mt_path.is_absolute():
                # Reuse the standard containment check for explicit paths.
                try:
                    _path_within(mt_from_str, self.project_root)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid multi_table_from: {exc}"
                    ) from exc
        payload["run_name"] = _mint_run_id(payload.get("model"))
        return self._launch(
            "detection",
            build_detection_argv(payload),
            run_id=payload["run_name"],
        )

    def launch_report(self, run_id: str) -> Job:
        run_dir = self.project_root / "outputs" / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run not found: {run_id}")
        # Defence in depth: refuse to launch if the resolved run dir
        # escapes the project root (e.g. via a symlinked outputs/<id>).
        try:
            _path_within(str(run_dir), self.project_root)
        except ValueError as exc:
            raise ValueError(f"invalid run_id: {exc}") from exc
        return self._launch("report", build_report_argv(run_dir))

    def launch_relations(self, payload: Dict[str, Any]) -> Job:
        payload = dict(payload)
        _validate_payload_paths(
            payload,
            project_root=self.project_root,
            path_keys=_RELATIONS_PATH_KEYS,
            glob_keys=_RELATIONS_GLOB_KEYS,
        )
        return self._launch("relations", build_relations_argv(payload))

    def launch_relation_detection(self, payload: Dict[str, Any]) -> Job:
        """Step 2: per-cell violation detection from confirmed relations.

        Accepts a ``discovery_run`` (path to a Step-1 directory) in the
        payload; the script reads ``relations.jsonl`` + ``relation_reviews.jsonl``
        from there and writes a new run dir with cleaning-detection-shape
        outputs that the labeling UI surfaces automatically.
        """
        payload = dict(payload)
        # ``discovery_run`` is required and must live under the project
        # root (same containment guard as every other payload path).
        if not payload.get("discovery_run"):
            raise ValueError(
                "relation_detection: 'discovery_run' is required (path to "
                "a Step-1 discovery run directory)"
            )
        _validate_payload_paths(
            payload,
            project_root=self.project_root,
            path_keys=_RELATION_DETECTION_PATH_KEYS,
            glob_keys=(),
        )
        payload["run_name"] = _mint_run_id(payload.get("model"))
        return self._launch(
            "relation_detection",
            build_relation_detection_argv(payload),
            run_id=payload["run_name"],
        )

    def launch_relation_report(self, run_id: str) -> Job:
        run_dir = self.project_root / "outputs" / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run not found: {run_id}")
        try:
            _path_within(str(run_dir), self.project_root)
        except ValueError as exc:
            raise ValueError(f"invalid run_id: {exc}") from exc
        return self._launch(
            "relation_report", build_relation_report_argv(run_dir),
        )

    def launch_relation_validation(
        self,
        run_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Job:
        run_dir = self.project_root / "outputs" / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run not found: {run_id}")
        try:
            _path_within(str(run_dir), self.project_root)
        except ValueError as exc:
            raise ValueError(f"invalid run_id: {exc}") from exc
        params = dict(params or {})
        argv = build_relation_validation_argv(
            run_dir,
            layer=params.get("layer"),
            concurrency=params.get("concurrency"),
            limit=params.get("limit"),
        )
        return self._launch("relation_validation", argv)

    def launch_web_verify(
        self,
        run_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """Spawn ``experiments.run_web_verify`` for one detection run.

        Batch-verifies every gate-eligible cluster (the cells the LLM
        flagged as errors) against the web. Driven by the "Verify on
        web" button on each detection row in the runs dashboard.

        Optional ``params`` keys: ``concurrency``, ``max_clusters``,
        ``error_types`` (comma-separated whitelist), ``model``,
        ``reasoning_effort``, ``force``, ``agent_endpoint``,
        ``agent_id``. Anything missing falls back to the script's
        defaults / environment.
        """
        run_dir = self.project_root / "outputs" / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run not found: {run_id}")
        try:
            _path_within(str(run_dir), self.project_root)
        except ValueError as exc:
            raise ValueError(f"invalid run_id: {exc}") from exc
        if not (run_dir / "clusters.json").exists():
            raise ValueError(
                f"run {run_id!r} has no clusters.json — nothing to verify"
            )
        params = dict(params or {})
        argv = build_web_verify_argv(
            run_id=run_id,
            agent_endpoint=params.get("agent_endpoint") or None,
            agent_id=params.get("agent_id") or None,
            model=params.get("model") or None,
            reasoning_effort=params.get("reasoning_effort") or None,
            concurrency=params.get("concurrency"),
            max_clusters=params.get("max_clusters"),
            error_types=params.get("error_types") or None,
            force=bool(params.get("force")),
        )
        return self._launch("web_verify", argv)

    def launch_column_verify(
        self,
        run_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """Spawn ``experiments.run_web_verify_column`` for one run.

        ``params`` accepts ``source`` + ``column`` for a single-column
        audit, OR ``all_budget_capped=True`` to scan every (source,
        column) whose adaptive arms are exhausted. Optional knobs
        (``concurrency``, ``max_distinct``, ``skip_flagged``,
        ``force``, ``model``, ``reasoning_effort``) are passed through.
        """
        run_dir = self.project_root / "outputs" / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run not found: {run_id}")
        try:
            _path_within(str(run_dir), self.project_root)
        except ValueError as exc:
            raise ValueError(f"invalid run_id: {exc}") from exc
        params = dict(params or {})
        all_capped = bool(params.get("all_budget_capped"))
        source = params.get("source")
        column = params.get("column")
        if not all_capped and (not source or not column):
            raise ValueError(
                "launch_column_verify requires either "
                "(source + column) or all_budget_capped=True"
            )
        argv = build_column_verify_argv(
            run_id=run_id,
            source=str(source) if source else None,
            column=str(column) if column else None,
            all_budget_capped=all_capped,
            concurrency=params.get("concurrency"),
            max_distinct=params.get("max_distinct"),
            skip_flagged=bool(params.get("skip_flagged")),
            force=bool(params.get("force")),
            model=params.get("model"),
            reasoning_effort=params.get("reasoning_effort"),
        )
        return self._launch("column_verify", argv)

    def launch_curate(self, payload: Dict[str, Any]) -> Job:
        payload = dict(payload)
        _validate_payload_paths(
            payload,
            project_root=self.project_root,
            path_keys=_CURATE_PATH_KEYS,
        )
        return self._launch("curate", build_curate_argv(payload))

    # Backwards-compat aliases. The unified launcher dispatches both
    # legacy kinds onto the same curate process so saved Job entries
    # continue to display, and any caller still posting the old kinds
    # keeps working.
    def launch_curate_tables(self, payload: Dict[str, Any]) -> Job:
        return self.launch_curate(payload)

    def launch_curate_pbi(self, payload: Dict[str, Any]) -> Job:
        return self.launch_curate(payload)

    def _launch(
        self, kind: str, argv: List[str], *, run_id: Optional[str] = None,
    ) -> Job:
        # ``run_id`` is set for the kinds that *create* a new output dir
        # (detection / relation_detection) -- the launcher assigns it up
        # front (``_mint_run_id``) and passes it to the child via
        # ``--run-name``, so it is known and correct immediately rather
        # than inferred after the job exits. ``None`` for kinds that
        # operate on an existing run (report, web_verify, ...).
        job_id = uuid.uuid4().hex[:12]
        log_path = self.log_dir / f"{job_id}.log"

        # Snapshot existing runs so we can detect the new one this job creates.
        existing = sorted({r.run_id for r in self._registry.list()})

        log_fh = log_path.open("w", buffering=1, encoding="utf-8")
        log_fh.write(
            f"# launched: {datetime.now(timezone.utc).isoformat()}\n"
            f"# cwd: {self.project_root}\n"
            f"# argv: {self.python_executable} {' '.join(argv)}\n\n"
        )
        log_fh.flush()

        try:
            proc = subprocess.Popen(
                [self.python_executable, *argv],
                cwd=str(self.project_root),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                # New process group so SIGTERM on cancel reaches children.
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            )
        except Exception:
            # Popen can fail (bad interpreter, fork/ENOMEM) before the
            # reaper thread that owns closing log_fh ever starts.
            log_fh.close()
            raise

        job = Job(
            job_id=job_id,
            kind=kind,
            argv=list(argv),
            log_path=log_path,
            started_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            pid=proc.pid,
            existing_run_ids=list(existing),
            _proc=proc,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)

        thread = threading.Thread(
            target=self._reap,
            args=(job, log_fh),
            daemon=True,
            name=f"job-{job_id}",
        )
        job._thread = thread
        thread.start()
        return job

    def _reap(
        self,
        job: Job,
        log_fh: Any,
    ) -> None:
        try:
            rc = job._proc.wait() if job._proc else -1
        except Exception as exc:  # noqa: BLE001
            rc = -1
            log_fh.write(f"\n# wait error: {exc}\n")
        finally:
            try:
                log_fh.flush()
                log_fh.close()
            except Exception:  # noqa: BLE001
                pass

        job.exit_code = rc
        job.ended_at = datetime.now(timezone.utc).isoformat()
        if job.status != "cancelled":
            job.status = "succeeded" if rc == 0 else "failed"
        self._resolve_run_id(job)
        self._persist(job)

    # -------------------------------------------------- persistence + recovery

    def _persist(self, job: Job) -> None:
        """Write ``<log_dir>/<job_id>.json`` atomically."""
        target = self.log_dir / f"{job.job_id}.json"
        tmp = target.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
            os.replace(tmp, target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to persist job %s: %s", job.job_id, exc)

    def _resolve_run_id(self, job: Job) -> None:
        """Legacy fallback: infer the output dir for a detection job that
        has no ``run_id`` yet.

        New jobs get a deterministic ``run_id`` at launch (``_mint_run_id``
        -> ``--run-name``), so this returns immediately for them. It now
        only fires for jobs persisted by an older build (``run_id`` still
        ``None`` when recovered after a restart). The registry-diff +
        mtime guess below is inherently racy under concurrency -- which is
        exactly why launch-time assignment replaced it -- but it stays as
        a best-effort recovery path for those historical jobs.
        """
        if job.kind != "detection" or job.run_id:
            return
        try:
            existing = set(job.existing_run_ids)
            now_runs = {r.run_id for r in self._registry.list()}
            new_runs = now_runs - existing
            if len(new_runs) == 1:
                job.run_id = next(iter(new_runs))
            elif new_runs:
                newest = max(
                    (self._registry.get(r) for r in new_runs if self._registry.get(r)),
                    key=lambda r: r.path.stat().st_mtime,
                    default=None,
                )
                if newest is not None:
                    job.run_id = newest.run_id
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to detect new run for job %s: %s", job.job_id, exc)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """True if a process with ``pid`` currently exists."""
        if pid is None or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but is owned by someone else - still "alive".
            return True
        return True

    @staticmethod
    def _pid_looks_like_ours(pid: int, argv: List[str]) -> bool:
        """Cheap PID-reuse guard: /proc/{pid}/cmdline should mention run_cleaning_detection."""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read().decode("utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, OSError):
            # If we can't read it, be permissive.
            return True
        needle = argv[1] if len(argv) > 1 else ""
        return (needle and needle in cmdline) or "run_cleaning_detection" in cmdline or "build_report" in cmdline

    def _recover_from_disk(self) -> None:
        """Rehydrate Jobs from ``<log_dir>/*.json`` on startup.

        Running jobs whose pid is still alive get a watcher thread that polls
        liveness (we are not the parent so ``wait()`` does not work). Running
        jobs whose pid is dead are marked ``orphaned`` and we best-effort
        resolve their ``run_id`` from newly-created ``outputs/`` dirs.
        """
        for meta_path in sorted(self.log_dir.glob("*.json")):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                job = Job.from_dict(data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("skipping malformed job metadata %s: %s", meta_path, exc)
                continue
            with self._lock:
                self._jobs[job.job_id] = job
            if job.status != "running":
                continue
            # Was "running" when server last touched it. Check liveness.
            job.recovered = True
            if job.pid and self._pid_alive(job.pid) and self._pid_looks_like_ours(job.pid, job.argv):
                logger.info("recovered running job %s (pid=%s)", job.job_id, job.pid)
                thread = threading.Thread(
                    target=self._watch_recovered,
                    args=(job,),
                    daemon=True,
                    name=f"job-{job.job_id}-watch",
                )
                job._thread = thread
                thread.start()
            else:
                logger.info("job %s pid=%s is gone - marking orphaned", job.job_id, job.pid)
                job.status = "orphaned"
                job.ended_at = job.ended_at or datetime.now(timezone.utc).isoformat()
                self._resolve_run_id(job)
                self._persist(job)

    def _watch_recovered(self, job: Job, poll_s: float = 5.0) -> None:
        """Poll a recovered (orphaned-but-alive) child until its pid disappears."""
        import time
        while job.status == "running" and self._pid_alive(job.pid or -1):
            time.sleep(poll_s)
        # Process is gone OR we were cancelled. Either way, finalise metadata.
        if job.ended_at is None:
            job.ended_at = datetime.now(timezone.utc).isoformat()
        self._resolve_run_id(job)
        if job.status == "running":
            # Natural exit (not cancel). Infer success from summary.json presence.
            succeeded = False
            if job.run_id:
                info = self._registry.get(job.run_id)
                if info and info.has_summary:
                    succeeded = True
            job.status = "succeeded" if succeeded else "failed"
            job.exit_code = 0 if succeeded else None
        self._persist(job)

    # ------------------------------------------------------------- cancel

    def cancel(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status != "running":
            return job
        pid = job.pid or (job._proc.pid if job._proc else None)
        if not pid:
            return job
        try:
            # Send SIGTERM to the whole process group. Works for both our
            # own child processes and recovered/orphaned ones (setsid'd at launch).
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (AttributeError, ProcessLookupError):
                if job._proc is not None:
                    job._proc.terminate()
                else:
                    os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        job.status = "cancelled"
        self._persist(job)
        return job

    # ------------------------------------------------------------- delete

    def delete(self, job_id: str) -> Job:
        """Remove a finished job's metadata + log file. Refuses to delete
        a job that's still running (cancel it first)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status == "running":
                raise ValueError(
                    f"job {job_id} is still running - cancel it before deleting"
                )
            self._jobs.pop(job_id, None)
        for path in (
            self.log_dir / f"{job_id}.json",
            self.log_dir / f"{job_id}.json.tmp",
            job.log_path,
        ):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:  # noqa: BLE001
                logger.warning("failed to remove %s: %s", path, exc)
        return job

    # ------------------------------------------------------------ queries

    def list(self) -> List[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def read_log(self, job_id: str, offset: int = 0, max_bytes: int = 256_000) -> Dict[str, Any]:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        size = 0
        text = ""
        next_offset = offset
        try:
            with job.log_path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                if offset < 0:
                    offset = max(0, size + offset)
                fh.seek(offset)
                data = fh.read(max_bytes)
                text = data.decode("utf-8", errors="replace")
                # Advance by the RAW bytes consumed, not the re-encoded
                # text: errors="replace" turns each invalid byte into a
                # 3-byte U+FFFD, so len(text.encode()) != len(data) and
                # the incremental tail would skip/dupe bytes.
                next_offset = min(size, offset + len(data))
        except FileNotFoundError:
            pass
        return {
            "job_id": job_id,
            "offset": offset,
            "next_offset": next_offset,
            "size": size,
            "text": text,
            "status": job.status,
            "exit_code": job.exit_code,
            "run_id": job.run_id,
        }


# ---------------------------------------------------------------------------
# Form defaults helpers
# ---------------------------------------------------------------------------


def list_datasets(data_dir: Path) -> List[Dict[str, Any]]:
    """Enumerate subdirectories under ``data_dir`` that contain CSVs, and
    loose CSV files at the top level. Used to populate the launcher form.
    """
    data_dir = Path(data_dir)
    out: List[Dict[str, Any]] = []
    if not data_dir.is_dir():
        return out
    for child in sorted(data_dir.iterdir()):
        if child.is_dir():
            # Cap scan at 5000 - some corpora have tens of thousands of
            # CSVs and we only need a yes/no presence + a 5-name sample.
            # Recursive so folders that hold per-table subdirs
            # (e.g. curated_pbi/<workspace>/*.csv, Bing21/<id>/*.csv)
            # are detected.
            csvs: List[str] = []
            truncated = False
            for i, p in enumerate(child.rglob("*.csv")):
                if i >= 5000:
                    truncated = True
                    break
                csvs.append(p.name)
            if not csvs:
                continue
            out.append({
                "type": "dir",
                "path": str(child),
                "name": child.name,
                "csv_count": (len(csvs) if not truncated else f"{len(csvs)}+"),
                "sample": sorted(csvs)[:5],
            })
        elif child.suffix.lower() == ".csv":
            out.append({
                "type": "file",
                "path": str(child),
                "name": child.name,
                "csv_count": 1,
            })
    return out


def load_taxonomy_keys(project_root: Path) -> List[Dict[str, Any]]:
    """Return the list of error-type keys + their category, read from the
    default prompts/error_types.yaml. Fails soft (returns [])."""
    import yaml  # deferred

    path = project_root / "prompts" / "error_types.yaml"
    if not path.exists():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return []
    types = raw.get("types") or raw.get("error_types") or []
    out: List[Dict[str, Any]] = []
    if isinstance(types, list):
        for item in types:
            if not isinstance(item, dict):
                continue
            out.append({
                "key": item.get("key") or item.get("id") or "",
                "name": item.get("name") or item.get("key") or "",
                "category": item.get("category") or "",
            })
    elif isinstance(types, dict):
        for key, item in types.items():
            if not isinstance(item, dict):
                continue
            out.append({
                "key": key,
                "name": item.get("name") or key,
                "category": item.get("category") or "",
            })
    return out


def load_config_defaults(project_root: Path) -> Dict[str, Any]:
    """Best-effort parse of ``config/experiment.yaml`` for form defaults."""
    import yaml

    path = project_root / "config" / "experiment.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}
