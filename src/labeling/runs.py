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
    return out


def _save_run_metadata(run_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``meta`` into the sidecar and write atomically.

    Only known keys (``display_name``, ``starred``, ``notes``) are
    persisted. Returns the full post-merge metadata dict. Writes use
    a tmp-file + ``os.replace`` so a crashing server can't leave a
    half-written JSON behind.
    """
    current = _load_run_metadata(run_dir)
    merged: Dict[str, Any] = {
        "display_name": current.get("display_name", ""),
        "starred": bool(current.get("starred", False)),
        "notes": current.get("notes", ""),
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

    path = run_dir / _RUN_METADATA_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return merged


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
            csv_path = cfg.get("csv_path") or (paths[0] if paths else None)
            if csv_path:
                p = Path(csv_path)
                # If many CSVs, label by their common parent dir; if a single
                # file run, label by the file name.
                if len(paths) > 1:
                    # ``csv_path`` in single-file mode is the file; in
                    # directory mode it's the directory itself.
                    dataset = Path(cfg.get("csv_path") or p.parent).name
                else:
                    dataset = p.name
            if not is_relations:
                n_sources = int(cfg.get("source_count") or len(paths) or 0)
            ni = cfg.get("normalize_input")
            if isinstance(ni, dict):
                normalize_input = bool(ni.get("enabled"))
            else:
                normalize_input = bool(ni)
        except Exception as exc:  # noqa: BLE001
            logger.debug("summary.json parse fail for %s: %s", path, exc)
    if clusters_path.exists():
        # cheap count: iterate the JSON array as lines would not be guaranteed
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
        return self._launch("detection", build_detection_argv(payload))

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
        return self._launch(
            "relation_detection", build_relation_detection_argv(payload),
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

    def _launch(self, kind: str, argv: List[str]) -> Job:
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
