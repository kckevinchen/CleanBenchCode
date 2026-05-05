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
        }


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
    # Good-Turing booleans - handled below with paired --no-* flags.
}


def _append_if(argv: List[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, (list, tuple)):
        # Lists from the JSON launcher payload (e.g. ``error_types``,
        # ``scopes``, ``columns``) need to be flattened into the CLI's
        # comma-separated form. ``str([...])`` would smuggle Python
        # repr noise (brackets, quotes) into argv.
        items = [str(v).strip() for v in value if str(v).strip()]
        if not items:
            return
        argv.extend([flag, ",".join(items)])
        return
    argv.extend([flag, str(value)])


def _append_bool(
    argv: List[str],
    value: Optional[bool],
    on_flag: str,
    off_flag: str,
) -> None:
    if value is None:
        return
    argv.append(on_flag if value else off_flag)


# Path keys that the launcher accepts from the UI / API and forwards to
# child processes via CLI flags. Each entry must be validated against
# the project root before the subprocess is spawned, otherwise an
# attacker who can hit the unauth API could read/write arbitrary
# paths using the server's credentials.
_DETECTION_PATH_KEYS = ("config", "csv", "csv_dir", "output_dir")
_DETECTION_GLOB_KEYS = ("csv_glob",)  # patterns, not paths
_CURATE_PATH_KEYS = ("source_dir", "output_dir")


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


def build_detection_argv(payload: Dict[str, Any]) -> List[str]:
    """Translate a UI form payload into a ``run_cleaning_detection`` argv.

    Everything goes through named flags so we never need shell quoting.
    The Python executable is the same one the server runs under (to keep
    the detector in the same venv); the caller prepends it.
    """
    argv: List[str] = ["-m", "experiments.run_cleaning_detection"]

    _append_if(argv, "--config", payload.get("config"))
    _append_if(argv, "--mode", payload.get("mode"))
    _append_if(argv, "--csv", payload.get("csv"))
    _append_if(argv, "--csv-dir", payload.get("csv_dir"))
    _append_if(argv, "--csv-glob", payload.get("csv_glob"))
    _append_if(argv, "--model", payload.get("model"))
    _append_if(argv, "--chunk-size", payload.get("chunk_size"))
    _append_if(argv, "--max-chunks", payload.get("max_chunks"))
    _append_if(argv, "--column-mode", payload.get("column_mode"))
    _append_if(argv, "--columns", payload.get("columns"))
    _append_if(argv, "--reasoning-effort", payload.get("reasoning_effort"))
    _append_if(argv, "--concurrency", payload.get("concurrency"))
    _append_if(argv, "--error-types", payload.get("error_types"))
    _append_if(argv, "--scopes", payload.get("scopes"))
    _append_if(argv, "--synthesis-concurrency", payload.get("synthesis_concurrency"))
    _append_if(argv, "--synthesis-effort", payload.get("synthesis_effort"))
    _append_if(argv, "--output-dir", payload.get("output_dir"))

    if payload.get("no_synthesis"):
        argv.append("--no-synthesis")

    # Input normalization (deterministic pre-pass; ablation flag).
    if payload.get("normalize_input"):
        argv.append("--normalize-input")
    if payload.get("normalize_fold_punctuation"):
        argv.append("--normalize-fold-punctuation")
    if payload.get("normalize_lowercase"):
        argv.append("--normalize-lowercase")

    # row_duplication detector knobs (only consulted when
    # ``row_duplication`` is in --error-types). Tri-state booleans are
    # forwarded via paired --no-* flags so the user can explicitly
    # override the YAML in either direction.
    _append_if(
        argv,
        "--row-dup-detection-mode",
        payload.get("row_dup_detection_mode"),
    )
    _append_bool(
        argv,
        payload.get("row_dup_normalize_keys"),
        "--row-dup-normalize-keys",
        "--no-row-dup-normalize-keys",
    )
    _append_bool(
        argv,
        payload.get("row_dup_adjudicate"),
        "--row-dup-adjudicate",
        "--no-row-dup-adjudicate",
    )
    _append_if(
        argv,
        "--row-dup-max-groups-per-source",
        payload.get("row_dup_max_groups_per_source"),
    )
    _append_bool(
        argv,
        payload.get("row_dup_canonicalize_keys"),
        "--row-dup-canonicalize-keys",
        "--no-row-dup-canonicalize-keys",
    )
    _append_if(
        argv,
        "--row-dup-canonicalize-max-cardinality",
        payload.get("row_dup_canonicalize_max_cardinality"),
    )
    _append_bool(
        argv,
        payload.get("row_dup_blocking_enabled"),
        "--row-dup-blocking",
        "--no-row-dup-blocking",
    )
    _append_bool(
        argv,
        payload.get("row_dup_minhash_enabled"),
        "--row-dup-minhash",
        "--no-row-dup-minhash",
    )
    _append_if(
        argv,
        "--row-dup-minhash-threshold",
        payload.get("row_dup_minhash_threshold"),
    )
    _append_bool(
        argv,
        payload.get("row_dup_embedding_enabled"),
        "--row-dup-embedding",
        "--no-row-dup-embedding",
    )
    _append_if(
        argv,
        "--row-dup-embedding-threshold",
        payload.get("row_dup_embedding_threshold"),
    )
    _append_if(
        argv,
        "--row-dup-embedding-model",
        payload.get("row_dup_embedding_model"),
    )

    # Good-Turing (adaptive mode).
    _append_if(argv, "--gt-mode", payload.get("gt_mode"))
    _append_if(argv, "--gt-num-runs", payload.get("gt_num_runs"))
    _append_if(argv, "--gt-min-runs", payload.get("gt_min_runs"))
    _append_if(argv, "--gt-target-missing-mass", payload.get("gt_target_missing_mass"))
    _append_if(argv, "--gt-stop-scope", payload.get("gt_stop_scope"))
    # Back-compat: still forward --gt-target-policy if a stale form
    # submission supplies it (the launcher drops it on the next page
    # load now that the dropdown was renamed).
    _append_if(argv, "--gt-target-policy", payload.get("gt_target_policy"))
    _append_if(argv, "--gt-base-seed", payload.get("gt_base_seed"))
    _append_bool(argv, payload.get("gt_shuffle_rows"), "--gt-shuffle-rows", "--no-gt-shuffle-rows")
    _append_bool(argv, payload.get("gt_shuffle_columns"), "--gt-shuffle-columns", "--no-gt-shuffle-columns")
    _append_bool(argv, payload.get("gt_prompt_ensemble"), "--gt-prompt-ensemble", "--no-gt-prompt-ensemble")
    _append_bool(argv, payload.get("gt_skip_identity_run"), "--gt-skip-identity-run", "--no-gt-skip-identity-run")
    _append_if(argv, "--gt-variant-effort", payload.get("gt_variant_effort"))

    # Scope-aware per-arm overrides. Loop-level target_missing_mass /
    # min_runs / num_runs feed the per-arm epsilon / k_min / k_max so
    # those are not duplicated here. The remaining per-arm knobs --
    # require_two_consecutive, stop_when, epsilon_chao1 -- are exposed
    # individually so the launcher form can experiment with them
    # without editing the YAML.
    _append_bool(
        argv,
        payload.get("gt_per_arm_two_consecutive"),
        "--gt-per-arm-two-consecutive",
        "--no-gt-per-arm-two-consecutive",
    )
    _append_if(
        argv,
        "--gt-per-arm-stop-when",
        payload.get("gt_per_arm_stop_when"),
    )
    _append_if(
        argv,
        "--gt-per-arm-epsilon-chao1",
        payload.get("gt_per_arm_epsilon_chao1"),
    )

    return argv


def build_report_argv(run_dir: Path) -> List[str]:
    return ["-m", "experiments.build_report", str(run_dir)]


# ---- Relation discovery (multi-column relations) -----------------------------

# Path/glob keys forwarded from the launcher form to
# ``run_relation_discovery``.  Validated for project-root containment
# in ``launch_relations`` before the subprocess is spawned.
_RELATIONS_PATH_KEYS = ("config", "csv", "csv_dir", "output_dir",
                        "discovery_prompt", "aggregate_prompt",
                        "generalize_prompt", "validate_prompt")
_RELATIONS_GLOB_KEYS = ("csv_glob",)


def build_relations_argv(payload: Dict[str, Any]) -> List[str]:
    """Translate a UI form payload into a ``run_relation_discovery`` argv."""
    argv: List[str] = ["-m", "experiments.run_relation_discovery"]
    _append_if(argv, "--config", payload.get("config"))
    _append_if(argv, "--csv", payload.get("csv"))
    _append_if(argv, "--csv-dir", payload.get("csv_dir"))
    _append_if(argv, "--csv-glob", payload.get("csv_glob"))
    _append_if(argv, "--model", payload.get("model"))
    _append_if(argv, "--reasoning-effort", payload.get("reasoning_effort"))
    _append_if(argv, "--concurrency", payload.get("concurrency"))
    _append_if(argv, "--error-types", payload.get("error_types"))
    _append_if(argv, "--n-row-samples", payload.get("n_row_samples"))
    _append_if(argv, "--output-dir", payload.get("output_dir"))
    _append_if(argv, "--discovery-prompt", payload.get("discovery_prompt"))
    if payload.get("include_raw"):
        argv.append("--include-raw")
    if payload.get("adaptive"):
        argv.append("--adaptive")
        _append_if(argv, "--max-passes", payload.get("max_passes"))
        _append_if(argv, "--min-passes", payload.get("min_passes"))
        _append_if(argv, "--gt-threshold", payload.get("gt_threshold"))
        _append_if(argv, "--stop-when", payload.get("stop_when"))
        _append_if(argv, "--epsilon-chao1", payload.get("epsilon_chao1"))
        _append_if(argv, "--base-seed", payload.get("base_seed"))
    # Post-discovery passes. Both default ON in the CLI; the launcher
    # form sends ``aggregate``/``generalize`` as bools so we can flip
    # the negative flag without leaking flag-name semantics into the
    # form payload schema.
    if payload.get("aggregate") is False:
        argv.append("--no-aggregate")
    else:
        _append_if(argv, "--aggregate-prompt", payload.get("aggregate_prompt"))
        _append_if(argv, "--aggregate-min-cluster",
                   payload.get("aggregate_min_cluster"))
    if payload.get("generalize") is False:
        argv.append("--no-generalize")
    else:
        _append_if(argv, "--generalize-prompt", payload.get("generalize_prompt"))
        _append_if(argv, "--generalize-min-cluster",
                   payload.get("generalize_min_cluster"))
    # Empirical-validation pass also defaults ON; flip the negative
    # flag when the form explicitly opts out.
    if payload.get("validate") is False:
        argv.append("--no-validate")
    else:
        _append_if(argv, "--validate-prompt", payload.get("validate_prompt"))
        _append_if(argv, "--validate-layer", payload.get("validate_layer"))
        _append_if(argv, "--validate-sample-rows",
                   payload.get("validate_sample_rows"))
        _append_if(argv, "--validate-timeout",
                   payload.get("validate_timeout"))
        _append_if(argv, "--validate-max-violations",
                   payload.get("validate_max_violations"))
        _append_if(argv, "--validate-limit", payload.get("validate_limit"))
    return argv


def build_relation_report_argv(run_dir: Path) -> List[str]:
    return ["-m", "experiments.build_relation_report", str(run_dir)]


def build_relation_validation_argv(
    run_dir: Path,
    *,
    layer: Optional[str] = None,
    concurrency: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """Argv for ``experiments.run_relation_validation``.

    Optional kwargs map onto the script's CLI flags. They are exposed
    as job-launcher params so the UI can offer a small form (layer
    select + max-relations cap for a smoke test) without hardcoding.
    """
    argv: List[str] = ["-m", "experiments.run_relation_validation", str(run_dir)]
    if layer:
        argv.extend(["--layer", str(layer)])
    if concurrency is not None:
        argv.extend(["--concurrency", str(int(concurrency))])
    if limit is not None:
        argv.extend(["--limit", str(int(limit))])
    return argv


# ---- Data-prep (unified curate) -------------------------------------------

# Field schema for the unified curate launcher. Mirrors
# ``src/data_prep/curate.py``'s parse_args(); CSV layout (flat vs grouped)
# is auto-detected by the curate module so the same payload schema covers
# both the legacy curate_tables (single-table) and curate_pbi (multi-table)
# use cases.
_CURATE_FIELDS: List[tuple] = [
    # (payload_key, cli_flag)
    ("source_dir",                "--source-dir"),
    ("output_dir",                "--output-dir"),
    ("num_targets",               "--num-targets"),
    ("min_columns",               "--min-columns"),
    ("max_columns",               "--max-columns"),
    ("min_rows",                  "--min-rows"),
    ("max_rows",                  "--max-rows"),
    ("min_tables",                "--min-tables"),
    ("max_tables",                "--max-tables"),
    ("seed",                      "--seed"),
    ("max_scan",                  "--max-scan"),
    ("oversize",                  "--oversize"),
    ("language_check",            "--language-check"),
    ("language_min_english_prob", "--language-min-english-prob"),
    ("language_min_confidence",   "--language-min-confidence"),
    ("language_llm_model",        "--language-llm-model"),
    ("language_llm_effort",       "--language-llm-effort"),
    ("max_non_latin_chars",       "--max-non-latin-chars"),
    ("max_empty_ratio",           "--max-empty-ratio"),
    ("max_cell_chars",            "--max-cell-chars"),
    ("max_cell_words",            "--max-cell-words"),
    ("max_oversized_row_ratio",   "--max-oversized-row-ratio"),
]

# Legacy payload keys that callers may still send. We translate them onto
# the unified schema so existing UI/API clients keep working through one
# release; remove after the next major.
_CURATE_LEGACY_KEY_ALIASES: Dict[str, str] = {
    "num_tables":       "num_targets",   # old curate_tables
    "num_spreadsheets": "num_targets",   # old curate_pbi
}


def _normalise_curate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy payload keys onto the unified schema."""
    out = dict(payload)
    for old, new in _CURATE_LEGACY_KEY_ALIASES.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
        elif old in out:
            out.pop(old)
    return out


def build_curate_argv(payload: Dict[str, Any]) -> List[str]:
    payload = _normalise_curate_payload(payload)
    if not payload.get("source_dir"):
        raise ValueError("source_dir is required for curate")
    if not payload.get("output_dir"):
        raise ValueError("output_dir is required for curate")
    argv: List[str] = ["-m", "experiments.data_prep.curate"]
    for key, flag in _CURATE_FIELDS:
        _append_if(argv, flag, payload.get(key))
    if payload.get("overwrite"):
        argv.append("--overwrite")
    # Boolean flag pair: dedupe_schema_within_group defaults ON in the CLI,
    # so only emit the negative form when the caller explicitly disables it.
    dedupe = payload.get("dedupe_schema_within_group")
    if dedupe is False or (isinstance(dedupe, str) and dedupe.lower() in ("false", "0", "no", "off")):
        argv.append("--no-dedupe-schema-within-group")
    return argv


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
        try:
            with job.log_path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                if offset < 0:
                    offset = max(0, size + offset)
                fh.seek(offset)
                data = fh.read(max_bytes)
                text = data.decode("utf-8", errors="replace")
        except FileNotFoundError:
            pass
        return {
            "job_id": job_id,
            "offset": offset,
            "next_offset": min(size, offset + len(text.encode("utf-8"))),
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
