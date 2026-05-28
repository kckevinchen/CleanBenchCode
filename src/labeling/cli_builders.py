"""CLI-argv builders that translate launcher JSON payloads into subprocess argv.

Each ``build_*_argv`` function takes a (mostly) free-form JSON payload from
the launcher form and emits a ``-m <module> [flag value]...`` list that can
be passed straight to ``subprocess.Popen``. Path/glob validation for the
subset of fields that resolve to filesystem locations lives in
:mod:`src.labeling.runs` (``_path_within`` / ``_validate_payload_paths``)
and runs at job-spawn time, immediately before the argv is consumed.

Kept in its own module because the eight builders + their helpers form a
self-contained ~550 LOC block that used to bloat ``runs.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# argv-builder helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


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
    _append_if(argv, "--synthesis-effort", payload.get("synthesis_effort"))
    _append_if(argv, "--output-dir", payload.get("output_dir"))

    if payload.get("no_synthesis"):
        argv.append("--no-synthesis")
    if payload.get("no_synthesise_singletons"):
        argv.append("--no-synthesise-singletons")

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

    # Multi-table cross-sheet pass. The cleaning runner owns the
    # whole multi-table pipeline (steps 1+2+3 plus optional step 4)
    # so artifacts land in the same run as the rest of the
    # detection output. ``inter_table_inconsistency`` in
    # --error-types auto-triggers the pass; --multi-table-from
    # short-circuits steps 1+2 by reusing groups from a prior
    # discovery run.
    if payload.get("no_multi_table"):
        argv.append("--no-multi-table")
    _append_if(argv, "--multi-table-from", payload.get("multi_table_from"))
    _append_if(argv, "--multi-table-prompt", payload.get("multi_table_prompt"))
    _append_if(
        argv,
        "--multi-table-strategy",
        payload.get("multi_table_strategy"),
    )
    _append_if(
        argv,
        "--multi-table-detect-prompt",
        payload.get("multi_table_detect_prompt"),
    )
    _append_if(
        argv,
        "--multi-table-verify-prompt",
        payload.get("multi_table_verify_prompt"),
    )
    if payload.get("no_multi_table_verify"):
        argv.append("--no-multi-table-verify")
    _append_if(
        argv,
        "--multi-table-semantic-rows",
        payload.get("multi_table_semantic_rows"),
    )
    _append_if(
        argv,
        "--multi-table-semantic-max-pages",
        payload.get("multi_table_semantic_max_pages"),
    )

    return argv


def build_report_argv(run_dir: Path) -> List[str]:
    return ["-m", "experiments.build_report", str(run_dir)]


# ---------------------------------------------------------------------------
# Relation discovery
# ---------------------------------------------------------------------------


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
    # Post-discovery passes. Defaults ON in the CLI; the launcher
    # form sends ``aggregate`` as a bool so we can flip the negative
    # flag without leaking flag-name semantics into the form payload
    # schema.
    if payload.get("aggregate") is False:
        argv.append("--no-aggregate")
    else:
        _append_if(argv, "--aggregate-prompt", payload.get("aggregate_prompt"))
        _append_if(argv, "--aggregate-min-cluster",
                   payload.get("aggregate_min_cluster"))
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


def build_relation_detection_argv(payload: Dict[str, Any]) -> List[str]:
    """Argv for ``experiments.run_relation_detection`` (Step 2).

    Translates the UI launcher form into a CLI invocation. The payload's
    ``discovery_run`` is the absolute path to a Step-1 run directory;
    it's a positional arg to the script, so we append it directly.
    """
    discovery_run = payload.get("discovery_run")
    if not discovery_run:
        raise ValueError("relation_detection: 'discovery_run' is required")
    argv: List[str] = [
        "-m", "experiments.run_relation_detection", str(discovery_run),
    ]
    _append_if(argv, "--config", payload.get("config"))
    _append_if(argv, "--output-dir", payload.get("output_dir"))
    _append_if(argv, "--run-name", payload.get("run_name"))
    _append_if(argv, "--error-types", payload.get("error_types"))
    _append_if(argv, "--review-filter", payload.get("review_filter"))
    _append_if(argv, "--model", payload.get("model"))
    _append_if(argv, "--reasoning-effort", payload.get("reasoning_effort"))
    _append_if(argv, "--concurrency", payload.get("concurrency"))
    _append_if(argv, "--violations-min-pass-rate",
               payload.get("violations_min_pass_rate"))
    _append_if(argv, "--violations-max-pass-rate",
               payload.get("violations_max_pass_rate"))
    _append_if(argv, "--violations-max-per-rule",
               payload.get("violations_max_per_rule"))
    _append_if(argv, "--violations-min-confidence",
               payload.get("violations_min_confidence"))
    _append_if(argv, "--llm-detect-max-rows",
               payload.get("llm_detect_max_rows"))
    _append_if(argv, "--llm-detect-max-per-rule",
               payload.get("llm_detect_max_per_rule"))
    if payload.get("no_llm_detect"):
        argv.append("--no-llm-detect")
    if payload.get("no_synthesis"):
        argv.append("--no-synthesis")
    return argv


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


# ---------------------------------------------------------------------------
# Web verifiers
# ---------------------------------------------------------------------------


def build_web_verify_argv(
    run_id: str,
    *,
    agent_endpoint: Optional[str] = None,
    agent_id: Optional[str] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    concurrency: Optional[int] = None,
    max_clusters: Optional[int] = None,
    error_types: Optional[str] = None,
    force: bool = False,
) -> List[str]:
    """Argv for ``experiments.run_web_verify``.

    Batch-verifies every gate-eligible cluster in a finished detection
    run against the web. Used by the "Verify on web" button on each
    detection row in the runs dashboard. ``error_types`` is the
    comma-separated whitelist forwarded as ``--error-types``; pass
    ``None``/empty to verify every gate-eligible cluster.
    """
    argv: List[str] = ["-m", "experiments.run_web_verify",
                       "--run-id", run_id]
    if agent_endpoint:
        argv.extend(["--agent-endpoint", str(agent_endpoint)])
    if agent_id:
        argv.extend(["--agent-id", str(agent_id)])
    if model:
        argv.extend(["--model", str(model)])
    if reasoning_effort:
        argv.extend(["--reasoning-effort", str(reasoning_effort)])
    if concurrency is not None:
        argv.extend(["--concurrency", str(int(concurrency))])
    if max_clusters is not None:
        argv.extend(["--max-clusters", str(int(max_clusters))])
    if error_types:
        argv.extend(["--error-types", str(error_types)])
    if force:
        argv.append("--force")
    return argv


def build_column_verify_argv(
    run_id: str,
    *,
    source: Optional[str] = None,
    column: Optional[str] = None,
    all_budget_capped: bool = False,
    concurrency: Optional[int] = None,
    max_distinct: Optional[int] = None,
    skip_flagged: bool = False,
    force: bool = False,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> List[str]:
    """Argv for ``experiments.run_web_verify_column``.

    Either ``(source, column)`` must be supplied OR ``all_budget_capped``
    must be True. The caller (JobManager.launch_column_verify) is
    responsible for validating that condition before reaching here.
    """
    argv: List[str] = ["-m", "experiments.run_web_verify_column",
                       "--run-id", run_id]
    if all_budget_capped:
        argv.append("--all-budget-capped")
    else:
        if not source:
            raise ValueError("source is required when all_budget_capped is False")
        if not column:
            raise ValueError("column is required when all_budget_capped is False")
        argv.extend(["--source", source, "--column", column])
    if concurrency is not None:
        argv.extend(["--concurrency", str(int(concurrency))])
    if max_distinct is not None:
        argv.extend(["--max-distinct", str(int(max_distinct))])
    if skip_flagged:
        argv.append("--skip-flagged")
    if force:
        argv.append("--force")
    if model:
        argv.extend(["--model", str(model)])
    if reasoning_effort:
        argv.extend(["--reasoning-effort", str(reasoning_effort)])
    return argv


# ---------------------------------------------------------------------------
# Curate (data prep)
# ---------------------------------------------------------------------------


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
    ("dedupe_across_groups",      "--dedupe-across-groups"),
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
