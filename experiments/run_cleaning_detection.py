"""Unified entry point for the data-cleaning error-detection experiment.

Two experiment modes, selectable via ``experiment.mode`` in the YAML
config or ``--mode`` on the CLI:

``mode=single`` (default):
  1. Load every CSV and split into fixed-size chunks.
  2. For each ``(chunk, error_type)`` pair, issue one LLM call.
  3. Flatten detections and run Phase-A clustering
     (:func:`src.aggregator.cluster_issues`) to merge overlapping
     detections on the same row into one cluster per underlying defect.
  4. For every multi-type cluster, optionally fire one Phase-B LLM
     synthesis call that picks a ``primary_error_type`` and emits a
     canonical explanation + fix (disable with ``--no-synthesis``).
  5. Emit JSON / JSONL / Markdown artifacts + the HTML dashboard.

``mode=adaptive``:
  Multi-pass run with perturbed row/column order. After every pass we
  recompute the Good-Turing / Chao1 false-negative estimator; the loop
  stops either after ``good_turing.num_runs`` passes (``gt_mode=fixed``)
  or as soon as the Good-Turing missing mass drops to
  ``good_turing.target_missing_mass`` (``gt_mode=adaptive``, default),
  subject to a ``good_turing.min_runs`` floor. The union of detections
  from every completed pass feeds Phase-A clustering + Phase-B
  synthesis + the HTML dashboard exactly as in single mode; the
  estimator report lands under ``good_turing/`` inside the run dir.

Usage (from the project root)::

    # Classic single-pass run:
    python -m experiments.run_cleaning_detection \
        --config config/experiment.yaml

    # Adaptive multi-pass with Good-Turing stop condition:
    python -m experiments.run_cleaning_detection \
        --mode adaptive --gt-num-runs 8 --gt-target-missing-mass 0.05

Common overrides::

    # Smoke test (1 chunk, single-column core types, no synthesis):
    python -m experiments.run_cleaning_detection \
        --max-chunks 1 \
        --error-types lexical_inconsistency,formatting_inconsistency \
        --column-mode single \
        --no-synthesis

    # Fixed-K multi-pass (6 passes, no early stop):
    python -m experiments.run_cleaning_detection \
        --mode adaptive --gt-mode fixed --gt-num-runs 6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.aggregator import (  # noqa: E402
    ClusterSynthesiser,
    cluster_issues,
    load_synthesis_prompt,
)
from src.core.data_loader import (  # noqa: E402
    COLUMN_MODE_ALL,
    COLUMN_MODE_AUTO,
    COLUMN_MODE_SINGLE,
    VALID_COLUMN_MODES,
    VALID_DISPATCHER_COLUMN_MODES,
    CsvChunk,
    iter_chunks,
    load_csv,
)
from src.core.detector import PerTypeCleaningDetector  # noqa: E402
from src.core.error_taxonomy import load_taxonomy  # noqa: E402
from src.core.row_duplication_detector import (  # noqa: E402
    AdjudicatorSettings,
    RowDuplicationDetector,
)
from src.core.table_digest import TableDigest, build_table_digest  # noqa: E402
from src.adaptive.cell_gt import (  # noqa: E402
    aggregate_runs,
    estimate,
    estimate_by_group,
    render_markdown_report,
)
from src.adaptive.scoped_gt import (  # noqa: E402
    ArmConfig,
    aggregate_runs_by_scope,
    estimate_by_arm,
    evaluate_stop,
    is_applicable,
    render_per_arm_markdown,
    resolve_arm_config,
)
from src.core.llm_client import LLMClient  # noqa: E402
from src.adaptive.perturbations import perturb_frames  # noqa: E402
from src.adaptive.prompt_variants import (  # noqa: E402
    PromptVariantGenerator,
    save_variant_records,
)
from src.core.report_builder import build_run_artifacts, flatten_issues  # noqa: E402

logger = logging.getLogger("run_cleaning_detection")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


# Recognised scope keys (the prefix of ErrorType.category before " / ").
# Keep in sync with prompts/error_types.yaml category strings.
KNOWN_SCOPES = frozenset(
    {"single-column", "multi-column", "multi-row", "multi-table"}
)


def _scope_of(category: str) -> str:
    """Return the lower-cased scope prefix of an ErrorType.category."""
    head = (category or "").split("/", 1)[0]
    return " ".join(head.lower().split())


def _normalise_scope(value: Any) -> str:
    """Normalise a user-supplied scope token to the canonical form."""
    return " ".join(str(value).strip().lower().split())


def _resolve_scopes(
    cli_value: Optional[str],
    cfg_value: Any,
) -> Optional[set]:
    """Combine CLI + YAML scope inputs into a set, or None for "no filter".

    Accepts either a string (CSV-list, e.g. "single-column,multi-column")
    or a list. The pseudo-scope "all" disables filtering. Unknown scopes
    raise ``ValueError`` so typos surface immediately.
    """
    raw: Optional[List[str]]
    if cli_value is not None:
        raw = _parse_csv_list(cli_value)
    elif cfg_value is None:
        raw = None
    elif isinstance(cfg_value, str):
        raw = _parse_csv_list(cfg_value)
    else:
        raw = [str(v) for v in cfg_value if str(v).strip()] or None

    if not raw:
        return None
    cleaned = {_normalise_scope(s) for s in raw}
    if "all" in cleaned:
        return None
    unknown = cleaned - KNOWN_SCOPES
    if unknown:
        raise ValueError(
            f"Unknown scope(s) {sorted(unknown)}. "
            f"Valid scopes: {sorted(KNOWN_SCOPES) + ['all']}."
        )
    return cleaned


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect data-cleaning errors in a CSV by running one LLM "
            "call per (chunk, error_type) pair, then clustering and "
            "synthesising the results."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "experiment.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help=(
            "Path to a single CSV (overrides data.csv_path) OR to a "
            "directory of CSVs. If a directory is given, every file "
            "matching --csv-glob is included in the run."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        help=(
            "Directory of CSVs to process together (overrides "
            "data.csv_dir). Equivalent to passing the directory via --csv."
        ),
    )
    parser.add_argument(
        "--csv-glob",
        type=str,
        help=(
            "Glob pattern (e.g. '*.csv', 'sample_*.csv') used when a "
            "directory of CSVs is provided. Defaults to '*.csv'."
        ),
    )
    parser.add_argument("--model", type=str, help="Override llm.model.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help=(
            "Number of CSV rows per LLM call (a.k.a. row chunk size). "
            "Overrides data.chunk_size."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        help=(
            "Override data.max_chunks (use -1 for all chunks). Counts "
            "(row-slice, column) pairs in single-column mode."
        ),
    )
    parser.add_argument(
        "--column-mode",
        type=str,
        choices=sorted(VALID_DISPATCHER_COLUMN_MODES),
        help=(
            "How to split columns across LLM calls. "
            "'all' = one call sees every column for a row slice "
            "(useful for cross-column error types). "
            "'single' = one call per (row-slice, column) pair "
            "(use for single-column error types like typo, "
            "text_inconsistency, pattern_inconsistency, "
            "semantic_incompatible). 'auto' = dispatcher picks per "
            "error-type scope: single-column types run in 'single', "
            "multi-column / multi-row / multi-table types run in "
            "'all'. Overrides data.column_mode."
        ),
    )
    parser.add_argument(
        "--columns",
        type=str,
        help=(
            "Comma-separated list of CSV column names to analyse. "
            "Defaults to every column. Especially useful with "
            "--column-mode single to focus on a handful of columns."
        ),
    )
    parser.add_argument(
        "--reasoning-effort", type=str, help="Override llm.reasoning_effort."
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override output.directory."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Use this exact directory name (under --output-dir) for the "
            "run instead of the auto-generated <timestamp>_<model> form. "
            "Useful when scripting reproducible demos or CI artifacts."
        ),
    )
    parser.add_argument(
        "--error-types",
        type=str,
        help=(
            "Comma-separated list of error_type keys to run. Defaults to "
            "every type in the selected taxonomy. Run "
            "`--list-error-types` to see the keys available in the "
            "currently-loaded taxonomy. Example (core taxonomy): "
            "'lexical_inconsistency,formatting_inconsistency'."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help=(
            "Maximum number of concurrent LLM calls (detection). Defaults "
            "to min(8, #endpoints). Use 1 for deterministic sequential runs."
        ),
    )
    parser.add_argument(
        "--error-types-file",
        type=Path,
        help="Override prompts.error_types_file (path to error_types.yaml).",
    )
    parser.add_argument(
        "--synthesis-prompt-file",
        type=Path,
        help=(
            "Override prompts.synthesis_prompt_file (Phase-B YAML template)."
        ),
    )
    parser.add_argument(
        "--no-synthesis",
        action="store_true",
        help=(
            "Skip Phase-B LLM synthesis of multi-type clusters. Phase-A "
            "deterministic clustering is always performed."
        ),
    )
    parser.add_argument(
        "--synthesis-concurrency",
        type=int,
        help=(
            "Max concurrent Phase-B synthesis calls. Defaults to "
            "min(8, #endpoints)."
        ),
    )
    parser.add_argument(
        "--synthesis-effort",
        type=str,
        help=(
            "Override llm.synthesis_effort for the Phase-B call "
            "(default: medium)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Render the first (chunk, type) prompt to stdout and exit "
            "(no LLM call)."
        ),
    )
    parser.add_argument(
        "--normalize-input",
        action="store_true",
        help=(
            "Apply deterministic text normalization (NFKC, whitespace "
            "trim/collapse, fold NBSP/ZWSP) to every CSV cell before "
            "rendering it into the LLM prompt. Useful as an ablation: "
            "compare detections with vs without normalization to see "
            "how much LLM volume is driven by mechanical formatting "
            "noise. Casing and punctuation are NOT folded by default "
            "(see --normalize-fold-punctuation / --normalize-lowercase)."
        ),
    )
    parser.add_argument(
        "--normalize-fold-punctuation",
        action="store_true",
        help=(
            "When --normalize-input is set, also fold smart quotes, en/em "
            "dashes, ellipsis, etc. to ASCII. Off by default."
        ),
    )
    parser.add_argument(
        "--normalize-lowercase",
        action="store_true",
        help=(
            "When --normalize-input is set, also lowercase every string "
            "cell. DESTRUCTIVE - off by default; use only for casing-"
            "ablation studies."
        ),
    )
    parser.add_argument(
        "--no-table-digest",
        action="store_true",
        help=(
            "Disable the TABLE OVERVIEW block prepended to every "
            "per-type detection prompt. Overrides "
            "data.include_table_digest in the YAML config. The "
            "digest carries per-column stats, candidate FDs, and "
            "total-row hints used by the multi-column / multi-row "
            "error types; turn off for ablations or when running an "
            "exclusively single-column subset on very wide tables."
        ),
    )
    parser.add_argument(
        "--list-error-types",
        action="store_true",
        help="Print the loaded error-type taxonomy and exit.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help=(
            "Load + validate --config, --csv path(s) and --error-types "
            "filter, print a summary, and exit 0 if everything resolves "
            "(non-zero on any error). Useful in CI/demo scripts to "
            "catch typos before kicking off an LLM run."
        ),
    )
    parser.add_argument(
        "--scopes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of error-type SCOPES to keep. A scope is "
            "the prefix of an ErrorType.category up to the first '/' - one "
            "of: single-column, multi-column, multi-row, multi-table. "
            "Pass any combination, e.g. 'single-column' or "
            "'single-column,multi-column'. Applied AFTER --error-types "
            "subset selection. 'all' (or empty) disables the filter. "
            "Overrides prompts.scopes."
        ),
    )

    # --- row_duplication detector knobs ------------------------------
    # These override the ``row_duplication`` block in
    # ``config/experiment.yaml`` and are only consulted when
    # ``row_duplication`` is among the selected error types.
    parser.add_argument(
        "--row-dup-detection-mode",
        choices=("collision_adjudication", "legacy_chunk_scan", "whole_table"),
        default=None,
        help=(
            "Strategy for the row_duplication detector. "
            "'collision_adjudication' (default) runs a deterministic "
            "key-collision scan over the whole table and adjudicates "
            "each group with one LLM call. 'legacy_chunk_scan' falls "
            "back to the original per-chunk LLM scan (misses pairs that "
            "straddle chunk boundaries). 'whole_table' is a PoC mode "
            "that skips clustering entirely and hands the full CSV to "
            "one LLM call per source under the same Chao1/GT adaptive "
            "stop rule (capped by row_duplication.whole_table_max_rows). "
            "Overrides row_duplication.detection_mode."
        ),
    )
    parser.add_argument(
        "--row-dup-normalize-keys",
        dest="row_dup_normalize_keys",
        action="store_true",
        default=None,
        help=(
            "Apply case/punctuation normalisation when scanning for "
            "duplicate keys (catches 'Acme Corp.' vs 'ACME CORP'). "
            "Overrides row_duplication.normalize_keys."
        ),
    )
    parser.add_argument(
        "--no-row-dup-normalize-keys",
        dest="row_dup_normalize_keys",
        action="store_false",
        help="Disable key normalisation (overrides the config).",
    )
    parser.add_argument(
        "--row-dup-adjudicate",
        dest="row_dup_adjudicate",
        action="store_true",
        default=None,
        help=(
            "Have the LLM adjudicate every collision group. Overrides "
            "row_duplication.adjudicate_groups."
        ),
    )
    parser.add_argument(
        "--no-row-dup-adjudicate",
        dest="row_dup_adjudicate",
        action="store_false",
        help=(
            "Emit deterministic collision groups straight away with no "
            "LLM verification. Overrides the config."
        ),
    )
    parser.add_argument(
        "--row-dup-max-groups-per-source",
        type=int,
        default=None,
        help=(
            "Cap on the number of collision groups adjudicated per "
            "source. Groups are ranked by size (largest first). "
            "Overrides row_duplication.max_groups_per_source."
        ),
    )
    parser.add_argument(
        "--row-dup-canonicalize-keys",
        dest="row_dup_canonicalize_keys",
        action="store_true",
        default=None,
        help=(
            "Run a per-column LLM canonicalisation pass over each "
            "candidate key column before the collision scan, so "
            "semantically-equivalent surface forms ('US' / 'United "
            "States') collide deterministically. One LLM call per "
            "(source, key column). Overrides "
            "row_duplication.canonicalize_keys."
        ),
    )
    parser.add_argument(
        "--no-row-dup-canonicalize-keys",
        dest="row_dup_canonicalize_keys",
        action="store_false",
        help=(
            "Disable per-column LLM canonicalisation (overrides the "
            "config)."
        ),
    )
    parser.add_argument(
        "--row-dup-canonicalize-max-cardinality",
        type=int,
        default=None,
        help=(
            "Skip LLM canonicalisation for columns whose distinct "
            "non-null value count exceeds this cap (the prompt would "
            "be too long for a single reliable call). Overrides "
            "row_duplication.canonicalize_max_cardinality."
        ),
    )
    parser.add_argument(
        "--row-dup-blocking",
        dest="row_dup_blocking_enabled",
        action="store_true",
        default=None,
        help=(
            "Enable token-signature blocking: surface rows that share "
            "a signature (first token of each candidate-key column) "
            "as candidate duplicate clusters. Free; high recall."
        ),
    )
    parser.add_argument(
        "--no-row-dup-blocking",
        dest="row_dup_blocking_enabled",
        action="store_false",
        help="Disable token-signature blocking.",
    )
    parser.add_argument(
        "--row-dup-minhash",
        dest="row_dup_minhash_enabled",
        action="store_true",
        default=None,
        help=(
            "Enable MinHash LSH near-duplicate clusterer. Requires "
            "``datasketch`` installed. Catches row-level Jaccard "
            "near-duplicates."
        ),
    )
    parser.add_argument(
        "--no-row-dup-minhash",
        dest="row_dup_minhash_enabled",
        action="store_false",
        help="Disable MinHash LSH clusterer.",
    )
    parser.add_argument(
        "--row-dup-minhash-threshold",
        type=float,
        default=None,
        help=(
            "Jaccard threshold for the MinHash LSH clusterer "
            "(0..1). Lower = higher recall."
        ),
    )
    parser.add_argument(
        "--row-dup-embedding",
        dest="row_dup_embedding_enabled",
        action="store_true",
        default=None,
        help=(
            "Enable sentence-embedding near-duplicate clusterer. "
            "Requires ``sentence-transformers`` installed. Catches "
            "semantic / paraphrase near-duplicates."
        ),
    )
    parser.add_argument(
        "--no-row-dup-embedding",
        dest="row_dup_embedding_enabled",
        action="store_false",
        help="Disable embedding clusterer.",
    )
    parser.add_argument(
        "--row-dup-embedding-threshold",
        type=float,
        default=None,
        help=(
            "Cosine-similarity threshold for the embedding clusterer "
            "(0..1). Lower = higher recall."
        ),
    )
    parser.add_argument(
        "--row-dup-embedding-model",
        default=None,
        help=(
            "sentence-transformers model name. Default: "
            "all-MiniLM-L6-v2. Try all-mpnet-base-v2 for higher "
            "quality at ~3x cost."
        ),
    )

    # --- Experiment-mode + Good-Turing knobs --------------------------
    # These are only consulted when --mode=adaptive (or when
    # experiment.mode=adaptive in config). In single mode they are
    # silently ignored.
    parser.add_argument(
        "--mode",
        choices=("single", "adaptive"),
        help=(
            "Experiment strategy. 'single' = one detection pass (the "
            "classic flow). 'adaptive' = multi-pass with perturbations "
            "and a Good-Turing / Chao1 FN-rate estimator; stopping is "
            "governed by --gt-mode. Overrides experiment.mode."
        ),
    )
    parser.add_argument(
        "--gt-mode",
        choices=("fixed", "adaptive"),
        help=(
            "Within --mode=adaptive: 'fixed' runs exactly --gt-num-runs "
            "passes; 'adaptive' stops once Good-Turing missing mass "
            "<= --gt-target-missing-mass (min --gt-min-runs, max "
            "--gt-num-runs). Overrides good_turing.mode."
        ),
    )
    parser.add_argument(
        "--gt-num-runs",
        type=int,
        help=(
            "Max / exact number of perturbed runs for --mode=adaptive. "
            "Overrides good_turing.num_runs."
        ),
    )
    parser.add_argument(
        "--gt-min-runs",
        type=int,
        help=(
            "Minimum runs before early-stop is allowed. Overrides "
            "good_turing.min_runs."
        ),
    )
    parser.add_argument(
        "--gt-target-missing-mass",
        type=float,
        help=(
            "Good-Turing missing-mass stopping threshold (e.g. 0.05). "
            "Overrides good_turing.target_missing_mass."
        ),
    )
    parser.add_argument(
        "--gt-stop-scope",
        choices=(
            "dataset", "column", "column_pair", "row",
            "csv", "workbook", "cross_csv", "multi_table",
        ),
        help=(
            "Granularity at which --gt-mode=adaptive decides when to "
            "stop. 'column' (default) treats each (source, column, "
            "error_type) as an independent arm; the loop ends when "
            "every error_type has all its arms saturated. 'dataset' "
            "pools across columns/sources so each error_type has one "
            "arm (equivalent to the deprecated target_policy=per_type)."
        ),
    )
    parser.add_argument(
        "--gt-target-policy",
        choices=("per_type",),
        help=(
            "DEPRECATED. 'per_type' is now an alias for "
            "--gt-stop-scope=dataset. 'overall' has been removed; use "
            "--gt-stop-scope=dataset (one arm per error_type) or "
            "--gt-stop-scope=column (one arm per (column, error_type))."
        ),
    )
    parser.add_argument(
        "--gt-base-seed",
        type=int,
        help="Seed derivation root. Overrides good_turing.base_seed.",
    )
    parser.add_argument(
        "--gt-shuffle-rows",
        dest="gt_shuffle_rows",
        action="store_true",
        default=None,
        help="Permute row order per run. Overrides good_turing.shuffle_rows.",
    )
    parser.add_argument(
        "--no-gt-shuffle-rows",
        dest="gt_shuffle_rows",
        action="store_false",
        help="Disable row-order permutation (overrides the config).",
    )
    parser.add_argument(
        "--gt-shuffle-columns",
        dest="gt_shuffle_columns",
        action="store_true",
        default=None,
        help="Also permute column order per run. Overrides good_turing.shuffle_columns.",
    )
    parser.add_argument(
        "--no-gt-shuffle-columns",
        dest="gt_shuffle_columns",
        action="store_false",
        help="Disable column-order permutation (overrides the config).",
    )
    parser.add_argument(
        "--gt-prompt-ensemble",
        dest="gt_prompt_ensemble",
        action="store_true",
        default=None,
        help=(
            "On every non-identity pass, ask an LLM to rewrite each "
            "selected error type's description+notes+example (preserving "
            "scope). Recommended for cell-local error types. Overrides "
            "good_turing.prompt_ensemble."
        ),
    )
    parser.add_argument(
        "--no-gt-prompt-ensemble",
        dest="gt_prompt_ensemble",
        action="store_false",
        help="Disable LLM-generated prompt variants (overrides the config).",
    )
    parser.add_argument(
        "--gt-variant-template",
        type=Path,
        help=(
            "Path to the rewriter prompt YAML. Overrides "
            "good_turing.prompt_variant_template (default "
            "prompts/generate_prompt_variant.yaml)."
        ),
    )
    parser.add_argument(
        "--gt-variant-effort",
        type=str,
        help=(
            "Reasoning effort for the rewriter call. Overrides "
            "good_turing.variant_generation_effort (default 'low')."
        ),
    )
    parser.add_argument(
        "--gt-skip-identity-run",
        dest="gt_skip_identity_run",
        action="store_true",
        default=None,
        help=(
            "Perturb every run (no identity baseline). Overrides "
            "good_turing.skip_identity_run."
        ),
    )
    parser.add_argument(
        "--no-gt-skip-identity-run",
        dest="gt_skip_identity_run",
        action="store_false",
        help="Keep run 0 as the unshuffled baseline (overrides the config).",
    )

    # --- Per-arm scope-aware estimator overrides ---------------------
    # Override the ``good_turing.per_arm.defaults`` block. Each is a
    # global default applied to every arm unless overridden by a YAML
    # ``per_error_type`` entry. Leaving them as None falls through to
    # YAML, then to ArmConfig.default().
    parser.add_argument(
        "--gt-per-arm-two-consecutive",
        dest="gt_per_arm_two_consecutive",
        action="store_true",
        default=None,
        help=(
            "Require two consecutive passes meeting the threshold "
            "before declaring an arm saturated. Overrides "
            "good_turing.per_arm.defaults.require_two_consecutive."
        ),
    )
    parser.add_argument(
        "--no-gt-per-arm-two-consecutive",
        dest="gt_per_arm_two_consecutive",
        action="store_false",
        help=(
            "Allow saturation after a single passing run "
            "(overrides the config)."
        ),
    )
    parser.add_argument(
        "--gt-per-arm-stop-when",
        dest="gt_per_arm_stop_when",
        choices=["gt", "chao1", "both"],
        default=None,
        help=(
            "Which estimator(s) gate per-arm saturation. "
            "'gt' (default) -> Good-Turing point estimate p_0 <= "
            "epsilon (bounds probability mass on undetected "
            "species). 'chao1' -> Chao1 upper-95 missing-count "
            "fraction (chao1_ci_95_high - S_obs) / S_obs <= "
            "epsilon_chao1 (bounds count of undetected species). "
            "'both' -> require both criteria simultaneously "
            "(strongest end-to-end FN guarantee). Overrides "
            "good_turing.per_arm.defaults.stop_when."
        ),
    )
    parser.add_argument(
        "--gt-per-arm-epsilon-chao1",
        dest="gt_per_arm_epsilon_chao1",
        type=float,
        default=None,
        help=(
            "Threshold for the Chao1 stop criterion (only consulted "
            "when --gt-per-arm-stop-when is 'chao1' or 'both'). "
            "Default 0.05 = 'estimated missing count is at most 5%% "
            "of what we've already observed'. Overrides "
            "good_turing.per_arm.defaults.epsilon_chao1."
        ),
    )
    parser.add_argument(
        "--gt-no-stop",
        dest="gt_no_stop",
        action="store_true",
        default=False,
        help=(
            "Force every arm to run until ``--gt-num-runs`` (k_max) "
            "regardless of any saturation predicate. Sets epsilon and "
            "epsilon_chao1 to 0 internally so no arm can ever satisfy "
            "the stop predicate, and lifts k_min to k_max. Used in "
            "conjunction with the per-pass per-arm snapshot dump for "
            "offline hyperparameter search via "
            "``experiments/analyze_stop_thresholds.py`` -- one full "
            "K_max run produces a ground-truth trace against which "
            "any (stop_when, epsilon, epsilon_chao1, k_min, "
            "two_consec) combination can be replayed without "
            "spending more LLM tokens."
        ),
    )

    # --- Web verification (post-pass) ---------------------------------
    parser.add_argument(
        "--verify-web",
        action="store_true",
        help=(
            "After detection finishes, run the web-grounded verifier "
            "on eligible clusters (org / address / phone / url columns "
            "with a typo / text-inconsistency error type). Writes "
            "<run_dir>/web_verifications.json. Requires "
            "AZURE_AGENTS_ENDPOINT and AZURE_AGENTS_VERIFIER_AGENT_ID "
            "in the environment (or --web-agent-endpoint / --web-agent-id)."
        ),
    )
    parser.add_argument(
        "--web-agent-endpoint",
        type=str,
        default=None,
        help="Foundry project endpoint URL. Falls back to AZURE_AGENTS_ENDPOINT.",
    )
    parser.add_argument(
        "--web-agent-id",
        type=str,
        default=None,
        help="Foundry agent id (asst_...). Falls back to "
             "AZURE_AGENTS_VERIFIER_AGENT_ID.",
    )
    parser.add_argument(
        "--web-concurrency",
        type=int,
        default=4,
        help="Verifier agent-call concurrency (default: 4).",
    )
    parser.add_argument(
        "--web-max-clusters",
        type=int,
        default=None,
        help="Cap on clusters to verify (after gating). Useful for smoke tests.",
    )
    parser.add_argument(
        "--web-include-person",
        action="store_true",
        help=(
            "Also verify person-name columns. Off by default for "
            "privacy (PII)."
        ),
    )
    parser.add_argument(
        "--web-force",
        action="store_true",
        help=(
            "Re-verify even clusters that already have a prior "
            "verification, and clear the on-disk web cache first."
        ),
    )
    return parser.parse_args(argv)


def _resolve(path_value: Any, base: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else base / p


def _resolve_synthesis_enabled(
    args: argparse.Namespace, llm_cfg: Dict[str, Any]
) -> bool:
    if args.no_synthesis:
        return False
    # Config uses the explicit `enable_synthesis` key (default true) to
    # match the positive CLI flag convention.
    cfg_flag = llm_cfg.get("enable_synthesis", True)
    return bool(cfg_flag)


def _prepare_chunks(
    source_frames: Dict[str, "pd.DataFrame"],
    *,
    chunk_size: int,
    max_chunks: Optional[int],
    start_row: int,
    column_mode: str,
    active_columns: Optional[List[str]],
) -> Tuple[List[CsvChunk], Dict[str, int]]:
    """Build the per-(chunk, error_type) list for a given set of frames.

    Mirrors what the single-pass path does inline; factored out so the
    adaptive loop can call it once per perturbed pass. ``source_frames``
    is keyed by source name (matches ``CsvChunk.source``'s file name).
    """
    chunks: List[CsvChunk] = []
    per_source: Dict[str, int] = {}
    for name, df_i in source_frames.items():
        if active_columns is not None:
            present = [c for c in active_columns if c in df_i.columns]
            missing = [c for c in active_columns if c not in df_i.columns]
            if missing:
                logger.debug(
                    "  skipping columns not present in %s: %s", name, missing
                )
            if not present:
                logger.warning(
                    "  no requested columns present in %s - skipping file", name
                )
                continue
            cols_for_file: Optional[List[str]] = present
        else:
            cols_for_file = None
        file_chunks = list(
            iter_chunks(
                df_i,
                source=Path(name),
                chunk_size=chunk_size,
                max_chunks=max_chunks,
                start_row=start_row,
                column_mode=column_mode,
                columns=cols_for_file,
            )
        )
        chunks.extend(file_chunks)
        per_source[name] = len(file_chunks)
    return chunks, per_source


def _split_error_types_by_scope(
    column_mode: str,
    error_types: List[Any],
) -> List[Tuple[str, List[Any]]]:
    """Group error types by the column-mode they should run under.

    For ``column_mode=auto`` (dispatcher-only mode), split the selected
    error types so that single-column-scope types run under
    ``column_mode=single`` (one LLM call per (row-slice, column) pair)
    and every other-scope type runs under ``column_mode=all`` (one
    call per row-slice with every column visible). Other scopes
    (multi-column / multi-row / multi-table) intrinsically need to
    see the whole row at once, so forcing them through 'single' would
    silently neuter them.

    For non-auto modes, returns a single group preserving input
    order so the caller's behaviour is unchanged.

    Empty groups are dropped, so a returned list of length 0 means
    nothing to scan.
    """
    if column_mode != COLUMN_MODE_AUTO:
        return [(column_mode, list(error_types))] if error_types else []
    sc: List[Any] = []
    other: List[Any] = []
    for et in error_types:
        scope = _scope_of(getattr(et, "category", "") or "")
        if scope == "single-column":
            sc.append(et)
        else:
            other.append(et)
    groups: List[Tuple[str, List[Any]]] = []
    if sc:
        groups.append((COLUMN_MODE_SINGLE, sc))
    if other:
        groups.append((COLUMN_MODE_ALL, other))
    return groups


def _merge_adaptive_gt_artifacts(
    group_artifacts: List[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Combine per-scope-group adaptive ``gt_artifacts`` into one dict.

    In ``column_mode=auto`` the adaptive loop runs once per non-empty
    scope group (single-column types vs everything else). Each group
    produces an independent Good-Turing/Chao1 estimator over its own
    species pool; this helper stitches the per-group outputs into a
    single artifact bundle that the existing single-pass downstream
    code (config snapshot, ``good_turing/`` dump, HTML dashboard) can
    consume unchanged.

    Merging rules:

    * Each error type belongs to exactly one group, so
      ``per_error_type`` and ``species_records`` are simple unions.
    * ``overall_estimate`` and ``per_source`` are RECOMPUTED from the
      union of species records using ``num_runs = max`` across groups
      (the per-group runs were independent, so the unioned-species
      estimator is the appropriate combined view).
    * Lists tracking per-pass / per-run state (``progression``,
      ``per_run_dumps``, ``run_info``, ``per_pass_per_arm_snapshots``,
      ``variant_records_per_run``, ``saturation_history``) are
      concatenated with a ``group`` tag added for traceability.
    * ``per_run_dumps[*]['run_id']`` is offset across groups so
      downstream dump paths (``runs/run_NN/``) stay unique.
    """
    if len(group_artifacts) == 1:
        return group_artifacts[0][1]

    from src.adaptive.cell_gt import estimate as _estimate_overall

    species_records: List[Any] = []
    progression: List[Dict[str, Any]] = []
    per_run_dumps: List[Dict[str, Any]] = []
    variant_records_per_run: List[Any] = []
    per_pass_per_arm_snapshots: List[Dict[str, Any]] = []
    run_info: List[Any] = []
    saturated_types: set = set()
    saturation_history: List[Dict[str, Any]] = []
    per_error_type: Dict[str, Any] = {}
    runs_probed_per_type: Dict[str, int] = {}
    variant_passes_ok_per_type: Dict[str, int] = {}
    markdown_sections: List[str] = []
    stop_reason_parts: List[str] = []
    num_runs_completed = 0
    num_runs_max = 0
    min_runs_seen = 0
    target_missing_mass: Optional[float] = None
    stop_scope: Optional[str] = None
    target_policy: Optional[str] = None
    gt_mode: Optional[str] = None
    base_seed: Optional[int] = None
    shuffle_rows: Optional[bool] = None
    shuffle_columns: Optional[bool] = None
    include_identity_run: Optional[bool] = None
    prompt_ensemble: Optional[bool] = None
    variant_generation_effort: Optional[str] = None
    per_arm_arms: List[Any] = []
    per_arm_md_sections: List[str] = []
    per_arm_scope_kind: Optional[str] = None
    per_arm_num_runs = 0

    run_id_offset = 0
    for group_mode, art in group_artifacts:
        md = art.get("markdown_report") or ""
        markdown_sections.append(
            f"## Auto-mode group: column_mode={group_mode}\n\n{md}".rstrip()
        )
        stop_reason_parts.append(
            f"{group_mode}: {art.get('stop_reason', '') or ''}"
        )
        species_records.extend(art.get("species_records") or [])
        for row in art.get("progression") or []:
            new_row = dict(row)
            new_row.setdefault("group", group_mode)
            progression.append(new_row)
        group_dump_count = 0
        for d in art.get("per_run_dumps") or []:
            new_d = dict(d)
            try:
                rid = int(d.get("run_id", 0))
            except (TypeError, ValueError):
                rid = 0
            new_d["run_id"] = rid + run_id_offset
            new_d["group"] = group_mode
            new_d["original_run_id"] = rid
            # Carry the same shift into the meta blob if present so
            # the on-disk ``run.json`` stays consistent.
            if isinstance(new_d.get("meta"), dict):
                meta = dict(new_d["meta"])
                if "run_id" in meta:
                    try:
                        meta["run_id"] = int(meta["run_id"]) + run_id_offset
                    except (TypeError, ValueError):
                        pass
                meta.setdefault("group", group_mode)
                new_d["meta"] = meta
            per_run_dumps.append(new_d)
            group_dump_count += 1
        run_id_offset += group_dump_count
        for v in art.get("variant_records_per_run") or []:
            if isinstance(v, dict):
                new_v = dict(v)
                new_v.setdefault("group", group_mode)
                variant_records_per_run.append(new_v)
            else:
                variant_records_per_run.append(v)
        for snap in art.get("per_pass_per_arm_snapshots") or []:
            new_snap = dict(snap)
            new_snap.setdefault("group", group_mode)
            per_pass_per_arm_snapshots.append(new_snap)
        for r in art.get("run_info") or []:
            if isinstance(r, dict):
                new_r = dict(r)
                new_r.setdefault("group", group_mode)
                run_info.append(new_r)
            else:
                run_info.append(r)
        ep = art.get("estimates_payload") or {}
        for k in ep.get("saturated_types", []) or []:
            saturated_types.add(k)
        for h in ep.get("saturation_history", []) or []:
            new_h = dict(h)
            new_h.setdefault("group", group_mode)
            saturation_history.append(new_h)
        per_error_type.update(art.get("per_error_type") or {})
        runs_probed_per_type.update(ep.get("runs_probed_per_type") or {})
        variant_passes_ok_per_type.update(
            ep.get("variant_passes_ok_per_type") or {}
        )
        nrc = int(art.get("num_runs_completed", 0) or 0)
        num_runs_completed = max(num_runs_completed, nrc)
        if ep:
            num_runs_max = max(num_runs_max, int(ep.get("num_runs_max", 0) or 0))
            min_runs_seen = max(min_runs_seen, int(ep.get("min_runs", 0) or 0))
            target_missing_mass = ep.get("target_missing_mass", target_missing_mass)
            stop_scope = ep.get("stop_scope", stop_scope)
            target_policy = ep.get("target_policy", target_policy)
            gt_mode = ep.get("gt_mode", gt_mode)
            base_seed = ep.get("base_seed", base_seed)
            shuffle_rows = ep.get("shuffle_rows", shuffle_rows)
            shuffle_columns = ep.get("shuffle_columns", shuffle_columns)
            include_identity_run = ep.get(
                "include_identity_run", include_identity_run
            )
            prompt_ensemble = ep.get("prompt_ensemble", prompt_ensemble)
            variant_generation_effort = ep.get(
                "variant_generation_effort", variant_generation_effort
            )
        per_arm_payload = art.get("per_arm_payload")
        if per_arm_payload:
            per_arm_arms.extend(per_arm_payload.get("arms") or [])
            per_arm_scope_kind = (
                per_arm_payload.get("scope_kind") or per_arm_scope_kind
            )
            per_arm_num_runs = max(
                per_arm_num_runs,
                int(per_arm_payload.get("num_runs_completed", 0) or 0),
            )
        if art.get("per_arm_markdown"):
            per_arm_md_sections.append(
                f"## Auto-mode group: column_mode={group_mode}\n\n"
                + str(art["per_arm_markdown"])
            )

    overall = _estimate_overall(species_records, num_runs=num_runs_completed)

    per_source_dict: Dict[str, Any] = {}
    by_source: Dict[str, List[Any]] = {}
    for rec in species_records:
        src = getattr(rec, "source", None)
        if src is None:
            continue
        by_source.setdefault(str(src), []).append(rec)
    for src_name, recs in by_source.items():
        per_source_dict[src_name] = _estimate_overall(
            recs, num_runs=num_runs_completed
        ).to_dict()

    estimates_payload: Dict[str, Any] = {
        "gt_mode": gt_mode,
        "num_runs_completed": num_runs_completed,
        "num_runs_max": num_runs_max or num_runs_completed,
        "min_runs": min_runs_seen,
        "target_missing_mass": target_missing_mass,
        "stop_scope": stop_scope,
        "target_policy": target_policy,
        "saturated_types": sorted(saturated_types),
        "saturation_history": saturation_history,
        "runs_probed_per_type": runs_probed_per_type,
        "variant_passes_ok_per_type": variant_passes_ok_per_type,
        "stop_reason": "; ".join(stop_reason_parts),
        "base_seed": base_seed,
        "shuffle_rows": shuffle_rows,
        "shuffle_columns": shuffle_columns,
        "include_identity_run": include_identity_run,
        "prompt_ensemble": prompt_ensemble,
        "variant_generation_effort": variant_generation_effort,
        "overall": overall.to_dict(),
        "per_error_type": per_error_type,
        "per_source": per_source_dict,
        "progression": progression,
        "runs": run_info,
    }
    merged: Dict[str, Any] = {
        "num_runs_completed": num_runs_completed,
        "stop_reason": "; ".join(stop_reason_parts),
        "species_records": species_records,
        "progression": progression,
        "estimates_payload": estimates_payload,
        "markdown_report": "\n\n".join(markdown_sections).strip(),
        "overall_estimate": overall.to_dict(),
        "per_run_dumps": per_run_dumps,
        "variant_records_per_run": variant_records_per_run,
        "per_pass_per_arm_snapshots": per_pass_per_arm_snapshots,
        "per_error_type": per_error_type,
        "per_source": per_source_dict,
        "run_info": run_info,
        "per_arm_payload": (
            {
                "scope_kind": per_arm_scope_kind,
                "num_runs_completed": per_arm_num_runs,
                "arms": per_arm_arms,
            }
            if per_arm_arms
            else None
        ),
        "per_arm_markdown": (
            "\n\n".join(per_arm_md_sections).strip()
            if per_arm_md_sections
            else None
        ),
    }
    return merged


def _resolve_adaptive_settings(
    args: argparse.Namespace, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve Good-Turing adaptive-loop settings from CLI + YAML.

    CLI wins, else the ``good_turing:`` YAML block, else built-in
    defaults. Returns a plain dict of resolved values.
    """
    gt_cfg = config.get("good_turing", {}) or {}
    gt_mode_raw = args.gt_mode or str(gt_cfg.get("mode", "adaptive"))
    gt_mode = gt_mode_raw.lower()
    if gt_mode not in {"fixed", "adaptive"}:
        raise ValueError(
            f"good_turing.mode must be 'fixed' or 'adaptive', got {gt_mode_raw!r}"
        )
    num_runs = (
        args.gt_num_runs if args.gt_num_runs is not None
        else int(gt_cfg.get("num_runs", 5))
    )
    min_runs = (
        args.gt_min_runs if args.gt_min_runs is not None
        else int(gt_cfg.get("min_runs", 3))
    )
    target_missing_mass = (
        args.gt_target_missing_mass if args.gt_target_missing_mass is not None
        else float(gt_cfg.get("target_missing_mass", 0.05))
    )

    # ``stop_scope`` is the new canonical name for "what's an arm".
    # The legacy ``target_policy`` knob is now a compatibility shim:
    #
    #   target_policy: per_type -> stop_scope: dataset (one arm per
    #                                          error_type, pooled
    #                                          across columns/sources).
    #   target_policy: overall  -> hard error (removed).
    #
    # Resolution order is CLI > YAML; if both name and alias appear,
    # the new name wins.
    stop_scope_cli = getattr(args, "gt_stop_scope", None)
    target_policy_cli = getattr(args, "gt_target_policy", None)
    stop_scope_raw: Optional[str] = (
        stop_scope_cli
        if stop_scope_cli is not None
        else gt_cfg.get("stop_scope")
    )
    if stop_scope_raw is None:
        # Fall back to the deprecated knob.
        legacy = (
            target_policy_cli
            if target_policy_cli is not None
            else gt_cfg.get("target_policy")
        )
        if legacy is not None:
            legacy = str(legacy).lower()
            if legacy == "overall":
                raise ValueError(
                    "good_turing.target_policy='overall' has been removed. "
                    "Use --gt-stop-scope dataset (or column) instead. The "
                    "loop now stops when every active error_type's arm at "
                    "the chosen scope has saturated; there is no longer a "
                    "single pooled missing-mass threshold across types."
                )
            if legacy == "per_type":
                logger.warning(
                    "target_policy=per_type is deprecated; "
                    "use stop_scope=dataset"
                )
                stop_scope_raw = "dataset"
            else:
                raise ValueError(
                    "good_turing.target_policy must be 'per_type' "
                    "(deprecated alias for stop_scope=dataset); "
                    f"got {legacy!r}"
                )
    if stop_scope_raw is None:
        stop_scope_raw = "column"  # new default
    stop_scope = str(stop_scope_raw).lower()
    if stop_scope not in {
        "dataset", "column", "column_pair", "row",
        "csv", "workbook", "cross_csv", "multi_table",
    }:
        raise ValueError(
            "good_turing.stop_scope must be one of "
            "'dataset','column','column_pair','row','csv','workbook',"
            f"'cross_csv','multi_table'; got {stop_scope_raw!r}"
        )
    base_seed = (
        args.gt_base_seed if args.gt_base_seed is not None
        else int(gt_cfg.get("base_seed", 42))
    )
    shuffle_rows = (
        args.gt_shuffle_rows if args.gt_shuffle_rows is not None
        else bool(gt_cfg.get("shuffle_rows", False))
    )
    shuffle_columns = (
        args.gt_shuffle_columns if args.gt_shuffle_columns is not None
        else bool(gt_cfg.get("shuffle_columns", False))
    )
    skip_identity_run = (
        args.gt_skip_identity_run if args.gt_skip_identity_run is not None
        else bool(gt_cfg.get("skip_identity_run", False))
    )
    prompt_ensemble = (
        args.gt_prompt_ensemble if args.gt_prompt_ensemble is not None
        else bool(gt_cfg.get("prompt_ensemble", True))
    )
    variant_template_raw = (
        args.gt_variant_template if args.gt_variant_template is not None
        else gt_cfg.get(
            "prompt_variant_template", "prompts/generate_prompt_variant.yaml"
        )
    )
    variant_generation_effort = (
        args.gt_variant_effort if args.gt_variant_effort is not None
        else str(gt_cfg.get("variant_generation_effort", "low"))
    )
    if num_runs < 1:
        raise ValueError(f"good_turing.num_runs must be >= 1, got {num_runs}")
    if min_runs < 1:
        raise ValueError(f"good_turing.min_runs must be >= 1, got {min_runs}")
    if min_runs > num_runs:
        logger.warning(
            "good_turing.min_runs (%d) > num_runs (%d); clamping min_runs",
            min_runs, num_runs,
        )
        min_runs = num_runs
    if not (0.0 <= target_missing_mass <= 1.0):
        raise ValueError(
            "good_turing.target_missing_mass must be in [0, 1], "
            f"got {target_missing_mass}"
        )
    return {
        "gt_mode": gt_mode,
        "num_runs": num_runs,
        "min_runs": min_runs,
        "target_missing_mass": target_missing_mass,
        "stop_scope": stop_scope,
        # Back-compat alias kept on the settings dict so consumers
        # written before the rename still work; new code should read
        # ``stop_scope`` directly.
        "target_policy": (
            "per_type" if stop_scope == "dataset" else "stop_scope"
        ),
        "base_seed": base_seed,
        "shuffle_rows": shuffle_rows,
        "shuffle_columns": shuffle_columns,
        "skip_identity_run": skip_identity_run,
        "prompt_ensemble": prompt_ensemble,
        "prompt_variant_template": str(variant_template_raw),
        "variant_generation_effort": variant_generation_effort,
        # Per-arm scope-aware estimator config; consumed by the
        # post-loop additive artifact builder. Defaults are baked into
        # ArmConfig.default() so passing nothing yields the
        # recommended settings. ``target_missing_mass`` is plumbed in
        # as the fallback epsilon, and ``min_runs`` / ``num_runs``
        # are plumbed in as the per-arm ``k_min`` / ``k_max`` defaults
        # (these used to be duplicate knobs pre-merge).
        "per_arm": _resolve_per_arm_config(
            config, args,
            fallback_epsilon=target_missing_mass,
            fallback_k_min=min_runs,
            fallback_k_max=num_runs,
        ),
    }


def _resolve_per_arm_config(
    config: Dict[str, Any],
    args: Optional[argparse.Namespace] = None,
    fallback_epsilon: Optional[float] = None,
    fallback_k_min: Optional[int] = None,
    fallback_k_max: Optional[int] = None,
) -> Tuple[ArmConfig, Dict[str, ArmConfig], Optional[Dict[str, List[str]]]]:
    """Pull the optional ``good_turing.per_arm`` block out of YAML.

    Schema (all optional)::

        good_turing:
          per_arm:
            defaults:        # global ArmConfig (epsilon / k_min / k_max / ...)
              epsilon: 0.05
              k_min: 3
              k_max: 10
              require_two_consecutive: true
            per_error_type:  # override defaults for one error_type
              typo:
                epsilon: 0.03
            applicability:   # whitelist {scope_kind: [error_types]}
              column: [typo, casing_inconsistency, unit_mismatch]

    Missing block / fields => :meth:`ArmConfig.default` everywhere and
    no applicability filter (every observed (scope_kind, error_type)
    arm is allowed). Validation is light - bad fields are passed
    through to ``ArmConfig`` which will surface a clear TypeError.

    CLI overrides on ``args`` (when provided) take precedence over the
    YAML ``defaults`` block; only fields explicitly passed on the CLI
    are applied. ``args`` is optional so the function is safe to call
    from tests / library code that doesn't construct an argparse
    namespace.
    """
    block = ((config.get("good_turing") or {}).get("per_arm") or {})
    defaults_raw = dict(block.get("defaults") or {})
    # Backward-compat: silently drop legacy ``s_min`` from YAML so
    # existing configs don't blow up after the field was removed.
    defaults_raw.pop("s_min", None)
    if args is not None:
        cli_overrides = {
            "require_two_consecutive": getattr(
                args, "gt_per_arm_two_consecutive", None
            ),
            "stop_when": getattr(args, "gt_per_arm_stop_when", None),
            "epsilon_chao1": getattr(
                args, "gt_per_arm_epsilon_chao1", None
            ),
        }
        for k, v in cli_overrides.items():
            if v is not None:
                defaults_raw[k] = v
    # ``fallback_*`` plumb the loop-level knobs into the per-arm
    # defaults: ``min_runs`` / ``num_runs`` ARE the per-arm
    # ``k_min`` / ``k_max`` defaults (they were duplicates pre-merge).
    # Per-error-type overrides in YAML ``per_arm.per_error_type`` can
    # still tune them per type.
    if fallback_epsilon is not None and "epsilon" not in defaults_raw:
        defaults_raw["epsilon"] = float(fallback_epsilon)
    if fallback_k_min is not None and "k_min" not in defaults_raw:
        defaults_raw["k_min"] = int(fallback_k_min)
    if fallback_k_max is not None and "k_max" not in defaults_raw:
        defaults_raw["k_max"] = int(fallback_k_max)
    # ``--gt-no-stop`` is the highest-priority override: it forces
    # every arm to run all the way to k_max regardless of any other
    # threshold setting. We force epsilon and epsilon_chao1 to 0 so
    # neither GT nor Chao1 criterion can ever satisfy the predicate,
    # AND lift k_min to k_max so the ``num_runs >= k_min`` gate also
    # never fires before the final pass. Both per_arm.per_error_type
    # overrides AND any earlier CLI / YAML defaults are clobbered for
    # consistency -- partial overrides defeat the purpose of forcing
    # a complete sweep.
    if args is not None and getattr(args, "gt_no_stop", False):
        k_max_floor = int(
            defaults_raw.get("k_max")
            or fallback_k_max
            or ArmConfig.default().k_max
            or 10
        )
        defaults_raw["epsilon"] = 0.0
        defaults_raw["epsilon_chao1"] = 0.0
        defaults_raw["k_min"] = k_max_floor
        defaults_raw["k_max"] = k_max_floor
    global_default = ArmConfig.default().merge_into(ArmConfig.default())
    if defaults_raw:
        global_default = ArmConfig(**defaults_raw).merge_into(
            ArmConfig.default()
        )
    per_type_raw = block.get("per_error_type") or {}
    if args is not None and getattr(args, "gt_no_stop", False):
        # Wipe per-error-type overrides too: any non-default epsilon /
        # k_min / k_max here would let some arms stop early and break
        # the "every arm runs to k_max" invariant the caller asked for.
        per_type_raw = {}
    per_type_cfg: Dict[str, ArmConfig] = {
        # Drop legacy ``s_min`` here too so per-error-type overrides
        # from older configs don't crash the resolver.
        str(k): ArmConfig(**{kk: vv for kk, vv in (v or {}).items() if kk != "s_min"})
        for k, v in per_type_raw.items()
    }
    appl_raw = block.get("applicability")
    applicability: Optional[Dict[str, List[str]]] = None
    if appl_raw:
        applicability = {
            str(k): [str(x) for x in (v or [])]
            for k, v in appl_raw.items()
        }
    return global_default, per_type_cfg, applicability


def _run_adaptive_loop(
    *,
    llm: LLMClient,
    model: str,
    base_taxonomy: Any,
    seed_error_types: List[Any],
    reasoning_effort: str,
    reasoning_summary: str,
    concurrency: Optional[int],
    source_frames: Dict[str, pd.DataFrame],
    chunk_size: int,
    max_chunks: Optional[int],
    start_row: int,
    column_mode: str,
    active_columns: Optional[List[str]],
    enable_synthesis: bool,
    settings: Dict[str, Any],
    variant_generator: Optional[PromptVariantGenerator],
    run_dir: Path,
    digests: Optional[Dict[str, TableDigest]] = None,
) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Run the adaptive (multi-pass + Good-Turing) detection loop.

    Returns a tuple of ``(all_reports, all_flat_issues, gt_artifacts)``
    where ``gt_artifacts`` bundles the estimator output for the caller
    to persist (progression.jsonl, estimates.json, report.md, and the
    species + per-run dumps).

    Each pass may apply two orthogonal perturbations:

    * **Frame perturbation** - row / column shuffle (seed derived from
      ``base_seed`` + pass index + source name).
    * **Prompt-ensemble perturbation** - if ``variant_generator`` is
      provided, the generator is invoked once per selected error type
      on every non-identity pass to rewrite ``description`` / ``notes``
      / ``example``. A fresh :class:`PerTypeCleaningDetector` is
      constructed for that pass using the variant taxonomy; the shared
      system prompt and user template stay fixed so raw detections
      remain comparable across runs.

    After every pass the cumulative Good-Turing / Chao1 estimate is
    recomputed; in ``gt_mode='adaptive'`` the loop breaks early as
    soon as the missing-mass threshold is met (and at least
    ``min_runs`` runs have completed).
    """
    gt_mode: str = settings["gt_mode"]
    num_runs: int = settings["num_runs"]
    min_runs: int = settings["min_runs"]
    target_missing_mass: float = settings["target_missing_mass"]
    stop_scope: str = settings.get("stop_scope", "column")
    per_arm_default, per_arm_by_type, per_arm_appl = settings.get(
        "per_arm",
        (
            ArmConfig.default().merge_into(ArmConfig.default()),
            {},
            None,
        ),
    )
    base_seed: int = settings["base_seed"]
    shuffle_rows: bool = settings["shuffle_rows"]
    shuffle_columns: bool = settings["shuffle_columns"]
    skip_identity_run: bool = settings["skip_identity_run"]
    prompt_ensemble_on: bool = bool(settings["prompt_ensemble"]) and (
        variant_generator is not None
    )

    all_reports: List[Any] = []
    all_flat_issues: List[Dict[str, Any]] = []
    detections_per_run: List[List[Dict[str, Any]]] = []
    progression: List[Dict[str, Any]] = []
    run_info: List[Dict[str, Any]] = []
    per_run_dumps: List[Dict[str, Any]] = []
    variant_records_per_run: List[Dict[str, Any]] = []
    stop_reason: Optional[str] = None
    # Per-type adaptive bookkeeping. Keys are ErrorType.key strings.
    # Once a type's GT missing mass falls to/below the target (after
    # at least ``min_runs`` passes), it is dropped from later passes
    # so we don't keep paying tokens to redetect well-saturated types.
    seed_type_keys: List[str] = [t.key for t in seed_error_types]
    saturated_types: set = set()
    saturation_history: List[Dict[str, Any]] = []
    # Per-type effective-K bookkeeping. ``runs_probed_per_type[k]`` is
    # the number of passes in which type ``k`` was actually evaluated
    # (i.e. it was in the active set that pass). Once a type saturates
    # it is dropped from later passes, so its effective K can be less
    # than the total number of completed runs - using the total would
    # mis-classify cells found only in its last probed pass as
    # "singletons against K=runs_completed" when in fact they were
    # singletons against K=runs_probed_per_type[k]. Pass this through
    # to ``estimate_by_group`` so per-type Chao1 / GT figures are
    # statistically sound after pruning.
    runs_probed_per_type: Dict[str, int] = {k: 0 for k in seed_type_keys}
    # Per-arm adaptive bookkeeping. ``saturated_arms`` tracks
    # arm-keys (((scope_kind, scope_id_tuple), error_type_key))
    # that have reached saturated/exhausted/not_applicable status
    # under the configured stop_scope. We use this to prune the next
    # pass at (chunk, error_type) granularity so the tqdm progress
    # bar shrinks as arms saturate. ``runs_probed_per_arm`` records
    # the effective K per arm; for unobserved arms (no detections
    # yet) it stays at the per-type fallback.
    saturated_arms: set = set()
    runs_probed_per_arm: Dict[Any, int] = {}
    # Per-pass per-arm snapshot for offline hyperparameter search.
    # Each entry is one ``(pass_idx, arm)`` row with the full
    # estimator state AS OF that pass's cumulative aggregation. Used
    # by ``experiments/analyze_stop_thresholds.py`` to replay any
    # ``(stop_when, epsilon, epsilon_chao1, k_min, two_consec)``
    # combination offline against a single forced-K_max run, so the
    # threshold sweep doesn't need new LLM calls per configuration.
    per_pass_per_arm_snapshots: List[Dict[str, Any]] = []
    # Per-arm streak / status memo across passes. ``evaluate_stop``
    # is *stateless* - to honour ``require_two_consecutive`` we have
    # to thread the previous pass's ``(status, streak)`` into the
    # next call ourselves. Without this the streak resets to 0 every
    # pass, the threshold is never reached, and arms can only ever
    # turn ``exhausted`` at K==k_max instead of ``saturated`` mid-run.
    arm_prev_status: Dict[Any, str] = {}
    arm_prev_streak: Dict[Any, int] = {}
    # Successful-variant count per type. Used to defer saturation
    # until the type has been tested under at least one rewritten
    # prompt - otherwise the very point of prompt_ensemble (injecting
    # variance the seed prompt cannot produce) is wasted.
    variant_passes_ok_per_type: Dict[str, int] = {k: 0 for k in seed_type_keys}

    def _arm_key_for_chunk(chunk, et_key: str):
        """Map a (chunk, error_type) pair to its stop_scope arm key.

        Returns ``None`` when the current ``stop_scope`` cannot be
        cleanly mapped to a single chunk (e.g. column-pair / row-level
        scopes, or column-scope under ``column_mode=all`` where one
        chunk covers many column arms). Callers fall back to the
        existing type-level pruning in those cases.
        """
        if stop_scope == "csv":
            return (("csv", (str(chunk.source),)), et_key)
        if stop_scope == "column" and chunk.is_single_column and chunk.headers:
            return (("column", (str(chunk.source), chunk.headers[0])), et_key)
        if stop_scope == "dataset":
            return (("dataset", ()), et_key)
        return None

    for i in range(num_runs):
        # Run-ID offset: when skip_identity_run is False, run 0 is the
        # un-perturbed baseline (matches what `mode=single` would see).
        # When True, we shift so every run is perturbed.
        effective_run_id = i if not skip_identity_run else i + 1

        # Frame perturbation (row / column shuffle). effective_run_id=0
        # returns the identity view, so this is a no-op when the
        # baseline pass is in use.
        views = perturb_frames(
            source_frames,
            run_id=effective_run_id,
            base_seed=base_seed,
            shuffle_rows=shuffle_rows,
            shuffle_columns=shuffle_columns,
        )
        pass_frames: Dict[str, pd.DataFrame] = {v.source: v.frame for v in views}
        pass_chunks, _ = _prepare_chunks(
            pass_frames,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
            start_row=start_row,
            column_mode=column_mode,
            active_columns=active_columns,
        )

        # Prompt-ensemble perturbation. Only fires on non-identity
        # passes so run 0 (when kept) matches the seed prompt exactly.
        use_variants = prompt_ensemble_on and effective_run_id > 0
        # Per-type policy: skip already-saturated types this pass.
        active_seed_types = [
            t for t in seed_error_types if t.key not in saturated_types
        ]
        if saturated_types:
            # Make the per-type pruning visible: the detector progress
            # bar will shrink AND the prompt-variant generator (when
            # enabled) will skip these types entirely.
            logger.info(
                "  Pruned %d/%d saturated type(s) from pass %d: %s",
                len(saturated_types), len(seed_type_keys), i + 1,
                ", ".join(sorted(saturated_types)),
            )
        if not active_seed_types:
            # Nothing left to detect -- defensive guard; the loop's
            # post-pass check below normally catches this first.
            stop_reason = (
                f"adaptive[stop_scope={stop_scope}]: all "
                f"{len(seed_type_keys)} seed type(s) "
                f"saturated before pass {i + 1}"
            )
            logger.info("Early stop: %s", stop_reason)
            break
        variant_records: List[Dict[str, Any]] = []
        if use_variants:
            assert variant_generator is not None  # narrowed by prompt_ensemble_on
            logger.info(
                "Pass %d: generating LLM prompt variants for %d error type(s) "
                "(effort=%s)",
                i + 1, len(active_seed_types), variant_generator.reasoning_effort,
            )
            pass_taxonomy, pass_error_types, variant_records = (
                variant_generator.generate_for_taxonomy(
                    base_taxonomy=base_taxonomy,
                    seed_types=active_seed_types,
                    pass_index=effective_run_id,
                )
            )
            # Persist variants so reviewers can inspect what the
            # rewriter actually produced for this pass.
            save_variant_records(
                variant_records,
                run_subdir=run_dir / "prompt_variants" / f"run_{i:02d}",
            )
        else:
            pass_taxonomy = base_taxonomy
            pass_error_types = list(active_seed_types)

        # Account for which types were actually probed this pass
        # (whether under seed or variant prompt). Used to drive a
        # statistically sound per-type estimate where types pruned
        # mid-loop have their own effective K.
        for et in active_seed_types:
            runs_probed_per_type[et.key] = runs_probed_per_type.get(et.key, 0) + 1
        if use_variants and variant_records:
            for r in variant_records:
                if r.get("ok"):
                    tkey = r.get("type_key")
                    if isinstance(tkey, str):
                        variant_passes_ok_per_type[tkey] = (
                            variant_passes_ok_per_type.get(tkey, 0) + 1
                        )

        pass_detector = PerTypeCleaningDetector(
            llm=llm,
            taxonomy=pass_taxonomy,
            model=model,
            error_types=pass_error_types,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            concurrency=concurrency,
            digests=digests,
        )

        # Per-arm pruning predicate. Skips (chunk, type) pairs whose
        # arm has already saturated under the configured stop_scope.
        # Returns False (i.e. keep) for any (chunk, type) we cannot
        # map cleanly to a single arm - those rely on the higher-level
        # type-level saturation set above.
        def _skip_pair(chunk, et) -> bool:
            arm_key = _arm_key_for_chunk(chunk, et.key)
            if arm_key is None:
                return False
            return arm_key in saturated_arms

        # Tally per-arm effective K for the (chunk, type) pairs we
        # are about to actually probe this pass. Only meaningful for
        # scopes where ``_arm_key_for_chunk`` returns a non-None key.
        kept_arm_probes: set = set()
        for chunk in pass_chunks:
            for et in pass_error_types:
                arm_key = _arm_key_for_chunk(chunk, et.key)
                if arm_key is None or arm_key in saturated_arms:
                    continue
                kept_arm_probes.add(arm_key)
        for arm_key in kept_arm_probes:
            runs_probed_per_arm[arm_key] = (
                runs_probed_per_arm.get(arm_key, 0) + 1
            )

        frame_desc = views[0].description if views else "empty"
        prompt_desc = "prompt=variant" if use_variants else "prompt=seed"
        logger.info(
            "=== Pass %d/%d [%s | %s] ===",
            i + 1, num_runs, frame_desc, prompt_desc,
        )
        reports_run = pass_detector.run(pass_chunks, skip_predicate=_skip_pair)
        flat_run = flatten_issues(
            reports_run,
            source_frames=pass_frames,
            include_full_row=enable_synthesis,
        )
        all_reports.extend(reports_run)
        all_flat_issues.extend(flat_run)
        detections_per_run.append(flat_run)
        variant_meta: Dict[str, Any] = {
            "enabled": use_variants,
            "count_ok": sum(1 for r in variant_records if r["ok"]),
            "count_fallback": sum(1 for r in variant_records if not r["ok"]),
            "error_type_keys": [r["type_key"] for r in variant_records],
        }
        if use_variants:
            # Slim in-memory copy for the HTML dashboard (skip the raw
            # LLM reply and the full generator user-prompt - they can
            # push the embedded payload past a megabyte on larger runs
            # and are preserved on disk anyway).
            variant_records_per_run.append({
                "run_id": i,
                "pass_index": effective_run_id,
                "records": [
                    {
                        "type_key": r["type_key"],
                        "pass_index": r.get("pass_index"),
                        "ok": r["ok"],
                        "error": r.get("error"),
                        "seed_description": r.get("seed_description", ""),
                        "seed_notes": r.get("seed_notes", ""),
                        "seed_example": r.get("seed_example", ""),
                        "variant_description": r.get("variant_description", ""),
                        "variant_notes": r.get("variant_notes", ""),
                        "variant_example": r.get("variant_example", ""),
                    }
                    for r in variant_records
                ],
            })
        run_info.append(
            {
                "run_id": i,
                "effective_seed": views[0].effective_seed if views else None,
                "sources": [v.source for v in views],
                "descriptions": {v.source: v.description for v in views},
                "detections": len(flat_run),
                "failed_calls": sum(1 for r in reports_run if r.error),
                "prompt_variants": variant_meta,
            }
        )
        per_run_dumps.append(
            {
                "run_id": i,
                "detections": flat_run,
                "meta": {
                    "run_id": i,
                    "effective_seeds": {v.source: v.effective_seed for v in views},
                    "descriptions": {v.source: v.description for v in views},
                    "detections": len(flat_run),
                    "prompt_variants": variant_meta,
                },
            }
        )

        runs_done = i + 1
        cum_species = list(aggregate_runs(detections_per_run).values())
        cum_est = estimate(cum_species, num_runs=runs_done)
        progression.append(
            {
                "run_id": i,
                "runs_so_far": runs_done,
                "detections_this_run": len(flat_run),
                "n_observed": cum_est.n_observed,
                "good_turing_missing_mass": cum_est.good_turing_missing_mass,
                "chao1_estimated_total": cum_est.chao1_estimated_total,
                "chao1_ci_95_low": cum_est.chao1_ci_95_low,
                "chao1_ci_95_high": cum_est.chao1_ci_95_high,
                "fn_rate_point": cum_est.fn_rate_point,
                "fn_rate_ci_95_low": cum_est.fn_rate_ci_95_low,
                "fn_rate_ci_95_high": cum_est.fn_rate_ci_95_high,
            }
        )
        logger.info(
            "  After pass %d: S_obs=%d, GT missing mass=%.2f%%, "
            "Chao1=%.1f [95%% CI %.1f-%.1f]",
            runs_done,
            cum_est.n_observed,
            100.0 * cum_est.good_turing_missing_mass,
            cum_est.chao1_estimated_total,
            cum_est.chao1_ci_95_low,
            cum_est.chao1_ci_95_high,
        )

        if gt_mode == "adaptive":
            # Unified per-arm stop. Build the per-arm table at the
            # configured ``stop_scope``, evaluate the per-arm stop
            # predicate, and treat an error_type as "saturated" iff
            # every applicable arm of that type has reached
            # ``saturated`` / ``exhausted`` (or ``not_applicable``).
            #
            # When stop_scope=="dataset" each error_type has exactly
            # one arm and this collapses to the previous "per_type"
            # behaviour. When stop_scope=="column" arms partition
            # further by (source, column).
            #
            # ``target_missing_mass`` continues to act as the default
            # epsilon so existing configs keep working unchanged; the
            # YAML ``per_arm.defaults.epsilon`` (and CLI overrides)
            # still take precedence.
            stop_arm_table = aggregate_runs_by_scope(
                detections_per_run, scope_kind=stop_scope  # type: ignore[arg-type]
            )
            # Per-arm K: prefer the per-arm probe count (which reflects
            # arm-level pruning when stop_scope is chunk-mappable) and
            # fall back to the per-type probe count for unobserved /
            # unmappable arms.
            num_runs_by_arm = {}
            for arm in stop_arm_table.keys():
                k_arm = runs_probed_per_arm.get(arm)
                if k_arm is None:
                    k_arm = int(runs_probed_per_type.get(arm[1], runs_done))
                num_runs_by_arm[arm] = int(k_arm)
            stop_arm_ests = estimate_by_arm(
                stop_arm_table, num_runs_by_arm=num_runs_by_arm
            )
            # Per-arm status: re-use the existing predicate so behaviour
            # matches what the artifact writer reports later. Two-pass
            # tracking is per-arm, not maintained across iterations
            # here (we recompute from cumulative species each pass).
            arms_by_type: Dict[str, List[str]] = {}
            saturated_arms_by_type: Dict[str, List[str]] = {}
            newly_saturated_arms: List[Dict[str, Any]] = []
            for arm, est in stop_arm_ests.items():
                etype = arm[1]
                arms_by_type.setdefault(etype, []).append(arm[0][0])
                if not is_applicable(arm, whitelist=per_arm_appl):
                    est.status = "not_applicable"
                else:
                    cfg = resolve_arm_config(
                        arm,
                        global_default=per_arm_default,
                        by_error_type=per_arm_by_type,
                    )
                    # Thread the previous pass's (status, streak) so
                    # ``require_two_consecutive`` actually fires across
                    # passes. ``evaluate_stop`` is sticky for terminal
                    # statuses, so once an arm saturates / exhausts /
                    # is marked not_applicable it stays that way.
                    prev_status = arm_prev_status.get(arm, "active")
                    prev_streak = arm_prev_streak.get(arm, 0)
                    est.status, est.saturated_streak = evaluate_stop(
                        est, cfg,
                        previous_status=prev_status,  # type: ignore[arg-type]
                        previous_streak=prev_streak,
                    )
                arm_prev_status[arm] = est.status
                arm_prev_streak[arm] = est.saturated_streak
                if est.status in {"saturated", "exhausted", "not_applicable"}:
                    saturated_arms_by_type.setdefault(etype, []).append(
                        arm[0][0]
                    )
                    # Gate per-arm pruning behind the same min_runs
                    # floor as per-type saturation: otherwise the loop
                    # would prune all (chunk, type) pairs immediately
                    # under a permissive epsilon and spin doing nothing
                    # until min_runs catches up.
                    if runs_done >= min_runs and arm not in saturated_arms:
                        # Newly saturated this pass - record for both
                        # logging and (next-pass) chunk-level pruning.
                        saturated_arms.add(arm)
                        scope_id = arm[0][1]
                        scope_label = (
                            ":".join(str(p) for p in scope_id)
                            if scope_id else "(dataset)"
                        )
                        newly_saturated_arms.append({
                            "arm": arm,
                            "scope_kind": arm[0][0],
                            "scope_label": scope_label,
                            "type_key": etype,
                            "status": est.status,
                            "n_observed": est.n_observed,
                            "good_turing_missing_mass":
                                est.good_turing_missing_mass,
                            "k": num_runs_by_arm.get(arm, runs_done),
                        })
            # Log per-arm transitions so the user can watch arms drop
            # out of the active set in real time. Each line ties a
            # scope_id to the type_key it just saturated for.
            for entry in newly_saturated_arms:
                logger.info(
                    "  [stop_scope=%s] arm SATURATED: %s | type=%s | "
                    "S_obs=%d, GT mass=%.2f%%, status=%s, K=%d",
                    entry["scope_kind"],
                    entry["scope_label"],
                    entry["type_key"],
                    entry["n_observed"],
                    100.0 * entry["good_turing_missing_mass"],
                    entry["status"],
                    entry["k"],
                )
            # Active-arms-only Good-Turing summary. The pooled
            # ``cum_est.good_turing_missing_mass`` logged above
            # treats every species as if its arm would still be
            # probed in the next pass, but saturated arms are frozen
            # - their singletons keep inflating the global numerator
            # while their hits stop accumulating. The honest
            # "probability the next emission belongs to a never-yet-
            # seen species" is ``sum(n1) / sum(N)`` over arms still
            # ``active`` (and thus actually scheduled for the next
            # pass), where ``N = sum_x R_x`` is the textbook GT
            # denominator: total raw detection emissions across the
            # arm with no within-pass de-duplication.
            active_n1 = 0
            active_n_total = 0
            active_n_observed = 0
            active_arm_count = 0
            for arm, est in stop_arm_ests.items():
                if est.status != "active":
                    continue
                active_arm_count += 1
                active_n1 += int(est.frequency_counts.get(1, 0))
                active_n_total += int(est.total_hits)
                active_n_observed += int(est.n_observed)
            active_gt_missing_mass = (
                (active_n1 / active_n_total) if active_n_total > 0 else 0.0
            )
            logger.info(
                "  [stop_scope=%s] active-arm GT missing mass=%.2f%% "
                "(S_obs=%d across %d active arm(s); excludes %d "
                "saturated/exhausted arm(s))",
                stop_scope,
                100.0 * active_gt_missing_mass,
                active_n_observed,
                active_arm_count,
                len(stop_arm_ests) - active_arm_count,
            )
            # Annotate this pass's progression entry so downstream
            # reports / compare-runs tooling can plot the active-arm
            # mass over time. The legacy pooled fields stay untouched
            # for back-compat.
            progression[-1]["active_arm_gt_missing_mass"] = active_gt_missing_mass
            progression[-1]["active_arm_n_observed"] = active_n_observed
            progression[-1]["active_arm_count"] = active_arm_count
            progression[-1]["saturated_arm_count"] = (
                len(stop_arm_ests) - active_arm_count
            )
            # Per-pass per-arm snapshot for offline threshold sweeps.
            # We persist a minimal column-set to keep the JSONL
            # compact across long runs; richer fields are recoverable
            # from ``per_arm_estimates.json`` (final state) if needed.
            for arm, est in stop_arm_ests.items():
                scope_kind, scope_id = arm[0]
                per_pass_per_arm_snapshots.append({
                    "pass_idx": runs_done - 1,
                    "runs_so_far": runs_done,
                    "scope_kind": scope_kind,
                    "scope_id": list(scope_id),
                    "error_type": arm[1],
                    "k_arm": int(num_runs_by_arm.get(arm, runs_done)),
                    "n_observed": int(est.n_observed),
                    "n1": int(est.frequency_counts.get(1, 0)),
                    "n2": int(est.frequency_counts.get(2, 0)),
                    "total_hits": int(est.total_hits),
                    "good_turing_missing_mass":
                        float(est.good_turing_missing_mass),
                    "chao1_estimated_missed":
                        float(est.chao1_estimated_missed),
                    "chao1_estimated_total":
                        float(est.chao1_estimated_total),
                    "chao1_ci_95_low": float(est.chao1_ci_95_low),
                    "chao1_ci_95_high": float(est.chao1_ci_95_high),
                    "fn_rate_point": float(est.fn_rate_point),
                    "status": est.status,
                    "saturated_streak": int(est.saturated_streak),
                })
            if newly_saturated_arms:
                # Estimate next-pass savings so the user can predict
                # how much smaller the upcoming progress bar will be.
                next_skipped_pairs = 0
                for chunk in pass_chunks:
                    for et in seed_error_types:
                        if et.key in saturated_types:
                            continue
                        arm_key = _arm_key_for_chunk(chunk, et.key)
                        if arm_key is not None and arm_key in saturated_arms:
                            next_skipped_pairs += 1
                if next_skipped_pairs > 0:
                    logger.info(
                        "  [stop_scope=%s] arm pruning will skip "
                        "%d (chunk x type) pair(s) on the next pass "
                        "(%d arm(s) saturated cumulatively).",
                        stop_scope,
                        next_skipped_pairs,
                        len(saturated_arms),
                    )
            # An error_type is "saturated" iff every observed arm of
            # that type is non-active. Types that produced zero
            # detections at the chosen scope have no arms in the table
            # and remain active until either they appear or num_runs
            # runs out - which is the conservative behaviour.
            newly_saturated: List[str] = []
            if runs_done >= min_runs:
                for tkey in seed_type_keys:
                    if tkey in saturated_types:
                        continue
                    arms_for_type = arms_by_type.get(tkey, [])
                    if not arms_for_type:
                        # Zero observations after ``min_runs`` passes:
                        # we treat the type as saturated (assume the
                        # error simply isn't present in this corpus)
                        # rather than letting it pin the loop open
                        # until ``num_runs``. ``min_runs`` is the
                        # caller-controlled floor for "how many empty
                        # passes is enough evidence of absence". Data
                        # cleaning rarely surfaces every error type
                        # in every dataset, so this is the right
                        # default.
                        if (
                            prompt_ensemble_on
                            and variant_passes_ok_per_type.get(tkey, 0) < 1
                        ):
                            # Still defer: a rewritten prompt may yet
                            # surface a detection the seed couldn't.
                            continue
                        saturated_types.add(tkey)
                        newly_saturated.append(tkey)
                        continue
                    if (
                        prompt_ensemble_on
                        and variant_passes_ok_per_type.get(tkey, 0) < 1
                    ):
                        # Defer saturation until the type has been
                        # tested under at least one rewritten prompt.
                        continue
                    sat_arms = saturated_arms_by_type.get(tkey, [])
                    if len(sat_arms) >= len(arms_for_type):
                        saturated_types.add(tkey)
                        newly_saturated.append(tkey)
            if newly_saturated:
                logger.info(
                    "  [stop_scope=%s] saturated this pass: %s "
                    "(total saturated: %d/%d)",
                    stop_scope,
                    ", ".join(newly_saturated),
                    len(saturated_types), len(seed_type_keys),
                )
            saturation_history.append({
                "run_id": i,
                "runs_so_far": runs_done,
                "newly_saturated": newly_saturated,
                "saturated_total": sorted(saturated_types),
                "active_types_for_next_pass": [
                    k for k in seed_type_keys if k not in saturated_types
                ],
                # Per-arm detail. ``newly_saturated_arms`` lists every
                # (scope_id, type) pair that crossed the stop predicate
                # this pass; ``saturated_arms_total`` is the cumulative
                # count for at-a-glance progress tracking.
                "newly_saturated_arms": [
                    {
                        "scope_kind": e["scope_kind"],
                        "scope_label": e["scope_label"],
                        "type_key": e["type_key"],
                        "status": e["status"],
                        "n_observed": e["n_observed"],
                        "good_turing_missing_mass":
                            e["good_turing_missing_mass"],
                        "k": e["k"],
                    }
                    for e in newly_saturated_arms
                ],
                "saturated_arms_total": len(saturated_arms),
            })
            if (
                runs_done >= min_runs
                and saturated_types == set(seed_type_keys)
            ):
                stop_reason = (
                    f"adaptive[stop_scope={stop_scope}]: all "
                    f"{len(seed_type_keys)} seed type(s) saturated "
                    f"(per-arm GT missing mass <= {target_missing_mass:.4f}) "
                    f"after {runs_done} run(s) (min_runs={min_runs})"
                )
                logger.info("Early stop: %s", stop_reason)
                break

    if stop_reason is None:
        if gt_mode == "adaptive":
            stop_reason = (
                f"adaptive[stop_scope={stop_scope}]: hit ceiling "
                f"num_runs={num_runs} with {len(saturated_types)}/"
                f"{len(seed_type_keys)} seed type(s) saturated at target "
                f"missing_mass <= {target_missing_mass:.4f}"
            )
        else:
            stop_reason = f"fixed: completed {len(detections_per_run)} run(s)"
    num_runs_completed = len(detections_per_run)
    logger.info("Loop terminated: %s", stop_reason)

    species_records = list(aggregate_runs(detections_per_run).values())
    overall = estimate(species_records, num_runs=num_runs_completed)
    per_type = estimate_by_group(
        species_records,
        num_runs=num_runs_completed,
        key="error_type",
        num_runs_by_key=runs_probed_per_type,
    )
    per_source_est = estimate_by_group(
        species_records, num_runs=num_runs_completed, key="source"
    )

    estimates_payload: Dict[str, Any] = {
        "gt_mode": gt_mode,
        "num_runs_completed": num_runs_completed,
        "num_runs_max": num_runs,
        "min_runs": min_runs,
        "target_missing_mass": target_missing_mass,
        "stop_scope": stop_scope,
        # Back-compat alias preserved on the payload so old report
        # templates / dashboards keep rendering. New consumers should
        # prefer ``stop_scope``.
        "target_policy": (
            "per_type" if stop_scope == "dataset" else "stop_scope"
        ),
        "saturated_types": sorted(saturated_types),
        "saturation_history": saturation_history,
        "runs_probed_per_type": dict(runs_probed_per_type),
        "variant_passes_ok_per_type": dict(variant_passes_ok_per_type),
        "stop_reason": stop_reason,
        "base_seed": base_seed,
        "shuffle_rows": shuffle_rows,
        "shuffle_columns": shuffle_columns,
        "include_identity_run": not skip_identity_run,
        "prompt_ensemble": prompt_ensemble_on,
        "variant_generation_effort": (
            variant_generator.reasoning_effort
            if variant_generator is not None else None
        ),
        "overall": overall.to_dict(),
        "per_error_type": {k: v.to_dict() for k, v in per_type.items()},
        "per_source": {k: v.to_dict() for k, v in per_source_est.items()},
        "progression": progression,
        "runs": run_info,
    }
    markdown_report = render_markdown_report(
        overall=overall,
        per_type=per_type,
        per_source=per_source_est,
        run_info=run_info,
        num_runs=num_runs_completed,
        mode=gt_mode,
        num_runs_max=num_runs,
        min_runs=min_runs,
        target_missing_mass=target_missing_mass,
        stop_reason=stop_reason,
        progression=progression,
    )

    gt_artifacts: Dict[str, Any] = {
        "num_runs_completed": num_runs_completed,
        "stop_reason": stop_reason,
        "species_records": species_records,
        "progression": progression,
        "estimates_payload": estimates_payload,
        "markdown_report": markdown_report,
        "overall_estimate": overall.to_dict(),
        "per_run_dumps": per_run_dumps,
        "variant_records_per_run": variant_records_per_run,
        # Per-pass per-arm snapshot stream for offline hyperparameter
        # sweeps via ``experiments/analyze_stop_thresholds.py``.
        "per_pass_per_arm_snapshots": per_pass_per_arm_snapshots,
        # Re-exposed for the HTML dashboard so it doesn't have to re-open
        # estimates.json. These are the SAME dicts stored in estimates_payload.
        "per_error_type": {k: v.to_dict() for k, v in per_type.items()},
        "per_source": {k: v.to_dict() for k, v in per_source_est.items()},
        "run_info": run_info,
    }

    # --- Additive: scope-aware per-arm estimator -----------------------
    #
    # Computed unconditionally as a sibling artifact. Does not affect
    # the loop's stopping behaviour (the existing global rule still
    # decides when to halt) - this is purely a richer report so users
    # can see per-(scope, error_type) saturation. See
    # ``src/adaptive/scoped_gt.py`` for the underlying math.
    try:
        # ``per_arm_default`` / ``per_arm_by_type`` / ``per_arm_appl``
        # were already unpacked at the top of the loop. Re-fetch
        # defensively so this block stays callable even if a future
        # refactor moves things around.
        per_arm_default, per_arm_by_type, per_arm_appl = settings.get(
            "per_arm",
            (
                ArmConfig.default().merge_into(ArmConfig.default()),
                {},
                None,
            ),
        )
        # Aggregate K runs into the per-arm species table at the SAME
        # scope_kind the loop used for stopping. Reporting at a
        # different scope than the one that drove pruning would mean
        # the artifact's status flags wouldn't line up with the
        # in-loop ``arm_prev_status`` we carry over below.
        report_scope_kind = stop_scope
        arm_table = aggregate_runs_by_scope(
            detections_per_run, scope_kind=report_scope_kind  # type: ignore[arg-type]
        )
        # Per-arm K: prefer the explicit per-arm probe count tracked
        # by the loop (which reflects arm-level pruning when the
        # stop_scope is chunk-mappable) and fall back to the per-type
        # probe count for unobserved / unmappable arms. Mirrors the
        # in-loop computation so the status carry-over below is
        # consistent.
        num_runs_by_arm = {}
        for arm in arm_table.keys():
            k_arm = runs_probed_per_arm.get(arm)
            if k_arm is None:
                k_arm = int(
                    runs_probed_per_type.get(arm[1], num_runs_completed)
                )
            num_runs_by_arm[arm] = int(k_arm)
        scoped_estimates = estimate_by_arm(
            arm_table, num_runs_by_arm=num_runs_by_arm
        )
        # Stop predicate -> status field on each ArmEstimate. For arms
        # the loop already evaluated, ``arm_prev_status`` /
        # ``arm_prev_streak`` already hold the AFTER-last-pass state,
        # so we MUST NOT call ``evaluate_stop`` again on the same
        # cumulative data - that would increment the streak a second
        # time for the same pass and could spuriously promote an
        # ``active`` arm with streak=1 to ``saturated`` in the
        # artifact even though no new pass occurred. For arms not
        # tracked in-loop (e.g. ``gt_mode == 'fixed'``, or arms that
        # only surfaced in the final aggregator) we run a fresh
        # ``evaluate_stop`` from default ``("active", 0)``.
        for arm, est in scoped_estimates.items():
            if not is_applicable(arm, whitelist=per_arm_appl):
                est.status = "not_applicable"
                continue
            if arm in arm_prev_status:
                # Trust the in-loop decision verbatim - it already
                # reflects the streak / two-consecutive bookkeeping
                # the loop maintained pass-by-pass.
                est.status = arm_prev_status[arm]  # type: ignore[assignment]
                est.saturated_streak = arm_prev_streak.get(arm, 0)
                continue
            cfg = resolve_arm_config(
                arm,
                global_default=per_arm_default,
                by_error_type=per_arm_by_type,
            )
            est.status, est.saturated_streak = evaluate_stop(est, cfg)
        per_arm_payload: Optional[Dict[str, Any]] = {
            "scope_kind": report_scope_kind,
            "num_runs_completed": num_runs_completed,
            "arms": [est.to_dict() for est in scoped_estimates.values()],
        }
        per_arm_md: Optional[str] = render_per_arm_markdown(scoped_estimates)
    except Exception:  # noqa: BLE001
        # Never let the additive artifact break the run. Log and move on.
        logger.exception(
            "scope-aware per-arm estimator failed; skipping artifact"
        )
        per_arm_payload = None
        per_arm_md = None
    gt_artifacts["per_arm_payload"] = per_arm_payload
    gt_artifacts["per_arm_markdown"] = per_arm_md

    return all_reports, all_flat_issues, gt_artifacts


def _write_run_status(
    run_dir: "Optional[Path]",
    status: str,
    *,
    detail: Optional[str] = None,
) -> None:
    """Write a tiny ``RUN_STATUS.json`` so a partial run is visible.

    ``status`` is one of ``in_progress``, ``completed``, ``failed``.
    Best-effort: never raises - status reporting must not mask the
    real exit path.
    """
    if run_dir is None:
        return
    try:
        from datetime import datetime, timezone
        payload = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if detail:
            payload["detail"] = detail[:500]
        (run_dir / "RUN_STATUS.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.debug("could not write RUN_STATUS.json", exc_info=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, str(config.get("logging", {}).get("level", "INFO")).upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Silence noisy third-party loggers: httpx/httpcore log one INFO
    # line per HTTP request, and with K passes x N error-types x M
    # chunks that floods the console. Warnings still come through.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "azure"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data_cfg = config.get("data", {})
    llm_cfg = config.get("llm", {})
    prompts_cfg = config.get("prompts", {})
    output_cfg = config.get("output", {})

    # CSV resolution supports three modes:
    #   1. --csv FILE                    -> single file
    #   2. --csv DIR  or --csv-dir DIR   -> every CSV in DIR matching --csv-glob
    #   3. (no flags)                    -> data.csv_path (single) or data.csv_dir
    # Directory discovery is RECURSIVE: CSVs nested in subdirectories
    # (e.g. data/curated_pbi/<spreadsheet-id>/<table>.csv) are included.
    # Pass a glob containing a path component (e.g. "*/*.csv") to
    # restrict to a single depth.
    csv_glob = (
        args.csv_glob or str(data_cfg.get("csv_glob", "*.csv")) or "*.csv"
    )
    csv_dir_override = args.csv_dir
    csv_path: Optional[Path] = None
    csv_paths: List[Path]

    def _discover_csvs(root: Path, pattern: str) -> List[Path]:
        # Path.rglob treats "*.csv" as "match anywhere in the tree";
        # patterns that include explicit separators or "**" use glob
        # so caller-supplied depth anchors are respected.
        if "/" in pattern or "**" in pattern:
            matches = root.glob(pattern)
        else:
            matches = root.rglob(pattern)
        return sorted(p for p in matches if p.is_file())

    if csv_dir_override is not None:
        csv_dir = _resolve(csv_dir_override, PROJECT_ROOT)
        csv_paths = _discover_csvs(csv_dir, csv_glob)
        if not csv_paths:
            raise FileNotFoundError(
                f"No CSVs matched {csv_glob!r} under {csv_dir}"
            )
        csv_path = csv_dir
    elif args.csv is not None:
        candidate = _resolve(args.csv, PROJECT_ROOT)
        if not candidate.exists():
            raise FileNotFoundError(
                f"--csv path does not exist: {candidate}"
            )
        if candidate.is_dir():
            csv_paths = _discover_csvs(candidate, csv_glob)
            if not csv_paths:
                raise FileNotFoundError(
                    f"No CSVs matched {csv_glob!r} under {candidate}"
                )
            csv_path = candidate
        else:
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"--csv is not a regular file: {candidate}"
                )
            csv_paths = [candidate]
            csv_path = candidate
    elif data_cfg.get("csv_dir"):
        csv_dir = _resolve(data_cfg["csv_dir"], PROJECT_ROOT)
        if not csv_dir.is_dir():
            raise FileNotFoundError(
                f"data.csv_dir does not exist or is not a directory: {csv_dir}"
            )
        csv_paths = _discover_csvs(csv_dir, csv_glob)
        if not csv_paths:
            raise FileNotFoundError(
                f"No CSVs matched {csv_glob!r} under {csv_dir}"
            )
        csv_path = csv_dir
    else:
        csv_path = _resolve(data_cfg["csv_path"], PROJECT_ROOT)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"data.csv_path does not exist: {csv_path}"
            )
        csv_paths = [csv_path]

    output_dir = _resolve(
        args.output_dir or output_cfg.get("directory", "outputs"),
        PROJECT_ROOT,
    )
    error_types_file = _resolve(
        args.error_types_file
        or prompts_cfg.get("error_types_file", "prompts/error_types.yaml"),
        PROJECT_ROOT,
    )
    synthesis_prompt_file = _resolve(
        args.synthesis_prompt_file
        or prompts_cfg.get(
            "synthesis_prompt_file", "prompts/synthesise_cluster.yaml"
        ),
        PROJECT_ROOT,
    )

    model = args.model or llm_cfg.get("model", "gpt-5.2")
    chunk_size = args.chunk_size or int(data_cfg.get("chunk_size", 15))
    start_row = int(data_cfg.get("start_row", 0))

    column_mode = (
        args.column_mode
        or str(data_cfg.get("column_mode", COLUMN_MODE_ALL)).strip()
    )
    if column_mode not in VALID_DISPATCHER_COLUMN_MODES:
        raise ValueError(
            f"column_mode must be one of {sorted(VALID_DISPATCHER_COLUMN_MODES)}, "
            f"got {column_mode!r}"
        )

    cli_columns = _parse_csv_list(args.columns)
    cfg_columns = data_cfg.get("columns")
    if cli_columns is not None:
        active_columns: Optional[List[str]] = cli_columns
    elif cfg_columns:
        active_columns = [str(c).strip() for c in cfg_columns if str(c).strip()]
    else:
        active_columns = None

    if args.max_chunks is not None:
        max_chunks = None if args.max_chunks < 0 else args.max_chunks
    else:
        raw = data_cfg.get("max_chunks", None)
        max_chunks = None if raw in (None, "null") else int(raw)

    reasoning_effort = args.reasoning_effort or llm_cfg.get("reasoning_effort", "medium")
    reasoning_summary = llm_cfg.get("reasoning_summary", "detailed")
    synthesis_effort = args.synthesis_effort or llm_cfg.get(
        "synthesis_effort", reasoning_effort
    )

    # Error-type subset resolution: CLI overrides config; empty/None = all.
    cli_types = _parse_csv_list(args.error_types)
    cfg_types = prompts_cfg.get("error_types")
    if cli_types is not None:
        selected_type_keys: Optional[List[str]] = cli_types
    elif cfg_types:
        selected_type_keys = [str(k).strip() for k in cfg_types if str(k).strip()]
    else:
        selected_type_keys = None

    concurrency = args.concurrency if args.concurrency else llm_cfg.get("concurrency")
    if concurrency is not None:
        concurrency = int(concurrency)

    synthesis_concurrency = (
        args.synthesis_concurrency
        if args.synthesis_concurrency
        else llm_cfg.get("synthesis_concurrency")
    )
    if synthesis_concurrency is not None:
        synthesis_concurrency = int(synthesis_concurrency)

    enable_synthesis = _resolve_synthesis_enabled(args, llm_cfg)

    experiment_cfg = config.get("experiment", {}) or {}
    mode_raw = args.mode or str(experiment_cfg.get("mode", "single"))
    mode = mode_raw.lower()
    if mode not in {"single", "adaptive"}:
        raise ValueError(
            f"experiment.mode must be 'single' or 'adaptive', got {mode_raw!r}"
        )
    adaptive_settings = (
        _resolve_adaptive_settings(args, config) if mode == "adaptive" else None
    )

    taxonomy = load_taxonomy(error_types_file)

    if args.list_error_types:
        for et in taxonomy.types:
            print(f"  {et.key:40s}  {et.summary_line()}")
        return 0

    if getattr(args, "validate_config", False):
        # All upstream validation has already run by this point:
        # config schema, mode, adaptive settings, taxonomy.
        # Emit a one-line OK summary and exit successfully.
        n_selected = (
            len(selected_type_keys) if selected_type_keys is not None else "all"
        )
        print(
            f"validate-config: OK  config={args.config}  mode={mode}  "
            f"types={len(taxonomy.types)}  selected={n_selected}"
        )
        return 0

    error_types = taxonomy.select(selected_type_keys)

    # Optional scope filter (CLI > config). A scope is the prefix of
    # ErrorType.category up to the first '/', normalised to lowercase
    # with internal whitespace collapsed (e.g. "Single-Column / Typo"
    # -> "single-column"). The pseudo-scope "all" disables the filter.
    active_scopes = _resolve_scopes(args.scopes, prompts_cfg.get("scopes"))
    if active_scopes is not None:
        before_keys = [et.key for et in error_types]
        error_types = [
            et for et in error_types if _scope_of(et.category) in active_scopes
        ]
        dropped = [k for k in before_keys if k not in {et.key for et in error_types}]
        if not error_types:
            raise ValueError(
                f"scopes={sorted(active_scopes)} filter dropped every selected "
                "error type. Either widen the scope list or include matching "
                f"types in --error-types / prompts.error_types. Dropped: {dropped}"
            )
        if dropped:
            logger.info(
                "scopes=%s: kept %d type(s), dropped %d non-matching type(s): %s",
                sorted(active_scopes), len(error_types), len(dropped), dropped,
            )

    # ---- Row-duplication detector ---------------------------------
    # When ``row_duplication`` is among the selected types, swap the
    # per-chunk LLM scan for a whole-table key-collision detector +
    # per-group LLM adjudicator. The deterministic stage finds duplicate
    # pairs that straddle chunk boundaries; the adjudicator then
    # confirms each candidate group with a single, focused LLM call.
    row_dup_cfg = dict(config.get("row_duplication") or {})
    if args.row_dup_detection_mode is not None:
        row_dup_cfg["detection_mode"] = args.row_dup_detection_mode
    if args.row_dup_normalize_keys is not None:
        row_dup_cfg["normalize_keys"] = args.row_dup_normalize_keys
    if args.row_dup_adjudicate is not None:
        row_dup_cfg["adjudicate_groups"] = args.row_dup_adjudicate
    if args.row_dup_max_groups_per_source is not None:
        row_dup_cfg["max_groups_per_source"] = args.row_dup_max_groups_per_source
    if args.row_dup_canonicalize_keys is not None:
        row_dup_cfg["canonicalize_keys"] = args.row_dup_canonicalize_keys
    if args.row_dup_canonicalize_max_cardinality is not None:
        row_dup_cfg["canonicalize_max_cardinality"] = (
            args.row_dup_canonicalize_max_cardinality
        )
    if args.row_dup_blocking_enabled is not None:
        row_dup_cfg["blocking_enabled"] = args.row_dup_blocking_enabled
    if args.row_dup_minhash_enabled is not None:
        row_dup_cfg["minhash_enabled"] = args.row_dup_minhash_enabled
    if args.row_dup_minhash_threshold is not None:
        row_dup_cfg["minhash_threshold"] = args.row_dup_minhash_threshold
    if args.row_dup_embedding_enabled is not None:
        row_dup_cfg["embedding_enabled"] = args.row_dup_embedding_enabled
    if args.row_dup_embedding_threshold is not None:
        row_dup_cfg["embedding_threshold"] = args.row_dup_embedding_threshold
    if args.row_dup_embedding_model is not None:
        row_dup_cfg["embedding_model"] = args.row_dup_embedding_model
    row_dup_settings = AdjudicatorSettings.from_config(row_dup_cfg)
    # Resolve the per-arm ArmConfig once, regardless of mode. The
    # row-duplication detector's adaptive inner loops (cluster
    # adjudication, key canonicalisation) share this same config
    # surface with the experiment-mode Good-Turing per-arm loop, so
    # the operator only configures FN-control behaviour in one place.
    row_dup_arm_config: ArmConfig = _resolve_per_arm_config(config, args)[0]
    wants_row_dup = any(et.key == "row_duplication" for et in error_types)
    use_row_dup_adjudicator = (
        wants_row_dup
        and row_dup_settings.detection_mode
        in ("collision_adjudication", "whole_table")
    )
    if use_row_dup_adjudicator:
        # Multi-row detection needs the LLM to see every column on
        # each row. The adjudicator builds its own per-table prompts
        # outside the chunked scan, so it works regardless of how the
        # chunked scan is dispatched - 'auto' is fine here too. We
        # only nudge the chunked scan to 'all' when the user pinned
        # 'single' (which makes no sense for the remaining types if
        # any of them are cross-column).
        if column_mode == COLUMN_MODE_SINGLE:
            logger.warning(
                "row_duplication is selected; switching column_mode "
                "from 'single' to 'auto' so cross-column types still "
                "work. Set data.column_mode=auto explicitly to "
                "silence this warning."
            )
            column_mode = COLUMN_MODE_AUTO
        # Hide row_duplication from the per-chunk detector so we don't
        # double-scan; the adjudicator emits its findings separately.
        chunk_error_types = [et for et in error_types if et.key != "row_duplication"]
        # Adaptive mode is meaningless when only row_duplication is
        # selected (the adjudicator runs as a single deterministic
        # pass over each table). Coerce the run to single mode and
        # warn so the operator knows the GT settings are inert.
        if not chunk_error_types and mode == "adaptive":
            logger.warning(
                "Only row_duplication is selected; switching mode "
                "from 'adaptive' to 'single' since the adjudicator "
                "is a single-pass detector. Adaptive settings are "
                "ignored for this run."
            )
            mode = "single"
            adaptive_settings = None
        logger.info(
            "row_duplication: %s enabled "
            "(normalize_keys=%s, adjudicate_groups=%s, "
            "max_groups_per_source=%d). Chunk-based scan now runs %d "
            "type(s) instead of %d.",
            (
                "whole-table LLM pass"
                if row_dup_settings.detection_mode == "whole_table"
                else "collision-adjudicator"
            ),
            row_dup_settings.normalize_keys,
            row_dup_settings.adjudicate_groups,
            row_dup_settings.max_groups_per_source,
            len(chunk_error_types),
            len(error_types),
        )
    else:
        chunk_error_types = list(error_types)

    logger.info("Project root: %s", PROJECT_ROOT)
    if len(csv_paths) == 1:
        logger.info("CSV: %s", csv_paths[0])
    else:
        logger.info(
            "CSVs: %d files (root=%s, glob=%s)",
            len(csv_paths),
            csv_path,
            csv_glob,
        )
    logger.info(
        "Model=%s chunk_size=%d max_chunks=%s column_mode=%s "
        "active_columns=%s error_types=%d concurrency=%s",
        model,
        chunk_size,
        max_chunks,
        column_mode,
        len(active_columns) if active_columns else "all",
        len(error_types),
        concurrency,
    )
    logger.info(
        "Phase B synthesis: %s (concurrency=%s, effort=%s)",
        "ON" if enable_synthesis else "OFF",
        synthesis_concurrency,
        synthesis_effort,
    )

    # Load every CSV up front.
    from src.core.text_normalize import NormalizeOptions, normalize_frame
    norm_opts = NormalizeOptions(
        enabled=bool(args.normalize_input),
        fold_punctuation=bool(args.normalize_fold_punctuation),
        lowercase=bool(args.normalize_lowercase),
    )
    if norm_opts.enabled:
        logger.info("Input normalization: ON (%s)", dict(norm_opts.describe()))
    source_frames: Dict[str, pd.DataFrame] = {}
    for path in csv_paths:
        df_i = load_csv(path)
        df_i = normalize_frame(df_i, norm_opts)
        # Recursive directory discovery can surface CSVs with the same
        # basename in different subdirectories (e.g. two spreadsheets
        # that both have a Table.csv). Use the path relative to the
        # discovery root in that case so every source key stays unique.
        name = path.name
        if name in source_frames:
            try:
                rel = path.relative_to(csv_path) if csv_path and csv_path.is_dir() else path
            except ValueError:
                rel = path
            name = str(rel).replace("\\", "/")
        source_frames[name] = df_i
        # Per-file load chatter is noisy for 20+ CSV runs; demoted to
        # DEBUG so default runs only show the single aggregate summary
        # ("Prepared N chunk(s) across M file(s)") below.
        logger.debug(
            "Loaded %s: %d rows x %d cols",
            path.name,
            len(df_i),
            len(df_i.columns),
        )

    # Whole-table digests (TABLE OVERVIEW block prepended to every
    # per-type prompt). Built once per CSV; the cost is O(columns)
    # tokens per call, so this stays affordable on tall tables. Off
    # via `--no-table-digest` or `data.include_table_digest: false`.
    #
    # In ``column_mode=single`` each prompt only sees one column, so
    # the whole-table overview is largely wasted tokens for the
    # single-column / cell-local error types this mode targets. We
    # therefore default the digest OFF in single-column mode unless
    # the user has explicitly opted in via ``data.include_table_digest:
    # true`` in the YAML config. ``--no-table-digest`` still forces it
    # off in either mode.
    cfg_digest = data_cfg.get("include_table_digest", None)
    if args.no_table_digest:
        include_digest = False
    elif cfg_digest is not None:
        include_digest = bool(cfg_digest)
    else:
        # Default: ON unless we are in pure single-column mode (where
        # per-cell prompts gain little from a whole-table overview).
        # In 'auto' mode the all-column group benefits from the digest
        # too, so we keep it on.
        include_digest = column_mode != COLUMN_MODE_SINGLE
    digests: Dict[str, TableDigest] = {}
    if include_digest:
        for name, df_i in source_frames.items():
            try:
                digests[name] = build_table_digest(df_i, source=name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Table digest build failed for %s: %s; "
                    "continuing without digest for this source",
                    name, exc,
                )
        logger.info(
            "Built table digests for %d/%d source(s) [include_table_digest=on]",
            len(digests), len(source_frames),
        )
    else:
        if column_mode == COLUMN_MODE_SINGLE and cfg_digest is None and not args.no_table_digest:
            logger.info(
                "Table digest disabled (column_mode=single default; "
                "set data.include_table_digest=true to force on)"
            )
        else:
            logger.info("Table digest disabled (include_table_digest=off)")

    # ---- Chunk preparation ----------------------------------------
    # In ``auto`` mode we split the selected error types by scope and
    # build chunks per scope: single-column-scope types get
    # ``column_mode=single`` chunks (one cell-shaped prompt at a time),
    # everything else gets ``column_mode=all`` chunks (whole row
    # visible). For non-auto modes there is exactly one group.
    chunk_groups_input = _split_error_types_by_scope(
        column_mode, chunk_error_types
    )
    chunk_groups: List[Tuple[str, List[CsvChunk], List[Any], Dict[str, int]]] = []
    for group_mode, group_types in chunk_groups_input:
        group_chunks, group_per_source = _prepare_chunks(
            source_frames,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
            start_row=start_row,
            column_mode=group_mode,
            active_columns=active_columns,
        )
        chunk_groups.append(
            (group_mode, group_chunks, group_types, group_per_source)
        )

    chunks: List[CsvChunk] = [c for (_, gc, _, _) in chunk_groups for c in gc]
    # Union of per-source counts for the log line. With auto mode a
    # source can appear under both 'single' and 'all'; we sum so the
    # log reflects total chunk volume.
    per_source_chunks: Dict[str, int] = {}
    for _, _, _, gps in chunk_groups:
        for k, v in gps.items():
            per_source_chunks[k] = per_source_chunks.get(k, 0) + v
    expected_calls_per_pass = sum(
        len(gc) * len(gt) for (_, gc, gt, _) in chunk_groups
    )
    if column_mode == COLUMN_MODE_AUTO:
        breakdown = ", ".join(
            f"{gm}:{len(gc)}\u00d7{len(gt)}"
            for (gm, gc, gt, _) in chunk_groups
        ) or "(none)"
        logger.info(
            "Prepared %d chunk(s) across %d file(s) [column_mode=auto: %s] "
            "-> %d detection LLM call(s) per pass",
            len(chunks),
            len(per_source_chunks),
            breakdown,
            expected_calls_per_pass,
        )
    else:
        logger.info(
            "Prepared %d chunk(s) across %d file(s) [column_mode=%s] -> "
            "%d detection LLM call(s) per pass (chunks * error_types)",
            len(chunks),
            len(per_source_chunks),
            column_mode,
            expected_calls_per_pass,
        )
    if mode == "adaptive":
        gt_mode_label = adaptive_settings["gt_mode"]
        max_passes = adaptive_settings["num_runs"]
        logger.info(
            "Adaptive mode: gt_mode=%s, max passes=%d, min=%d, "
            "target missing mass=%.4f -> up to %d total detection call(s)",
            gt_mode_label,
            max_passes,
            adaptive_settings["min_runs"],
            adaptive_settings["target_missing_mass"],
            expected_calls_per_pass * max_passes,
        )

    if args.dry_run:
        if not chunk_groups or all(not gc for (_, gc, _, _) in chunk_groups):
            print("(no chunks to render)")
            return 0
        # Pick the first group with at least one chunk and at least
        # one error type so the rendered prompt is representative.
        first_chunk = None
        first_type = None
        for _, gc, gt, _ in chunk_groups:
            if gc and gt:
                first_chunk = gc[0]
                first_type = gt[0]
                break
        if first_chunk is None or first_type is None:
            print("(no chunks to render)")
            return 0
        print(f"# error_type: {first_type.key}  ({first_type.summary_line()})")
        print(f"# source: {first_chunk.source}")
        print(f"# system: {taxonomy.system_prompt}")
        print("# --- user prompt ---")
        print(taxonomy.render_user_prompt(first_type, first_chunk))
        return 0

    # Run directory is created BEFORE detection so adaptive mode can
    # persist per-pass artifacts (prompt variants, per-run detections)
    # while the loop is in flight.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = getattr(args, "run_name", None)
    if run_name:
        run_id = str(run_name)
    else:
        run_id = f"{timestamp}_{model}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_status(run_dir, "in_progress")

    llm = LLMClient()

    # ---- Detection: single pass OR adaptive multi-pass --------------
    gt_artifacts: Optional[Dict[str, Any]] = None

    if mode == "single":
        if chunk_error_types:
            reports = []
            representative_concurrency = concurrency or 1
            # In ``auto`` mode we run the detector once per scope
            # group: single-column-scope types against single-column
            # chunks, multi-* types against all-column chunks. Other
            # modes have a single group so this collapses to one
            # detector invocation, identical to the prior behaviour.
            for group_mode, group_chunks, group_types, _ in chunk_groups:
                if not group_chunks or not group_types:
                    continue
                detector = PerTypeCleaningDetector(
                    llm=llm,
                    taxonomy=taxonomy,
                    model=model,
                    error_types=group_types,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                    concurrency=concurrency,
                    digests=digests,
                )
                reports.extend(detector.run(group_chunks))
                # Track the most recent detector's concurrency for
                # config-snapshot purposes; in single-group runs this
                # is the same value as before, in auto-mode it picks
                # whichever group's detector ran last.
                representative_concurrency = detector.concurrency
        else:
            # row_duplication is the only selected type; skip the
            # chunk-based detector entirely.
            reports = []
            representative_concurrency = concurrency or 1
        if use_row_dup_adjudicator:
            row_dup_detector = RowDuplicationDetector(
                llm=llm,
                taxonomy=taxonomy,
                model=model,
                settings=row_dup_settings,
                digests=digests,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                concurrency=concurrency,
                arm_config=row_dup_arm_config,
            )
            reports.extend(row_dup_detector.run(source_frames))
        flat_issues = flatten_issues(
            reports,
            source_frames=source_frames,
            include_full_row=enable_synthesis,
        )
        logger.info("Flattened into %d raw detections", len(flat_issues))
        # Used by the config_snapshot below; single mode keeps a
        # representative detector instance around.
    else:  # mode == "adaptive"
        # Build the prompt-variant generator if the user left
        # prompt_ensemble on. When off, the adaptive loop falls back
        # to frame-only perturbation.
        variant_generator: Optional[PromptVariantGenerator] = None
        if adaptive_settings["prompt_ensemble"]:
            variant_template_path = _resolve(
                adaptive_settings["prompt_variant_template"], PROJECT_ROOT
            )
            variant_generator = PromptVariantGenerator(
                llm=llm,
                model=model,
                template_path=variant_template_path,
                reasoning_effort=adaptive_settings["variant_generation_effort"],
                reasoning_summary=reasoning_summary,
                concurrency=concurrency,
            )
            logger.info(
                "Prompt-ensemble ON: rewriter=%s effort=%s (generates %d "
                "variant(s) per non-identity pass)",
                variant_template_path,
                adaptive_settings["variant_generation_effort"],
                len(error_types),
            )
        else:
            logger.info(
                "Prompt-ensemble OFF: relying on frame-only perturbation "
                "(shuffle_rows=%s, shuffle_columns=%s)",
                adaptive_settings["shuffle_rows"],
                adaptive_settings["shuffle_columns"],
            )

        # In ``column_mode=auto`` the adaptive loop runs once per
        # non-empty scope group (single-column types, then everything
        # else) and the per-group ``gt_artifacts`` are merged at the
        # end. For non-auto modes there is exactly one group so this
        # collapses to the historical single-call behaviour.
        adaptive_groups: List[Tuple[str, List[Any]]] = (
            _split_error_types_by_scope(column_mode, chunk_error_types)
        )
        # Filter out empty groups defensively (already done by the
        # splitter, but keep the loop body simple).
        adaptive_groups = [(gm, gt) for (gm, gt) in adaptive_groups if gt]
        is_multi_group = len(adaptive_groups) > 1
        if is_multi_group:
            logger.info(
                "Adaptive auto-mode: running %d scope group(s) sequentially: %s",
                len(adaptive_groups),
                ", ".join(
                    f"{gm}({len(gt)})" for (gm, gt) in adaptive_groups
                ),
            )

        reports = []
        flat_issues = []
        per_group_gt: List[Tuple[str, Dict[str, Any]]] = []
        for group_mode, group_types in adaptive_groups:
            # Each group writes its own prompt_variants/ subtree to
            # avoid clobbering when multiple groups run.
            group_run_dir = (
                run_dir / f"auto_{group_mode}" if is_multi_group else run_dir
            )
            if is_multi_group:
                group_run_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Adaptive group: column_mode=%s types=%s",
                    group_mode,
                    [et.key for et in group_types],
                )
            g_reports, g_flat, g_artifacts = _run_adaptive_loop(
                llm=llm,
                model=model,
                base_taxonomy=taxonomy,
                seed_error_types=group_types,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                concurrency=concurrency,
                source_frames=source_frames,
                chunk_size=chunk_size,
                max_chunks=max_chunks,
                start_row=start_row,
                column_mode=group_mode,
                active_columns=active_columns,
                enable_synthesis=enable_synthesis,
                settings=adaptive_settings,
                variant_generator=variant_generator,
                run_dir=group_run_dir,
                digests=digests,
            )
            reports.extend(g_reports)
            flat_issues.extend(g_flat)
            per_group_gt.append((group_mode, g_artifacts))
        gt_artifacts = _merge_adaptive_gt_artifacts(per_group_gt)
        if use_row_dup_adjudicator:
            row_dup_detector = RowDuplicationDetector(
                llm=llm,
                taxonomy=taxonomy,
                model=model,
                settings=row_dup_settings,
                digests=digests,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                concurrency=concurrency,
                arm_config=row_dup_arm_config,
            )
            extra_reports = row_dup_detector.run(source_frames)
            reports.extend(extra_reports)
            flat_issues.extend(
                flatten_issues(
                    extra_reports,
                    source_frames=source_frames,
                    include_full_row=enable_synthesis,
                )
            )
            logger.info(
                "row_duplication: appended %d adjudicator report(s) -> "
                "%d unified raw detections total",
                len(extra_reports),
                len(flat_issues),
            )
        logger.info(
            "Adaptive loop finished: %d pass(es) done, %d unified raw detections",
            gt_artifacts["num_runs_completed"],
            len(flat_issues),
        )
        # No single canonical detector in adaptive mode - record the
        # concurrency each pass actually used by sniffing a throwaway
        # instance built with the same knobs (cheap; no LLM calls).
        representative_concurrency = PerTypeCleaningDetector(
            llm=llm,
            taxonomy=taxonomy,
            model=model,
            error_types=error_types,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            concurrency=concurrency,
        ).concurrency

    # Phase A: cluster on the full (possibly multi-pass) pool.
    clusters = cluster_issues(flat_issues)

    # Phase B: LLM synthesis of multi-type clusters --------------------
    synthesis_meta: Dict[str, Any] = {
        "enabled": enable_synthesis,
        "effort": synthesis_effort if enable_synthesis else None,
        "concurrency": None,
        "prompt_file": str(synthesis_prompt_file) if enable_synthesis else None,
    }
    if enable_synthesis:
        synthesis_prompt = load_synthesis_prompt(synthesis_prompt_file)
        synthesiser = ClusterSynthesiser(
            llm=llm,
            taxonomy=taxonomy,
            prompt=synthesis_prompt,
            model=model,
            reasoning_effort=synthesis_effort,
            reasoning_summary=reasoning_summary,
            concurrency=synthesis_concurrency,
        )
        clusters = synthesiser.run(clusters)
        synthesis_meta["concurrency"] = synthesiser.concurrency

    sources_list = [p.name for p in csv_paths]
    config_snapshot: Dict[str, Any] = {
        "mode": mode,
        "model": model,
        "chunk_size": chunk_size,
        "max_chunks": max_chunks,
        "column_mode": column_mode,
        "active_columns": active_columns,
        "csv_path": str(csv_path),
        "csv_glob": csv_glob if len(csv_paths) > 1 else None,
        "csv_paths": [str(p) for p in csv_paths],
        "sources": sources_list,
        "source_count": len(sources_list),
        "chunks_per_source": per_source_chunks,
        "reasoning_effort": reasoning_effort,
        "reasoning_summary": reasoning_summary,
        "error_types": [et.key for et in error_types],
        "error_types_file": str(error_types_file),
        "scopes": sorted(active_scopes) if active_scopes else None,
        "concurrency": representative_concurrency,
        "chunk_count": len(chunks),
        "detection_call_count_per_pass": expected_calls_per_pass,
        "detection_call_count": (
            expected_calls_per_pass
            if mode == "single"
            else expected_calls_per_pass * gt_artifacts["num_runs_completed"]
        ),
        "synthesis": synthesis_meta,
        "normalize_input": dict(norm_opts.describe()),
        "include_table_digest": include_digest,
    }
    if mode == "adaptive":
        config_snapshot["good_turing"] = {
            "gt_mode": adaptive_settings["gt_mode"],
            "num_runs_completed": gt_artifacts["num_runs_completed"],
            "num_runs_max": adaptive_settings["num_runs"],
            "min_runs": adaptive_settings["min_runs"],
            "target_missing_mass": adaptive_settings["target_missing_mass"],
            "stop_scope": adaptive_settings["stop_scope"],
            "target_policy": adaptive_settings["target_policy"],  # back-compat alias
            "saturated_types": gt_artifacts["estimates_payload"].get("saturated_types", []),
            "base_seed": adaptive_settings["base_seed"],
            "shuffle_rows": adaptive_settings["shuffle_rows"],
            "shuffle_columns": adaptive_settings["shuffle_columns"],
            "include_identity_run": not adaptive_settings["skip_identity_run"],
            "prompt_ensemble": adaptive_settings["prompt_ensemble"],
            "prompt_variant_template": adaptive_settings["prompt_variant_template"],
            "variant_generation_effort": adaptive_settings["variant_generation_effort"],
            "stop_reason": gt_artifacts["stop_reason"],
            "overall_estimate": gt_artifacts["overall_estimate"],
        }
    run_metadata: Dict[str, Any] = dict(config_snapshot)
    run_metadata.update(
        {
            "run_id": run_id,
            "generated_at": timestamp,
            "source": (
                csv_paths[0].name if len(csv_paths) == 1
                else f"{len(csv_paths)} files"
            ),
            "sources": sources_list,
            "error_types_count": len(error_types),
        }
    )

    # In adaptive mode, dump the Good-Turing artifacts alongside the main
    # dashboard so users can inspect the estimator progression + per-run
    # detections without leaving the run directory.
    if gt_artifacts is not None:
        gt_dir = run_dir / "good_turing"
        gt_dir.mkdir(parents=True, exist_ok=True)
        with (gt_dir / "species.jsonl").open("w", encoding="utf-8") as fh:
            for rec in gt_artifacts["species_records"]:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        with (gt_dir / "progression.jsonl").open("w", encoding="utf-8") as fh:
            for row in gt_artifacts["progression"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        (gt_dir / "estimates.json").write_text(
            json.dumps(gt_artifacts["estimates_payload"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (gt_dir / "report.md").write_text(
            gt_artifacts["markdown_report"], encoding="utf-8"
        )
        # Additive scope-aware artifacts (per-arm, per-column).
        # Written alongside the legacy global block so existing
        # tooling keeps working unchanged.
        per_arm_payload = gt_artifacts.get("per_arm_payload")
        if per_arm_payload is not None:
            (gt_dir / "per_arm_estimates.json").write_text(
                json.dumps(per_arm_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        per_arm_md = gt_artifacts.get("per_arm_markdown")
        if per_arm_md is not None:
            (gt_dir / "per_arm_report.md").write_text(
                per_arm_md, encoding="utf-8"
            )
        # Per-pass per-arm snapshot stream for offline hyperparameter
        # sweeps. One JSON line per ``(pass_idx, arm)`` row; consumed
        # by ``experiments/analyze_stop_thresholds.py``.
        per_pass_snaps = gt_artifacts.get("per_pass_per_arm_snapshots") or []
        if per_pass_snaps:
            with (gt_dir / "per_pass_per_arm.jsonl").open(
                "w", encoding="utf-8"
            ) as fh:
                for row in per_pass_snaps:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        runs_subdir = run_dir / "runs"
        runs_subdir.mkdir(parents=True, exist_ok=True)
        for run_dump in gt_artifacts["per_run_dumps"]:
            sub = runs_subdir / f"run_{run_dump['run_id']:02d}"
            sub.mkdir(parents=True, exist_ok=True)
            with (sub / "detections.jsonl").open("w", encoding="utf-8") as fh:
                for d in run_dump["detections"]:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            (sub / "run.json").write_text(
                json.dumps(run_dump["meta"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    html_template = PROJECT_ROOT / "frontend" / "report_template.html"

    # Slim GT payload for the HTML dashboard. `None` in single mode so
    # the frontend hides the Good-Turing tab entirely. Persisted to
    # ``good_turing/gt_dashboard.json`` so ``experiments.build_report``
    # can rebuild the HTML later without re-running the loop.
    gt_dashboard: Optional[Dict[str, Any]] = None
    if gt_artifacts is not None and adaptive_settings is not None:
        gt_dashboard = {
            "enabled": True,
            "gt_mode": adaptive_settings["gt_mode"],
            "num_runs_completed": gt_artifacts["num_runs_completed"],
            "num_runs_max": adaptive_settings["num_runs"],
            "min_runs": adaptive_settings["min_runs"],
            "target_missing_mass": adaptive_settings["target_missing_mass"],
            "stop_scope": adaptive_settings["stop_scope"],
            "target_policy": adaptive_settings["target_policy"],  # back-compat alias
            "saturated_types": gt_artifacts["estimates_payload"].get("saturated_types", []),
            "saturation_history": gt_artifacts["estimates_payload"].get("saturation_history", []),
            "base_seed": adaptive_settings["base_seed"],
            "shuffle_rows": adaptive_settings["shuffle_rows"],
            "shuffle_columns": adaptive_settings["shuffle_columns"],
            "include_identity_run": not adaptive_settings["skip_identity_run"],
            "prompt_ensemble": adaptive_settings["prompt_ensemble"],
            "prompt_variant_template": adaptive_settings["prompt_variant_template"],
            "variant_generation_effort": adaptive_settings["variant_generation_effort"],
            "stop_reason": gt_artifacts["stop_reason"],
            "overall": gt_artifacts["overall_estimate"],
            "per_error_type": gt_artifacts["per_error_type"],
            "per_source": gt_artifacts["per_source"],
            "progression": gt_artifacts["progression"],
            "runs": gt_artifacts["run_info"],
            "variant_records_per_run": gt_artifacts["variant_records_per_run"],
            # Additive: scope-aware per-arm estimator (column scope only
            # for now). ``None`` if the per-arm computation was skipped
            # or failed; the dashboard hides its panel in that case.
            "per_arm": gt_artifacts.get("per_arm_payload"),
        }
        (run_dir / "good_turing" / "gt_dashboard.json").write_text(
            json.dumps(gt_dashboard, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Report rendering is now a SEPARATE step. Run `python -m
    # experiments.build_report <run_dir>` after detection finishes to
    # produce report.html. This keeps the detection run itself free of
    # frontend concerns and lets users regenerate the dashboard without
    # re-running the (expensive) LLM calls.
    artifacts = build_run_artifacts(
        reports,
        run_dir,
        flat_issues=flat_issues,
        clusters=clusters,
        config=config_snapshot,
        run_metadata=run_metadata,
        html_template_path=None,
        source_frames=source_frames,
        good_turing=gt_dashboard,
    )

    summary = json.loads(artifacts["summary_json"].read_text(encoding="utf-8"))
    logger.info(
        "Summary: %s",
        json.dumps(
            {k: v for k, v in summary.items() if k != "config"},
            ensure_ascii=False,
        ),
    )

    # --- Optional web verification post-pass --------------------------
    # Decoupled from the detection pipeline so users can opt-in per
    # run; failures here are logged but do not fail the whole run
    # (the detection artifacts are already on disk at this point).
    if getattr(args, "verify_web", False):
        try:
            from experiments.run_web_verify import verify_run as _verify_run
            logger.info("web verification: starting (agent_id=%s)", args.web_agent_id or "<env>")
            wv = _verify_run(
                run_dir,
                agent_endpoint=args.web_agent_endpoint,
                agent_id=args.web_agent_id,
                model=str(config.get("llm", {}).get("model", "gpt-5.2")),
                concurrency=args.web_concurrency,
                max_clusters=args.web_max_clusters,
                include_person=args.web_include_person,
                force=args.web_force,
            )
            logger.info(
                "web verification: %d/%d eligible, verified=%d, counts=%s -> %s",
                wv["eligible"], wv["total_clusters"], wv["verified_now"],
                wv["counts"],
                wv["output_path"].relative_to(PROJECT_ROOT)
                if wv["output_path"].is_relative_to(PROJECT_ROOT)
                else wv["output_path"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("web verification failed: %s", exc)
    if gt_artifacts is not None:
        overall = gt_artifacts["overall_estimate"]  # Estimate.to_dict()
        chao1 = overall["chao1"]
        fn = overall["estimated_fn_rate"]
        logger.info(
            "Good-Turing: runs=%d/%d (%s) | S_obs=%d | missing mass=%.2f%% | "
            "Chao1 total=%.1f [95%% CI %.1f-%.1f] | FN rate=%.2f%% [95%% CI %.2f%%-%.2f%%]",
            gt_artifacts["num_runs_completed"],
            adaptive_settings["num_runs"],
            adaptive_settings["gt_mode"],
            overall["n_observed"],
            100.0 * overall["good_turing_missing_mass"],
            chao1["estimated_total"],
            chao1["ci_95"][0],
            chao1["ci_95"][1],
            100.0 * fn["point"],
            100.0 * fn["ci_95"][0],
            100.0 * fn["ci_95"][1],
        )
    def _rel(p: Path) -> Path:
        try:
            return p.relative_to(PROJECT_ROOT)
        except ValueError:
            return p

    logger.info("Artifacts written under: %s", run_dir)
    for name, path in artifacts.items():
        logger.info("  %-22s %s", name, _rel(path))
    if gt_artifacts is not None:
        gt_dir = run_dir / "good_turing"
        for fname in ("report.md", "estimates.json", "progression.jsonl", "species.jsonl"):
            fp = gt_dir / fname
            if fp.is_file():
                logger.info("  %-22s %s", f"gt_{fname}", _rel(fp))
    logger.info(
        "To build the HTML dashboard, run: "
        "python -m experiments.build_report %s", _rel(run_dir),
    )
    _write_run_status(run_dir, "completed")
    return 0


def _main_with_status(argv: Optional[list[str]] = None) -> int:
    """Top-level entry that ensures partial runs are marked failed
    on disk so users don't mistake a half-written ``outputs/<ts>/``
    for a successful run.
    """
    try:
        return main(argv)
    except SystemExit:
        # argparse / explicit exit codes - re-raise unchanged.
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_cleaning_detection failed: %s", exc)
        # Best-effort: walk the most recent run dir under outputs/ and
        # mark it failed. We don't know the run_dir at this scope, so
        # rely on RUN_STATUS.json having been seeded as "in_progress"
        # by main(); search for the freshest one and stamp it.
        try:
            outputs_root = Path("outputs")
            if outputs_root.is_dir():
                candidates = [
                    p for p in outputs_root.iterdir()
                    if p.is_dir() and (p / "RUN_STATUS.json").is_file()
                ]
                if candidates:
                    latest = max(candidates, key=lambda p: p.stat().st_mtime)
                    cur = json.loads((latest / "RUN_STATUS.json").read_text())
                    if cur.get("status") == "in_progress":
                        _write_run_status(
                            latest, "failed", detail=f"{type(exc).__name__}: {exc}",
                        )
        except Exception:  # noqa: BLE001
            logger.debug("could not stamp failed status", exc_info=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main_with_status())
