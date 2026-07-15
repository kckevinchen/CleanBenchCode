"""Single-pass relation discovery (Phase D).

Run ONE LLM call per (source, error_type) to enumerate every multi-
column / multi-row / multi-table relation that should hold over each
CSV. Output is persisted as ``relations.jsonl`` (one line per
relation) plus a slim per-call ``discovery_reports.json`` for
debugging.

This is the single-pass scaffold; the adaptive Good-Turing wrapper
(per-pass perturbation + missing-mass stop) graduates onto this in a
follow-up. For now the point is to (1) lock in the prompt and JSON
schema, (2) eyeball the relations on real CSVs, and (3) produce an
artifact the downstream per-row detector can consume.

Example::

    python -m experiments.run_relation_discovery \\
        --csv-dir data/curated_pbi \\
        --error-types fd_violation,row_calculation_inconsistency

Defaults to the multi-column / multi-row / multi-table types that
have entries in ``prompts/discover_relations.yaml``; pass
``--error-types`` to restrict further.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import load_csv  # noqa: E402
from src.core.error_taxonomy import load_taxonomy  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402
from src.core.relations import (  # noqa: E402
    SchemaDiscoverer,
    load_discovery_prompt,
    write_discovery_reports_json,
    write_relations_jsonl,
)
from src.core.table_digest import build_table_digest  # noqa: E402
from src.core.relations.aggregate import (  # noqa: E402
    AggregationResult,
    aggregate_relations,
    load_aggregation_prompt,
    write_aggregated_jsonl,
    write_aggregation_report_json,
)

logger = logging.getLogger("run_relation_discovery")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return raw if isinstance(raw, dict) else {}


def _resolve(path_like: str, root: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (root / p).resolve()


def _csv_paths(csv_path: Optional[Path], csv_dir: Optional[Path], glob: str) -> List[Path]:
    paths: List[Path] = []
    if csv_path is not None and csv_path.is_file():
        paths.append(csv_path)
    elif csv_path is not None and csv_path.is_dir():
        paths.extend(sorted(csv_path.rglob(glob)))
    if csv_dir is not None:
        paths.extend(sorted(csv_dir.rglob(glob)))
    # Drop dupes preserving order.
    seen: set[Path] = set()
    deduped: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(p)
    return deduped


def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def _render_dashboard(run_dir: Path) -> None:
    """Render the HTML dashboard for a completed discovery run.

    Failure here is non-fatal: the JSON artifacts are the source of
    truth, the HTML is a convenience.
    """
    template_path = PROJECT_ROOT / "frontend" / "relations_template.html"
    if not template_path.is_file():
        logger.warning("dashboard template missing: %s", template_path)
        return
    has_relations = (run_dir / "relations.jsonl").is_file()
    has_multi = (
        (run_dir / "multi_table_groups.jsonl").is_file()
        or (run_dir / "multi_table_violations.jsonl").is_file()
    )
    if not (has_relations or has_multi):
        logger.info(
            "dashboard skipped: %s has neither relations.jsonl nor "
            "multi_table_*.jsonl artifacts", run_dir,
        )
        return
    try:
        from experiments.build_relation_report import build_relation_report  # noqa: WPS433
        out = build_relation_report(run_dir, template_path=template_path)
        logger.info("dashboard: %s", out)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("dashboard rendering failed: %s", exc)


def _run_aggregation_pass(
    *,
    run_dir: Path,
    relations: Sequence[Any],
    digests: Mapping[str, Any],
    source_frames: Mapping[str, Any],
    llm: Any,
    model: str,
    reasoning_effort: str,
    reasoning_summary: str,
    n_row_samples: int,
    concurrency: int,
    prompt_path: Path,
    min_cluster_size: int,
) -> tuple[Optional[Dict[str, Any]], Sequence[Any]]:
    """Cluster discovered rules per (source, error_type, columns)
    and ask the LLM to partition each cluster into canonical rules,
    folding together paraphrases. Returns ``(stats, post_relations)``
    where ``post_relations`` is the union of (a) canonical rules and
    (b) originals not covered by any canonical rule. When the pass
    is skipped, returns ``(None, relations)`` unchanged so callers
    can pass through.
    """
    if not prompt_path.is_file():
        logger.warning(
            "aggregation prompt missing: %s (skipping pass)", prompt_path,
        )
        return None, relations
    prompt_cfg = load_aggregation_prompt(prompt_path)
    result: AggregationResult = aggregate_relations(
        relations,
        digests=digests,
        source_frames=source_frames,
        prompt_cfg=prompt_cfg,
        llm=llm,
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        n_row_samples=n_row_samples,
        min_cluster_size=min_cluster_size,
        concurrency=concurrency,
    )
    if result.n_clusters == 0:
        logger.info("Aggregation: no multi-rule arms to merge.")
        return {
            "n_clusters": 0,
            "n_aggregated_clusters": 0,
            "n_canonical_rules": 0,
            "n_declined": 0,
            "n_errored": 0,
            "n_originals_collapsed": 0,
        }, relations
    n_written = write_aggregated_jsonl(
        result, run_dir / "relations_aggregated.jsonl",
    )
    write_aggregation_report_json(
        result, run_dir / "aggregation_report.json",
    )
    logger.info(
        "Aggregation: %d cluster(s), %d merged into %d canonical rules, "
        "%d declined, %d errored, %d originals collapsed.",
        result.n_clusters, result.n_aggregated_clusters,
        result.n_canonical_rules, result.n_declined,
        result.n_errored, result.n_originals_collapsed,
    )
    stats = {
        "n_clusters": result.n_clusters,
        "n_aggregated_clusters": result.n_aggregated_clusters,
        "n_canonical_rules": n_written,
        "n_declined": result.n_declined,
        "n_errored": result.n_errored,
        "n_originals_collapsed": result.n_originals_collapsed,
    }
    return stats, result.post_aggregation_relations(relations)


def _run_multi_table_pass(
    *,
    run_dir: Path,
    workbooks: Mapping[str, Sequence[str]],
    source_frames: Mapping[str, Any],
    digests: Mapping[str, Any],
    llm: Any,
    model: str,
    prompt_path: Path,
    concurrency: int,
) -> Dict[str, Any]:
    """Run the multi-table cross-sheet **discovery** pass (steps 1+2).

    Discovery emits the *relation* (``multi_table_groups.jsonl``) so
    downstream consumers can decide what to do with it. Per-row
    violation detection (step 3) and the optional LLM verifier
    (step 4) live in the cleaning-detection runner; they consume the
    groups file via ``--multi-table-from`` (or run end-to-end
    inline). Splitting the pipeline this way keeps each runner doing
    one thing and matches the single-table convention where
    ``run_relation_discovery`` produces relations and
    ``run_cleaning_detection`` produces error findings.

    Output: ``multi_table_groups.jsonl`` - one line per workbook,
    carrying every confirmed (and rejected) group + column mapping
    the LLM emitted at step 2.

    Returns a stats dict for ``summary.json``. Workbooks with <2
    sheets are skipped silently; an empty stats dict is returned
    when the entire workbook map has none.
    """
    from src.core.multi_table.pipeline import resolve_groups_pass

    _, stats = resolve_groups_pass(
        run_dir=run_dir,
        workbooks=workbooks,
        source_frames=source_frames,
        digests=digests,
        llm=llm,
        model=model,
        prompt_path=prompt_path,
        concurrency=concurrency,
    )
    return stats


def _run_validation_pass(
    *,
    run_dir: Path,
    source_frames: Mapping[str, Any],
    llm: Any,
    model: str,
    prompt_path: Path,
    layer: str,
    concurrency: int,
    sample_rows: int,
    timeout_seconds: float,
    max_violation_samples: int,
    limit: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Empirically validate the discovered relations.

    Picks the highest available rule layer (aggregated > originals)
    when ``layer == "auto"`` and asks the LLM to write a
    ``check(df)`` function for each rule. The function runs in a
    restricted sandbox; pass-rate and sample violations are persisted
    to ``relation_verifications.jsonl``. Returns a stats dict for
    inclusion in ``summary.json`` or ``None`` when the pass was
    skipped (e.g. prompt missing).

    The runner has the source DataFrames and an LLM client already
    in hand, so we avoid the standalone CLI's ``summary.json``
    parsing round-trip and instead call the validator directly.
    """
    if not prompt_path.is_file():
        logger.warning(
            "validation prompt missing: %s (skipping pass)", prompt_path,
        )
        return None
    # Imported lazily so the discovery module's import surface
    # (and the dry-run path) doesn't pay for it.
    from src.core.relations.validator import (  # noqa: WPS433
        load_validation_prompt,
        validate_relations,
        write_verifications_jsonl,
    )
    from src.core.jsonl_io import read_jsonl  # noqa: WPS433

    layer_files = {
        "aggregated":  "relations_aggregated.jsonl",
        "originals":   "relations.jsonl",
    }

    # Apply the per-pass limit during the read so very large relation
    # files don't get materialised in full just to slice off the
    # leading N records.
    read_limit = limit if (limit is not None and limit > 0) else None

    chosen: str = layer
    relations: List[Dict[str, Any]] = []
    if layer == "union":
        # Merge aggregated canonicals + originals that no canonical
        # absorbed. Singleton originals (no paraphrase cluster) are
        # NEVER written to relations_aggregated.jsonl, so the
        # 'aggregated' and 'auto' layers silently drop them. 'union'
        # keeps them.
        agg_rels = read_jsonl(
            run_dir / layer_files["aggregated"], limit=None,
        )
        orig_rels = read_jsonl(
            run_dir / layer_files["originals"], limit=None,
        )
        absorbed: set[str] = set()
        for r in agg_rels:
            for k in (r.get("aggregated_from") or []):
                if k:
                    absorbed.add(str(k))
        kept_orig = [
            r for r in orig_rels
            if str(r.get("species_key") or "") not in absorbed
        ]
        relations = list(agg_rels) + kept_orig
        if read_limit is not None:
            relations = relations[:read_limit]
        chosen = "union"
    elif layer == "auto":
        for name in ("aggregated", "originals"):
            relations = read_jsonl(run_dir / layer_files[name], limit=read_limit)
            if relations:
                chosen = name
                break
    elif layer in layer_files:
        relations = read_jsonl(run_dir / layer_files[layer], limit=read_limit)
    else:
        logger.warning("unknown validation layer %r; skipping", layer)
        return None

    if not relations:
        logger.info("validation: no relations to score (layer=%s)", chosen)
        return {
            "layer": chosen,
            "n_relations": 0,
            "n_verifiable": 0,
            "n_scored": 0,
            "n_perfect": 0,
            "n_errors": 0,
        }

    prompt_cfg = load_validation_prompt(prompt_path)

    def _progress(done: int, total: int) -> None:
        # Log every ~5%% so adaptive runs over hundreds of relations
        # don't drown the runner's console.
        if done == total or done % max(1, total // 20) == 0:
            logger.info("validation: %d/%d relations scored", done, total)

    records = validate_relations(
        relations,
        source_frames=source_frames,
        llm=llm,
        model=model,
        prompt_cfg=prompt_cfg,
        concurrency=max(1, int(concurrency)),
        sample_rows=int(sample_rows),
        timeout_seconds=float(timeout_seconds),
        max_violation_samples=int(max_violation_samples),
        layer=chosen,
        progress_cb=_progress,
    )

    out_path = run_dir / "relation_verifications.jsonl"
    n_written = write_verifications_jsonl(records, out_path)

    n_verifiable = sum(1 for r in records if r.verifiable)
    n_scored = sum(1 for r in records if r.pass_rate is not None)
    n_perfect = sum(
        1 for r in records
        if r.pass_rate is not None and r.pass_rate >= 0.999
    )
    n_mostly = sum(
        1 for r in records
        if r.pass_rate is not None and 0.85 <= r.pass_rate < 0.999
    )
    n_poor = sum(
        1 for r in records
        if r.pass_rate is not None and r.pass_rate < 0.85
    )
    n_errors = sum(1 for r in records if r.error)
    logger.info(
        "Validation: layer=%s, %d relations, %d verifiable, %d scored, "
        "%d perfect, %d mostly, %d poor, %d errored. -> %s",
        chosen, len(records), n_verifiable, n_scored,
        n_perfect, n_mostly, n_poor, n_errors, out_path,
    )
    return {
        "layer": chosen,
        "n_relations": len(records),
        "n_written": n_written,
        "n_verifiable": n_verifiable,
        "n_scored": n_scored,
        "n_perfect": n_perfect,
        "n_mostly": n_mostly,
        "n_poor": n_poor,
        "n_errors": n_errors,
    }


def _run_violation_compilation_pass(
    *,
    run_dir: Path,
    source_frames: Mapping[str, Any],
    min_pass_rate: float,
    max_pass_rate: float,
    timeout_seconds: float,
) -> Optional[Dict[str, Any]]:
    """Compile per-row violations from the validated relations.

    Pipeline (linear, no LLM detect):

    1. Load ``relation_verifications.jsonl`` (the validator's
       per-rule ``check()`` code + ``pass_rate``).
    2. For each ``(rule, source)`` pair, classify the outcome via
       :func:`relation_violations._classify_relation_outcome`:

       * ``pass_rate >= max_pass_rate`` (rule holds) **or**
         ``pass_rate < min_pass_rate`` (false rule) \u2192 rejected.
         The rule is dropped; nothing reaches the labelling UI.
       * ``min_pass_rate \u2264 pass_rate < max_pass_rate`` \u2192 compile.
         Re-execute the sandboxed check and emit one
         ``code_verifier`` violation per failing row.
       * verifier could not score (``verifiable=False``, missing
         code, sandbox error, no ``pass_rate``) \u2192 emit a single
         ``needs_review`` placeholder so a human can review the
         rule against the table.

    3. Aggregate violations into clusters via ``cluster_issues``
       and write ``relation_violations.jsonl`` /
       ``clusters.json[l]`` / ``issues.json[l]`` so the unified
       label UI picks the findings up automatically.

    Returns a stats dict for ``summary.json``. Returns ``None`` when
    the validation artifact is missing.
    """
    # Lazy imports keep the module import cheap when the violation
    # compilation pass is disabled.
    from src.core.aggregator import cluster_issues  # noqa: WPS433
    from src.core.jsonl_io import read_jsonl, write_jsonl  # noqa: WPS433
    from src.core.report_builder import write_json  # noqa: WPS433
    from src.core.relations.violations import (  # noqa: WPS433
        OUTCOME_NEEDS_REVIEW,
        OUTCOME_REJECT,
        compile_code_violations,
        relation_violations_to_flat_issues,
    )

    verifications_path = run_dir / "relation_verifications.jsonl"
    if not verifications_path.is_file():
        logger.info(
            "violation compilation: no relation_verifications.jsonl in %s; "
            "skipping (run validation first)",
            run_dir,
        )
        return None

    verifications_list = read_jsonl(verifications_path)

    # Load the relation layer the validator actually scored.
    chosen_layer: Optional[str] = None
    for v in verifications_list:
        if v.get("layer"):
            chosen_layer = str(v["layer"])
            break
    layer_files = {
        "aggregated":  "relations_aggregated.jsonl",
        "originals":   "relations.jsonl",
    }
    relations_path = run_dir / layer_files.get(chosen_layer or "originals", "relations.jsonl")
    if not relations_path.is_file():
        for name in ("aggregated", "originals"):
            candidate = run_dir / layer_files[name]
            if candidate.is_file():
                relations_path = candidate
                chosen_layer = name
                break
    relations_list = read_jsonl(relations_path)
    if not relations_list:
        logger.info(
            "violation compilation: no relations to compile from in %s",
            relations_path,
        )
        return None

    # Build species_key -> relation lookup once.
    relations_by_species: Dict[str, Dict[str, Any]] = {}
    for r in relations_list:
        key = r.get("species_key")
        if not key:
            continue
        relations_by_species[str(key)] = r

    # One-pass compile: code-path violations + needs-review
    # placeholders are produced together by the compiler.
    violations, skip_log = compile_code_violations(
        relations_by_species=relations_by_species,
        verifications=verifications_list,
        source_frames=source_frames,
        min_pass_rate=float(min_pass_rate),
        max_pass_rate=float(max_pass_rate),
        timeout_seconds=float(timeout_seconds),
    )

    # Persist the audit log of skipped/needs-review rules so callers
    # can inspect why each rule landed where it did.
    audit_records = [
        {
            "species_key": entry.get("species_key", ""),
            "source": entry.get("source", ""),
            "outcome": entry.get("outcome", ""),
            "reason": entry.get("reason", ""),
        }
        for entry in skip_log
    ]
    if audit_records:
        write_jsonl(run_dir / "relation_skipped_rules.jsonl", audit_records)

    # Persist raw violations regardless of whether the compiler found
    # anything - an empty file is itself useful traceability.
    out_violations = run_dir / "relation_violations.jsonl"
    write_jsonl(out_violations, [v.as_dict() for v in violations])

    # Convert to flat-issue contract, cluster, and emit
    # clusters.json[l] / issues.json[l] for the label UI.
    flat_issues = relation_violations_to_flat_issues(violations)
    clusters = cluster_issues(flat_issues)
    cluster_dicts = [c.as_dict() for c in clusters]
    write_jsonl(run_dir / "clusters.jsonl", cluster_dicts)
    write_json(run_dir / "clusters.json", cluster_dicts)
    write_jsonl(run_dir / "issues.jsonl", flat_issues)
    write_json(run_dir / "issues.json", list(flat_issues))

    n_code_violations = sum(1 for v in violations if v.detection_method == "code_verifier")
    n_needs_review = sum(1 for v in violations if v.detection_method == "needs_review")
    n_rules_with_violations = len({v.relation_species_key for v in violations})
    n_rejected = sum(1 for e in skip_log if e.get("outcome") == OUTCOME_REJECT)
    n_needs_review_logged = sum(1 for e in skip_log if e.get("outcome") == OUTCOME_NEEDS_REVIEW)

    logger.info(
        "violation compilation: total %d violation(s) across %d rule(s) "
        "(code=%d, needs_review=%d) -> %d cluster(s); rejected=%d. "
        "Artifacts: %s, clusters.json",
        len(violations), n_rules_with_violations,
        n_code_violations, n_needs_review,
        len(clusters), n_rejected,
        out_violations.name,
    )
    return {
        "layer": chosen_layer,
        "n_violations": len(violations),
        "n_code_violations": n_code_violations,
        "n_needs_review_placeholders": n_needs_review,
        "n_rules_with_violations": n_rules_with_violations,
        "n_rules_rejected": n_rejected,
        "n_rules_needs_review": n_needs_review_logged,
        "n_clusters": len(clusters),
        "skipped_rules": _summarize_skip_log(skip_log),
        "config": {
            "min_pass_rate": float(min_pass_rate),
            "max_pass_rate": float(max_pass_rate),
            "timeout_seconds": float(timeout_seconds),
        },
    }


def _summarize_skip_log(skip_log: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Histogram skip-log entries by ``outcome`` and ``reason`` for
    ``summary.json``. Per-rule diagnostics stay in
    ``relation_skipped_rules.jsonl`` on disk; here we just want the
    counts so the dashboard can summarize at a glance."""
    by_outcome: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    for entry in skip_log:
        outcome = str(entry.get("outcome") or "unknown")
        reason = str(entry.get("reason") or "unknown")
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1
    return [
        {"reason": f"__outcome__:{o}", "count": c}
        for o, c in sorted(by_outcome.items())
    ] + [
        {"reason": r, "count": c}
        for r, c in sorted(by_reason.items())
    ]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-pass relation discovery of multi-column relations.",
    )
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "experiment.yaml"),
        help="Experiment config YAML (used for csv_dir / model / endpoints).",
    )
    p.add_argument("--csv", type=str, default=None, help="CSV file or directory.")
    p.add_argument("--csv-dir", type=str, default=None, help="Directory of CSVs.")
    p.add_argument("--csv-glob", type=str, default=None, help="Glob within csv_dir.")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--reasoning-effort", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument(
        "--error-types",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of error types to discover relations "
            "for. Defaults to every type that has an entry in "
            "prompts/discover_relations.yaml (i.e. all multi-column / "
            "multi-row / multi-table types)."
        ),
    )
    p.add_argument(
        "--n-row-samples",
        type=int,
        default=5,
        help="Stratified rows shown to the LLM alongside the digest.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override outputs directory (defaults to outputs/).",
    )
    p.add_argument(
        "--include-raw",
        action="store_true",
        help=(
            "Persist raw LLM responses + prompts in discovery_reports.json. "
            "Off by default to keep the artifact small."
        ),
    )
    p.add_argument(
        "--discovery-prompt",
        type=str,
        default="prompts/discover_relations.yaml",
        help="Path to the discovery prompt YAML.",
    )
    p.add_argument(
        "--list-error-types",
        action="store_true",
        help="Print every error type that has a discovery entry and exit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Render the first (source, error_type) discovery prompt to "
            "stdout and exit (no LLM call)."
        ),
    )
    # Adaptive multi-pass options.
    p.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Run multiple discovery passes with different row-sample "
            "seeds and stop each (source, error_type) arm via "
            "Good-Turing missing-mass. Without this flag the runner "
            "executes a single deterministic pass."
        ),
    )
    p.add_argument(
        "--max-passes", "--gt-max-runs",
        type=int,
        default=5,
        dest="max_passes",
        help="Adaptive: hard cap on passes per arm (default 5).",
    )
    p.add_argument(
        "--min-passes", "--gt-min-runs",
        type=int,
        default=2,
        dest="min_passes",
        help=(
            "Adaptive: minimum passes per arm before saturation can "
            "be declared (default 2)."
        ),
    )
    p.add_argument(
        "--gt-threshold",
        type=float,
        default=0.05,
        help=(
            "Adaptive: stop an arm once its Good-Turing missing-mass "
            "estimate falls below this (default 0.05 = 5%%). Used "
            "when --stop-when is 'gt' or 'both'."
        ),
    )
    p.add_argument(
        "--stop-when", "--gt-stop-when",
        choices=["gt", "chao1", "both"],
        default="both",
        dest="stop_when",
        help=(
            "Adaptive: which estimator gates per-arm saturation. "
            "'both' (default) -- require Good-Turing AND Chao1 "
            "criteria simultaneously. 'gt' -- Good-Turing missing-mass "
            "below --gt-threshold. 'chao1' -- upper-95 CI on Chao1 "
            "missing count, normalised by S_obs, below --epsilon-chao1."
        ),
    )
    p.add_argument(
        "--epsilon-chao1", "--gt-epsilon-chao1",
        type=float,
        default=0.05,
        dest="epsilon_chao1",
        help=(
            "Adaptive: stop an arm once the Chao1 upper-95 CI on the "
            "missing count, divided by S_obs, falls below this "
            "(default 0.05). Used when --stop-when is 'chao1' or "
            "'both'."
        ),
    )
    p.add_argument(
        "--base-seed", "--gt-base-seed",
        type=int,
        default=1000,
        dest="base_seed",
        help="Adaptive: seed offset for per-pass row sampling.",
    )
    p.add_argument(
        "--no-aggregate",
        action="store_true",
        help=(
            "Skip the post-discovery aggregation pass. By default, "
            "discovered rules in the same (source, error_type, columns) "
            "arm are clustered and the LLM is asked to merge paraphrases "
            "into a minimal set of canonical rules."
        ),
    )
    p.add_argument(
        "--aggregate-prompt",
        type=str,
        default="prompts/aggregate_relations.yaml",
        help="Path to the aggregation prompt YAML.",
    )
    p.add_argument(
        "--aggregate-min-cluster",
        type=int,
        default=2,
        help=(
            "Skip aggregation for arms with fewer than this many rules "
            "(default 2 - singletons can't be merged)."
        ),
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "Skip the post-discovery empirical-validation pass. By "
            "default, after aggregation the runner asks the LLM to "
            "write a check(df) function for each rule and runs it "
            "in a restricted sandbox to record per-rule pass-rates "
            "in relation_verifications.jsonl."
        ),
    )
    p.add_argument(
        "--no-multi-table",
        action="store_true",
        help=(
            "Skip the cross-sheet UNION/JOIN discovery pass for "
            "inter_table_inconsistency. When the pass runs it "
            "REPLACES the LLM check()-codegen path for that error "
            "type: structural priors are computed and the LLM is "
            "asked once per workbook to confirm groups + provide a "
            "canonical column mapping. The result is written to "
            "multi_table_groups.jsonl. Per-row violations (step 3) "
            "and the optional LLM verifier (step 4) are produced by "
            "run_cleaning_detection (which can consume this run's "
            "groups via --multi-table-from)."
        ),
    )
    p.add_argument(
        "--multi-table-prompt",
        type=str,
        default="prompts/confirm_union_join.yaml",
        help=(
            "Path to the union/join confirmation prompt YAML "
            "(default: prompts/confirm_union_join.yaml)."
        ),
    )
    p.add_argument(
        "--validate-prompt",
        type=str,
        default="prompts/validate_relation.yaml",
        help="Path to the validation prompt YAML.",
    )
    p.add_argument(
        "--validate-layer",
        choices=["auto", "union", "aggregated", "originals"],
        default="union",
        help=(
            "Which rule layer to validate. 'union' (default) covers all "
            "post-aggregation rules: aggregated canonicals + singleton "
            "originals that no canonical absorbed. 'auto' falls back "
            "from aggregated to originals, but DROPS singleton originals "
            "when aggregated is non-empty."
        ),
    )
    p.add_argument(
        "--validate-sample-rows",
        type=int,
        default=30,
        help="Head rows included in the validation prompt (default 30).",
    )
    p.add_argument(
        "--validate-timeout",
        type=float,
        default=8.0,
        help="Per-rule check() wall-clock cap in seconds (default 8).",
    )
    p.add_argument(
        "--validate-max-violations",
        type=int,
        default=5,
        help="Sample violating rows to record per rule (default 5).",
    )
    p.add_argument(
        "--validate-limit",
        type=int,
        default=None,
        help=(
            "Cap the number of relations validated (default: no cap). "
            "Useful for smoke tests on large adaptive runs."
        ),
    )
    # ---- Violation compilation (post-validation, pre-multi-table) ----
    p.add_argument(
        "--no-compile-violations",
        action="store_true",
        help=(
            "Skip the violation compilation pass. By default, after "
            "validation, the runner classifies each validated rule "
            "by pass-rate: rules in the keep band re-execute their "
            "check(df) to materialize per-row violations; rules the "
            "verifier could not score become needs-review "
            "placeholders. The combined output is clustered and "
            "persisted to clusters.json so the unified label UI "
            "surfaces every kept rule for human review."
        ),
    )
    p.add_argument(
        "--violations-min-pass-rate",
        type=float,
        default=0.90,
        help=(
            "Reject rules whose validated pass_rate is below this "
            "threshold (default 0.90: a real rule holds on essentially "
            "every row and the few violations are the actual errors; a "
            "rule violated by >10%% of rows is not a real rule and its "
            "'violations' are legitimate variation). Lower toward 0.5 "
            "for high-recall/low-precision."
        ),
    )
    p.add_argument(
        "--violations-max-pass-rate",
        type=float,
        default=1.0 - 1e-9,
        help=(
            "Reject rules whose pass_rate is at or above this "
            "threshold (default ~1.0). A fully-satisfied rule has no "
            "violations to surface. Set to 1.0 to keep pass_rate=1.0 "
            "rules."
        ),
    )
    p.add_argument(
        "--violations-timeout",
        type=float,
        default=8.0,
        help=(
            "Per-rule check() wall-clock cap during re-execution "
            "(default 8s). Independent of --validate-timeout because "
            "the compilation pass typically runs against larger "
            "frames than the validator did."
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        level=logging.INFO,
    )
    # Silence noisy per-request HTTP logs from the OpenAI/Azure client
    # transports. They flood the discovery UI without adding signal -
    # the LLMClient already logs request/response summaries.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = _build_argparser().parse_args(argv)

    cfg = _load_yaml(_resolve(args.config, PROJECT_ROOT))
    data_cfg = cfg.get("data", {}) or {}
    llm_cfg = cfg.get("llm", {}) or {}
    prompts_cfg = cfg.get("prompts", {}) or {}
    output_cfg = cfg.get("output", {}) or {}

    # Taxonomy (shared with detection).
    taxonomy_path = _resolve(
        prompts_cfg.get("error_types_file", "prompts/error_types.yaml"),
        PROJECT_ROOT,
    )
    taxonomy = load_taxonomy(taxonomy_path)

    # Discovery prompt config.
    discovery_path = _resolve(args.discovery_prompt, PROJECT_ROOT)
    discovery_cfg = load_discovery_prompt(discovery_path)

    if args.list_error_types:
        print("Error types with a discovery entry "
              f"({discovery_path}):")
        for key, spec in discovery_cfg.types.items():
            kinds = ", ".join(spec.kinds)
            print(f"  - {key}  (kinds: {kinds})")
        return 0

    # Resolve CSV paths. CLI --csv / --csv-dir override the config's
    # csv_dir entirely (mirrors the cleaning runner's precedence).
    csv_arg = Path(args.csv).expanduser() if args.csv else None
    if args.csv_dir:
        csv_dir_arg: Optional[Path] = Path(args.csv_dir).expanduser()
    elif csv_arg is not None:
        csv_dir_arg = None
    elif data_cfg.get("csv_dir"):
        csv_dir_arg = _resolve(data_cfg["csv_dir"], PROJECT_ROOT)
    else:
        csv_dir_arg = None
    csv_glob = args.csv_glob or str(data_cfg.get("csv_glob", "*.csv"))
    csv_paths = _csv_paths(csv_arg, csv_dir_arg, csv_glob)
    if not csv_paths:
        logger.error(
            "No CSVs found. --csv=%s --csv-dir=%s glob=%s",
            args.csv, args.csv_dir, csv_glob,
        )
        return 2

    # Error-type subset.
    cli_types = _parse_csv_list(args.error_types)
    cfg_types = prompts_cfg.get("error_types") or None
    requested = cli_types if cli_types else cfg_types
    error_types = taxonomy.select(requested)
    # Filter to types with a discovery entry (single-column types
    # never have one and would crash the renderer).
    error_types = [et for et in error_types if et.key in discovery_cfg.types]

    # Multi-table pass routing: by default ``inter_table_inconsistency``
    # is handled by the new structural priors -> LLM-confirm path
    # (see ``_run_multi_table_pass``), NOT by the LLM check()-codegen
    # flow. When the new pass is active we drop the type from
    # ``error_types`` so SchemaDiscoverer never issues a discovery call
    # for it. ``--no-multi-table`` flips both: keep the type, run the
    # legacy LLM relation discovery + check() validation path. Step 3
    # (per-row violation detection) lives in run_cleaning_detection.
    multi_table_enabled = (
        not args.no_multi_table
        and any(et.key == "inter_table_inconsistency" for et in error_types)
    )
    if multi_table_enabled:
        error_types = [
            et for et in error_types if et.key != "inter_table_inconsistency"
        ]
        logger.info(
            "Routing 'inter_table_inconsistency' through the multi-table "
            "cross-sheet discovery pass (priors -> LLM-confirm groups). "
            "Per-row violation detection is run by run_cleaning_detection. "
            "Pass --no-multi-table to use the legacy check()-codegen path."
        )

    if not error_types and not multi_table_enabled:
        logger.error(
            "No requested error types have an entry in %s. Available: %s",
            discovery_path, sorted(discovery_cfg.types),
        )
        return 2

    # Load every CSV and build digests.
    source_frames: Dict[str, pd.DataFrame] = {}
    # Workbook grouping: parent directory of the CSV is the
    # workbook id (xlsx stem in the bing21 layout, where each
    # ``<hash>.xlsx/`` contains the sheets exported as CSV). For
    # flat directories (no per-workbook parent), every source is
    # its own workbook and the table-scope discovery arms are
    # skipped automatically.
    workbooks: Dict[str, List[str]] = {}
    skipped_csvs: List[Path] = []
    for path in csv_paths:
        try:
            df = load_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            # The bing21 export occasionally contains whitespace-only
            # or otherwise unparseable sheets. Skip them with a warning
            # so a single bad CSV does not abort an entire workbook
            # batch; ``group_workbooks`` downstream handles workbooks
            # that drop below the multi-sheet threshold.
            logger.warning("Skipping unreadable CSV %s: %s", path, exc)
            skipped_csvs.append(path)
            continue
        # Source key is the bare filename, with relative path
        # disambiguation when basenames collide (mirrors the cleaning
        # runner's convention so artifacts cross-reference cleanly).
        name = path.name
        if name in source_frames:
            try:
                rel = path.relative_to(csv_dir_arg) if csv_dir_arg else path
            except ValueError:
                rel = path
            name = str(rel).replace("\\", "/")
        source_frames[name] = df
        workbook_id = path.parent.name or "_root"
        workbooks.setdefault(workbook_id, []).append(name)
        logger.debug("Loaded %s: %d rows x %d cols", name, len(df), len(df.columns))
    if skipped_csvs:
        logger.warning(
            "Skipped %d/%d CSV(s) that pandas could not parse "
            "(empty header, malformed, or mid-file corruption).",
            len(skipped_csvs), len(csv_paths),
        )
    if not source_frames:
        raise RuntimeError(
            f"No readable CSVs in {csv_dir_arg} "
            f"(tried {len(csv_paths)}, all empty or malformed)."
        )

    digests = {
        name: build_table_digest(df, source=name, sample_k_values=8)
        for name, df in source_frames.items()
    }

    model = args.model or llm_cfg.get("model", "gpt-5.2")
    reasoning_effort = args.reasoning_effort or llm_cfg.get("reasoning_effort", "high")
    reasoning_summary = llm_cfg.get("reasoning_summary", "detailed")
    concurrency = args.concurrency

    if args.dry_run:
        # Render first (source, error_type) prompt and exit.
        first_source = next(iter(source_frames))
        if not error_types:
            # Multi-table-only run: the single-relation types were routed
            # to the multi_table pass (error_types emptied), so there is
            # no relation-discovery prompt to preview here.
            print(f"# source     : {first_source}")
            print(
                "# (multi-table-only run: no single-relation discovery "
                "prompt to preview; the multi_table pass renders its own)"
            )
            return 0
        first_type = error_types[0]
        print(f"# source     : {first_source}")
        print(f"# error_type : {first_type.key}")
        print(f"# system     :\n{discovery_cfg.system_prompt}\n")
        from src.core.relations import sample_rows_for_prompt
        digest = digests[first_source]
        row_sample = sample_rows_for_prompt(
            source_frames[first_source],
            n_samples=args.n_row_samples,
            total_row_indices=digest.total_row_indices,
        )
        # Match what the runtime path would produce: include the
        # multi-row priors section when the type is group-scoped.
        priors_section = ""
        spec = discovery_cfg.types.get(first_type.key)
        if spec is not None and spec.scope == "group":
            from src.core.row_dup.priors import (
                compute_priors, render_group_blocks, render_priors_block,
            )
            priors = compute_priors(
                df=source_frames[first_source], digest=digest,
                error_type=first_type.key, source=first_source,
            )
            priors_section = (
                render_priors_block(source_frames[first_source], priors)
                + "\n"
                + render_group_blocks(source_frames[first_source], priors)
            )
        print("# user prompt:\n")
        print(discovery_cfg.render(
            error_type=first_type,
            digest=digest,
            row_sample_md=row_sample,
            priors_section=priors_section,
        ))
        return 0

    llm = LLMClient()
    discoverer: Optional[SchemaDiscoverer] = None
    if error_types:
        discoverer = SchemaDiscoverer(
            llm=llm,
            prompt_config=discovery_cfg,
            taxonomy=taxonomy,
            model=model,
            digests=digests,
            source_frames=source_frames,
            error_types=error_types,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            concurrency=concurrency,
            n_row_samples=args.n_row_samples,
            workbooks=workbooks,
        )
    elif multi_table_enabled:
        logger.info(
            "Only inter_table_inconsistency requested -> skipping legacy "
            "relation discovery; running the multi-table cross-sheet pass only."
        )

    n_multi_sheet = sum(1 for sheets in workbooks.values() if len(sheets) >= 2)
    if n_multi_sheet:
        logger.info(
            "Workbook grouping: %d workbook(s) with >=2 sheets eligible "
            "for cross-table discovery (out of %d total workbook(s))",
            n_multi_sheet, len(workbooks),
        )

    # Output dir is shared by both modes.
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _resolve(str(output_cfg.get("directory", "outputs")), PROJECT_ROOT)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_relations_adaptive" if args.adaptive else "_relations"
    run_dir = output_root / f"{timestamp}_{model}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist a compact per-source context dump up front (independent
    # of run mode). The report dashboard reads this to show "what does
    # each CSV look like?" alongside the relations it discovered.
    # `preview_rows` lets users audit relations against actual data
    # without re-loading the CSV: capped at 50 rows so the file
    # stays small on multi-source runs.
    PREVIEW_ROW_CAP = 50

    def _source_payload(name: str) -> Dict[str, Any]:
        payload = digests[name].to_dict()
        df = source_frames[name]
        head = df.head(PREVIEW_ROW_CAP)
        payload["preview_rows"] = [
            {col: ("" if pd.isna(v) else str(v)) for col, v in row.items()}
            for _, row in head.iterrows()
        ]
        payload["preview_total"] = int(len(df))
        return payload

    (run_dir / "sources.json").write_text(
        json.dumps(
            [_source_payload(name) for name in source_frames.keys()],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.adaptive:
        if discoverer is None:
            logger.error(
                "--adaptive requires at least one non-multi-table error type "
                "in --error-types. The multi-table pass runs in single-call "
                "mode and is not adaptive."
            )
            return 2
        from src.core.relations.adaptive import (  # noqa: WPS433
            AdaptiveSchemaDiscoverer,
            adaptive_summary_dict,
        )

        adaptive = AdaptiveSchemaDiscoverer(
            discoverer,
            max_passes=args.max_passes,
            min_passes=args.min_passes,
            gt_threshold=args.gt_threshold,
            base_seed=args.base_seed,
            stop_when=args.stop_when,
            epsilon_chao1=args.epsilon_chao1,
        )
        result = adaptive.run()

        # Flatten every pass into one combined report list (for the
        # union-relations and discovery-reports artifact).
        all_reports: List[Any] = []
        for pass_idx, reports in enumerate(result.pass_reports):
            for rep in reports:
                # Tag the report with its pass index in raw_text-adjacent
                # metadata: store on the in-memory object since
                # DiscoveryReport doesn't have a pass field.
                setattr(rep, "_adaptive_pass", pass_idx)
                all_reports.append(rep)

        relations_path = run_dir / "relations.jsonl"
        # Write union relations with per-species frequency metadata.
        with relations_path.open("w", encoding="utf-8") as fh:
            for rel in result.union_relations:
                payload = rel.as_dict()
                meta = result.union_metadata.get(rel.species_key(), {})
                payload["pass_frequency"] = meta.get("frequency", 0)
                payload["seen_in_passes"] = meta.get("seen_in_passes", [])
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        n_relations = len(result.union_relations)

        write_discovery_reports_json(
            all_reports, run_dir / "discovery_reports.json",
            include_raw=args.include_raw,
        )

        # GT artifact.
        gt_summary = adaptive_summary_dict(result)
        (run_dir / "gt_summary.json").write_text(
            json.dumps(gt_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Optional aggregation pass: merge paraphrases of the same
        # rule (same source / error_type / columns) into a minimal
        # canonical set. The validator + violation compiler
        # downstream read either the aggregated layer or the raw
        # originals depending on what's available.
        agg_stats: Optional[Dict[str, Any]] = None
        agg_relations: Sequence[Any] = result.union_relations
        if not args.no_aggregate:
            agg_stats, agg_relations = _run_aggregation_pass(
                run_dir=run_dir,
                relations=result.union_relations,
                digests=digests,
                source_frames=source_frames,
                llm=llm,
                model=model,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                n_row_samples=args.n_row_samples,
                concurrency=concurrency or 4,
                prompt_path=_resolve(args.aggregate_prompt, PROJECT_ROOT),
                min_cluster_size=args.aggregate_min_cluster,
            )

        # Top-level summary.
        by_type: Dict[str, int] = {}
        for rel in result.union_relations:
            by_type[rel.error_type] = by_type.get(rel.error_type, 0) + 1
        n_llm_errors = sum(1 for r in all_reports if r.error)
        n_parse_errors = sum(1 for r in all_reports if r.parse_error)

        # Empirical validation pass (optional). Runs AFTER aggregation
        # so we score the rules the user will actually see on the
        # dashboard, but BEFORE summary.json + dashboard render so the
        # verification stats land in summary.json and the report.html
        # picks up relation_verifications.jsonl.
        validation_stats: Optional[Dict[str, Any]] = None
        if not args.no_validate:
            validation_stats = _run_validation_pass(
                run_dir=run_dir,
                source_frames=source_frames,
                llm=llm,
                model=model,
                prompt_path=_resolve(args.validate_prompt, PROJECT_ROOT),
                layer=args.validate_layer,
                concurrency=concurrency or 4,
                sample_rows=args.validate_sample_rows,
                timeout_seconds=args.validate_timeout,
                max_violation_samples=args.validate_max_violations,
                limit=args.validate_limit,
            )

        # Violation compilation pass: turn validated rules into per-row
        # findings + clusters. Runs after validation (which produces
        # the ``check()`` code we re-execute) and before multi-table
        # (which has its own row-level pipeline).
        #
        # Pipeline (linear, no LLM detect):
        #   * pass_rate < min  or  pass_rate >= max  -> reject (drop).
        #   * min <= pass_rate < max                 -> compile rows.
        #   * no pass_rate / verifier could not run  -> needs-review
        #     placeholder so the labelling UI surfaces the rule.
        violations_stats: Optional[Dict[str, Any]] = None
        if not args.no_compile_violations and validation_stats is not None:
            violations_stats = _run_violation_compilation_pass(
                run_dir=run_dir,
                source_frames=source_frames,
                min_pass_rate=args.violations_min_pass_rate,
                max_pass_rate=args.violations_max_pass_rate,
                timeout_seconds=args.violations_timeout,
            )

        # Multi-table cross-sheet pass for inter_table_inconsistency.
        # Discovery only runs steps 1+2 (priors -> LLM-confirm groups);
        # step 3 (semantic detector) and step 4 (verifier) are owned
        # by the cleaning-detection runner, which can either consume
        # this run's ``multi_table_groups.jsonl`` via
        # ``--multi-table-from`` or run the whole pipeline end-to-end.
        multi_table_stats: Optional[Dict[str, Any]] = None
        if multi_table_enabled:
            multi_table_stats = _run_multi_table_pass(
                run_dir=run_dir,
                workbooks=workbooks,
                source_frames=source_frames,
                digests=digests,
                llm=llm,
                model=model,
                prompt_path=_resolve(args.multi_table_prompt, PROJECT_ROOT),
                concurrency=concurrency or 4,
            )

        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "mode": "adaptive",
                    "model": model,
                    "n_sources": len(source_frames),
                    "n_error_types": len(error_types),
                    "n_passes": result.n_passes,
                    "n_calls": len(all_reports),
                    "n_relations": n_relations,
                    "n_llm_errors": n_llm_errors,
                    "n_parse_errors": n_parse_errors,
                    "relations_by_error_type": by_type,
                    "n_arms": gt_summary["n_arms"],
                    "n_saturated_arms": gt_summary["n_saturated_arms"],
                    "saturated_arms_by_pass": gt_summary["saturated_arms_by_pass"],
                    # Merge the adaptive-discovery knobs (max_passes,
                    # gt_threshold, ...) with the data-config keys the
                    # labelling app + downstream report builders read
                    # (csv_paths, csv_glob). Keeping them under one
                    # ``config`` namespace mirrors the cleaning runner.
                    "config": {
                        **result.config,
                        "csv_paths": [str(p) for p in csv_paths],
                        "csv_glob": csv_glob,
                    },
                    "csv_paths": [str(p) for p in csv_paths],
                    "aggregation": agg_stats,
                    "validation": validation_stats,
                    "violations": violations_stats,
                    "multi_table": multi_table_stats,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _render_dashboard(run_dir)
        logger.info(
            "Adaptive discovery complete: %d passes, %d union relations, "
            "%d/%d arms saturated. Artifacts: %s",
            result.n_passes, n_relations,
            gt_summary["n_saturated_arms"], gt_summary["n_arms"], run_dir,
        )
        return 0

    reports: List[Any] = []
    n_relations = 0
    all_rels: List[Any] = []
    agg_relations: Sequence[Any] = all_rels
    agg_stats: Optional[Dict[str, Any]] = None
    validation_stats: Optional[Dict[str, Any]] = None
    violations_stats: Optional[Dict[str, Any]] = None

    if discoverer is not None:
        reports = discoverer.run()

        relations_path = run_dir / "relations.jsonl"
        reports_path = run_dir / "discovery_reports.json"

        n_relations = write_relations_jsonl(reports, relations_path)
        write_discovery_reports_json(reports, reports_path, include_raw=args.include_raw)

        # Optional aggregation pass: merge paraphrases of the same
        # rule (same source / error_type / columns) into a minimal
        # canonical set.
        for rep in reports:
            all_rels.extend(rep.relations)
        agg_relations = all_rels
        if not args.no_aggregate:
            agg_stats, agg_relations = _run_aggregation_pass(
                run_dir=run_dir,
                relations=all_rels,
                digests=digests,
                source_frames=source_frames,
                llm=llm,
                model=model,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                n_row_samples=args.n_row_samples,
                concurrency=concurrency or 4,
                prompt_path=_resolve(args.aggregate_prompt, PROJECT_ROOT),
                min_cluster_size=args.aggregate_min_cluster,
            )

        # Empirical-validation pass (optional). Mirrors the adaptive
        # branch: runs after aggregation so the rules being scored
        # match what the user sees on the dashboard.
        if not args.no_validate:
            validation_stats = _run_validation_pass(
                run_dir=run_dir,
                source_frames=source_frames,
                llm=llm,
                model=model,
                prompt_path=_resolve(args.validate_prompt, PROJECT_ROOT),
                layer=args.validate_layer,
                concurrency=concurrency or 4,
                sample_rows=args.validate_sample_rows,
                timeout_seconds=args.validate_timeout,
                max_violation_samples=args.validate_max_violations,
                limit=args.validate_limit,
            )

        # Violation compilation (mirrors the adaptive branch).
        # Rules the verifier could not score are kept as
        # needs-review placeholders; rules with pass_rate in
        # {>=max, <min} are rejected; the rest compile per-row.
        if not args.no_compile_violations and validation_stats is not None:
            violations_stats = _run_violation_compilation_pass(
                run_dir=run_dir,
                source_frames=source_frames,
                min_pass_rate=args.violations_min_pass_rate,
                max_pass_rate=args.violations_max_pass_rate,
                timeout_seconds=args.violations_timeout,
            )

    # Multi-table cross-sheet pass for inter_table_inconsistency.
    # Discovery only runs steps 1+2 (priors -> LLM-confirm groups);
    # step 3 (semantic detector) and step 4 (verifier) are owned by
    # the cleaning-detection runner. See ``_run_multi_table_pass`` and
    # ``src.core.multi_table.pipeline`` for the split.
    multi_table_stats: Optional[Dict[str, Any]] = None
    if multi_table_enabled:
        multi_table_stats = _run_multi_table_pass(
            run_dir=run_dir,
            workbooks=workbooks,
            source_frames=source_frames,
            digests=digests,
            llm=llm,
            model=model,
            prompt_path=_resolve(args.multi_table_prompt, PROJECT_ROOT),
            concurrency=concurrency or 4,
        )

    # Lightweight summary.
    by_type: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    parse_failed = 0
    llm_errored = 0
    for rep in reports:
        if rep.error:
            llm_errored += 1
        if rep.parse_error:
            parse_failed += 1
        by_type[rep.error_type] = by_type.get(rep.error_type, 0) + len(rep.relations)
        by_source[rep.source] = by_source.get(rep.source, 0) + len(rep.relations)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "mode": "single_pass",
                "model": model,
                "n_sources": len(source_frames),
                "n_error_types": len(error_types),
                "n_calls": len(reports),
                "n_relations": n_relations,
                "n_llm_errors": llm_errored,
                "n_parse_errors": parse_failed,
                "relations_by_error_type": by_type,
                "relations_by_source": by_source,
                # Surface csv_paths under ``config`` so the labelling
                # app's RunContext can resolve sheet basenames back to
                # files (used by the multi-table label page to fetch
                # full row contents around a flagged cell).
                "config": {
                    "csv_paths": [str(p) for p in csv_paths],
                    "csv_glob": csv_glob,
                },
                "aggregation": agg_stats,
                "validation": validation_stats,
                "violations": violations_stats,
                "multi_table": multi_table_stats,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _render_dashboard(run_dir)
    logger.info(
        "Relation discovery complete: %d relations across %d call(s) "
        "(%d LLM errors, %d parse errors). Artifacts: %s",
        n_relations, len(reports), llm_errored, parse_failed, run_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
