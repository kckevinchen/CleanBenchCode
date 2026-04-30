"""Single-pass schema discovery (Phase D).

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
from typing import Any, Dict, List, Optional

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
    try:
        from experiments.build_relation_report import build_relation_report  # noqa: WPS433
        out = build_relation_report(run_dir, template_path=template_path)
        logger.info("dashboard: %s", out)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("dashboard rendering failed: %s", exc)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-pass schema discovery of multi-column relations.",
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
        "--max-passes",
        type=int,
        default=5,
        help="Adaptive: hard cap on passes per arm (default 5).",
    )
    p.add_argument(
        "--min-passes",
        type=int,
        default=2,
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
            "estimate falls below this (default 0.05 = 5%%)."
        ),
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Adaptive: seed offset for per-pass row sampling.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        level=logging.INFO,
    )
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
    if not error_types:
        logger.error(
            "No requested error types have an entry in %s. Available: %s",
            discovery_path, sorted(discovery_cfg.types),
        )
        return 2

    # Load every CSV and build digests.
    source_frames: Dict[str, pd.DataFrame] = {}
    for path in csv_paths:
        df = load_csv(path)
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
        logger.debug("Loaded %s: %d rows x %d cols", name, len(df), len(df.columns))

    digests = {
        name: build_table_digest(df, source=name, sample_k_values=8)
        for name, df in source_frames.items()
    }

    model = args.model or llm_cfg.get("model", "gpt-5.2")
    reasoning_effort = args.reasoning_effort or llm_cfg.get("reasoning_effort", "medium")
    reasoning_summary = llm_cfg.get("reasoning_summary", "detailed")
    concurrency = args.concurrency

    if args.dry_run:
        # Render first (source, error_type) prompt and exit.
        first_source = next(iter(source_frames))
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
        print("# user prompt:\n")
        print(discovery_cfg.render(
            error_type=first_type,
            digest=digest,
            row_sample_md=row_sample,
        ))
        return 0

    llm = LLMClient()
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
        from src.core.relation_adaptive import (  # noqa: WPS433
            AdaptiveSchemaDiscoverer,
            adaptive_summary_dict,
        )

        adaptive = AdaptiveSchemaDiscoverer(
            discoverer,
            max_passes=args.max_passes,
            min_passes=args.min_passes,
            gt_threshold=args.gt_threshold,
            base_seed=args.base_seed,
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

        # Top-level summary.
        by_type: Dict[str, int] = {}
        for rel in result.union_relations:
            by_type[rel.error_type] = by_type.get(rel.error_type, 0) + 1
        n_llm_errors = sum(1 for r in all_reports if r.error)
        n_parse_errors = sum(1 for r in all_reports if r.parse_error)
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
                    "config": result.config,
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

    reports = discoverer.run()

    relations_path = run_dir / "relations.jsonl"
    reports_path = run_dir / "discovery_reports.json"

    n_relations = write_relations_jsonl(reports, relations_path)
    write_discovery_reports_json(reports, reports_path, include_raw=args.include_raw)

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
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _render_dashboard(run_dir)
    logger.info(
        "Schema discovery complete: %d relations across %d call(s) "
        "(%d LLM errors, %d parse errors). Artifacts: %s",
        n_relations, len(reports), llm_errored, parse_failed, run_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
