"""Step 2 of the relation pipeline: violation detection + fix synthesis.

Takes a Step-1 (relation-discovery) run directory, filters the discovered
relations to those the operator has approved in the review UI, runs the same
violation-compilation pipeline (code-path re-execution + LLM table-scan) that
the discovery runner uses for its preview compile, then fires the cluster
synthesis Phase B for canonical fixes.

Output is written to a NEW run directory with the same artifact shape as
``run_cleaning_detection`` (``clusters.json``, ``issues.jsonl``,
``summary.json``, ``RUN_STATUS.json``) so the unified labeling UI picks it
up automatically without any frontend changes.

Example::

    python -m experiments.run_relation_detection \\
        outputs/20260515T235305Z_..._relations_adaptive \\
        --error-types row_calculation_inconsistency,string_calculation_inconsistency \\
        --concurrency 20
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

from src.core.aggregator import (  # noqa: E402
    ClusterSynthesiser,
    cluster_issues,
    load_synthesis_prompt,
)
from src.core.data_loader import load_csv  # noqa: E402
from src.core.error_taxonomy import load_taxonomy  # noqa: E402
from src.core.jsonl_io import read_jsonl, write_jsonl  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402
from src.core.relations.llm_detector import (  # noqa: E402
    detect_violations_for_relations,
    load_detect_prompt,
)
from src.core.relations.violations import (  # noqa: E402
    compile_code_violations,
    relation_violations_to_flat_issues,
)
from src.core.report_builder import write_json  # noqa: E402

logger = logging.getLogger("run_relation_detection")

_APPROVE_VERDICTS = {"approved", "added"}


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p)


def _write_run_status(run_dir: Path, status: str, *, detail: Optional[str] = None) -> None:
    """Mirror ``run_cleaning_detection._write_run_status`` so the labeling
    UI sees the same progress markers (``in_progress`` / ``completed`` /
    ``failed``). Best-effort; never raises."""
    try:
        payload: Dict[str, Any] = {
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


def _load_relations(discovery_dir: Path) -> tuple[List[Dict[str, Any]], str]:
    """Prefer aggregated relations; fall back to originals.

    The aggregated layer is the canonical post-discovery rule set: paraphrase
    clusters have been merged so each species has one row. If aggregation
    was skipped, ``relations.jsonl`` is what the validator saw.
    """
    candidates = [
        ("aggregated", discovery_dir / "relations_aggregated.jsonl"),
        ("originals", discovery_dir / "relations.jsonl"),
    ]
    for layer, path in candidates:
        if path.is_file():
            return read_jsonl(path), layer
    raise FileNotFoundError(
        f"No relations file found in {discovery_dir} "
        f"(checked relations_aggregated.jsonl and relations.jsonl)"
    )


def _load_reviews(discovery_dir: Path) -> Dict[str, str]:
    """Return ``{relation_id: latest_decision}`` for the discovery run.

    The review store is append-only; the last decision per ``relation_id``
    wins. Missing file is fine — every relation just falls through to
    ``never_reviewed``.
    """
    path = discovery_dir / "relation_reviews.jsonl"
    if not path.is_file():
        return {}
    latest: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            rid = rec.get("relation_id")
            decision = rec.get("decision")
            if rid and decision:
                latest[str(rid)] = str(decision)
    return latest


def _filter_relations(
    relations: Sequence[Mapping[str, Any]],
    reviews: Mapping[str, str],
    *,
    review_filter: str,
    error_types: Optional[set[str]],
) -> List[Dict[str, Any]]:
    """Apply review-verdict and error-type filters.

    ``review_filter`` decides which verdicts survive:
      * ``approved`` (default) — only ``approved`` + ``added``
      * ``approved+unreviewed`` — same plus relations with no recorded
        verdict (useful for a first pass before any human review).
      * ``all`` — keep everything except explicit ``rejected`` / ``cleared``.
    """
    kept: List[Dict[str, Any]] = []
    for r in relations:
        sk = r.get("species_key") or ""
        verdict = reviews.get(sk)
        if review_filter == "approved":
            if verdict not in _APPROVE_VERDICTS:
                continue
        elif review_filter == "approved+unreviewed":
            if verdict not in (None, *_APPROVE_VERDICTS):
                continue
        elif review_filter == "all":
            if verdict in ("rejected", "cleared"):
                continue
        else:
            raise ValueError(f"unknown --review-filter={review_filter!r}")
        if error_types is not None and r.get("error_type") not in error_types:
            continue
        kept.append(dict(r))
    return kept


def _load_source_frames(
    csv_paths: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Load every CSV referenced by the discovery run, keyed by basename
    (matches the convention the validator / detector use)."""
    frames: Dict[str, pd.DataFrame] = {}
    for path_str in csv_paths:
        p = Path(path_str)
        if not p.is_file():
            logger.warning("source CSV missing, skipping: %s", p)
            continue
        try:
            df = load_csv(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load %s: %s", p, exc)
            continue
        frames[p.name] = df
    return frames


def _build_summary(
    *,
    args: argparse.Namespace,
    discovery_dir: Path,
    relation_layer: str,
    n_relations_total: int,
    n_relations_kept: int,
    review_filter: str,
    error_types: Optional[set[str]],
    csv_paths: Sequence[str],
    sources: Sequence[str],
    model: str,
    n_violations: int,
    n_code_violations: int,
    n_llm_violations: int,
    n_clusters: int,
    n_synthesised: int,
    enable_synthesis: bool,
) -> Dict[str, Any]:
    return {
        "mode": "single",  # downstream UI keys; this is a one-pass detection
        "model": model,
        "total_issues": n_violations,
        "total_clusters": n_clusters,
        "kind": "relation_detection",
        "config": {
            "mode": "single",
            "model": model,
            "csv_path": str(discovery_dir),
            "csv_paths": list(csv_paths),
            "sources": list(sources),
            "source_count": len(sources),
            "error_types": sorted(error_types) if error_types else None,
            "reasoning_effort": args.reasoning_effort,
            "concurrency": args.concurrency,
            "discovery_run": str(discovery_dir),
            "relation_layer": relation_layer,
            "review_filter": review_filter,
            "n_relations_total": n_relations_total,
            "n_relations_kept": n_relations_kept,
            "violations_min_pass_rate": args.violations_min_pass_rate,
            "violations_max_pass_rate": args.violations_max_pass_rate,
            "violations_max_per_rule": args.violations_max_per_rule,
            "violations_min_confidence": args.violations_min_confidence,
            "enable_llm_detect": not args.no_llm_detect,
            "enable_synthesis": enable_synthesis,
        },
        "violations": {
            "n_total": n_violations,
            "n_code": n_code_violations,
            "n_llm": n_llm_violations,
            "n_clusters": n_clusters,
            "n_synthesised": n_synthesised,
        },
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "discovery_run",
        type=Path,
        help="Step-1 run directory (must contain relations.jsonl and "
             "relation_verifications.jsonl).",
    )
    p.add_argument(
        "--config",
        default="config/experiment.yaml",
        help="Experiment config (for default prompts + model).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the new run dir. Default: outputs/<ts>_relation_detection/",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override the new run dir's basename (under --output-dir).",
    )
    p.add_argument(
        "--error-types",
        type=str,
        default=None,
        help="Comma-separated error_type keys to keep. Default: all.",
    )
    p.add_argument(
        "--review-filter",
        choices=("approved", "approved+unreviewed", "all"),
        default="approved",
        help=(
            "Which review verdicts to keep. Default: only 'approved'/'added'. "
            "'approved+unreviewed' also runs against relations the operator "
            "has not yet reviewed (useful for the first pass). 'all' keeps "
            "everything except explicitly 'rejected'/'cleared'."
        ),
    )
    p.add_argument("--model", default=None, help="Override llm.model.")
    p.add_argument(
        "--reasoning-effort", default=None,
        help="Override llm.reasoning_effort.",
    )
    p.add_argument(
        "--concurrency", type=int, default=None,
        help="LLM concurrency for the detect + synthesis steps.",
    )
    p.add_argument(
        "--detect-prompt",
        default="prompts/detect_relation_violation.yaml",
        help="LLM table-scan prompt YAML.",
    )
    p.add_argument(
        "--no-llm-detect", action="store_true",
        help="Skip the LLM table-scan path; only run the deterministic "
             "code-verifier re-execution.",
    )
    p.add_argument(
        "--no-synthesis", action="store_true",
        help="Skip the Phase-B cluster synthesis (no canonical_fix).",
    )
    p.add_argument(
        "--violations-min-pass-rate", type=float, default=0.90,
        help=(
            "Drop rules with pass_rate below this (default 0.90: a real "
            "rule holds on essentially every row and the few violations "
            "are the actual errors; a rule violated by >10%% of rows is "
            "not a real rule and its 'violations' are legitimate "
            "variation). Lower toward 0.5 for high-recall/low-precision."
        ),
    )
    p.add_argument(
        "--violations-max-pass-rate", type=float, default=1.0 - 1e-9,
        help="Drop rules at/above this pass_rate (default ~1.0).",
    )
    p.add_argument(
        "--violations-max-per-rule", type=int, default=200,
        help="Cap rows compiled per rule (default 200).",
    )
    p.add_argument(
        "--violations-min-confidence",
        choices=["low", "medium", "high"], default="medium",
    )
    p.add_argument(
        "--violations-timeout", type=float, default=8.0,
        help="Per-rule check() wall-clock cap (s).",
    )
    p.add_argument(
        "--llm-detect-max-rows", type=int, default=300,
        help="Row cap per relation handed to the LLM table-scan.",
    )
    p.add_argument(
        "--llm-detect-max-per-rule", type=int, default=50,
        help="Cap on violations emitted per relation by the LLM scan.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        level=logging.INFO,
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    args = _build_argparser().parse_args(argv)
    discovery_dir = args.discovery_run.expanduser().resolve()
    if not discovery_dir.is_dir():
        logger.error("discovery run dir not found: %s", discovery_dir)
        return 2

    cfg_path = _resolve(args.config, PROJECT_ROOT)
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    llm_cfg = cfg.get("llm", {}) or {}
    prompts_cfg = cfg.get("prompts", {}) or {}
    output_cfg = cfg.get("output", {}) or {}

    model = args.model or llm_cfg.get("model", "gpt-5.2")
    reasoning_effort = (
        args.reasoning_effort or llm_cfg.get("reasoning_effort", "medium")
    )
    reasoning_summary = llm_cfg.get("reasoning_summary", "detailed")

    # Resolve output dir. Default mirrors run_cleaning_detection's
    # ``outputs/<UTC timestamp>_<model>_relation_detection`` shape so the
    # labeling UI's run list sorts these alongside cleaning runs.
    out_root = (
        args.output_dir
        or _resolve(output_cfg.get("directory", "outputs"), PROJECT_ROOT)
    )
    out_root = Path(out_root).expanduser().resolve()
    if args.run_name:
        run_dir = out_root / args.run_name
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_model = model.replace("/", "_")[:40]
        run_dir = out_root / f"{ts}_{safe_model}_relation_detection"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_status(run_dir, "in_progress")
    logger.info("Run dir: %s", run_dir)

    try:
        taxonomy_path = _resolve(
            prompts_cfg.get("error_types_file", "prompts/error_types.yaml"),
            PROJECT_ROOT,
        )
        taxonomy = load_taxonomy(taxonomy_path)

        # ---- Load relations + reviews + verifications -------------------
        relations_all, layer = _load_relations(discovery_dir)
        reviews = _load_reviews(discovery_dir)
        wanted_types: Optional[set[str]] = (
            {s.strip() for s in args.error_types.split(",") if s.strip()}
            if args.error_types else None
        )
        relations = _filter_relations(
            relations_all, reviews,
            review_filter=args.review_filter,
            error_types=wanted_types,
        )
        logger.info(
            "Loaded %d %s relation(s); %d survive filter "
            "(review_filter=%s, error_types=%s)",
            len(relations_all), layer, len(relations),
            args.review_filter,
            sorted(wanted_types) if wanted_types else "all",
        )
        if not relations:
            logger.warning("No relations survived filtering; exiting.")
            _write_run_status(run_dir, "completed", detail="no relations to detect")
            return 0

        ver_path = discovery_dir / "relation_verifications.jsonl"
        if not ver_path.is_file():
            logger.error(
                "relation_verifications.jsonl missing in %s — run validation first",
                discovery_dir,
            )
            _write_run_status(run_dir, "failed", detail="no verifications")
            return 2
        verifications_all = read_jsonl(ver_path)
        kept_species = {r.get("species_key") for r in relations if r.get("species_key")}
        verifications = [
            v for v in verifications_all
            if v.get("species_key") in kept_species
        ]
        verifications_by_key = {
            str(v["species_key"]): v
            for v in verifications
            if v.get("species_key")
        }
        relations_by_species = {
            str(r["species_key"]): r for r in relations if r.get("species_key")
        }

        # ---- Source frames ---------------------------------------------
        # Prefer csv_paths from the discovery run's summary.json; we don't
        # re-discover sources here because the validator already ran
        # against this exact set.
        summary_path = discovery_dir / "summary.json"
        csv_paths: List[str] = []
        if summary_path.is_file():
            try:
                discovery_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                csv_paths = list(
                    discovery_summary.get("csv_paths")
                    or (discovery_summary.get("config") or {}).get("csv_paths")
                    or []
                )
            except Exception:  # noqa: BLE001
                logger.warning("could not parse discovery summary.json")
        if not csv_paths:
            logger.error(
                "no csv_paths found in %s/summary.json", discovery_dir,
            )
            _write_run_status(run_dir, "failed", detail="no csv_paths")
            return 2

        source_frames = _load_source_frames(csv_paths)
        # Restrict to sources the kept relations touch.
        used_sources = {r.get("source") for r in relations if r.get("source")}
        source_frames = {
            k: v for k, v in source_frames.items() if k in used_sources
        }
        logger.info(
            "Loaded %d source frame(s) covering %d relation(s)",
            len(source_frames), len(relations),
        )

        # ---- LLM client + detection prompt -----------------------------
        llm = LLMClient()
        detect_prompt_path = _resolve(args.detect_prompt, PROJECT_ROOT)

        # ---- Compile code-path violations ------------------------------
        # The compiler now emits both code-verifier rows and
        # needs-review placeholders in a single pass; rules outside
        # the pass-rate keep band are rejected and don't appear in
        # ``violations``.
        violations, skip_log = compile_code_violations(
            relations_by_species=relations_by_species,
            verifications=verifications,
            source_frames=source_frames,
            min_pass_rate=args.violations_min_pass_rate,
            max_pass_rate=args.violations_max_pass_rate,
            timeout_seconds=args.violations_timeout,
        )
        n_code_violations = len(violations)
        logger.info(
            "Code path: %d violation(s) across %d rule(s); skipped %d rule(s)",
            n_code_violations,
            len({v.relation_species_key for v in violations}),
            len(skip_log),
        )

        # ---- LLM table-scan path --------------------------------------
        n_llm_violations = 0
        if not args.no_llm_detect:
            if not detect_prompt_path.is_file():
                logger.warning(
                    "detect prompt missing at %s; skipping LLM path",
                    detect_prompt_path,
                )
            else:
                prompt_cfg = load_detect_prompt(detect_prompt_path)
                llm_violations = detect_violations_for_relations(
                    relations,
                    verifications=verifications_by_key,
                    source_frames=source_frames,
                    llm=llm,
                    model=model,
                    prompt_cfg=prompt_cfg,
                    max_rows_per_call=args.llm_detect_max_rows,
                    max_violations_per_rule=args.llm_detect_max_per_rule,
                    min_pass_rate=args.violations_min_pass_rate,
                    max_pass_rate=args.violations_max_pass_rate,
                )
                n_llm_violations = len(llm_violations)
                violations.extend(llm_violations)
                logger.info(
                    "LLM table-scan: %d violation(s)", n_llm_violations,
                )

        # Persist raw violations regardless of cluster outcome.
        write_jsonl(
            run_dir / "relation_violations.jsonl",
            [v.as_dict() for v in violations],
        )

        # ---- Flat issues + clustering ---------------------------------
        flat_issues = list(relation_violations_to_flat_issues(violations))
        clusters = cluster_issues(flat_issues)
        logger.info(
            "Clustering: %d issue(s) -> %d cluster(s)",
            len(flat_issues), len(clusters),
        )

        # ---- Phase B synthesis (canonical_fix) ------------------------
        n_synthesised = 0
        enable_synthesis = (not args.no_synthesis) and bool(clusters)
        if enable_synthesis:
            synthesis_prompt_path = _resolve(
                prompts_cfg.get(
                    "synthesis_prompt_file",
                    "prompts/synthesise_cluster.yaml",
                ),
                PROJECT_ROOT,
            )
            synthesis_prompt = load_synthesis_prompt(synthesis_prompt_path)
            synthesiser = ClusterSynthesiser(
                llm=llm,
                taxonomy=taxonomy,
                prompt=synthesis_prompt,
                model=model,
                reasoning_effort=(
                    args.reasoning_effort
                    or llm_cfg.get("synthesis_effort", reasoning_effort)
                ),
                reasoning_summary=reasoning_summary,
                concurrency=args.concurrency,
                include_singletons=True,
            )
            clusters = synthesiser.run(clusters)
            n_synthesised = sum(
                1 for c in clusters
                if c.synthesis and c.synthesis.canonical_fix not in (None, "")
            )
            logger.info(
                "Synthesis: %d cluster(s) gained canonical_fix",
                n_synthesised,
            )

        # ---- Write standard cleaning-detection artifacts --------------
        cluster_dicts = [c.as_dict() for c in clusters]
        write_jsonl(run_dir / "issues.jsonl", flat_issues)
        write_json(run_dir / "issues.json", flat_issues)
        write_jsonl(run_dir / "clusters.jsonl", cluster_dicts)
        write_json(run_dir / "clusters.json", cluster_dicts)

        summary = _build_summary(
            args=args,
            discovery_dir=discovery_dir,
            relation_layer=layer,
            n_relations_total=len(relations_all),
            n_relations_kept=len(relations),
            review_filter=args.review_filter,
            error_types=wanted_types,
            csv_paths=csv_paths,
            sources=sorted(source_frames.keys()),
            model=model,
            n_violations=len(violations),
            n_code_violations=n_code_violations,
            n_llm_violations=n_llm_violations,
            n_clusters=len(clusters),
            n_synthesised=n_synthesised,
            enable_synthesis=enable_synthesis,
        )
        write_json(run_dir / "summary.json", summary)

        _write_run_status(run_dir, "completed")
        logger.info(
            "Done. %d violation(s) -> %d cluster(s) (%d with canonical_fix). %s",
            len(violations), len(clusters), n_synthesised, run_dir,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("relation detection failed")
        _write_run_status(run_dir, "failed", detail=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
