"""Run Phase-B synthesis (canonical_fix / canonical_explanation) over the
clusters in an existing relation-discovery run dir.

The relation pipeline (:mod:`experiments.run_relation_discovery`) emits
``issues.jsonl`` and ``clusters.jsonl`` but skips the LLM synthesis step
that adds ``synthesis.canonical_fix`` to each cluster. This helper does
exactly that: re-runs ``cluster_issues`` over the existing flat issues,
optionally filters by error type, and fires ``ClusterSynthesiser``.

The output is written to ``clusters_synthesised.jsonl`` /
``clusters_synthesised.json`` so the original ``clusters.jsonl`` from the
discovery run is not overwritten.

Example::

    python -m experiments.synthesise_relation_clusters \\
        outputs/20260515T235305Z_gpt-5.2,gpt-5.4,gpt-5.5_relations_adaptive \\
        --error-types row_calculation_inconsistency,string_calculation_inconsistency \\
        --concurrency 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.core.aggregator import (  # noqa: E402
    Cluster,
    ClusterSynthesiser,
    cluster_issues,
    load_synthesis_prompt,
)
from src.core.error_taxonomy import load_taxonomy  # noqa: E402
from src.core.jsonl_io import read_jsonl, write_jsonl  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402
from src.core.report_builder import write_json  # noqa: E402

logger = logging.getLogger("synthesise_relation_clusters")


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p)


def _filter_clusters(
    clusters: Sequence[Cluster], wanted: Optional[set[str]],
) -> List[Cluster]:
    if not wanted:
        return list(clusters)
    return [c for c in clusters if any(et in wanted for et in c.error_types)]


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        level=logging.INFO,
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "run_dir",
        type=Path,
        help="Relation-discovery run directory containing issues.jsonl.",
    )
    p.add_argument(
        "--config",
        default="config/experiment.yaml",
        help="Experiment config for model + prompt paths.",
    )
    p.add_argument(
        "--error-types",
        default=None,
        help=(
            "Comma-separated error_type keys to keep. Default: all "
            "clusters in the run dir."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override llm.model from the config.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Concurrency for the synthesis pass.",
    )
    p.add_argument(
        "--reasoning-effort",
        default=None,
        help="Override llm.synthesis_effort from the config.",
    )
    p.add_argument(
        "--include-singletons",
        dest="include_singletons",
        action="store_true",
        default=True,
        help="Synthesise single-detection clusters too (default).",
    )
    p.add_argument(
        "--no-include-singletons",
        dest="include_singletons",
        action="store_false",
        help="Restrict synthesis to multi-type clusters only.",
    )
    args = p.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        logger.error("run_dir not found: %s", run_dir)
        return 2

    issues_path = run_dir / "issues.jsonl"
    if not issues_path.is_file():
        logger.error("issues.jsonl missing in %s", run_dir)
        return 2

    cfg_path = _resolve(args.config, PROJECT_ROOT)
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    llm_cfg = cfg.get("llm", {}) or {}
    prompts_cfg = cfg.get("prompts", {}) or {}

    taxonomy_path = _resolve(
        prompts_cfg.get("error_types_file", "prompts/error_types.yaml"),
        PROJECT_ROOT,
    )
    taxonomy = load_taxonomy(taxonomy_path)

    synthesis_prompt_file = _resolve(
        prompts_cfg.get(
            "synthesis_prompt_file", "prompts/synthesise_cluster.yaml"
        ),
        PROJECT_ROOT,
    )
    synthesis_prompt = load_synthesis_prompt(synthesis_prompt_file)

    model = args.model or llm_cfg.get("model", "gpt-5.2")
    reasoning_effort = (
        args.reasoning_effort
        or llm_cfg.get("synthesis_effort")
        or llm_cfg.get("reasoning_effort", "medium")
    )
    reasoning_summary = llm_cfg.get("reasoning_summary", "detailed")

    wanted = None
    if args.error_types:
        wanted = {s.strip() for s in args.error_types.split(",") if s.strip()}
        logger.info("Filtering to error types: %s", sorted(wanted))

    flat_issues = read_jsonl(issues_path)
    logger.info("Loaded %d flat issue(s) from %s", len(flat_issues), issues_path)

    clusters = cluster_issues(flat_issues)
    logger.info("Re-clustered into %d cluster(s)", len(clusters))

    clusters = _filter_clusters(clusters, wanted)
    logger.info(
        "After error-type filter: %d cluster(s) to synthesise", len(clusters),
    )
    if not clusters:
        logger.info("Nothing to synthesise; exiting.")
        return 0

    llm = LLMClient()
    synthesiser = ClusterSynthesiser(
        llm=llm,
        taxonomy=taxonomy,
        prompt=synthesis_prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        concurrency=args.concurrency,
        include_singletons=args.include_singletons,
    )
    clusters = synthesiser.run(clusters)

    out_jsonl = run_dir / "clusters_synthesised.jsonl"
    out_json = run_dir / "clusters_synthesised.json"
    payload = [c.as_dict() for c in clusters]
    write_jsonl(out_jsonl, payload)
    write_json(out_json, payload)
    n_with_fix = sum(
        1 for c in clusters
        if c.synthesis and c.synthesis.canonical_fix not in (None, "")
    )
    logger.info(
        "Wrote %d cluster(s) -> %s (%d with canonical_fix)",
        len(clusters), out_jsonl.name, n_with_fix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
