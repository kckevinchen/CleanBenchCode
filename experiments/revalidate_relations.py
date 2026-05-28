"""Re-run the empirical-validation step of an existing relation
discovery run, after prompt/sandbox fixes. Reuses the saved
``relations.jsonl`` (the LLM discovery output -- the expensive part of
the pipeline) and re-issues only the cheap validation LLM calls plus
the no-LLM violation compilation. Outputs overwrite
``relation_verifications.jsonl``, ``relation_violations.jsonl`` and
``clusters.*`` in place; ``summary.json`` is patched with the fresh
sub-blocks.

Use this when you've changed the validator prompt
(``prompts/validate_relation.yaml``), the sandbox whitelist
(``src/core/relation_validator.py``), the violation compiler, or the
needs-review placeholder shape -- anywhere downstream of discovery.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import load_csv  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402
from experiments.run_relation_discovery import (  # noqa: E402
    _run_validation_pass,
    _run_violation_compilation_pass,
)


logger = logging.getLogger("revalidate_relations")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--csv-dir", required=True)
    p.add_argument(
        "--prompt",
        default="prompts/validate_relation.yaml",
        help="Validator prompt YAML (default: prompts/validate_relation.yaml).",
    )
    p.add_argument(
        "--model",
        default="gpt-5.2,gpt-5.4,gpt-5.5",
        help="LLM model spec passed to the validator (same routing the "
             "original discovery run used).",
    )
    p.add_argument(
        "--layer",
        default="union",
        choices=["auto", "union", "originals", "aggregated"],
        help="Which relation layer to validate. 'union' (default) "
             "covers aggregated canonicals + singleton originals "
             "(rules with no paraphrase cluster, which the 'aggregated' "
             "and 'auto' layers silently drop).",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--sample-rows", type=int, default=20)
    p.add_argument("--validate-timeout", type=float, default=8.0)
    p.add_argument("--validate-max-violations", type=int, default=20)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Validate only the first N relations (debug knob).",
    )
    p.add_argument("--min-pass-rate", type=float, default=0.5)
    p.add_argument("--max-pass-rate", type=float, default=1.0 - 1e-9)
    p.add_argument("--violation-timeout", type=float, default=8.0)
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation and only re-run violation compilation "
             "against the existing relation_verifications.jsonl.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_dir = Path(args.run_dir).resolve()
    csv_dir = Path(args.csv_dir).resolve()

    if not (run_dir / "relations.jsonl").is_file():
        logger.error("no relations.jsonl in %s; nothing to revalidate", run_dir)
        return 1

    # ---- Load source frames ----------------------------------------
    sources_path = run_dir / "sources.json"
    if not sources_path.is_file():
        logger.error("no sources.json in %s", run_dir)
        return 1
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    source_names = {s["source"] for s in sources}
    source_frames: Dict[str, pd.DataFrame] = {}
    for csv_path in csv_dir.rglob("*.csv"):
        if csv_path.name not in source_names:
            continue
        try:
            source_frames[csv_path.name] = load_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip %s: %s", csv_path, exc)
    logger.info(
        "loaded %d/%d source frames",
        len(source_frames), len(source_names),
    )

    # ---- Validation pass -------------------------------------------
    if not args.skip_validation:
        llm = LLMClient()
        prompt_path = (PROJECT_ROOT / args.prompt).resolve()
        val_stats: Optional[Dict] = _run_validation_pass(
            run_dir=run_dir,
            source_frames=source_frames,
            llm=llm,
            model=args.model,
            prompt_path=prompt_path,
            layer=args.layer,
            concurrency=args.concurrency,
            sample_rows=args.sample_rows,
            timeout_seconds=args.validate_timeout,
            max_violation_samples=args.validate_max_violations,
            limit=args.limit,
        )
        logger.info("validation stats: %s", val_stats)
    else:
        val_stats = None
        logger.info("validation pass skipped (--skip-validation)")

    # ---- Violation compilation -------------------------------------
    vio_stats = _run_violation_compilation_pass(
        run_dir=run_dir,
        source_frames=source_frames,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
        timeout_seconds=args.violation_timeout,
    )

    # ---- Patch summary.json ----------------------------------------
    summary_out = run_dir / "summary.json"
    if summary_out.is_file():
        try:
            summary_doc = json.loads(summary_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("summary.json is not valid JSON; skipping merge")
            summary_doc = None
        if isinstance(summary_doc, dict):
            if val_stats is not None:
                summary_doc["validation"] = val_stats
            if vio_stats is not None:
                summary_doc["violations"] = vio_stats
            summary_out.write_text(
                json.dumps(summary_doc, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("updated %s", summary_out)

    # ---- Refresh report.html ---------------------------------------
    # Mirrors what ``run_relation_discovery`` and
    # ``run_relation_validation`` do at the end of their pipelines:
    # re-render the static dashboard so the new pass-rates / violation
    # counts surface without a manual rebuild step. Non-fatal on
    # failure - the JSONL artifacts on disk are the source of truth.
    try:
        from experiments.build_relation_report import build_relation_report  # noqa: WPS433
        template_path = PROJECT_ROOT / "frontend" / "relations_template.html"
        if template_path.is_file():
            out = build_relation_report(run_dir, template_path=template_path)
            logger.info("dashboard refreshed: %s", out)
        else:
            logger.warning("dashboard template missing: %s", template_path)
    except Exception as exc:  # noqa: BLE001 - non-fatal
        logger.warning("dashboard refresh failed: %s", exc)

    print(json.dumps({
        "validation": val_stats,
        "violations": vio_stats,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
