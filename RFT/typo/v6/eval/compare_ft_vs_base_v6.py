"""v6 wrapper around the v5 compare_ft_vs_base evaluator.

Only the model registry and default eval set change; everything else
(graders, scoring, P/R/F1 logic) is reused unchanged from
``RFT.typo.v5.eval.compare_ft_vs_base``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v5.eval.compare_ft_vs_base import ModelSpec, run  # noqa: E402

logger = logging.getLogger("compare_ft_vs_base_v6")

DEFAULT_EVAL = "RFT/typo/v6/data/eval.jsonl"

DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(
        label="base",
        endpoint="https://dmxaz-test-aoai-eu2.openai.azure.com/",
        deployment="o4-mini",
    ),
    ModelSpec(
        label="longer",
        endpoint="https://dmxaz-test-aoai-eu2.openai.azure.com/",
        deployment="o4-mini-v6-longer",
    ),
    ModelSpec(
        label="sharper",
        endpoint="https://yeyehe-aoai.openai.azure.com/",
        deployment="o4-mini-v6-sharper",
    ),
    ModelSpec(
        label="baseline",
        endpoint="https://sc-rf-ma30hqbc-eastus2.openai.azure.com/",
        deployment="o4-mini-v6-baseline",
    ),
    ModelSpec(
        label="deep",
        endpoint="https://dmxaz-test-aoai-eu2.openai.azure.com/",
        deployment="o4-mini-v6-deep",
    ),
    ModelSpec(
        label="gpt-5.2",
        endpoint="https://dmxaz-test-aoai-eu2.openai.azure.com/",
        deployment="gpt-5.2",
        kind="responses",
    ),
    ModelSpec(
        label="gpt-5.5",
        endpoint="https://dmxaz-test-aoai-eu2.openai.azure.com/",
        deployment="gpt-5.5",
        kind="responses",
    ),
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", default=DEFAULT_EVAL, type=Path)
    p.add_argument("--out-dir", default="outputs/v6_compare", type=Path)
    p.add_argument(
        "--grader-models",
        default="gpt-5.2,gpt-5.4,gpt-5.5",
    )
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of labels (e.g. base,longer,sharper).")
    args = p.parse_args(argv)

    models = list(DEFAULT_MODELS)
    if args.only:
        keep = set(args.only.split(","))
        models = [m for m in models if m.label in keep]
        if "base" not in keep:
            logger.warning("base excluded -- delta-vs-base columns will be omitted.")

    run(
        models=models,
        eval_path=args.eval,
        out_dir=args.out_dir,
        grader_models=args.grader_models,
        reasoning_effort=args.reasoning_effort,
        limit=args.limit,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
