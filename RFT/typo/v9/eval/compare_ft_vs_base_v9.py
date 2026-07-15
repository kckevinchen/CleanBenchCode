"""Compare the v9 RFT fine-tuned models against base o4-mini + GPT-5 baselines.

Wrapper around v5's compare_ft_vs_base evaluator (graders, P/R/F1 logic reused
unchanged). Only the model registry + default eval set change. Eval rows bake
PROMPT_V9 as their developer message, so every model is run with the v9 prompt.

Headline metric is the canonical cell-level micro-F1 (RFT.md §3): sum tp/fp/fn
across rows of each model's ``*.rollouts.jsonl``.

    python -m RFT.typo.v9.eval.compare_ft_vs_base_v9 --reasoning-effort medium
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

logger = logging.getLogger("compare_ft_vs_base_v9")

DEFAULT_EVAL = "RFT/typo/v9/data/eval.jsonl"

EU2 = "https://dmxaz-test-aoai-eu2.openai.azure.com/"
YEYEHE = "https://yeyehe-aoai.openai.azure.com/"
MA30 = "https://sc-rf-ma30hqbc-eastus2.cognitiveservices.azure.com/"
M8UD = "https://sc-rf-m8udnqmj-swedencentral.cognitiveservices.azure.com/"
DMXAZ = "https://dmxaz-test-aoai.openai.azure.com/"

DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(label="base", endpoint=EU2, deployment="o4-mini"),
    ModelSpec(label="v9-r1-e2", endpoint=EU2, deployment="o4-mini-v9-r1-e2"),
    ModelSpec(label="v9-r1-e3", endpoint=M8UD, deployment="o4-mini-v9-r1-e3"),
    ModelSpec(label="v9-r2-base", endpoint=YEYEHE, deployment="o4-mini-2025-04-16-v9-r2-base"),  # user's Developer-tier deployment
    ModelSpec(label="v9-r3-base", endpoint=MA30, deployment="o4-mini-v9-r3-base"),
    ModelSpec(label="v9-r1-lr2", endpoint=M8UD, deployment="o4-mini-v9-r1-lr2"),
    # v9-r1-base: trained on M8UD (ftjob-bfbd...c32d89), deployed Standard tier
    # (GlobalStandard finetune quota there is full from v9-r1-lr2).
    ModelSpec(label="v9-r1-base", endpoint=M8UD, deployment="o4-mini-v9-r1-base"),
    # prior-version FT model, redeployed for comparison (runs with PROMPT_V8):
    ModelSpec(label="v8-r1-low", endpoint=EU2, deployment="o4-mini-v8-r1-low"),
    # NOTE: GPT-5 baselines are evaluated at reasoning_effort=none (run them in a
    # separate pass: --only gpt-5.5,gpt-5.2 --reasoning-effort none). The o4-mini
    # base + RFT models use their trained effort (medium).
    ModelSpec(label="gpt-5.2", endpoint=EU2, deployment="gpt-5.2", kind="responses"),
    ModelSpec(label="gpt-5.5", endpoint=EU2, deployment="gpt-5.5", kind="responses"),
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", default=DEFAULT_EVAL, type=Path)
    p.add_argument("--out-dir", default="outputs/v9_compare", type=Path)
    p.add_argument("--grader-models", default="gpt-5.2,gpt-5.4,gpt-5.5")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of labels (e.g. base,v9-r2-base).")
    args = p.parse_args(argv)

    models = list(DEFAULT_MODELS)
    if args.only:
        keep = set(args.only.split(","))
        models = [m for m in models if m.label in keep]
        if "base" not in keep:
            logger.warning("base excluded -- delta-vs-base columns omitted.")

    run(models=models, eval_path=args.eval, out_dir=args.out_dir,
        grader_models=args.grader_models, reasoning_effort=args.reasoning_effort,
        limit=args.limit, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
