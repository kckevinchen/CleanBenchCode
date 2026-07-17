"""Compare the v10 RFT fine-tuned models against base o4-mini + GPT-5 baselines.

Thin wrapper around v5's compare_ft_vs_base evaluator (graders, P/R/F1 logic
reused unchanged). Only the model registry + default eval set change. Eval rows
bake PROMPT_V10 as their developer message, so every model runs with the v10
prompt.

The three v10 RFT arms differ only in reward:
  * v10-r1        -- gold-based grader (reference).
  * v10-nogold    -- gold-free precision + trivial only (degenerate: empty
                     output scored max during training).
  * v10-nogoldrec -- gold-free F1(LLM precision, LLM recall) judge; the empty
                     -output degeneracy is removed (see eval/judge_pr_check.py).

Headline metric is the canonical cell-level micro-F1 (RFT.md §3): sum tp/fp/fn
across rows of each model's ``*.rollouts.jsonl``.

    python -m RFT.typo.v10.eval.compare_ft_vs_base_v10 --reasoning-effort medium
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v5.eval.compare_ft_vs_base import (  # noqa: E402
    ModelSpec,
    _configured_endpoint,
    run,
)

logger = logging.getLogger("compare_ft_vs_base_v10")

DEFAULT_EVAL = "RFT/typo/v10/data/eval.jsonl"

PRIMARY_ENDPOINT = _configured_endpoint(0)

DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(label="base", endpoint=PRIMARY_ENDPOINT, deployment="o4-mini"),
    ModelSpec(label="v10-r1", endpoint=PRIMARY_ENDPOINT, deployment="o4-mini-v10-r1"),
    ModelSpec(label="v10-nogold", endpoint=PRIMARY_ENDPOINT, deployment="o4-mini-v10-nogold"),
    ModelSpec(label="v10-nogoldrec", endpoint=PRIMARY_ENDPOINT,
              deployment="o4-mini-v10-nogoldrec"),
    # NOTE: GPT-5 baselines are evaluated at reasoning_effort=none (run them in a
    # separate pass: --only gpt-5.5,gpt-5.2 --reasoning-effort none). The o4-mini
    # base + RFT models use their trained effort (medium).
    ModelSpec(label="gpt-5.2", endpoint=PRIMARY_ENDPOINT, deployment="gpt-5.2", kind="responses"),
    ModelSpec(label="gpt-5.5", endpoint=PRIMARY_ENDPOINT, deployment="gpt-5.5", kind="responses"),
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", default=DEFAULT_EVAL, type=Path)
    p.add_argument("--out-dir", default="outputs/v10_compare", type=Path)
    p.add_argument("--grader-models", default="gpt-5.2,gpt-5.4,gpt-5.5")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of labels (e.g. base,v10-r1).")
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
