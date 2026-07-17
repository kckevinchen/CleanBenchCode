"""v7 wrapper around the v5 compare_ft_vs_base evaluator.

Same as v6's wrapper but with the v7 FT models added. Deployment
names follow the pattern ``o4-mini-v7-{r1,r2,r3}``.
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

logger = logging.getLogger("compare_ft_vs_base_v7")

DEFAULT_EVAL = "RFT/typo/v7/data/eval.jsonl"

DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(
        label="base",
        endpoint=_configured_endpoint(0),
        deployment="o4-mini",
    ),
    # v6 best FT (kept for direct comparison vs v7)
    ModelSpec(
        label="v6-baseline",
        endpoint=_configured_endpoint(2),
        deployment="o4-mini-v6-baseline",
    ),
    # v7 FT models (deployment names follow the convention used by
    # earlier v6 deployments). Created on the SAME endpoint the
    # corresponding RFT job ran on.
    ModelSpec(
        label="v7-r1",
        endpoint=_configured_endpoint(0),
        deployment="o4-mini-v7-r1",
    ),
    ModelSpec(
        label="v7-r2",
        endpoint=_configured_endpoint(1),
        deployment="o4-mini-v7-r2",
    ),
    ModelSpec(
        label="v7-r3",
        endpoint=_configured_endpoint(2),
        deployment="o4-mini-v7-r3",
    ),
    ModelSpec(
        label="gpt-5.2",
        endpoint=_configured_endpoint(0),
        deployment="gpt-5.2",
        kind="responses",
    ),
    ModelSpec(
        label="gpt-5.5",
        endpoint=_configured_endpoint(0),
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
    p.add_argument("--out-dir", default="outputs/v7_compare", type=Path)
    p.add_argument(
        "--grader-models",
        default="gpt-5.2,gpt-5.4,gpt-5.5",
    )
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--only", default=None,
        help="Comma-separated subset of labels (e.g. base,v7-r1,v7-r2).",
    )
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
