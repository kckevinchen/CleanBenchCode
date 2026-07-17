"""Run the v5 eval harness against the v8 deployments.

Compares:
    - base       : o4-mini (no FT)
    - v8-r1-low  : o4-mini-v8-r1-low  (r1 reward, 2 epochs, low reasoning)
    - v8-slim    : o4-mini-v8-slim (slim 2-grader reward, 2 epochs)

Uses RFT/typo/v8/data/eval.jsonl.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from RFT.typo.v5.eval.compare_ft_vs_base import (  # noqa: E402
    ModelSpec,
    _configured_endpoint,
    run,
)

MODELS = [
    ModelSpec(
        label="base",
        endpoint=_configured_endpoint(0),
        deployment="o4-mini",
    ),
    ModelSpec(
        label="v8-r1-low",
        endpoint=_configured_endpoint(0),
        deployment="o4-mini-v8-r1-low",
    ),
    ModelSpec(
        label="v8-slim",
        endpoint=_configured_endpoint(1),
        deployment="o4-mini-v8-slim",
    ),
]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(
        models=MODELS,
        eval_path=Path("RFT/typo/v8/data/eval.jsonl"),
        out_dir=Path("outputs/v8_compare"),
        grader_models="gpt-5.2,gpt-5.4,gpt-5.5",
        reasoning_effort="medium",
        limit=None,
        workers=8,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
