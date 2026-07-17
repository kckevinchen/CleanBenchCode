"""gpt-5.5 reasoning=none against the v8 eval rows BUT using the v7 prompt."""

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
        label="gpt-5.5-v7prompt",
        endpoint=_configured_endpoint(0),
        deployment="gpt-5.5",
        kind="responses",
    ),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(
        models=MODELS,
        eval_path=Path("RFT/typo/v8/data/eval_v7prompt.jsonl"),
        out_dir=Path("outputs/v8_compare_gpt55_v7prompt"),
        grader_models="gpt-5.2,gpt-5.4,gpt-5.5",
        reasoning_effort="none",
        limit=None,
        workers=8,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
