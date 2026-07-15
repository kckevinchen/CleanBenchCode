"""gpt-5.5 reasoning=none against the v8 eval rows BUT using the r4 prompt
(from RFT/typo/v7/eval/sweep_trainee_prompt.py)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from RFT.typo.v5.eval.compare_ft_vs_base import ModelSpec, run  # noqa: E402


MODELS = [
    ModelSpec(
        label="gpt-5.5-r4",
        endpoint="https://dmxaz-test-aoai-eu2.openai.azure.com/",
        deployment="gpt-5.5",
        kind="responses",
    ),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(
        models=MODELS,
        eval_path=Path("RFT/typo/v8/data/eval_r4prompt.jsonl"),
        out_dir=Path("outputs/v8_compare_gpt55_r4"),
        grader_models="gpt-5.2,gpt-5.4,gpt-5.5",
        reasoning_effort="none",
        limit=None,
        workers=8,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
