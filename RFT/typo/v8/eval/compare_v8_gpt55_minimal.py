"""Run the v5 eval harness against gpt-5.5 with reasoning=minimal.

Same eval.jsonl as compare_v8.py, but only one trainee model
(gpt-5.5 via the Responses API at the lowest reasoning level).
Results go to outputs/v8_compare_gpt55_min/ so they don't clobber
the o4-mini comparison.
"""

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
        label="gpt-5.5-min",
        endpoint="https://dmxaz-test-aoai-eu2.openai.azure.com/",
        deployment="gpt-5.5",
        kind="responses",
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
        out_dir=Path("outputs/v8_compare_gpt55_min"),
        grader_models="gpt-5.2,gpt-5.4,gpt-5.5",
        reasoning_effort="none",
        limit=None,
        workers=8,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
