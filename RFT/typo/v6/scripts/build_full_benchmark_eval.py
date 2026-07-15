"""Build a single eval file containing EVERY column from the human benchmark.

Used to run a model (e.g. gpt-5.5) against the *full* labelled benchmark
rather than just the 30 % v6 eval split.

Output is in the same per-column JSONL format as ``RFT/typo/v6/data/eval.jsonl``
so it can be passed directly to ``RFT/typo/v6/eval/compare_ft_vs_base_v6.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v5 import _task as _typo_v5_task  # noqa: E402
from RFT.typo.v5.scripts.split_typo_benchmark_v5 import (  # noqa: E402
    _aggregate,
    _build_csv_index,
)

logger = logging.getLogger("build_full_benchmark_eval")

DEFAULT_BENCHMARK = Path("benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl")
DEFAULT_CSV_ROOT = Path("data/benchmark_sources_v6")
DEFAULT_OUTPUT = Path("RFT/typo/v6/data/eval_full.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--max-distinct", type=int, default=500)
    ap.add_argument("--english-only", action="store_true", default=True)
    ap.add_argument("--english-threshold", type=float, default=0.6)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    csv_index = _build_csv_index(args.csv_root)
    rows = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("Loaded %d benchmark rows", len(rows))

    items, stats = _aggregate(
        rows, csv_index,
        max_distinct=args.max_distinct,
        english_only=args.english_only,
        english_threshold=args.english_threshold,
    )
    logger.info("Aggregated %d columns (typo=%d, amb_only=%d, clean=%d)",
                stats["columns_emitted"],
                stats["columns_with_gold_typos"],
                stats["columns_with_only_ambiguous"],
                stats["columns_pure_clean"])

    items.sort(key=lambda it: (it["source"], it["column"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for it in items:
            rec = _typo_v5_task.build_record(it)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Wrote %d records to %s", n, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
