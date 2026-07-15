"""Grader-only sanity test.

Construct a *perfect* synthetic trainee output for every column in
the strict-filter benchmark and score it with the v5 multigrader.

A perfect output is::

    {"typos": [{"value": v, "correction": c, "flavour": "spelling",
                "rationale": "benchmark gold"} for v, c in gold]}

If the grader is correctly implemented the total score should be
5.0 on every column. Anything below ~4.5 is a grader bug or
borderline judge wording in the score-model leg.

This avoids any trainee-model variance and isolates the grader.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v5 import _task as t  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402

# Reuse helpers from the full grader test driver.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_grader_on_benchmark import (  # noqa: E402
    _build_embedded_grade, load_kept_columns, run_score_model_leg,
)

logger = logging.getLogger("test_grader_only")


def perfect_output(gold_typos: list[dict]) -> str:
    """Emit the trainee output a perfect model would produce."""
    typos = [
        {
            "value": g["value"],
            "correction": g["correction"],
            "flavour": "spelling",
            "rationale": "benchmark gold",
        }
        for g in gold_typos
    ]
    return json.dumps({"typos": typos}, ensure_ascii=False)


def empty_output() -> str:
    return json.dumps({"typos": []})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",
                        default="benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl",
                        type=Path)
    parser.add_argument("--csv-root",
                        default="data/source_inconsistency_bing21",
                        type=Path)
    parser.add_argument("--out-dir",
                        default=Path("outputs/grader_only_test"),
                        type=Path)
    parser.add_argument("--grader-model",
                        default="gpt-5.2,gpt-5.4,gpt-5.5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["perfect", "empty"],
                        default="perfect",
                        help="perfect: feed canonical gold (expect 5.0); "
                        "empty: feed {} (expect 5.0 on G=0, low otherwise).")
    parser.add_argument("--skip-sm", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading + filtering benchmark...")
    items = load_kept_columns(args.benchmark, args.csv_root)
    logger.info("Kept %d columns after strict filter.", len(items))
    if args.limit:
        items = items[: args.limit]

    records = []
    for it in items:
        rec = t.build_record(it)
        rec["_n_gold_cells"] = it["n_gold_cells"]
        rec["_n_clean_cells"] = it["n_clean_cells"]
        records.append(rec)

    py_grade = _build_embedded_grade()
    grader_spec = t.build_grader()
    sm_sub = grader_spec["graders"]["typo_v5_score_model_extras"]
    gw = t.TYPO_V5_GOLD_WEIGHT
    ew = 1.0 - gw

    client = None if args.skip_sm else LLMClient()

    results = []
    by_g = defaultdict(list)
    failures = []  # columns scoring < 4.5
    t0 = time.monotonic()
    for i, rec in enumerate(records, 1):
        gold_typos = json.loads(rec["gold_typos"])
        if args.mode == "perfect":
            output_text = perfect_output(gold_typos)
        else:
            output_text = empty_output()

        py_score = float(py_grade({"output_text": output_text}, rec))
        if args.skip_sm:
            sm_score = float("nan")
            sm_reply = "(skipped)"
        else:
            try:
                sm_score, sm_reply, _ = run_score_model_leg(
                    client, args.grader_model, sm_sub, rec, output_text,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("SM call failed: %s", exc)
                sm_score, sm_reply = 0.0, f"<exception: {exc}>"

        total = (
            5.0 * (gw * py_score + ew * sm_score)
            if not args.skip_sm else 5.0 * py_score
        )

        G = rec["_n_gold_cells"]
        by_g[G].append(total)

        row = {
            "source": rec["source"],
            "column": rec["column"],
            "G": G,
            "n_clean_cells": rec["_n_clean_cells"],
            "py_score": py_score,
            "sm_score": None if args.skip_sm else sm_score,
            "total": total,
            "sm_reply": sm_reply,
            "synthetic_output": output_text,
        }
        results.append(row)

        if total < 4.5:
            failures.append(row)
            logger.info(
                "[%d/%d] LOW  total=%.3f py=%.2f sm=%s  %s::%s (G=%d)",
                i, len(records), total, py_score,
                f"{sm_score:.2f}" if not args.skip_sm else "skip",
                rec["source"], rec["column"], G,
            )
        elif i % 10 == 0:
            logger.info("[%d/%d] running mean=%.3f",
                        i, len(records),
                        statistics.mean(r["total"] for r in results))

        if i % 10 == 0 or i == len(records):
            (args.out_dir / "per_row.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False)
                          for r in results) + "\n",
                encoding="utf-8",
            )

    elapsed = time.monotonic() - t0

    totals = [r["total"] for r in results]
    summary = {
        "n_columns": len(results),
        "mode": args.mode,
        "elapsed_seconds": round(elapsed, 1),
        "mean_total": statistics.mean(totals),
        "min_total": min(totals),
        "max_total": max(totals),
        "std_total": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
        "n_below_4_5": sum(1 for t_ in totals if t_ < 4.5),
        "n_perfect_5": sum(1 for t_ in totals if t_ >= 4.999),
        "by_G_bucket": {
            str(g): {
                "n": len(v),
                "mean": statistics.mean(v),
                "min": min(v),
            }
            for g, v in sorted(by_g.items())
        },
        "failures_below_4_5": [
            {"source": f["source"], "column": f["column"], "G": f["G"],
             "py": f["py_score"], "sm": f["sm_score"], "total": f["total"]}
            for f in failures
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 60)
    print("GRADER-ONLY SANITY TEST")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
