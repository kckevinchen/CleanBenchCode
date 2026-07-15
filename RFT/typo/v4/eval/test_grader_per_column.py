"""Column-level grader test: aggregated score vs label-derived ideal.

The cell-level discrimination test characterises the grader's per-
decision soundness. In RFT the grader sees a WHOLE column's worth of
flagged typos and returns a single column score. This test measures
how close that aggregated score is to what we would get if we mapped
each flagged cell to its benchmark label and averaged.

Per column we synthesise ONE trainee output that contains:

* every gold cell, with its canonical correction        -> label=valid    (reward 1.0)
* a random subset of clean cells, with a fake correction -> label=fp       (reward 0.0)
* every trivial-error cell, with its trivial correction  -> label=trivial_fp (reward 0.0)

Ambiguous-labelled cells are not included.

The IDEAL column score is the mean of per-cell rewards from the label
mapping above. The GRADER column score is what
``typo_v4_score_model_extras`` returns when given that combined output.
A perfect aggregator under our cell-level study would give grader_score
== ideal_score. Differences come from (a) the grader classifying a cell
differently from our label-derived mapping (e.g. judging a gold cell
ambiguous instead of valid), or (b) aggregation noise.

Outputs (under --out-dir):

  per_column.jsonl   -- one row per column with n_valid, n_fp,
                        n_trivial, ideal_score, grader_score, delta.
  summary.json       -- MAE, signed mean error, Pearson r, and
                        the breakdown of delta by configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v4 import _task as t  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_grader_on_benchmark import (  # noqa: E402
    load_kept_columns, run_score_model_leg,
)
from test_grader_sm_discrimination import fake_correction  # noqa: E402

logger = logging.getLogger("test_grader_per_column")


def _typo_entry(value: str, correction: str) -> dict:
    return {
        "value": value, "correction": correction,
        "flavour": "spelling",
        "rationale": "synthetic column-level grader probe",
    }


def build_column_output(
    item: dict,
    rec: dict,
    rng: random.Random,
    max_clean: int,
    max_trivial: int,
) -> tuple[str, dict]:
    """Synthesise a trainee output for ``item`` and return its label tally."""
    gold_typos = json.loads(rec["gold_typos"])
    gold_values = {g["value"]: g["correction"] for g in gold_typos}
    trivial_values = list(item.get("trivial_values", []))
    ambiguous_values = set(item.get("ambiguous_values", []))

    vlist = json.loads(rec["column_values_json"])
    clean_pool = [
        v for v, _ in vlist
        if v not in gold_values
        and v not in trivial_values
        and v not in ambiguous_values
    ]
    rng.shuffle(clean_pool)
    clean_pick = clean_pool[:max_clean]

    rng.shuffle(trivial_values)
    trivial_pick = trivial_values[:max_trivial]

    typos: list[dict] = []
    for v, c in gold_values.items():
        typos.append(_typo_entry(v, c))
    for v in clean_pick:
        typos.append(_typo_entry(v, fake_correction(v)))
    # Trivial cells lift the actual trivial correction from the
    # benchmark when we have it; otherwise just lower-case the value
    # (which is itself a casing-only edit, i.e. still trivial).
    bench_trivial_corr = item.get("trivial_corrections") or {}
    for v in trivial_pick:
        c = bench_trivial_corr.get(v) or v.lower() or v + " "
        typos.append(_typo_entry(v, c))

    rng.shuffle(typos)
    output_text = json.dumps({"typos": typos}, ensure_ascii=False)

    tally = {
        "n_valid": len(gold_values),
        "n_fp": len(clean_pick),
        "n_trivial": len(trivial_pick),
        "n_total": len(typos),
    }
    return output_text, tally


def ideal_score(tally: dict) -> float:
    n = tally["n_total"]
    if n == 0:
        return 1.0
    return tally["n_valid"] / n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default="benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--csv-root",
        default="data/source_inconsistency_bing21",
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default=Path("outputs/grader_per_column"),
        type=Path,
    )
    parser.add_argument("--grader-model", default="gpt-4o")
    parser.add_argument("--max-clean", type=int, default=3,
                        help="Max fake-flag clean cells per column.")
    parser.add_argument("--max-trivial", type=int, default=2,
                        help="Max trivial-flag cells per column.")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading + filtering benchmark...")
    items = load_kept_columns(
        args.benchmark, args.csv_root,
        drop_trivial_columns=False,
        drop_ambiguous_columns=False,
    )
    # Restrict to columns with at least one non-trivial gold typo --
    # otherwise the output contains only clean/trivial fps which are
    # informative but don't tell us anything about valid-recognition.
    items = [it for it in items if it.get("n_gold_cells", 0) > 0]
    logger.info("Columns with >=1 gold cell: %d", len(items))

    grader_spec = t.build_grader()
    sm_sub = grader_spec["graders"]["typo_v4_score_model_extras"]
    client = LLMClient()

    rng = random.Random(args.seed)

    jobs = []
    for it in items:
        rec = t.build_record(it)
        output_text, tally = build_column_output(
            it, rec, rng, args.max_clean, args.max_trivial,
        )
        jobs.append((rec, it, output_text, tally))

    logger.info("Scoring %d columns with grader-model=%s...",
                len(jobs), args.grader_model)

    results = []
    results_lock = threading.Lock()
    t0 = time.monotonic()

    def _run_one(job):
        rec, it, output_text, tally = job
        last_reply = "<no attempt>"
        for attempt in range(args.max_retries + 1):
            try:
                score, reply, _ = run_score_model_leg(
                    client, args.grader_model, sm_sub, rec, output_text,
                )
            except Exception as exc:  # noqa: BLE001
                score, reply = None, f"<exception: {exc}>"
            last_reply = reply
            transient = (
                score is None
                or reply.startswith("<query failed")
                or reply.startswith("<exception")
                or reply == "<score line missing>"
            )
            if not transient:
                ideal = ideal_score(tally)
                return {
                    "source": rec["source"], "column": rec["column"],
                    **tally,
                    "ideal_score": ideal,
                    "grader_score": score,
                    "delta": score - ideal,
                    "attempts": attempt + 1,
                    "grader_reply_head": reply[:400],
                }
            if attempt < args.max_retries:
                time.sleep(2 ** attempt)
        return {
            "source": rec["source"], "column": rec["column"],
            **tally,
            "ideal_score": ideal_score(tally),
            "grader_score": None,
            "delta": None,
            "attempts": args.max_retries + 1,
            "failed": True,
            "grader_reply_head": last_reply[:400],
        }

    def _flush(done):
        elapsed = time.monotonic() - t0
        (args.out_dir / "per_column.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
            + "\n",
            encoding="utf-8",
        )
        deltas = [r["delta"] for r in results if r["delta"] is not None]
        if deltas:
            mae = statistics.mean(abs(d) for d in deltas)
            me = statistics.mean(deltas)
        else:
            mae = me = float("nan")
        logger.info(
            "[%d/%d] elapsed=%.1fs  MAE=%.3f  mean_signed_err=%+.3f",
            done, len(jobs), elapsed, mae, me,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_run_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            with results_lock:
                results.append(r)
                done = len(results)
            if done % 20 == 0 or done == len(jobs):
                with results_lock:
                    _flush(done)

    deltas = [r["delta"] for r in results if r["delta"] is not None]
    grader_scores = [r["grader_score"] for r in results if r["grader_score"] is not None]
    ideals = [r["ideal_score"] for r in results if r["grader_score"] is not None]
    n = len(deltas)

    def pearson(xs, ys):
        if len(xs) < 2:
            return None
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        nu = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        if dx == 0 or dy == 0:
            return None
        return nu / (dx * dy)

    summary = {
        "n_columns_scored": n,
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "grader_model": args.grader_model,
        "mae": statistics.mean(abs(d) for d in deltas) if deltas else None,
        "mean_signed_error": statistics.mean(deltas) if deltas else None,
        "stdev_delta": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "pearson_r_ideal_vs_grader": pearson(ideals, grader_scores),
        "ideal_mean": statistics.mean(ideals) if ideals else None,
        "grader_mean": statistics.mean(grader_scores) if grader_scores else None,
        "pct_within_0_05": (
            100.0 * sum(1 for d in deltas if abs(d) <= 0.05) / max(n, 1)
        ),
        "pct_within_0_10": (
            100.0 * sum(1 for d in deltas if abs(d) <= 0.10) / max(n, 1)
        ),
        "pct_grader_higher": (
            100.0 * sum(1 for d in deltas if d > 0) / max(n, 1)
        ),
        "pct_grader_lower": (
            100.0 * sum(1 for d in deltas if d < 0) / max(n, 1)
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 60)
    print("PER-COLUMN GRADER vs IDEAL (label-derived)")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
