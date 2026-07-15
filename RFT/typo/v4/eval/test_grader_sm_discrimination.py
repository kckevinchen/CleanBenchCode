"""Score-model grader discrimination test.

In the RFT setting, every item handed to the score-model grader
is something the trainee flagged as a typo. We want to know:

    Does the LLM grader give HIGH scores to actual typos and
    LOW scores to clean values mistakenly flagged?

For each cluster (distinct value across the 112 strict-filter
columns), we construct a trainee output containing ONLY that
single flagged cluster::

    {"typos": [{"value": v, "correction": c,
                "flavour": "spelling", "rationale": "..."}]}

* If the cluster is a benchmark gold typo we use the canonical
  correction -- the grader should award full credit because
  the flagged value really is a typo.
* If the cluster is a benchmark-clean value we use a plausibly
  wrong correction (e.g. just lower-case it, or swap two
  characters). The grader should award little or no credit
  because the trainee invented a typo that isn't there.

We score with the v4 score-model leg only (the python leg is
purely numeric and has already been characterised). The output
is a per-cluster table plus headline stats: mean/median score
for true-positives vs false-positives, plus the AUC-like split
(% of gold scoring above mean clean score).

Cost note: every cluster = one LLM call. We sample ~1000
clusters by default (all gold + a random clean subset) so a
full run takes ~30-60 min with the gpt-5 pool.
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

logger = logging.getLogger("test_grader_sm_discrimination")


def fake_correction(value: str) -> str:
    """Return a plausibly-wrong correction for a clean value.

    Heuristics: if the value has any upper-case letter, lower-case
    everything; else swap the last two characters; if the value is
    very short, just append 's'. The fake should look like the
    kind of thing a trigger-happy detector might propose.
    """
    if not value:
        return "x"
    if any(ch.isupper() for ch in value):
        out = value.lower()
        if out != value:
            return out
    if len(value) >= 2:
        return value[:-2] + value[-1] + value[-2]
    return value + "s"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",
                        default="benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl",
                        type=Path)
    parser.add_argument("--csv-root",
                        default="data/source_inconsistency_bing21",
                        type=Path)
    parser.add_argument("--out-dir",
                        default=Path("outputs/grader_sm_discrimination"),
                        type=Path)
    parser.add_argument("--grader-model", default="gpt-5.2,gpt-5.4,gpt-5.5")
    parser.add_argument("--n-clean", type=int, default=964,
                        help="How many clean clusters to sample (gold "
                        "clusters are always included in full).")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent grader calls.")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Retries per job on transient grader "
                        "failure (empty reply / missing score line / "
                        "HTTP error). Exponential backoff between "
                        "attempts.")
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
    logger.info("Kept %d columns.", len(items))

    records = [(t.build_record(it), it) for it in items]
    rng = random.Random(args.seed)

    # Enumerate clusters: (col_idx, value, true_label, correction).
    # Trivial-error and ambiguous cell values are skipped entirely
    # (no test case), per the cell-level filter design.
    gold_jobs, clean_jobs = [], []
    n_trivial_skipped = 0
    n_ambiguous_skipped = 0
    for ci, (rec, it) in enumerate(records):
        gold_typos = json.loads(rec["gold_typos"])
        gold_map = {g["value"]: g["correction"] for g in gold_typos}
        trivial_set = set(it.get("trivial_values", []))
        ambiguous_set = set(it.get("ambiguous_values", []))
        vlist = json.loads(rec["column_values_json"])
        seen_gold = set()
        for v, _ in vlist:
            if v in trivial_set:
                n_trivial_skipped += 1
                continue
            if v in ambiguous_set:
                n_ambiguous_skipped += 1
                continue
            if v in gold_map:
                if v in seen_gold:
                    continue
                seen_gold.add(v)
                gold_jobs.append((ci, v, "gold", gold_map[v]))
            else:
                clean_jobs.append((ci, v, "clean", fake_correction(v)))

    rng.shuffle(clean_jobs)
    clean_jobs = clean_jobs[: args.n_clean]
    jobs = gold_jobs + clean_jobs
    rng.shuffle(jobs)
    logger.info(
        "Total LLM calls: %d  (gold=%d, clean=%d, "
        "trivial_skipped=%d, ambiguous_skipped=%d)",
        len(jobs), len(gold_jobs), len(clean_jobs),
        n_trivial_skipped, n_ambiguous_skipped,
    )

    grader_spec = t.build_grader()
    sm_sub = grader_spec["graders"]["typo_v4_score_model_extras"]
    client = LLMClient()

    results = []
    results_lock = threading.Lock()
    t0 = time.monotonic()

    def _run_one(job):
        ci, v, label, correction = job
        rec, _ = records[ci]
        output_text = json.dumps({
            "typos": [{
                "value": v, "correction": correction,
                "flavour": "spelling",
                "rationale": "single-cluster grader probe",
            }],
        }, ensure_ascii=False)
        last_reply = "<no attempt>"
        for attempt in range(args.max_retries + 1):
            try:
                score, reply, _full = run_score_model_leg(
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
                return {
                    "source": rec["source"], "column": rec["column"],
                    "value": v, "correction": correction,
                    "true_label": label, "sm_score": score,
                    "sm_reply_head": reply[:240],
                    "attempts": attempt + 1,
                }
            if attempt < args.max_retries:
                time.sleep(2 ** attempt)
        logger.warning(
            "Job failed after %d attempts: col=%r value=%r reply=%s",
            args.max_retries + 1, rec["column"], v, last_reply[:120],
        )
        return {
            "source": rec["source"], "column": rec["column"],
            "value": v, "correction": correction,
            "true_label": label, "sm_score": 0.0,
            "sm_reply_head": last_reply[:240],
            "attempts": args.max_retries + 1,
            "failed": True,
        }

    def _flush_and_log(done):
        elapsed = time.monotonic() - t0
        (args.out_dir / "per_cluster.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False)
                      for r in results) + "\n",
            encoding="utf-8",
        )
        gold_scores = [r["sm_score"] for r in results
                       if r["true_label"] == "gold"]
        clean_scores = [r["sm_score"] for r in results
                        if r["true_label"] == "clean"]
        g_mean = statistics.mean(gold_scores) if gold_scores else 0.0
        c_mean = statistics.mean(clean_scores) if clean_scores else 0.0
        logger.info(
            "[%d/%d] elapsed=%.1fs  gold_mean=%.3f clean_mean=%.3f",
            done, len(jobs), elapsed, g_mean, c_mean,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_run_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            with results_lock:
                results.append(r)
                done = len(results)
            if done % 50 == 0 or done == len(jobs):
                with results_lock:
                    _flush_and_log(done)

    elapsed = time.monotonic() - t0

    gold_scores = [r["sm_score"] for r in results
                   if r["true_label"] == "gold"]
    clean_scores = [r["sm_score"] for r in results
                    if r["true_label"] == "clean"]

    # AUC-like discrimination: probability a random gold ranks above
    # a random clean (Mann-Whitney U / (n*m)). Ties count as 0.5.
    def auc(pos, neg):
        if not pos or not neg:
            return None
        wins, ties = 0, 0
        for p in pos:
            for n in neg:
                if p > n:
                    wins += 1
                elif p == n:
                    ties += 1
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    def stats_block(name, scores):
        return {
            "n": len(scores),
            "mean": statistics.mean(scores) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "p10": (statistics.quantiles(scores, n=10)[0]
                    if len(scores) >= 10 else None),
            "p90": (statistics.quantiles(scores, n=10)[-1]
                    if len(scores) >= 10 else None),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        }

    summary = {
        "n_calls": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "gold": stats_block("gold", gold_scores),
        "clean": stats_block("clean", clean_scores),
        "discrimination_auc": auc(gold_scores, clean_scores),
        "pct_clean_below_0_5": (
            100.0 * sum(1 for s in clean_scores if s < 0.5)
            / max(len(clean_scores), 1)
        ),
        "pct_gold_above_0_5": (
            100.0 * sum(1 for s in gold_scores if s > 0.5)
            / max(len(gold_scores), 1)
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 60)
    print("GRADER SCORE-MODEL DISCRIMINATION TEST")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
