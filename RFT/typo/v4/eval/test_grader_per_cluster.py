"""Per-cluster grader test.

For every distinct-value cluster across the 112 strict-filter
columns, we construct two synthetic trainee outputs and score
each with the v4 multigrader:

* ``clean_pred``  -- model claims the focal cluster is clean
                     (but correctly flags every other gold typo
                     in the column).
* ``flag_pred``   -- model claims the focal cluster is a typo
                     (correct canonical correction if it's a
                     gold cluster; a fake correction if it's
                     clean), AND correctly flags every other
                     gold typo in the column.

For each cluster we record (s_clean, s_flag) and the delta. A
well-calibrated grader should produce::

    gold cluster:  s_flag > s_clean   (flagging gets credit)
    clean cluster: s_clean > s_flag   (flagging gets penalty)

This isolates the grader's per-cell decision: ~1000 clusters
become ~2000 grader evaluations (python leg is instant; the
score-model leg is gated behind --use-sm because it's ~5h of
LLM calls).
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

from RFT.typo.v4 import _task as t  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_grader_on_benchmark import (  # noqa: E402
    _build_embedded_grade, load_kept_columns, run_score_model_leg,
)

logger = logging.getLogger("test_grader_per_cluster")


def _typo_entry(value: str, correction: str) -> dict:
    return {
        "value": value, "correction": correction,
        "flavour": "spelling", "rationale": "x",
    }


def _make_output(typos: list[dict]) -> str:
    return json.dumps({"typos": typos}, ensure_ascii=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",
                        default="benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl",
                        type=Path)
    parser.add_argument("--csv-root",
                        default="data/source_inconsistency_bing21",
                        type=Path)
    parser.add_argument("--out-dir",
                        default=Path("outputs/grader_per_cluster"),
                        type=Path)
    parser.add_argument("--use-sm", action="store_true",
                        help="Also score with the score-model leg "
                        "(slow: ~2x clusters in LLM calls).")
    parser.add_argument("--grader-model", default="gpt-5.2,gpt-5.4,gpt-5.5")
    parser.add_argument("--limit-clusters", type=int, default=None,
                        help="Cap total clusters processed.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading + filtering benchmark...")
    items = load_kept_columns(args.benchmark, args.csv_root)
    logger.info("Kept %d columns after strict filter.", len(items))

    records = [(t.build_record(it), it) for it in items]
    py_grade = _build_embedded_grade()

    grader_spec = t.build_grader()
    sm_sub = grader_spec["graders"]["typo_v4_score_model_extras"]
    gw = t.TYPO_V4_GOLD_WEIGHT
    ew = 1.0 - gw
    client = LLMClient() if args.use_sm else None

    # Enumerate clusters across all columns.
    cluster_jobs = []  # (col_idx, value, is_gold, gold_correction or None)
    for ci, (rec, it) in enumerate(records):
        gold_typos = json.loads(rec["gold_typos"])
        gold_map = {g["value"]: g["correction"] for g in gold_typos}
        vlist = json.loads(rec["column_values_json"])  # [[value, count], ...]
        seen = set()
        for v, _cnt in vlist:
            if v in seen:
                continue
            seen.add(v)
            cluster_jobs.append((
                ci, v, v in gold_map, gold_map.get(v),
            ))

    if args.limit_clusters:
        cluster_jobs = cluster_jobs[: args.limit_clusters]
    n_jobs = len(cluster_jobs)
    n_gold = sum(1 for _, _, g, _ in cluster_jobs if g)
    n_clean = n_jobs - n_gold
    logger.info(
        "Total clusters: %d  (gold=%d, clean=%d) -> %d grader evaluations",
        n_jobs, n_gold, n_clean, 2 * n_jobs,
    )

    results = []
    t0 = time.monotonic()
    for j, (ci, v, is_gold, gold_corr) in enumerate(cluster_jobs, 1):
        rec, it = records[ci]
        gold_typos = json.loads(rec["gold_typos"])
        # Build "everything correct except focal cluster"
        others = [_typo_entry(g["value"], g["correction"])
                  for g in gold_typos if g["value"] != v]

        # clean_pred: model says focal cluster is clean
        clean_typos = list(others)
        # flag_pred: model flags focal cluster
        if is_gold:
            flag_typos = others + [_typo_entry(v, gold_corr)]
        else:
            # Fabricate a plausible-looking but wrong correction.
            flag_typos = others + [_typo_entry(v, v + "_FAKE")]

        out_clean = _make_output(clean_typos)
        out_flag = _make_output(flag_typos)

        py_clean = float(py_grade({"output_text": out_clean}, rec))
        py_flag = float(py_grade({"output_text": out_flag}, rec))

        if args.use_sm:
            try:
                sm_clean, _, _ = run_score_model_leg(
                    client, args.grader_model, sm_sub, rec, out_clean,
                )
                sm_flag, _, _ = run_score_model_leg(
                    client, args.grader_model, sm_sub, rec, out_flag,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("SM call failed: %s", exc)
                sm_clean = sm_flag = 0.0
            total_clean = 5.0 * (gw * py_clean + ew * sm_clean)
            total_flag = 5.0 * (gw * py_flag + ew * sm_flag)
        else:
            sm_clean = sm_flag = None
            total_clean = 5.0 * py_clean
            total_flag = 5.0 * py_flag

        results.append({
            "source": rec["source"], "column": rec["column"], "value": v,
            "true_label": "gold" if is_gold else "clean",
            "py_clean": py_clean, "py_flag": py_flag,
            "sm_clean": sm_clean, "sm_flag": sm_flag,
            "total_clean": total_clean, "total_flag": total_flag,
            "delta_flag_minus_clean": total_flag - total_clean,
            "grader_prefers": (
                "flag" if total_flag > total_clean
                else ("clean" if total_clean > total_flag else "tie")
            ),
        })

        if j % 100 == 0 or j == n_jobs:
            (args.out_dir / "per_cluster.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False)
                          for r in results) + "\n",
                encoding="utf-8",
            )
            logger.info("[%d/%d]  elapsed=%.1fs", j, n_jobs,
                        time.monotonic() - t0)

    elapsed = time.monotonic() - t0

    # -- Summarise --
    by_label = defaultdict(list)
    for r in results:
        by_label[r["true_label"]].append(r)

    def stats(rows, key):
        vals = [r[key] for r in rows]
        return {
            "n": len(vals),
            "mean": statistics.mean(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }

    summary = {
        "n_clusters": n_jobs,
        "n_gold_clusters": n_gold,
        "n_clean_clusters": n_clean,
        "use_sm": args.use_sm,
        "elapsed_seconds": round(elapsed, 1),
        "gold_clusters": {
            "score_clean":  stats(by_label["gold"], "total_clean"),
            "score_flag":   stats(by_label["gold"], "total_flag"),
            "delta_flag_minus_clean":
                stats(by_label["gold"], "delta_flag_minus_clean"),
            "grader_correct_pct": round(100.0 * sum(
                1 for r in by_label["gold"] if r["grader_prefers"] == "flag"
            ) / max(len(by_label["gold"]), 1), 2),
        },
        "clean_clusters": {
            "score_clean":  stats(by_label["clean"], "total_clean"),
            "score_flag":   stats(by_label["clean"], "total_flag"),
            "delta_flag_minus_clean":
                stats(by_label["clean"], "delta_flag_minus_clean"),
            "grader_correct_pct": round(100.0 * sum(
                1 for r in by_label["clean"] if r["grader_prefers"] == "clean"
            ) / max(len(by_label["clean"]), 1), 2),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 60)
    print("PER-CLUSTER GRADER TEST")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
