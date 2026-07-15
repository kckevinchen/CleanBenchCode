"""Cell-level grader probe -- NO filtering.

Every labelled cell in the v4 typo benchmark is turned into a single
grader call. Cells are partitioned into four groups based on their
benchmark label:

  * ``clean``           -- label='clean'.
  * ``error_nontrivial``-- label='error' and the (original, corrected)
                            edit is NOT a v4 trivial edit, and the
                            ``note`` field is not ``"trivial"``.
  * ``error_trivial``   -- label='error' and the edit IS a v4 trivial
                            edit (or the row carries ``note=trivial``).
  * ``ambiguous``       -- label='ambiguous'.

For each cell we send one (column, source, value, correction) probe to
the score-model grader. ``correction`` is:

  * For ``clean`` and ``ambiguous``: a realistic LLM-proposed
    correction sourced from the ``--corrections-cache`` JSONL
    (produced by ``generate_cell_corrections``). The cache holds
    one GPT-5-family suggestion per (source, column, original_value)
    so the candidate looks like a genuine typo-detector proposal
    rather than a synthetic perturbation. Cells with no cache entry
    fall back to ``fake_correction`` only when
    ``--allow-perturbation-fallback`` is set; otherwise they are
    skipped.
  * For ``error_*``: the benchmark's ``corrected_value`` (canonical
    ground-truth fix).

The grader still operates in NO-GOLD mode: ``gold_typos`` and
``ambiguous_values`` rendered into the prompt are empty (just like the
discrimination test). What we are measuring is how the grader scores
each benchmark group on its own merits.

The script reports mean/median/n per group plus a histogram of
{0.0, 0.5, 1.0} scores per group. Both raw per-row results
(``per_cell.jsonl``) and the summary (``summary.json``) are written
under ``--out-dir``.

Usage::

    python -m RFT.typo.v4.eval.test_grader_sm_cell_level \\
        --benchmark benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl \\
        --csv-root data/source_inconsistency_bing21 \\
        --grader-model gpt-4o \\
        --concurrency 20 --max-retries 3 \\
        --out-dir outputs/grader_sm_cell_level_gpt4o
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
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

logger = logging.getLogger("test_grader_sm_cell_level")


def _classify_row(r: dict) -> str | None:
    lbl = r.get("label")
    if lbl == "clean":
        return "clean"
    if lbl == "ambiguous":
        return "ambiguous"
    if lbl == "error":
        o = r.get("original_value")
        c = r.get("corrected_value")
        note = (r.get("note") or "").strip().lower()
        is_trivial = (
            note == "trivial"
            or (c is not None and t.is_v4_trivial_correction(o, c))
        )
        return "error_trivial" if is_trivial else "error_nontrivial"
    return None


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
        default=Path("outputs/grader_sm_cell_level"),
        type=Path,
    )
    parser.add_argument("--grader-model", default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--corrections-cache",
        type=Path,
        default=Path("outputs/cell_level_corrections.jsonl"),
        help=(
            "JSONL produced by generate_cell_corrections; supplies a "
            "realistic correction candidate for clean + ambiguous cells."
        ),
    )
    parser.add_argument(
        "--allow-perturbation-fallback",
        action="store_true",
        help=(
            "If a clean/ambiguous cell has no entry in the corrections "
            "cache, fall back to fake_correction (perturbation). "
            "Default is to skip the cell so results aren't contaminated "
            "by synthetic mutations."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load every column (no ambiguous/trivial filtering) so we have a
    # CSV-distinct value listing per (source, column) for the grader
    # prompt. The grader still runs in no-gold mode (empty gold_typos /
    # ambiguous_values are passed at the prompt template level via
    # build_record).
    logger.info("Loading + filtering benchmark...")
    items = load_kept_columns(
        args.benchmark, args.csv_root,
        drop_trivial_columns=False, drop_ambiguous_columns=False,
    )
    logger.info("Kept %d columns.", len(items))

    # Wipe gold + ambiguous from each item so the grader sees no labels.
    for it in items:
        it["gold_typos"] = []
        it["ambiguous_values"] = []

    rec_by_key = {
        (it["source"], it["column"]): t.build_record(it) for it in items
    }

    # Load LLM-generated correction candidates for clean + ambiguous
    # cells (one entry per distinct (source, column, original_value)).
    corrections_cache: dict[tuple[str, str, str], str] = {}
    if args.corrections_cache.exists():
        with args.corrections_cache.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                corr = row.get("correction")
                if not isinstance(corr, str) or not corr:
                    continue
                key = (row.get("source"), row.get("column"),
                       row.get("original"))
                if all(isinstance(k, str) for k in key):
                    corrections_cache[key] = corr
        logger.info(
            "Loaded %d cached corrections from %s",
            len(corrections_cache), args.corrections_cache,
        )
    else:
        logger.warning(
            "Corrections cache %s not found; clean/ambiguous cells "
            "without a cache entry will be %s.",
            args.corrections_cache,
            "perturbed" if args.allow_perturbation_fallback else "skipped",
        )

    # Iterate every labelled benchmark row and assemble a job per cell.
    jobs = []
    missing_csv = collections.Counter()
    no_cache_skips = collections.Counter()
    with args.benchmark.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            r = json.loads(line)
            grp = _classify_row(r)
            if grp is None:
                continue
            key = (r["source"], r["column"])
            rec = rec_by_key.get(key)
            if rec is None:
                missing_csv[key] += 1
                continue
            value = r.get("original_value")
            if not isinstance(value, str):
                continue
            if grp in ("clean", "ambiguous"):
                correction = corrections_cache.get(
                    (r["source"], r["column"], value)
                )
                if not correction:
                    if args.allow_perturbation_fallback:
                        correction = fake_correction(value)
                    else:
                        no_cache_skips[grp] += 1
                        continue
            else:
                correction = r.get("corrected_value")
                if not isinstance(correction, str) or not correction:
                    correction = r.get("llm_suggested_value")
                if not isinstance(correction, str) or not correction:
                    continue
            jobs.append({
                "row_idx": idx, "group": grp,
                "source": r["source"], "column": r["column"],
                "value": value, "correction": correction,
                "note": r.get("note") or "",
            })

    by_group = collections.Counter(j["group"] for j in jobs)
    logger.info(
        "Total LLM calls: %d  groups=%s  missing_csv_cols=%d  no_cache_skips=%s",
        len(jobs), dict(by_group), sum(missing_csv.values()),
        dict(no_cache_skips),
    )

    grader_spec = t.build_grader()
    sm_sub = grader_spec["graders"]["typo_v4_score_model_extras"]
    client = LLMClient()

    results = []
    results_lock = threading.Lock()
    t0 = time.monotonic()

    def _run_one(job):
        rec = rec_by_key[(job["source"], job["column"])]
        output_text = json.dumps({
            "typos": [{
                "value": job["value"], "correction": job["correction"],
                "flavour": "spelling",
                "rationale": "single-cell grader probe",
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
                    **job,
                    "sm_score": score,
                    "sm_reply_head": reply[:240],
                    "attempts": attempt + 1,
                }
            if attempt < args.max_retries:
                time.sleep(2 ** attempt)
        logger.warning(
            "Job failed after %d attempts: col=%r value=%r reply=%s",
            args.max_retries + 1, job["column"], job["value"],
            last_reply[:120],
        )
        return {
            **job,
            "sm_score": 0.0,
            "sm_reply_head": last_reply[:240],
            "attempts": args.max_retries + 1,
            "failed": True,
        }

    def _flush(done):
        elapsed = time.monotonic() - t0
        (args.out_dir / "per_cell.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False)
                      for r in results) + "\n",
            encoding="utf-8",
        )
        per_grp_mean = {
            g: round(statistics.mean(
                [r["sm_score"] for r in results if r["group"] == g]
            ), 3) if any(r["group"] == g for r in results) else None
            for g in by_group
        }
        logger.info(
            "[%d/%d] elapsed=%.1fs  means=%s",
            done, len(jobs), elapsed, per_grp_mean,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            with results_lock:
                results.append(r)
                done = len(results)
            if done % 100 == 0 or done == len(jobs):
                with results_lock:
                    _flush(done)

    # Per-group stats + histogram.
    def _stats(scores):
        if not scores:
            return None
        return {
            "n": len(scores),
            "mean": round(statistics.mean(scores), 4),
            "median": statistics.median(scores),
            "stdev": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
            "min": min(scores), "max": max(scores),
            "hist": {
                "score_0.0": sum(1 for s in scores if s == 0.0),
                "score_0.5": sum(1 for s in scores if s == 0.5),
                "score_1.0": sum(1 for s in scores if s == 1.0),
            },
        }

    summary = {
        "n_calls": len(results),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "groups": {
            g: _stats([r["sm_score"] for r in results if r["group"] == g])
            for g in by_group
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 60)
    print("GRADER SCORE-MODEL CELL-LEVEL TEST (no filter)")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
