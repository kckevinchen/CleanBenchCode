"""Quick iterative discrimination test.

Like ``test_grader_sm_discrimination.py`` but with a stratified
sample of fake-correction kinds so we can see, per prompt
revision, how the judge behaves on each perturbation type:

    * casing_only    -- value.lower() (or .upper() if already lc)
    * identical      -- correction equals value
    * last2_swap     -- swap final two characters
    * trailing_s     -- append "s"
    * one_char_sub   -- substitute one random middle character

For each kind we sample N_PER_KIND clean clusters; every gold
cluster is always included.

Usage:
    python RFT/typo/v5/eval/test_grader_sm_quick.py [--n-per-kind 25]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_grader_on_benchmark import (  # noqa: E402
    load_kept_columns, run_score_model_leg,
)

logger = logging.getLogger("test_grader_sm_quick")


def make_casing(v: str) -> str | None:
    if any(c.isupper() for c in v):
        out = v.lower()
        return out if out != v else None
    if any(c.isalpha() for c in v):
        return v.upper()
    return None


def make_identical(v: str) -> str | None:
    return v


def make_last2_swap(v: str) -> str | None:
    if len(v) < 2 or v[-1] == v[-2]:
        return None
    return v[:-2] + v[-1] + v[-2]


def make_trailing_s(v: str) -> str | None:
    if v.endswith("s") or v.endswith("S") or not v:
        return None
    return v + "s"


def make_one_char_sub(v: str, rng: random.Random) -> str | None:
    if len(v) < 3:
        return None
    idx = rng.randrange(1, len(v) - 1)
    if not v[idx].isalpha():
        return None
    repl = "x" if v[idx].lower() != "x" else "z"
    if v[idx].isupper():
        repl = repl.upper()
    return v[:idx] + repl + v[idx + 1:]


KIND_BUILDERS = {
    "casing_only": lambda v, rng: make_casing(v),
    "identical": lambda v, rng: make_identical(v),
    "last2_swap": lambda v, rng: make_last2_swap(v),
    "trailing_s": lambda v, rng: make_trailing_s(v),
    "one_char_sub": lambda v, rng: make_one_char_sub(v, rng),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",
                        default="benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl",
                        type=Path)
    parser.add_argument("--csv-root",
                        default="data/source_inconsistency_bing21",
                        type=Path)
    parser.add_argument("--out-dir",
                        default=Path("outputs/grader_sm_quick"),
                        type=Path)
    parser.add_argument("--grader-model", default="gpt-5.2,gpt-5.4,gpt-5.5")
    parser.add_argument("--n-per-kind", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-gold", action="store_true", default=True)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = load_kept_columns(args.benchmark, args.csv_root)
    records = [(t.build_record(it), it) for it in items]
    rng = random.Random(args.seed)

    # Pools per kind
    pools: dict[str, list] = defaultdict(list)
    gold_jobs = []
    for ci, (rec, it) in enumerate(records):
        gold_typos = json.loads(rec["gold_typos"])
        gold_map = {g["value"]: g["correction"] for g in gold_typos}
        vlist = json.loads(rec["column_values_json"])
        for v, _ in vlist:
            if v in gold_map:
                gold_jobs.append((ci, v, "gold", gold_map[v]))
                continue
            for kind, fn in KIND_BUILDERS.items():
                c = fn(v, rng)
                if c is None:
                    continue
                pools[kind].append((ci, v, kind, c))

    # Dedupe gold (some columns have duplicate value clusters)
    seen = set()
    gold_jobs_d = []
    for ci, v, lbl, c in gold_jobs:
        key = (ci, v)
        if key in seen:
            continue
        seen.add(key)
        gold_jobs_d.append((ci, v, lbl, c))
    gold_jobs = gold_jobs_d

    jobs = []
    if args.include_gold:
        jobs.extend(gold_jobs)
    for kind in KIND_BUILDERS:
        rng.shuffle(pools[kind])
        jobs.extend(pools[kind][: args.n_per_kind])
    rng.shuffle(jobs)
    logger.info("Jobs: %d (gold=%d) per-kind cap=%d",
                len(jobs), len(gold_jobs), args.n_per_kind)

    sm_sub = t.build_grader()["graders"]["typo_v5_score_model_extras"]
    client = LLMClient()

    results = []
    t0 = time.monotonic()
    for i, (ci, v, kind, correction) in enumerate(jobs, 1):
        rec, _ = records[ci]
        output_text = json.dumps({
            "typos": [{
                "value": v, "correction": correction,
                "flavour": "spelling",
                "rationale": "single-cluster grader probe",
            }],
        }, ensure_ascii=False)
        try:
            score, reply, _ = run_score_model_leg(
                client, args.grader_model, sm_sub, rec, output_text,
            )
        except Exception as exc:  # noqa: BLE001
            score, reply = 0.0, f"<exception: {exc}>"
        results.append({
            "source": rec["source"], "column": rec["column"],
            "value": v, "correction": correction,
            "kind": kind, "sm_score": score,
            "sm_reply_head": reply[:240],
        })
        if i % 30 == 0 or i == len(jobs):
            elapsed = time.monotonic() - t0
            (args.out_dir / "per_cluster.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False)
                          for r in results) + "\n",
                encoding="utf-8",
            )
            logger.info("[%d/%d] %.1fs", i, len(jobs), elapsed)

    # Summary by kind
    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]].append(r["sm_score"])

    summary = {
        "n_calls": len(results),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "by_kind": {
            kind: {
                "n": len(scores),
                "mean": round(statistics.mean(scores), 3) if scores else None,
                "median": (round(statistics.median(scores), 3)
                           if scores else None),
                "pct_correct": round(
                    100.0 * (
                        sum(1 for s in scores if s >= 0.5)
                        if kind == "gold"
                        else sum(1 for s in scores if s < 0.5)
                    ) / max(len(scores), 1), 1,
                ),
            }
            for kind, scores in sorted(by_kind.items())
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 60)
    print("PROMPT REVISION TEST")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
