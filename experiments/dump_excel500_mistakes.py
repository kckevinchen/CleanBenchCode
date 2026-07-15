"""Dump the cells where a typo run disagrees with the excel_500_dedup benchmark.

One JSONL record per mistake (ambiguous/skip cells are excluded, matching
``score_excel500_benchmark``):

  * ``"fp"`` -- the model flagged a value that is NOT a gold ``error`` (it is
    labelled ``clean``, or sits in an unlabelled column). Carries the model's
    suggested correction + the benchmark label, so real typos the benchmark
    marked clean (gold under-labels) are easy to spot.
  * ``"fn"`` -- a gold ``error`` cell the model did NOT flag (missed). Carries
    the gold correction.

    python -m experiments.dump_excel500_mistakes <run_dir> [--out path.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.score_excel500_benchmark import BENCH, EXCLUDE_LABELS, norm


def load_gold_records() -> dict:
    """(source, column, norm value) -> gold record (error label wins ties)."""
    by_key: dict = {}
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "typo" not in (r.get("error_types") or []):
            continue
        key = (r["source"], r["column"], norm(r.get("original_value")))
        prev = by_key.get(key)
        if prev is None or (r.get("label") == "error" and prev["label"] != "error"):
            by_key[key] = {
                "label": r.get("label"),
                "original_value": r.get("original_value"),
                "corrected_value": r.get("corrected_value"),
                "note": r.get("note"),
            }
    return by_key


def load_pred_records(run_dir: Path) -> dict:
    """(source, column, norm value) -> the model's issue detail (first seen)."""
    by_key: dict = {}
    for line in (run_dir / "issues.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error_type") != "typo":
            continue
        s, v = r.get("source"), norm(r.get("observed"))
        for col in (r.get("columns") or []):
            by_key.setdefault((s, col, v), {
                "observed": r.get("observed"),
                "corrected_value": r.get("corrected_value"),
                "suggested_fix": r.get("suggested_fix"),
                "explanation": r.get("explanation"),
            })
    return by_key


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gold = load_gold_records()
    pred = load_pred_records(args.run_dir)
    pos = {k for k, g in gold.items() if g["label"] == "error"}
    excl = {k for k, g in gold.items() if g["label"] in EXCLUDE_LABELS}
    pred_keys = set(pred) - excl
    fp = pred_keys - pos
    fn = pos - pred_keys

    out = args.out or (args.run_dir / "mistakes.jsonl")
    n_fp = n_fn = 0
    with open(out, "w", encoding="utf-8") as f:
        for k in sorted(fp):
            s, c, _ = k
            pr = pred[k]
            g = gold.get(k)
            f.write(json.dumps({
                "mistake": "fp",
                "source": s, "column": c,
                "value": pr["observed"],
                "gold_label": g["label"] if g else "unlabeled",
                "model_correction": pr.get("corrected_value"),
                "model_suggested_fix": pr.get("suggested_fix"),
                "model_explanation": pr.get("explanation"),
            }, ensure_ascii=False) + "\n")
            n_fp += 1
        for k in sorted(fn):
            s, c, _ = k
            g = gold[k]
            f.write(json.dumps({
                "mistake": "fn",
                "source": s, "column": c,
                "value": g["original_value"],
                "gold_label": "error",
                "gold_correction": g["corrected_value"],
                "note": g.get("note"),
            }, ensure_ascii=False) + "\n")
            n_fn += 1
    print(f"wrote {n_fp} FP + {n_fn} FN = {n_fp + n_fn} mistakes -> {out}")


if __name__ == "__main__":
    main()
