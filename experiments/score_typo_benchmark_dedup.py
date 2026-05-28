"""Score typo_benchmark runs vs human gold, deduped by (source, column, value).

Repeated identical values within a column collapse to one entry. Useful
when a single typo string appears in many rows of the same column.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from RFT.typo_v5._task import is_v5_trivial_correction

BENCHMARK = Path("benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl")


def load_gold():
    pos = set()
    ambig = set()
    for line in BENCHMARK.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "typo" not in (r.get("error_types") or []):
            continue
        key = (r["source"], r["column"], str(r.get("original_value")))
        label = r.get("label")
        if label == "error":
            if is_v5_trivial_correction(r.get("original_value"), r.get("corrected_value")):
                continue
            pos.add(key)
        elif label == "ambiguous":
            ambig.add(key)
    return pos, ambig


def load_pred(run_dir: Path):
    cells = set()
    for line in (run_dir / "issues.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error_type") != "typo":
            continue
        src = r.get("source")
        observed = str(r.get("observed"))
        for col in (r.get("columns") or []):
            cells.add((src, col, observed))
    return cells


def score(run_dir: Path, pos, ambig):
    pred = load_pred(run_dir)
    tp = pred & pos
    ambig_hit = pred & ambig
    fp = pred - pos - ambig
    fn = pos - pred
    n_tp, n_fp, n_fn, n_amb = len(tp), len(fp), len(fn), len(ambig_hit)
    p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    r = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "run": run_dir.name,
        "predicted": len(pred),
        "tp": n_tp,
        "fp": n_fp,
        "fn": n_fn,
        "ambig_hit": n_amb,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }


def main(argv):
    if not argv:
        print("usage: score_typo_benchmark_dedup.py <run_dir> [<run_dir> ...]")
        sys.exit(2)
    pos, ambig = load_gold()
    print(f"Gold non-trivial typo VALUES (deduped): {len(pos)}  ambiguous: {len(ambig)}")
    rows = [score(Path(d), pos, ambig) for d in argv]
    cols = ["run", "predicted", "tp", "fp", "fn", "ambig_hit", "precision", "recall", "f1"]
    widths = [max(len(c), max(len(str(r[c])) for r in rows)) for c in cols]
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


if __name__ == "__main__":
    main(sys.argv[1:])
