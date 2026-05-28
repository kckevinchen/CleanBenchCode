"""Score typo_benchmark runs against the human-labeled gold benchmark.

Cell-level comparison keyed by (source, row, column). Gold positives are
rows with label='error' AND 'typo' in error_types, excluding v5-trivial
(original, corrected_value) pairs. Predictions matching an 'ambiguous'
cell are tallied separately and excluded from FP.

Usage:
    python -m experiments.score_typo_benchmark_runs <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from RFT.typo_v5._task import is_v5_trivial_correction

BENCHMARK = Path("benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl")


def load_gold(path: Path):
    pos = set()       # (source, row, column) non-trivial typo errors
    ambig = set()
    by_cell = {}      # (source, row, column) -> gold record
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "typo" not in (r.get("error_types") or []):
            continue
        key = (r["source"], int(r["row"]), r["column"])
        by_cell[key] = r
        label = r.get("label")
        if label == "error":
            if is_v5_trivial_correction(r.get("original_value"), r.get("corrected_value")):
                continue
            pos.add(key)
        elif label == "ambiguous":
            ambig.add(key)
    return pos, ambig, by_cell


def load_pred(run_dir: Path):
    issues_path = run_dir / "issues.jsonl"
    cells = set()
    for line in issues_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error_type") != "typo":
            continue
        src = r.get("source")
        for col in (r.get("columns") or []):
            for row in (r.get("affected_rows") or [r.get("row")]):
                if row is None:
                    continue
                cells.add((src, int(row), col))
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
        print("usage: score_typo_benchmark_runs.py <run_dir> [<run_dir> ...]")
        sys.exit(2)
    pos, ambig, _ = load_gold(BENCHMARK)
    print(f"Gold non-trivial typo cells: {len(pos)}  ambiguous: {len(ambig)}")
    rows = [score(Path(d), pos, ambig) for d in argv]
    cols = ["run", "predicted", "tp", "fp", "fn", "ambig_hit", "precision", "recall", "f1"]
    widths = [max(len(c), max(len(str(r[c])) for r in rows)) for c in cols]
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


if __name__ == "__main__":
    main(sys.argv[1:])
