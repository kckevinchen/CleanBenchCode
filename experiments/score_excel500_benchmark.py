"""Score a typo detection run against the excel_500_dedup human benchmark.

Cell-level micro-F1 by (source, column, normalised value), restricted to the
(source, column) pairs the benchmark labelled. Per the eval convention here:

  * gold positive = benchmark label ``error``;
  * ``ambiguous`` and ``skip`` cells are EXCLUDED (immunity -- no TP/FP/FN);
  * ``clean`` (and any other labelled-but-not-error value in a scored column)
    is a negative: a prediction on it is a FALSE POSITIVE. The detector runs
    over ALL the data including never-flagged cells, so flagging a clean value
    counts against precision.

Usage:
    python -m experiments.score_excel500_benchmark <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BENCH = Path("benchmark/benchmark_typo_excel_500_dedup.jsonl")
EXCLUDE_LABELS = {"ambiguous", "skip"}

# Visible whitespace / arrow markers the detector renders into ``observed``;
# map ␣/· to a literal space, drop arrows, so observed matches the raw gold value.
_MARKERS = str.maketrans({"␣": " ", "·": " ", "→": "", "↵": ""})


def norm(v) -> str:
    if v is None:
        return ""
    return str(v).translate(_MARKERS).strip()


def load_gold():
    pos, excl, clean, bench_cols = set(), set(), set(), set()
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "typo" not in (r.get("error_types") or []):
            continue
        s, c = r["source"], r["column"]
        key = (s, c, norm(r.get("original_value")))
        bench_cols.add((s, c))
        label = r.get("label")
        if label == "error":
            pos.add(key)
        elif label in EXCLUDE_LABELS:
            excl.add(key)
        else:  # clean (or anything else) -> negative
            clean.add(key)
    # pos takes precedence over excl/clean on any conflicting (s,c,v)
    excl -= pos
    clean -= pos | excl
    return pos, excl, clean, bench_cols


def load_pred(run_dir: Path) -> set:
    """Every typo the model flagged across the ENTIRE dataset (all columns)."""
    cells = set()
    for line in (run_dir / "issues.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error_type") != "typo":
            continue
        s = r.get("source")
        v = norm(r.get("observed"))
        for col in (r.get("columns") or []):
            cells.add((s, col, v))
    return cells


def score(run_dir: Path) -> dict:
    pos, excl, clean, bench_cols = load_gold()
    pred = load_pred(run_dir)
    pred -= excl  # flags on ambiguous/skip cells get immunity (neither TP nor FP)
    tp = pred & pos
    fp = pred - pos  # everything else flagged across the whole CSV = false positive
    fn = pos - pred
    n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
    p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    r = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    fp_labeled_clean = len(fp & clean)  # flags on benchmark-labelled clean cells
    fp_unlabeled = n_fp - fp_labeled_clean  # flags outside the labelled columns/cells
    return {
        "run": run_dir.name,
        "gold_err": len(pos),
        "excluded": len(excl),
        "predicted": len(pred),
        "tp": n_tp, "fp": n_fp, "fn": n_fn,
        "fp_clean": fp_labeled_clean, "fp_unlab": fp_unlabeled,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "_samples": {
            "fp": sorted(list(fp))[:8],
            "fn": sorted(list(fn))[:8],
        },
    }


def main(argv):
    if not argv:
        print("usage: python -m experiments.score_excel500_benchmark <run_dir> ...")
        sys.exit(2)
    rows = [score(Path(d)) for d in argv]
    cols = ["run", "gold_err", "excluded", "predicted",
            "tp", "fp", "fn", "fp_clean", "fp_unlab", "precision", "recall", "f1"]
    w = [max(len(c), max(len(str(r[c])) for r in rows)) for c in cols]
    print(" | ".join(c.ljust(x) for c, x in zip(cols, w)))
    print("-+-".join("-" * x for x in w))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(x) for c, x in zip(cols, w)))
    for r in rows:
        print(f"\n[{r['run']}] sample FP (pred on clean): "
              f"{[t[2] for t in r['_samples']['fp']]}")
        print(f"[{r['run']}] sample FN (missed error): "
              f"{[t[2] for t in r['_samples']['fn']]}")


if __name__ == "__main__":
    main(sys.argv[1:])
