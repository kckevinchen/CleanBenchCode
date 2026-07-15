"""Score typo detection runs against the benchmark/typo export -- two ways.

Gold = ``benchmark/typo/typo_benchmark.jsonl`` (one row per (source,row,column)
CELL): ``error``=positive, ``clean``=negative, ``ambiguous``=excluded.
``benchmark/typo/typo_skipped.jsonl`` (label ``skip``) is ALWAYS excluded.

The ``low_value`` cells (clean cells the LLM flagged where its fix differs from
the original by only spacing or a single punctuation mark) are scored two ways:
  * low_value_as_fp   -- normal clean: flagging one is a FP. STRICT (Eval 1).
  * low_value_as_skip -- excluded (immunity): flagging has no effect (Eval 2).

Two granularities are reported:
  * PER-CELL  -- each (source,row,column) cell counts; a value that repeats
    across K cells counts K times (gold from the benchmark rows; the run's
    ``affected_rows_total`` weights flags on un-benchmarked values).
  * PER-VALUE -- each distinct (source,column,value) counts once (deduped).

Entire-CSV: a flag on any non-gold-error, non-excluded value is a FP.

    python -m experiments.score_typo_dir_benchmark <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Default benchmark dir; override with TYPO_BENCH_DIR=benchmark/<name>.
BENCH_DIR = Path(os.environ.get("TYPO_BENCH_DIR", "benchmark/typo_excel_500"))
GOLD = BENCH_DIR / "typo_benchmark.jsonl"
SKIPPED = BENCH_DIR / "typo_skipped.jsonl"

_MARKERS = str.maketrans({"␣": " ", "·": " ", "→": "", "↵": ""})


def norm(v) -> str:
    if v is None:
        return ""
    return str(v).translate(_MARKERS).strip()


def _load_exclusions():
    """Operator csv-/column-scope exclusions (active only). A flag on an
    excluded csv or column must NOT count -- those slices were dismissed."""
    path = BENCH_DIR / "typo_exclusions.jsonl"
    csvs, cols = set(), set()
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("deleted"):
                continue
            s = r.get("source")
            if r.get("scope") == "csv":
                csvs.add(s)
            elif r.get("scope") == "columns":
                for c in (r.get("columns") or []):
                    cols.add((s, (c or "").strip()))
    return csvs, cols


EXCL_CSVS, EXCL_COLS = _load_exclusions()


def is_excluded(source, column) -> bool:
    return source in EXCL_CSVS or (source, column) in EXCL_COLS


def load_gold():
    """Return (cells, pos, amb, clean, low).

    ``cells`` is the per-cell list ``[(source, column, normvalue, label, low)]``,
    one per benchmark row. ``pos/amb/clean/low`` are deduped value-key sets.
    """
    cells = []
    pos, amb, clean, low = set(), set(), set(), set()
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        s, c = r["source"], (r["column"] or "").strip()
        if is_excluded(s, c):
            continue
        v = norm(r.get("original_value"))
        lab = r.get("label")
        lv = bool(r.get("low_value"))
        cells.append((s, c, v, lab, lv))
        key = (s, c, v)
        if lab == "error":
            pos.add(key)
        elif lab == "ambiguous":
            amb.add(key)
        elif lab == "clean":
            clean.add(key)
            if lv:
                low.add(key)
    amb -= pos
    clean -= pos
    low -= pos
    return cells, pos, amb, clean, low


def load_skip_keys() -> set:
    # Exports name the skip sidecar either typo_skipped.jsonl or
    # typo_benchmark_skipped.jsonl; accept whichever exists (empty if neither).
    path = SKIPPED if SKIPPED.exists() else (BENCH_DIR / "typo_benchmark_skipped.jsonl")
    if not path.exists():
        return set()
    return {
        (r["source"], (r["column"] or "").strip(), norm(r.get("original_value")))
        for r in (json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip())
    }


def load_pred(run_dir: Path):
    """Return (pred_set, weight) where weight[(s,c,v)] = max affected_rows_total
    (number of source cells that value occupies)."""
    pred, weight = set(), {}
    for line in (run_dir / "issues.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error_type") != "typo":
            continue
        s, v = r.get("source"), norm(r.get("observed"))
        tot = r.get("affected_rows_total") or 1
        for col in (r.get("columns") or []):
            col = (col or "").strip()  # headers may carry stray whitespace
            if is_excluded(s, col):  # flag on a dismissed csv/column -> ignore
                continue
            k = (s, col, v)
            pred.add(k)
            weight[k] = max(weight.get(k, 0), tot)
    return pred, weight


def _f1(tp, fp, fn):
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 1.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F1


def score_per_value(pos, amb, clean, low, skip, pred, mode):
    excluded = amb | skip | (low if mode == "low_value_as_skip" else set())
    p = pred - excluded
    tp, fp, fn = len(p & pos), len(p - pos), len(pos - pred)
    return tp, fp, fn, *_f1(tp, fp, fn)


def score_per_cell(cells, pos_keys, skip, pred, weight, mode):
    """Per (source,row,column) cell. Benchmark rows supply gold cell counts;
    un-benchmarked flagged values add affected_rows_total cells as FP."""
    tp = fp = fn = 0
    bkeys = set()
    for (s, c, v, lab, lv) in cells:
        key = (s, c, v)
        bkeys.add(key)
        if lab == "ambiguous":
            continue
        if mode == "low_value_as_skip" and lab == "clean" and lv:
            continue
        flagged = key in pred
        if lab == "error":
            tp += 1 if flagged else 0
            fn += 0 if flagged else 1
        elif lab == "clean":
            fp += 1 if flagged else 0
    # Flags on values that have no benchmark cell -> unlabeled FP, weighted by
    # how many source cells they occupy. Skip-listed values get immunity.
    for k, w in weight.items():
        if k in bkeys or k in skip:
            continue
        fp += w
    return tp, fp, fn, *_f1(tp, fp, fn)


def main(argv):
    if not argv:
        print("usage: python -m experiments.score_typo_dir_benchmark <run_dir> ...")
        sys.exit(2)
    cells, pos, amb, clean, low = load_gold()
    skip = load_skip_keys()
    print(f"gold (per-value): error={len(pos)} clean={len(clean)} (low_value={len(low)}) "
          f"ambiguous={len(amb)} | skipped values={len(skip)}\n")
    hdr = f"{'run':42} {'mode':18} {'TP':>4} {'FP':>5} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}"
    print(hdr)
    print("-" * len(hdr))
    for d in argv:
        run = Path(d)
        pred, _ = load_pred(run)
        for mode in ("low_value_as_fp", "low_value_as_skip"):
            tp, fp, fn, P, R, F1 = score_per_value(pos, amb, clean, low, skip, pred, mode)
            print(f"{run.name[:42]:42} {mode:18} {tp:>4} {fp:>5} {fn:>4} "
                  f"{P:>6.3f} {R:>6.3f} {F1:>6.3f}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
