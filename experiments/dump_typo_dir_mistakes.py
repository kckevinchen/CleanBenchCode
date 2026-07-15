"""Dump every TP/FP/FN a typo run produces vs benchmark/typo -- one file PER MODE.

Writes two JSONL files for a run (prefix from ``--out-prefix``):
  * ``<prefix>.eval1_low_value_as_fp.jsonl``   -- low-value flags ARE errors:
      records = all TP + all FN + all FP.
  * ``<prefix>.eval2_low_value_as_skip.jsonl`` -- low-value flags are skipped:
      records = all TP + all FN + FP where ``low_value=false``.

Unified record schema (ambiguous + skipped + excluded cells are dropped):
  ``{error_type, source, original, gold_correction, model_correction}``
  * ``"tp"`` -- run caught a real gold error: BOTH ``gold_correction`` (the
    canonical gold fix) and ``model_correction`` (the model's fix) populated.
  * ``"fp"`` -- run flagged a non-gold-error value: ``gold_correction=null``
    (gold says clean), ``model_correction`` = the model's proposed fix.
  * ``"fn"`` -- a gold ``error`` the run missed: ``model_correction=null``,
    ``gold_correction`` = the canonical gold fix.

    python -m experiments.dump_typo_dir_mistakes <run_dir> --out-prefix outputs/gpt55_clean
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.score_typo_dir_benchmark import (
    GOLD,
    is_excluded,
    load_skip_keys,
    norm,
)


def load_gold_records():
    by_key: dict = {}
    cell_count: Counter = Counter()
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = (r["column"] or "").strip()
        if is_excluded(r["source"], c):
            continue
        key = (r["source"], c, norm(r.get("original_value")))
        cell_count[key] += 1
        prev = by_key.get(key)
        if prev is None or (r.get("label") == "error" and prev["label"] != "error"):
            by_key[key] = {
                "label": r.get("label"),
                "original_value": r.get("original_value"),
                "corrected_value": r.get("corrected_value"),
                "llm_suggested_value": r.get("llm_suggested_value"),
                "low_value": bool(r.get("low_value")),
            }
    return by_key, cell_count


def load_pred_records(run_dir: Path):
    """(source,column,normvalue) -> issue detail + cell count (affected_rows_total)."""
    by_key: dict = {}
    for line in (run_dir / "issues.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error_type") != "typo":
            continue
        s, v = r.get("source"), norm(r.get("observed"))
        tot = r.get("affected_rows_total") or 1
        for col in (r.get("columns") or []):
            col = (col or "").strip()
            if is_excluded(s, col):
                continue
            k = (s, col, v)
            rec = by_key.setdefault(k, {
                "observed": r.get("observed"),
                "corrected_value": r.get("corrected_value"),
                "suggested_fix": r.get("suggested_fix"),
                "explanation": r.get("explanation"),
                "cells": 0,
            })
            rec["cells"] = max(rec["cells"], tot)
    return by_key


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out-prefix", required=True,
                    help="Output path prefix; '.eval1_*' / '.eval2_*' .jsonl appended.")
    args = ap.parse_args()

    gold, gold_cells = load_gold_records()
    pred = load_pred_records(args.run_dir)
    skip = load_skip_keys()

    pos = {k for k, g in gold.items() if g["label"] == "error"}
    amb = {k for k, g in gold.items() if g["label"] == "ambiguous"}
    low = {k for k, g in gold.items() if g["label"] == "clean" and g["low_value"]}
    excluded = (amb | skip) - pos

    tp_keys = set(pred) & pos
    fp_keys = set(pred) - pos - excluded
    fn_keys = pos - set(pred)

    # Unified schema for every outcome:
    #   error_type | source | original | gold_correction | model_correction
    #   TP -> caught a real error: BOTH gold fix and model fix populated.
    #   FP -> gold says clean (gold_correction=null); model proposed a fix.
    #   FN -> model missed it (model_correction=null); gold has the fix.
    tp_records = []
    for k in sorted(tp_keys):
        s, _, _ = k
        g, pr = gold[k], pred[k]
        tp_records.append({
            "error_type": "tp",
            "source": s,
            "original": g["original_value"],
            "gold_correction": g["llm_suggested_value"],
            "model_correction": pr.get("suggested_fix") or pr.get("corrected_value"),
        })
    fp_records = []  # (is_low, record)
    for k in sorted(fp_keys):
        s, c, _ = k
        pr = pred[k]
        fp_records.append((k in low, {
            "error_type": "fp",
            "source": s,
            "column": c,
            "original": pr["observed"],
            "gold_correction": None,
            "model_correction": pr.get("suggested_fix") or pr.get("corrected_value"),
        }))
    fn_records = []
    for k in sorted(fn_keys):
        s, c, _ = k
        g = gold[k]
        fn_records.append({
            "error_type": "fn",
            "source": s,
            "column": c,
            "original": g["original_value"],
            # The canonical clean fix (corrected_value is operator prose like
            # `Change "Ptd" to "Ltd"`, which often isn't even the full value).
            "gold_correction": g["llm_suggested_value"],
            "model_correction": None,
        })

    modes = [
        ("eval1_low_value_as_fp", False),   # keep all FP
        ("eval2_low_value_as_skip", True),  # drop low_value FP
    ]
    for suffix, drop_low in modes:
        out = Path(f"{args.out_prefix}.{suffix}.jsonl")
        n_fp = 0
        with out.open("w", encoding="utf-8") as f:
            for rec in tp_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for is_low, rec in fp_records:
                if drop_low and is_low:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_fp += 1
            for rec in fn_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {out.name}: {len(tp_records)} TP + {n_fp} FP + {len(fn_records)} FN "
              f"= {len(tp_records) + n_fp + len(fn_records)}")


if __name__ == "__main__":
    main()
