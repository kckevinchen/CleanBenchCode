"""Per-iteration analysis for the gpt-5.5 rulebook-tuning loop.

Reads the cached predictions for one (model, effort) -> prompt_hash and
reports, split by TUNE (train+valid sources) vs HELD-OUT (eval sources):
  * P / R / F1 (distinct-value, ambiguous-immune, incl. clean-col FP)
  * FN miss-classes by sub-pattern   (TUNE only -- what to fix)
  * FP over-flag modes               (TUNE only -- what to stop)

Overfitting guard: only TUNE composition is shown for editing; HELD-OUT
is measured but never inspected. Edit the rulebook for general classes
seen in TUNE, then confirm HELD-OUT F1 moves the same direction.

    python -m RFT.inconsistency.v1.eval.analyze --model gpt-5.5 --effort none
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from RFT.inconsistency.v1.eval.run_baseline import (
    BENCHMARK, _acceptable_set, _auditable_columns, _csv_lookup,
    _norm, _parse_findings, _prompt_hash,
)

CACHE = Path("outputs/inconsistency_v1_baseline/cache")
MANIFEST = "RFT/inconsistency/v1/data/manifest.json"


def _fp_mode(v, pc):
    if pc is None or _norm(pc) == _norm(v):
        return "noop"
    if _norm(v).casefold() == _norm(str(pc)).casefold():
        return "casing-only"
    a = "".join(ch for ch in v.casefold() if ch.isalnum())
    b = "".join(ch for ch in str(pc).casefold() if ch.isalnum())
    if a == b:
        return "punct/space"
    return "substantive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--examples", type=int, default=10)
    a = ap.parse_args()
    ph = _prompt_hash(a.model, a.effort, a.temperature)

    man = json.load(open(a.manifest))
    heldout = set(man["sources"]["eval"])
    tune = set(man["sources"]["train"]) | set(man["sources"]["valid"])

    def split_of(s):
        return "heldout" if s in heldout else ("tune" if s in tune else "other")

    gold = collections.defaultdict(dict)
    amb = collections.defaultdict(set)
    for line in open(BENCHMARK):
        r = json.loads(line)
        k = (r["source"], r["column"]); nv = _norm(r["original_value"])
        if r["label"] == "error":
            kind = (r.get("cluster_kinds") or ["<none>"])[0]
            gold[k][nv] = {"orig": r["original_value"], "corr": r.get("corrected_value"),
                           "kind": kind, "acc": _acceptable_set(r)}
        elif r["label"] == "ambiguous":
            amb[k].add(nv)

    def pred_for(s, c):
        key = hashlib.sha256(f"{ph}|{s}|{c}".encode()).hexdigest()[:16]
        p = CACHE / f"{key}.json"
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except Exception:
            return None
        return None if d.get("raw") is None else _parse_findings(d["raw"])

    csv_map = _csv_lookup()
    agg = collections.defaultdict(lambda: {"tp": 0, "fp_lab": 0, "fp_clean": 0, "fn": 0})
    fn_rows, fp_rows = [], []
    miss_cache = 0

    labeled_keys = set(gold) | set(amb)
    for s in sorted({src for src, _ in labeled_keys}):
        sp = split_of(s)
        csv = csv_map.get(s)
        if not csv:
            continue
        labeled = {c for (ss, c) in labeled_keys if ss == s}
        for c, vals in _auditable_columns(csv).items():
            pred = pred_for(s, c)
            if pred is None:
                miss_cache += 1
                continue
            if c in labeled:
                g, am = gold[(s, c)], amb[(s, c)]
                for nv, rec in g.items():
                    if nv in pred:
                        agg[sp]["tp"] += 1
                    else:
                        agg[sp]["fn"] += 1; fn_rows.append((sp, s, c, rec))
                for nv, pc in pred.items():
                    if nv in g or nv in am:
                        continue
                    agg[sp]["fp_lab"] += 1; fp_rows.append((sp, "labeled", s, c, nv, pc))
            else:
                for nv, pc in pred.items():
                    agg[sp]["fp_clean"] += 1; fp_rows.append((sp, "clean", s, c, nv, pc))

    def prf(d):
        tp = d["tp"]; fp = d["fp_lab"] + d["fp_clean"]; fn = d["fn"]
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        return P, R, F

    print(f"prompt_hash={ph}  model={a.model}/{a.effort}  (cache-miss cols: {miss_cache})")
    full = {k: agg["tune"][k] + agg["heldout"][k] for k in ("tp", "fp_lab", "fp_clean", "fn")}
    for name, d in [("TUNE", agg["tune"]), ("HELD-OUT", agg["heldout"]), ("FULL", full)]:
        P, R, F = prf(d)
        print(f"  {name:9} TP={d['tp']:3} FN={d['fn']:3} FPlab={d['fp_lab']:3} "
              f"FPclean={d['fp_clean']:3} | P={P:.3f} R={R:.3f} F1={F:.3f}")

    # FN miss-classes (TUNE)
    tune_fn = collections.Counter(); tune_gold = collections.Counter()
    for (s, c), g in gold.items():
        if split_of(s) == "tune":
            for rec in g.values():
                tune_gold[rec["kind"]] += 1
    for sp, s, c, rec in fn_rows:
        if sp == "tune":
            tune_fn[rec["kind"]] += 1
    print("\n=== TUNE recall by kind (missed / gold) ===")
    for k in sorted(tune_gold, key=lambda x: -tune_gold[x]):
        g = tune_gold[k]; m = tune_fn[k]
        print(f"  {k:16} {g-m:3}/{g:<3} = {(g-m)/g:.0%}   (missed {m})")

    random.seed(0)
    bk = collections.defaultdict(list)
    for sp, s, c, rec in fn_rows:
        if sp == "tune":
            bk[rec["kind"]].append((c, rec))
    print("\n=== TUNE FN examples (orig -> gold corrected) ===")
    for k in ["pattern", "punctuation", "text", "casing", "<none>"]:
        rows = bk.get(k, [])
        if not rows:
            continue
        print(f"-- {k} ({len(rows)}) --")
        for c, rec in random.sample(rows, min(a.examples, len(rows))):
            print(f"   [{c[:18]:18}] {rec['orig'][:32]!r:34} -> {str(rec['corr'])[:26]!r}")

    # FP modes (TUNE)
    fpm = collections.Counter(); fp_ex = collections.defaultdict(list)
    for sp, kc, s, c, v, pc in fp_rows:
        if sp == "tune":
            m = _fp_mode(v, pc); fpm[(kc, m)] += 1; fp_ex[(kc, m)].append((c, v, pc))
    print("\n=== TUNE FP modes (col-kind, edit-mode, count) ===")
    for (kc, m), n in fpm.most_common():
        print(f"  {kc:8} {m:14} {n}")
    print("\n=== TUNE clean-col FP examples (by mode) ===")
    for m in ["casing-only", "substantive", "punct/space", "noop"]:
        rows = fp_ex.get(("clean", m), [])
        if not rows:
            continue
        print(f"-- clean / {m} ({len(rows)}) --")
        for c, v, pc in random.sample(rows, min(6, len(rows))):
            print(f"   [{c[:18]:18}] {v[:30]!r:32} -> {str(pc)[:24]!r}")


if __name__ == "__main__":
    main()
