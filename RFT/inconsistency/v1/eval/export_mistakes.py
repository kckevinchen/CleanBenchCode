"""Export GPT-5.5's mistakes on the inconsistency benchmark to JSONL.

Reads the cached predictions for one (model, effort, temperature) and
emits one line per mistake:

  * "miss"            -- a gold inconsistency the model did NOT flag (FN)
  * "false_positive"  -- a value the model flagged that is not gold-error
                         and not ambiguous (on a labeled OR clean column)
  * "wrong_correction"-- a gold cell the model flagged, but with a
                         correction outside the gold's acceptable set

Use temp=0 for the deterministic, trustworthy run (gpt-5.5 is stochastic
at temp=1). Default writes the temp=0 comprehensive-rulebook run.

    python -m RFT.inconsistency.v1.eval.export_mistakes \
        --model gpt-5.5 --effort none --temperature 0
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

# One uniform schema for every row (miss / false_positive / wrong_correction):
FIELDS = [
    "error_kind", "source", "column", "value", "column_type",
    "gold_is_error", "gold_corrected_value", "gold_subpattern",
    "gpt5_flagged", "gpt5_corrected_value", "gpt5_subpattern", "gpt5_rationale",
    "split",
]


def _s(x):
    """List of acceptable forms -> readable string; pass through otherwise."""
    if isinstance(x, list):
        return " | ".join(str(e) for e in x)
    return x


def mkrow(kind, source, column, value, column_type, split, *,
          gold_corr=None, gold_sub=None,
          gpt5_corr=None, gpt5_sub=None, gpt5_rat=None):
    return {
        "error_kind": kind,
        "source": source,
        "column": column,
        "value": value,
        "column_type": column_type,
        "gold_is_error": kind in ("miss", "wrong_correction"),
        "gold_corrected_value": _s(gold_corr),
        "gold_subpattern": gold_sub,
        "gpt5_flagged": kind in ("false_positive", "wrong_correction"),
        "gpt5_corrected_value": _s(gpt5_corr),
        "gpt5_subpattern": gpt5_sub,
        "gpt5_rationale": gpt5_rat,
        "split": split,
    }

from RFT.inconsistency.v1.eval.run_baseline import (
    BENCHMARK, _acceptable_set, _auditable_columns, _csv_lookup,
    _norm, _prompt_hash,
)

CACHE = Path("outputs/inconsistency_v1_baseline/cache")
OUT = Path("outputs/inconsistency_v1_baseline")
MANIFEST = "RFT/inconsistency/v1/data/manifest.json"


def _parse_full(raw):
    """Return {norm_value: {value, corrected_value, kind, rationale}}."""
    if not raw:
        return {}
    s = raw.strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return {}
    out = {}
    for it in obj.get("findings", []) or []:
        v = it.get("value")
        if isinstance(v, str):
            out[_norm(v)] = {
                "value": v,
                "corrected_value": it.get("corrected_value"),
                "kind": it.get("kind"),
                "rationale": it.get("rationale"),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--benchmark", default=None,
                    help="Override the benchmark JSONL (re-score cached "
                         "predictions against a different gold set).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bench = a.benchmark or BENCHMARK
    ph = _prompt_hash(a.model, a.effort, a.temperature)
    out_path = Path(a.out) if a.out else OUT / f"gpt55_mistakes_{ph}.jsonl"

    man = json.load(open(a.manifest))
    heldout = set(man["sources"]["eval"])

    def split_of(s):
        return "heldout" if s in heldout else "tune"

    # gold: error cells with kind + acceptable set; ambiguous set
    gold = collections.defaultdict(dict)   # (s,c) -> {nv: {orig, corr, kind, acc}}
    amb = collections.defaultdict(set)
    for line in open(bench):
        r = json.loads(line)
        k = (r["source"], r["column"]); nv = _norm(r["original_value"])
        if r["label"] == "error":
            gold[k][nv] = {
                "orig": r["original_value"],
                "corr": r.get("corrected_value"),
                "kind": (r.get("cluster_kinds") or [None])[0],
                "acc": _acceptable_set(r),
            }
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
        return None if d.get("raw") is None else _parse_full(d["raw"])

    csv_map = _csv_lookup()
    rows = []
    counts = collections.Counter()
    no_cache = 0

    # A column is "labeled" if it carries ANY error or ambiguous gold; the
    # source set is every benchmark source (matches run_baseline).
    labeled_keys = set(gold) | set(amb)
    all_sources = sorted({s for s, _ in labeled_keys})

    for s in all_sources:
        src_csv = csv_map.get(s)
        if not src_csv:
            continue
        labeled = {c for (ss, c) in labeled_keys if ss == s}
        sp = split_of(s)
        for c, _vals in _auditable_columns(src_csv).items():
            pred = pred_for(s, c)
            if pred is None:
                no_cache += 1
                continue
            if c in labeled:
                g, am = gold.get((s, c), {}), amb.get((s, c), set())
                # misses + wrong corrections
                for nv, rec in g.items():
                    if nv not in pred:
                        rows.append(mkrow("miss", s, c, rec["orig"], "labeled", sp,
                                          gold_corr=rec["corr"], gold_sub=rec["kind"]))
                        counts["miss"] += 1
                    else:
                        pc = pred[nv].get("corrected_value")
                        ok = isinstance(pc, str) and _norm(pc) in rec["acc"]
                        if not ok:
                            rows.append(mkrow(
                                "wrong_correction", s, c, rec["orig"], "labeled", sp,
                                gold_corr=rec["corr"], gold_sub=rec["kind"],
                                gpt5_corr=pc, gpt5_sub=pred[nv].get("kind"),
                                gpt5_rat=pred[nv].get("rationale")))
                            counts["wrong_correction"] += 1
                # false positives on labeled cols
                for nv, p in pred.items():
                    if nv in g or nv in am:
                        continue
                    rows.append(mkrow("false_positive", s, c, p["value"], "labeled", sp,
                                      gpt5_corr=p.get("corrected_value"),
                                      gpt5_sub=p.get("kind"), gpt5_rat=p.get("rationale")))
                    counts["false_positive"] += 1
            else:
                # clean column: every prediction is a false positive
                for nv, p in pred.items():
                    rows.append(mkrow("false_positive", s, c, p["value"], "clean", sp,
                                      gpt5_corr=p.get("corrected_value"),
                                      gpt5_sub=p.get("kind"), gpt5_rat=p.get("rationale")))
                    counts["false_positive"] += 1

    # group for readability: all misses, then wrong-corrections, then FPs
    order = {"miss": 0, "wrong_correction": 1, "false_positive": 2}
    rows.sort(key=lambda r: (order[r["error_kind"]], r["source"], r["column"],
                             str(r["value"])))

    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"prompt_hash={ph}  (cache-miss cols: {no_cache})")
    print(f"wrote {len(rows)} rows (unified schema) -> {out_path}")
    print(f"             and CSV -> {csv_path}")
    print("by error_kind:", dict(counts))
    print("  miss by gold_subpattern:",
          dict(collections.Counter(r["gold_subpattern"]
                                   for r in rows if r["error_kind"] == "miss")))


if __name__ == "__main__":
    main()
