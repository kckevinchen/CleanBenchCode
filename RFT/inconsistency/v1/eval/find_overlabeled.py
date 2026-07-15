"""Find OVER-LABELED gold error cells that violate per-column inference.

Inconsistency is per-column: a value is only an error if the column has a
dominant convention on the relevant axis AND the value is a minority
deviation from it. This finds gold ERROR cells where that fails:

  * "no-convention"  -- the column has no dominant form on the edit's axis
                        (e.g. the PACK column mixes cans/ctn/ctns/pkgs).
  * "value-conforms" -- the flagged value already matches the column's
                        majority form on that axis (it is not a deviation;
                        the gold is normalising the majority to a minority).

Method: derive the EDIT AXIS from (original -> corrected) -- not the noisy
cluster_kind label -- then compute the column's signature distribution on
exactly that axis (title-case-aware for casing).

    python -m RFT.inconsistency.v1.eval.find_overlabeled
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pandas as pd

from RFT.inconsistency.v1.eval.run_baseline import _csv_lookup

BENCH = Path("benchmark/benchmark_inconsistency_20260515T010112Z_gpt5-family.jsonl")
OUT = Path("outputs/inconsistency_v1_baseline/gpt55_overlabeled_candidates.jsonl")
DOM_THRESHOLD = 0.50  # below this dominant share -> no clear convention


def _ws(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _alnum(s):
    return "".join(c for c in str(s).casefold() if c.isalnum())


def _digits(s):
    return re.sub(r"\D", "", str(s))


def _first(x):
    return x[0] if isinstance(x, list) and x else (x if isinstance(x, str) else "")


def edit_axis(o, c):
    o_raw, c_raw = o, c
    if not _ws(c):
        return "empty-correction"   # gold bug: error cell with no fix
    o, c = _ws(o), _ws(c)
    if o == c:
        return "identical" if o_raw == c_raw else "whitespace"
    if o.casefold() == c.casefold():
        return "casing"
    if _alnum(o) == _alnum(c):
        return "punctuation"    # same letters/digits, differ in punct/spacing
    if len(_digits(o)) >= 6 and _digits(o) == _digits(c):
        return "number-regroup"
    return "text/structural"    # not deterministically checkable


def case_sig(v):
    a = [ch for ch in v if ch.isalpha()]
    if not a:
        return "none"
    if all(ch.isupper() for ch in a):
        return "UPPER"
    if all(ch.islower() for ch in a):
        return "lower"
    if v == v.title():
        return "Title"
    return "Mixed"


def punct_sig(v):
    return "".join(c for c in v if not c.isalnum() and not c.isspace())


def numshape_sig(v):
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", v))


AXIS_SIG = {"casing": case_sig, "punctuation": punct_sig,
            "number-regroup": numshape_sig}


def main():
    # distinct gold error cells -> (raw original, corrected); keyed on the
    # whitespace-normalised value to dedupe, but keep the RAW original so the
    # whitespace axis is detectable.
    cells = {}
    for l in open(BENCH):
        r = json.loads(l)
        if r["label"] != "error":
            continue
        cells[(r["source"], r["column"], _ws(r["original_value"]))] = \
            (r["original_value"], _first(r.get("corrected_value")))

    csv_map = _csv_lookup()
    sig_cache = {}

    def col_sig_dist(s, c, axis):
        key = (s, c, axis)
        if key in sig_cache:
            return sig_cache[key]
        fn = AXIS_SIG[axis]
        df = pd.read_csv(csv_map[s], dtype=str, keep_default_na=False)
        dist = collections.Counter(fn(_ws(v)) for v in df[c] if _ws(v))
        sig_cache[key] = dist
        return dist

    by_axis = collections.Counter()
    overlab = collections.Counter()
    reason_ct = collections.Counter()
    out_rows = []
    examples = collections.defaultdict(list)

    for (s, c, v), (v_raw, corr) in cells.items():
        axis = edit_axis(v_raw, corr)
        by_axis[axis] += 1
        if axis in ("text/structural", "whitespace"):
            continue             # whitespace = trivial-but-real; text = not evaluable
        if axis in ("empty-correction", "identical"):  # gold bugs
            overlab[axis] += 1
            reason_ct["gold bug: " + axis] += 1
            out_rows.append({"source": s, "column": c, "value": v,
                             "gold_corrected_value": corr, "edit_axis": axis,
                             "reason": "gold bug (" + axis + ")"})
            examples[axis].append((c, v, "gold bug", None))
            continue
        try:
            dist = col_sig_dist(s, c, axis)
        except Exception:  # noqa: BLE001
            continue
        tot = sum(dist.values())
        if tot < 3:
            continue
        dom_sig, dom_n = dist.most_common(1)[0]
        dom_share = dom_n / tot
        vsig = AXIS_SIG[axis](v)
        reason = None
        if vsig == dom_sig:
            reason = "value conforms to column majority"
        elif dom_share < DOM_THRESHOLD:
            reason = "no dominant convention"
        if reason:
            overlab[axis] += 1
            reason_ct[reason] += 1
            out_rows.append({
                "source": s, "column": c, "value": v,
                "gold_corrected_value": corr, "edit_axis": axis,
                "reason": reason, "dom_share": round(dom_share, 2),
                "dom_sig": dom_sig, "value_sig": vsig,
            })
            examples[axis].append((c, v, reason, round(dom_share, 2)))

    with OUT.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("gold error cells by EDIT AXIS:", dict(by_axis))
    print()
    print("OVER-LABELED / problematic (violate per-column inference):")
    for ax in ("casing", "punctuation", "number-regroup", "empty-correction", "identical"):
        ev = by_axis[ax]
        print(f"  {ax:16} {overlab[ax]:3} / {ev:<3}")
    print(f"  (whitespace: {by_axis['whitespace']} trivial-but-real, not counted; "
          f"text/structural: {by_axis['text/structural']} not evaluable)")
    print("reasons:", dict(reason_ct))
    print(f"\ntotal over-labeled: {len(out_rows)} -> {OUT}")
    for ax in ("casing", "punctuation", "number-regroup"):
        if not examples[ax]:
            continue
        print(f"\n-- {ax} --")
        for col, v, reason, sh in examples[ax][:6]:
            print(f"   [{col[:20]:20}] {v[:30]!r:32} dom_share={sh}  ({reason})")


if __name__ == "__main__":
    main()
