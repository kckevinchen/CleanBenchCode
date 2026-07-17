"""Export likely GOLD-GAP candidates (casing) for benchmark relabeling.

These are values GPT-5.5 flagged as casing inconsistencies that the
benchmark did NOT label (so they scored as false positives), on columns
that genuinely have a dominant case convention -- i.e. probably real
inconsistencies the labelers missed.

Each row carries the source CSV, column, value, the CSV ROW indices where
the value appears (benchmark is keyed by (source,row,column)), the model's
prediction + rationale, and the column's case-convention evidence + a
verdict so you can triage.

    python -m RFT.inconsistency.v1.eval.export_gold_gaps
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pandas as pd

from RFT.inconsistency.v1.eval.run_baseline import _csv_lookup

OUT = Path("outputs/inconsistency_v1_baseline")
MISTAKES = OUT / "gpt55_mistakes_46b4d9166ec0.jsonl"


def _norm(s):
    return " ".join(str(s).split())


def _first(x):
    return x[0] if isinstance(x, list) and x else (x if isinstance(x, str) else "")


def _case_class(v):
    a = [c for c in v if c.isalpha()]
    if not a:
        return "none"
    if all(c.isupper() for c in a):
        return "ALLCAPS"
    if all(c.islower() for c in a):
        return "lower"
    return "Title/Mixed"


# Column types where CASING carries no meaning, so a casing "fix" is an
# over-flag regardless of how dominant a case is: emails (case-insensitive),
# URLs, and free-text (descriptions / notes / comments).
_CASING_IRRELEVANT_KEYS = (
    "email", "e-mail", "url", "website", "web site", "http",
    "desc", "description", "note", "comment", "remark", "narrative",
    "summary", "detail", "free text", "freetext", "message",
)


def _casing_irrelevant(col):
    c = col.lower()
    return any(k in c for k in _CASING_IRRELEVANT_KEYS)


# Proper-noun / address columns: casing is incidental entry style, not a
# defect (neither 'Sullivan' nor 'SULLIVAN' is wrong) -> not a gold gap.
_PROPER_NOUN_KEYS = (
    "name", "applicant", "agency", "site", "district", "individual",
    "organi", "company", "vendor", "supplier", "school", "address",
    "street", "facility", "accredited", "issuer", "owner", "contact",
    "person", "author", "city", "county", "town", "place",
)


def _proper_noun_col(col):
    return any(k in col.lower() for k in _PROPER_NOUN_KEYS)


def _verdict(dom, share, col):
    if _casing_irrelevant(col):
        return "over-flag: casing not meaningful (email/url/free-text)"
    if _proper_noun_col(col):
        return "over-flag: casing on proper noun/address (not a defect)"
    if dom in ("ALLCAPS", "lower") and share >= 0.80:
        return "clear gold gap"
    if dom in ("ALLCAPS", "lower") and share >= 0.65:
        return "likely gold gap"
    if dom == "Title/Mixed":
        return "fuzzy (no single case convention)"
    return "probable over-flag (weak convention)"


def main():
    rows = [json.loads(l) for l in open(MISTAKES)]
    # casing-only false positives
    fp = [r for r in rows
          if r["error_kind"] == "false_positive"
          and _norm(r["value"]).casefold() == _norm(_first(r.get("predicted_corrected_value"))).casefold()
          and _norm(r["value"]) != _norm(_first(r.get("predicted_corrected_value")))]

    csv_map = _csv_lookup()
    # cache columns + convention per (source, column)
    col_cache = {}
    out_rows = []
    for r in fp:
        s, c = r["source"], r["column"]
        key = (s, c)
        if key not in col_cache:
            try:
                df = pd.read_csv(csv_map[s], dtype=str, keep_default_na=False)
                series = [_norm(x) for x in df[c]]
            except Exception:  # noqa: BLE001
                col_cache[key] = None
            else:
                cc = collections.Counter(_case_class(v) for v in series if v)
                dom, domn = cc.most_common(1)[0]
                col_cache[key] = {
                    "series": series,
                    "dom": dom, "share": domn / sum(cc.values()),
                    "distinct": len({v for v in series if v}),
                }
        info = col_cache[key]
        if info is None:
            continue
        val = _norm(r["value"])
        row_idx = [i for i, x in enumerate(info["series"]) if x == val]
        out_rows.append({
            "source": s,
            "column": c,
            "value": val,
            "rows": row_idx,                       # 0-based CSV row indices
            "count": len(row_idx),
            "value_case": _case_class(val),
            "column_dominant_case": info["dom"],
            "column_dominant_share": round(info["share"], 3),
            "column_distinct": info["distinct"],
            "gpt5_corrected_value": _first(r.get("predicted_corrected_value")),
            "gpt5_subpattern": r.get("predicted_subpattern"),
            "gpt5_rationale": r.get("predicted_rationale"),
            "verdict": _verdict(info["dom"], info["share"], c),
            "split": r.get("split"),
        })

    # sort: clear gaps first, then by column
    order = {"clear gold gap": 0, "likely gold gap": 1,
             "fuzzy (no single case convention)": 2,
             "probable over-flag (weak convention)": 3,
             "over-flag: casing on proper noun/address (not a defect)": 4,
             "over-flag: casing not meaningful (email/url/free-text)": 5}
    out_rows.sort(key=lambda r: (order.get(r["verdict"], 9), r["source"], r["column"]))

    # Keep only the actual relabel candidates; drop the over-flag rows
    # (casing-not-meaningful columns like email/free-text, and weak-
    # convention columns) -- those are not gold gaps.
    GAP_VERDICTS = {"clear gold gap", "likely gold gap",
                    "fuzzy (no single case convention)"}
    dropped = [r for r in out_rows if r["verdict"] not in GAP_VERDICTS]
    out_rows = [r for r in out_rows if r["verdict"] in GAP_VERDICTS]

    out_path = OUT / "gpt55_gold_gap_candidates_casing.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(out_rows)} casing gold-gap candidates -> {out_path}")
    print("by verdict:", dict(collections.Counter(r["verdict"] for r in out_rows)))
    print("distinct columns:", len({(r["source"], r["column"]) for r in out_rows}))
    print("total CSV cells to review (sum of counts):",
          sum(r["count"] for r in out_rows))
    print(f"dropped {len(dropped)} over-flag rows (not gold gaps):",
          dict(collections.Counter(r["verdict"] for r in dropped)))


if __name__ == "__main__":
    main()
