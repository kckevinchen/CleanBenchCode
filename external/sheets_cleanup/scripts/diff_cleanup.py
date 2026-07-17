"""Diff each table's before/after CSV export to recover the exact per-cell fix
that Google Sheets' 'Cleanup suggestions' Apply produced, and emit REIN-format
predictions for scoring on the v8 benchmark.

    python scripts/diff_cleanup.py

scripts/apply_and_diff.py saved, per table, a clipboard TSV of the sheet's used
range before and after clicking the accept buttons:
  outputs/applied/<source>.before.tsv
  outputs/applied/<source>.after.tsv
Both come from Sheets' own clipboard copy, so a cell differs only when Apply
actually changed it. This writes:
  outputs/applied_fixes.jsonl / .csv        - one record per changed cell
  <rein>/results/predictions_gsheets.jsonl  - {dataset,detector,row,col} for scoring
  <rein>/results/predictions_gsheets.jsonl.ran.json - coverage (datasets we ran on)

`cell` is the A1 ref; data_row = sheet_row - 2 (benchmark 0-based; row 1 is the
header, first data row is sheet row 2). Predictions use the 0-based data_row as
`row` and the 0-based column index as `col`, matching the Excel baseline.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys

from harness import OUT

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
V8_MANIFEST = os.path.join(REPO_ROOT, "external", "rein", "rein-benchmark",
                           "datasets", "_manifest.json")
REIN_RESULTS = os.path.join(REPO_ROOT, "external", "rein", "results")

APPLIED_DIR = os.path.join(OUT, "applied")
AFTER_SUFFIX = ".after.tsv"
BEFORE_SUFFIX = ".before.tsv"
DETECTOR = "gsheets"


def index_to_col(idx):
    """0-based column index -> letter (0 -> A, 26 -> AA)."""
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def slugify(source):
    name = source[:-4] if source.lower().endswith(".csv") else source
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def parse_tsv(text):
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.reader(io.StringIO(text), delimiter="\t"))


def norm(s):
    return (s or "").strip()


def load_v8_sources():
    """slugified-ds set present in the v8 benchmark (only these can be scored)."""
    m = json.load(open(V8_MANIFEST))
    return set(m["datasets"].keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=DETECTOR)
    args = ap.parse_args()

    v8 = load_v8_sources()
    fixes = []
    preds = []
    covered = set()          # slugified ds we ran on AND that exist in v8
    covered_all = set()      # all slugified ds we ran on (for reporting)
    n_tables = 0
    n_shape_mismatch = 0

    for af in sorted(os.listdir(APPLIED_DIR)):
        if not af.endswith(AFTER_SUFFIX):
            continue
        source = af[: -len(AFTER_SUFFIX)]
        bf = os.path.join(APPLIED_DIR, source + BEFORE_SUFFIX)
        ap_after = os.path.join(APPLIED_DIR, af)
        if not os.path.exists(bf) or os.path.getsize(ap_after) == 0:
            continue
        before = parse_tsv(open(bf, encoding="utf-8").read())
        after = parse_tsv(open(ap_after, encoding="utf-8").read())
        n_tables += 1
        ds = slugify(source)
        covered_all.add(ds)
        in_v8 = ds in v8
        if in_v8:
            covered.add(ds)

        if len(before) != len(after):
            # Row count changed (a Remove slipped through, or export race). We
            # only diff the aligned prefix; misaligned tails are skipped rather
            # than producing spurious fixes.
            n_shape_mismatch += 1

        for r in range(min(len(before), len(after))):
            brow = before[r]
            arow = after[r]
            for c in range(max(len(brow), len(arow))):
                bv = norm(brow[c]) if c < len(brow) else ""
                av = norm(arow[c]) if c < len(arow) else ""
                if bv == av:
                    continue
                col = index_to_col(c)
                sheet_row = r + 1          # CSV row 0 == sheet row 1 (header)
                data_row = sheet_row - 2   # benchmark 0-based data row
                fixes.append({
                    "source": source, "cell": f"{col}{sheet_row}", "column": col,
                    "sheet_row": sheet_row, "data_row": data_row,
                    "before": bv, "after": av,
                })
                if data_row >= 0 and in_v8:
                    preds.append({"dataset": ds, "detector": args.detector,
                                  "row": data_row, "col": c})

    # ---- outputs/applied_fixes.{jsonl,csv} ----
    with open(os.path.join(OUT, "applied_fixes.jsonl"), "w", encoding="utf-8") as f:
        for r in fixes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["source", "cell", "column", "sheet_row", "data_row", "before", "after"]
    with open(os.path.join(OUT, "applied_fixes.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in fixes:
            w.writerow([r[c] for c in cols])

    # ---- rein predictions + coverage ----
    os.makedirs(REIN_RESULTS, exist_ok=True)
    pred_path = os.path.join(REIN_RESULTS, f"predictions_{args.detector}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")
    with open(pred_path + ".ran.json", "w", encoding="utf-8") as f:
        json.dump({"per_detector": {args.detector: sorted(covered)}}, f)

    print(f"{n_tables} tables diffed -> {len(fixes)} changed cells "
          f"({n_shape_mismatch} shape-mismatch)", flush=True)
    print(f"coverage: {len(covered_all)} tables ran, {len(covered)} in v8", flush=True)
    print(f"predictions (data cells, in v8): {len(preds)}", flush=True)
    print(f"wrote {OUT}/applied_fixes.csv, {OUT}/applied_fixes.jsonl", flush=True)
    print(f"wrote {pred_path} (+ .ran.json)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
