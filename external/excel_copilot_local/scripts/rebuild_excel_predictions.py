"""Regenerate Excel Copilot rein predictions against the CURRENT materialized v8
datasets and rescore.

The Excel findings live in ``excel_finding.csv`` (one row per flagged cell, with
Excel's column *name* in ``header`` and 0-based ``data_row``). We map each flagged
cell to the current ``datasets/<ds>/dirty.csv`` column index by NAME (the mapping
that exactly reproduced the previous ``predictions_excel.jsonl``), applying the
inclusion rule that reproduced the previous 518-cell set:

    include a flagged cell iff it has a non-blank applied ``suggested_fix``
    OR its Excel type is "Invalid email addresses" (highlight-only, no fix).

Cells whose dataset was dropped from v8, whose column no longer exists, or whose
row is now out of range are dropped (they no longer correspond to a real v8 cell).

Coverage scope = the datasets Excel scanned that still exist in the current
manifest (so recall has a valid denominator and dropped datasets don't count).

Writes (overwriting):
  external/rein/results/predictions_excel.jsonl        {dataset,detector,row,col}
  external/rein/results/predictions_excel.jsonl.ran.json  {per_detector:{excel:[...]}}
Then prints the canonical score.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REIN = Path(__file__).resolve().parents[3] / "external" / "rein"
sys.path.insert(0, str(REIN))
import common as C  # noqa: E402

EXCEL_OUT = Path(__file__).resolve().parents[1] / "outputs"
FINDINGS = EXCEL_OUT / "excel_finding.csv"
PRED = REIN / "results" / "predictions_excel.jsonl"


def include(row: dict) -> bool:
    return bool(row["suggested_fix"].strip()) or row["type"] == "Invalid email addresses"


def dirty_header(ds: str, cache: dict) -> dict | None:
    if ds in cache:
        return cache[ds]
    p = C.DATASETS_DIR / ds / "dirty.csv"
    hdr = None
    ncols = 0
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            head = next(reader, [])
            ncols = len(head)
            hdr = {name: i for i, name in enumerate(head)}
            # number of data rows for range checks
            ndata = sum(1 for _ in reader)
        cache[ds] = (hdr, ncols, ndata)
    else:
        cache[ds] = None
    return cache[ds]


def main() -> int:
    manifest = json.loads((C.DATASETS_DIR / "_manifest.json").read_text())
    current_ds = set(manifest["datasets"])

    # previous coverage scope (datasets Excel scanned)
    prev_ran = json.loads((Path(str(PRED) + ".ran.json")).read_text())
    prev_scope = set(prev_ran.get("per_detector", {}).get("excel", []))
    # scope now = scanned datasets that still exist in v8
    scope = sorted(prev_scope & current_ds)

    cache: dict = {}
    preds: set[tuple[str, int, int]] = set()
    stats = {"rows": 0, "kept": 0, "drop_rule": 0, "drop_removed_ds": 0,
             "drop_no_col": 0, "drop_row_oob": 0, "drop_no_scope": 0}
    for row in csv.DictReader(open(FINDINGS, encoding="utf-8-sig")):
        stats["rows"] += 1
        ds = C.slugify_dataset(row["source"])
        dr = row.get("data_row")
        if dr in (None, ""):
            continue
        dr = int(dr)
        if dr < 0:
            continue
        if not include(row):
            stats["drop_rule"] += 1
            continue
        if ds not in current_ds:
            stats["drop_removed_ds"] += 1
            continue
        if ds not in scope:
            stats["drop_no_scope"] += 1
            continue
        info = dirty_header(ds, cache)
        if info is None:
            stats["drop_removed_ds"] += 1
            continue
        hdr, ncols, ndata = info
        j = hdr.get(row["header"])
        if j is None:
            stats["drop_no_col"] += 1
            continue
        if dr >= ndata:
            stats["drop_row_oob"] += 1
            continue
        preds.add((ds, dr, j))
        stats["kept"] += 1

    # write predictions + coverage
    with open(PRED, "w", encoding="utf-8") as f:
        for (ds, i, j) in sorted(preds):
            f.write(json.dumps({"dataset": ds, "detector": "excel", "row": i, "col": j}) + "\n")
    with open(str(PRED) + ".ran.json", "w", encoding="utf-8") as f:
        json.dump({"per_detector": {"excel": scope}}, f)

    print("=== rebuild stats ===")
    for k, v in stats.items():
        print(f"  {k:18s}: {v}")
    print(f"  unique pred cells  : {len(preds)}")
    print(f"  coverage datasets  : {len(scope)} (was {len(prev_scope)})")
    print(f"\nwrote {PRED}")
    print(f"wrote {PRED}.ran.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
