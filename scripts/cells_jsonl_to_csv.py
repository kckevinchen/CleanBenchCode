"""Flatten the merged benchmark ``benchmark.json`` into a faithful ``benchmark.csv``.

One CSV row per cell, columns mirroring the ``benchmark.json`` schema 1:1
(same names, same order). Only non-scalar fields are serialised so they
fit a spreadsheet cell:

* the ``error_types`` list -> ``"; "``-joined.
* ``corrected_value`` -> the string as-is when one-directional; a JSON
  array (leading ``[``) when bi-directional (multiple acceptable values);
  empty when null. A value beginning with ``[`` is a multi-value fix.

Every other field is copied verbatim, so the CSV columns line up exactly
with the JSON keys.

Usage::

    python scripts/cells_jsonl_to_csv.py
    python scripts/cells_jsonl_to_csv.py --in benchmark/merged_v1/benchmark.json \
        --out benchmark/merged_v1/benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "benchmark" / "merged_v1" / "benchmark.json"

# Mirrors the benchmark.json key order exactly.
COLUMNS = [
    "source", "row", "column", "original_value",
    "status", "error_types", "corrected_value",
    "dataset",
]
LIST_FIELDS = {"error_types"}


def _cell(field: str, value: Any) -> str:
    if value is None:
        return ""
    if field == "corrected_value" and isinstance(value, list):
        # bi-directional fix -> JSON array so it stays unambiguous.
        return json.dumps(value, ensure_ascii=False)
    if field in LIST_FIELDS and isinstance(value, list):
        return "; ".join("" if x is None else str(x) for x in value)
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default="")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out) if args.out else in_path.with_name("benchmark.csv")

    rows = [json.loads(line) for line in in_path.open() if line.strip()]

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([_cell(col, r.get(col)) for col in COLUMNS])

    print(f"wrote {out_path} ({len(rows)} rows, {len(COLUMNS)} columns)")


if __name__ == "__main__":
    main()
