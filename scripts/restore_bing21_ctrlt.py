#!/usr/bin/env python3
"""Reconstruct original CSV tables from the melted Bing21-CtrlT TSV.

Input layout (tab-separated, no header, CRLF), one line per table COLUMN:
    [1] source file path
    [2] table id  (e.g. "tid0_0.json_line1_Public Case Reviews_A2:F12")
    [3] (always empty)
    [4] column index within the table (0-based)
    [5] column header (first cell of the column)
    [6] the column's data cells joined by "___" (empty cells -> empty splits)

Each output CSV is one table: columns ordered by field 4, header row from
field 5, data rows from transposing the field-6 splits.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

SEP = "___"


def sanitize(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._:-]+", "_", name)
    return name.strip("_")[:150]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/Bing21-CtrlT-processed.head30000.tsv")
    ap.add_argument("--out-dir", default="data/Bing21-CtrlT-restored")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve first-seen order; collect columns per table.
    tables: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
    order: list[tuple[str, str]] = []

    with in_path.open("r", encoding="utf-8", newline="") as fh:
        for lineno, raw in enumerate(fh, 1):
            row = raw.rstrip("\r\n").split("\t")
            if len(row) < 6:
                continue
            src, tid, _blank, col_idx, header, payload = row[:6]
            key = (src, tid)
            if key not in tables:
                order.append(key)
            try:
                ci = int(col_idx)
            except ValueError:
                ci = len(tables[key])
            tables[key].append((ci, header, payload))

    used: set[str] = set()
    written = 0
    for key in order:
        src, tid = key
        cols = sorted(tables[key], key=lambda c: c[0])
        headers = [c[1] for c in cols]
        col_cells = [c[2].split(SEP) for c in cols]
        nrows = max((len(c) for c in col_cells), default=0)

        # Build a unique filename from the source file + table id.
        base = f"{sanitize(src)}__{sanitize(tid)}"
        name = base
        i = 1
        while name in used:
            i += 1
            name = f"{base}__{i}"
        used.add(name)

        out_path = out_dir / f"{name}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as out:
            w = csv.writer(out)
            w.writerow(headers)
            for r in range(nrows):
                w.writerow([cells[r] if r < len(cells) else "" for cells in col_cells])
        written += 1

    print(f"Reconstructed {written} tables -> {out_dir}")


if __name__ == "__main__":
    main()
