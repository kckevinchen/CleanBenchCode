"""Diagnostic: for every cell-scoped Google prediction, show the mapped
(source,row,col), the actual CSV value there, and any gold/ambiguous label at
row-1/row/row+1 (to catch off-by-one). Also verify the 3 ambiguous hits.
"""

import csv
import json
import os
import re

from harness import OUT
from parse_cleanup_html import parse_html

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH = os.path.join(REPO, "benchmark", "merged_v2", "benchmark_cell.json")
MANIFEST = os.path.join(OUT, "manifest.json")
HTML_DIR = os.path.join(OUT, "html")
A1_CELL = re.compile(r"^([A-Z]+)(\d+)$")
CELL_CATS_DEFAULT = {"WHITESPACE", "INVALID DATA"}


def col_letter_to_idx(col):
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def main():
    manifest = {m["source"]: m for m in json.load(open(MANIFEST))}
    gold = {}
    for line in open(BENCH):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        gold[(c["source"], int(c["row"]), c["column"])] = c.get("status") or c.get("label")

    # gold index by (source,row) -> list of (col,status) to check off-by-one
    gold_by_sr = {}
    for (src, row, col), st in gold.items():
        gold_by_sr.setdefault((src, row), []).append((col, st))

    rows_cache = {}

    def get_csv(src):
        if src not in rows_cache:
            path = manifest[src]["csv_path"]
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                rows_cache[src] = list(csv.reader(f))
        return rows_cache[src]

    n_pred = 0
    n_value_match = 0
    print("=== cell-scoped predictions (WHITESPACE / INVALID DATA) ===")
    for fname in sorted(os.listdir(HTML_DIR)):
        if not fname.endswith(".html"):
            continue
        src = fname[:-5]
        parsed = parse_html(open(os.path.join(HTML_DIR, fname)).read())
        header = manifest.get(src, {}).get("header", [])
        for s in parsed["suggestions"]:
            if s["category"] not in CELL_CATS_DEFAULT:
                continue
            for chip in s["ranges"]:
                m = A1_CELL.match(chip.strip())
                if not m:
                    continue
                col_letter, sheet_row = m.group(1), int(m.group(2))
                bench_row = sheet_row - 2
                ci = col_letter_to_idx(col_letter)
                col_name = header[ci] if ci < len(header) else f"<col{ci}>"
                # actual CSV value at sheet coords: sheet row r -> csv line r-1
                csvrows = get_csv(src)
                csv_line = sheet_row - 1
                val = ""
                if 0 <= csv_line < len(csvrows) and ci < len(csvrows[csv_line]):
                    val = csvrows[csv_line][ci]
                n_pred += 1
                # check gold at row-1/row/row+1
                nearby = []
                for dr in (-1, 0, 1):
                    for (gc, gst) in gold_by_sr.get((src, bench_row + dr), []):
                        nearby.append(f"r{bench_row+dr}:{gc}={gst}")
                vmatch = (s["value"].strip() == val.strip()) if val else False
                if vmatch:
                    n_value_match += 1
                print(f"{src[:26]:26} {s['category'][:12]:12} {chip:5} "
                      f"-> row{bench_row} col='{col_name[:18]}' "
                      f"csv_val='{val[:22]}' panel_val='{s['value'][:22]}' "
                      f"vmatch={vmatch} gold_nearby={nearby[:3]}")

    print(f"\ncell predictions: {n_pred} | panel-value matches CSV: {n_value_match}")
    print("(If vmatch is mostly True, row/col mapping is correct and recall=0 is real.)")


if __name__ == "__main__":
    main()
