"""Export every Google Sheets Cleanup detection to a flat CSV, one row per
suggestion, keyed by source table.

Columns:
  source, dataset, category, scope, cell_ref, column_letter, column_name,
  sheet_row, bench_row, flagged_value, suggested_value, action, raw_text
"""

import csv
import json
import os
import re

from harness import OUT
from parse_cleanup_html import parse_html

MANIFEST = os.path.join(OUT, "manifest.json")
MANIFEST_CLEAN = os.path.join(OUT, "manifest_clean.json")
HTML_DIR = os.path.join(OUT, "html")
OUT_CSV = os.path.join(OUT, "sheets_cleanup_findings.csv")
A1_CELL = re.compile(r"^([A-Z]+)(\d+)$")

ROW_CATS = {"DUPLICATE ROWS"}
COL_CATS = {"COLUMN FORMAT"}


def col_to_name(letter, header):
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    idx -= 1
    return (header[idx] if 0 <= idx < len(header) else ""), idx


def main():
    manifest = {m["source"]: m for m in json.load(open(MANIFEST))}
    bench_sources = set(manifest)
    if os.path.exists(MANIFEST_CLEAN):
        for m in json.load(open(MANIFEST_CLEAN)):
            manifest.setdefault(m["source"], m)

    # source -> dataset, from the benchmark (manifest doesn't carry it).
    bench = os.path.join(
        os.path.dirname(MANIFEST), "..", "..", "..", "benchmark", "merged_v2",
        "benchmark_cell.json",
    )
    src2ds = {}
    if os.path.exists(bench):
        for line in open(bench):
            line = line.strip()
            if line:
                r = json.loads(line)
                src2ds.setdefault(r["source"], r.get("dataset", ""))

    fields = [
        "source", "set", "dataset", "category", "scope", "cell_ref", "column_letter",
        "column_name", "sheet_row", "bench_row", "flagged_value",
        "suggested_value", "action", "raw_text",
    ]
    n = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for fname in sorted(os.listdir(HTML_DIR)):
            if not fname.endswith(".html"):
                continue
            src = fname[:-5]
            header = manifest.get(src, {}).get("header", [])
            dataset = src2ds.get(src, "")
            table_set = "benchmark" if src in bench_sources else "clean"
            parsed = parse_html(open(os.path.join(HTML_DIR, fname)).read())
            for s in parsed["suggestions"]:
                cat = s["category"]
                suggested = s["extra"].get("to", "")
                if cat in ROW_CATS:
                    for chip in s["ranges"]:
                        if not chip.isdigit():
                            continue
                        w.writerow({
                            "source": src, "set": table_set, "dataset": dataset,
                            "category": cat,
                            "scope": "row", "cell_ref": "", "column_letter": "",
                            "column_name": "", "sheet_row": chip,
                            "bench_row": int(chip) - 2,
                            "flagged_value": s["value"], "suggested_value": "",
                            "action": s["action"], "raw_text": s["raw_text"][:200],
                        })
                        n += 1
                elif cat in COL_CATS:
                    col = s["ranges"][0].split(":")[0] if s["ranges"] else ""
                    name, _ = col_to_name(col, header) if col else ("", -1)
                    w.writerow({
                        "source": src, "set": table_set, "dataset": dataset,
                        "category": cat,
                        "scope": "column", "cell_ref": "", "column_letter": col,
                        "column_name": name, "sheet_row": "", "bench_row": "",
                        "flagged_value": s["extra"].get("from", s["value"]),
                        "suggested_value": suggested, "action": s["action"],
                        "raw_text": s["raw_text"][:200],
                    })
                    n += 1
                else:  # cell-scoped: WHITESPACE, INVALID DATA, other
                    chips = s["ranges"] or [""]
                    for chip in chips:
                        m = A1_CELL.match(chip.strip())
                        if m:
                            letter, srow = m.group(1), int(m.group(2))
                            name, _ = col_to_name(letter, header)
                            w.writerow({
                                "source": src, "set": table_set, "dataset": dataset,
                                "category": cat,
                                "scope": "cell", "cell_ref": chip,
                                "column_letter": letter, "column_name": name,
                                "sheet_row": srow, "bench_row": srow - 2,
                                "flagged_value": s["value"],
                                "suggested_value": suggested, "action": s["action"],
                                "raw_text": s["raw_text"][:200],
                            })
                        else:
                            w.writerow({
                                "source": src, "set": table_set, "dataset": dataset,
                                "category": cat,
                                "scope": "cell", "cell_ref": chip,
                                "column_letter": "", "column_name": "",
                                "sheet_row": "", "bench_row": "",
                                "flagged_value": s["value"],
                                "suggested_value": suggested, "action": s["action"],
                                "raw_text": s["raw_text"][:200],
                            })
                        n += 1

    print(f"wrote {OUT_CSV}")
    print(f"rows: {n}")


if __name__ == "__main__":
    main()
