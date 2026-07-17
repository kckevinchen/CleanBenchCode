"""Analyze false-positive / trigger rate of Google Sheets cleanup on the 785
"clean" (non-benchmark) tables.

Since these tables have no gold labels, any suggestion is treated as a
false-positive *trigger*. We report: how many tables triggered, by category;
total suggestions; cell/column/row counts; and the cell-level FP rate
(cell-scoped suggestions per total cells scanned).

For comparison we also recompute the same trigger stats on the 217 benchmark
tables, so FP behavior on clean vs labeled data sits side by side.
"""

import collections
import json
import os
import re

from harness import OUT
from parse_cleanup_html import expand_ranges, parse_html

MANIFEST_CLEAN = os.path.join(OUT, "manifest_clean.json")
MANIFEST_BENCH = os.path.join(OUT, "manifest.json")
HTML_DIR = os.path.join(OUT, "html")
A1_CELL = re.compile(r"^([A-Z]+)(\d+)$")

CELL_CATS = {"WHITESPACE", "INVALID DATA"}
COL_CATS = {"COLUMN FORMAT"}
ROW_CATS = {"DUPLICATE ROWS"}


def analyze(manifest_path, label):
    manifest = json.load(open(manifest_path))
    total_tables = len(manifest)
    total_cells = sum(m["n_rows"] * m["n_cols"] for m in manifest)
    total_rows = sum(m["n_rows"] for m in manifest)

    tables_triggered = 0
    cat_tables = collections.Counter()
    cat_suggestions = collections.Counter()
    n_cell = n_col = n_row = 0
    missing = 0

    for m in manifest:
        src = m["source"]
        path = os.path.join(HTML_DIR, src + ".html")
        if not os.path.exists(path):
            missing += 1
            continue
        parsed = parse_html(open(path).read())
        if not parsed["suggestions"]:
            continue
        tables_triggered += 1
        cats_here = set()
        for s in parsed["suggestions"]:
            cat = s["category"]
            cat_suggestions[cat] += 1
            cats_here.add(cat)
            for chip in s["ranges"]:
                chip = chip.strip()
                if cat in ROW_CATS and chip.isdigit():
                    n_row += 1
                elif cat in COL_CATS:
                    n_col += 1
                elif A1_CELL.match(chip):
                    n_cell += 1
        for c in cats_here:
            cat_tables[c] += 1

    return {
        "label": label,
        "total_tables": total_tables,
        "total_rows": total_rows,
        "total_cells": total_cells,
        "missing_html": missing,
        "tables_triggered": tables_triggered,
        "tables_triggered_pct": round(100 * tables_triggered / total_tables, 1),
        "by_category_tables": dict(cat_tables),
        "by_category_suggestions": dict(cat_suggestions),
        "cell_scoped_suggestions": n_cell,
        "column_scoped_suggestions": n_col,
        "row_scoped_suggestions": n_row,
        "cell_fp_rate_per_1k_cells": round(1000 * n_cell / total_cells, 3) if total_cells else 0,
    }


def fmt(rep):
    L = [
        f"### {rep['label']}",
        f"- Tables: **{rep['total_tables']}** "
        f"({rep['total_rows']:,} rows, {rep['total_cells']:,} cells)",
        f"- Tables with >=1 suggestion: **{rep['tables_triggered']}** "
        f"({rep['tables_triggered_pct']}%)",
        f"- Suggestions by category (tables): `{rep['by_category_tables']}`",
        f"- Suggestions by category (count): `{rep['by_category_suggestions']}`",
        f"- Scope: cell={rep['cell_scoped_suggestions']} "
        f"col={rep['column_scoped_suggestions']} row={rep['row_scoped_suggestions']}",
        f"- **Cell-level FP rate: {rep['cell_fp_rate_per_1k_cells']} per 1,000 cells**",
    ]
    if rep["missing_html"]:
        L.append(f"- (missing HTML: {rep['missing_html']})")
    return "\n".join(L)


def main():
    clean = analyze(MANIFEST_CLEAN, "Clean set (785 non-benchmark tables)")
    bench = analyze(MANIFEST_BENCH, "Benchmark set (217 labeled tables)")

    out = {"clean": clean, "benchmark": bench}
    with open(os.path.join(OUT, "fp_report.json"), "w") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Google Sheets Cleanup — False-Positive / Trigger Analysis",
        "",
        "On the **clean set** these tables carry no gold errors, so every "
        "suggestion is a false-positive *trigger*.",
        "",
        fmt(clean),
        "",
        fmt(bench),
        "",
        "## Interpretation",
        "- `DUPLICATE ROWS` and `COLUMN FORMAT` dominate; both are "
        "formatting/structure heuristics, not content-error detection.",
        "- Cell-scoped content suggestions (`WHITESPACE`, `INVALID DATA`) are "
        "rare even on clean data — consistent with the benchmark recall of 0 on "
        "typo/spelling/inconsistency errors.",
    ]
    report = "\n".join(md) + "\n"
    with open(os.path.join(OUT, "fp_report.md"), "w") as f:
        f.write(report)
    print(report)
    print("wrote outputs/fp_report.json and outputs/fp_report.md")


if __name__ == "__main__":
    main()
