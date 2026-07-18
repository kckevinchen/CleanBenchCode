"""Score Google Sheets Cleanup detections against CleanBench (cell-level).

Uses manifest_full.json (headers for all 1002 captured tables) and parses every
captured panel HTML. Reports overall + per-error-type recall/precision, and how
Google's suggestion categories map onto the benchmark's error taxonomy.
"""

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from parse_cleanup_html import parse_html  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH = os.path.join(REPO, "benchmark", "CleanBench", "benchmark_cell.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
HTML_DIR = os.path.join(OUT, "html")
MANIFEST = os.path.join(OUT, "manifest_full.json")

A1_CELL = re.compile(r"^([A-Z]+)(\d+)$")
CELL_CATS = {"WHITESPACE", "INVALID DATA"}   # cell-scoped Google categories
COL_CATS = {"COLUMN FORMAT"}
ROW_CATS = {"DUPLICATE ROWS"}


def col_letter_to_idx(col):
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def main():
    # gold
    gold = {}
    for line in open(BENCH):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        gold[(c["source"], int(c["row"]), c["column"])] = {
            "status": c.get("status"),
            "types": c.get("error_types") or [],
        }
    gold_err = {k for k, v in gold.items() if v["status"] == "error"}
    gold_amb = {k for k, v in gold.items() if v["status"] == "ambiguous"}
    benchmark_tables = {k[0] for k in gold}

    manifest = {m["source"]: m for m in json.load(open(MANIFEST))}
    captured = {f[:-5] for f in os.listdir(HTML_DIR) if f.endswith(".html")}

    # Aggregate predictions only for benchmark tables, for well-defined recall.
    cell_preds = set()      # (source,row,col_name)
    col_preds = set()       # (source,col_name)
    row_preds = set()       # (source,row)
    cat_counts = collections.Counter()
    cat_tables = collections.Counter()

    for src in benchmark_tables & captured:
        header = manifest.get(src, {}).get("header", [])
        parsed = parse_html(open(os.path.join(HTML_DIR, src + ".html")).read())
        seen_cats = set()
        for s in parsed["suggestions"]:
            cat = s["category"]
            cat_counts[cat] += 1
            seen_cats.add(cat)
            for chip in s["ranges"]:
                chip = chip.strip()
                if cat in ROW_CATS and chip.isdigit():
                    row_preds.add((src, int(chip) - 2))
                elif cat in COL_CATS:
                    ci = col_letter_to_idx(chip.split(":")[0]) if chip.split(":")[0].isalpha() else -1
                    if 0 <= ci < len(header):
                        col_preds.add((src, header[ci]))
                else:
                    m = A1_CELL.match(chip)
                    if m:
                        ci = col_letter_to_idx(m.group(1))
                        if 0 <= ci < len(header):
                            cell_preds.add((src, int(m.group(2)) - 2, header[ci]))
        for c in seen_cats:
            cat_tables[c] += 1

    # Cell-level scoring against gold errors in captured benchmark tables.
    scope_err = {k for k in gold_err if k[0] in captured}
    scope_amb = {k for k in gold_amb if k[0] in captured}
    tp = cell_preds & scope_err
    tp_amb = cell_preds & scope_amb
    on_labeled = cell_preds & set(gold.keys())
    fp = cell_preds - set(gold.keys())
    recall = len(tp) / len(scope_err) if scope_err else 0.0
    prec = len(tp) / len(cell_preds) if cell_preds else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    # per-type recall (a gold cell counts as hit if ANY google cell-pred lands on it)
    type_total = collections.Counter()
    type_hit = collections.Counter()
    for k in scope_err:
        for t in gold[k]["types"]:
            type_total[t] += 1
            if k in cell_preds:
                type_hit[t] += 1

    report = {
        "benchmark": "CleanBench",
        "benchmark_tables": len(benchmark_tables),
        "captured_benchmark_tables": len(benchmark_tables & captured),
        "google_categories_on_cleanbench": dict(cat_counts),
        "google_category_tables": dict(cat_tables),
        "predictions": {"cell": len(cell_preds), "column": len(col_preds), "row": len(row_preds)},
        "gold_error_cells_scored": len(scope_err),
        "cell_level": {
            "tp": len(tp), "fp_unlabeled": len(fp), "fn": len(scope_err) - len(tp),
            "tp_on_ambiguous": len(tp_amb), "preds_on_labeled_cells": len(on_labeled),
            "recall": round(recall, 4), "precision_strict": round(prec, 4), "f1": round(f1, 4),
        },
        "per_type_recall": {
            t: {"hit": type_hit[t], "total": type_total[t],
                "recall": round(type_hit[t] / type_total[t], 4) if type_total[t] else 0.0}
            for t in sorted(type_total)
        },
    }
    with open(os.path.join(OUT, "score_cleanbench.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
