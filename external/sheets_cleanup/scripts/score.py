"""Score Google Sheets 'Cleanup suggestions' against the merged_v2 cell benchmark.

Pipeline: parse each captured panel HTML -> interpret suggestions by category
scope (cell / column / row) -> map Sheets A1 refs to benchmark (source, row,
column-name) cells -> compare to gold.

Mapping notes:
  * On CSV import, Sheets row 1 is the header; the first data row is Sheets row 2.
    The benchmark uses 0-based data rows, so  bench_row = sheet_row - 2.
  * Column letter -> column name via the table's CSV header (from manifest.json).

Google's categories and their scope:
  WHITESPACE, INVALID DATA -> cell-level (chip like "J78")
  COLUMN FORMAT            -> column-level (chip like "C")        [display format]
  DUPLICATE ROWS           -> row-level (chips are row numbers)   [no merged_v2 gold]

merged_v2 contains only cell-level single-column error types, so the headline
metric is cell-level micro-F1 using Google's cell-scoped predictions.
"""

import argparse
import collections
import json
import os
import re

from harness import OUT
from parse_cleanup_html import _col_to_idx, parse_html

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH = os.path.join(REPO, "benchmark", "merged_v2", "benchmark_cell.json")
NEUTRAL = os.path.join(REPO, "benchmark", "merged_v2", "neutral_cells.jsonl")
MANIFEST = os.path.join(OUT, "manifest.json")
HTML_DIR = os.path.join(OUT, "html")

CELL_CATS = {"WHITESPACE", "INVALID DATA"}
COL_CATS = {"COLUMN FORMAT"}
ROW_CATS = {"DUPLICATE ROWS"}
A1_CELL = re.compile(r"^([A-Z]+)(\d+)$")


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_benchmark():
    cells = load_jsonl(BENCH)
    gold = {}
    for c in cells:
        key = (c["source"], int(c["row"]), c["column"])
        status = c.get("status") or c.get("label")
        gold[key] = {
            "status": status,
            "error_types": c.get("error_types"),
            "value": c.get("original_value"),
        }
    return gold


def header_maps(manifest):
    """source -> {col_letter: column_name} and set of column names."""
    maps = {}
    for m in manifest:
        header = m.get("header") or []
        letter2name = {}
        for i, name in enumerate(header):
            letter2name[_idx_to_letter(i + 1)] = name
        maps[m["source"]] = letter2name
    return maps


def _idx_to_letter(idx):
    s = ""
    while idx > 0:
        idx, r = divmod(idx - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def interpret(source, suggestions, letter2name):
    """Return dict with cell/col/row predictions for one table."""
    cell_preds = set()  # (source, bench_row, col_name)
    col_preds = set()  # (source, col_name)
    row_preds = set()  # (source, bench_row)
    detail = []
    for s in suggestions:
        cat = s["category"]
        for chip in s["ranges"]:
            chip = chip.strip()
            if cat in ROW_CATS and chip.isdigit():
                sheet_row = int(chip)
                row_preds.add((source, sheet_row - 2))
                detail.append((cat, "row", sheet_row - 2, None))
            elif cat in COL_CATS:
                col = chip.split(":")[0]
                name = letter2name.get(col)
                if name is not None:
                    col_preds.add((source, name))
                    detail.append((cat, "col", None, name))
            else:  # cell-scoped (WHITESPACE, INVALID DATA, default)
                m = A1_CELL.match(chip)
                if m:
                    col, sheet_row = m.group(1), int(m.group(2))
                    name = letter2name.get(col)
                    if name is not None:
                        cell_preds.add((source, sheet_row - 2, name))
                        detail.append((cat, "cell", sheet_row - 2, name))
    return cell_preds, col_preds, row_preds, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "score_report"))
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    letter_maps = header_maps(manifest)
    gold = load_benchmark()
    # neutral_cells.jsonl marks human-skipped/neutral cells. Schema is loose:
    # some rows use singular row/column, a few use plural lists, and many have an
    # empty column (whole-row skip). Collect a best-effort set of (source, row,
    # column) tuples for "known not-an-error" signal; ignore malformed entries.
    neutral = set()
    for n in load_jsonl(NEUTRAL):
        src = n.get("source")
        if not src:
            continue
        col = n.get("column", "")
        if "row" in n and isinstance(n["row"], int):
            neutral.add((src, n["row"], col))

    gold_error = {k for k, v in gold.items() if v["status"] == "error"}
    gold_ambig = {k for k, v in gold.items() if v["status"] == "ambiguous"}
    tables_with_gold = {k[0] for k in gold}

    # Aggregate predictions across all captured tables.
    all_cell_preds = set()
    all_col_preds = set()
    all_row_preds = set()
    cat_counter = collections.Counter()
    tables_scored = 0

    for fname in os.listdir(HTML_DIR):
        if not fname.endswith(".html"):
            continue
        source = fname[:-5]  # strip .html
        letter2name = letter_maps.get(source, {})
        parsed = parse_html(open(os.path.join(HTML_DIR, fname)).read())
        if not parsed["present"]:
            continue
        tables_scored += 1
        for s in parsed["suggestions"]:
            cat_counter[s["category"]] += 1
        cp, kp, rp, _ = interpret(source, parsed["suggestions"], letter2name)
        all_cell_preds |= cp
        all_col_preds |= kp
        all_row_preds |= rp

    # Cell-level scoring: restrict to tables we actually captured AND that have
    # gold, so recall is well-defined.
    captured_sources = {f[:-5] for f in os.listdir(HTML_DIR) if f.endswith(".html")}
    scope_gold_error = {k for k in gold_error if k[0] in captured_sources}

    tp = all_cell_preds & scope_gold_error
    tp_ambig = all_cell_preds & gold_ambig
    tp_neutral = all_cell_preds & neutral
    pred_on_gold = all_cell_preds & set(gold.keys())
    fp = all_cell_preds - set(gold.keys()) - neutral  # predicted, not in any labeled set
    fn = scope_gold_error - all_cell_preds

    precision_strict = len(tp) / len(all_cell_preds) if all_cell_preds else 0.0
    recall = len(tp) / len(scope_gold_error) if scope_gold_error else 0.0
    f1 = (2 * precision_strict * recall / (precision_strict + recall)
          if (precision_strict + recall) else 0.0)

    # Per-error-type recall: how many gold-error cells of each type did Google's
    # cell-scoped predictions catch?
    type_total = collections.Counter()
    type_hit = collections.Counter()
    for k in scope_gold_error:
        et = gold[k]["error_types"]
        et = et if isinstance(et, str) else "-".join(et or [])
        for tok in re.split(r"[-,;|]", et or ""):
            tok = tok.strip()
            if tok:
                type_total[tok] += 1
                if k in all_cell_preds:
                    type_hit[tok] += 1

    report = {
        "tables_captured": len(captured_sources),
        "tables_scored": tables_scored,
        "suggestion_categories": dict(cat_counter),
        "cell_predictions": len(all_cell_preds),
        "column_predictions": len(all_col_preds),
        "row_predictions": len(all_row_preds),
        "gold_error_cells_in_captured": len(scope_gold_error),
        "cell_level": {
            "tp": len(tp),
            "fp_unlabeled": len(fp),
            "fn": len(fn),
            "tp_on_ambiguous": len(tp_ambig),
            "tp_on_neutral_clean": len(tp_neutral),
            "predictions_landing_on_labeled_cells": len(pred_on_gold),
            "precision_strict": round(precision_strict, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "per_type_recall": {
            t: {"hit": type_hit[t], "total": type_total[t],
                "recall": round(type_hit[t] / type_total[t], 4) if type_total[t] else 0.0}
            for t in sorted(type_total)
        },
    }

    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)

    # Markdown summary
    lines = [
        "# Google Sheets Cleanup vs merged_v2 benchmark",
        "",
        f"- Tables captured: **{report['tables_captured']}** "
        f"(scored {report['tables_scored']})",
        f"- Suggestion categories seen: `{report['suggestion_categories']}`",
        f"- Cell-scoped predictions: **{report['cell_predictions']}** | "
        f"column: {report['column_predictions']} | row: {report['row_predictions']}",
        f"- Gold *error* cells in captured tables: "
        f"**{report['gold_error_cells_in_captured']}**",
        "",
        "## Cell-level match",
        f"- TP (hit a gold error cell): **{report['cell_level']['tp']}**",
        f"- TP on *ambiguous* cells: {report['cell_level']['tp_on_ambiguous']}",
        f"- Predictions on known-clean (neutral) cells: "
        f"{report['cell_level']['tp_on_neutral_clean']}",
        f"- FP (predicted, unlabeled): {report['cell_level']['fp_unlabeled']}",
        f"- FN (missed gold errors): {report['cell_level']['fn']}",
        f"- **Recall = {report['cell_level']['recall']}**, "
        f"strict precision = {report['cell_level']['precision_strict']}, "
        f"F1 = {report['cell_level']['f1']}",
        "",
        "## Per-error-type recall (Google cell predictions vs gold)",
        "| error_type | hit | total | recall |",
        "|---|---:|---:|---:|",
    ]
    for t, d in report["per_type_recall"].items():
        lines.append(f"| {t} | {d['hit']} | {d['total']} | {d['recall']} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nwrote", args.out + ".json", "and", args.out + ".md")


if __name__ == "__main__":
    main()
