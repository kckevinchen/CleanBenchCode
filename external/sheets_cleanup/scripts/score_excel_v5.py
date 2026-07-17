"""Score Excel Copilot findings (external/excel_finding.csv) against merged_v5.

The findings CSV already provides `data_row` (0-based) and `column` name, so
mapping to the benchmark's (source,row,column) keys is direct. We report overall
cell-level recall/precision and per-error-type recall, plus how Excel's detection
`type` labels distribute over the gold error types they hit.
"""

import collections
import csv
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH = os.path.join(REPO, "benchmark", "merged_v5", "benchmark_cell.json")
FINDINGS = os.path.join(REPO, "external", "excel_finding.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_findings():
    rows = []
    with open(FINDINGS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    preds = set()
    by_key = {}
    for r in rows:
        if r.get("data_row") in (None, ""):
            continue
        # Key on the column NAME (`header`), not the letter (`column`); Excel's
        # `data_row` is already benchmark 0-based (excel_row - 2).
        key = (r["source"], int(r["data_row"]), r["header"])
        preds.add(key)
        by_key.setdefault(key, []).append(r.get("type", ""))
    return rows, preds, by_key


def main():
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
    v5_tables = {k[0] for k in gold}

    raw, preds_all, by_key = load_findings()
    pred_tables = {k[0] for k in preds_all}

    # Restrict scoring to tables that are BOTH in v5 and processed by Excel, so
    # gold exists for every prediction (fair precision) and recall is defined.
    covered = v5_tables & pred_tables
    preds = {k for k in preds_all if k[0] in covered}
    scope_err = {k for k in gold_err if k[0] in covered}
    scope_amb = {k for k in gold_amb if k[0] in covered}

    tp = preds & scope_err
    tp_amb = preds & scope_amb
    on_labeled = preds & set(gold.keys())
    fp = preds - set(gold.keys())
    recall = len(tp) / len(scope_err) if scope_err else 0.0
    prec = len(tp) / len(preds) if preds else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    # per gold-type recall (restricted to covered tables)
    type_total = collections.Counter()
    type_hit = collections.Counter()
    for k in scope_err:
        for t in gold[k]["types"]:
            type_total[t] += 1
            if k in preds:
                type_hit[t] += 1

    # what Excel type-labels land on true errors
    excel_type_on_tp = collections.Counter()
    for k in tp:
        for t in by_key.get(k, []):
            excel_type_on_tp[t] += 1

    report = {
        "benchmark": "merged_v5",
        "excel_detections": len(raw),
        "excel_pred_cells_all": len(preds_all),
        "excel_pred_cells_scored": len(preds),
        "excel_tables": len(pred_tables),
        "v5_tables": len(v5_tables),
        "v5_tables_excel_covered": len(covered),
        "gold_error_cells_in_covered_tables": len(scope_err),
        "cell_level": {
            "tp": len(tp),
            "fp_unlabeled": len(fp),
            "fn": len(scope_err) - len(tp),
            "tp_on_ambiguous": len(tp_amb),
            "preds_on_labeled_cells": len(on_labeled),
            "recall": round(recall, 4),
            "precision_strict": round(prec, 4),
            "f1": round(f1, 4),
        },
        "per_type_recall": {
            t: {"hit": type_hit[t], "total": type_total[t],
                "recall": round(type_hit[t] / type_total[t], 4) if type_total[t] else 0.0}
            for t in sorted(type_total)
        },
        "excel_type_labels_on_true_positives": dict(excel_type_on_tp.most_common()),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "score_excel_v5.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
