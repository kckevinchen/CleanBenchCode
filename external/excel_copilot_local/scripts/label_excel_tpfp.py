"""Label Excel Copilot predictions TP/FP against CleanBench and report
per-Excel-type performance.

Authoritative source of the scored cells is
``external/rein/results/predictions_excel.jsonl`` (the exact set the canonical
scorer ``external/rein/score.py`` grades -> P 0.692 / R 0.258 / F1 0.375,
TP 355 / FP 158). Every prediction cell is joined back to its Excel-declared
error ``type`` via ``excel_finding.csv`` using the column-NAME -> dirty.csv index
mapping (the same mapping used to build the predictions), then labeled against the
v8 gold cells with the *exact* score.py contract (``--ambiguous exclude``:
predicted-ambiguous cells are neutral; GT = error cells only).

Outputs (under external/excel_copilot_local/outputs/):
  * excel_finding_labeled.csv    one row per scored prediction cell + verdict
  * excel_pertype_performance.csv per Excel-type TP/FP/precision + recall share

Run:  external/rein/venv/bin/python \
        external/excel_copilot_local/scripts/label_excel_tpfp.py
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

# --- wire up the canonical scorer's helpers so labels match score.py exactly ---
REIN = Path(__file__).resolve().parents[3] / "external" / "rein"
import sys

sys.path.insert(0, str(REIN))
import common as C  # noqa: E402
import score  # noqa: E402

EXCEL_OUT = Path(__file__).resolve().parents[1] / "outputs"
PRED = REIN / "results" / "predictions_excel.jsonl"
FINDINGS = EXCEL_OUT / "excel_finding.csv"


def dirty_cols(ds: str, cache: dict) -> dict:
    if ds in cache:
        return cache[ds]
    p = C.DATASETS_DIR / ds / "dirty.csv"
    hdr = []
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            hdr = next(csv.reader(f), [])
    cache[ds] = {name: i for i, name in enumerate(hdr)}
    return cache[ds]


def main() -> int:
    # --- scored prediction cells + run scope (datasets excel actually ran on) ---
    ran = json.loads((Path(str(PRED) + ".ran.json")).read_text())
    scope = list(ran["per_detector"]["excel"])
    scope_set = set(scope)
    pred_cells: set[tuple[str, int, int]] = set()
    for line in PRED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            pred_cells.add((r["dataset"], int(r["row"]), int(r["col"])))

    # --- v8 ground truth (exactly as score.py loads it) ---
    error_cells, ambiguous_cells, _types, _gt_by_type, _gt_fine = score.load_ground_truth(
        C.DATASETS_DIR, scope)
    total_gt_errors = len(error_cells)

    # raw gold error types per cell (for TP diagnostics)
    gold_types: dict[tuple[str, int, int], list[str]] = {}
    for ds in scope_set:
        p = C.DATASETS_DIR / ds / "gt_cells.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") == C.STATUS_ERROR:
                gold_types[(ds, int(rec["i"]), int(rec["j"]))] = rec.get("error_types") or []

    # --- join each prediction cell -> Excel finding row (type/value/fix/text) ---
    colcache: dict = {}
    cell_meta: dict[tuple[str, int, int], dict] = {}
    for row in csv.DictReader(open(FINDINGS, encoding="utf-8-sig")):
        ds = C.slugify_dataset(row["source"])
        if ds not in scope_set:
            continue
        dr = row.get("data_row")
        if dr in (None, ""):
            continue
        dr = int(dr)
        if dr < 0:
            continue
        j = dirty_cols(ds, colcache).get(row["header"])
        if j is None:
            continue
        cell = (ds, dr, j)
        if cell not in pred_cells:
            continue
        # first finding row for the cell wins the display fields; collect all types
        m = cell_meta.setdefault(cell, {
            "source": row["source"], "cell": row["cell"], "header": row["header"],
            "value": row["value"], "suggested_fix": row["suggested_fix"],
            "text": row.get("text", ""), "types": [],
        })
        m["types"].append(row["type"])

    # --- verdict per cell, mirroring score.py (ambiguous exclude; no excel neutral) ---
    labeled = []
    tp = fp = amb = 0
    for cell in sorted(pred_cells):
        ds, i, j = cell
        meta = cell_meta.get(cell, {})
        etypes = meta.get("types", [])
        excel_type = etypes[0] if etypes else "(unjoined)"
        if cell in ambiguous_cells and cell not in error_cells:
            verdict, amb = "ambiguous", amb + 1
        elif cell in error_cells:
            verdict, tp = "TP", tp + 1
        else:
            verdict, fp = "FP", fp + 1
        labeled.append({
            "dataset": ds, "source": meta.get("source", ""), "cell": meta.get("cell", ""),
            "data_row": i, "col": j, "header": meta.get("header", ""),
            "value": meta.get("value", ""), "excel_type": excel_type,
            "excel_types_all": "|".join(dict.fromkeys(etypes)),
            "suggested_fix": meta.get("suggested_fix", ""),
            "verdict": verdict,
            "gold_error_types": "|".join(gold_types.get(cell, [])),
        })

    out_cols = ["dataset", "source", "cell", "data_row", "col", "header", "value",
                "excel_type", "excel_types_all", "suggested_fix", "verdict",
                "gold_error_types"]
    lp = EXCEL_OUT / "excel_finding_labeled.csv"
    with open(lp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols)
        w.writeheader()
        w.writerows(labeled)

    # --- per-Excel-type performance (precision well-defined; recall share too) ---
    by_type = collections.defaultdict(lambda: {"TP": 0, "FP": 0, "ambiguous": 0})
    for rec in labeled:
        by_type[rec["excel_type"]][rec["verdict"]] += 1

    rows_out = []
    for t in sorted(by_type, key=lambda x: -(by_type[x]["TP"] + by_type[x]["FP"])):
        d = by_type[t]
        scored = d["TP"] + d["FP"]
        prec = d["TP"] / scored if scored else 0.0
        rec_share = d["TP"] / total_gt_errors if total_gt_errors else 0.0
        rows_out.append({
            "excel_type": t, "flagged_scored": scored, "TP": d["TP"], "FP": d["FP"],
            "ambiguous_excluded": d["ambiguous"], "precision": round(prec, 4),
            "recall_share_of_v8": round(rec_share, 4),
        })
    pt_cols = ["excel_type", "flagged_scored", "TP", "FP", "ambiguous_excluded",
               "precision", "recall_share_of_v8"]
    pp = EXCEL_OUT / "excel_pertype_performance.csv"
    with open(pp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=pt_cols)
        w.writeheader()
        w.writerows(rows_out)

    # --- console summary ---
    scored = tp + fp
    prec = tp / scored if scored else 0.0
    rec = tp / total_gt_errors if total_gt_errors else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"v8 gold error cells (excel scope) : {total_gt_errors}")
    print(f"excel prediction cells (raw)      : {len(pred_cells)}")
    print(f"  ambiguous (excluded, neutral)   : {amb}")
    print(f"  scored (TP+FP)                  : {scored}")
    print(f"  TP {tp}  FP {fp}  FN {total_gt_errors - tp}")
    print(f"  precision {prec:.3f}  recall {rec:.3f}  f1 {f1:.3f}")
    print(f"\nwrote {lp}")
    print(f"wrote {pp}\n")
    print(f"{'excel_type':26s}{'scored':>8}{'TP':>6}{'FP':>6}{'amb':>5}"
          f"{'prec':>8}{'rec_share':>11}")
    for r in rows_out:
        print(f"{r['excel_type']:26s}{r['flagged_scored']:8d}{r['TP']:6d}{r['FP']:6d}"
              f"{r['ambiguous_excluded']:5d}{r['precision']:8.3f}{r['recall_share_of_v8']:11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
