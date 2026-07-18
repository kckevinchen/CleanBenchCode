"""Label Google Sheets 'Data cleanup' detections TP/FP against CleanBench,
mirroring the Excel labeled CSV.

Google's cleanup is apply-then-diff: scripts/diff_cleanup.py recovered every cell
Apply changed (outputs/applied_fixes.jsonl, before->after). Those changed cells ARE
its "detections". This joins each changed cell (restricted to the scored scope in
predictions_gsheets.jsonl.ran.json) to the current v8 gold with the exact score.py
contract (--ambiguous exclude: predicted-ambiguous cells neutral; GT = error cells),
labels TP/FP/ambiguous, and classifies each fix into a change_kind that stands in for
Google's (unlabeled) "detection type".

Outputs (under external/sheets_cleanup/outputs/):
  * gsheets_finding_labeled.csv      one row per scored detection cell + verdict
  * gsheets_perkind_performance.csv  per change_kind TP/FP/precision + recall share
"""

from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

REIN = Path(__file__).resolve().parents[3] / "external" / "rein"
sys.path.insert(0, str(REIN))
import common as C  # noqa: E402
import score  # noqa: E402

SHEETS_OUT = Path(__file__).resolve().parents[1] / "outputs"
FIXES = SHEETS_OUT / "applied_fixes.jsonl"
PRED = REIN / "results" / "predictions_gsheets.jsonl"


def col_letter_to_index(letter: str) -> int:
    n = 0
    for ch in letter.strip().upper():
        if "A" <= ch <= "Z":
            n = n * 26 + (ord(ch) - 64)
    return n - 1


def _num(x: str) -> bool:
    try:
        float(re.sub(r"[,$%\s]", "", x))
        return True
    except ValueError:
        return False


def change_kind(bv: str, av: str) -> str:
    ws = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    if ws(bv) == ws(av):
        return "whitespace"
    if "%" in av and "%" not in bv:
        return "percent_format"
    if _num(bv) and _num(av):
        return "numeric_format"
    if re.search(r"\d{4}-\d{2}-\d{2}", av) or re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", av):
        return "date_format"
    return "other"


def dirty_header(ds: str, cache: dict) -> list:
    if ds not in cache:
        p = C.DATASETS_DIR / ds / "dirty.csv"
        cache[ds] = next(csv.reader(open(p, encoding="utf-8")), []) if p.exists() else []
    return cache[ds]


def main() -> int:
    ran = json.loads((Path(str(PRED) + ".ran.json")).read_text())
    scope = list(ran["per_detector"]["gsheets"])
    scope_set = set(scope)

    pred_cells = set()
    for line in PRED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            pred_cells.add((r["dataset"], int(r["row"]), int(r["col"])))

    error_cells, ambiguous_cells, *_ = score.load_ground_truth(C.DATASETS_DIR, scope)
    total_gt_errors = len(error_cells)

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

    # join each changed cell -> its scored prediction cell (dedup: one row per cell)
    hcache: dict = {}
    cell_fix: dict[tuple[str, int, int], dict] = {}
    for line in FIXES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ds = C.slugify_dataset(r["source"])
        if ds not in scope_set:
            continue
        dr = int(r["data_row"])
        if dr < 0:
            continue
        j = col_letter_to_index(r["column"])
        cell = (ds, dr, j)
        if cell not in pred_cells or cell in cell_fix:
            continue
        hdr = dirty_header(ds, hcache)
        cell_fix[cell] = {
            "source": r["source"], "cell": r["cell"],
            "header": hdr[j] if j < len(hdr) else "",
            "before": r["before"], "after": r["after"],
            "change_kind": change_kind(r["before"], r["after"]),
        }

    labeled = []
    tp = fp = amb = 0
    for cell in sorted(pred_cells):
        ds, i, j = cell
        m = cell_fix.get(cell, {})
        if cell in ambiguous_cells and cell not in error_cells:
            verdict, amb = "ambiguous", amb + 1
        elif cell in error_cells:
            verdict, tp = "TP", tp + 1
        else:
            verdict, fp = "FP", fp + 1
        labeled.append({
            "dataset": ds, "source": m.get("source", ""), "cell": m.get("cell", ""),
            "data_row": i, "col": j, "header": m.get("header", ""),
            "before": m.get("before", ""), "after": m.get("after", ""),
            "change_kind": m.get("change_kind", "(unjoined)"), "verdict": verdict,
            "gold_error_types": "|".join(gold_types.get(cell, [])),
        })

    cols = ["dataset", "source", "cell", "data_row", "col", "header", "before",
            "after", "change_kind", "verdict", "gold_error_types"]
    lp = SHEETS_OUT / "gsheets_finding_labeled.csv"
    with open(lp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(labeled)

    by_kind = collections.defaultdict(lambda: {"TP": 0, "FP": 0, "ambiguous": 0})
    for rec in labeled:
        by_kind[rec["change_kind"]][rec["verdict"]] += 1
    rows_out = []
    for k in sorted(by_kind, key=lambda x: -(by_kind[x]["TP"] + by_kind[x]["FP"])):
        d = by_kind[k]
        scored = d["TP"] + d["FP"]
        rows_out.append({
            "change_kind": k, "flagged_scored": scored, "TP": d["TP"], "FP": d["FP"],
            "ambiguous_excluded": d["ambiguous"],
            "precision": round(d["TP"] / scored, 4) if scored else 0.0,
            "recall_share_of_v8": round(d["TP"] / total_gt_errors, 4) if total_gt_errors else 0.0,
        })
    pp = SHEETS_OUT / "gsheets_perkind_performance.csv"
    with open(pp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    scored = tp + fp
    prec = tp / scored if scored else 0.0
    rec = tp / total_gt_errors if total_gt_errors else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print("=== Google Sheets 'Data cleanup' labeled ===")
    print(f"v8 gold error cells (scope)  : {total_gt_errors}")
    print(f"detection cells (scored)     : {len(pred_cells)}")
    print(f"  ambiguous (neutral)        : {amb}")
    print(f"  scored (TP+FP)             : {scored}")
    print(f"  TP {tp}  FP {fp}  FN {total_gt_errors - tp}")
    print(f"  P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}\n")
    print(f"{'change_kind':16s}{'scored':>8}{'TP':>5}{'FP':>7}{'amb':>5}{'prec':>7}{'rec_share':>11}")
    for r in rows_out:
        print(f"{r['change_kind']:16s}{r['flagged_scored']:8d}{r['TP']:5d}{r['FP']:7d}"
              f"{r['ambiguous_excluded']:5d}{r['precision']:7.3f}{r['recall_share_of_v8']:11.4f}")
    print(f"\nwrote {lp}\nwrote {pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
