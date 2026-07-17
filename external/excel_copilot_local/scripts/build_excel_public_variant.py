"""Build a PRESENTATION ("public product") variant of the Excel baseline that
drops two internal-only detection types:

    Invalid email addresses   (highlight-only, not in the shipped product)
    Inconsistent order        (internal-only)

A flagged cell is kept iff it has at least one qualifying finding row whose type
is NOT excluded (so a cell flagged as both "Invalid email addresses" and a kept
type such as "Inconsistent pattern" survives, grouped under the kept type).

Everything else mirrors the canonical rebuild + labeling: map Excel findings to
current v8 ``dirty.csv`` cells by column NAME, keep cells with a non-blank applied
fix (the email special-case is now moot since email is excluded), score with the
exact ``score.py`` contract (``--ambiguous exclude``).

Writes:
  external/rein/results/predictions_excel_public.jsonl (+ .ran.json)  detector=excel_public
  external/excel_copilot_local/outputs/excel_finding_labeled_public.csv
  external/excel_copilot_local/outputs/excel_pertype_performance_public.csv
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

REIN = Path(__file__).resolve().parents[3] / "external" / "rein"
sys.path.insert(0, str(REIN))
import common as C  # noqa: E402
import score  # noqa: E402

EXCLUDE_TYPES = {"Invalid email addresses", "Inconsistent order"}
DETECTOR = "excel_public"

EXCEL_OUT = Path(__file__).resolve().parents[1] / "outputs"
FINDINGS = EXCEL_OUT / "excel_finding.csv"
PRED = REIN / "results" / f"predictions_{DETECTOR}.jsonl"


def qualifies(row: dict) -> bool:
    """A finding row contributes a prediction iff it has a real applied fix and
    its Excel type is not internal-only. (Email had no fix and is excluded, so the
    old email special-case is unnecessary here.)"""
    return bool(row["suggested_fix"].strip()) and row["type"] not in EXCLUDE_TYPES


def dirty_header(ds: str, cache: dict):
    if ds in cache:
        return cache[ds]
    p = C.DATASETS_DIR / ds / "dirty.csv"
    info = None
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            head = next(reader, [])
            ndata = sum(1 for _ in reader)
        info = ({name: i for i, name in enumerate(head)}, ndata)
    cache[ds] = info
    return info


def main() -> int:
    manifest = json.loads((C.DATASETS_DIR / "_manifest.json").read_text())
    current_ds = set(manifest["datasets"])
    source2ds = {e["source"]: ds for ds, e in manifest["datasets"].items()}

    prev_ran = json.loads((REIN / "results" / "predictions_excel.jsonl.ran.json").read_text())
    prev_scope = set(prev_ran.get("per_detector", {}).get("excel", []))
    scope = sorted(prev_scope & current_ds)
    scope_set = set(scope)

    cache: dict = {}
    cell_types: dict[tuple[str, int, int], list[str]] = collections.defaultdict(list)
    cell_meta: dict[tuple[str, int, int], dict] = {}
    for row in csv.DictReader(open(FINDINGS, encoding="utf-8-sig")):
        ds = C.slugify_dataset(row["source"])
        if ds not in scope_set:
            continue
        dr = row.get("data_row")
        if dr in (None, "") or int(dr) < 0:
            continue
        dr = int(dr)
        if not qualifies(row):
            continue
        info = dirty_header(ds, cache)
        if info is None:
            continue
        hdr, ndata = info
        j = hdr.get(row["header"])
        if j is None or dr >= ndata:
            continue
        cell = (ds, dr, j)
        cell_types[cell].append(row["type"])
        cell_meta.setdefault(cell, {
            "source": row["source"], "cell": row["cell"], "header": row["header"],
            "value": row["value"], "suggested_fix": row["suggested_fix"],
        })

    preds = set(cell_types)

    with open(PRED, "w", encoding="utf-8") as f:
        for (ds, i, j) in sorted(preds):
            f.write(json.dumps({"dataset": ds, "detector": DETECTOR, "row": i, "col": j}) + "\n")
    with open(str(PRED) + ".ran.json", "w", encoding="utf-8") as f:
        json.dump({"per_detector": {DETECTOR: scope}}, f)

    # ground truth + verdicts (exact score.py contract)
    error_cells, ambiguous_cells, *_ = score.load_ground_truth(C.DATASETS_DIR, scope)
    total_gt = len(error_cells)
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

    labeled = []
    tp = fp = amb = 0
    for cell in sorted(preds):
        ds, i, j = cell
        meta = cell_meta[cell]
        etypes = [t for t in cell_types[cell]]  # already excludes internal types
        excel_type = etypes[0] if etypes else "(unknown)"
        if cell in ambiguous_cells and cell not in error_cells:
            verdict, amb = "ambiguous", amb + 1
        elif cell in error_cells:
            verdict, tp = "TP", tp + 1
        else:
            verdict, fp = "FP", fp + 1
        labeled.append({
            "dataset": ds, "source": meta["source"], "cell": meta["cell"],
            "data_row": i, "col": j, "header": meta["header"], "value": meta["value"],
            "excel_type": excel_type, "excel_types_all": "|".join(dict.fromkeys(etypes)),
            "suggested_fix": meta["suggested_fix"], "verdict": verdict,
            "gold_error_types": "|".join(gold_types.get(cell, [])),
        })

    lp = EXCEL_OUT / "excel_finding_labeled_public.csv"
    cols = ["dataset", "source", "cell", "data_row", "col", "header", "value",
            "excel_type", "excel_types_all", "suggested_fix", "verdict", "gold_error_types"]
    with open(lp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(labeled)

    by_type = collections.defaultdict(lambda: {"TP": 0, "FP": 0, "ambiguous": 0})
    for rec in labeled:
        by_type[rec["excel_type"]][rec["verdict"]] += 1
    rows_out = []
    for t in sorted(by_type, key=lambda x: -(by_type[x]["TP"] + by_type[x]["FP"])):
        d = by_type[t]
        scored = d["TP"] + d["FP"]
        rows_out.append({
            "excel_type": t, "flagged_scored": scored, "TP": d["TP"], "FP": d["FP"],
            "ambiguous_excluded": d["ambiguous"],
            "precision": round(d["TP"] / scored, 4) if scored else 0.0,
            "recall_share_of_v8": round(d["TP"] / total_gt, 4) if total_gt else 0.0,
        })
    pp = EXCEL_OUT / "excel_pertype_performance_public.csv"
    with open(pp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    scored = tp + fp
    prec = tp / scored if scored else 0.0
    rec = tp / total_gt if total_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"EXCLUDED types           : {sorted(EXCLUDE_TYPES)}")
    print(f"v8 gold error cells      : {total_gt}")
    print(f"prediction cells (raw)   : {len(preds)}")
    print(f"  ambiguous (neutral)    : {amb}")
    print(f"  scored (TP+FP)         : {scored}")
    print(f"  TP {tp}  FP {fp}  FN {total_gt - tp}")
    print(f"  P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}\n")
    print(f"{'excel_type':22s}{'scored':>8}{'TP':>6}{'FP':>6}{'amb':>5}{'prec':>8}{'rec_share':>11}")
    for r in rows_out:
        print(f"{r['excel_type']:22s}{r['flagged_scored']:8d}{r['TP']:6d}{r['FP']:6d}"
              f"{r['ambiguous_excluded']:5d}{r['precision']:8.3f}{r['recall_share_of_v8']:11.4f}")
    print(f"\nwrote {PRED}\nwrote {lp}\nwrote {pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
