"""Per-error-type precision/recall breakdown across all baselines.

Reuses ``score.py``'s validated internals (ground-truth loader + neutral-set
logic) so numbers match the leaderboard. For every predictions file in the
results dir it reports:

* overall precision / recall / F1, and
* per-type RECALL (well-defined for every detector: hits_T / gt_T).

For detectors whose predictions are TYPED (a companion ``issues_<tag>.jsonl``
with an ``error_type`` per cell -- currently only the GPT-5.5 detector), it also
reports per-type PRECISION (type-aware: a cell predicted as T is correct iff the
labeled error at that cell has type T).

    python external/rein/analyze_by_type.py
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import score as S  # noqa: E402

TYPES = ["typo", "inconsistency", "semantic_incompatible", "numeric_outlier"]


def _load_preds(path: Path):
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["dataset"], int(r["row"]), int(r["col"])))
    return out


def _load_typed(path: Path):
    """(ds,row,col) -> coarse predicted type, from an issues_*.jsonl file."""
    typed = {}
    if not path.exists():
        return typed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        typed[(r["dataset"], int(r["row"]), int(r["col"]))] = S.coarse_type(r.get("error_type", ""))
    return typed


def _primary_type_sets(ds_root: Path, scope):
    """Partition labeled ERROR cells into ONE primary coarse type each.

    Uses the first ``error_types`` entry (coarse) so the per-type attributed
    precision columns sum exactly to the method's overall precision.
    """
    out = defaultdict(set)
    for ds in scope:
        p = ds_root / ds / "gt_cells.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != C.STATUS_ERROR:
                continue
            ets = rec.get("error_types") or []
            prim = S.coarse_type(ets[0]) if ets else "unspecified"
            out[prim].add((ds, int(rec["i"]), int(rec["j"])))
    return out


def analyze():
    ds_root = C.DATASETS_DIR
    manifest = json.loads((ds_root / "_manifest.json").read_text())["datasets"]
    all_ds = sorted(manifest)
    source2ds = {e["source"]: ds for ds, e in manifest.items()}
    neutral = S.load_neutral(C.DEFAULT_BENCHMARK_DIR / "neutral_cells.jsonl", ds_root, source2ds)

    rows = []
    for pf in sorted(glob.glob(str(C.RESULTS_DIR / "predictions_*.jsonl"))):
        pf = Path(pf)
        tag = pf.stem.replace("predictions_", "")
        ranp = Path(str(pf) + ".ran.json")
        scope = all_ds
        if ranp.exists():
            ran = json.loads(ranp.read_text())
            if isinstance(ran, dict):
                scope = ran.get("per_detector", {}).get(tag) or ran.get("all_loaded") or all_ds
            elif isinstance(ran, list) and ran:
                scope = ran

        err, amb, _t, gt_by_type = S.load_ground_truth(ds_root, scope)
        # Primary-type partition of error cells (each error -> exactly one type)
        # so attributed per-type precision sums to overall precision.
        primary = _primary_type_sets(ds_root, scope)
        preds = _load_preds(pf)

        # Held-out: for few-shot detectors (raha) exclude labeled-training rows
        # from BOTH predictions and ground truth (don't credit leaked error cells).
        labeled_path = Path(str(pf) + ".labeled_rows.json")
        if labeled_path.exists():
            raw = json.loads(labeled_path.read_text())
            labeled = {ds: set(int(r) for r in rs) for ds, rs in raw.items()}

            def ho(c):
                return c[1] in labeled.get(c[0], ())

            preds = {p for p in preds if not ho(p)}
            err = {c for c in err if not ho(c)}
            amb = {c for c in amb if not ho(c)}
            gt_by_type = {t: {c for c in cs if not ho(c)} for t, cs in gt_by_type.items()}
            primary = {t: {c for c in cs if not ho(c)} for t, cs in primary.items()}

        preds = {p for p in preds if p not in amb}
        preds = {p for p in preds if (p in err) or not S._is_neutral(p[0], p[1], p[2], neutral)}

        tp = len(preds & err)
        P = tp / len(preds) if preds else 0.0
        R = tp / len(err) if err else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0

        per_type = {}
        for t in TYPES:
            g = gt_by_type.get(t, set())
            hit = len(preds & g)
            gp = primary.get(t, set())
            tp_t = len(preds & gp)  # correct type-T hits (primary)
            per_type[t] = {
                "gt": len(g),
                "rec": (hit / len(g) if g else 0.0),
                # attributed precision: share of ALL flags that are correct type-T
                "prec_attr": (tp_t / len(preds) if preds else 0.0),
                "tp": tp_t,
            }

        typed = _load_typed(C.RESULTS_DIR / f"issues_{tag}.jsonl")
        if typed:
            typed = {k: v for k, v in typed.items() if k in preds}
            for t in TYPES:
                pred_t = {k for k, v in typed.items() if v == t}
                tp_t = len(pred_t & gt_by_type.get(t, set()))
                per_type[t]["prec_typed"] = (tp_t / len(pred_t) if pred_t else 0.0)
                per_type[t]["pred"] = len(pred_t)

        rows.append({"tag": tag, "cov": len(scope), "P": P, "R": R, "F1": F1,
                     "n_err": len(err), "n_pred": len(preds),
                     "per_type": per_type, "typed": bool(typed)})

    rows.sort(key=lambda r: -r["P"])

    # ---- overall ----
    print("=" * 82)
    print("OVERALL (cell-level micro, ambiguous+skipped+excluded neutralized) -- sorted by PRECISION")
    print("=" * 82)
    print(f"{'method':<26}{'cov':>5}{'predN':>8}{'P':>8}{'R':>8}{'F1':>8}{'errGT':>7}")
    for r in rows:
        print(f"{r['tag']:<26}{r['cov']:>5}{r['n_pred']:>8}{r['P']:>8.3f}{r['R']:>8.3f}"
              f"{r['F1']:>8.3f}{r['n_err']:>7}")

    # ---- per-type PRECISION (share of all flags; do NOT sum) ----
    print("\n" + "=" * 82)
    print("PER-TYPE PRECISION  (correct type-T hits / ALL flags; read per-cell, do NOT sum)")
    print("=" * 82)
    print(f"{'method':<26}" + "".join(f"{t[:9]:>12}" for t in TYPES))
    for r in rows:
        cells = "".join(f"{r['per_type'][t]['prec_attr']:>12.3f}" for t in TYPES)
        print(f"{r['tag']:<26}{cells}")

    # ---- per-type RECALL ----
    print("\n" + "=" * 82)
    print("PER-TYPE RECALL  (hits_T / gt_T)")
    print("=" * 82)
    print(f"{'method':<26}" + "".join(f"{t[:9]:>12}" for t in TYPES))
    for r in rows:
        cells = "".join(f"{r['per_type'][t]['rec']:>12.3f}" for t in TYPES)
        print(f"{r['tag']:<26}{cells}")
    print(f"{'(gt count, full scope)':<26}" + "".join(
        f"{rows[0]['per_type'][t]['gt']:>12}" for t in TYPES))

    # ---- typed detectors: self-labeled per-type P/R/F1 ----
    for r in rows:
        if not r["typed"]:
            continue
        print("\n" + "=" * 82)
        print(f"TYPE-AWARE P/R/F1  ({r['tag']})  -- precision uses the model's OWN type label")
        print("=" * 82)
        print(f"{'type':<24}{'predN':>8}{'gt':>7}{'P':>8}{'R':>8}{'F1':>8}")
        for t in TYPES:
            pt = r["per_type"][t]
            p, rec = pt.get("prec_typed", 0.0), pt["rec"]
            f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
            print(f"{t:<24}{pt.get('pred', 0):>8}{pt['gt']:>7}{p:>8.3f}{rec:>8.3f}{f1:>8.3f}")


if __name__ == "__main__":
    analyze()
