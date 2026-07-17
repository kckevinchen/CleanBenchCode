"""Scorer: cell-level micro precision/recall/F1 for REIN detector predictions.

Grades the JSONL predictions from ``run_rein.py`` against the benchmark labels
materialized per dataset (``datasets/<ds>/gt_cells.jsonl``). Metric mirrors REIN's
internal ``__evaluate`` and RFT.md (cell-level micro-F1).

Ambiguous handling (``--ambiguous``):
* ``exclude`` (default) -- predicted ambiguous cells are neutral (neither TP nor
  FP); ground truth = ``error`` cells only.
* ``fp``                -- predicted ambiguous cells count as false positives.
* ``error``             -- ambiguous cells are added to ground truth (lenient).

Scope: only the datasets the detector actually ran on (read from the
``<predictions>.ran.json`` sidecar, else all manifest datasets).

Example:
    python3 external/rein/score.py --predictions external/rein/results/predictions_raha.jsonl
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def coarse_type(t: str) -> str:
    """Collapse compound labels (``typo-spelling`` -> ``typo``)."""
    return t.split("-", 1)[0]


def _ds_columns(ds_root: Path, ds: str, cache: dict[str, dict[str, int]]) -> dict[str, int]:
    """Column-name -> integer index for a dataset's dirty.csv (cached)."""
    if ds in cache:
        return cache[ds]
    cols: list[str] = []
    p = ds_root / ds / "dirty.csv"
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            cols = next(csv.reader(f), [])
    cache[ds] = {name: i for i, name in enumerate(cols)}
    return cache[ds]


def load_neutral(path: Path, ds_root: Path, source2ds: dict[str, str]):
    """Parse neutral_cells.jsonl into (cells, files, rows, cols, meta).

    Records ``skip`` / ``exclude_csv`` / ``exclude_rows`` / ``exclude_cols`` are
    mapped from (source, column-name) into the (dataset, row, col-index) space.
    A ``skip`` with an empty column is a *row-level* finding (row-level error
    types, not part of the cell benchmark) and is ignored for cell scoring.
    """
    cells: set[tuple[str, int, int]] = set()
    files: set[str] = set()
    rows: dict[str, set[int]] = defaultdict(set)
    cols: dict[str, set[int]] = defaultdict(set)
    colcache: dict[str, dict[str, int]] = {}
    meta = {"unmatched": 0, "rowlevel_skip": 0}
    if not path or not path.exists():
        return cells, files, rows, cols, meta
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ds = source2ds.get(r.get("source"))
        if ds is None:  # source not among materialized datasets
            continue
        kind = r.get("kind")
        if kind == "skip":
            col = r.get("column")
            if not col:  # row-level skip: irrelevant to cell scoring
                meta["rowlevel_skip"] += 1
                continue
            j = _ds_columns(ds_root, ds, colcache).get(col)
            if j is None or r.get("row") is None:
                meta["unmatched"] += 1
                continue
            cells.add((ds, int(r["row"]), j))
        elif kind == "exclude_csv":
            files.add(ds)
        elif kind == "exclude_rows":
            for rr in r.get("rows", []):
                rows[ds].add(int(rr))
        elif kind == "exclude_cols":
            idx = _ds_columns(ds_root, ds, colcache)
            for name in r.get("columns", []):
                j = idx.get(name)
                if j is None:
                    meta["unmatched"] += 1
                    continue
                cols[ds].add(j)
    return cells, files, rows, cols, meta


def _is_neutral(ds: str, i: int, j: int, neutral) -> bool:
    cells, files, rows, cols = neutral[0], neutral[1], neutral[2], neutral[3]
    return (ds in files or i in rows.get(ds, ()) or j in cols.get(ds, ())
            or (ds, i, j) in cells)


def load_ground_truth(ds_root: Path, scope: list[str]):
    """Return (error_cells, ambiguous_cells, types_by_cell, gt_by_type, gt_by_type_fine)."""
    error_cells: set[tuple[str, int, int]] = set()
    ambiguous_cells: set[tuple[str, int, int]] = set()
    types_by_cell: dict[tuple[str, int, int], set[str]] = {}
    gt_by_type: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    gt_by_type_fine: dict[str, set[tuple[str, int, int]]] = defaultdict(set)

    for ds in scope:
        p = ds_root / ds / "gt_cells.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (ds, int(rec["i"]), int(rec["j"]))
            if rec.get("status") == C.STATUS_ERROR:
                error_cells.add(key)
                raw = rec.get("error_types") or []
                types = {coarse_type(t) for t in raw} or {"unspecified"}
                types_by_cell[key] = types
                for t in types:
                    gt_by_type[t].add(key)
                for t in (set(raw) or {"unspecified"}):
                    gt_by_type_fine[t].add(key)
            else:
                ambiguous_cells.add(key)
    return error_cells, ambiguous_cells, types_by_cell, gt_by_type, gt_by_type_fine


def load_seen_error_signatures(ds_root: Path, scope: list[str],
                               labeled_rows: dict[str, set[int]]):
    """For each dataset, the set of ``(col, dirty_value)`` signatures that were
    shown as **errors in a labeled (training) row**. A held-out error cell whose
    own ``(col, dirty_value)`` matches one of these was effectively memorizable:
    the few-shot learner saw that exact erroneous value flagged during training,
    so re-flagging it elsewhere is re-application, not novel detection."""
    sigs: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for ds in scope:
        rows = labeled_rows.get(ds)
        if not rows:
            continue
        p = ds_root / ds / "gt_cells.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") == C.STATUS_ERROR and int(rec["i"]) in rows:
                sigs[ds].add((int(rec["j"]), str(rec.get("original_value"))))
    return sigs


def memorized_error_cells(ds_root: Path, scope: list[str],
                          labeled_rows: dict[str, set[int]]):
    """Held-out error cells whose ``(col, dirty_value)`` was seen as an error in a
    training row (= memorizable). Returns a set of ``(ds, i, j)`` keys."""
    sigs = load_seen_error_signatures(ds_root, scope, labeled_rows)
    out: set[tuple[str, int, int]] = set()
    for ds in scope:
        ds_sigs = sigs.get(ds)
        rows = labeled_rows.get(ds, set())
        if not ds_sigs:
            continue
        p = ds_root / ds / "gt_cells.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            i, j = int(rec["i"]), int(rec["j"])
            if rec.get("status") == C.STATUS_ERROR and i not in rows:
                if (j, str(rec.get("original_value"))) in ds_sigs:
                    out.add((ds, i, j))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, help="predictions JSONL from run_rein.py")
    ap.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    ap.add_argument("--ambiguous", default="exclude", choices=["exclude", "fp", "error"])
    ap.add_argument("--neutral-file",
                    default=str(C.DEFAULT_BENCHMARK_DIR / "neutral_cells.jsonl"),
                    help="skipped/excluded cells to neutralize (from export_neutral_cells.py)")
    ap.add_argument("--no-neutral", action="store_true",
                    help="do not neutralize skipped/excluded cells")
    ap.add_argument("--holdout-labeled-rows", dest="holdout", action="store_true", default=True,
                    help="for few-shot raha: exclude its labeled rows from BOTH predictions and "
                         "ground truth (held-out eval; don't credit leaked error cells). On by "
                         "default when a <predictions>.labeled_rows.json sidecar exists.")
    ap.add_argument("--no-holdout-labeled-rows", dest="holdout", action="store_false",
                    help="score raw (count raha's leaked labeled error cells as TPs).")
    ap.add_argument("--no-synthetic-neutralize", action="store_true",
                    help="do NOT neutralize synthetic-injected cells (default: neutralize "
                         "them when a <predictions>.synthetic_cells.json sidecar exists).")
    ap.add_argument("--out", default="", help="write scores JSON here (default: results/scores_<tag>.json)")
    ap.add_argument("--restrict-to-manifest", dest="restrict", action="store_true", default=True,
                    help="drop predictions (and scope entries) for datasets absent from the current "
                         "_manifest.json. The manifest defines the benchmark universe, so flags on "
                         "datasets removed/replaced by a benchmark update are out of scope, not FPs. "
                         "On by default.")
    ap.add_argument("--no-restrict-to-manifest", dest="restrict", action="store_false",
                    help="score every predicted cell, including datasets absent from the manifest "
                         "(legacy behavior; such flags count as FPs).")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"Predictions not found: {pred_path}")
    ds_root = Path(args.datasets_dir)
    manifest = json.loads((ds_root / "_manifest.json").read_text())
    all_datasets = sorted(manifest["datasets"])
    source2ds = {entry["source"]: ds for ds, entry in manifest["datasets"].items()}

    # Skipped/excluded cells -> neutral (never scored as clean/FP); error wins.
    neutral = (set(), set(), {}, {}, {})
    if not args.no_neutral:
        neutral = load_neutral(Path(args.neutral_file), ds_root, source2ds)
        nc, nf, nr, ncol, meta = neutral
        extra = []
        if meta.get("rowlevel_skip"):
            extra.append(f"{meta['rowlevel_skip']} row-level skips ignored")
        if meta.get("unmatched"):
            extra.append(f"{meta['unmatched']} unmatched col-names")
        print(f"neutral set: {len(nc)} skip-cells, {len(nf)} files, "
              f"{sum(len(v) for v in nr.values())} rows, {sum(len(v) for v in ncol.values())} cols"
              f"{(' (' + ', '.join(extra) + ')') if extra else ''}\n")

    # Per-detector scope (datasets the detector actually produced output on).
    ran_path = Path(str(pred_path) + ".ran.json")
    per_detector_scope: dict[str, list[str]] = {}
    if ran_path.exists():
        ran = json.loads(ran_path.read_text())
        if isinstance(ran, dict) and "per_detector" in ran:
            per_detector_scope = ran["per_detector"]

    # Held-out evaluation: rows a (few-shot) detector was given ground-truth labels
    # for. Cells in these rows are excluded from BOTH predictions and ground truth
    # so the detector is scored only on cells it had to generalize to.
    labeled_rows: dict[str, set[int]] = {}
    labeled_path = Path(str(pred_path) + ".labeled_rows.json")
    if args.holdout and labeled_path.exists():
        raw = json.loads(labeled_path.read_text())
        labeled_rows = {ds: set(int(r) for r in rows) for ds, rows in raw.items()}
        n_rows = sum(len(v) for v in labeled_rows.values())
        print(f"held-out: excluding {n_rows} labeled rows over {len(labeled_rows)} datasets "
              f"(few-shot training cells not scored)\n")

    def _heldout(cell):
        return cell[1] in labeled_rows.get(cell[0], ())

    # Synthetic few-shot: cells we injected as synthetic training errors. A flag on
    # one is detecting a SYNTHETIC error (the training signal) -- neither a real TP
    # nor FP -- so we neutralize predictions there. Default ON when the sidecar
    # exists (raha/baran/tabclean --synthetic-fewshot runs).
    synth_cells: set[tuple[str, int, int]] = set()
    synth_path = Path(str(pred_path) + ".synthetic_cells.json")
    if not args.no_synthetic_neutralize and synth_path.exists():
        raw = json.loads(synth_path.read_text())
        for ds, cells in raw.items():
            for (i, j) in cells:
                synth_cells.add((ds, int(i), int(j)))
        print(f"synthetic few-shot: neutralizing {len(synth_cells)} injected cells "
              f"over {len(raw)} datasets (training signal, not scored)\n")

    # Predictions grouped by detector.
    preds_by_det: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        preds_by_det[r["detector"]].add((r["dataset"], int(r["row"]), int(r["col"])))
    # A valid detector may predict no cells. Preserve its declared run scope so
    # it is scored as zero recall rather than disappearing from the result file.
    for detector in per_detector_scope:
        preds_by_det.setdefault(detector, set())

    # Restrict to the current benchmark universe: drop predictions and scope
    # entries for datasets absent from _manifest.json. A benchmark update that
    # removes/replaces a source table takes it out of scope entirely -- a flag on
    # a table that is no longer part of the benchmark is neither a TP nor an FP.
    if args.restrict:
        manifest_set = set(all_datasets)
        n_dropped_preds = 0
        for det in list(preds_by_det):
            before = len(preds_by_det[det])
            preds_by_det[det] = {p for p in preds_by_det[det] if p[0] in manifest_set}
            n_dropped_preds += before - len(preds_by_det[det])
        for det in list(per_detector_scope):
            per_detector_scope[det] = [d for d in per_detector_scope[det] if d in manifest_set]
        if n_dropped_preds:
            print(f"restrict-to-manifest: dropped {n_dropped_preds} predicted cells on "
                  f"datasets absent from the current manifest (out of benchmark scope)\n")

    results = {}
    for det, preds in sorted(preds_by_det.items()):
        scope = per_detector_scope.get(det) or all_datasets
        error_cells, ambiguous_cells, _types, gt_by_type, gt_by_type_fine = load_ground_truth(
            ds_root, scope)
        # Hold out the detector's labeled (training) rows from preds AND ground truth.
        n_heldout_pred = 0
        if labeled_rows:
            before = len(preds)
            preds = {p for p in preds if not _heldout(p)}
            n_heldout_pred = before - len(preds)
            error_cells = {c for c in error_cells if not _heldout(c)}
            ambiguous_cells = {c for c in ambiguous_cells if not _heldout(c)}
            gt_by_type = {t: {c for c in cells if not _heldout(c)} for t, cells in gt_by_type.items()}
            gt_by_type_fine = {t: {c for c in cells if not _heldout(c)}
                               for t, cells in gt_by_type_fine.items()}
        gt = (error_cells | ambiguous_cells) if args.ambiguous == "error" else set(error_cells)

        if args.ambiguous == "exclude":
            preds = preds - ambiguous_cells  # neutral
        # Neutralize synthetic-injected cells (the few-shot training signal).
        if synth_cells:
            preds = preds - synth_cells
        # Drop skipped/excluded predictions (but a labeled error always wins).
        n_neutral_dropped = 0
        if not args.no_neutral:
            kept = {p for p in preds if (p in error_cells) or not _is_neutral(p[0], p[1], p[2], neutral)}
            n_neutral_dropped = len(preds) - len(kept)
            preds = kept
        tp = len(preds & gt)
        fp = len(preds) - tp
        fn = len(gt) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_type = {}
        for t, cells in sorted(gt_by_type.items()):
            hit = len(preds & cells)
            per_type[t] = {"recall": hit / len(cells) if cells else 0.0, "gt": len(cells), "hit": hit}

        per_type_fine = {}
        for t, cells in sorted(gt_by_type_fine.items()):
            hit = len(preds & cells)
            per_type_fine[t] = {"recall": hit / len(cells) if cells else 0.0,
                                "gt": len(cells), "hit": hit}

        results[det] = {
            "scope_datasets": len(scope), "n_errors": len(error_cells), "n_ambiguous": len(ambiguous_cells),
            "tp": tp, "fp": fp, "fn": fn, "n_neutral_dropped": n_neutral_dropped,
            "precision": precision, "recall": recall, "f1": f1,
            "n_predicted": len(preds), "per_type_recall": per_type,
            "per_type_recall_fine": per_type_fine,
        }

    # ---- report ----
    print(f"=== REIN scores ({pred_path.name}) ===")
    print(f"total datasets available : {len(all_datasets)}  | ambiguous mode: {args.ambiguous}\n")
    print(f"{'detector':>16} {'cov':>7} {'errGT':>6} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>7} {'FN':>6} {'neut':>6}")
    for det, m in results.items():
        print(f"{det:>16} {m['scope_datasets']:>4}ds {m['n_errors']:6d} "
              f"{m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
              f"{m['tp']:6d} {m['fp']:7d} {m['fn']:6d} {m['n_neutral_dropped']:6d}")
    print("\nper-error-type recall (coarse):")
    types = sorted({t for m in results.values() for t in m["per_type_recall"]})
    print(f"{'detector':>16} " + " ".join(f"{t[:12]:>13}" for t in types))
    for det, m in results.items():
        row = " ".join(
            f"{m['per_type_recall'][t]['recall']:7.3f}({m['per_type_recall'][t]['gt']:>4})"
            if t in m["per_type_recall"] else f"{'-':>13}"
            for t in types
        )
        print(f"{det:>16} {row}")

    print("\nper-error-type recall (fine):")
    types_f = sorted({t for m in results.values() for t in m.get("per_type_recall_fine", {})})
    print(f"{'detector':>16} " + " ".join(f"{t[:14]:>15}" for t in types_f))
    for det, m in results.items():
        pf = m.get("per_type_recall_fine", {})
        row = " ".join(
            f"{pf[t]['recall']:8.3f}({pf[t]['gt']:>4})"
            if t in pf else f"{'-':>15}"
            for t in types_f
        )
        print(f"{det:>16} {row}")
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = pred_path.stem.replace("predictions_", "")
    out_path = Path(args.out) if args.out else C.RESULTS_DIR / f"scores_{tag}.json"
    out_path.write_text(json.dumps(
        {"total_datasets": len(all_datasets), "ambiguous_mode": args.ambiguous, "results": results}, indent=2))
    print(f"\nscores -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
