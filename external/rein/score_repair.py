"""Scorer: cell-level micro precision/recall/F1 for value-level repair.

Companion to ``score.py`` (which grades *detection* — was a cell flagged?).
This scorer grades the *value* a method proposes for a cell against the gold
``clean.csv``, in two tasks:

* ``--task correction`` — **oracle-location correction** (§5.5): the method is
  handed the true error locations and only proposes replacement values. Scored
  over exactly the gold error cells:
    - TP = gold error cell with a usable proposed value equal to the gold value
    - FP = gold error cell the method *attempted* (usable value) but got wrong
    - FN = gold error cell left uncorrected (abstained or wrong)
    - precision = TP / attempts, recall = TP / |errors|.

* ``--task e2e`` — **end-to-end cleaning** (§5.6): the method both finds and
  fixes over the full corpus. A "change" is any predicted cell with a usable
  value (or, with ``--repaired-dir``, any cell whose repaired value differs from
  dirty). Scored over the whole run scope:
    - TP = change at a real gold error cell whose value equals the gold value
    - FP = change that is not a fully-correct clean (wrong cell OR wrong value)
    - FN = |errors| - TP
    - precision = TP / changes, recall = TP / |errors|.

Value equality mirrors the main pipeline (``common.values_equal``): whitespace-
normalized string match with numeric-surface-form equivalence; ``--casefold``
adds case-insensitivity, ``--no-numeric`` disables numeric equivalence.

Input is either a predictions JSONL carrying a value field
(``corrected_value`` / ``value`` / ``new_value`` / ``repaired`` / ``correction``)
with the usual ``.ran.json`` / ``.labeled_rows.json`` / ``.synthetic_cells.json``
sidecars, or ``--repaired-dir <dir>`` holding one ``<dataset>/repaired.csv`` per
dataset (diffed against the benchmark ``dirty.csv``).

Examples:
    python3 external/rein/score_repair.py --task e2e \
        --predictions external/rein/results/predictions_cleanagent.jsonl
    python3 external/rein/score_repair.py --task e2e --detector-name holoclean \
        --repaired-dir external/holoclean/cache/v8 --repaired-suffix __<commit>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import score as S  # noqa: E402  (reuse ground-truth / neutral / holdout loaders)

_VALUE_KEYS = ("corrected_value", "value", "new_value", "repaired", "correction")


def _extract_value(rec: dict):
    for k in _VALUE_KEYS:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None


def load_valued_predictions(pred_path: Path):
    """Parse a predictions JSONL into ``{detector: {(ds, i, j): value}}``."""
    preds: dict[str, dict[tuple[str, int, int], object]] = defaultdict(dict)
    n_novalue = 0
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        val = _extract_value(r)
        if val is None:
            n_novalue += 1
        key = (r["dataset"], int(r["row"]), int(r["col"]))
        preds[r["detector"]][key] = val
    return preds, n_novalue


def load_repaired_dir(
    repaired_dir: Path,
    ds_root: Path,
    all_datasets: list[str],
    detector: str,
    suffix: str,
    diff_casefold: bool,
):
    """Diff ``<repaired_dir>/<ds><suffix>/repaired.csv`` against benchmark dirty.

    Returns ``({detector: {(ds, i, j): repaired_value}}, scope)`` where a cell is
    a change iff its repaired value differs from the dirty value (exact by
    default; normalized/case-folded when ``diff_casefold`` — which suppresses
    pure re-casing / whitespace preprocessing artifacts).
    """
    tvcache: dict = {}
    preds: dict[tuple[str, int, int], object] = {}
    scope: list[str] = []
    for ds in all_datasets:
        rep_path = repaired_dir / f"{ds}{suffix}" / "repaired.csv"
        if not rep_path.exists():
            # tolerate the flat layout <repaired_dir>/<ds>.csv or <ds>/repaired.csv
            alt = repaired_dir / f"{ds}{suffix}.csv"
            rep_path = alt if alt.exists() else rep_path
        if not rep_path.exists():
            continue
        scope.append(ds)
        rep_rows = C.load_table_rows(rep_path)
        rep_data = rep_rows[1:] if rep_rows else []
        dirty_data, _clean = C.load_dirty_clean(ds_root, ds, tvcache)
        n = min(len(rep_data), len(dirty_data))
        for i in range(n):
            width = min(len(rep_data[i]), len(dirty_data[i]))
            for j in range(width):
                rv, dv = rep_data[i][j], dirty_data[i][j]
                changed = (
                    C.normalize_cell(rv, casefold=True) != C.normalize_cell(dv, casefold=True)
                    if diff_casefold
                    else rv != dv
                )
                if changed:
                    preds[(ds, i, j)] = rv
    return {detector: preds}, scope


def _filter_heldout(cellmap, labeled_rows):
    return {c: v for c, v in cellmap.items() if c[1] not in labeled_rows.get(c[0], ())}


def load_accepted_sets(bench_cell_json: Path, ds_root: Path, source2ds: dict[str, str]):
    """Load per-cell accepted correction sets from ``benchmark_cell.json``.

    Returns ``(accepted, noop)`` where:
    * ``accepted[(ds, i, j)]`` = set of gold accepted replacement strings (a cell's
      ``corrected_value`` may be a single string or a list of accepted variants).
    * ``noop`` = cells whose accepted set includes the *original* value
      (no-op-acceptable, e.g. inconsistency labels). These are neutralized for
      both correction and end-to-end scoring per the evaluation protocol.
    """
    accepted: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    noop: set[tuple[str, int, int]] = set()
    colcache: dict = {}
    if not bench_cell_json.exists():
        return {}, set()
    for line in bench_cell_json.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != C.STATUS_ERROR:
            continue
        ds = source2ds.get(rec.get("source"))
        if ds is None:
            continue
        j = S._ds_columns(ds_root, ds, colcache).get(rec.get("column"))
        if j is None or rec.get("row") is None:
            continue
        key = (ds, int(rec["row"]), j)
        cv = rec.get("corrected_value")
        cands = cv if isinstance(cv, list) else [cv]
        cands = [str(x) for x in cands if C.is_usable_correction(x)]
        if not cands:
            continue
        accepted[key].update(cands)
        ov = rec.get("original_value")
        if any(C.values_equal(x, ov) for x in cands):
            noop.add(key)
    return dict(accepted), noop


def _grade_correction(eligible, preds, match, gt_by_type):
    """Oracle-location: score over eligible gold error cells (no-op-acceptable removed).

    ``match(cell, value) -> bool`` tests a proposed value against the cell's
    accepted correction set. Returns counts + the set of correctly-repaired cells.
    """
    tp = attempts = 0
    correct_cells: set = set()
    for c in eligible:
        v = preds.get(c)
        if not C.is_usable_correction(v):
            continue
        attempts += 1
        if match(c, v):
            tp += 1
            correct_cells.add(c)
    fp = attempts - tp
    fn = len(eligible) - tp
    return tp, fp, fn, attempts, correct_cells


def _grade_e2e(eligible, preds, match):
    """End-to-end: score over every proposed change against the eligible error set."""
    changes = {c: v for c, v in preds.items() if C.is_usable_correction(v)}
    tp = 0
    tp_loc = 0  # correct location (an eligible error), regardless of value
    correct_cells: set = set()
    for c, v in changes.items():
        if c in eligible:
            tp_loc += 1
            if match(c, v):
                tp += 1
                correct_cells.add(c)
    fp = len(changes) - tp
    fn = len(eligible) - tp
    return tp, fp, fn, len(changes), tp_loc, correct_cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["correction", "e2e"],
                    help="correction=oracle-location repair (§5.5); e2e=detect+fix (§5.6)")
    ap.add_argument("--predictions", help="valued predictions JSONL (with corrected_value)")
    ap.add_argument("--repaired-dir", help="dir of <dataset>/repaired.csv full-table outputs")
    ap.add_argument("--repaired-suffix", default="",
                    help="suffix appended to <dataset> dir name under --repaired-dir")
    ap.add_argument("--detector-name", default="repair",
                    help="detector label to use with --repaired-dir")
    ap.add_argument("--diff-casefold", action="store_true",
                    help="with --repaired-dir: ignore pure re-casing/whitespace changes when "
                         "detecting edits (suppresses pipeline preprocessing artifacts)")
    ap.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    ap.add_argument("--ambiguous", default="exclude", choices=["exclude", "fp", "error"])
    ap.add_argument("--neutral-file",
                    default=str(C.DEFAULT_BENCHMARK_DIR / "neutral_cells.jsonl"))
    ap.add_argument("--no-neutral", action="store_true")
    ap.add_argument("--no-holdout-labeled-rows", dest="holdout", action="store_false", default=True)
    ap.add_argument("--no-synthetic-neutralize", action="store_true")
    ap.add_argument("--casefold", action="store_true", help="case-insensitive value match")
    ap.add_argument("--no-numeric", action="store_true", help="disable numeric-equivalence match")
    ap.add_argument("--accepted-sets",
                    default=str(C.DEFAULT_BENCHMARK_DIR / "benchmark_cell.json"),
                    help="benchmark_cell.json with per-cell accepted correction sets + "
                         "no-op-acceptable neutralization (default: benchmark of record)")
    ap.add_argument("--no-accepted-sets", action="store_true",
                    help="ignore accepted-set file; fall back to single clean.csv gold with no "
                         "no-op neutralization (less faithful; for ad-hoc scoring)")
    ap.add_argument("--out", default="", help="write scores JSON here")
    ap.add_argument("--restrict-to-manifest", dest="restrict", action="store_true", default=True,
                    help="drop predictions for datasets absent from the current _manifest.json "
                         "(benchmark universe). On by default; --repaired-dir mode is inherently "
                         "restricted since it iterates the manifest.")
    ap.add_argument("--no-restrict-to-manifest", dest="restrict", action="store_false",
                    help="score predicted cells even on datasets absent from the manifest.")
    args = ap.parse_args()

    if not args.predictions and not args.repaired_dir:
        raise SystemExit("provide --predictions or --repaired-dir")
    numeric = not args.no_numeric
    casefold = args.casefold

    ds_root = Path(args.datasets_dir)
    manifest = json.loads((ds_root / "_manifest.json").read_text())
    all_datasets = sorted(manifest["datasets"])
    source2ds = {entry["source"]: ds for ds, entry in manifest["datasets"].items()}

    # Skipped/excluded cells -> neutral (matches score.py).
    neutral = (set(), set(), {}, {}, {})
    if not args.no_neutral:
        neutral = S.load_neutral(Path(args.neutral_file), ds_root, source2ds)

    # ---- assemble predictions + per-detector scope ----
    per_detector_scope: dict[str, list[str]] = {}
    labeled_rows: dict[str, set[int]] = {}
    synth_cells: set[tuple[str, int, int]] = set()

    if args.repaired_dir:
        preds_by_det, scope = load_repaired_dir(
            Path(args.repaired_dir), ds_root, all_datasets,
            args.detector_name, args.repaired_suffix, args.diff_casefold)
        per_detector_scope[args.detector_name] = scope
        print(f"repaired-dir: {len(scope)} datasets with repaired.csv; "
              f"{sum(len(v) for v in preds_by_det.values())} changed cells "
              f"(diff_casefold={args.diff_casefold})\n")
    else:
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            raise SystemExit(f"Predictions not found: {pred_path}")
        preds_by_det, n_novalue = load_valued_predictions(pred_path)
        if n_novalue:
            print(f"WARNING: {n_novalue} prediction rows carried no value field "
                  f"({'/'.join(_VALUE_KEYS)}); they cannot be a repair TP.\n")
        ran_path = Path(str(pred_path) + ".ran.json")
        if ran_path.exists():
            ran = json.loads(ran_path.read_text())
            if isinstance(ran, dict) and "per_detector" in ran:
                per_detector_scope = ran["per_detector"]
        labeled_path = Path(str(pred_path) + ".labeled_rows.json")
        if args.holdout and labeled_path.exists():
            raw = json.loads(labeled_path.read_text())
            labeled_rows = {ds: {int(r) for r in rows} for ds, rows in raw.items()}
            print(f"held-out: excluding {sum(len(v) for v in labeled_rows.values())} labeled "
                  f"rows over {len(labeled_rows)} datasets\n")
        synth_path = Path(str(pred_path) + ".synthetic_cells.json")
        if not args.no_synthetic_neutralize and synth_path.exists():
            raw = json.loads(synth_path.read_text())
            for ds, cells in raw.items():
                for (i, j) in cells:
                    synth_cells.add((ds, int(i), int(j)))
            print(f"synthetic few-shot: neutralizing {len(synth_cells)} injected cells\n")

    tvcache: dict = {}

    # Restrict to the current benchmark universe (predictions mode): drop cells on
    # datasets absent from _manifest.json. --repaired-dir mode already iterates the
    # manifest, so it is inherently restricted.
    if args.restrict and not args.repaired_dir:
        manifest_set = set(all_datasets)
        n_dropped = 0
        for det in list(preds_by_det):
            before = len(preds_by_det[det])
            preds_by_det[det] = {c: v for c, v in preds_by_det[det].items()
                                 if c[0] in manifest_set}
            n_dropped += before - len(preds_by_det[det])
        for det in list(per_detector_scope):
            per_detector_scope[det] = [d for d in per_detector_scope[det] if d in manifest_set]
        if n_dropped:
            print(f"restrict-to-manifest: dropped {n_dropped} predicted cells on datasets "
                  f"absent from the current manifest (out of benchmark scope)\n")

    # Per-cell accepted correction sets + no-op-acceptable neutralization.
    accepted: dict[tuple[str, int, int], set[str]] = {}
    noop_acceptable: set[tuple[str, int, int]] = set()
    if not args.no_accepted_sets:
        accepted, noop_acceptable = load_accepted_sets(
            Path(args.accepted_sets), ds_root, source2ds)
        print(f"accepted sets: {len(accepted)} error cells "
              f"({sum(len(v) for v in accepted.values())} accepted values); "
              f"neutralizing {len(noop_acceptable)} no-op-acceptable cells\n")

    def gold_set(c):
        """Accepted replacement strings for a cell (fallback: single clean.csv value)."""
        if c in accepted:
            return accepted[c]
        ds, i, j = c
        _dirty, clean = C.load_dirty_clean(ds_root, ds, tvcache)
        cv = C.cell_value(clean, i, j)
        return {cv} if cv is not None else set()

    def match(c, v):
        return any(C.values_equal(v, g, numeric=numeric, casefold=casefold) for g in gold_set(c))

    results = {}
    for det, preds in sorted(preds_by_det.items()):
        scope = per_detector_scope.get(det) or all_datasets
        error_cells, ambiguous_cells, _types, gt_by_type, gt_by_type_fine = S.load_ground_truth(
            ds_root, scope)

        # Held-out (few-shot) rows excluded from BOTH predictions and gold.
        if labeled_rows:
            preds = _filter_heldout(preds, labeled_rows)
            hd = lambda c: c[1] in labeled_rows.get(c[0], ())  # noqa: E731
            error_cells = {c for c in error_cells if not hd(c)}
            ambiguous_cells = {c for c in ambiguous_cells if not hd(c)}
            gt_by_type = {t: {c for c in cs if not hd(c)} for t, cs in gt_by_type.items()}
            gt_by_type_fine = {t: {c for c in cs if not hd(c)} for t, cs in gt_by_type_fine.items()}
        # Synthetic-injected training cells neutralized.
        if synth_cells:
            preds = {c: v for c, v in preds.items() if c not in synth_cells}

        # No-op-acceptable cells (accepted set includes the original) are neutral for
        # BOTH tasks: removed from the eligible/gold denominator and from per-type sets.
        eligible = {c for c in error_cells if c not in noop_acceptable}
        gt_by_type = {t: {c for c in cs if c not in noop_acceptable} for t, cs in gt_by_type.items()}
        gt_by_type_fine = {t: {c for c in cs if c not in noop_acceptable}
                           for t, cs in gt_by_type_fine.items()}

        if args.task == "correction":
            tp, fp, fn, attempts, correct_cells = _grade_correction(
                eligible, preds, match, gt_by_type)
            denom_note = {"attempts": attempts}
            tp_loc = attempts  # every oracle cell is correctly located -> valAcc == precision
        else:  # e2e
            # Ambiguous predicted cells are neutral (exclude mode).
            if args.ambiguous == "exclude":
                preds = {c: v for c, v in preds.items() if c not in ambiguous_cells}
            # No-op-acceptable predicted cells are neutral (unchanged is already fine).
            preds = {c: v for c, v in preds.items() if c not in noop_acceptable}
            # Drop predictions on skipped/excluded cells (a gold error always wins).
            if not args.no_neutral:
                preds = {c: v for c, v in preds.items()
                         if (c in eligible) or not S._is_neutral(c[0], c[1], c[2], neutral)}
            tp, fp, fn, n_changes, tp_loc, correct_cells = _grade_e2e(eligible, preds, match)
            denom_note = {"n_changes": n_changes, "tp_location": tp_loc}

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        value_acc_located = (tp / tp_loc) if tp_loc else 0.0

        per_type = {}
        for t, cells in sorted(gt_by_type.items()):
            hit = len(cells & correct_cells)
            per_type[t] = {"recall": hit / len(cells) if cells else 0.0, "gt": len(cells), "hit": hit}
        per_type_fine = {}
        for t, cells in sorted(gt_by_type_fine.items()):
            hit = len(cells & correct_cells)
            per_type_fine[t] = {"recall": hit / len(cells) if cells else 0.0,
                                "gt": len(cells), "hit": hit}

        results[det] = {
            "task": args.task, "scope_datasets": len(scope), "n_errors": len(eligible),
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "value_acc_of_located": value_acc_located, **denom_note,
            "per_type_recall": per_type, "per_type_recall_fine": per_type_fine,
        }

    # ---- report ----
    src = args.repaired_dir or args.predictions
    print(f"=== repair scores [{args.task}] ({Path(src).name}) ===")
    print(f"total datasets available : {len(all_datasets)}  | value match: "
          f"{'casefold ' if casefold else ''}{'exact' if args.no_numeric else 'numeric-aware'}\n")
    extra_h = "attempts" if args.task == "correction" else "changes"
    print(f"{'detector':>18} {'cov':>7} {'errGT':>6} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'TP':>6} {'FP':>7} {'FN':>6} {extra_h:>8} {'valAcc':>7}")
    for det, m in results.items():
        extra_v = m.get("attempts", m.get("n_changes", 0))
        print(f"{det:>18} {m['scope_datasets']:>4}ds {m['n_errors']:6d} "
              f"{m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
              f"{m['tp']:6d} {m['fp']:7d} {m['fn']:6d} {extra_v:8d} "
              f"{m['value_acc_of_located']:7.3f}")

    print("\nper-error-type recall (coarse):")
    types = sorted({t for m in results.values() for t in m["per_type_recall"]})
    print(f"{'detector':>18} " + " ".join(f"{t[:12]:>13}" for t in types))
    for det, m in results.items():
        row = " ".join(
            f"{m['per_type_recall'][t]['recall']:7.3f}({m['per_type_recall'][t]['gt']:>4})"
            if t in m["per_type_recall"] else f"{'-':>13}" for t in types)
        print(f"{det:>18} {row}")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = Path(src).stem.replace("predictions_", "")
    out_path = Path(args.out) if args.out else C.RESULTS_DIR / f"repair_{args.task}_{tag}.json"
    out_path.write_text(json.dumps(
        {"task": args.task, "total_datasets": len(all_datasets), "results": results}, indent=2))
    print(f"\nscores -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
