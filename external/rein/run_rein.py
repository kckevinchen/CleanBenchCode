"""Runner: execute REIN error detectors on the materialized benchmark datasets.

To stay runnable on Python 3.12 (REIN's monolithic ``rein/detectors.py`` eagerly
imports torch/tensorflow/holoclean/... which do not install on 3.12), this script
**reimplements the lightweight detectors faithfully** from
``rein-benchmark/rein/detectors.py`` and invokes the heavier tools directly:

* ``mvdetector``       explicit missing values (empty string)        [pure]
* ``outlierdetector``  per-column SD / IQR / IsolationForest         [pure]
* ``dboost``           histogram/gaussian outlier models (vendored)  [lazy]
* ``raha``             config-free ensemble detector (``pip raha``)  [lazy]

Each detector emits a set of ``(row_i, col_j)`` cells per dataset -- exactly the
representation REIN scores. Predictions are written as JSONL
(``{dataset, detector, row, col}``) for ``score.py``.

Examples (repo root, main venv):
    python3 external/rein/run_rein.py --detector outlierdetector --outlier-method IF
    python3 external/rein/run_rein.py --detector mvdetector --detector outlierdetector
    python3 external/rein/run_rein.py --detector raha            # needs: pip install raha
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# raha forks multiprocessing Pools inside both detection (run_strategies) and
# Baran correction. Forking after numpy/sklearn have spawned BLAS/OpenMP threads
# deadlocks; pin thread counts to 1 BEFORE numpy is imported so the fork is safe.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

Cell = tuple[int, int]


# --------------------------------------------------------------------------- #
# Numeric coercion matching REIN's ``df.apply(pd.to_numeric, errors="ignore")``
# (removed in pandas 3.0): a column becomes numeric only if EVERY value parses.
# --------------------------------------------------------------------------- #
def _to_numeric_ignore(s: pd.Series) -> pd.Series:
    conv = pd.to_numeric(s, errors="coerce")
    return s if conv.isna().any() else conv


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(_to_numeric_ignore)


# --------------------------------------------------------------------------- #
# Detectors (faithful ports of rein/detectors.py)
# --------------------------------------------------------------------------- #
def detect_mvdetector(df: pd.DataFrame, ds_dir: Path, args) -> set[Cell]:
    out: set[Cell] = set()
    for j, col in enumerate(df.columns):
        for i in range(len(df)):
            if df.iat[i, j] == "":
                out.add((i, j))
    return out


def detect_outlierdetector(df: pd.DataFrame, ds_dir: Path, args) -> set[Cell]:
    method = args.outlier_method

    def SD(x, nstd=3.0):
        mean, std = np.mean(x), np.std(x)
        cut = std * nstd
        lo, hi = mean - cut, mean + cut
        return lambda y: (y > hi) | (y < lo)

    def IQR(x, k=1.5):
        q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
        cut = (q75 - q25) * k
        lo, hi = q25 - cut, q75 + cut
        return lambda y: (y > hi) | (y < lo)

    def IF(x, contamination=0.01):
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(contamination=contamination, random_state=0)
        clf.fit(x.reshape(-1, 1))
        return lambda y: (clf.predict(y.reshape(-1, 1)) == -1)

    fn = {"SD": SD, "IQR": IQR, "IF": IF}[method]

    num = _coerce_numeric(df).select_dtypes(include="number")
    out: set[Cell] = set()
    for col in num.columns:
        j = df.columns.get_loc(col)
        x = num[col].to_numpy(dtype=float)
        if x.size == 0:
            continue
        try:
            is_outlier = fn(x)(x)
        except Exception:
            continue
        for i, flag in enumerate(np.asarray(is_outlier).ravel()):
            if bool(flag):
                out.add((i, j))
    return out


def detect_dboost(df: pd.DataFrame, ds_dir: Path, args) -> set[Cell]:
    """Vendored dBoost with REIN's grid search.

    NOTE 1: like REIN, the best config is chosen by F1 against ground truth, so
    this is a per-dataset *oracle upper bound*, not a blind detector.
    NOTE 2: the dBoost copy vendored in ``mohamedyd/rein-benchmark`` is INCOMPLETE
    (its ``dboost/utils/`` package is absent), so this detector is EXPERIMENTAL.
    To enable it, vendor ``utils/`` from upstream (github.com/cpitclaudel/dBoost)
    into ``rein-benchmark/cleaners/dBoost/dboost/``. For a working, non-oracle
    outlier baseline use ``--detector outlierdetector`` instead.
    """
    import hashlib
    import itertools
    import json as _json
    import random
    import tempfile

    dboost_pkg = C.REIN_REPO / "cleaners" / "dBoost"
    if not (dboost_pkg / "dboost" / "utils").is_dir():
        raise ImportError(
            "vendored dBoost is incomplete (missing dboost/utils/); see detect_dboost "
            "docstring. Use --detector outlierdetector for a working outlier baseline."
        )
    sys.path.insert(0, str(dboost_pkg))
    from dboost.imported_dboost import run_dboost  # type: ignore

    actual = _load_actual_errors(ds_dir)
    grid = [list(a) for a in (
        list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                               ["0.1", "0.3", "0.5", "0.7", "0.9"]))
        + list(itertools.product(["gaussian"], ["1.0", "1.5", "2.0", "2.5", "3.0"]))
    )]
    random.shuffle(grid)

    best_f1, best_cells = -1.0, set()
    for config in grid:
        name = _json.dumps(["OD", config])
        h = str(int(hashlib.sha1(name.encode()).hexdigest(), 16))
        tmp = Path(tempfile.gettempdir()) / f"{ds_dir.name}-{h}.csv"
        df.to_csv(tmp, sep=",", header=True, index=False, encoding="utf-8")
        params = ["-F", ",", "--statistical", "0.5", "--" + config[0]] + config[1:] + [str(tmp)]
        cells: set[Cell] = set()
        try:
            run_dboost(params)
            res = Path(str(tmp) + "-dboost_output.csv")
            if res.exists():
                ocdf = pd.read_csv(res, sep=",", header=None, dtype=str,
                                   keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
                for i, j in ocdf.values.tolist():
                    if int(i) > 0:
                        cells.add((int(i) - 1, int(j)))
                res.unlink()
        except Exception:
            pass
        finally:
            if tmp.exists():
                tmp.unlink()
        f1 = _f1(cells, actual)
        if f1 > best_f1:
            best_f1, best_cells = f1, cells
    return best_cells


# raha/baran label whole rows with ground truth; we capture which rows so the
# scorer can hold them out (a few-shot learner must not be tested on its own
# labeled examples — those leaked error cells are otherwise echoed back as
# guaranteed TPs).
_RAHA_LABELED_ROWS: dict[str, list[int]] = {}
_BARAN_LABELED_ROWS: dict[str, list[int]] = {}


def _budget_for(df: pd.DataFrame, args) -> int:
    """raha/baran label whole ROWS with ground truth, so cell-leakage == fraction
    of rows labeled. A proportional budget caps that leakage per table; a fixed
    budget (default 20) leaks far more on small tables."""
    import math

    if getattr(args, "raha_budget_frac", 0.0) and args.raha_budget_frac > 0:
        return max(1, math.ceil(args.raha_budget_frac * len(df)))
    return args.raha_labeling_budget


def _run_raha_detection(dataset_dict: dict, df: pd.DataFrame, args):
    """Replicate raha Detection.run() so we can capture d.labeled_tuples (run()
    returns only the detections). Returns the dataset object ``d`` (with
    ``d.detected_cells`` populated) and the sorted list of GT-labeled rows."""
    from raha.detection import Detection  # lazy: pip install raha

    app = Detection()
    if hasattr(app, "LABELING_BUDGET"):
        app.LABELING_BUDGET = _budget_for(df, args)
    app.VERBOSE = False
    d = app.initialize_dataset(dataset_dict)
    app.run_strategies(d)
    app.generate_features(d)
    app.build_clusters(d)
    while len(d.labeled_tuples) < app.LABELING_BUDGET and len(d.labeled_tuples) < d.dataframe.shape[0]:
        app.sample_tuple(d)
        if d.has_ground_truth:
            app.label_with_ground_truth(d)
    app.propagate_labels(d)
    app.predict_labels(d)
    return d, sorted(int(r) for r in d.labeled_tuples)


def _dataset_dict(ds_dir: Path) -> dict:
    return {
        "name": ds_dir.name,
        "path": str((ds_dir / "dirty.csv").resolve()),
        "clean_path": str((ds_dir / "clean.csv").resolve()),
    }


# --------------------------------------------------------------------------- #
# Synthetic few-shot support: train raha/baran on SYNTHETIC errors only (never
# the real benchmark labels). We inject a fixed synthetic recipe into every row
# (src/synth, via external/synth_fewshot/inject.py) to get ``df_syn``; the
# few-shot oracle is then just the ORIGINAL table, so diffing reveals only the
# synthetic injections. raha/baran sample+label rows under this synthetic oracle
# and are scored on the REAL errors (synthetic cells neutralized; labeled rows
# held out). See external/synth_fewshot/inject.py for the rationale.
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth_fewshot"))
_SYNTH_CELLS: dict[str, list] = {}
_SYNTH_CACHE: dict[str, tuple] = {}
_SYNTH_ROOT = Path(__file__).resolve().parents[1] / "synth_fewshot" / "tmp_rein"


def _synthetic_dataset(ds_dir: Path, args):
    """Build (and cache) the synthetic-error version of a dataset: ``df_syn`` as
    dirty, the ORIGINAL table as the clean oracle. Returns
    ``(dataset_dict, df_syn, injections)``. Density per ``args.synthetic_density``:
    ``realistic`` matches the table's real error count, ``dense`` uses the fixed
    ~half-the-rows recipe."""
    name = ds_dir.name
    if name in _SYNTH_CACHE:
        return _SYNTH_CACHE[name]
    import inject as SI  # external/synth_fewshot/inject.py

    df = pd.read_csv(ds_dir / "dirty.csv", dtype=str, keep_default_na=False)
    protected = SI.load_protected_cells(ds_dir / "gt_cells.jsonl")
    density = getattr(args, "synthetic_density", "dense")
    n_errors = SI.count_real_errors(ds_dir / "gt_cells.jsonl") if density == "realistic" else None
    df_syn, injs = SI.build_synthetic_table(df, protected, n_errors=n_errors)
    out = _SYNTH_ROOT / density / name
    out.mkdir(parents=True, exist_ok=True)
    # Drop any stale raha/Baran strategy caches from a prior materialization: those
    # .dictionary files are keyed by strategy config, NOT data content, so raha
    # would silently reuse feature vectors computed on an OLD synthetic dirty.csv
    # (e.g. after the benchmark sources were re-sanitized). Mirrors prepare_datasets.
    for stale in out.glob("raha-baran-results-*"):
        if stale.is_dir():
            shutil.rmtree(stale)
    # Round-trip BOTH through pandas so non-injected cells are byte-identical and
    # raha's dirty-vs-clean diff is exactly the synthetic injections.
    df_syn.to_csv(out / "dirty.csv", index=False, encoding="utf-8")
    df.to_csv(out / "clean.csv", index=False, encoding="utf-8")
    dd = {"name": name, "path": str((out / "dirty.csv").resolve()),
          "clean_path": str((out / "clean.csv").resolve())}
    _SYNTH_CELLS[name] = [[x["i"], x["j"]] for x in injs]
    res = (dd, df_syn, injs)
    _SYNTH_CACHE[name] = res
    return res


_BARAN_PATCHED = False


def _patch_baran_serial() -> None:
    """Run Baran's two ``multiprocessing.Pool`` stages serially.

    raha's ``Correction`` forks a Pool inside ``generate_features`` (train
    features) and ``predict_correction_multicore`` (prediction), once per label
    round. Forking after numpy/sklearn have spawned BLAS/OpenMP threads
    deadlocks (and ``pool.close()`` is never ``join``-ed). Our tables are tiny,
    so serial execution is both correct and fast. The inner ``*_process``
    helpers already accept an explicit ``dataset``/``cls_model``, so we just
    drive them without a Pool.
    """
    global _BARAN_PATCHED
    if _BARAN_PATCHED:
        return
    import itertools

    from raha.correction import Correction

    def generate_features(self, d, cells):
        if len(cells) == 0:
            yield {}, []
            return
        for chunk in itertools.zip_longest(*[iter(cells)] * self.CHUNK_SIZE):
            _, pair_features_out, cell_list = self._feature_generator_process(chunk, dataset=d)
            yield pair_features_out, cell_list

    def predict_correction_multicore(self, classification_model, used_cells_test, d, all_zeros, all_ones):
        for chunk in itertools.zip_longest(*[iter(used_cells_test)] * self.CHUNK_SIZE):
            correction_dict = self._prediction_process(
                chunk, all_ones=all_ones, all_zeros=all_zeros,
                dataset=d, cls_model=classification_model,
            )
            d.corrected_cells.update(correction_dict)

    Correction.generate_features = generate_features
    Correction.predict_correction_multicore = predict_correction_multicore
    _BARAN_PATCHED = True


def detect_raha(df: pd.DataFrame, ds_dir: Path, args) -> set[Cell]:
    if getattr(args, "synthetic_fewshot", False):
        dataset_dict, _df_syn, _ = _synthetic_dataset(ds_dir, args)
    else:
        dataset_dict = _dataset_dict(ds_dir)
    d, rows = _run_raha_detection(dataset_dict, df, args)
    _RAHA_LABELED_ROWS[ds_dir.name] = rows
    return {(int(i), int(j)) for (i, j) in d.detected_cells.keys()}


def detect_baran(df: pd.DataFrame, ds_dir: Path, args) -> set[Cell]:
    """Baran (raha's correction engine) as a detector: Raha *detects* candidate
    error cells, Baran learns repair transformations from a few GT-labeled tuples
    and proposes corrections, and a cell counts as a detection iff Baran actually
    *changes* its value (correction != dirty). Few-shot, scored held-out on the
    union of both stages' labeled rows. This is the realistic Raha+Baran system,
    not Baran handed perfect (ground-truth) detections.
    """
    import raha.dataset
    from raha.correction import Correction

    _patch_baran_serial()
    if getattr(args, "synthetic_fewshot", False):
        dataset_dict, compare_df, _ = _synthetic_dataset(ds_dir, args)
    else:
        dataset_dict = _dataset_dict(ds_dir)
        compare_df = df

    # Stage 1 — Raha detection gives realistic (imperfect) candidate error cells.
    d_det, raha_rows = _run_raha_detection(dataset_dict, df, args)
    detected = dict(d_det.detected_cells)
    if not detected:
        _BARAN_LABELED_ROWS[ds_dir.name] = raha_rows
        return set()

    # Stage 2 — Baran correction on a FRESH dataset seeded with those detections.
    data = raha.dataset.Dataset(dataset_dict)
    data.detected_cells = detected
    cor = Correction()
    cor.VERBOSE = False
    cor.SAVE_RESULTS = False  # don't write raha-baran-results-* dirs next to datasets
    cor.NUM_WORKERS = max(1, int(getattr(args, "baran_workers", 2)))
    # raha's Correction.run() loops `while len(labeled_tuples) < LABELING_BUDGET`
    # with no row-count guard (unlike detection), so a budget exceeding the row
    # count spins forever re-sampling already-labeled rows. Cap at n_rows.
    cor.LABELING_BUDGET = max(1, min(_budget_for(df, args), len(df)))
    corrected = cor.run(data)  # {(i, j): corrected_value}

    baran_rows = {int(r) for r in getattr(data, "labeled_tuples", {})}
    _BARAN_LABELED_ROWS[ds_dir.name] = sorted(set(raha_rows) | baran_rows)

    # Detection = cells whose Baran-proposed value actually differs from the dirty
    # value it saw (a no-op "correction" is not a detection). In synthetic mode the
    # table it saw is df_syn, so compare against that.
    changed: set[Cell] = set()
    n_rows, n_cols = compare_df.shape
    for (i, j), val in corrected.items():
        if 0 <= i < n_rows and 0 <= j < n_cols and str(val) != str(compare_df.iat[i, j]):
            changed.add((int(i), int(j)))
    return changed


DETECTORS = {
    "mvdetector": detect_mvdetector,
    "outlierdetector": detect_outlierdetector,
    "dboost": detect_dboost,
    "raha": detect_raha,
    "baran": detect_baran,
}
ORACLE_DETECTORS = {"dboost"}  # peek at labels to pick config


# --------------------------------------------------------------------------- #
def _load_actual_errors(ds_dir: Path) -> set[Cell]:
    p = ds_dir / "actual_errors.csv"
    out: set[Cell] = set()
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.split(",")
                if len(parts) >= 2:
                    out.add((int(parts[0]), int(parts[1])))
    return out


def _f1(pred: set[Cell], actual: set[Cell]) -> float:
    if not pred or not actual:
        return 0.0
    tp = len(pred & actual)
    prec = tp / len(pred)
    rec = tp / len(actual)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", action="append", required=True, choices=list(DETECTORS),
                    help="detector(s) to run (repeatable)")
    ap.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    ap.add_argument("--outlier-method", default="IF", choices=["SD", "IQR", "IF"])
    ap.add_argument("--raha-labeling-budget", type=int, default=20,
                    help="raha: fixed number of ground-truth-labeled tuples (rows) per table")
    ap.add_argument("--raha-budget-frac", type=float, default=0.0,
                    help="raha/baran: instead use a per-table budget = ceil(frac * n_rows), "
                         "capping ground-truth cell-leakage to ~frac (e.g. 0.05).")
    ap.add_argument("--baran-workers", type=int, default=2,
                    help="baran: worker processes for correction prediction (default 2; "
                         "raha's default of all CPUs storms a shared box).")
    ap.add_argument("--synthetic-fewshot", action="store_true",
                    help="raha/baran: train ONLY on synthetic injected errors (fixed "
                         "src/synth recipe), never the real benchmark labels. The real "
                         "errors become the held-out test set (synthetic cells neutralized, "
                         "labeled rows held out). Honest few-shot eval without value-leak.")
    ap.add_argument("--synthetic-density", choices=["dense", "realistic"], default="dense",
                    help="synthetic-fewshot density: 'dense' (~half the rows, generous "
                         "signal) or 'realistic' (match each table's real error count).")
    ap.add_argument("--tag", default="", help="override output detector label / filename tag")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="", help="predictions JSONL (default: results/predictions_<detectors>.jsonl)")
    args = ap.parse_args()

    ds_root = Path(args.datasets_dir)
    manifest_path = ds_root / "_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Run prepare_datasets.py first.")
    manifest = json.loads(manifest_path.read_text())
    ds_names = sorted(manifest["datasets"])
    if args.limit:
        ds_names = ds_names[: args.limit]

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(args.detector)
    if "outlierdetector" in args.detector:
        tag += f"-{args.outlier_method}"
    if args.tag:
        tag = args.tag
    out_path = Path(args.out) if args.out else C.RESULTS_DIR / f"predictions_{tag}.jsonl"

    # Output label for each detector (so a re-parameterized run, e.g. low-budget
    # raha, is scored as its own distinct detector).
    def label_of(det: str) -> str:
        return args.tag if args.tag else det

    n_pred = 0
    ran_datasets: list[str] = []
    succeeded: dict[str, list[str]] = {label_of(d): [] for d in args.detector}
    meta = {d: {"datasets": 0, "cells": 0, "runtime_s": 0.0, "errors": 0} for d in args.detector}
    with open(out_path, "w", encoding="utf-8") as fout:
        for k, ds in enumerate(ds_names, 1):
            ds_dir = ds_root / ds
            try:
                df = C.load_rein_df(ds_dir / "dirty.csv")
            except Exception as e:
                print(f"[skip] {ds}: cannot load dirty.csv ({e})")
                continue
            ran_datasets.append(ds)
            for det in args.detector:
                t0 = time.time()
                try:
                    cells = DETECTORS[det](df, ds_dir, args)
                except Exception as e:
                    meta[det]["errors"] += 1
                    print(f"[err ] {det} on {ds}: {e}")
                    continue
                dt = time.time() - t0
                succeeded[label_of(det)].append(ds)
                meta[det]["datasets"] += 1
                meta[det]["cells"] += len(cells)
                meta[det]["runtime_s"] += dt
                for (i, j) in cells:
                    fout.write(json.dumps({"dataset": ds, "detector": label_of(det), "row": i, "col": j}) + "\n")
                    n_pred += 1
            if k % 25 == 0:
                print(f"  ...{k}/{len(ds_names)} datasets")

    ran_path = Path(str(out_path) + ".ran.json")
    ran_path.write_text(json.dumps({"per_detector": succeeded, "all_loaded": ran_datasets}))

    # few-shot detectors (raha/baran) label whole rows with ground truth; record
    # them so the scorer can hold those rows out of evaluation (don't credit a
    # few-shot learner for the GT error cells it was handed / echoes back).
    _fewshot_rows = {"raha": _RAHA_LABELED_ROWS, "baran": _BARAN_LABELED_ROWS}
    merged_labeled: dict[str, set[int]] = {}
    for det in args.detector:
        for ds, rows in _fewshot_rows.get(det, {}).items():
            merged_labeled.setdefault(ds, set()).update(rows)
    if merged_labeled:
        labeled_path = Path(str(out_path) + ".labeled_rows.json")
        labeled_path.write_text(json.dumps({ds: sorted(r) for ds, r in merged_labeled.items()}))
        print(f"few-shot labeled rows -> {labeled_path} ({len(merged_labeled)} datasets)")

    # synthetic few-shot: record the injected (synthetic) cells so the scorer can
    # neutralize them (a flag on a synthetic error is neither a real TP nor FP).
    if getattr(args, "synthetic_fewshot", False) and _SYNTH_CELLS:
        synth_path = Path(str(out_path) + ".synthetic_cells.json")
        synth_path.write_text(json.dumps({ds: cells for ds, cells in _SYNTH_CELLS.items()
                                          if ds in set(ran_datasets)}))
        print(f"synthetic cells -> {synth_path} ({len(_SYNTH_CELLS)} datasets)")

    print("=== run_rein summary ===")
    for det in args.detector:
        m = meta[det]
        oracle = "  [ORACLE upper-bound: peeks at labels]" if det in ORACLE_DETECTORS else ""
        print(f"{det:>16}: {m['cells']:6d} cells over {m['datasets']:3d} datasets, "
              f"{m['runtime_s']:6.1f}s, {m['errors']} errors{oracle}")
    print(f"predictions -> {out_path} ({n_pred} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
