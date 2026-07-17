"""Adapter for the ED2 upstream implementation.

The upstream ED2 code is vendored unchanged under
``external/ed2/SequentialPatternErrorDetection``.  It is Python-2-era code, so
this adapter converts a working copy with ``fissix`` outside the clone, patches
Python 3 compatibility issues in that copy, and drives the upstream feature
generation / active-learning helpers under the benchmark's synthetic few-shot
protocol.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import random
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

ED2_DIR = Path(__file__).resolve().parent
REPO_ROOT = ED2_DIR.parents[1]
UPSTREAM_ROOT = ED2_DIR / "SequentialPatternErrorDetection"
UPSTREAM_ML = UPSTREAM_ROOT / "model" / "ml"
PY3_VENDOR = ED2_DIR / "py3_vendor"
PY3_ML = PY3_VENDOR / "ml"

DETECTOR = "ed2"
SEED = 20240601
DEFAULT_LABELING_BUDGET = 20
DEFAULT_XGB_ROUNDS = 25


@dataclasses.dataclass
class DetectionResult:
    cells: set[tuple[int, int]]
    labeled_rows: list[int]
    synthetic_cells: list[list[int]]
    n_labels: int
    n_injections: int
    error: str | None = None


def pinned_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def ensure_py3_vendor(force: bool = False) -> Path:
    """Create the converted Python 3 copy used for imports.

    The vendored clone is never edited.  The converted copy is keyed by the
    upstream commit so stale conversions are rebuilt automatically.
    """

    marker = PY3_VENDOR / ".converted_commit"
    commit = pinned_commit()
    if not force and PY3_ML.exists() and marker.exists() and marker.read_text().strip() == commit:
        return PY3_VENDOR

    from fissix.refactor import RefactoringTool, get_fixers_from_package

    if PY3_VENDOR.exists():
        shutil.rmtree(PY3_VENDOR)
    PY3_ML.mkdir(parents=True, exist_ok=True)

    tool = RefactoringTool(get_fixers_from_package("fissix.fixes"))
    for path in UPSTREAM_ML.rglob("*"):
        rel = path.relative_to(UPSTREAM_ML)
        out = PY3_ML / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".py":
            shutil.copy2(path, out)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.endswith("\n"):
            text += "\n"
        converted = str(tool.refactor_string(text, str(path)))
        converted = converted.replace("from sets import Set", "Set = set")
        converted = converted.replace(
            'feature[i,0] = re.findall("^[A-Za-z_]+$", value)',
            'feature[i,0] = bool(re.findall("^[A-Za-z_]+$", value))',
        )
        out.write_text(converted, encoding="utf-8")

    marker.write_text(commit + "\n", encoding="utf-8")
    return PY3_VENDOR


def _load_upstream() -> tuple[Any, Any]:
    ensure_py3_vendor()
    vendor = str(PY3_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from ml.active_learning import library as ed2_lib
    from ml.datasets.DataSet import DataSet

    return DataSet, ed2_lib


def _build_synthetic_dataset(ds_dir: Path, density: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    synth_dir = REPO_ROOT / "external" / "synth_fewshot"
    if str(synth_dir) not in sys.path:
        sys.path.insert(0, str(synth_dir))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import inject as synth_inject

    df = pd.read_csv(ds_dir / "dirty.csv", dtype=str, keep_default_na=False, low_memory=False)
    protected = synth_inject.load_protected_cells(ds_dir / "gt_cells.jsonl")
    n_errors = synth_inject.count_real_errors(ds_dir / "gt_cells.jsonl") if density == "realistic" else None
    df_syn, injections = synth_inject.build_synthetic_table(df, protected, n_errors=n_errors)
    return df, df_syn, injections


def _train_predict_xgb(train_x, train_y, x_all, *, rounds: int, seed: int):
    import xgboost as xgb

    y = np.asarray(train_y).ravel().astype(int)
    if len(np.unique(y)) < 2:
        proba = np.zeros(x_all.shape[0], dtype=float)
        return proba, proba > 0.5

    params = {
        "learning_rate": 0.1,
        "colsample_bytree": 0.8,
        "seed": seed,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": 1,
        "verbosity": 0,
        "max_depth": 3,
        # The original ED2 grid included min_child_weight=1 for old XGBoost.
        # Modern XGBoost will not split four-label seed sets with that value,
        # yielding constant 0.5 predictions; 0.01 preserves the intended small
        # active-learning seed behavior.
        "min_child_weight": 0.01,
        "subsample": 0.9,
    }
    booster = xgb.train(
        params,
        xgb.DMatrix(train_x, label=y),
        num_boost_round=max(1, int(rounds)),
        verbose_eval=False,
    )
    proba = booster.predict(xgb.DMatrix(x_all))
    return proba, proba > 0.5


def detect_dataset(
    ds_dir: Path,
    *,
    labeling_budget: int = DEFAULT_LABELING_BUDGET,
    synthetic_density: str = "dense",
    seed: int = SEED,
    xgb_rounds: int = DEFAULT_XGB_ROUNDS,
    ngrams: int = 1,
    use_metadata: bool = True,
    use_correlation: bool = True,
    quiet: bool = True,
) -> DetectionResult:
    """Run ED2 on one benchmark dataset using only synthetic few-shot labels."""

    try:
        DataSet, ed2_lib = _load_upstream()
        random.seed(seed)
        np.random.seed(seed)

        df, df_syn, injections = _build_synthetic_dataset(ds_dir, synthetic_density)
        synthetic_cells = [[int(x["i"]), int(x["j"])] for x in injections]
        if df.empty or df.shape[1] == 0 or not injections:
            return DetectionResult(set(), [], synthetic_cells, 0, len(injections))

        data = DataSet(ds_dir.name, df_syn.copy(), df.copy())
        n_rows, n_cols = data.shape
        budget = max(0, int(labeling_budget))
        if budget < 4:
            return DetectionResult(set(), [], synthetic_cells, 0, len(injections))

        train_indices = list(range(n_rows))
        test_indices: list[int] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_matrix, all_matrix_test, feature_names = ed2_lib.create_features(
                data,
                train_indices,
                test_indices,
                ngrams,
                False,
                False,
                True,
            )
            if use_metadata:
                all_matrix, all_matrix_test, feature_names = ed2_lib.add_metadata_features(
                    data,
                    train_indices,
                    test_indices,
                    all_matrix,
                    all_matrix_test,
                    feature_names,
                    False,
                )
        feature_matrix = all_matrix.tocsr()

        target_matrix = data.matrix_is_error
        all_error_status = np.zeros((n_rows, n_cols), dtype=bool)
        train: dict[int, Any] = {}
        train_target: dict[int, Any] = {}
        train_chosen_ids: dict[int, list[int]] = {}
        current_predictions: dict[int, np.ndarray] = {}
        diff_certainty: dict[int, np.ndarray] = {}
        labeled_rows: set[int] = set()
        labels_used = 0
        exhausted: set[int] = set()

        def fit_column(column_id: int) -> bool:
            data_x_matrix = train[column_id].copy()
            x_all = feature_matrix.copy()
            if use_correlation and current_predictions:
                try:
                    data_x_matrix, x_all = ed2_lib.augment_features_with_predictions(
                        data_x_matrix,
                        feature_matrix,
                        current_predictions,
                        column_id,
                        train_chosen_ids,
                    )
                except Exception:
                    data_x_matrix = train[column_id].copy()
                    x_all = feature_matrix.copy()
            proba, pred = _train_predict_xgb(
                data_x_matrix,
                train_target[column_id],
                x_all,
                rounds=xgb_rounds,
                seed=seed,
            )
            current_predictions[column_id] = np.asarray(proba)
            diff_certainty[column_id] = np.absolute(np.asarray(proba) - 0.5)
            all_error_status[:, column_id] = np.asarray(pred, dtype=bool)
            return True

        applicable = [
            c
            for c in range(n_cols)
            if int(np.sum(target_matrix[:, c] == True)) >= 2
            and int(np.sum(target_matrix[:, c] == False)) >= 2
        ]
        for column_id in applicable:
            if labels_used + 4 > budget:
                break
            target = target_matrix[:, column_id]
            start_x, start_y, start_ids = ed2_lib.create_user_start_data(
                feature_matrix,
                target,
                2,
                return_ids=True,
            )
            if start_x is None:
                exhausted.add(column_id)
                continue
            ids = [int(i) for i in start_ids]
            train[column_id] = start_x
            train_target[column_id] = np.asarray(start_y).ravel()
            train_chosen_ids[column_id] = ids
            labels_used += len(ids)
            labeled_rows.update(ids)
            fit_column(column_id)

        while labels_used < budget:
            candidates = [c for c in train if c not in exhausted and len(train_chosen_ids[c]) < n_rows]
            if not candidates:
                break
            column_id = min(
                candidates,
                key=lambda c: (
                    float(np.mean(diff_certainty.get(c, np.array([0.5])))),
                    c,
                ),
            )
            target = target_matrix[:, column_id]
            try:
                x_next, y_next, diff, id_next = ed2_lib.create_next_part(
                    feature_matrix,
                    target,
                    current_predictions[column_id],
                    1,
                    data,
                    column_id,
                    0.0,
                    train_chosen_ids[column_id],
                    seed,
                )
            except Exception:
                exhausted.add(column_id)
                continue
            new_ids = [int(i) for i in id_next]
            if not new_ids:
                exhausted.add(column_id)
                continue
            train[column_id], train_target[column_id], train_chosen_ids[column_id] = (
                ed2_lib.add_data_next(
                    train[column_id],
                    train_target[column_id],
                    train_chosen_ids[column_id],
                    x_next,
                    y_next,
                    new_ids,
                )
            )
            labels_used += len(new_ids)
            labeled_rows.update(new_ids)
            diff_certainty[column_id] = np.asarray(diff)
            fit_column(column_id)

        cells = {
            (int(i), int(j))
            for i, j in zip(*np.where(all_error_status), strict=False)
            if 0 <= int(i) < n_rows and 0 <= int(j) < n_cols
        }
        return DetectionResult(cells, sorted(labeled_rows), synthetic_cells, labels_used, len(injections))
    except Exception as exc:
        if quiet:
            return DetectionResult(set(), [], [], 0, 0, error=f"{type(exc).__name__}: {exc}")
        raise


def detect_dataset_quiet(ds_dir: Path, **kwargs) -> DetectionResult:
    if not kwargs.get("quiet", True):
        return detect_dataset(ds_dir, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return detect_dataset(ds_dir, **kwargs)
