"""Adapter from REIN dirty tables to upstream ZeroER's unsupervised EM model."""

from __future__ import annotations

import itertools
import math
import re
import sys
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler

ZEROER_SRC = Path(__file__).resolve().parent / "zeroer"
if str(ZEROER_SRC) not in sys.path:
    sys.path.insert(0, str(ZEROER_SRC))

import model as zeroer_model_module  # type: ignore  # noqa: E402
from model import ZeroerModel, get_y_init_given_threshold  # type: ignore  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ZeroERResult:
    """Predicted duplicate pairs and the cell projection used by the REIN scorer."""

    pairs: list[tuple[int, int, float]]
    cells: set[tuple[int, int]]
    stats: dict[str, int | float | str | bool]


class _QuietTqdm:
    def __init__(self, iterable):
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def __iter__(self):
        return iter(self.iterable)

    def set_description_str(self, _text: str) -> None:
        return None


def _norm(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().casefold().split())


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text))


def _qgrams(text: str, q: int = 3) -> set[str]:
    compact = re.sub(r"\s+", " ", text)
    if len(compact) <= q:
        return {compact} if compact else set()
    return {compact[i : i + q] for i in range(len(compact) - q + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _number(value: str) -> float | None:
    match = NUMBER_RE.search(value.replace("$", "").replace("%", ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _numeric_similarity(a: str, b: str) -> float:
    na = _number(a)
    nb = _number(b)
    if na is None or nb is None:
        return 0.0
    denom = max(abs(na), abs(nb), 1.0)
    return max(0.0, 1.0 - abs(na - nb) / denom)


def _sequence_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _pair_features(left: list[str], right: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    per_col: list[float] = []
    nonempty_agree = 0
    comparable = 0
    for idx, (raw_a, raw_b) in enumerate(zip(left, right, strict=False)):
        a = _norm(raw_a)
        b = _norm(raw_b)
        exact = 1.0 if a == b else 0.0
        seq = _sequence_ratio(a, b)
        token_j = _jaccard(_tokens(a), _tokens(b))
        qgram_j = _jaccard(_qgrams(a), _qgrams(b))
        num = _numeric_similarity(a, b)
        prefix = f"c{idx}"
        features[f"{prefix}_exact"] = exact
        features[f"{prefix}_seq"] = seq
        features[f"{prefix}_token_jaccard"] = token_j
        features[f"{prefix}_qgram_jaccard"] = qgram_j
        features[f"{prefix}_numeric"] = num
        per_col.append(max(exact, seq, token_j, qgram_j, num))
        if a or b:
            comparable += 1
            if exact:
                nonempty_agree += 1
    features["row_mean"] = float(np.mean(per_col)) if per_col else 0.0
    features["row_min"] = float(np.min(per_col)) if per_col else 0.0
    features["row_exact_fraction"] = nonempty_agree / comparable if comparable else 1.0
    features["row_nonempty_fraction"] = comparable / len(left) if left else 0.0
    return features


def _candidate_pairs(df: pd.DataFrame) -> tuple[list[tuple[int, int]], list[dict[str, float]]]:
    records = df.fillna("").astype(str).values.tolist()
    pairs: list[tuple[int, int]] = []
    feature_rows: list[dict[str, float]] = []
    for i, j in itertools.combinations(range(len(records)), 2):
        left = records[i]
        right = records[j]
        if not any(_norm(v) for v in left) and not any(_norm(v) for v in right):
            continue
        pairs.append((i, j))
        feature_rows.append(_pair_features(left, right))
    return pairs, feature_rows


def _safe_similarity_frame(feature_rows: list[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(feature_rows, dtype=float)
    frame.replace([np.inf, -np.inf], 0.0, inplace=True)
    frame.fillna(0.0, inplace=True)
    variable = [col for col in frame.columns if frame[col].nunique(dropna=False) > 1]
    return frame[variable]


def _has_two_class_seed(y_init: list[int]) -> bool:
    counts = {label: y_init.count(label) for label in set(y_init)}
    return counts.get(0, 0) >= 2 and counts.get(1, 0) >= 2


def _topk_seed(features: pd.DataFrame) -> list[int]:
    if len(features) < 4:
        return [0] * len(features)
    scaled = MinMaxScaler().fit_transform(features.values)
    scores = MinMaxScaler().fit_transform(np.sum(scaled, axis=1).reshape(-1, 1)).ravel()
    top_k = min(max(2, math.ceil(0.01 * len(features))), len(features) - 2)
    selected = set(np.argsort(-scores)[:top_k])
    return [1 if idx in selected else 0 for idx in range(len(features))]


def _zeroer_probabilities(
    features: pd.DataFrame,
    pairs: list[tuple[int, int]],
    run_transitivity: bool,
) -> tuple[np.ndarray, float]:
    y_init = get_y_init_given_threshold(features)
    init_threshold = 0.8
    if not _has_two_class_seed(y_init):
        for candidate in (0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4):
            candidate_y = get_y_init_given_threshold(features, threshold=candidate)
            if _has_two_class_seed(candidate_y):
                y_init = candidate_y
                init_threshold = candidate
                break
    if not _has_two_class_seed(y_init):
        y_init = _topk_seed(features)
        init_threshold = -1.0
    if not _has_two_class_seed(y_init):
        return np.zeros(len(features), dtype=float), init_threshold
    id_df = pd.DataFrame(pairs, columns=["ltable_id", "rtable_id"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        original_seed = zeroer_model_module.get_y_init_given_threshold
        original_tqdm = zeroer_model_module.tqdm
        zeroer_model_module.get_y_init_given_threshold = lambda _df, threshold=0.8: list(y_init)
        zeroer_model_module.tqdm = _QuietTqdm
        try:
            _model, probs = ZeroerModel.run_em(
                [features.values, None, None],
                features.columns,
                [y_init, None, None],
                (id_df, None, None),
                False,
                True,
                run_transitivity,
                y_true=None,
                hard=False,
                c_bay=0.1,
            )
        finally:
            zeroer_model_module.get_y_init_given_threshold = original_seed
            zeroer_model_module.tqdm = original_tqdm
    probs = np.asarray(probs, dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(probs, 0.0, 1.0), init_threshold


def detect_duplicate_rows(
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    run_transitivity: bool = False,
) -> ZeroERResult:
    """Run upstream ZeroER in single-table deduplication mode on ``df``.

    The benchmark's row-duplication labels mark whole duplicate rows
    (``columns: []`` in ``benchmark_row.json``). The REIN scorer is cell-level,
    so every cell in every row participating in a predicted duplicate pair is
    emitted.
    """

    n_rows, n_cols = df.shape
    stats: dict[str, int | float | str | bool] = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "candidate_pairs": 0,
        "feature_count": 0,
        "predicted_pairs": 0,
        "predicted_rows": 0,
        "threshold": threshold,
        "run_transitivity": run_transitivity,
        "status": "ok",
    }
    if n_rows < 2 or n_cols == 0:
        stats["status"] = "too_small"
        return ZeroERResult([], set(), stats)

    pairs, feature_rows = _candidate_pairs(df)
    stats["candidate_pairs"] = len(pairs)
    if not pairs:
        stats["status"] = "no_candidates"
        return ZeroERResult([], set(), stats)

    features = _safe_similarity_frame(feature_rows)
    stats["feature_count"] = int(features.shape[1])
    if features.empty:
        stats["status"] = "no_variable_features"
        return ZeroERResult([], set(), stats)

    try:
        probs, init_threshold = _zeroer_probabilities(features, pairs, run_transitivity)
        stats["init_threshold"] = init_threshold
    except Exception as exc:  # upstream can fail on degenerate Gaussian fits
        stats["status"] = f"zeroer_error:{type(exc).__name__}"
        stats["error"] = str(exc)[:500]
        return ZeroERResult([], set(), stats)

    pred_pairs = [
        (i, j, float(prob))
        for (i, j), prob in zip(pairs, probs, strict=True)
        if math.isfinite(float(prob)) and float(prob) > threshold
    ]
    duplicate_rows = {row for i, j, _prob in pred_pairs for row in (i, j)}
    cells = {(row, col) for row in duplicate_rows for col in range(n_cols)}
    stats["predicted_pairs"] = len(pred_pairs)
    stats["predicted_rows"] = len(duplicate_rows)
    return ZeroERResult(pred_pairs, cells, stats)
