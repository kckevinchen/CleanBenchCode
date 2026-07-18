"""Shared helpers for the REIN-on-benchmark integration.

These utilities are intentionally dependency-light (stdlib + pandas) so that the
adapter (``prepare_datasets.py``) and scorer (``score.py``) run in the project's
main ``venv``. Only ``run_rein.py`` pulls in detector-specific dependencies.
"""

from __future__ import annotations

import csv
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

# external/rein/common.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
REIN_DIR = REPO_ROOT / "external" / "rein"
REIN_REPO = REIN_DIR / "rein-benchmark"
DATASETS_DIR = REIN_REPO / "datasets"
RESULTS_DIR = REIN_DIR / "results"

# Benchmark of record. Override with environment variables without editing code.
#   DQ_BENCHMARK_DIR=/path/to/benchmark DQ_SOURCES_DIR=/path/to/sources ...
_DEFAULT_BENCHMARK = REPO_ROOT / "benchmark" / "CleanBench"
_DEFAULT_SOURCES = REPO_ROOT / "data" / "CleanBench"
DEFAULT_BENCHMARK_DIR = Path(os.environ.get("DQ_BENCHMARK_DIR", str(_DEFAULT_BENCHMARK)))
DEFAULT_SOURCES_DIR = Path(os.environ.get("DQ_SOURCES_DIR", str(_DEFAULT_SOURCES)))

# Status values in the benchmark cell records.
STATUS_ERROR = "error"
STATUS_AMBIGUOUS = "ambiguous"


def slugify_dataset(source: str) -> str:
    """Map a benchmark ``source`` basename to a filesystem-safe REIN dataset name.

    The benchmark ``source`` is a unique CSV basename (e.g.
    ``005c....xlsx_ed8fbac9.csv``); we strip the ``.csv`` suffix and keep only
    safe characters. A manifest records the exact reverse mapping.
    """
    name = source[:-4] if source.lower().endswith(".csv") else source
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def load_rein_df(path: str | Path) -> pd.DataFrame:
    """Read a CSV exactly the way REIN's ``datasets.load_data`` does.

    Matching REIN (``dtype=str``, ``keep_default_na=False``) keeps cell indices
    and values aligned with both the benchmark and the detectors.
    """
    return pd.read_csv(
        path,
        dtype=str,
        header="infer",
        encoding="utf-8",
        keep_default_na=False,
        low_memory=False,
    )


def read_benchmark_cells(benchmark_dir: str | Path) -> list[dict]:
    """Load the cell-level benchmark records (``benchmark_cell.json``)."""
    p = Path(benchmark_dir) / "benchmark_cell.json"
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cells_by_source(cells: list[dict]) -> dict[str, list[dict]]:
    """Group benchmark cell records by their ``source`` basename."""
    out: dict[str, list[dict]] = {}
    for rec in cells:
        out.setdefault(rec["source"], []).append(rec)
    return out


def is_usable_correction(value) -> bool:
    """True if ``corrected_value`` is a usable non-empty replacement string."""
    return value is not None and str(value).strip() != ""


# ---------------------------------------------------------------------------
# Value-equality helpers for the correction / end-to-end scorers.
#
# These mirror ``src/core/value_types.py`` so a repaired cell is judged the same
# way the main pipeline judges a proposed ``corrected_value``. Kept here (not
# imported) so the scorer stays runnable from the dependency-light ``venv``.
# ---------------------------------------------------------------------------

_PURE_NUMERIC_RE = re.compile(
    r"^[+-]?" r"(?:(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?)|(?:\.\d+))" r"(?:[eE][+-]?\d+)?$"
)
_DECIMAL_COMMA_RE = re.compile(r"^[+-]?" r"(?:(?:\d+(?:\.\d{3})*|),\d+)" r"(?:[eE][+-]?\d+)?$")


def _parse_numeric_value(value: object) -> Decimal | None:
    """Parse a supported numeric surface form into a ``Decimal`` (else ``None``)."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if _PURE_NUMERIC_RE.fullmatch(text):
        normalized = text.replace(",", "")
    elif _DECIMAL_COMMA_RE.fullmatch(text):
        normalized = text.replace(".", "").replace(",", ".")
    else:
        return None
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def numeric_values_equal(left: object, right: object) -> bool:
    """Whether two supported numeric surface forms denote the same value."""
    left_number = _parse_numeric_value(left)
    right_number = _parse_numeric_value(right)
    return left_number is not None and right_number is not None and left_number == right_number


def normalize_cell(value: object, *, casefold: bool = False) -> str:
    """Whitespace-normalize a cell string for value comparison.

    Strips leading/trailing whitespace and collapses internal runs of
    whitespace to a single space (CSV round-trips and LLM outputs differ only in
    incidental spacing surprisingly often). Optionally case-folds.
    """
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold() if casefold else text


def values_equal(
    proposed: object,
    gold: object,
    *,
    numeric: bool = True,
    casefold: bool = False,
) -> bool:
    """Judge a proposed corrected value against the gold value.

    * Exact string match after :func:`normalize_cell` (optionally case-folded).
    * If ``numeric`` (default), numerically-equal surface forms also match
      (``"1,000"`` == ``"1000"`` == ``"1000.0"``).
    """
    if normalize_cell(proposed, casefold=casefold) == normalize_cell(gold, casefold=casefold):
        return True
    if numeric and numeric_values_equal(proposed, gold):
        return True
    return False


def load_table_rows(path: str | Path) -> list[list[str]]:
    """Read a CSV as a list of rows (header + data), matching REIN's ``dtype=str``.

    Uses the stdlib ``csv`` reader (not pandas) so cell strings are byte-faithful
    to what the detectors and the benchmark saw.
    """
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def load_dirty_clean(
    ds_root: str | Path,
    ds: str,
    cache: dict[str, tuple[list[list[str]], list[list[str]]]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Return ``(dirty_rows, clean_rows)`` for a dataset (data rows only, cached).

    Both are 0-indexed by the benchmark ``(i, j)`` convention: ``rows[i][j]`` is
    the cell at data-row ``i`` (header excluded), column ``j``.
    """
    if ds in cache:
        return cache[ds]
    root = Path(ds_root) / ds
    dirty = load_table_rows(root / "dirty.csv")
    clean = load_table_rows(root / "clean.csv")
    dirty_data = dirty[1:] if dirty else []
    clean_data = clean[1:] if clean else []
    cache[ds] = (dirty_data, clean_data)
    return cache[ds]


def cell_value(rows: list[list[str]], i: int, j: int) -> str | None:
    """Safe ``rows[i][j]`` (``None`` if out of range)."""
    if 0 <= i < len(rows) and 0 <= j < len(rows[i]):
        return rows[i][j]
    return None
