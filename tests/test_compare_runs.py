"""Tests for the run-comparison logic powering the labeling UI's
``Compare runs`` panel.

Pins:
  * ``ERROR_TYPE_BUCKETS`` covers every error_type in the canonical
    13-type taxonomy (no error_type silently maps to ``unknown``).
  * ``HEURISTIC_BUCKETS`` lists every bucket that has at least one
    heuristic detector under ``src/heuristics/`` or
    ``src/core/row_duplication_detector.py``.
  * ``_index_cells`` preserves whole-row issues (``columns: []``)
    such as ``row_duplication``, slotting them under a sentinel
    ``(whole row)`` column instead of dropping them.
  * ``diff_runs`` produces a ``by_bucket`` row for every bucket
    either run emitted, including the new multi-row / multi-column
    / multi-table types.
  * The Bing-grounded ``_fold_col`` collapses the sentinel cleanly
    so the same whole-row diff round-trips through
    ``diff_cell_detail`` (the drawer endpoint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from experiments.compare_runs import (
    ERROR_TYPE_BUCKETS,
    HEURISTIC_BUCKETS,
    WHOLE_ROW_COL,
    _bucket,
    _fold_col,
    _index_cells,
    diff_cell_detail,
    diff_runs,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _issue(
    *,
    error_type: str,
    source: str = "x.csv",
    row: int = 0,
    columns: List[str] | None = None,
    affected_rows: List[int] | None = None,
    detector: str = "llm",
    observed: str = "obs",
    explanation: str = "why",
    suggested_fix: str | None = None,
    severity: str | None = None,
) -> Dict[str, Any]:
    """Build an issues.jsonl-shaped dict."""
    out: Dict[str, Any] = {
        "id": f"{source}-{row}-{error_type}",
        "source": source,
        "row": row,
        "columns": columns if columns is not None else [],
        "error_type": error_type,
        "detector": detector,
        "observed": observed,
        "explanation": explanation,
        "suggested_fix": suggested_fix,
        "severity": severity,
    }
    if affected_rows is not None:
        out["affected_rows"] = list(affected_rows)
    return out


def _write_run(tmp_path: Path, run_id: str, issues: List[Dict[str, Any]]) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "issues.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in issues) + "\n",
        encoding="utf-8",
    )
    return run_dir


# ---------------------------------------------------------------------------
# Bucket map coverage
# ---------------------------------------------------------------------------


# Source of truth: the canonical 13-type taxonomy from
# ``prompts/error_types.yaml``. If a new type is added there, it must
# also be added to ``ERROR_TYPE_BUCKETS`` (this test enforces it).
_CANONICAL_ERROR_TYPES = frozenset({
    # single-column
    "typo",
    "text_inconsistency",
    "casing_inconsistency",
    "pattern_inconsistency",
    "punctuation_inconsistency",
    "number_format_inconsistency",
    "semantic_incompatible",
    "numeric_outlier",
    # multi-column
    "fd_violation",
    "row_calculation_inconsistency",
    "string_calculation_inconsistency",
    # multi-row
    "row_duplication",
    # multi-table
    "inter_table_inconsistency",
})


def test_bucket_map_covers_every_canonical_error_type():
    missing = _CANONICAL_ERROR_TYPES - set(ERROR_TYPE_BUCKETS)
    assert not missing, (
        "ERROR_TYPE_BUCKETS is missing canonical error_types: "
        f"{sorted(missing)}. Add identity mappings (or a deliberate "
        "fold) so the compare-runs panel can surface these types."
    )


def test_bucket_map_no_unknown_fallback_for_canonical_types():
    """Every canonical error_type maps to a non-empty, non-``unknown``
    bucket name."""
    for et in _CANONICAL_ERROR_TYPES:
        b = _bucket(et)
        assert b
        assert b != "unknown"


def test_bucket_map_known_folds_are_preserved():
    """``number_format_inconsistency`` is deliberately folded into
    ``pattern_inconsistency`` because the heuristics treat them as
    one class. Keep that fold pinned so a refactor doesn't silently
    split them."""
    assert _bucket("number_format_inconsistency") == "pattern_inconsistency"


def test_unknown_error_type_passes_through_as_its_own_bucket():
    """Anything not in the map keeps its raw name (so a new
    detector's error_type still shows up as a bucket the operator
    can see, even before someone updates the map)."""
    assert _bucket("someones_new_type") == "someones_new_type"


def test_empty_error_type_falls_back_to_unknown():
    assert _bucket("") == "unknown"


def test_heuristic_buckets_cover_each_shipped_heuristic():
    """Every heuristic detector emits one of these bucket names. Keep
    the set in sync with ``src/heuristics/`` + the row-duplication
    detector under ``src/core/``."""
    expected_subset = {
        # src/heuristics/
        "casing_inconsistency",         # casing.py
        "fd_violation",                 # fd_violation.py
        "typo",                         # key_collision.py
        "numeric_outlier",              # numeric_outlier.py
        "pattern_inconsistency",        # pattern_minority.py
        "punctuation_inconsistency",    # unicode_norm.py
        "text_inconsistency",           # whitespace.py
        # src/core/row_duplication_detector.py
        "row_duplication",
    }
    missing = expected_subset - HEURISTIC_BUCKETS
    assert not missing, (
        "HEURISTIC_BUCKETS missing entries for shipped detectors: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# _index_cells: whole-row issues
# ---------------------------------------------------------------------------


def test_index_cells_keeps_row_duplication_issues():
    """Row-level issues (``columns: []``) used to be silently
    dropped. They must now slot under a sentinel column so the
    bucket overlap and diff list can surface them."""
    issues = [
        _issue(
            error_type="row_duplication",
            row=27,
            columns=[],
            affected_rows=[27, 29],
            observed="2 rows match exactly",
        ),
        _issue(
            error_type="row_duplication",
            row=29,
            columns=[],
            affected_rows=[27, 29],
            observed="2 rows match exactly",
        ),
    ]
    idx, display = _index_cells(issues)
    folded = _fold_col(WHOLE_ROW_COL)
    assert ("x.csv", 27, folded) in idx
    assert ("x.csv", 29, folded) in idx
    assert "row_duplication" in idx[("x.csv", 27, folded)]
    # Display map round-trips back to the human-readable label.
    assert display[("x.csv", 27, folded)] == WHOLE_ROW_COL


def test_index_cells_still_skips_issues_without_a_row():
    """Issues lacking a numeric ``row`` (corpus-level findings) are
    still skipped - the comparison is cell-keyed."""
    issues = [_issue(error_type="row_duplication", columns=[])]
    issues[0]["row"] = None  # type: ignore[assignment]
    idx, _ = _index_cells(issues)
    assert idx == {}


def test_index_cells_per_column_still_works_for_normal_issues():
    """Normal per-cell issues continue to get one entry per column.
    This pins the existing behavior so the whole-row branch doesn't
    accidentally swallow them."""
    issues = [
        _issue(error_type="typo", row=5, columns=["Vendor"]),
        _issue(error_type="fd_violation", row=5,
               columns=["LICENSE_ID", "BUSINESS NAME"]),
    ]
    idx, _ = _index_cells(issues)
    assert ("x.csv", 5, _fold_col("Vendor")) in idx
    assert ("x.csv", 5, _fold_col("LICENSE_ID")) in idx
    assert ("x.csv", 5, _fold_col("BUSINESS NAME")) in idx
    # No spurious whole-row entry when columns are populated.
    assert ("x.csv", 5, _fold_col(WHOLE_ROW_COL)) not in idx


# ---------------------------------------------------------------------------
# diff_runs end-to-end: row_duplication appears in by_bucket + diff_cells
# ---------------------------------------------------------------------------


def test_diff_runs_surfaces_row_duplication_disagreement(tmp_path: Path):
    run_a = _write_run(tmp_path, "run-a", [
        _issue(error_type="row_duplication", row=10, columns=[],
               affected_rows=[10, 14], observed="duplicate group A"),
        _issue(error_type="row_duplication", row=14, columns=[],
               affected_rows=[10, 14], observed="duplicate group A"),
        # Run A also caught a typo on a real cell.
        _issue(error_type="typo", row=2, columns=["Vendor"],
               observed="AcmeC"),
    ])
    run_b = _write_run(tmp_path, "run-b", [
        # Run B agreed on the typo.
        _issue(error_type="typo", row=2, columns=["Vendor"],
               observed="AcmeC"),
        # ... but missed the duplicate group entirely.
    ])

    diff = diff_runs(run_a, run_b)

    # Per-bucket overlap surfaces row_duplication AND typo.
    by_bucket = {r["bucket"]: r for r in diff["by_bucket"]}
    assert "row_duplication" in by_bucket
    assert "typo" in by_bucket
    rd = by_bucket["row_duplication"]
    assert rd["a_hits"] == 2
    assert rd["b_hits"] == 0
    assert rd["a_only"] == 2
    assert rd["b_only"] == 0
    typo = by_bucket["typo"]
    # typo agreed on row 2 -> intersection 1, side="both" so it is
    # NOT in diff_cells (include_both=False).
    assert typo["intersection"] == 1
    assert typo["a_only"] == 0
    assert typo["b_only"] == 0

    # Diff cell list shows the two duplicate rows as a_only with
    # column ``(whole row)``.
    diff_cols = sorted(
        (c["row"], c["column"], c["side"])
        for c in diff["diff_cells"]
    )
    assert (10, WHOLE_ROW_COL, "a_only") in diff_cols
    assert (14, WHOLE_ROW_COL, "a_only") in diff_cols
    # No row_duplication entry from run B (it didn't flag it).
    assert all(c["side"] == "a_only" for c in diff["diff_cells"])
    assert diff["n_diff_cells"] == 2


def test_diff_runs_surfaces_each_new_taxonomy_type(tmp_path: Path):
    """Every error_type from the new taxonomy ends up in
    ``by_bucket`` when at least one run emits it."""
    new_types_per_cell = [
        ("semantic_incompatible", ["Phone"]),
        ("row_calculation_inconsistency", ["Total", "Item1", "Item2"]),
        ("string_calculation_inconsistency", ["FullName", "First", "Last"]),
        ("inter_table_inconsistency", ["Vendor"]),
    ]
    issues_a: List[Dict[str, Any]] = []
    for et, cols in new_types_per_cell:
        issues_a.append(_issue(error_type=et, row=0, columns=cols))
    # Whole-row issue too.
    issues_a.append(
        _issue(error_type="row_duplication", row=0, columns=[],
               affected_rows=[0, 1])
    )

    run_a = _write_run(tmp_path, "run-a", issues_a)
    run_b = _write_run(tmp_path, "run-b", [])  # empty

    diff = diff_runs(run_a, run_b)
    bucket_names = {r["bucket"] for r in diff["by_bucket"]}
    expected = {et for et, _ in new_types_per_cell} | {"row_duplication"}
    assert expected.issubset(bucket_names), (
        f"missing buckets: {sorted(expected - bucket_names)} - "
        "either ERROR_TYPE_BUCKETS folded them away unexpectedly or "
        "_index_cells dropped them."
    )


def test_diff_cell_detail_handles_whole_row_column(tmp_path: Path):
    """The drawer endpoint must look up whole-row issues using the
    same sentinel column the diff list emits."""
    run_a = _write_run(tmp_path, "run-a", [
        _issue(error_type="row_duplication", row=10, columns=[],
               affected_rows=[10, 14], observed="dup"),
    ])
    run_b = _write_run(tmp_path, "run-b", [])

    detail = diff_cell_detail(run_a, run_b,
                              source="x.csv", row=10, column=WHOLE_ROW_COL)
    assert len(detail["a_issues"]) == 1
    assert detail["a_issues"][0]["bucket"] == "row_duplication"
    assert detail["b_issues"] == []
