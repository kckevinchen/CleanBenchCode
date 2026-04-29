"""Schema-deduplication within grouped (multi-table) workbooks.

When ``layout=grouped`` and a single workbook contains two CSVs whose
post-clean column headers are identical, only the first occurrence is
kept. The duplicate is recorded in the manifest and counted under
``skip_reason_counts['duplicate_schema_within_group']``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_prep.curate import curate


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_rows(n: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({
        "name":  [f"{prefix}-name-{i}" for i in range(n)],
        "city":  [f"{prefix}-city-{i}" for i in range(n)],
        "score": list(range(n)),
    })


@pytest.fixture
def grouped_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "corpus"
    # workbook_a: two tables sharing the same schema, one distinct
    _write_csv(src / "workbook_a" / "a1.csv", _make_rows(20, "a1"))
    _write_csv(src / "workbook_a" / "a2.csv", _make_rows(25, "a2"))  # dup of a1
    _write_csv(src / "workbook_a" / "a3.csv", pd.DataFrame({
        "product": [f"p{i}" for i in range(15)],
        "price":   [i * 1.5 for i in range(15)],
    }))
    # workbook_b: two tables, same schema. After dedup only one remains.
    _write_csv(src / "workbook_b" / "b1.csv", _make_rows(12, "b1"))
    _write_csv(src / "workbook_b" / "b2.csv", _make_rows(14, "b2"))
    return src


def _curate_kwargs(out: Path) -> dict:
    return dict(
        output_dir=out,
        num_targets=10,
        min_rows=5, max_rows=200,
        min_columns=2, max_columns=10,
        min_tables=1, max_tables=10,
        seed=0,
        language_check="off",
        max_non_latin_chars=10_000,
        max_empty_ratio=1.0,
        max_cell_chars=None, max_cell_words=None,
        max_oversized_row_ratio=1.0,
    )


def test_dedup_drops_duplicate_schema_within_group(grouped_corpus, tmp_path):
    out = tmp_path / "out"
    manifest = curate(
        source_dir=grouped_corpus,
        dedupe_schema_within_group=True,
        **_curate_kwargs(out),
    )

    assert manifest["layout"] == "grouped"
    by_id = {g["id"]: g for g in manifest["selected"]}

    # workbook_a kept a1 + a3 (a2 dropped as duplicate of a1)
    wa = by_id["workbook_a"]
    kept_a = sorted(t["filename"] for t in wa["tables"])
    assert kept_a == ["a1.csv", "a3.csv"]
    assert wa["dropped_duplicate_schema"] == [
        {"filename": "a2.csv", "duplicate_of": "a1.csv"}
    ]
    # only the kept files were written
    assert (out / "workbook_a" / "a1.csv").exists()
    assert (out / "workbook_a" / "a3.csv").exists()
    assert not (out / "workbook_a" / "a2.csv").exists()

    # workbook_b kept just b1 (b2 dropped)
    wb = by_id["workbook_b"]
    kept_b = sorted(t["filename"] for t in wb["tables"])
    assert kept_b == ["b1.csv"]
    assert wb["dropped_duplicate_schema"] == [
        {"filename": "b2.csv", "duplicate_of": "b1.csv"}
    ]
    assert not (out / "workbook_b" / "b2.csv").exists()

    # skip_reason_counts surfaces the dedup activity
    assert manifest["skip_reason_counts"].get("duplicate_schema_within_group") == 2
    assert manifest["filters"]["dedupe_schema_within_group"] is True


def test_dedup_disabled_keeps_all_duplicates(grouped_corpus, tmp_path):
    out = tmp_path / "out"
    manifest = curate(
        source_dir=grouped_corpus,
        dedupe_schema_within_group=False,
        **_curate_kwargs(out),
    )

    by_id = {g["id"]: g for g in manifest["selected"]}
    kept_a = sorted(t["filename"] for t in by_id["workbook_a"]["tables"])
    assert kept_a == ["a1.csv", "a2.csv", "a3.csv"]
    kept_b = sorted(t["filename"] for t in by_id["workbook_b"]["tables"])
    assert kept_b == ["b1.csv", "b2.csv"]
    assert by_id["workbook_a"]["dropped_duplicate_schema"] == []
    assert by_id["workbook_b"]["dropped_duplicate_schema"] == []
    assert "duplicate_schema_within_group" not in manifest["skip_reason_counts"]
    assert manifest["filters"]["dedupe_schema_within_group"] is False


def test_dedup_below_min_tables_rejects_workbook(tmp_path):
    src = tmp_path / "corpus"
    # Both tables in workbook_x share schema; with min_tables=2 the
    # workbook becomes ineligible after dedup collapses it to 1 table.
    _write_csv(src / "workbook_x" / "t1.csv", _make_rows(10, "t1"))
    _write_csv(src / "workbook_x" / "t2.csv", _make_rows(10, "t2"))

    out = tmp_path / "out"
    kwargs = _curate_kwargs(out)
    kwargs["min_tables"] = 2
    manifest = curate(
        source_dir=src,
        dedupe_schema_within_group=True,
        **kwargs,
    )

    assert manifest["num_targets_selected"] == 0
    assert manifest["skip_reason_counts"].get("too_few_tables_after_dedup") == 1
    assert not (out / "workbook_x").exists() or not any((out / "workbook_x").iterdir())


def test_dedup_noop_in_flat_layout(tmp_path):
    src = tmp_path / "flat"
    src.mkdir()
    # Two flat CSVs with identical schemas — flat layout treats each as
    # its own group, so cross-group duplicates are NOT deduped.
    _write_csv(src / "f1.csv", _make_rows(8, "f1"))
    _write_csv(src / "f2.csv", _make_rows(8, "f2"))

    out = tmp_path / "out"
    manifest = curate(
        source_dir=src,
        dedupe_schema_within_group=True,
        **_curate_kwargs(out),
    )

    assert manifest["layout"] == "flat"
    kept = sorted(t["filename"] for g in manifest["selected"] for t in g["tables"])
    assert kept == ["f1.csv", "f2.csv"]
    assert "duplicate_schema_within_group" not in manifest["skip_reason_counts"]
    assert manifest["filters"]["dedupe_schema_within_group"] is None
