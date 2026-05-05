"""Tests for `src.core.table_digest`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.data_loader import COLUMN_MODE_ALL, CsvChunk
from src.core.error_taxonomy import load_taxonomy
from src.core.table_digest import (
    ColumnDigest,
    TableDigest,
    build_table_digest,
)
from tests.test_error_taxonomy import _good_taxonomy_dict, _write_yaml


def _df_with_total() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Region": ["North", "South", "East", "West", "Total"],
            "Sales": ["10", "20", "30", "40", "100"],
            "Code": ["N1", "S1", "E1", "W1", ""],
        }
    )


class TestBuildTableDigest:
    def test_basic_shape_and_columns(self) -> None:
        df = _df_with_total()
        d = build_table_digest(df, source="x.csv")
        assert d.n_rows == 5
        assert d.n_columns == 3
        assert d.headers == ["Region", "Sales", "Code"]
        sales = d.columns["Sales"]
        assert sales.dtype_kind == "numeric"
        assert sales.numeric_summary is not None
        assert sales.numeric_summary["min"] == 10.0
        assert sales.numeric_summary["max"] == 100.0
        assert d.columns["Region"].dtype_kind == "text"

    def test_total_row_detected(self) -> None:
        d = build_table_digest(_df_with_total(), source="x.csv")
        assert 4 in d.total_row_indices

    def test_identity_candidate_detected(self) -> None:
        df = pd.DataFrame(
            {
                "id": [f"E{i:03d}" for i in range(20)],
                "dept": ["A"] * 10 + ["B"] * 10,
            }
        )
        d = build_table_digest(df, source="x.csv")
        assert "id" in d.candidate_identity_columns
        assert "dept" not in d.candidate_identity_columns

    def test_candidate_fd_detected(self) -> None:
        df = pd.DataFrame(
            {
                "Zip": ["94103", "94103", "94103", "10001", "10001", "10001"],
                "State": ["CA", "CA", "CA", "NY", "NY", "NY"],
                "Note": list("abcdef"),
            }
        )
        d = build_table_digest(df, source="x.csv")
        keys = {(fd["lhs"], fd["rhs"]) for fd in d.candidate_fds}
        assert ("Zip", "State") in keys

    def test_to_prompt_block_smoke(self) -> None:
        d = build_table_digest(_df_with_total(), source="x.csv")
        block = d.to_prompt_block()
        assert "TABLE OVERVIEW" in block
        assert "x.csv" in block
        assert "Region" in block
        assert "Sales" in block
        assert "4" in block

    def test_empty_table_returns_empty_block(self) -> None:
        df = pd.DataFrame({"a": [], "b": []})
        d = build_table_digest(df, source="x.csv")
        assert d.n_rows == 0
        out = d.to_prompt_block()
        assert "empty table" in out


class TestRenderUserPromptWithDigest:
    def _make_template_with_overview(self) -> dict:
        d = _good_taxonomy_dict()
        d["base"]["user_template"] = (
            "TYPE={type_key} OVERVIEW={table_overview} ROWS:\n{rows}"
        )
        return d

    def test_digest_substituted_when_supplied(self, tmp_path: Path) -> None:
        tax = load_taxonomy(_write_yaml(tmp_path, self._make_template_with_overview()))
        et = tax.get("typo")
        df = pd.DataFrame({"name": ["alice", "bob"]})
        chunk = CsvChunk(
            source=Path("a.csv"), start_row=0, end_row=2,
            headers=["name"], frame=df, column_mode=COLUMN_MODE_ALL,
        )
        digest = build_table_digest(df, source="a.csv")
        out = tax.render_user_prompt(et, chunk, digest=digest)
        assert "OVERVIEW=TABLE OVERVIEW" in out
        assert "name" in out

    def test_no_digest_uses_placeholder(self, tmp_path: Path) -> None:
        tax = load_taxonomy(_write_yaml(tmp_path, self._make_template_with_overview()))
        et = tax.get("typo")
        df = pd.DataFrame({"name": ["alice"]})
        chunk = CsvChunk(
            source=Path("a.csv"), start_row=0, end_row=1,
            headers=["name"], frame=df, column_mode=COLUMN_MODE_ALL,
        )
        out = tax.render_user_prompt(et, chunk)
        assert "OVERVIEW=(table overview unavailable" in out

    def test_template_without_overview_placeholder_still_works(
        self, tmp_path: Path
    ) -> None:
        tax = load_taxonomy(_write_yaml(tmp_path, _good_taxonomy_dict()))
        et = tax.get("typo")
        df = pd.DataFrame({"name": ["alice"]})
        chunk = CsvChunk(
            source=Path("a.csv"), start_row=0, end_row=1,
            headers=["name"], frame=df, column_mode=COLUMN_MODE_ALL,
        )
        out = tax.render_user_prompt(et, chunk, digest=build_table_digest(df, source="a"))
        assert "TYPE=typo" in out


# ---------------------------------------------------------------------------
# Richer evidence: extended numeric / length / rare-value summaries
# ---------------------------------------------------------------------------


class TestRicherEvidence:
    def test_numeric_summary_has_quartiles_and_flags(self) -> None:
        df = pd.DataFrame({"v": [str(x) for x in [-2, 0, 0, 1, 2, 3, 100]]})
        d = build_table_digest(df, source="t")
        ns = d.columns["v"].numeric_summary
        assert ns is not None
        assert {"min", "p25", "median", "p75", "max", "mean",
                "n_zeros", "n_negatives"} <= set(ns)
        assert ns["n_zeros"] == 2
        assert ns["n_negatives"] == 1

    def test_text_columns_have_length_summary(self) -> None:
        df = pd.DataFrame({"s": ["a", "bb", "ccc", "dddd"]})
        d = build_table_digest(df, source="t")
        ls = d.columns["s"].length_summary
        assert ls is not None
        assert ls == {"min": 1, "median": 2, "max": 4}

    def test_rare_values_picked_from_long_tail(self) -> None:
        # Fat head ("A" x4) and a long tail of singletons.
        df = pd.DataFrame({"c": ["A"] * 4 + ["x", "y", "z"]})
        d = build_table_digest(df, source="t")
        col = d.columns["c"]
        assert col.rare_values, "expected long-tail values"
        assert all(v in {"x", "y", "z"} for v in col.rare_values)

    def test_fd_violation_examples_surfaced(self) -> None:
        # Zip -> State holds for 5/6 rows. Row 5 violates.
        df = pd.DataFrame(
            {
                "Zip": ["10001", "10001", "10001", "94016", "94016", "94016"],
                "State": ["NY", "NY", "NY", "CA", "CA", "WA"],
            }
        )
        d = build_table_digest(df, source="t",
                               fd_min_support=0.5, fd_min_group_size=2,
                               fd_rule_modal_share=0.5)
        fds = [fd for fd in d.candidate_fds if fd["lhs"] == "Zip" and fd["rhs"] == "State"]
        assert fds, "expected Zip -> State candidate FD"
        fd = fds[0]
        assert fd["support"] < 1.0
        assert fd["violation_examples"], "expected violation row example"
        # Violator should be index 5 (Zip=94016 -> State=WA, modal=CA).
        ex = fd["violation_examples"][0]
        assert ex["row"] == 5
        assert ex["lhs_value"] == "94016"
        assert ex["rhs_value"] == "WA"
        assert ex["expected_rhs"] == "CA"

    def test_to_prompt_block_renders_extended_fields(self) -> None:
        df = pd.DataFrame(
            {
                "v": [str(x) for x in [-1, 0, 1, 2, 3, 4, 100]],
                "s": ["short", "medium-len", "and a much longer one!", "x", "y", "z", "w"],
            }
        )
        d = build_table_digest(df, source="t")
        out = d.to_prompt_block()
        assert "p25=" in out and "p75=" in out
        assert "negatives=" in out
        assert "cell length" in out

    def test_to_dict_persists_new_fields(self) -> None:
        df = pd.DataFrame({"v": [str(x) for x in [0, 0, 1, 2]]})
        d = build_table_digest(df, source="t")
        blob = d.to_dict()
        col = blob["columns"]["v"]
        assert "n_zeros" in col["numeric_summary"]
        assert "rare_values" in col


# ---------------------------------------------------------------------------
# Outlier row picking
# ---------------------------------------------------------------------------


class TestPickOutlierIndices:
    def test_numeric_outlier_picked(self) -> None:
        from src.core.relations import pick_outlier_indices
        df = pd.DataFrame({"v": [str(x) for x in [10, 11, 9, 10, 12, 1000]]})
        d = build_table_digest(df, source="t")
        picks = pick_outlier_indices(df, d, n=3)
        assert 5 in picks  # row 5 is the obvious extreme

    def test_total_rows_excluded(self) -> None:
        from src.core.relations import pick_outlier_indices
        df = pd.DataFrame({
            "lab": ["a", "b", "c", "d", "Total"],
            "v": ["1", "2", "3", "4", "10000"],
        })
        d = build_table_digest(df, source="t")
        picks = pick_outlier_indices(df, d, n=3)
        # Index 4 is "Total" - must NOT be picked even though it has
        # the largest numeric deviation.
        assert 4 not in picks

    def test_rare_value_row_picked_when_no_numeric_signal(self) -> None:
        from src.core.relations import pick_outlier_indices
        df = pd.DataFrame({"c": ["A"] * 4 + ["unique-x"]})
        d = build_table_digest(df, source="t")
        picks = pick_outlier_indices(df, d, n=3)
        assert 4 in picks


class TestSampleRowsWithOutliers:
    def test_outlier_section_appended(self) -> None:
        from src.core.relations import sample_rows_for_prompt
        df = pd.DataFrame({"x": list(range(10))})
        out = sample_rows_for_prompt(df, n_samples=3, outlier_indices=[7])
        assert "stratified" in out
        assert "outliers" in out

    def test_outlier_dedup_against_stratified(self) -> None:
        from src.core.relations import sample_rows_for_prompt
        df = pd.DataFrame({"x": list(range(10))})
        # Index 0 is always picked by stratified -> must not appear in
        # the outlier section.
        out = sample_rows_for_prompt(df, n_samples=3, outlier_indices=[0])
        assert "outliers" not in out
