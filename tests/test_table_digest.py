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
