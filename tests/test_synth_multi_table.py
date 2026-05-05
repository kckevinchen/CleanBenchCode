"""Tests for cross-sheet (workbook-scope) synthetic injectors."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from src.synth import WORKBOOK_INJECTOR_REGISTRY
from src.synth.multi_table import (
    CrossSheetAliasSwap,
    CrossSheetAttributeConflict,
    CrossSheetOrphanInject,
    CrossSheetRenameHeader,
    CrossSheetUnitFlip,
    _shared_columns,
)


def _two_sheet_workbook() -> dict:
    customers = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "country": ["United States", "United Kingdom", "China", "Germany"],
        "tier": ["gold", "silver", "gold", "bronze"],
    })
    orders = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C5"],
        "country": ["United States", "United Kingdom", "China", "France"],
        "amount_usd": [100.0, 200.0, 300.0, 400.0],
    })
    return {"customers.csv": customers, "orders.csv": orders}


def _numeric_two_sheet() -> dict:
    a = pd.DataFrame({"id": [1, 2, 3], "weight": [10.0, 20.0, 30.0]})
    b = pd.DataFrame({"id": [1, 2, 3], "weight": [10.0, 20.0, 30.0]})
    return {"a.csv": a, "b.csv": b}


class TestRegistry:
    def test_all_injectors_registered(self) -> None:
        for name in (
            "cross_sheet_alias_swap",
            "cross_sheet_unit_flip",
            "cross_sheet_orphan_inject",
            "cross_sheet_attribute_conflict",
            "cross_sheet_rename_header",
        ):
            assert name in WORKBOOK_INJECTOR_REGISTRY


class TestSharedColumns:
    def test_finds_shared_columns(self) -> None:
        triples = _shared_columns(_two_sheet_workbook())
        cols = {c for _, _, c in triples}
        assert {"customer_id", "country"} <= cols

    def test_no_shared_when_disjoint(self) -> None:
        frames = {
            "x.csv": pd.DataFrame({"a": [1]}),
            "y.csv": pd.DataFrame({"b": [1]}),
        }
        assert _shared_columns(frames) == []


class TestCrossSheetAliasSwap:
    def test_rewrites_to_alias(self) -> None:
        frames = _two_sheet_workbook()
        result = CrossSheetAliasSwap().try_inject(frames, rng=random.Random(0))
        assert result is not None
        new_frames, records = result
        assert len(records) == 1
        rec = records[0]
        assert rec.error_type == "inter_table_inconsistency"
        # The injected value should be one of the known aliases.
        assert rec.injected_value in {
            "USA", "U.S.", "America", "UK", "PRC", "Deutschland",
        }
        # Original should be a known canonical.
        assert rec.original_value.lower() in {
            "united states", "united kingdom", "china", "germany",
        }

    def test_returns_none_when_no_shared(self) -> None:
        frames = {
            "x.csv": pd.DataFrame({"a": ["United States"]}),
            "y.csv": pd.DataFrame({"b": ["United States"]}),
        }
        assert CrossSheetAliasSwap().try_inject(frames, rng=random.Random(0)) is None


class TestCrossSheetUnitFlip:
    def test_multiplies_by_ratio(self) -> None:
        frames = _numeric_two_sheet()
        result = CrossSheetUnitFlip().try_inject(frames, rng=random.Random(1))
        assert result is not None
        new_frames, records = result
        assert len(records) == 3  # all rows scaled
        ratio = records[0].params["ratio"]
        for rec in records:
            assert pytest.approx(float(rec.injected_value)) == float(rec.original_value) * ratio

    def test_skips_non_numeric(self) -> None:
        frames = {
            "a.csv": pd.DataFrame({"x": ["foo", "bar"]}),
            "b.csv": pd.DataFrame({"x": ["baz", "qux"]}),
        }
        assert CrossSheetUnitFlip().try_inject(frames, rng=random.Random(0)) is None


class TestCrossSheetOrphanInject:
    def test_inserts_orphan_in_child(self) -> None:
        frames = _two_sheet_workbook()
        result = CrossSheetOrphanInject().try_inject(frames, rng=random.Random(2))
        assert result is not None
        new_frames, records = result
        assert len(records) == 1
        rec = records[0]
        assert "__SYNTH_ORPHAN_" in rec.injected_value
        # The orphan should now appear in exactly one sheet and not the other.
        col = rec.column
        in_orphan = new_frames[rec.source][col].astype(str).str.contains("__SYNTH_ORPHAN_").any()
        parent = rec.params["parent_sheet"]
        in_parent = new_frames[parent][col].astype(str).str.contains("__SYNTH_ORPHAN_").any()
        assert in_orphan and not in_parent


class TestCrossSheetAttributeConflict:
    def test_flips_attribute_for_shared_key(self) -> None:
        frames = _two_sheet_workbook()
        result = CrossSheetAttributeConflict().try_inject(frames, rng=random.Random(3))
        assert result is not None
        new_frames, records = result
        assert len(records) == 1
        rec = records[0]
        assert rec.injected_value.endswith("_CONFLICT")
        assert "key_column" in rec.params and "key_value" in rec.params


class TestCrossSheetRenameHeader:
    def test_renames_one_side(self) -> None:
        frames = _two_sheet_workbook()
        result = CrossSheetRenameHeader().try_inject(frames, rng=random.Random(4))
        assert result is not None
        new_frames, records = result
        assert len(records) == 1
        rec = records[0]
        # Renamed column should appear in injected sheet but not under original name.
        assert rec.injected_value not in frames[rec.source].columns
        assert rec.injected_value in new_frames[rec.source].columns
        assert rec.original_value not in new_frames[rec.source].columns
        # Sibling untouched.
        sibling = rec.params["sibling_sheet"]
        assert rec.original_value in new_frames[sibling].columns
