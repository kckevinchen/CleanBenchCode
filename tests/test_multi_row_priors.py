"""Tests for ``src.core.multi_row_priors``."""

from __future__ import annotations

import pandas as pd

from src.core.multi_row_priors import (
    DuplicateSubset,
    IdentityConflict,
    MultiRowPriors,
    TotalRowBlock,
    compute_priors,
    enumerate_collision_groups,
    find_duplicate_subsets,
    find_duplicate_subsets_normalized,
    find_identity_conflicts,
    find_total_row_blocks,
    normalize_for_dedup,
    render_group_blocks,
    render_priors_block,
)
from src.core.table_digest import build_table_digest


# ---------------------------------------------------------------------------
# find_total_row_blocks
# ---------------------------------------------------------------------------


class TestFindTotalRowBlocks:
    def test_finds_total_row_with_label(self) -> None:
        df = pd.DataFrame({
            "Region": ["East", "West", "North", "Total"],
            "Sales":  [100, 200, 150, 450],
        })
        d = build_table_digest(df, source="t.csv")
        blocks = find_total_row_blocks(df, digest=d)
        assert len(blocks) == 1
        b = blocks[0]
        assert b.total_row == 3
        assert "total" in b.label.lower()
        # The three preceding rows are detail rows.
        assert b.detail_rows == (0, 1, 2)

    def test_falls_back_to_regex_when_digest_missing(self) -> None:
        df = pd.DataFrame({
            "Period":   ["Q1", "Q2", "Q3", "Grand Total"],
            "Revenue":  [10, 20, 30, 60],
        })
        blocks = find_total_row_blocks(df, digest=None)
        assert len(blocks) == 1
        assert blocks[0].total_row == 3

    def test_empty_frame(self) -> None:
        assert find_total_row_blocks(pd.DataFrame(), digest=None) == []

    def test_no_total_rows_returns_empty(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        d = build_table_digest(df, source="t.csv")
        assert find_total_row_blocks(df, digest=d) == []


# ---------------------------------------------------------------------------
# find_duplicate_subsets
# ---------------------------------------------------------------------------


class TestFindDuplicateSubsets:
    def test_single_column_duplicates(self) -> None:
        df = pd.DataFrame({
            "Email":    ["a@x", "b@x", "a@x", "c@x", "a@x"],
            "Name":     ["Ana", "Bob", "Ana", "Cy",  "Ana"],
            "Age":      [30, 25, 30, 22, 31],
        })
        d = build_table_digest(df, source="t.csv")
        subs = find_duplicate_subsets(df, digest=d)
        # Email shows duplicates (3 rows for "a@x")
        emails = [s for s in subs if s.columns == ("Email",)]
        assert len(emails) == 1
        s = emails[0]
        assert s.n_duplicate_rows == 3
        assert s.n_groups == 1
        # Sample group has the three rows where Email == "a@x"
        assert any(set(g) == {0, 2, 4} for g in s.sample_groups)

    def test_no_duplicates(self) -> None:
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        d = build_table_digest(df, source="t.csv")
        assert find_duplicate_subsets(df, digest=d) == []

    def test_skips_float_columns(self) -> None:
        # Floats should not be considered candidate identity keys.
        df = pd.DataFrame({"x": [1.5, 1.5, 2.5, 2.5]})
        d = build_table_digest(df, source="t.csv")
        # No object/int candidates -> no duplicates surfaced.
        assert find_duplicate_subsets(df, digest=d) == []


# ---------------------------------------------------------------------------
# find_identity_conflicts
# ---------------------------------------------------------------------------


class TestFindIdentityConflicts:
    def test_simple_conflict(self) -> None:
        df = pd.DataFrame({
            "EmployeeID": ["E1", "E2", "E1", "E3"],
            "DOB":        ["1990-01-01", "1985-05-05", "1990-06-03", "1992-08-12"],
            "Address":    ["12 Elm", "34 Oak", "12 Elm",   "78 Pine"],
        })
        d = build_table_digest(df, source="t.csv")
        conflicts = find_identity_conflicts(df, digest=d)
        # EmployeeID has groups (E1: rows 0,2). DOB conflicts (1990-01-01
        # vs 1990-06-03) inside that group.
        assert len(conflicts) >= 1
        emp_conflicts = [c for c in conflicts if c.key_columns == ("EmployeeID",)]
        assert len(emp_conflicts) == 1
        c = emp_conflicts[0]
        assert "DOB" in c.descriptive_columns
        # Sample group should contain rows 0 and 2.
        assert any(set(g) == {0, 2} for g in c.sample_groups)

    def test_no_conflict_when_groups_agree(self) -> None:
        df = pd.DataFrame({
            "EmployeeID": ["E1", "E1", "E2", "E2"],
            "DOB":        ["1990", "1990", "1985", "1985"],
            "Address":    ["A", "A", "B", "B"],
        })
        d = build_table_digest(df, source="t.csv")
        # Groups exist but every group agrees -> no conflict.
        conflicts = find_identity_conflicts(df, digest=d)
        assert conflicts == []

    def test_handles_singleton_groups(self) -> None:
        df = pd.DataFrame({
            "ID":  ["X1", "X2", "X3"],
            "Val": ["a", "b", "c"],
        })
        d = build_table_digest(df, source="t.csv")
        # Every group size is 1 -> nothing to conflict on.
        assert find_identity_conflicts(df, digest=d) == []


# ---------------------------------------------------------------------------
# compute_priors top-level
# ---------------------------------------------------------------------------


class TestComputePriors:
    def test_aggregation_type_no_longer_dispatches(self) -> None:
        # column_aggregation_inconsistency was removed from the active
        # taxonomy; compute_priors now treats it as an unknown type and
        # returns empty priors.
        df = pd.DataFrame({
            "Region": ["East", "West", "Total"],
            "Sales":  [100, 200, 300],
        })
        d = build_table_digest(df, source="t.csv")
        priors = compute_priors(
            df=df, digest=d,
            error_type="column_aggregation_inconsistency", source="t.csv",
        )
        assert priors.is_empty()

    def test_duplication_type_returns_dup_subsets_only(self) -> None:
        df = pd.DataFrame({
            "Email": ["a", "b", "a", "c"],
            "Name":  ["Ana", "Bob", "Ana", "Cy"],
        })
        d = build_table_digest(df, source="t.csv")
        priors = compute_priors(
            df=df, digest=d, error_type="row_duplication", source="t.csv",
        )
        assert priors.duplicate_subsets
        assert not priors.total_row_blocks
        assert not priors.identity_conflicts

    def test_cross_record_type_no_longer_dispatches(self) -> None:
        # cross_record_inconsistency was removed from the active taxonomy.
        df = pd.DataFrame({
            "EmpID": ["E1", "E1", "E2"],
            "DOB":   ["1990", "1991", "1985"],
        })
        d = build_table_digest(df, source="t.csv")
        priors = compute_priors(
            df=df, digest=d,
            error_type="cross_record_inconsistency", source="t.csv",
        )
        assert priors.is_empty()

    def test_unknown_error_type_returns_empty_priors(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        d = build_table_digest(df, source="t.csv")
        priors = compute_priors(
            df=df, digest=d, error_type="something_else", source="t.csv",
        )
        assert priors.is_empty()


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class TestRenderers:
    def test_render_priors_block_when_empty_says_none(self) -> None:
        priors = MultiRowPriors(
            error_type="column_aggregation_inconsistency", source="t.csv",
        )
        out = render_priors_block(pd.DataFrame(), priors)
        assert "STRUCTURAL CANDIDATES" in out
        assert "none found" in out

    def test_render_priors_block_with_total_row(self) -> None:
        priors = MultiRowPriors(
            error_type="column_aggregation_inconsistency",
            source="t.csv",
            total_row_blocks=(
                TotalRowBlock(total_row=5, label="Grand Total",
                              detail_rows=(2, 3, 4)),
            ),
        )
        out = render_priors_block(pd.DataFrame(), priors)
        assert "Grand Total" in out
        assert "row 5" in out
        assert "2, 3, 4" in out

    def test_render_group_blocks_returns_markdown(self) -> None:
        df = pd.DataFrame({
            "Email": ["a", "b", "a"],
            "Name":  ["Ana", "Bob", "Ana"],
        })
        priors = MultiRowPriors(
            error_type="row_duplication", source="t.csv",
            duplicate_subsets=(
                DuplicateSubset(
                    columns=("Email",), n_duplicate_rows=2, n_groups=1,
                    dup_rate=2 / 3, sample_groups=((0, 2),),
                ),
            ),
        )
        out = render_group_blocks(df, priors)
        assert "GROUP SAMPLES" in out
        assert "duplicate group" in out
        # Both rows of the duplicate group appear in the rendered table.
        assert "Ana" in out


# ---------------------------------------------------------------------------
# normalize_for_dedup
# ---------------------------------------------------------------------------


class TestNormalizeForDedup:
    def test_strings_lowercased_and_stripped(self) -> None:
        assert normalize_for_dedup("Acme Corp.") == "acme corp"
        assert normalize_for_dedup("ACME CORPORATION") == "acme corporation"
        assert normalize_for_dedup("  Hello,  World!  ") == "hello world"

    def test_nan_and_none_collapse_to_empty(self) -> None:
        assert normalize_for_dedup(None) == ""
        assert normalize_for_dedup(float("nan")) == ""

    def test_numeric_canonicalisation(self) -> None:
        # 1000.0 and 1000 share a canonical form.
        assert normalize_for_dedup(1000) == normalize_for_dedup(1000.0)
        assert normalize_for_dedup(1000.0) == "1000"
        # Punctuation in numeric strings is treated as a separator.
        assert normalize_for_dedup("1,000.00") == "1 000 00"


# ---------------------------------------------------------------------------
# find_duplicate_subsets_normalized
# ---------------------------------------------------------------------------


class TestFindDuplicateSubsetsNormalized:
    def test_catches_case_variants(self) -> None:
        # Differs only in case / trailing punctuation - exactly what
        # the simple normaliser is designed to catch.
        df = pd.DataFrame({
            "Org": ["Acme Corp.", "ACME CORP", "Other Inc"],
            "Id":  [1, 2, 3],
        })
        d = build_table_digest(df, source="t.csv")
        # Strict scan finds nothing (different strings).
        strict = find_duplicate_subsets(df, digest=d)
        assert all(s.columns != ("Org",) for s in strict)
        # Normalised scan finds the case/punctuation collision.
        loose = find_duplicate_subsets_normalized(df, digest=d)
        org_subs = [s for s in loose if s.columns == ("Org",)]
        assert len(org_subs) == 1
        assert org_subs[0].n_duplicate_rows == 2


# ---------------------------------------------------------------------------
# enumerate_collision_groups
# ---------------------------------------------------------------------------


class TestEnumerateCollisionGroups:
    def test_returns_all_groups_not_just_samples(self) -> None:
        df = pd.DataFrame({
            "Email": ["a", "b", "a", "c", "b", "d"],
            "X":     [1, 2, 1, 3, 2, 4],
        })
        groups = enumerate_collision_groups(df, ["Email"])
        sets = [set(g) for g in groups]
        assert {0, 2} in sets
        assert {1, 4} in sets
        assert len(groups) == 2  # only groups of size >= 2

    def test_skips_rows_with_null_keys(self) -> None:
        df = pd.DataFrame({
            "K": ["a", None, "a", None],
        })
        groups = enumerate_collision_groups(df, ["K"])
        # Only the two non-null "a" rows form a group.
        assert len(groups) == 1
        assert set(groups[0]) == {0, 2}

    def test_normalize_groups_case_variants(self) -> None:
        df = pd.DataFrame({
            "Org": ["Acme Corp.", "ACME CORP", "Other Inc"],
        })
        groups_strict = enumerate_collision_groups(df, ["Org"], normalize=False)
        assert groups_strict == []
        groups_norm = enumerate_collision_groups(df, ["Org"], normalize=True)
        assert len(groups_norm) == 1
        assert set(groups_norm[0]) == {0, 1}
