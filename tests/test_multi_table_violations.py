"""Tests for ``src.core.multi_table_violations`` (the deterministic
compiler that turns LLM-confirmed cross-sheet groups into per-row
violation records).

The compiler must:
    * Skip groups where ``confirmed=False``.
    * Emit one record per (sheet, raw_column) for header_synonym
      and dtype_drift findings.
    * Emit one record per (sheet, row) for duplicate_row and
      attribute_conflict findings, all linked by the same
      ``violation_id``.
    * Emit fk_orphan records only for confirmed JOIN groups, and
      only for child rows whose key is not in the parent's
      normalized value set.
"""

from __future__ import annotations

import pandas as pd

from src.core.multi_table_resolver import (
    ColumnMappingEntry,
    JoinSpec,
    ResolvedGroup,
    ResolvedWorkbook,
)
from src.core.multi_table_violations import (
    MultiTableViolation,
    compile_violations,
)


def _entry(sheet: str, column: str) -> ColumnMappingEntry:
    return ColumnMappingEntry(sheet=sheet, column=column)


def _union_group(
    *,
    workbook: str = "wb.xlsx",
    group_id: str = "u1",
    sheets,
    column_mapping,
    primary_key=(),
    confirmed: bool = True,
) -> ResolvedGroup:
    return ResolvedGroup(
        workbook=workbook,
        group_id=group_id,
        kind="union",
        confirmed=confirmed,
        rationale="test",
        sheets=tuple(sheets),
        column_mapping={
            k: tuple(v) for k, v in column_mapping.items()
        },
        primary_key=tuple(primary_key),
    )


def _wb(workbook: str, sheets, groups):
    return ResolvedWorkbook(
        workbook=workbook, sheets=tuple(sheets), groups=list(groups),
    )


class TestUnconfirmedGroupsSkipped:
    def test_no_violations_when_unconfirmed(self) -> None:
        df_a = pd.DataFrame({"K": [1, 2], "V": [10, 20]})
        df_b = pd.DataFrame({"K": [1, 3], "V": [99, 30]})
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "K": [_entry("a.csv", "K"), _entry("b.csv", "K")],
                "V": [_entry("a.csv", "V"), _entry("b.csv", "V")],
            },
            primary_key=["K"],
            confirmed=False,
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        assert violations == []


class TestHeaderSynonym:
    def test_three_variants_emit_three_linked_records(self) -> None:
        # Three sheets carry the same canonical column under three
        # different raw headers - one record per (sheet, raw_col).
        df_a = pd.DataFrame({"Score %": [10]})
        df_b = pd.DataFrame({"Score (%)": [20]})
        df_c = pd.DataFrame({"score_pct": [30]})
        g = _union_group(
            sheets=["a.csv", "b.csv", "c.csv"],
            column_mapping={
                "Score %": [
                    _entry("a.csv", "Score %"),
                    _entry("b.csv", "Score (%)"),
                    _entry("c.csv", "score_pct"),
                ],
            },
        )
        rw = _wb("wb", ["a.csv", "b.csv", "c.csv"], [g])
        violations = compile_violations(
            resolved=rw,
            frames={"a.csv": df_a, "b.csv": df_b, "c.csv": df_c},
        )
        synonyms = [v for v in violations if v.kind == "header_synonym"]
        assert len(synonyms) == 3
        # All three share a (sheet, raw_col) pair and link to the
        # other two locations.
        sheet_set = {v.sheet for v in synonyms}
        assert sheet_set == {"a.csv", "b.csv", "c.csv"}
        for v in synonyms:
            assert v.row == -1   # header-level finding
            assert len(v.linked) == 2
            assert v.values["canonical"] == "Score %"

    def test_no_synonym_when_all_headers_match(self) -> None:
        df_a = pd.DataFrame({"K": [1]})
        df_b = pd.DataFrame({"K": [2]})
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "K": [_entry("a.csv", "K"), _entry("b.csv", "K")],
            },
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        assert [v for v in violations if v.kind == "header_synonym"] == []


class TestDtypeDrift:
    def test_string_vs_float_drift(self) -> None:
        df_a = pd.DataFrame({"Score": [86.7, 73.3]})        # float
        df_b = pd.DataFrame({"Score": ["90", "90"]})        # str
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "Score": [_entry("a.csv", "Score"), _entry("b.csv", "Score")],
            },
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        drifts = [v for v in violations if v.kind == "dtype_drift"]
        assert len(drifts) == 2
        kinds = {v.values["dtype_kind"] for v in drifts}
        assert kinds == {"float", "str"}

    def test_no_drift_when_dtypes_match(self) -> None:
        df_a = pd.DataFrame({"K": [1, 2]})
        df_b = pd.DataFrame({"K": [3, 4]})
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "K": [_entry("a.csv", "K"), _entry("b.csv", "K")],
            },
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        assert [v for v in violations if v.kind == "dtype_drift"] == []


class TestDuplicateRow:
    def test_shared_pk_identical_attrs(self) -> None:
        df_a = pd.DataFrame({"K": ["A", "B"], "V": [10, 20]})
        df_b = pd.DataFrame({"K": ["A", "C"], "V": [10, 30]})  # A duplicate
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "K": [_entry("a.csv", "K"), _entry("b.csv", "K")],
                "V": [_entry("a.csv", "V"), _entry("b.csv", "V")],
            },
            primary_key=["K"],
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        dups = [v for v in violations if v.kind == "duplicate_row"]
        assert len(dups) == 2   # one per (sheet, row)
        # Linked records share violation_id.
        assert dups[0].violation_id == dups[1].violation_id
        # Each links to the other.
        sheets = {v.sheet for v in dups}
        assert sheets == {"a.csv", "b.csv"}

    def test_pk_normalisation_strips_case_and_whitespace(self) -> None:
        df_a = pd.DataFrame({"K": ["abc"]})
        df_b = pd.DataFrame({"K": ["  ABC  "]})
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "K": [_entry("a.csv", "K"), _entry("b.csv", "K")],
            },
            primary_key=["K"],
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        dups = [v for v in violations if v.kind == "duplicate_row"]
        assert len(dups) == 2

    def test_no_duplicate_when_pk_disjoint(self) -> None:
        df_a = pd.DataFrame({"K": ["A"]})
        df_b = pd.DataFrame({"K": ["B"]})
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "K": [_entry("a.csv", "K"), _entry("b.csv", "K")],
            },
            primary_key=["K"],
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        assert [v for v in violations if v.kind == "duplicate_row"] == []


class TestAttributeConflict:
    def test_shared_pk_disagreeing_attrs(self) -> None:
        df_a = pd.DataFrame({"K": ["A"], "V": [10], "Loc": ["NY"]})
        df_b = pd.DataFrame({"K": ["A"], "V": [99], "Loc": ["LA"]})  # both V and Loc differ
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "K": [_entry("a.csv", "K"), _entry("b.csv", "K")],
                "V": [_entry("a.csv", "V"), _entry("b.csv", "V")],
                "Loc": [_entry("a.csv", "Loc"), _entry("b.csv", "Loc")],
            },
            primary_key=["K"],
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        confs = [v for v in violations if v.kind == "attribute_conflict"]
        assert len(confs) == 2
        assert confs[0].violation_id == confs[1].violation_id
        # Both V and Loc are listed as conflicting columns.
        for v in confs:
            assert set(v.columns) == {"V", "Loc"}
            assert "V" in v.values
            assert "Loc" in v.values

    def test_partial_conflict_only_disagreeing_columns(self) -> None:
        df_a = pd.DataFrame({"K": ["A"], "V": [10], "Loc": ["NY"]})
        df_b = pd.DataFrame({"K": ["A"], "V": [10], "Loc": ["LA"]})  # only Loc differs
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "K": [_entry("a.csv", "K"), _entry("b.csv", "K")],
                "V": [_entry("a.csv", "V"), _entry("b.csv", "V")],
                "Loc": [_entry("a.csv", "Loc"), _entry("b.csv", "Loc")],
            },
            primary_key=["K"],
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        confs = [v for v in violations if v.kind == "attribute_conflict"]
        assert len(confs) == 2
        for v in confs:
            assert set(v.columns) == {"Loc"}


class TestFKOrphan:
    def test_orphan_child_rows(self) -> None:
        parent = pd.DataFrame({"id": ["P1", "P2", "P3"]})
        child = pd.DataFrame({"pid": ["P1", "P2", "ZZ"]})  # ZZ is orphan
        g = ResolvedGroup(
            workbook="wb",
            group_id="j1",
            kind="join",
            confirmed=True,
            rationale="test",
            sheets=("parent.csv", "child.csv"),
            column_mapping={
                "id":  (_entry("parent.csv", "id"),),
                "pid": (_entry("child.csv", "pid"),),
            },
            primary_key=(),
            join=JoinSpec(
                parent_sheet="parent.csv",
                parent_columns=("id",),
                child_sheet="child.csv",
                child_columns=("pid",),
            ),
        )
        rw = _wb("wb", ["parent.csv", "child.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"parent.csv": parent, "child.csv": child},
        )
        orphans = [v for v in violations if v.kind == "fk_orphan"]
        assert len(orphans) == 1
        assert orphans[0].sheet == "child.csv"
        assert orphans[0].row == 2
        assert orphans[0].values["child_key"] == ["zz"]   # normalised


class TestEndToEndStableOrder:
    def test_output_order(self) -> None:
        df_a = pd.DataFrame({
            "K": ["A", "B"],
            "Score %": [86.7, 73.3],
        })
        df_b = pd.DataFrame({
            "K": ["A", "C"],
            "Score (%)": [86.7, 90.0],
        })
        g = _union_group(
            sheets=["a.csv", "b.csv"],
            column_mapping={
                "K": [_entry("a.csv", "K"), _entry("b.csv", "K")],
                "Score %": [
                    _entry("a.csv", "Score %"),
                    _entry("b.csv", "Score (%)"),
                ],
            },
            primary_key=["K"],
        )
        rw = _wb("wb", ["a.csv", "b.csv"], [g])
        violations = compile_violations(
            resolved=rw, frames={"a.csv": df_a, "b.csv": df_b},
        )
        # Order: header_synonym (2), dtype_drift (0 - both float),
        # duplicate_row (2 from K=A), attribute_conflict (0), fk_orphan (0)
        kinds_order = [v.kind for v in violations]
        assert kinds_order == [
            "header_synonym", "header_synonym",
            "duplicate_row", "duplicate_row",
        ]
