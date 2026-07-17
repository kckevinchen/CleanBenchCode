"""Tests for ``src.core.multi_table.pipeline.group_workbooks`` grouping
modes.

``group_by`` decides how flat CSV corpora are bucketed into cross-sheet
workbooks:

* ``parent-dir`` (default, unchanged): parent directory name; flat
  files collapse to ``"_root"``.
* ``prefix``: basename up to the first ``"_"`` - the fix for flat
  PBI-style ``<reportID>_<TableName>.csv`` dumps where every table of
  one report shares a numeric prefix but sits in one directory.
* ``regex``: first capture group of a user-supplied pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.multi_table.pipeline import group_workbooks


# ---------------------------------------------------------------------------
# parent-dir (default; must stay backward-compatible)
# ---------------------------------------------------------------------------


class TestParentDir:
    def test_groups_by_parent_directory(self) -> None:
        paths = [
            Path("data/wbX/a.csv"),
            Path("data/wbX/b.csv"),
            Path("data/wbY/c.csv"),
        ]
        wb, _ = group_workbooks(paths)
        assert wb["wbX"] == ["a.csv", "b.csv"]
        assert wb["wbY"] == ["c.csv"]

    def test_flat_file_collapses_to_root(self) -> None:
        # A file directly under the scan root has an empty parent
        # name -> "_root" (historical behaviour).
        paths = [Path("a.csv"), Path("b.csv")]
        wb, _ = group_workbooks(paths)
        assert set(wb) == {"_root"}
        assert sorted(wb["_root"]) == ["a.csv", "b.csv"]

    def test_basename_collision_disambiguated(self) -> None:
        paths = [Path("wbX/a.csv"), Path("wbY/a.csv")]
        wb, source_keys = group_workbooks(paths, csv_dir=Path("."))
        # Second a.csv keeps a relative-path key so the two don't
        # clobber each other in source_frames.
        keys = list(source_keys.values())
        assert keys[0] == "a.csv"
        assert keys[1] != "a.csv"
        assert keys[1].endswith("a.csv")


# ---------------------------------------------------------------------------
# prefix (flat PBI corpora)
# ---------------------------------------------------------------------------


class TestPrefix:
    def test_groups_flat_pbi_by_numeric_prefix(self) -> None:
        paths = [
            Path("pbi/781731115_Sales.csv"),
            Path("pbi/781731115_Returns.csv"),
            Path("pbi/99_A.csv"),
            Path("pbi/99_B.csv"),
        ]
        wb, _ = group_workbooks(paths, group_by="prefix")
        assert wb["781731115"] == [
            "781731115_Sales.csv", "781731115_Returns.csv",
        ]
        assert wb["99"] == ["99_A.csv", "99_B.csv"]

    def test_no_underscore_becomes_singleton(self) -> None:
        # A basename with no "_" stays in its own group (full name as
        # id) rather than polluting a shared bucket.
        paths = [Path("pbi/lonely.csv"), Path("pbi/10_x.csv")]
        wb, _ = group_workbooks(paths, group_by="prefix")
        assert wb["lonely.csv"] == ["lonely.csv"]
        assert wb["10"] == ["10_x.csv"]

    def test_prefix_ignores_parent_directory(self) -> None:
        # Same report id split across two directories still groups
        # together under prefix mode.
        paths = [
            Path("dirA/55_one.csv"),
            Path("dirB/55_two.csv"),
        ]
        wb, _ = group_workbooks(paths, group_by="prefix")
        assert sorted(wb["55"]) == ["55_one.csv", "55_two.csv"]


# ---------------------------------------------------------------------------
# regex
# ---------------------------------------------------------------------------


class TestRegex:
    def test_first_capture_group_is_workbook_id(self) -> None:
        paths = [
            Path("pbi/781731115_Sales.csv"),
            Path("pbi/781731115_Returns.csv"),
        ]
        wb, _ = group_workbooks(
            paths, group_by="regex", group_regex=r"^(\d+)_",
        )
        assert wb["781731115"] == [
            "781731115_Sales.csv", "781731115_Returns.csv",
        ]

    def test_no_groups_uses_whole_match(self) -> None:
        paths = [Path("rep_alpha_1.csv"), Path("rep_alpha_2.csv")]
        wb, _ = group_workbooks(
            paths, group_by="regex", group_regex=r"rep_alpha",
        )
        assert sorted(wb["rep_alpha"]) == [
            "rep_alpha_1.csv", "rep_alpha_2.csv",
        ]

    def test_non_matching_files_are_singletons(self) -> None:
        paths = [Path("12_a.csv"), Path("nomatch.csv")]
        wb, _ = group_workbooks(
            paths, group_by="regex", group_regex=r"^(\d+)_",
        )
        assert wb["12"] == ["12_a.csv"]
        assert any(
            k.startswith("_ungrouped") and v == ["nomatch.csv"]
            for k, v in wb.items()
        )

    def test_missing_regex_raises(self) -> None:
        with pytest.raises(ValueError):
            group_workbooks([Path("a.csv")], group_by="regex")

    def test_invalid_regex_raises(self) -> None:
        with pytest.raises(ValueError):
            group_workbooks(
                [Path("a.csv")], group_by="regex", group_regex="(",
            )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_group_by_raises(self) -> None:
        with pytest.raises(ValueError):
            group_workbooks([Path("a.csv")], group_by="nope")
