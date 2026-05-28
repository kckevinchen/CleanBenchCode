"""Tests for ``src.core.data_loader``: CSV loading + chunk iteration.

Covers the load/chunk machinery the rest of the pipeline depends on:

* dtype=str preservation and keep_default_na=False (numbers stay strings,
  empty cells stay ``""`` rather than becoming NaN)
* duplicate-header disambiguation
* whitespace-stripped header collisions
* embedded newlines in cell values
* chunk slicing math (start_row / end_row half-open, last partial chunk,
  max_chunks early stop)
* column_mode = "single" yields one chunk per column
* preserved frame index (perturbation experiments rely on this)
* error paths: missing file, bad chunk_size, unknown column_mode,
  unknown column subset
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from src.core.data_loader import (
    COLUMN_MODE_ALL,
    COLUMN_MODE_SINGLE,
    CsvChunk,
    _drop_duplicate_columns,
    _visible_whitespace,
    iter_chunks,
    load_csv,
)


# ---------------------------------------------------------------------------
# _drop_duplicate_columns
# ---------------------------------------------------------------------------


class TestDropDuplicateColumns:
    def test_no_duplicates_passthrough(self) -> None:
        names, keep, dropped = _drop_duplicate_columns(["a", "b", "c"])
        assert names == ["a", "b", "c"]
        assert keep == [0, 1, 2]
        assert dropped == []

    def test_basic_duplicate_drops_later(self) -> None:
        # "a" appears 3x; only the first position survives.
        names, keep, dropped = _drop_duplicate_columns(["a", "a", "a"])
        assert names == ["a"]
        assert keep == [0]
        assert dropped == ["a"]

    def test_interleaved_duplicates(self) -> None:
        # First occurrences win regardless of distance.
        names, keep, dropped = _drop_duplicate_columns(["a", "b", "a", "b"])
        assert names == ["a", "b"]
        assert keep == [0, 1]
        assert dropped == ["a", "b"]

    def test_empty_input(self) -> None:
        assert _drop_duplicate_columns([]) == ([], [], [])


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_dtype_str_preserves_numbers(self, tmp_path: Path) -> None:
        p = tmp_path / "x.csv"
        p.write_text("id,price\n1,1.50\n2,1.500\n")
        df = load_csv(p)
        # All values stay strings — "1.500" must NOT collapse to "1.5".
        assert list(df["price"]) == ["1.50", "1.500"]
        assert list(df["id"]) == ["1", "2"]
        # Whatever the storage backend (object/string), the values
        # round-trip as Python strings.
        assert all(isinstance(v, str) for v in df["price"])
        assert all(isinstance(v, str) for v in df["id"])

    def test_empty_cells_stay_empty_string(self, tmp_path: Path) -> None:
        # keep_default_na=False means a blank cell is "" not NaN.
        p = tmp_path / "x.csv"
        p.write_text("a,b\n,foo\nbar,\n")
        df = load_csv(p)
        assert list(df["a"]) == ["", "bar"]
        assert list(df["b"]) == ["foo", ""]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nope.csv")

    def test_duplicate_headers_disambiguated(self, tmp_path: Path) -> None:
        p = tmp_path / "x.csv"
        p.write_text("name,name\nalice,bob\n")
        df = load_csv(p)
        # pandas itself dedupes during read; our extra pass is a no-op
        # for already-distinct labels but the load must round-trip.
        assert len(df.columns) == 2
        assert df.columns[0] == "name"
        # Single Series per column (not DataFrame).
        assert isinstance(df[df.columns[0]], pd.Series)
        assert isinstance(df[df.columns[1]], pd.Series)

    def test_whitespace_collision_dedupes(self, tmp_path: Path) -> None:
        # Two distinct on-disk labels (" Numeric" + "Numeric") collide
        # after .strip(); the SECOND one is dropped entirely so callers
        # always get a single Series per column name.
        p = tmp_path / "x.csv"
        p.write_text(" Numeric,Numeric\n1,2\n3,4\n")
        df = load_csv(p)
        assert list(df.columns) == ["Numeric"]
        assert isinstance(df["Numeric"], pd.Series)
        # The kept column is the FIRST occurrence (values "1" and "3").
        assert list(df["Numeric"]) == ["1", "3"]

    def test_embedded_newlines_in_cell_preserved(self, tmp_path: Path) -> None:
        # A quoted cell with an internal newline must survive as a single
        # row containing the literal "\n" inside the value.
        p = tmp_path / "x.csv"
        p.write_text('a,b\n"line1\nline2",ok\n')
        df = load_csv(p)
        assert len(df) == 1
        assert df.loc[0, "a"] == "line1\nline2"
        assert df.loc[0, "b"] == "ok"

    def test_cells_strip_leading_trailing_whitespace(self, tmp_path: Path) -> None:
        # Leading/trailing whitespace is treated as a CSV-format padding
        # artefact and normalised at load time. Internal whitespace
        # (repeated spaces, embedded newlines) is preserved verbatim so
        # the typo / inconsistency detectors still see it.
        p = tmp_path / "x.csv"
        p.write_text(
            'name,note\n'
            '" James Smith","ok"\n'
            '"James Smith ","trailing"\n'
            '"James   Smith","internal-repeat"\n'
            '"a","\tindented"\n'
            '"  ","whitespace-only collapses to empty"\n'
            '"alice","line1\nline2"\n'
        )
        df = load_csv(p)
        # Leading / trailing trimmed.
        assert df.loc[0, "name"] == "James Smith"
        assert df.loc[1, "name"] == "James Smith"
        # Internal repeated space preserved.
        assert df.loc[2, "name"] == "James   Smith"
        # Leading tab is whitespace -> stripped.
        assert df.loc[3, "note"] == "indented"
        # Whitespace-only collapses to empty string.
        assert df.loc[4, "name"] == ""
        # Embedded newline survives (str.strip only touches the edges).
        assert df.loc[5, "note"] == "line1\nline2"


# ---------------------------------------------------------------------------
# iter_chunks
# ---------------------------------------------------------------------------


def _frame(rows: int = 10, cols: int = 3) -> pd.DataFrame:
    data = {f"c{i}": [f"r{r}c{i}" for r in range(rows)] for i in range(cols)}
    return pd.DataFrame(data)


class TestIterChunks:
    def test_all_mode_one_chunk_per_slice(self) -> None:
        df = _frame(rows=10, cols=3)
        chunks = list(iter_chunks(df, source=Path("x.csv"), chunk_size=4))
        # Slices: [0,4), [4,8), [8,10) — three chunks, each with all 3 cols.
        assert len(chunks) == 3
        assert [(c.start_row, c.end_row) for c in chunks] == [(0, 4), (4, 8), (8, 10)]
        assert all(c.headers == ["c0", "c1", "c2"] for c in chunks)
        assert all(c.column_mode == COLUMN_MODE_ALL for c in chunks)
        # Last chunk is the remainder (size 2).
        assert chunks[-1].row_count == 2

    def test_single_mode_one_chunk_per_column(self) -> None:
        df = _frame(rows=4, cols=3)
        chunks = list(iter_chunks(
            df, source=Path("x.csv"), chunk_size=10,
            column_mode=COLUMN_MODE_SINGLE,
        ))
        # 1 row-slice × 3 columns = 3 chunks, one column each.
        assert len(chunks) == 3
        assert [c.headers for c in chunks] == [["c0"], ["c1"], ["c2"]]
        assert all(c.column_count == 1 for c in chunks)
        assert all(c.is_single_column for c in chunks)

    def test_single_mode_combined_with_chunk_size(self) -> None:
        df = _frame(rows=10, cols=2)
        chunks = list(iter_chunks(
            df, source=Path("x.csv"), chunk_size=4,
            column_mode=COLUMN_MODE_SINGLE,
        ))
        # 3 row-slices × 2 columns = 6 chunks.
        assert len(chunks) == 6
        # Order: row-slice MAJOR, column MINOR (matches generator).
        assert [(c.start_row, c.headers[0]) for c in chunks] == [
            (0, "c0"), (0, "c1"),
            (4, "c0"), (4, "c1"),
            (8, "c0"), (8, "c1"),
        ]

    def test_max_chunks_early_stop(self) -> None:
        df = _frame(rows=100, cols=3)
        chunks = list(iter_chunks(df, source=Path("x.csv"), chunk_size=10, max_chunks=2))
        assert len(chunks) == 2
        assert (chunks[0].start_row, chunks[0].end_row) == (0, 10)
        assert (chunks[1].start_row, chunks[1].end_row) == (10, 20)

    def test_max_chunks_zero_yields_nothing(self) -> None:
        df = _frame(rows=10, cols=3)
        chunks = list(iter_chunks(df, source=Path("x.csv"), chunk_size=4, max_chunks=0))
        assert chunks == []

    def test_start_row_offsets_slicing(self) -> None:
        df = _frame(rows=10, cols=2)
        chunks = list(iter_chunks(
            df, source=Path("x.csv"), chunk_size=4, start_row=3,
        ))
        # Slices begin at 3: [3,7), [7,10).
        assert [(c.start_row, c.end_row) for c in chunks] == [(3, 7), (7, 10)]

    def test_explicit_columns_subset(self) -> None:
        df = _frame(rows=4, cols=4)
        chunks = list(iter_chunks(
            df, source=Path("x.csv"), chunk_size=10,
            columns=["c0", "c2"],
        ))
        assert len(chunks) == 1
        assert chunks[0].headers == ["c0", "c2"]

    def test_unknown_column_raises_keyerror(self) -> None:
        df = _frame(rows=4, cols=2)
        with pytest.raises(KeyError, match="Requested columns not present"):
            list(iter_chunks(
                df, source=Path("x.csv"), chunk_size=10,
                columns=["c0", "doesnotexist"],
            ))

    def test_chunk_size_must_be_positive(self) -> None:
        df = _frame(rows=4, cols=2)
        with pytest.raises(ValueError, match="chunk_size"):
            list(iter_chunks(df, source=Path("x.csv"), chunk_size=0))
        with pytest.raises(ValueError, match="chunk_size"):
            list(iter_chunks(df, source=Path("x.csv"), chunk_size=-1))

    def test_invalid_column_mode_raises(self) -> None:
        df = _frame(rows=4, cols=2)
        with pytest.raises(ValueError, match="column_mode"):
            list(iter_chunks(
                df, source=Path("x.csv"), chunk_size=4, column_mode="bogus",
            ))

    def test_empty_columns_subset_raises(self) -> None:
        df = _frame(rows=4, cols=2)
        with pytest.raises(ValueError, match="No active columns"):
            list(iter_chunks(
                df, source=Path("x.csv"), chunk_size=4, columns=[],
            ))

    def test_frame_index_preserved_for_shuffled_input(self) -> None:
        # Perturbation experiments shuffle rows BEFORE chunking and rely
        # on the preserved index to map LLM output back to the original
        # CSV row number. This regresses if iter_chunks ever resets.
        df = _frame(rows=6, cols=2)
        shuffled = df.iloc[[3, 0, 5, 2, 4, 1]]
        chunks = list(iter_chunks(shuffled, source=Path("x.csv"), chunk_size=3))
        assert len(chunks) == 2
        assert list(chunks[0].frame.index) == [3, 0, 5]
        assert list(chunks[1].frame.index) == [2, 4, 1]


# ---------------------------------------------------------------------------
# CsvChunk.to_markdown
# ---------------------------------------------------------------------------


class TestCsvChunkToMarkdown:
    def test_includes_row_column_with_index(self) -> None:
        df = _frame(rows=4, cols=2).iloc[[2, 0]]
        chunk = CsvChunk(
            source=Path("x.csv"),
            start_row=0, end_row=4,
            headers=["c0", "c1"],
            frame=df, column_mode=COLUMN_MODE_ALL,
        )
        md = chunk.to_markdown(max_cell_chars=120)
        # Original CSV row numbers (2, 0) must appear, not a 0..N reset.
        assert "__row__" in md
        # Data rows: skip the markdown header ("| __row__") and separator
        # ("|---"). The first data row should carry the preserved index 2.
        data_rows = [
            ln for ln in md.splitlines()
            if ln.startswith("|") and "__row__" not in ln and "---" not in ln
        ]
        assert len(data_rows) == 2
        # First/second data rows carry the preserved indexes 2 and 0.
        first_cells = [c.strip() for c in data_rows[0].split("|")[1:-1]]
        second_cells = [c.strip() for c in data_rows[1].split("|")[1:-1]]
        assert first_cells[0] == "2"
        assert second_cells[0] == "0"

    def test_truncates_long_cells(self) -> None:
        df = pd.DataFrame({"big": ["x" * 500]})
        chunk = CsvChunk(
            source=Path("x.csv"),
            start_row=0, end_row=1,
            headers=["big"],
            frame=df, column_mode=COLUMN_MODE_ALL,
        )
        md = chunk.to_markdown(max_cell_chars=20)
        # Trail ellipsis appears; raw 500-char string does not.
        assert "\u2026" in md
        assert "x" * 500 not in md

    def test_marks_trailing_space(self) -> None:
        # The Markdown table renderer (tabulate) right-pads cells with
        # spaces - so trailing whitespace inside a cell would normally
        # be invisible. ``to_markdown`` must inject a sentinel so the
        # LLM can still see it.
        df = pd.DataFrame({"City": ["Centerville ", "Adams"]})
        chunk = CsvChunk(
            source=Path("x.csv"),
            start_row=0, end_row=2,
            headers=["City"],
            frame=df, column_mode=COLUMN_MODE_ALL,
        )
        md = chunk.to_markdown(max_cell_chars=120)
        assert "Centerville\u2423" in md
        # The "clean" cell is rendered without any sentinel.
        assert " Adams " in md  # surrounded by table padding spaces

    def test_marks_leading_space_and_embedded_newline(self) -> None:
        df = pd.DataFrame({"Note": [" foo", "bar\nbaz"]})
        chunk = CsvChunk(
            source=Path("x.csv"),
            start_row=0, end_row=2,
            headers=["Note"],
            frame=df, column_mode=COLUMN_MODE_ALL,
        )
        md = chunk.to_markdown(max_cell_chars=120)
        assert "\u2423foo" in md         # leading space
        assert "bar\u21B5baz" in md      # embedded \n


# ---------------------------------------------------------------------------
# _visible_whitespace
# ---------------------------------------------------------------------------


class TestVisibleWhitespace:
    def test_no_change_for_clean_value(self) -> None:
        assert _visible_whitespace("hello") == "hello"
        # Internal single-space runs are preserved as-is.
        assert _visible_whitespace("hello world") == "hello world"
        # Internal multi-space runs are also preserved (tabulate keeps
        # them; only edge whitespace and control chars get sentinels).
        assert _visible_whitespace("foo  bar") == "foo  bar"

    def test_leading_and_trailing_runs_are_marked(self) -> None:
        assert _visible_whitespace(" foo ") == "\u2423foo\u2423"
        assert _visible_whitespace("   foo") == "\u2423\u2423\u2423foo"
        assert _visible_whitespace("foo   ") == "foo\u2423\u2423\u2423"
        assert _visible_whitespace("   ") == "\u2423\u2423\u2423"

    def test_special_chars_are_substituted(self) -> None:
        assert _visible_whitespace("a\tb") == "a\u2192b"
        assert _visible_whitespace("a\nb") == "a\u21B5b"
        assert _visible_whitespace("a\rb") == "a\u21B5b"
        assert _visible_whitespace("a\u00A0b") == "a\u2423b"     # NBSP -> open box
        assert _visible_whitespace("a\u200Bb") == "a\u00B7b"      # ZWSP -> middle dot
        assert _visible_whitespace("a\uFEFFb") == "a\u00B7b"      # BOM -> middle dot

    def test_empty_string_unchanged(self) -> None:
        assert _visible_whitespace("") == ""
