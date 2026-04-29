"""Tests for ``src.data_prep.curate`` pure-helper edge cases.

Pinned by Pass 6c MED gap. Covers the deterministic helpers
(``_cell_is_empty``, ``_cell_is_oversized``, ``_empty_cell_ratio``,
``_clean_dataframe``, ``_is_non_latin_letter``, ``_scan_non_latin``,
``_looks_like_text_token``, ``_parse_lang_json``). The full ``curate``
entry-point integrates a langdetect/LLM stage and a filesystem walk
that's out-of-scope for these unit tests.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data_prep.curate import (
    _cell_is_empty,
    _cell_is_oversized,
    _clean_dataframe,
    _empty_cell_ratio,
    _is_non_latin_letter,
    _looks_like_text_token,
    _parse_lang_json,
    _scan_non_latin,
)


# ---------------------------------------------------------------------------
# _cell_is_empty
# ---------------------------------------------------------------------------


class TestCellIsEmpty:
    def test_none_is_empty(self) -> None:
        assert _cell_is_empty(None) is True

    def test_nan_is_empty(self) -> None:
        assert _cell_is_empty(float("nan")) is True
        assert _cell_is_empty(np.nan) is True
        assert _cell_is_empty(pd.NA) is True

    def test_empty_string_is_empty(self) -> None:
        assert _cell_is_empty("") is True

    def test_whitespace_only_is_empty(self) -> None:
        assert _cell_is_empty("   ") is True
        assert _cell_is_empty("\t\n") is True

    def test_sentinels_case_insensitive(self) -> None:
        for s in ["-", "--", "*", "?", ".", "N/A", "n/a", "NA",
                   "null", "None", "NaN", "#N/A"]:
            assert _cell_is_empty(s) is True, s

    def test_padded_sentinels(self) -> None:
        assert _cell_is_empty("  -  ") is True
        assert _cell_is_empty("  N/A  ") is True

    def test_real_text_not_empty(self) -> None:
        assert _cell_is_empty("hello") is False
        assert _cell_is_empty("0") is False  # zero string is meaningful
        assert _cell_is_empty("--actual-value--") is False

    def test_zero_int_not_empty(self) -> None:
        assert _cell_is_empty(0) is False

    def test_non_string_non_nan_not_empty(self) -> None:
        assert _cell_is_empty(42) is False
        assert _cell_is_empty(3.14) is False
        assert _cell_is_empty(False) is False  # explicit boolean


# ---------------------------------------------------------------------------
# _cell_is_oversized
# ---------------------------------------------------------------------------


class TestCellIsOversized:
    def test_no_thresholds_means_never_oversized(self) -> None:
        assert _cell_is_oversized("x" * 10_000, max_chars=None,
                                    max_words=None) is False

    def test_chars_threshold(self) -> None:
        assert _cell_is_oversized("x" * 50, max_chars=49,
                                    max_words=None) is True
        assert _cell_is_oversized("x" * 50, max_chars=50,
                                    max_words=None) is False

    def test_words_threshold(self) -> None:
        s = " ".join(["w"] * 10)
        assert _cell_is_oversized(s, max_chars=None, max_words=9) is True
        assert _cell_is_oversized(s, max_chars=None, max_words=10) is False

    def test_either_side_triggers(self) -> None:
        # Char limit OK but word limit exceeded.
        s = "a b c d e"
        assert _cell_is_oversized(s, max_chars=100, max_words=4) is True

    def test_zero_threshold_disables(self) -> None:
        assert _cell_is_oversized("x" * 999, max_chars=0,
                                    max_words=None) is False

    def test_none_value_not_oversized(self) -> None:
        assert _cell_is_oversized(None, max_chars=1, max_words=1) is False

    def test_non_string_coerced(self) -> None:
        # int 12345 → "12345" → 5 chars.
        assert _cell_is_oversized(12345, max_chars=4, max_words=None) is True
        assert _cell_is_oversized(12345, max_chars=10, max_words=None) is False


# ---------------------------------------------------------------------------
# _empty_cell_ratio
# ---------------------------------------------------------------------------


class TestEmptyCellRatio:
    def test_empty_frame_returns_zero(self) -> None:
        assert _empty_cell_ratio(pd.DataFrame()) == 0.0

    def test_no_empty_cells(self) -> None:
        df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
        assert _empty_cell_ratio(df) == 0.0

    def test_half_empty(self) -> None:
        df = pd.DataFrame({"a": ["x", ""], "b": ["", "y"]})
        assert _empty_cell_ratio(df) == 0.5

    def test_sentinels_count_as_empty(self) -> None:
        df = pd.DataFrame({"a": ["-", "x"], "b": ["N/A", "y"]})
        assert _empty_cell_ratio(df) == 0.5


# ---------------------------------------------------------------------------
# _clean_dataframe
# ---------------------------------------------------------------------------


class TestCleanDataFrame:
    def test_drops_all_empty_columns(self) -> None:
        df = pd.DataFrame({
            "a": ["x", "y"],
            "b": ["", "  "],   # all-empty
            "c": ["1", "2"],
            "d": ["-", "N/A"],  # all-sentinel-empty
        })
        clean, dropped_cols, dropped_rows_empty, dropped_rows_big = (
            _clean_dataframe(df)
        )
        assert dropped_cols == 2
        assert list(clean.columns) == ["a", "c"]
        assert dropped_rows_empty == 0
        assert dropped_rows_big == 0

    def test_drops_rows_with_any_empty(self) -> None:
        df = pd.DataFrame({
            "a": ["x", "y", "z"],
            "b": ["1", "", "3"],  # row 1 has empty
        })
        clean, _, dropped_rows_empty, _ = _clean_dataframe(df)
        assert dropped_rows_empty == 1
        assert list(clean["a"]) == ["x", "z"]

    def test_oversized_row_dropping(self) -> None:
        df = pd.DataFrame({
            "a": ["x", "y" * 200, "ok"],
            "b": ["1", "2", "3"],
        })
        clean, _, dropped_rows_empty, dropped_rows_big = _clean_dataframe(
            df, max_cell_chars=50, max_cell_words=None,
        )
        assert dropped_rows_empty == 0
        assert dropped_rows_big == 1
        assert list(clean["a"]) == ["x", "ok"]

    def test_oversized_not_double_counted_with_empty(self) -> None:
        # A row that is BOTH empty AND oversized counts only as empty.
        df = pd.DataFrame({
            "a": ["x", "huge" * 100],
            "b": ["1", ""],  # row 1 also empty
        })
        clean, _, dropped_rows_empty, dropped_rows_big = _clean_dataframe(
            df, max_cell_chars=50, max_cell_words=None,
        )
        assert dropped_rows_empty == 1
        assert dropped_rows_big == 0
        assert len(clean) == 1

    def test_clean_resets_index(self) -> None:
        df = pd.DataFrame({"a": ["x", "", "z"]})
        clean, *_ = _clean_dataframe(df)
        assert list(clean.index) == [0, 1]

    def test_completely_empty_frame_passthrough(self) -> None:
        df = pd.DataFrame()
        clean, dc, dre, drb = _clean_dataframe(df)
        assert clean.empty
        assert (dc, dre, drb) == (0, 0, 0)


# ---------------------------------------------------------------------------
# _is_non_latin_letter / _scan_non_latin
# ---------------------------------------------------------------------------


class TestIsNonLatinLetter:
    def test_ascii_letters_are_latin(self) -> None:
        for ch in "abcXYZ":
            assert _is_non_latin_letter(ch) is False, ch

    def test_accented_latin_letters_are_latin(self) -> None:
        for ch in "éñüÆçß":
            assert _is_non_latin_letter(ch) is False, ch

    def test_cjk_is_non_latin(self) -> None:
        for ch in "中文漢字한국어":
            assert _is_non_latin_letter(ch) is True, ch

    def test_cyrillic_arabic_devanagari_are_non_latin(self) -> None:
        for ch in "Привет":  # Cyrillic
            assert _is_non_latin_letter(ch) is True, ch
        for ch in "مرحبا":  # Arabic
            assert _is_non_latin_letter(ch) is True, ch
        for ch in "नमस्ते"[:1]:  # Devanagari letter (skip combining mark)
            assert _is_non_latin_letter(ch) is True, ch

    def test_digits_punctuation_pass(self) -> None:
        for ch in "0123 .,!?€¥":
            assert _is_non_latin_letter(ch) is False, ch


class TestScanNonLatin:
    def test_pure_latin_returns_zero(self) -> None:
        df = pd.DataFrame({"name": ["Alice", "Bob", "Caf\u00e9"]})
        count, example = _scan_non_latin(df)
        assert count == 0
        assert example == ""

    def test_counts_chars_and_returns_first_example(self) -> None:
        df = pd.DataFrame({
            "name": ["Alice", "中文", "Bob"],
        })
        count, example = _scan_non_latin(df)
        assert count == 2  # two CJK chars in "中文"
        assert example == "中"

    def test_scans_headers_too(self) -> None:
        df = pd.DataFrame({"中": ["a"], "name": ["b"]})
        count, example = _scan_non_latin(df)
        assert count >= 1
        assert example == "中"


# ---------------------------------------------------------------------------
# _looks_like_text_token
# ---------------------------------------------------------------------------


class TestLooksLikeTextToken:
    def test_short_token_rejected(self) -> None:
        assert _looks_like_text_token("a") is False
        assert _looks_like_text_token("") is False

    def test_alpha_token_kept(self) -> None:
        assert _looks_like_text_token("hello") is True
        assert _looks_like_text_token("caf\u00e9") is True

    def test_pure_numeric_rejected(self) -> None:
        assert _looks_like_text_token("12345") is False

    def test_alphanumeric_kept(self) -> None:
        # Mixed alpha+digit: the helper accepts ANY alpha presence so id-like
        # tokens such as ``ABC123`` pass. (Documented behaviour.)
        assert _looks_like_text_token("ABC123") is True


# ---------------------------------------------------------------------------
# _parse_lang_json
# ---------------------------------------------------------------------------


class TestParseLangJson:
    def test_empty_returns_none(self) -> None:
        assert _parse_lang_json("") is None
        assert _parse_lang_json("   ") is None

    def test_bare_object(self) -> None:
        assert _parse_lang_json('{"language":"en"}') == {"language": "en"}

    def test_fenced_json(self) -> None:
        assert _parse_lang_json('```json\n{"language":"en"}\n```') == {
            "language": "en"
        }

    def test_fenced_no_lang_tag(self) -> None:
        assert _parse_lang_json('```\n{"x":1}\n```') == {"x": 1}

    def test_object_with_surrounding_prose(self) -> None:
        # The regex ``\{.*\}`` is greedy, so prose around an object still
        # extracts the object.
        out = _parse_lang_json('Sure! {"language":"fr"} hope that helps')
        assert out == {"language": "fr"}

    def test_no_json_at_all_returns_none(self) -> None:
        assert _parse_lang_json("just prose") is None

    def test_invalid_json_returns_none(self) -> None:
        assert _parse_lang_json('{"bad json') is None
