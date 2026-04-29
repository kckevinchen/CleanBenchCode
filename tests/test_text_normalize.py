"""Tests for ``src.core.text_normalize``: per-cell + per-frame normalisation.

Pinned by Pass 6c MED gap. Profile-permutation coverage: each flag in
``NormalizeOptions`` is exercised in isolation and in combination, and
the no-op short-circuits (``enabled=False``, non-string, empty string)
are tested.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.core.text_normalize import (
    NormalizeOptions,
    normalize_cell,
    normalize_frame,
)


# ---------------------------------------------------------------------------
# normalize_cell — single-flag behaviour
# ---------------------------------------------------------------------------


class TestNormalizeCellShortCircuits:
    def test_disabled_returns_input_unchanged(self) -> None:
        opts = NormalizeOptions(enabled=False, strip=True, fold_whitespace=True)
        # Even with a value that WOULD be touched, disabled wins.
        assert normalize_cell("  hi  ", opts) == "  hi  "

    def test_non_string_passthrough(self) -> None:
        opts = NormalizeOptions(enabled=True)
        assert normalize_cell(42, opts) == 42
        nan = float("nan")
        out = normalize_cell(nan, opts)
        assert isinstance(out, float)
        assert math.isnan(out)
        assert normalize_cell(None, opts) is None

    def test_empty_string_passthrough(self) -> None:
        opts = NormalizeOptions(enabled=True, strip=True, fold_whitespace=True)
        assert normalize_cell("", opts) == ""


class TestNormalizeCellFlags:
    def test_strip_only(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=True, collapse_spaces=False,
            fold_whitespace=False, nfkc=False,
        )
        assert normalize_cell("  hello  ", opts) == "hello"

    def test_collapse_spaces_only(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=False, collapse_spaces=True,
            fold_whitespace=False, nfkc=False,
        )
        assert normalize_cell("a   b    c", opts) == "a b c"

    def test_fold_whitespace_nbsp_to_space(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=False, collapse_spaces=False,
            fold_whitespace=True, nfkc=False,
        )
        assert normalize_cell("a\u00A0b", opts) == "a b"
        assert normalize_cell("a\u3000b", opts) == "a b"

    def test_fold_whitespace_strips_zero_width(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=False, collapse_spaces=False,
            fold_whitespace=True, nfkc=False,
        )
        # ZWSP/ZWNJ/ZWJ/BOM removed entirely.
        assert normalize_cell("a\u200Bb", opts) == "ab"
        assert normalize_cell("a\u200Cb\u200Dc\uFEFFd", opts) == "abcd"

    def test_fold_whitespace_tabs_and_newlines(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=False, collapse_spaces=False,
            fold_whitespace=True, nfkc=False,
        )
        assert normalize_cell("a\tb\nc\rd", opts) == "a b c d"

    def test_nfkc_compatibility_fold(self) -> None:
        # Fullwidth digits → ASCII under NFKC.
        opts = NormalizeOptions(
            enabled=True, strip=False, collapse_spaces=False,
            fold_whitespace=False, nfkc=True,
        )
        assert normalize_cell("\uFF11\uFF12\uFF13", opts) == "123"

    def test_fold_punctuation_off_by_default(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=False, collapse_spaces=False,
            fold_whitespace=False, nfkc=False,
        )
        # Smart quotes preserved when fold_punctuation=False.
        assert normalize_cell("\u201Chi\u201D", opts) == "\u201Chi\u201D"

    def test_fold_punctuation_replaces_typographic_marks(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=False, collapse_spaces=False,
            fold_whitespace=False, nfkc=False, fold_punctuation=True,
        )
        assert normalize_cell("\u2018x\u2019", opts) == "'x'"
        assert normalize_cell("\u201Cy\u201D", opts) == '"y"'
        assert normalize_cell("a\u2014b\u2013c", opts) == "a-b-c"
        assert normalize_cell("e\u2026", opts) == "e..."

    def test_lowercase_destructive_off_by_default(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=False, collapse_spaces=False,
            fold_whitespace=False, nfkc=False,
        )
        assert normalize_cell("CamelCase", opts) == "CamelCase"

    def test_lowercase_when_enabled(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=False, collapse_spaces=False,
            fold_whitespace=False, nfkc=False, lowercase=True,
        )
        assert normalize_cell("CamelCase", opts) == "camelcase"


class TestNormalizeCellComposition:
    def test_default_profile(self) -> None:
        # Default enabled profile: strip + collapse + fold_ws + nfkc.
        opts = NormalizeOptions(enabled=True)
        # NBSP between words + leading tab + trailing newline + double space.
        assert normalize_cell("\thello\u00A0  world\n", opts) == "hello world"

    def test_full_pipeline(self) -> None:
        opts = NormalizeOptions(
            enabled=True, strip=True, collapse_spaces=True,
            fold_whitespace=True, nfkc=True, fold_punctuation=True,
            lowercase=True,
        )
        # Curly quotes + fullwidth digit + NBSP.
        assert normalize_cell(
            "  \u201CFOO\u00A0\uFF11\u201D  ", opts,
        ) == '"foo 1"'

    def test_describe_returns_flag_dict(self) -> None:
        opts = NormalizeOptions(enabled=True, lowercase=True)
        d = opts.describe()
        assert d["enabled"] is True
        assert d["lowercase"] is True
        assert d["fold_punctuation"] is False


# ---------------------------------------------------------------------------
# normalize_frame
# ---------------------------------------------------------------------------


class TestNormalizeFrame:
    def test_disabled_returns_input(self) -> None:
        df = pd.DataFrame({"a": ["  x  "]})
        opts = NormalizeOptions(enabled=False)
        out = normalize_frame(df, opts)
        # Non-copy fast-path: same object back.
        assert out is df

    def test_strips_string_cells_and_headers(self) -> None:
        df = pd.DataFrame({"  Col\u00A0  ": ["  hi\u00A0 ", "x"]})
        opts = NormalizeOptions(enabled=True)
        out = normalize_frame(df, opts)
        # Header normalised.
        assert list(out.columns) == ["Col"]
        # Cells normalised.
        assert list(out["Col"]) == ["hi", "x"]

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"c": ["  hi  "]})
        opts = NormalizeOptions(enabled=True)
        normalize_frame(df, opts)
        assert df["c"].iloc[0] == "  hi  "  # original untouched

    def test_non_string_columns_passthrough(self) -> None:
        df = pd.DataFrame({"n": [1, 2, 3], "s": ["  a  ", "b", "  c"]})
        opts = NormalizeOptions(enabled=True)
        out = normalize_frame(df, opts)
        assert list(out["n"]) == [1, 2, 3]
        assert list(out["s"]) == ["a", "b", "c"]

    def test_non_string_headers_preserved(self) -> None:
        df = pd.DataFrame({0: ["a"], 1: ["b"]})
        opts = NormalizeOptions(enabled=True)
        out = normalize_frame(df, opts)
        assert list(out.columns) == [0, 1]
