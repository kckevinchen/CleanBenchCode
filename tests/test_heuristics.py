"""Tests for the heuristic detectors.

Each detector is exercised with a hand-built positive + negative
DataFrame so we keep the precision/recall claims in the docstrings
honest.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.heuristics import HeuristicConfig, REGISTRY
from src.heuristics.casing import CasingFingerprintDetector, _casing_fingerprint
from src.heuristics.fd_violation import FdViolationDetector
from src.heuristics.key_collision import KeyCollisionTypoDetector, _fingerprint_key
from src.heuristics.numeric_outlier import NumericOutlierDetector, _coerce_float
from src.heuristics.pattern_minority import PatternMinorityDetector, _signature, _compress
from src.heuristics.unicode_norm import UnicodeNormalizationDetector
from src.heuristics.whitespace import WhitespaceDetector


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_has_seven_detectors() -> None:
    assert {
        "whitespace", "unicode_norm", "casing_fingerprint",
        "pattern_minority", "key_collision_typo",
        "fd_violation", "numeric_outlier",
    }.issubset(set(REGISTRY))


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------


class TestWhitespaceDetector:
    def test_flags_leading_trailing_and_nbsp(self) -> None:
        df = pd.DataFrame({
            "name": ["Alice", "Bob ", " Carol", "Dan", "Erin", "Faye\u00A0"],
        })
        m = WhitespaceDetector().detect(df, source="t.csv")
        rows = sorted({(x.row, x.column) for x in m})
        # Bob ", " Carol", and "Faye\u00A0" all flagged.
        assert (1, "name") in rows
        assert (2, "name") in rows
        assert (5, "name") in rows
        # "Alice" / "Dan" / "Erin" must not fire.
        assert (0, "name") not in rows
        assert (3, "name") not in rows
        assert (4, "name") not in rows

    def test_no_fp_on_clean_data(self) -> None:
        df = pd.DataFrame({
            "x": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
            "y": ["1", "2", "3", "4", "5", "6"],
        })
        assert WhitespaceDetector().detect(df, source="t.csv") == []

    def test_internal_double_space_in_short_value(self) -> None:
        df = pd.DataFrame({
            "tier": ["Tier 1", "Tier 2", "Tier 3", "Tier  4", "Tier 5", "Tier 6"],
        })
        m = WhitespaceDetector().detect(df, source="t.csv")
        assert any(x.row == 3 for x in m)

    def test_zero_width_space_invisible(self) -> None:
        df = pd.DataFrame({"id": ["A1", "A2\u200B", "A3", "A4", "A5", "A6"]})
        m = WhitespaceDetector().detect(df, source="t.csv")
        assert any(x.row == 1 and "ZERO_WIDTH_SPACE" in str(x.meta).upper()
                   or x.row == 1 for x in m)

    def test_fix_normalises_to_clean_string(self) -> None:
        df = pd.DataFrame({"x": ["  hello\u00A0world\t", "y", "z", "w", "v", "u"]})
        m = WhitespaceDetector().detect(df, source="t.csv")
        assert any(x.suggested_fix == "hello world" for x in m)


# ---------------------------------------------------------------------------
# Unicode normalisation
# ---------------------------------------------------------------------------


class TestUnicodeNormDetector:
    def test_flags_smart_quote_and_em_dash(self) -> None:
        df = pd.DataFrame({"text": ["it\u2019s", "x-y", "p", "q", "r", "long\u2014dash"]})
        m = UnicodeNormalizationDetector().detect(df, source="t.csv")
        flagged = {x.row for x in m}
        assert 0 in flagged
        assert 5 in flagged

    def test_flags_decomposed_combining(self) -> None:
        df = pd.DataFrame({"name": ["e\u0301cole", "abc", "def", "ghi", "jkl", "mno"]})
        m = UnicodeNormalizationDetector().detect(df, source="t.csv")
        assert any(x.row == 0 and "nfc_mismatch" in x.meta["signals"] for x in m)

    def test_flags_fullwidth_digits(self) -> None:
        df = pd.DataFrame({"id": ["\uFF11\uFF12\uFF13", "100", "200", "300", "400", "500"]})
        m = UnicodeNormalizationDetector().detect(df, source="t.csv")
        assert any(x.row == 0 and "nfkc_compat" in x.meta["signals"] for x in m)

    def test_no_fp_on_ascii_only(self) -> None:
        df = pd.DataFrame({"x": ["abc", "def", "ghi", "jkl", "mno", "pqr"]})
        assert UnicodeNormalizationDetector().detect(df, source="t.csv") == []


# ---------------------------------------------------------------------------
# Casing fingerprint
# ---------------------------------------------------------------------------


def test_casing_fingerprint_classes() -> None:
    assert _casing_fingerprint("UNITED STATES") == "UPPER"
    assert _casing_fingerprint("united states") == "LOWER"
    assert _casing_fingerprint("United States") == "TITLE"
    assert _casing_fingerprint("iPhone") == "MIXED"
    assert _casing_fingerprint("12345") == "NO_LETTERS"


class TestCasingFingerprintDetector:
    def test_flags_minority_casing(self) -> None:
        df = pd.DataFrame({"city": [
            "New York", "Boston", "Chicago", "Boston", "new york",
            "New York", "Boston", "Chicago", "Boston", "New York",
        ]})
        m = CasingFingerprintDetector().detect(df, source="t.csv")
        assert any(x.row == 4 and x.observed == "new york" for x in m)
        assert all(x.row != 0 for x in m)  # "New York" is the modal

    def test_skips_free_text_columns(self) -> None:
        df = pd.DataFrame({"desc": [f"sentence number {i}" for i in range(20)]})
        # Each value unique -> max_unique_share gate kicks in.
        assert CasingFingerprintDetector().detect(df, source="t.csv") == []


# ---------------------------------------------------------------------------
# Pattern minority
# ---------------------------------------------------------------------------


def test_pattern_signature() -> None:
    assert _signature("(415) 555-1234") == "(999) 999-9999"
    assert _signature("Aa9-X") == "Aa9-A"
    assert _compress("(999) 999-9999") == "(9+) 9+-9+"


class TestPatternMinorityDetector:
    def test_flags_minority_phone_format(self) -> None:
        df = pd.DataFrame({"phone": [
            "415-555-1234", "415-555-1235", "415-555-1236", "415-555-1237",
            "415-555-1238", "415-555-1239", "415-555-1240", "415-555-1241",
            "415-555-1242", "(415) 555-1234",  # the odd one out
        ]})
        m = PatternMinorityDetector().detect(df, source="t.csv")
        assert any(x.row == 9 for x in m)
        # Modal-format rows must not fire.
        assert all(x.row != 0 for x in m)

    def test_no_fp_when_column_too_diverse(self) -> None:
        df = pd.DataFrame({"text": [f"row-{i}-with-text-{i*7}" for i in range(20)]})
        assert PatternMinorityDetector().detect(df, source="t.csv") == []


# ---------------------------------------------------------------------------
# Key-collision typo
# ---------------------------------------------------------------------------


def test_fingerprint_key_examples() -> None:
    assert _fingerprint_key("Mr. John  SMITH") == _fingerprint_key("smith, john mr")
    assert _fingerprint_key("Caf\u00e9") == _fingerprint_key("cafe")


class TestKeyCollisionTypoDetector:
    def test_flags_rare_minor_spelling(self) -> None:
        # 9 Bostons vs 1 "Bsoton" -> typo.
        df = pd.DataFrame({"city": ["Boston"] * 9 + ["Bsoton"]})
        m = KeyCollisionTypoDetector().detect(df, source="t.csv")
        assert any(x.row == 9 and x.suggested_fix == "Boston" for x in m)

    def test_no_fp_when_no_dominant_spelling(self) -> None:
        df = pd.DataFrame({"city": [f"city_{i}" for i in range(15)]})
        assert KeyCollisionTypoDetector().detect(df, source="t.csv") == []


# ---------------------------------------------------------------------------
# FD violation
# ---------------------------------------------------------------------------


class TestFdViolationDetector:
    def test_flags_state_inconsistency(self) -> None:
        df = pd.DataFrame({
            "city": ["Boston", "Boston", "Boston", "Boston",
                     "NYC", "NYC", "NYC", "NYC",
                     "Boston", "Boston"],
            "state": ["MA", "MA", "MA", "MA",
                      "NY", "NY", "NY", "NY",
                      "MA", "NA"],  # last "Boston" mapped to wrong state
        })
        m = FdViolationDetector().detect(df, source="t.csv")
        assert any(x.row == 9 and x.column == "state" for x in m)

    def test_no_fp_when_lhs_is_unique(self) -> None:
        df = pd.DataFrame({
            "id": [str(i) for i in range(20)],
            "state": ["MA"] * 10 + ["NY"] * 10,
        })
        assert FdViolationDetector().detect(df, source="t.csv") == []


# ---------------------------------------------------------------------------
# Numeric outlier
# ---------------------------------------------------------------------------


def test_coerce_float_strips_currency() -> None:
    assert _coerce_float("$1,234.50") == 1234.5
    assert _coerce_float("12%") == 12.0
    assert _coerce_float("--") is None


class TestNumericOutlierDetector:
    def test_flags_extreme_value(self) -> None:
        # 14 small values + 1 very large -> modified-z > 3.5.
        df = pd.DataFrame({"price": [10, 11, 12, 11, 10, 9, 12, 11, 10, 11,
                                     9, 12, 10, 11, 10000]})
        m = NumericOutlierDetector().detect(df, source="t.csv")
        assert any(x.row == 14 for x in m)
        # All small values stay unflagged.
        assert all(x.row == 14 for x in m)

    def test_no_fp_on_constant_column(self) -> None:
        df = pd.DataFrame({"x": [10] * 20})
        assert NumericOutlierDetector().detect(df, source="t.csv") == []
