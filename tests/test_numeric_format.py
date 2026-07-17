"""Unit tests for :mod:`src.labeling.numeric_format`.

The module is pure stdlib. It flags *thousands-separator* inconsistencies
(``328,670`` among ``328670``) while never touching the ``.`` / ``-`` / ``+``
dimensions, so genuinely different numbers (``2.5`` vs ``25``) can never be
merged. Tests cover:

  * per-value style classification (numeric_style),
  * the separator-observable gate (has_observable_style),
  * commas-only reformatting (reformat_to_style),
  * per-column detection + majority vote (is_numeric_column /
    compute_majority_style),
  * end-to-end column scan (find_numeric_format_corrections),
  * the safety cases the whole rule exists for.
"""

from __future__ import annotations

from src.labeling.numeric_format import (
    NumericCorrection,
    compute_majority_style,
    find_numeric_format_corrections,
    has_observable_style,
    is_numeric_column,
    numeric_style,
    reformat_to_style,
)


# ------------------------------------------------------ style classification


def test_numeric_style_plain_vs_grouped() -> None:
    assert numeric_style("328670") == "plain"
    assert numeric_style("328,670") == "grouped"
    assert numeric_style("2,753,561") == "grouped"
    assert numeric_style("-1,234") == "grouped"
    assert numeric_style("+1234") == "plain"
    assert numeric_style("1,234.56") == "grouped"
    assert numeric_style("1234.56") == "plain"


def test_numeric_style_rejects_non_numbers() -> None:
    # Letters, currency, percent, dates -> not a clean number.
    for v in ("N/A", "$1,234", "15%", "8/29/2020", "abc", "", "  "):
        assert numeric_style(v) is None


def test_numeric_style_rejects_ambiguous_comma_grouping() -> None:
    # A lone comma that is NOT a valid 3-digit thousands group is ambiguous
    # (European decimal comma / Indian grouping) -> not classified.
    assert numeric_style("1,2") is None
    assert numeric_style("12,34") is None
    assert numeric_style("3,28,670") is None


# ------------------------------------------------------ observable gate


def test_has_observable_style_below_1000_is_neutral() -> None:
    # A plain sub-1000 number cannot carry a separator -> neutral.
    assert has_observable_style("56") is False
    assert has_observable_style("789") is False
    assert has_observable_style("999") is False
    # >= 1000 (4+ integer digits) is observable.
    assert has_observable_style("1000") is True
    assert has_observable_style("328670") is True
    # A grouped number is always observable (it has a comma).
    assert has_observable_style("1,000") is True


def test_has_observable_style_decimal_uses_integer_part() -> None:
    assert has_observable_style("999.99") is False   # int part 3 digits
    assert has_observable_style("1234.5") is True     # int part 4 digits


# ------------------------------------------------------ reformatting


def test_reformat_to_plain_strips_only_commas() -> None:
    assert reformat_to_style("328,670", "plain") == "328670"
    assert reformat_to_style("2,753,561", "plain") == "2753561"
    assert reformat_to_style("-1,234.50", "plain") == "-1234.50"
    # Already plain -> no change.
    assert reformat_to_style("328670", "plain") is None


def test_reformat_to_grouped_adds_thousands_commas() -> None:
    assert reformat_to_style("3210", "grouped") == "3,210"
    assert reformat_to_style("328670", "grouped") == "328,670"
    assert reformat_to_style("-1234.5", "grouped") == "-1,234.5"
    # Already grouped -> no change.
    assert reformat_to_style("3,210", "grouped") is None


def test_reformat_preserves_sign_and_decimal() -> None:
    # Only the comma dimension changes; sign + decimal are verbatim.
    assert reformat_to_style("+12345.678", "grouped") == "+12,345.678"
    assert reformat_to_style("-9,999", "plain") == "-9999"


# ------------------------------------------------------ column detection


def test_is_numeric_column_requires_numbers_and_observable_rows() -> None:
    plain = ["383481", "400262", "369435", "429452", "299884", "328,670"]
    assert is_numeric_column(plain) is True
    # Mostly text -> not numeric.
    assert is_numeric_column(["a", "b", "c", "d", "e", "328,670"]) is False
    # All sub-1000 -> numeric but nothing observable to canonicalise.
    assert is_numeric_column(["1", "2", "3", "4", "5", "6"]) is False


def test_compute_majority_style_needs_two_styles_and_a_majority() -> None:
    # Clear majority plain, one grouped.
    vals = ["383481", "400262", "369435", "429452", "299884", "328,670"]
    maj = compute_majority_style(vals)
    assert maj is not None
    assert maj[0] == "plain"

    # Single style -> nothing to fix.
    assert compute_majority_style(["1000", "2000", "3000", "4000"]) is None

    # 50/50 toss-up -> no clear canonical (default 0.6 threshold).
    tie = ["1,000", "2,000", "3000", "4000"]
    assert compute_majority_style(tie) is None


# ------------------------------------------------------ end-to-end scan


def _scan(values):
    return find_numeric_format_corrections(list(enumerate(values)))


def test_scan_flags_grouped_minority_in_plain_column() -> None:
    values = ["383481", "400262", "369435", "429452", "299884",
              "328,670", "291132", "300000"]
    majority, corrections = _scan(values)
    assert majority is not None and majority[0] == "plain"
    assert len(corrections) == 1
    c = corrections[0]
    assert isinstance(c, NumericCorrection)
    assert c.row == 5
    assert c.original == "328,670"
    assert c.canonical == "328670"
    assert c.style == "grouped"


def test_scan_flags_plain_minority_in_grouped_column() -> None:
    values = ["1,000", "2,345", "10,000", "3210", "9,999", "4,567", "7,777"]
    majority, corrections = _scan(values)
    assert majority is not None and majority[0] == "grouped"
    assert [c.original for c in corrections] == ["3210"]
    assert corrections[0].canonical == "3,210"


def test_scan_ignores_decimals_vs_integers() -> None:
    # SAFETY: 2.5 and 25 are both "plain" style -> single style -> nothing
    # flagged. The rule must never merge different numbers.
    majority, corrections = _scan(["2.5", "25", "3.7", "41", "5.0", "6.0"])
    assert majority is None
    assert corrections == []


def test_scan_ignores_sign_differences() -> None:
    majority, corrections = _scan(["-0.3", "0.3", "-1.2", "4.5", "6.7", "-8"])
    assert majority is None
    assert corrections == []


def test_scan_leaves_sub_1000_neutral_cells_alone() -> None:
    # Large numbers all plain; the sub-1000 values are neutral and never
    # flagged, and there is no grouped deviant, so nothing is written.
    values = ["1234", "5678", "56", "789", "9012", "3456", "12"]
    majority, corrections = _scan(values)
    assert corrections == []


def test_scan_preserves_decimal_when_grouping() -> None:
    values = ["1,234.50", "2,345.10", "3456.7", "1,111.1", "2,222.2", "3,333.3"]
    majority, corrections = _scan(values)
    assert majority is not None and majority[0] == "grouped"
    assert [c.original for c in corrections] == ["3456.7"]
    # Only the comma is added; the decimal is preserved verbatim.
    assert corrections[0].canonical == "3,456.7"
