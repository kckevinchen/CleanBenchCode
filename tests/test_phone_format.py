"""Unit tests for :mod:`src.labeling.phone_format`.

The module is pure stdlib; tests cover:
  * per-value classification (is_phone_shaped),
  * per-column detection (is_phone_column),
  * template-vote tie-breakers (compute_majority_template),
  * end-to-end column scan (find_phone_corrections).
"""

from __future__ import annotations

import pytest

from src.labeling.phone_format import (
    PhoneCorrection,
    compute_majority_template,
    digits_only,
    find_phone_corrections,
    is_phone_column,
    is_phone_shaped,
    reformat_to_template,
    template_of,
)


# ---------------------------------------------------------------- helpers


def test_digits_only_extracts_only_digits() -> None:
    assert digits_only("(575) 433-0200") == "5754330200"
    assert digits_only("+1 575.433.0200") == "15754330200"
    assert digits_only("") == ""
    assert digits_only("no digits!") == ""


def test_template_of_replaces_digits_with_D() -> None:
    assert template_of("(575) 433-0200") == "(DDD) DDD-DDDD"
    assert template_of("575-433-0200") == "DDD-DDD-DDDD"
    assert template_of("5754330200") == "DDDDDDDDDD"


# ---------------------------------------------------------------- per-value


def test_is_phone_shaped_accepts_standard_formats() -> None:
    """The canonical layouts the user pointed at -- (), -, ., space --
    must all classify as phone-shaped."""
    assert is_phone_shaped("(575) 433-0200")
    assert is_phone_shaped("(575) 4330200")
    assert is_phone_shaped("575-433-0200")
    assert is_phone_shaped("575.433.0200")
    assert is_phone_shaped("5754330200")
    assert is_phone_shaped("+1 575 433 0200")
    assert is_phone_shaped("+44 20 7946 0958")


def test_is_phone_shaped_rejects_obvious_non_phones() -> None:
    assert not is_phone_shaped("")
    assert not is_phone_shaped("123")             # too few digits
    assert not is_phone_shaped("12345678901234567")  # 17 digits, too many
    assert not is_phone_shaped("Acme Corp")       # letters
    assert not is_phone_shaped("1-800-FLOWERS")   # vanity letters
    assert not is_phone_shaped("$575.00")         # currency
    assert not is_phone_shaped("ID#12345678")     # '#'
    assert not is_phone_shaped(None)              # type: ignore[arg-type]


def test_is_phone_shaped_rejects_extensions() -> None:
    """Extensions are a real but separate identity -- the spec says
    leave them alone for manual review."""
    assert not is_phone_shaped("(575) 433-0200 x123")
    assert not is_phone_shaped("(575) 433-0200 ext. 5")
    assert not is_phone_shaped("(575) 433-0200 EXT 999")


# ---------------------------------------------------------------- per-column


def test_is_phone_column_detects_mixed_format_phone_column() -> None:
    """Mostly phone-shaped (with format variation) -> column qualifies."""
    column = [
        "(575) 433-0200",
        "(575) 433-0201",
        "(575) 4330202",     # missing space, will be a correction
        "575-433-0203",      # different layout, will be a correction
        "(575) 433-0204",
        "(575) 433-0205",
    ]
    assert is_phone_column(column)


def test_is_phone_column_rejects_when_not_enough_phones() -> None:
    """Column with mostly names and one number -> not a phone column."""
    column = [
        "Alice", "Bob", "Carol", "Diana", "Edward",
        "(575) 433-0200",
    ]
    assert not is_phone_column(column)


def test_is_phone_column_rejects_with_below_threshold() -> None:
    """50% phone-shaped -> below the 80% threshold."""
    column = [
        "(575) 433-0200",
        "(575) 433-0201",
        "Acme Corp",
        "Beta LLC",
    ]
    assert not is_phone_column(column)


def test_is_phone_column_respects_min_phone_rows() -> None:
    """Even 100% phone-shaped fails if too few rows -- avoids false
    positives on tiny columns where one value happens to look like a
    phone."""
    column = ["(575) 433-0200", "(575) 433-0201"]
    assert not is_phone_column(column, min_phone_rows=5)
    assert is_phone_column(column, min_phone_rows=2)


# ---------------------------------------------------------------- template vote


def test_compute_majority_template_picks_most_frequent() -> None:
    values = [
        "(575) 433-0200",
        "(575) 433-0201",
        "(575) 433-0202",
        "(575) 433-0203",
        "(575) 4330204",   # one stray missing-space variant
    ]
    template, n = compute_majority_template(values)
    assert template == "(DDD) DDD-DDDD"
    assert n == 10


def test_compute_majority_template_tiebreaks_to_more_punctuation() -> None:
    """3 rows in (DDD) DDD-DDDD vs 3 rows in DDDDDDDDDD: the more-
    punctuated template wins because human auditors verify formatted
    forms more reliably."""
    values = [
        "(575) 433-0200",
        "(575) 433-0201",
        "(575) 433-0202",
        "5754330203",
        "5754330204",
        "5754330205",
    ]
    template, _ = compute_majority_template(values)
    assert template == "(DDD) DDD-DDDD"


def test_compute_majority_template_returns_none_on_empty() -> None:
    assert compute_majority_template([]) is None
    assert compute_majority_template(["Alice", "Bob"]) is None


# ---------------------------------------------------------------- reformat


def test_reformat_to_template_fills_digits() -> None:
    assert reformat_to_template("5754330200", "(DDD) DDD-DDDD") == "(575) 433-0200"
    assert reformat_to_template("5754330200", "DDD.DDD.DDDD") == "575.433.0200"
    assert reformat_to_template("5754330200", "DDDDDDDDDD") == "5754330200"


def test_reformat_to_template_returns_none_on_digit_mismatch() -> None:
    """Don't silently pad or truncate -- digit-count mismatch is a
    real data problem, not a format variant."""
    assert reformat_to_template("575433020", "(DDD) DDD-DDDD") is None    # too few
    assert reformat_to_template("57543302001", "(DDD) DDD-DDDD") is None  # too many


# ---------------------------------------------------------------- end-to-end


def test_find_phone_corrections_emits_canonical_for_minority_rows() -> None:
    rows = [
        (0, "(575) 433-0200"),
        (1, "(575) 433-0201"),
        (2, "(575) 433-0202"),
        (3, "(575) 4330203"),         # missing space -> needs reformat
        (4, "575-433-0204"),          # different layout -> needs reformat
        (5, "(575) 433-0205"),
        (6, "(575) 433-0206"),
    ]
    majority, corrections = find_phone_corrections(rows, min_phone_rows=5)
    assert majority == ("(DDD) DDD-DDDD", 10)
    assert {c.row for c in corrections} == {3, 4}
    by_row = {c.row: c for c in corrections}
    assert by_row[3].canonical == "(575) 433-0203"
    assert by_row[3].original == "(575) 4330203"
    assert by_row[4].canonical == "(575) 433-0204"


def test_find_phone_corrections_leaves_digit_mismatch_alone() -> None:
    """7-digit local number in a 10-digit column is a different
    identity -- skip it rather than guess a country code."""
    rows = [
        (0, "(575) 433-0200"),
        (1, "(575) 433-0201"),
        (2, "(575) 433-0202"),
        (3, "(575) 433-0203"),
        (4, "(575) 433-0204"),
        (5, "433-0205"),              # 7 digits only; different identity
    ]
    majority, corrections = find_phone_corrections(rows, min_phone_rows=5)
    assert majority == ("(DDD) DDD-DDDD", 10)
    assert all(c.row != 5 for c in corrections)


def test_find_phone_corrections_returns_empty_when_column_not_phone() -> None:
    """All-letters column: nothing to do."""
    rows = [(i, f"Person {i}") for i in range(10)]
    majority, corrections = find_phone_corrections(rows)
    assert majority is None
    assert corrections == []


def test_find_phone_corrections_skips_clean_rows() -> None:
    """When everything matches the majority template, corrections is
    empty even though the column qualifies."""
    rows = [(i, f"(575) 433-020{i}") for i in range(10)]
    majority, corrections = find_phone_corrections(rows)
    assert majority == ("(DDD) DDD-DDDD", 10)
    assert corrections == []


def test_find_phone_corrections_handles_user_supplied_example() -> None:
    """Mirrors the de1df911.csv example from the gap report:
    `(575) 433 0200` is the column-majority style (with space), and
    `(575) 4330200` is the off-format row needing canonicalisation.
    Note: in the real file ONE row uses each form, so the dataset is
    a tie; the punctuation tie-breaker picks the more-punctuated one."""
    rows = [
        (0, "(575) 433 0200"),
        (1, "(575) 4330200"),
        (2, "(575) 543 0300"),
        (3, "(575) 543 0301"),
        (4, "(575) 543 0302"),
        (5, "(575) 543 0303"),
    ]
    majority, corrections = find_phone_corrections(rows, min_phone_rows=5)
    assert majority is not None
    template, n = majority
    assert n == 10
    # Either the space-form (5 votes) or the missing-space form (1
    # vote) -- the higher count wins.
    assert template == "(DDD) DDD DDDD"
    # Row 1 should be flagged for canonicalisation.
    flagged_rows = {c.row for c in corrections}
    assert 1 in flagged_rows
