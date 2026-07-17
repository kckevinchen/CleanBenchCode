from __future__ import annotations

import pytest

from src.core.value_types import is_pure_numeric_value, numeric_values_equal


@pytest.mark.parametrize(
    "value",
    [
        0,
        6.0,
        "6",
        " 6.00 ",
        "+.5",
        "-1,234.5",
        "1.",
        "2e3",
    ],
)
def test_is_pure_numeric_value_accepts_numeric_surfaces(value: object) -> None:
    assert is_pure_numeric_value(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        "",
        float("nan"),
        float("inf"),
        "555-123-4567",
        "2024-01-01",
        "6%",
        "$6",
        "ABC123",
        "1,23",
    ],
)
def test_is_pure_numeric_value_rejects_decorated_or_non_numeric_values(
    value: object,
) -> None:
    assert not is_pure_numeric_value(value)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("6", "6.00"),
        ("1000", "1,000"),
        ("199,5", "199,50"),
        ("1.234,5", "1.234,50"),
    ],
)
def test_numeric_values_equal_accepts_equivalent_presentations(
    left: object,
    right: object,
) -> None:
    assert numeric_values_equal(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("12.50", "12"),
        ("1133,21", "1133, 21"),
        ("555-123-4567", "5551234567"),
        ("2024-01-01", "20240101"),
    ],
)
def test_numeric_values_equal_rejects_content_and_decorated_values(
    left: object,
    right: object,
) -> None:
    assert not numeric_values_equal(left, right)
