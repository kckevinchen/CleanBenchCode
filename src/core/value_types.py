"""Strict scalar-shape predicates shared by detection and labeling."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_PURE_NUMERIC_RE = re.compile(
    r"^[+-]?" r"(?:(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?)|(?:\.\d+))" r"(?:[eE][+-]?\d+)?$"
)
_DECIMAL_COMMA_RE = re.compile(r"^[+-]?" r"(?:(?:\d+(?:\.\d{3})*|),\d+)" r"(?:[eE][+-]?\d+)?$")


def is_pure_numeric_value(value: object) -> bool:
    """Return whether ``value`` is only a finite numeric surface form.

    Valid forms include integers, decimals, scientific notation, and
    conventionally grouped thousands (for example ``6``, ``6.00``,
    ``-1,234.5``, and ``2e3``). Dates, phone numbers, identifiers, percentages,
    and currency-decorated values are deliberately excluded.
    """
    if value is None or isinstance(value, bool):
        return False
    text = str(value).strip()
    return bool(text and _PURE_NUMERIC_RE.fullmatch(text))


def numeric_values_equal(left: object, right: object) -> bool:
    """Return whether two supported numeric surface forms have the same value."""
    left_number = _parse_numeric_value(left)
    right_number = _parse_numeric_value(right)
    return left_number is not None and right_number is not None and left_number == right_number


def _parse_numeric_value(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if _PURE_NUMERIC_RE.fullmatch(text):
        normalized = text.replace(",", "")
    elif _DECIMAL_COMMA_RE.fullmatch(text):
        normalized = text.replace(".", "").replace(",", ".")
    else:
        return None
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None
