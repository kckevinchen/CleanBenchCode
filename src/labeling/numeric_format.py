"""Column-majority numeric thousands-separator canonicalisation.

This module detects columns that hold plain integers / decimals written
in mixed *thousands-separator* styles -- a minority carrying grouping
commas (``328,670``) while the majority does not (``328670``), or vice
versa -- and proposes a single canonical surface form for every
non-conforming cell.

Why a dedicated rule (and why it is SAFE):

  * The comma ``,`` is a **thousands separator**: adding or removing it
    does NOT change a number's value (``328,670`` == ``328670``). So a
    plain-vs-grouped difference in one column is a pure *formatting*
    inconsistency that can be canonicalised deterministically.
  * The characters ``.`` / ``-`` / ``+`` are the OPPOSITE: they DO change
    the value (``2.5`` != ``25``, ``-0.3`` != ``0.3``). This rule never
    touches them -- it only normalises the comma dimension -- so it can
    never merge two genuinely different numbers. That is exactly why the
    generic trivial-inconsistency tagger (which strips ALL punctuation)
    has to exclude numeric collisions, and why this narrower rule can
    safely include them.

The strategy mirrors :mod:`src.labeling.phone_format`:

1. **Column gate.** ``is_numeric_column`` requires >= ``threshold`` of the
   non-empty cells to be clean numbers (plain or thousands-grouped).
2. **Separator-observable subset.** A number reveals the column's
   separator convention only when a separator would actually appear --
   i.e. it already has a grouping comma, or its integer part has >= 4
   digits (``>= 1000``). Numbers below 1000 (``56``, ``789``) are
   separator-NEUTRAL: they conform to either convention, so they neither
   vote nor get flagged.
3. **Majority style vote.** Over the separator-observable numbers,
   ``compute_majority_style`` picks ``"plain"`` vs ``"grouped"`` and
   requires the winner to clear ``majority_threshold`` (else the column
   is a genuine toss-up and is left alone).
4. **Corrections.** Every separator-observable cell whose style differs
   from the majority is reformatted into the majority style, changing
   ONLY the commas.

All functions are pure and depend on nothing beyond the stdlib.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# A clean number with NO thousands separators: optional sign, one or more
# digits, optional decimal part. This is the "plain" surface style.
_PLAIN_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")

# A clean number WITH valid thousands grouping: optional sign, a 1-3 digit
# lead group, then one or more ``,DDD`` groups, optional decimal part. The
# strict ``\d{3}`` groups reject ambiguous shapes such as a European
# decimal comma (``1,2``) or Indian grouping (``3,28,670``), which fall
# through to "not a clean number" and disqualify the cell.
_GROUPED_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")

# Minimum fraction of non-empty cells that must be clean numbers for the
# column to be treated as numeric. Matches the phone rule's 80% gate.
DEFAULT_COLUMN_THRESHOLD = 0.80

# Below this many separator-observable numbers we refuse to vote a
# convention: the majority style is not reliable on a tiny sample.
DEFAULT_MIN_NUMERIC_ROWS = 5

# The dominant style must hold at least this fraction of the
# separator-observable numbers, else the column is a genuine mix with no
# clear canonical convention and is left untouched.
DEFAULT_MAJORITY_THRESHOLD = 0.60

# A plain integer part needs at least this many digits before a thousands
# separator would ever appear (>= 1000). Shorter numbers are neutral.
_MIN_OBSERVABLE_INT_DIGITS = 4


@dataclass(frozen=True)
class NumericCorrection:
    """One cell's canonical-form delta. Emitted by
    :func:`find_numeric_format_corrections` for a separator-observable
    cell whose thousands-separator style differs from the column majority."""

    row: int
    original: str
    canonical: str
    style: str            # this cell's style: "plain" | "grouped"


def numeric_style(value: str) -> Optional[str]:
    """Classify ``value``'s thousands-separator surface style.

    Returns ``"grouped"`` (valid thousands commas), ``"plain"`` (a clean
    number with no commas), or ``None`` when the value is not a clean
    number (letters, currency symbols, ``%``, an ambiguous lone comma,
    dates, etc.). Only the comma dimension is inspected; sign and decimal
    point are allowed but never used to differentiate the style.
    """
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    if _GROUPED_RE.match(v):
        return "grouped"
    if _PLAIN_RE.match(v):
        return "plain"
    return None


def _split_number(v: str) -> Tuple[str, str, str]:
    """Split a clean number into ``(sign, integer_digits, decimal_suffix)``.

    ``integer_digits`` has any commas removed; ``decimal_suffix`` is either
    ``""`` or ``".<digits>"``. Assumes ``v`` already passed
    :func:`numeric_style` (so the shape is valid)."""
    sign = ""
    body = v.strip()
    if body[:1] in "+-":
        sign, body = body[0], body[1:]
    int_part, dot, dec = body.partition(".")
    int_digits = int_part.replace(",", "")
    return sign, int_digits, (dot + dec if dot else "")


def _group_digits(digits: str) -> str:
    """Insert thousands commas every 3 digits from the right, preserving
    the exact digit sequence (including any leading zeros)."""
    n = len(digits)
    parts = [digits[max(0, i - 3):i] for i in range(n, 0, -3)]
    return ",".join(reversed(parts))


def has_observable_style(value: str) -> bool:
    """True iff ``value`` reveals the column's separator convention: it is
    a clean number that is EITHER already grouped (has a comma) OR whose
    integer part is long enough (>= 4 digits) that a separator would
    appear under a grouping convention. Numbers below 1000 are neutral."""
    style = numeric_style(value)
    if style is None:
        return False
    if style == "grouped":
        return True
    _, int_digits, _ = _split_number(value)
    return len(int_digits) >= _MIN_OBSERVABLE_INT_DIGITS


def reformat_to_style(value: str, style: str) -> Optional[str]:
    """Rewrite ``value`` into the target thousands-separator ``style``,
    changing ONLY the commas (sign + decimal part are preserved verbatim).

    Returns ``None`` if ``value`` is not a clean number or already matches
    the target style (no change needed)."""
    cur = numeric_style(value)
    if cur is None or cur == style:
        return None
    sign, int_digits, dec = _split_number(value)
    if style == "plain":
        out = f"{sign}{int_digits}{dec}"
    elif style == "grouped":
        out = f"{sign}{_group_digits(int_digits)}{dec}"
    else:
        return None
    return out if out != value else None


def is_numeric_column(
    values: Sequence[str],
    *,
    threshold: float = DEFAULT_COLUMN_THRESHOLD,
    min_numeric_rows: int = DEFAULT_MIN_NUMERIC_ROWS,
) -> bool:
    """Decide whether a column's non-empty values look like plain numbers.

    Requires the clean-number fraction over non-empty cells to clear
    ``threshold`` AND at least ``min_numeric_rows`` separator-observable
    numbers (so a convention can be voted). A column of only sub-1000
    integers is numeric but has nothing to canonicalise, so it fails the
    observable-rows gate."""
    non_empty = [v for v in values if isinstance(v, str) and v.strip()]
    if not non_empty:
        return False
    numeric = [v for v in non_empty if numeric_style(v) is not None]
    if (len(numeric) / len(non_empty)) < threshold:
        return False
    observable = [v for v in numeric if has_observable_style(v)]
    return len(observable) >= min_numeric_rows


def compute_majority_style(
    values: Sequence[str],
    *,
    majority_threshold: float = DEFAULT_MAJORITY_THRESHOLD,
) -> Optional[Tuple[str, int, int]]:
    """Pick the dominant thousands-separator style over the
    separator-observable numbers in ``values``.

    Returns ``(style, majority_count, observable_total)`` or ``None`` when
    there is no observable number, only one style is present (nothing to
    fix), or the winner fails to clear ``majority_threshold`` (a genuine
    mix with no clear canonical convention)."""
    counter: Counter[str] = Counter()
    for v in values:
        if has_observable_style(v):
            counter[numeric_style(v)] += 1  # type: ignore[index]
    total = sum(counter.values())
    if total == 0 or len(counter) < 2:
        return None
    style, count = counter.most_common(1)[0]
    if (count / total) < majority_threshold:
        return None
    return style, count, total


def find_numeric_format_corrections(
    rows: Sequence[Tuple[int, str]],
    *,
    threshold: float = DEFAULT_COLUMN_THRESHOLD,
    min_numeric_rows: int = DEFAULT_MIN_NUMERIC_ROWS,
    majority_threshold: float = DEFAULT_MAJORITY_THRESHOLD,
) -> Tuple[Optional[Tuple[str, int, int]], List[NumericCorrection]]:
    """End-to-end column scan.

    ``rows`` is an iterable of ``(row_index, value)`` pairs. Returns
    ``(majority, corrections)`` where:

    * ``majority`` is ``(style, majority_count, observable_total)`` for the
      dominant thousands-separator convention, or ``None`` if the column is
      not a numeric-format column or has no clear majority.
    * ``corrections`` is one :class:`NumericCorrection` per
      separator-observable cell whose style differs from the majority,
      carrying the majority-style canonical form (commas-only change).
    """
    values = [v for _, v in rows]
    if not is_numeric_column(
        values, threshold=threshold, min_numeric_rows=min_numeric_rows,
    ):
        return None, []

    majority = compute_majority_style(
        values, majority_threshold=majority_threshold,
    )
    if majority is None:
        return None, []

    dominant_style, _, _ = majority
    corrections: List[NumericCorrection] = []
    for row_idx, v in rows:
        if not has_observable_style(v):
            continue
        style = numeric_style(v)
        if style == dominant_style:
            continue
        canonical = reformat_to_style(v, dominant_style)
        if canonical is None or canonical == v:
            continue
        corrections.append(NumericCorrection(
            row=row_idx, original=v, canonical=canonical, style=style,  # type: ignore[arg-type]
        ))
    return majority, corrections
