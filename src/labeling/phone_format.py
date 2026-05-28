"""Column-majority phone-number format canonicalisation.

This module detects columns that hold phone numbers in mixed surface
formats (``(575) 433-0200`` vs ``575-433-0200`` vs ``5754330200``) and
proposes a single canonical surface form for every non-conforming row.

Two-stage approach:

1. **Per-value classification.** ``is_phone_shaped(v)`` returns True for
   strings whose digits-only form is 7..15 chars long *and* whose only
   non-digit characters belong to the standard phone-punctuation set
   ``[() +.\\-/]`` plus a small allow-list of extension keywords
   (``x``, ``ext``, ``ext.``). Anything with other letters is rejected
   (e.g. ``1-800-FLOWERS`` is left alone -- the literal letters are
   the entity, not a format choice).
2. **Per-column template vote.** ``compute_majority_template(values)``
   builds a ``Counter`` over the *format template* of each phone-shaped
   row (digits replaced with ``D``), and picks a winner with these
   tie-breakers, applied in order against any template that ties the
   max count within 10%:

     a. Most-punctuated template wins (more non-``D`` chars).
     b. Templates whose length is constant across the column win over
        templates that produce variable-length output.
     c. Standard NANP layout ``(DDD) DDD-DDDD`` wins if present.
     d. Lexicographic order over the template string -- last-resort
        deterministic fallback so two labellers always converge.

Once the dominant template is known, ``reformat_to_template`` takes a
row's digits-only identity and reformats it into the template. Rows
that *already* match the template are clean; rows whose digit-count
doesn't fit the template are flagged as ``ambiguous`` and left for
manual review (different identity, not a format variant).

The module is intentionally stdlib-only so it can run inside the
labelling app without adding a new runtime dependency. We deliberately
do *not* depend on ``phonenumbers``: the column-majority strategy
operates purely on the surface format, which is what the benchmark
cares about; validating against ITU-T E.164 is a separate concern.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


# Characters that are allowed as non-digit punctuation in a phone
# number's surface form. Anything else (letters, %, $, comma, etc.)
# disqualifies the row from phone-shaped status.
_PHONE_PUNCT = set("()+-./ \t\u00a0")

# Substrings that mark a number as carrying an extension. Cells with
# extensions are *not* candidates for reformatting because the
# extension is a separate functional identity (we'd need to preserve
# it across the reformat).
_EXT_TOKEN_RE = re.compile(r"\b(?:x|ext\.?)\s*\d+", re.IGNORECASE)

# Standard NANP national layout. Wins ties in step (c) of the
# template-vote tie-breaker so multi-team benchmarks converge on a
# familiar form when several templates are equally common.
_NANP_TEMPLATE = "(DDD) DDD-DDDD"

# Hard limits on the digit-count for a "phone-shaped" string. 7 is the
# minimum local NANP subscriber number; 15 is the ITU-T E.164 maximum.
_MIN_DIGITS = 7
_MAX_DIGITS = 15

# Minimum fraction of non-empty cells that must be phone-shaped for
# the column itself to be considered a phone-format column. Default
# matches the user's spec (>=80%).
DEFAULT_COLUMN_THRESHOLD = 0.80

# Below this many phone-shaped rows we refuse to canonicalise: the
# template vote isn't reliable on tiny samples and false positives
# from mistaking IDs for phones would hurt the benchmark.
DEFAULT_MIN_PHONE_ROWS = 5


@dataclass(frozen=True)
class PhoneCorrection:
    """One row's canonical-form delta. Emitted by
    :func:`find_phone_corrections` for cells whose surface form
    doesn't match the column's dominant template."""

    row: int
    original: str
    canonical: str
    digits: str


def digits_only(value: str) -> str:
    """Strip every non-digit char from ``value``. Used both for
    identity grouping and for the template-fill step."""
    return re.sub(r"\D", "", value or "")


def template_of(value: str) -> str:
    """Surface signature: every digit replaced with the literal ``D``.
    Punctuation and whitespace are preserved verbatim so two strings
    with the same layout collapse to the same key."""
    return re.sub(r"\d", "D", value or "")


def is_phone_shaped(value: str) -> bool:
    """A single value qualifies as phone-shaped iff:

    * it is a non-empty string,
    * it contains no extension marker (``x123`` / ``ext. 5``) - those
      are real but carry a separate identity we don't canonicalise,
    * its digits-only form has 7..15 chars,
    * every non-digit char is in the phone-punctuation set above.

    Letters disqualify the row even if they look like vanity dialling
    (``1-800-FLOWERS``): the user's spec is to leave letters alone
    rather than reformat them away.
    """
    if not isinstance(value, str) or not value:
        return False
    if _EXT_TOKEN_RE.search(value):
        return False
    d = digits_only(value)
    if not (_MIN_DIGITS <= len(d) <= _MAX_DIGITS):
        return False
    for ch in value:
        if ch.isdigit():
            continue
        if ch not in _PHONE_PUNCT:
            return False
    return True


def is_phone_column(
    values: Sequence[str],
    *,
    threshold: float = DEFAULT_COLUMN_THRESHOLD,
    min_phone_rows: int = DEFAULT_MIN_PHONE_ROWS,
) -> bool:
    """Decide whether a column's non-empty values look like phones.

    Requires at least ``min_phone_rows`` phone-shaped rows *and* the
    phone-shaped fraction over non-empty cells to clear ``threshold``.
    """
    non_empty = [v for v in values if isinstance(v, str) and v.strip()]
    if not non_empty:
        return False
    shaped = [v for v in non_empty if is_phone_shaped(v)]
    if len(shaped) < min_phone_rows:
        return False
    return (len(shaped) / len(non_empty)) >= threshold


def _digit_run_lengths(template: str) -> Tuple[int, ...]:
    """Return the lengths of consecutive ``D`` runs in ``template``.
    Used by :func:`reformat_to_template` (and indirectly by the tie-
    breaker for "fixed-length output")."""
    runs: List[int] = []
    current = 0
    for ch in template:
        if ch == "D":
            current += 1
        else:
            if current:
                runs.append(current)
                current = 0
    if current:
        runs.append(current)
    return tuple(runs)


def compute_majority_template(
    values: Iterable[str],
) -> Optional[Tuple[str, int]]:
    """Pick the dominant ``(template, digit_count)`` over the phone-
    shaped rows in ``values``.

    Tie-breakers (applied to any template whose count is within 10% of
    the max and whose digit_count matches the leader's digit_count):

    1. More punctuation chars (more ``len(template) - digits``).
    2. Fixed-length output (same digit_count every row).
    3. NANP standard ``(DDD) DDD-DDDD``.
    4. Lexicographic.

    Returns ``None`` if no phone-shaped row exists.
    """
    counter: Counter[Tuple[str, int]] = Counter()
    for v in values:
        if not is_phone_shaped(v):
            continue
        d = digits_only(v)
        counter[(template_of(v), len(d))] += 1
    if not counter:
        return None

    top_count = max(counter.values())
    cutoff = top_count * 0.90
    contenders = [(t, n) for (t, n), c in counter.items() if c >= cutoff]

    def punctuation_count(t: str) -> int:
        return len(t) - t.count("D")

    def fixed_length_score(t: str, n: int) -> int:
        # Returns 1 if the template fully consumes the digit count
        # with no slack, else 0. Templates whose D-count differs from
        # the digit count would yield None on reformat, so we prefer
        # the ones that line up.
        return 1 if t.count("D") == n else 0

    def nanp_score(t: str) -> int:
        return 1 if t == _NANP_TEMPLATE else 0

    contenders.sort(
        key=lambda tn: (
            -counter[tn],                       # most frequent first
            -punctuation_count(tn[0]),          # more punctuation
            -fixed_length_score(*tn),           # length-fit
            -nanp_score(tn[0]),                 # NANP preference
            tn[0],                              # lexicographic
        )
    )
    return contenders[0]


def reformat_to_template(digits: str, template: str) -> Optional[str]:
    """Fill ``template``'s ``D`` placeholders with ``digits``. Returns
    ``None`` if the digit count doesn't match the template's ``D``
    count exactly: we never silently drop or pad digits."""
    if template.count("D") != len(digits):
        return None
    out: List[str] = []
    i = 0
    for ch in template:
        if ch == "D":
            out.append(digits[i])
            i += 1
        else:
            out.append(ch)
    return "".join(out)


def find_phone_corrections(
    rows: Sequence[Tuple[int, str]],
    *,
    threshold: float = DEFAULT_COLUMN_THRESHOLD,
    min_phone_rows: int = DEFAULT_MIN_PHONE_ROWS,
) -> Tuple[Optional[Tuple[str, int]], List[PhoneCorrection]]:
    """End-to-end column scan.

    ``rows`` is an iterable of ``(row_index, value)`` pairs. Returns
    ``(majority, corrections)`` where:

    * ``majority`` is the dominant ``(template, digit_count)`` pair, or
      ``None`` if the column isn't phone-shaped.
    * ``corrections`` is the list of :class:`PhoneCorrection` for
      every row whose surface form differs from the majority but
      whose digit count matches (so a clean reformat is possible).
    """
    values = [v for _, v in rows]
    if not is_phone_column(
        values, threshold=threshold, min_phone_rows=min_phone_rows,
    ):
        return None, []

    majority = compute_majority_template(values)
    if majority is None:
        return None, []

    dominant_template, dominant_digits = majority
    corrections: List[PhoneCorrection] = []
    for row_idx, v in rows:
        if not is_phone_shaped(v):
            continue
        d = digits_only(v)
        if len(d) != dominant_digits:
            # Different identity (e.g. 7-digit local in a 10-digit
            # column). Leave it alone -- not a format variant.
            continue
        if template_of(v) == dominant_template:
            continue
        canonical = reformat_to_template(d, dominant_template)
        if canonical is None or canonical == v:
            continue
        corrections.append(PhoneCorrection(
            row=row_idx, original=v, canonical=canonical, digits=d,
        ))
    return majority, corrections
