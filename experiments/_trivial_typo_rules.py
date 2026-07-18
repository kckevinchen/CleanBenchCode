"""Trivial-typo deterministic rules for v4 audit.

The tracked typo policy in ``prompts/error_types.yaml``
authorises only ONE class of cells to be auto-flagged as
"Typo-but-trivial": **repeated / padding whitespace** (e.g.
``" James Smith"``, ``"James  Smith"``). The codebook lists
punctuation and casing as REAL Typos, not trivial.

Accordingly, only the two whitespace rules are active. The remaining
rules below (space-after-comma, footnote stripping, company-period,
phone canonicalisation, case canonicalisation, hyphen spacing) are
kept as documentation of the broader rule taxonomy but are NOT in
``RULES`` -- ``detect_trivial_typo`` must not fire them, or the audit
would over-flag real Typos as trivial.

Rules are intentionally conservative: each fires only when the fix is
unambiguous from the input alone.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

Rule = Callable[[str], Optional[tuple[str, str]]]


def _strip_padding_whitespace(s: str) -> Optional[tuple[str, str]]:
    """Strip leading/trailing whitespace."""
    if s != s.strip():
        return s.strip(), "strip leading/trailing whitespace"
    return None


def _collapse_internal_whitespace(s: str) -> Optional[tuple[str, str]]:
    """Normalise internal whitespace to a single ASCII space.

    Fires when:
      - the value has a run of 2+ whitespace characters anywhere, OR
      - the value contains any non-space whitespace (newline, tab,
        carriage return, non-breaking space, etc.) inside the string.
    Either case canonicalises every whitespace run to a single ' '.
    """
    has_run = bool(re.search(r"\s{2,}", s))
    inner = s.strip()
    has_nonspace_ws = bool(re.search(r"[^\S ]", inner))
    if not (has_run or has_nonspace_ws):
        return None
    out = re.sub(r"\s+", " ", s).strip()
    if out != s.strip():
        return out, "normalise internal whitespace to single ASCII space"
    return None


def _space_after_comma(s: str) -> Optional[tuple[str, str]]:
    """Insert a space after a comma when followed by a letter or digit.

    Conservative: do not fire on numbers like '1,000' (digit-comma-digit
    is fine) -- only act when the character before the comma is a letter
    (or end-of-token) AND the character after the comma is a non-space
    letter or digit.
    """
    if not re.search(r"[A-Za-z],[A-Za-z0-9]", s):
        return None
    out = re.sub(r"([A-Za-z]),([A-Za-z0-9])", r"\1, \2", s)
    if out != s:
        return out, "insert space after comma"
    return None


def _strip_trailing_footnote(s: str) -> Optional[tuple[str, str]]:
    """Strip a trailing run of footnote markers (*, dagger, double-dagger)."""
    m = re.search(r"[\*\u2020\u2021]+\s*$", s)
    if m:
        out = re.sub(r"[\*\u2020\u2021]+\s*$", "", s).rstrip()
        if out != s:
            return out, "strip trailing footnote markers"
    return None


def _append_company_period(s: str) -> Optional[tuple[str, str]]:
    """Append a terminal period to a trailing 'Inc' / 'Co' / 'Ltd' token."""
    m = re.search(r"\b(Inc|Co|Ltd|Corp)\s*$", s)
    if m:
        return s.rstrip() + ".", "append terminal period to company abbreviation"
    return None


def _phone_canonicalise(s: str) -> Optional[tuple[str, str]]:
    """Canonicalise NANP phone numbers to NNN-NNN-NNNN.

    Only fires when the value parses as exactly 10 NANP digits with
    common separators (space, hyphen, parentheses, dot). Conservative:
    skips anything that looks like an extension or country prefix.
    """
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) != 10:
        return None
    canonical = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if canonical == s:
        return None
    # only fire when the input looks "phone-shaped" (has at least one
    # separator from {space, -, (, ), .})
    if not re.search(r"[\s\-().]", s):
        return None
    return canonical, "phone-number canonicalisation to NNN-NNN-NNNN"


def _hyphen_spacing(s: str) -> Optional[tuple[str, str]]:
    """Add spaces around an internal hyphen when both sides look like
    multi-letter words and neither contains other hyphens (heuristic).

    Disabled by default -- hyphen-spacing is a policy choice and the
    rulebook explicitly says it is upstream, not for the LLM. We do not
    fire this rule against the benchmark; it is included only so the
    rule taxonomy is complete.
    """
    return None


def _case_canonicalise(s: str) -> Optional[tuple[str, str]]:
    """Case canonicalisation -- policy-dependent, no auto-fix here.

    The taxonomy entry exists so that the auditor knows casing is in
    scope for upstream, but we cannot canonicalise without knowing the
    column-level policy (Title Case vs UPPER vs source-preserved).
    """
    return None


# Only whitespace rules are authorised by the codebook as trivial.
# The remaining rules (space_after_comma, strip_trailing_footnote,
# append_company_period, phone_canonicalise) are punctuation-class
# edits that the codebook explicitly calls REAL Typos, so they are
# intentionally excluded from ``RULES``.
RULES: list[tuple[str, Rule]] = [
    ("strip_padding_whitespace", _strip_padding_whitespace),
    ("collapse_internal_whitespace", _collapse_internal_whitespace),
    # disabled (real Typos per codebook, not trivial):
    # ("space_after_comma", _space_after_comma),
    # ("strip_trailing_footnote", _strip_trailing_footnote),
    # ("append_company_period", _append_company_period),
    # ("phone_canonicalise", _phone_canonicalise),
    # disabled / policy-only:
    # ("hyphen_spacing", _hyphen_spacing),
    # ("case_canonicalise", _case_canonicalise),
]


def detect_trivial_typo(value: str) -> Optional[tuple[str, str, str]]:
    """Return (rule_name, canonical, reason) for the first matching rule,
    or None if no rule fires.
    """
    for name, rule in RULES:
        out = rule(value)
        if out is not None:
            canonical, reason = out
            return name, canonical, reason
    return None
