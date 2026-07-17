"""Deterministic (non-LLM) inconsistency oracle.

Representational inconsistency is defined *structurally*: a value is inconsistent
iff a **peer in the same column is the same thing written differently**. For the
structural sub-patterns (casing, punctuation/whitespace/separators, number
re-encoding) "same thing" = *normalises to the same canonical key*. That
equivalence is computable on ANY column — labeled or not — so this oracle needs
no gold and can label unlimited unlabeled data.

Algorithm (per column): for each normaliser, in precedence order, group the
column's distinct surface forms by ``key = normaliser(surface)``. A key whose
group holds >=2 distinct surfaces is a real inconsistency class; every surface
whose count is a *strict minority* (< the group's max count) is flagged, with the
max-count surface as the canonical correction. Each surface is assigned to the
FIRST (most specific) normaliser under which it joins such a group, so a
case-only difference is labelled ``casing`` rather than ``punctuation``.

By construction the oracle only ever flags a value that has a normalised twin in
its column, so it cannot "invent a convention" on a singleton/near-unique column
— the dominant GPT-5.5 false-positive mode.

Not handled here (the semantic residual, intentionally): ``text`` aliases /
abbreviations / missing-qualifier cases (``USA`` vs ``United States``,
``Acme Intl`` vs ``Acme International``) and date-component reordering. These
need world knowledge a fixed normaliser lacks; the measurement script quantifies
how large this residual is.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from external.baseline.inconsistency._task import norm

# --------------------------------------------------------------------------- #
# Normalisers — each maps a raw surface form to a canonical key (or None when it
# does not apply). Precedence: most specific first.
# --------------------------------------------------------------------------- #

_ALNUM = re.compile(r"[^0-9a-z]+")


def _key_casing(s: str) -> str | None:
    """Group values identical except for letter case (and incidental spaces)."""
    t = norm(s)
    if not any(ch.isalpha() for ch in t):
        return None
    return t.casefold()


def _key_alnum(s: str) -> str | None:
    """Group values equal after casefold + dropping all non-alphanumerics.

    Catches punctuation, whitespace, separator-slip and dotted-acronym diffs
    (``K Lee``/``K. Lee``; ``USA``/``U.S.A.``; ``(415) 555-0162``/``4155550162``).
    """
    t = _ALNUM.sub("", norm(s).casefold())
    return t or None


_CURRENCY = "$€£¥₹"


def _number_markers(s: str) -> tuple[bool, bool, bool, bool]:
    """Return the formatting markers present: (percent, currency, parens, grouping)."""
    t = norm(s)
    pct = t.rstrip().endswith("%")
    cur = any(ch in _CURRENCY for ch in t)
    paren = t.startswith("(") and t.endswith(")")
    grouping = bool(re.search(r"\d,\d", t))  # comma between digits
    return pct, cur, paren, grouping


def _parse_number(s: str) -> float | None:
    """Parse a formatted number to its magnitude, or ``None`` if not numeric.

    Handles thousands separators, currency symbols, leading/trailing ``%``
    (scaled by 1/100), and parenthesised negatives.
    """
    t = norm(s)
    if not t or not any(ch.isdigit() for ch in t):
        return None
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1]
    pct = t.rstrip().endswith("%")
    t = t.replace("%", "")
    t = re.sub(r"[^\d.\-]", "", t)  # drop $, commas, spaces, letters
    if t in ("", "-", ".", "-."):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    if neg:
        v = -abs(v)
    if pct:
        v = v / 100.0
    return v


def _key_number(s: str) -> str | None:
    """Magnitude key (trailing zeros stripped) for number-format re-encodings.

    The precedence loop additionally requires two grouped surfaces to differ in
    their *formatting-marker signature* (``$``, ``%``, parens, comma-grouping),
    so bare integers that merely differ by leading zeros (``0632`` vs ``632`` —
    distinct codes) are NOT treated as a re-encoding of one number.
    """
    v = _parse_number(s)
    if v is None:
        return None
    return format(v, ".12g")


# (kind, key_fn). Order = precedence.
_NORMALISERS: list[tuple[str, Callable[[str], str | None]]] = [
    ("casing", _key_casing),
    ("punctuation", _key_alnum),
    ("number_format", _key_number),
]


def _surface_only_precision_diff(a: str, b: str) -> bool:
    """True if a/b are numbers differing only by trailing-zero precision."""
    na, nb = _ALNUM.sub("", a.casefold()), _ALNUM.sub("", b.casefold())
    return na.rstrip("0").rstrip(".") == nb.rstrip("0").rstrip(".") and na != nb


def oracle_findings(
    values_with_counts: Sequence[tuple[str, int]],
    *,
    min_dominance: float = 0.0,
) -> list[dict[str, Any]]:
    """Return structural-inconsistency findings for one column.

    Each finding: ``{value, corrected_value, kind, rationale}`` (same schema the
    LLM detector emits). ``min_dominance`` optionally requires the canonical form
    to hold at least this fraction of the group before its minorities are flagged
    (0 = flag any strict minority).
    """
    # Distinct surfaces with counts (sum duplicates defensively).
    counts: dict[str, int] = {}
    for v, c in values_with_counts:
        sv = str(v)
        if sv.strip() == "":
            continue
        counts[sv] = counts.get(sv, 0) + int(c)
    surfaces = list(counts)
    total = sum(counts.values()) or 1

    findings: list[dict[str, Any]] = []
    assigned: set[str] = set()

    for kind, key_fn in _NORMALISERS:
        groups: dict[str, list[str]] = {}
        for sv in surfaces:
            if sv in assigned:
                continue
            k = key_fn(sv)
            if k is None:
                continue
            groups.setdefault(k, []).append(sv)

        for _k, members in groups.items():
            if len(members) < 2:
                continue
            # number_format: drop members that differ only by trailing-zero
            # precision from the canonical (not a real re-encoding).
            members_sorted = sorted(members, key=lambda s: (-counts[s], s))
            canonical = members_sorted[0]
            canon_count = counts[canonical]
            if canon_count / total < min_dominance:
                continue
            for sv in members_sorted[1:]:
                if counts[sv] >= canon_count:
                    continue  # not a strict minority (tie) -> skip for precision
                if kind == "number_format":
                    # Require a genuine re-encoding: same magnitude, different
                    # formatting-marker signature (not a leading-zero/precision diff).
                    if _number_markers(sv) == _number_markers(canonical):
                        continue
                    if _surface_only_precision_diff(sv, canonical):
                        continue
                assigned.add(sv)
                findings.append(
                    {
                        "value": sv,
                        "corrected_value": canonical,
                        "kind": kind,
                        "rationale": (
                            f"{kind}: '{sv}' is a minority form of '{canonical}' "
                            f"(same value after {kind} normalisation)."
                        ),
                    }
                )
    return findings


def has_structural_twin(value: str, values_with_counts: Sequence[tuple[str, int]]) -> bool:
    """True if ``value`` normalises equal to some OTHER surface in the column.

    Used to decide whether a (gold) value is structurally catchable at all — the
    basis for splitting recall misses into "structural" vs "semantic (no twin)".
    """
    nv = norm(value)
    others = [str(v) for v, _ in values_with_counts if norm(v) != nv and str(v).strip()]
    for _kind, key_fn in _NORMALISERS:
        k = key_fn(value)
        if k is None:
            continue
        if any(key_fn(o) == k for o in others):
            return True
    return False


# --------------------------------------------------------------------------- #
# Format-signature mode: a value can be inconsistent even with NO exact twin,
# when its *shape* is a minority versus the column's dominant shape (e.g. a lone
# dot-separated phone in a dash-separated column). Restricted to STRUCTURED
# values (low alphabetic content) so it never fires on free-text / name columns.
# --------------------------------------------------------------------------- #

_RUN = re.compile(r"\d+|[A-Za-z]+")


def _shape(s: str) -> str:
    """Format skeleton: digit runs -> 'D', letter runs -> 'A', separators literal."""
    return _RUN.sub(lambda m: "D" if m.group()[0].isdigit() else "A", norm(s))


def _alpha_frac(s: str) -> float:
    t = norm(s)
    letters = sum(ch.isalpha() for ch in t)
    return letters / len(t) if t else 1.0


def dominant_shape(
    values_with_counts: Sequence[tuple[str, int]], *, min_frac: float
) -> str | None:
    """Most common format shape if it covers >= ``min_frac`` of rows, else None."""
    shapes: dict[str, int] = {}
    total = 0
    for v, c in values_with_counts:
        if str(v).strip() == "":
            continue
        shapes[_shape(str(v))] = shapes.get(_shape(str(v)), 0) + int(c)
        total += int(c)
    if not total:
        return None
    top, n = max(shapes.items(), key=lambda kv: kv[1])
    return top if n / total >= min_frac and len(shapes) > 1 else None


def has_minority_format_signature(
    value: str,
    values_with_counts: Sequence[tuple[str, int]],
    *,
    dominance: float = 0.6,
    max_alpha: float = 0.4,
) -> bool:
    """True if ``value`` is a STRUCTURED value whose shape departs from the
    column's dominant shape (and the column is itself predominantly structured).
    """
    if _alpha_frac(value) > max_alpha:
        return False  # mostly letters -> treat as text, not a format pattern
    dom = dominant_shape(values_with_counts, min_frac=dominance)
    if dom is None:
        return False
    # Column must be mostly structured, else a dominant text shape is meaningless.
    struct = sum(c for v, c in values_with_counts if str(v).strip() and _alpha_frac(str(v)) <= max_alpha)
    total = sum(c for v, c in values_with_counts if str(v).strip() != "")
    if total and struct / total < 0.6:
        return False
    return _shape(value) != dom


def is_structurally_justified(
    value: str,
    values_with_counts: Sequence[tuple[str, int]],
    *,
    dominance: float = 0.6,
) -> bool:
    """Verifier predicate: is flagging ``value`` structurally defensible?

    True when EITHER a normalised twin exists (same thing written two ways) OR
    the value is a structured minority-format-signature. This is the
    high-precision, gold-free signal a precision-focused post-trained model can
    be rewarded for agreeing with: a flag that is *not* justified is, with high
    probability, the "invented-convention" false-positive mode.
    """
    return has_structural_twin(value, values_with_counts) or has_minority_format_signature(
        value, values_with_counts, dominance=dominance
    )


def make_oracle_detector(**kwargs: Any) -> Callable[[Any, str, str, Sequence[tuple[str, int]]], str]:
    """Adapt the oracle to the ``Detector`` interface used by ``evaluator.evaluate``.

    Signature ``(prompt, source, column, values) -> raw_json``; ``prompt`` is
    ignored. Lets the oracle reuse the exact scoring / exclusion / export path
    the LLM detector uses.
    """

    def detector(_prompt, _source, _column, values):
        return json.dumps({"findings": oracle_findings(values, **kwargs)})

    return detector
