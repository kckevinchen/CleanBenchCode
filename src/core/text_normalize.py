"""Deterministic text normalization for LLM input.

Used as an opt-in pre-pass on every CSV cell before it is rendered
into the LLM prompt. The point is to strip the cheap, mechanically-
detectable formatting noise (whitespace quirks, NFC/NFKC mismatches,
smart quotes) so the LLM sees a clean view of the data and we can
measure how much of its detection volume was driven by that noise
vs. genuine semantic / distributional defects.

The default profile mirrors the heuristics in :mod:`src.heuristics`
(``whitespace`` + ``unicode_norm``) so that turning normalization on
is roughly equivalent to "pretend those two heuristics already
cleaned the data". Casing and punctuation are *not* touched by
default - those carry semantic signal the LLM should keep seeing.

The function is a no-op on non-string cells (numeric NaNs, ints
read as numbers, etc.) and on the empty string.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

# Whitespace characters that should collapse to a regular ASCII space.
# Keep this list tight: only the characters that ``unicode_norm`` /
# ``whitespace`` heuristics already complain about.
_WS_TRANSLATE = {
    "\u00A0": " ",  # NBSP
    "\u2007": " ",  # FIGURE SPACE
    "\u202F": " ",  # NARROW NO-BREAK SPACE
    "\u3000": " ",  # IDEOGRAPHIC SPACE
    "\u200B": "",   # ZERO WIDTH SPACE
    "\u200C": "",   # ZWNJ
    "\u200D": "",   # ZWJ
    "\uFEFF": "",   # BOM
    "\t": " ",
    "\r": " ",
    "\n": " ",
}

# Smart / typographic punctuation that should fold to ASCII when
# ``fold_punctuation`` is enabled.
_PUNCT_TRANSLATE = {
    "\u2018": "'", "\u2019": "'",  # curly single
    "\u201C": '"', "\u201D": '"',  # curly double
    "\u2013": "-", "\u2014": "-",  # en/em dash
    "\u2212": "-",                 # minus sign
    "\u2026": "...",               # horizontal ellipsis
}

_MULTI_SPACE_RE = re.compile(r" {2,}")


@dataclass(frozen=True)
class NormalizeOptions:
    """Flags controlling :func:`normalize_cell`.

    The defaults are chosen to be semantically conservative: we strip
    whitespace and Unicode-fold, but DO NOT touch casing, punctuation
    (other than the smart-quote fold above when enabled), or numeric
    formatting. Each flag corresponds to a heuristic detector so
    ablations stay legible.
    """

    enabled: bool = False
    strip: bool = True              # leading/trailing ASCII whitespace
    collapse_spaces: bool = True    # internal "  " -> " "
    fold_whitespace: bool = True    # NBSP/ZWSP/tab/newline -> space
    nfkc: bool = True               # full Unicode NFKC normalization
    fold_punctuation: bool = False  # smart quotes / dashes -> ASCII
    lowercase: bool = False         # destructive; off by default

    def describe(self) -> Mapping[str, bool]:
        return {
            "enabled": self.enabled,
            "strip": self.strip,
            "collapse_spaces": self.collapse_spaces,
            "fold_whitespace": self.fold_whitespace,
            "nfkc": self.nfkc,
            "fold_punctuation": self.fold_punctuation,
            "lowercase": self.lowercase,
        }


def normalize_cell(value: object, opts: NormalizeOptions) -> object:
    """Return a normalized copy of ``value`` per ``opts``.

    Non-string values pass through unchanged. The empty string and
    ``NaN`` also pass through.
    """
    if not opts.enabled:
        return value
    if not isinstance(value, str) or value == "":
        return value

    s = value
    if opts.fold_whitespace:
        # Two-pass translate so multi-codepoint replacements work.
        s = s.translate(str.maketrans({k: v for k, v in _WS_TRANSLATE.items()
                                       if len(v) <= 1}))
        # Multi-char replacements (none today, future-proofing):
        for src, dst in _WS_TRANSLATE.items():
            if len(dst) > 1 and src in s:
                s = s.replace(src, dst)
    if opts.nfkc:
        s = unicodedata.normalize("NFKC", s)
    if opts.fold_punctuation:
        for src, dst in _PUNCT_TRANSLATE.items():
            if src in s:
                s = s.replace(src, dst)
    if opts.collapse_spaces:
        s = _MULTI_SPACE_RE.sub(" ", s)
    if opts.strip:
        s = s.strip()
    if opts.lowercase:
        s = s.lower()
    return s


def normalize_frame(df: pd.DataFrame, opts: NormalizeOptions) -> pd.DataFrame:
    """Return a copy of ``df`` with every string cell normalized.

    No-op (returns the input) when ``opts.enabled`` is ``False``.
    Column headers are normalized with the same rules so downstream
    code that keys by column name stays consistent.
    """
    if not opts.enabled:
        return df
    out = df.copy()
    # Headers
    out.columns = [normalize_cell(c, opts) if isinstance(c, str) else c
                   for c in out.columns]
    # Cells: only object/string columns need the per-cell pass.
    # ``object`` covers legacy text columns; ``string`` covers the
    # modern pandas StringDtype (default in pandas 3.x), which would
    # otherwise be silently skipped here and ship un-normalised text
    # to the LLM prompt.
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(lambda v: normalize_cell(v, opts))
    return out
