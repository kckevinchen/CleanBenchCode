"""Leaf helpers shared by the labelling app factory and its routers.

Kept light (stdlib + fastapi only) so both ``app.py`` and the
``routers/`` modules can import from here without a circular import.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, List, Optional, Tuple

from fastapi import HTTPException


def _scalar(v: Any) -> str:
    """Coerce a (possibly ``None``) cell value to a plain string.

    Used everywhere a CSV cell is compared or rendered so equality
    matches the operator's mental model ("this value equals that").
    """
    if v is None:
        return ""
    return str(v)


# ---------------------------------------------------------------------------
# Trivial whitespace anomaly detection + deterministic fix. Shared by the
# trivial-tagger endpoint (``routers/propagation.py``) AND the benchmark
# exporter (``export.py``) so a TRIVIAL cell's fix is computed the SAME way
# in both places -- the export uses it to recompute a per-cell fix rather
# than inherit a sibling row's (wrong) correction on a multi-row cluster.
# ---------------------------------------------------------------------------

# A genuine WHITESPACE anomaly. Deliberately STRICT -- two classes only:
#   1. edge or 2+ internal ASCII spaces (e.g. ``Elvira  Bowman``);
#   2. tab / NBSP / line breaks / zero-width / soft hyphen / BOM.
# No comma / bracket / casing reformats: those are presentation
# conventions in tabular data, not verifiable defects. Mirrored in
# ``frontend/labeling/label.js`` as ``_WS_MARKER_RE``.
_WHITESPACE_ANOMALY_RE = re.compile(
    r"^ | $|  +"
    r"|[\t\u00A0\n\r\u2028\u2029\u200B\u200C\u200D\u00AD\uFEFF]"
)


def _has_whitespace_anomaly(value: str) -> bool:
    if not value:
        return False
    return _WHITESPACE_ANOMALY_RE.search(value) is not None


# Mechanical fix for a whitespace anomaly: drop zero-width chars, fold tab /
# NBSP / line breaks to a space, collapse internal runs, strip the edges.
# Whitespace ONLY -- no comma / bracket / casing reformats.
_ZW_DROP = re.compile(r"[\u200B\u200C\u200D\u00AD\uFEFF]")
_CONTROL_TO_SPACE = re.compile(r"[\t\n\r\u2028\u2029]")
_NBSP_TO_SPACE = re.compile(r"[\u00A0]")
_MULTI_SPACE = re.compile(r" {2,}")


def _normalize_trivial(value: str) -> str:
    s = _ZW_DROP.sub("", value)
    s = _CONTROL_TO_SPACE.sub(" ", s)
    s = _NBSP_TO_SPACE.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Trivial PUNCTUATION anomaly detection + deterministic fix. The punctuation
# analogue of the whitespace pair above: a single-cell, value-alone rule whose
# fix is unambiguous from the value itself (no column context, no model call).
# Shared by the trivial-punctuation tagger (``routers/propagation.py``) and the
# benchmark exporter (``export.py``) so the fix is computed identically in both.
#
# Tier-1 rules ONLY (deliberately conservative -- each fix is self-evident):
#   1. Unicode -> ASCII fold: smart quotes, ellipsis, the U+2212 minus sign.
#   2. Collapse doubled comma / semicolon (``a,,b`` -> ``a,b``).
#   3. Remove padding space just inside a bracket (``( x`` -> ``(x``).
#   4. Remove a stray space before closing punctuation (``word ,`` -> ``word,``).
#   5. Insert the missing space before ``(`` (``word(x)`` -> ``word (x)``),
#      guarded so function calls ``f(x)``, sizes ``varchar(255)``, citations
#      ``Smith(2020)`` and formulae ``H2O(l)`` are left alone.
#   6. Insert the missing space after ``)`` (``(x)word`` -> ``(x) word``).
#   7. Insert the missing space after a comma (``a,b`` -> ``a, b``), guarded so
#      grouped numbers ``1,000`` are left alone.
#   8. Tighten ASYMMETRIC slash spacing (``A/ B`` or ``A /B`` -> ``A/B``).
#      Symmetric spacing (``A / B``, a deliberate style like ``Sales / Marketing``)
#      and a slash with no adjacent space are left alone.
#   9. Space a word-joining ampersand (``Black&Decker`` -> ``Black & Decker``,
#      also fixing the asymmetric ``Black &Decker`` / ``Black& Decker``). A
#      "word" must carry a lowercase letter, so SHORT all-caps acronyms keep
#      their conventional tight form (``AT&T``, ``R&D``, ``CF&CE``, ``H&M``).
#
# Pointedly does NOT touch general whitespace (leading / trailing / internal
# runs, tab / NBSP / zero-width): that is the whitespace rule's job, so the two
# taggers stay orthogonal. En / em dashes are NOT folded -- an en-dash range
# like ``1990-95`` is often correct, a Tier-3 judgement rather than trivial.
# Idempotent: applying the fix to its own output is a no-op.
# ---------------------------------------------------------------------------

# 1. Typographic Unicode punctuation that is an unambiguous variant of an ASCII
# mark. Single-char substitutions go through ``str.translate``; the ellipsis
# expands to three dots so it needs its own pass.
_PUNCT_UNICODE_MAP = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",   # single quotes
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',   # double quotes
    0x2212: "-",                                           # minus sign
}
_PUNCT_ELLIPSIS = re.compile("\u2026")                     # ... (single glyph)

# 2. Doubled comma / semicolon -> single. (NOT . ! ? -- ellipsis / emphasis.)
_PUNCT_DOUBLED = re.compile(r"([,;])\1+")

# 3/4. Space just inside an opening bracket, or before closing punctuation.
# Uses [ \t] (not \s) so newlines stay the whitespace rule's concern.
_PUNCT_SPACE_AFTER_OPEN = re.compile(r"([(\[{])[ \t]+")
_PUNCT_SPACE_BEFORE_CLOSE = re.compile(r"[ \t]+([,;:.!?)\]}])")

# 5. Missing space before ``(``: at least two letters before AND a letter right
# after, so numeric parentheticals (sizes, years, counts) are skipped. A final
# negative lookahead also skips the optional-plural / inflection idiom
# ``word(s)`` / ``POUND(S)`` / ``box(es)`` / ``categor(ies)`` -- there the
# parenthetical is a grammatical suffix, never a missing space.
_PUNCT_MISSING_SPACE_BEFORE_PAREN = re.compile(
    r"(?<=[A-Za-z]{2})\((?=[A-Za-z])(?!(?:[sS]|[eE][sS]|[iI][eE][sS])\))"
)

# 6. Missing space after ``)``: a letter on each side of the bracket.
_PUNCT_MISSING_SPACE_AFTER_PAREN = re.compile(r"(?<=[A-Za-z])\)(?=[A-Za-z])")

# 7. Missing space after a comma: not already followed by whitespace, and not a
# digit (``1,000`` is a grouped number, not a list).
_PUNCT_MISSING_SPACE_AFTER_COMMA = re.compile(r",(?=[^\s\d])")

# 8. Asymmetric space around a slash: a space on exactly ONE side of ``/``
# (``A/ B`` or ``A /B``) is a typo -> tighten to ``A/B``. The two patterns
# never match symmetric ``A / B`` (a deliberate style such as ``Sales /
# Marketing`` or ``5 / 5``): the "space after" pattern requires a NON-space
# immediately before the slash, the "space before" pattern requires a NON-space
# immediately after it, so a slash padded on both sides satisfies neither.
# ``[ \t]`` (not ``\s``) keeps newlines the whitespace rule's concern.
_PUNCT_SLASH_SPACE_AFTER = re.compile(r"(?<=\S)/[ \t]+")
_PUNCT_SLASH_SPACE_BEFORE = re.compile(r"[ \t]+/(?=\S)")


def _step_slash_spacing(s: str) -> str:
    s = _PUNCT_SLASH_SPACE_AFTER.sub("/", s)
    s = _PUNCT_SLASH_SPACE_BEFORE.sub("/", s)
    return s


# 9. Ampersand joining two WORDS with missing / asymmetric spacing:
# ``Word&Word`` / ``Word &Word`` / ``Word& Word`` -> ``Word & Word``. A "word"
# is a letter token that carries a lowercase letter; a short all-caps acronym /
# initialism (``AT``, ``T``, ``R``, ``D``, ``CF``, ``CE``, ``H``, ``M`` ...) is
# NOT a word and keeps its conventional tight ``&`` (``AT&T``, ``R&D``,
# ``CF&CE``, ``H&M``, ``B&B``). Both sides must be words, so half-acronym joins
# (``Smith&CO``, ``J&Johnson``) and code-like joins (``1320C&E``, ``1&1``) are
# left alone too. The right token is matched via a lookahead so it is not
# consumed -- a chain ``a&b&c`` is fully spaced in one pass. ``[ \t]?`` (0 or 1)
# on each side keeps multi-space runs the whitespace rule's concern and makes an
# already-correct ``Word & Word`` a no-op (idempotent).
_PUNCT_AMPERSAND = re.compile(r"([A-Za-z]+)[ \t]?&[ \t]?(?=([A-Za-z]+))")


def _amp_is_word(tok: str) -> bool:
    """True iff ``tok`` (a letter-only token) carries a lowercase letter.

    The ampersand rule's word-vs-acronym test: ``"Town" != "TOWN"`` is True (a
    word) while ``"AT" == "AT"`` makes ``AT`` an acronym (not a word).
    """
    return tok != tok.upper()


def _step_ampersand_spacing(s: str) -> str:
    def _repl(m: "re.Match[str]") -> str:
        left, right = m.group(1), m.group(2)
        if _amp_is_word(left) and _amp_is_word(right):
            return f"{left} & "
        # Not a word-join (acronym / code / half-acronym): leave the consumed
        # ``left[ ]?&[ ]?`` exactly as it was.
        return m.group(0)
    return _PUNCT_AMPERSAND.sub(_repl, s)


# Email / URL / web-address guard. The trivial-punctuation tagger must NEVER
# touch a cell that contains an e-mail address or a web URL: their dots,
# commas, slashes and parentheses are STRUCTURAL (``Foo(bar)`` inside a wiki
# URL, comma-joined domain / address lists, ``a@b.com``), so "fixing" them
# would corrupt the value. Any cell containing such a token is skipped
# wholesale by both the detector and the fix (see ``_contains_email_or_url``).
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Common TLDs for bare hosts (``example.com`` with no scheme / path). Kept to a
# curated list so abbreviations like ``U.S.`` / ``e.g.`` / ``Mr.Smith`` are NOT
# mistaken for domains. URLs with a scheme (``https://``), a ``www.`` prefix, or
# a ``host.tld/path`` slash-path are matched structurally and need no TLD list.
_URL_TLDS = (
    "com|org|net|edu|gov|mil|int|io|co|ai|app|dev|info|biz|name|xyz|online|"
    "site|tech|me|tv|us|uk|ca|de|fr|jp|cn|au|nl|ru|ch|it|es|se|no|fi|dk|be|"
    "at|in|pl|br|mx|kr|za|ie|nz|sg|hk|eu"
)
_URL_RE = re.compile(
    r"(?:https?|ftp|ftps)://\S+"                       # scheme://...
    r"|\bwww\.\S+"                                     # www.example.com/...
    r"|\b[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}/\S*"   # host.tld/path
    r"|\b[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.(?:" + _URL_TLDS + r")\b",  # bare host
    re.IGNORECASE,
)


def _contains_email_or_url(value: str) -> bool:
    """True iff ``value`` contains an e-mail address or a web URL anywhere.

    Used to exempt such cells from the punctuation tagger entirely (their
    punctuation is structural, not a defect).
    """
    if not value:
        return False
    return _EMAIL_RE.search(value) is not None or _URL_RE.search(value) is not None


def _step_unicode_fold(s: str) -> str:
    return _PUNCT_ELLIPSIS.sub("...", s.translate(_PUNCT_UNICODE_MAP))


# The Tier-1 rules as an ORDERED list of named ``(rule, fn)`` steps. Keeping
# them ordered + named (rather than an inline sequence of ``.sub`` calls) lets
# us (a) reproduce the exact original fix when every step runs, and (b) ask
# which individual rules fired on a value -- the signal the column-convention
# guard keys on (see ``_punctuation_column_findings``).
_PUNCT_STEPS: list[tuple[str, Any]] = [
    ("unicode_fold", _step_unicode_fold),
    ("doubled", lambda s: _PUNCT_DOUBLED.sub(r"\1", s)),
    ("space_after_open", lambda s: _PUNCT_SPACE_AFTER_OPEN.sub(r"\1", s)),
    ("space_before_close", lambda s: _PUNCT_SPACE_BEFORE_CLOSE.sub(r"\1", s)),
    ("missing_space_before_paren",
     lambda s: _PUNCT_MISSING_SPACE_BEFORE_PAREN.sub(" (", s)),
    ("missing_space_after_paren",
     lambda s: _PUNCT_MISSING_SPACE_AFTER_PAREN.sub(") ", s)),
    ("missing_space_after_comma",
     lambda s: _PUNCT_MISSING_SPACE_AFTER_COMMA.sub(", ", s)),
    ("slash_spacing", _step_slash_spacing),
    ("ampersand_spacing", _step_ampersand_spacing),
]


def _normalize_punctuation(value: str, skip: Iterable[str] = ()) -> str:
    """Deterministic Tier-1 punctuation fix (see section comment above).

    ``skip`` names rules to omit -- used by the column-convention guard so a
    cell's fix never "corrects" the punctuation pattern shared by its whole
    column. Returns ``value`` unchanged when no (non-skipped) rule applies.

    Cells containing an e-mail address or URL are returned UNCHANGED: their
    punctuation is structural (see ``_contains_email_or_url``).
    """
    if not value:
        return value
    if _contains_email_or_url(value):
        return value
    skip_set = frozenset(skip)
    s = value
    for name, step in _PUNCT_STEPS:
        if name in skip_set:
            continue
        s = step(s)
    return s


def _punctuation_rules_fired(value: str) -> frozenset[str]:
    """The set of Tier-1 rule names that change ``value`` (applied in order).

    Each step sees the output of the previous one, matching how the fix is
    actually built, so the reported set is exactly the rules that contribute
    to ``_normalize_punctuation(value)``.

    Returns an empty set for a cell containing an e-mail address or URL (its
    punctuation is structural, so no rule may fire).
    """
    if not value or _contains_email_or_url(value):
        return frozenset()
    fired: set[str] = set()
    s = value
    for name, step in _PUNCT_STEPS:
        nxt = step(s)
        if nxt != s:
            fired.add(name)
            s = nxt
    return frozenset(fired)


def _has_punctuation_anomaly(value: str) -> bool:
    """True iff the Tier-1 punctuation fix would change ``value``.

    Defined in terms of :func:`_normalize_punctuation` so the detector and the
    fix can never disagree (no cell is flagged that the fix leaves untouched).
    This is the value-alone test (no column context); the column-convention
    guard lives in :func:`_punctuation_column_findings`.
    """
    if not value:
        return False
    return _normalize_punctuation(value) != value


# A punctuation rule that fires on at least this FRACTION of a column's
# non-empty cells is treated as that column's CONVENTION, not a defect, and is
# suppressed for the whole column. Requires at least ``_PUNCT_CONVENTION_MIN``
# non-empty cells before any convention can be inferred (you cannot call two
# cells a "convention"). Tuned so uniform-format columns -- comma-joined word
# pairs (``people,analytics``), ``Total : State`` prefixes, ``A,B,C`` addresses
# -- are left alone, while an isolated anomaly in an otherwise-clean column is
# still flagged.
_PUNCT_CONVENTION_THRESHOLD = 0.9
_PUNCT_CONVENTION_MIN = 4


def _punctuation_column_findings(
    values: List[str],
    *,
    threshold: float = _PUNCT_CONVENTION_THRESHOLD,
    min_cells: int = _PUNCT_CONVENTION_MIN,
) -> List[Tuple[bool, str]]:
    """Column-aware punctuation findings: one ``(is_anomaly, fix)`` per value.

    Unlike the per-cell :func:`_has_punctuation_anomaly`, this inspects the
    whole column. Any Tier-1 rule that fires on ``>= threshold`` of the
    non-empty cells is the column's CONVENTION (e.g. every cell is
    ``word,word``) and is suppressed: it neither flags a cell nor appears in
    that cell's fix. A cell is an anomaly only if it still triggers some
    NON-convention rule, and its fix omits the convention rules so the shared
    pattern is preserved.

    Columns with fewer than ``min_cells`` non-empty values cannot establish a
    convention, so each cell falls back to the plain value-alone rule.
    """
    n = len(values)
    out: List[Tuple[bool, str]] = [(False, "")] * n
    nonempty = [(i, v) for i, v in enumerate(values) if v and v.strip()]
    if not nonempty:
        return out

    fired_by_idx = {i: _punctuation_rules_fired(v) for i, v in nonempty}
    convention: frozenset[str] = frozenset()
    if len(nonempty) >= min_cells:
        counts: dict[str, int] = {}
        for i, _ in nonempty:
            for rule in fired_by_idx[i]:
                counts[rule] = counts.get(rule, 0) + 1
        cutoff = threshold * len(nonempty)
        convention = frozenset(r for r, c in counts.items() if c >= cutoff)

    for i, v in nonempty:
        effective = fired_by_idx[i] - convention
        if effective:
            out[i] = (True, _normalize_punctuation(v, skip=convention))
    return out


# ---------------------------------------------------------------------------
# "Low value" fix detection. A correction is LOW VALUE when it changes the
# value only by SPACING or a SINGLE PUNCTUATION mark (case-insensitive) -- the
# letters/digits are otherwise identical, so the fix is cosmetic. Used by the
# benchmark exporter to flag "low value TP" cells (a real but trivial typo
# catch). A real letter change (PRESTIEGE -> PRESTIGE) is NOT low value.
# ---------------------------------------------------------------------------

# Unicode dash / hyphen variants typographically equivalent to ASCII '-'.
_DASH_LO, _DASH_HI, _MINUS = 0x2010, 0x2015, 0x2212
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _norm_dashes(s: str) -> str:
    return "".join(
        "-" if (_DASH_LO <= ord(ch) <= _DASH_HI or ord(ch) == _MINUS) else ch
        for ch in s
    )


def _letter_core(s: str) -> str:
    """Letters/digits only -- whitespace + punctuation stripped, case-folded."""
    return re.sub(r"\s+", "", _PUNCT_RE.sub("", s)).casefold()


def _is_low_value_fix(original: str, fix: str) -> bool:
    """True iff ``original -> fix`` differs ONLY by spacing or a single
    punctuation mark (case-insensitive) -- a cosmetically-trivial fix.

    Catches:
      * SPACING: identical once all whitespace is removed -- including
        word-boundary splits/joins (``log anberries`` -> ``loganberries``,
        ``Strubenvalley`` -> ``Struben Valley``) and padding/run collapse.
      * SINGLE PUNCTUATION: the letter/digit core is identical and the
        punctuation differs by at most one mark added / removed / changed
        (``Pty,) Ltd`` -> ``Pty) Ltd``, ``St Louis`` -> ``St. Louis``).

    A real letter/digit change (``PRESTIEGE`` -> ``PRESTIGE``, ``Pacfic`` ->
    ``Pacific``) is NOT low value.
    """
    if not isinstance(original, str) or not isinstance(fix, str):
        return False
    if not original or not fix or original == fix:
        return False
    a, b = _norm_dashes(original), _norm_dashes(fix)
    # spacing-only (case-insensitive): equal once whitespace is removed.
    if re.sub(r"\s+", "", a).casefold() == re.sub(r"\s+", "", b).casefold():
        return True
    # single punctuation: same letter/digit core, punctuation count differs
    # by <= 1 (one mark inserted, removed, or substituted).
    core = _letter_core(a)
    if core and core == _letter_core(b):
        if abs(len(_PUNCT_RE.findall(a)) - len(_PUNCT_RE.findall(b))) <= 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Instruction-vs-value detection. A correction recorded by an annotator or
# proposed by the LLM is sometimes PROSE -- an instruction describing the
# edit ("Change \"Strubenvalley\" to \"Struben Valley\"") rather than the
# concrete replacement value ("...Struben Valley..."). For a per-cell
# benchmark whose whole point is an ``original -> corrected`` value pair,
# prose is unusable, so the lean exporter detects it and prefers a concrete
# value instead.
#
# Heuristic (deliberately conservative to avoid misclassifying real values
# that merely start with an editing verb, e.g. the company name "Change
# Healthcare Inc" or "Set Design Studio"): the text must start with an
# imperative editing verb AND carry a SECOND instruction signal -- either
#   1. an instruction marker -- a quote glyph, an arrow, or one of the
#      connector words `` to `` / `` with `` -- that signals the
#      "<verb> X to/with Y" shape; OR
#   2. an editing-OBJECT word -- meta-textual vocabulary that names the thing
#      being edited ("trailing comma", "space before the colon", "closing
#      parenthesis"). These markerless instructions ("Remove the trailing
#      comma") slipped through the marker-only test and leaked into the
#      benchmark as if they were concrete values.
# The object list is meta-textual on purpose (punctuation / spacing / spelling
# terms) so a real value that merely opens with a verb -- "Change Healthcare",
# "Set Design Studio", "Remove Wall Anchors" -- carries none of these words and
# still passes through untouched. Neither the verb nor a signal alone flips it.
# ---------------------------------------------------------------------------

_INSTRUCTION_VERBS = (
    "change", "replace", "correct", "fix", "set", "remove", "delete",
    "insert", "use", "update", "capitalize", "capitalise", "expand",
    "standardize", "standardise", "swap", "rename", "convert", "modify",
    "edit", "normalize", "normalise", "reformat", "rewrite", "drop",
    "append", "prepend", "trim", "split", "join", "add",
)
_INSTRUCTION_VERB_RE = re.compile(
    r"^\s*(?:" + "|".join(_INSTRUCTION_VERBS) + r")\b", re.IGNORECASE
)
# Quote glyphs (ASCII + curly + guillemets), arrows, or the connector words.
_INSTRUCTION_MARKER_RE = re.compile(
    r"[\"'\u2018\u2019\u201c\u201d\u00ab\u00bb]|->|\u2192|\bto\b|\bwith\b",
    re.IGNORECASE,
)
# Editing-OBJECT vocabulary: meta-textual words that name a punctuation / spacing
# / spelling element being edited. Presence of one (after an imperative verb)
# marks the text as an instruction even without a quote/arrow/connector marker
# ("Remove the trailing comma", "Add closing parenthesis"). Kept to text-editing
# nouns/qualifiers so it never fires on an ordinary value that opens with a verb.
_INSTRUCTION_OBJECT_RE = re.compile(
    r"\b(?:trailing|leading|closing|opening|extra|missing|doubled?|stray|"
    r"redundant|superfluous|duplicate|"
    r"spaces?|whitespace|blank|comma|commas|colon|colons|semicolon|semicolons|"
    r"periods?|full stop|apostrophes?|quotes?|quotation|parenthes[ie]s|"
    r"brackets?|braces?|hyphens?|dashes?|slashes?|ampersands?|asterisks?|"
    r"underscores?|punctuation|capitali[sz]ation|characters?|letters?|digits?|"
    r"typos?|spelling|spacing)\b",
    re.IGNORECASE,
)


def _looks_like_instruction(text: Optional[str]) -> bool:
    """True iff ``text`` reads as an EDIT INSTRUCTION, not a replacement value.

    See the section comment above for the heuristic. Used by the benchmark
    exporters to prefer a concrete corrected value over an annotator's or the
    LLM's free-form "Change X to Y" / "Remove the trailing comma" note.
    """
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if not _INSTRUCTION_VERB_RE.match(s):
        return False
    return (
        _INSTRUCTION_MARKER_RE.search(s) is not None
        or _INSTRUCTION_OBJECT_RE.search(s) is not None
    )


# ---------------------------------------------------------------------------
# Faithful derivation of a CONCRETE fix from a trivial EDIT INSTRUCTION.
#
# When the only fix a cell carries is prose ("Remove the trailing comma"), the
# benchmark needs the literal RESULT of that edit, not the sentence. Blindly
# running the whole punctuation normaliser is UNSAFE -- on "Add closing
# parenthesis" it silently space-pads commas instead, on "Remove the trailing
# comma" it space-pads an ampersand and leaves the comma -- so the emitted value
# would not match the instruction. Instead we recognise a small set of
# UNAMBIGUOUS instruction classes and apply ONLY that specific edit, so the
# derived value is faithful by construction. Anything outside these classes
# (adding characters at an ambiguous position, replacing with an unknown value,
# blanking a cell, ...) returns ``None`` -> the caller emits ``null``.
#
# Supported classes (each deterministic + reversible from the value alone):
#   1. Remove a stray space BEFORE a named closing mark ("space before the
#      comma/colon") -> drop ``[ \t]+`` immediately preceding that mark.
#   2. Remove a TRAILING mark ("trailing comma / apostrophe / period") -> strip
#      that mark (and surrounding trivial whitespace) from the end.
#   3. Collapse a purely-whitespace anomaly ("remove the extra space", "trim the
#      whitespace") -> ``_normalize_trivial`` (only when the instruction names
#      no punctuation object, so it can't stand in for a punctuation edit).
# ---------------------------------------------------------------------------

# Named punctuation the instruction classes understand -> its literal glyph(s).
_NAMED_MARKS: dict[str, tuple[str, ...]] = {
    "comma": (",",), "commas": (",",),
    "colon": (":",), "colons": (":",),
    "semicolon": (";",), "semicolons": (";",),
    "period": (".",), "periods": (".",), "full stop": (".",), "dot": (".",),
    "apostrophe": ("'", "\u2019"), "apostrophes": ("'", "\u2019"),
    "quote": ('"', "'", "\u2019", "\u201d"),
    "quotes": ('"', "'", "\u2019", "\u201d"),
    "quotation mark": ('"', "\u201d"), "quotation marks": ('"', "\u201d"),
    "hyphen": ("-",), "hyphens": ("-",), "dash": ("-",), "dashes": ("-",),
    "slash": ("/",), "slashes": ("/",),
    "period.": (".",),
}
# Verbs that DELETE (so "<del-verb> ... trailing comma" / "... space before").
_DEL_VERB_RE = re.compile(r"^\s*(?:remove|delete|drop|strip|trim|eliminate)\b", re.IGNORECASE)
_TRAILING_RE = re.compile(r"\btrailing\b", re.IGNORECASE)
_SPACE_BEFORE_RE = re.compile(r"\bspaces?\b.*?\bbefore\b|\bbefore\b.*?\bspaces?\b", re.IGNORECASE)
# A whitespace-only edit: mentions a space/whitespace word but NO named mark
# and NO "before"/"trailing" qualifier (those are handled by classes 1 & 2).
_WS_WORD_RE = re.compile(r"\b(?:whitespace|spaces?|blank spaces?)\b", re.IGNORECASE)


def _named_marks_in(text: str) -> tuple[str, ...]:
    """The union of literal glyphs for every punctuation NAME the text mentions."""
    low = text.lower()
    glyphs: list[str] = []
    for name, chars in _NAMED_MARKS.items():
        if re.search(r"\b" + re.escape(name) + r"\b", low):
            for c in chars:
                if c not in glyphs:
                    glyphs.append(c)
    return tuple(glyphs)


def _derive_trivial_fix(original: Optional[str], instruction: Optional[str]) -> Optional[str]:
    """Return the literal result of a trivial edit INSTRUCTION, or ``None``.

    ``original`` is the cell's raw value; ``instruction`` is the prose fix. Only
    the unambiguous instruction classes documented above are honoured, and each
    is applied as its OWN specific edit (never the blanket normaliser), so the
    result is guaranteed to match what the instruction asked for. Returns
    ``None`` when the instruction is out of scope or the edit changes nothing.
    """
    if not isinstance(original, str) or not original:
        return None
    if not isinstance(instruction, str) or not _looks_like_instruction(instruction):
        return None
    marks = _named_marks_in(instruction)

    # Class 2: remove a TRAILING named mark (delete-verb + "trailing" + mark).
    if _DEL_VERB_RE.match(instruction) and _TRAILING_RE.search(instruction) and marks:
        s = original.rstrip()
        changed = False
        # Strip repeated trailing marks (and whitespace between them), e.g.
        # "X ," -> "X", "X''" -> "X".
        while s and s[-1] in marks:
            s = s[:-1].rstrip()
            changed = True
        if changed and s != original:
            return s
        return None

    # Class 1: remove a stray space BEFORE a named closing mark.
    if _DEL_VERB_RE.match(instruction) and _SPACE_BEFORE_RE.search(instruction) and marks:
        charclass = "".join(re.escape(c) for c in marks)
        derived = re.sub(r"[ \t]+([" + charclass + r"])", r"\1", original)
        if derived != original:
            return derived
        return None

    # Class 3: purely-whitespace edit (no named punctuation object involved).
    if (
        _DEL_VERB_RE.match(instruction)
        and _WS_WORD_RE.search(instruction)
        and not marks
        and not _TRAILING_RE.search(instruction)
        and _has_whitespace_anomaly(original)
    ):
        derived = _normalize_trivial(original)
        if derived and derived != original:
            return derived
    return None


# ---------------------------------------------------------------------------
# Trivial INCONSISTENCY detection. Two distinct cell values in the SAME
# column are "trivially inconsistent" when they collapse to the same string
# after a mechanical normalisation (case-fold + drop all whitespace + drop
# all punctuation -- i.e. ``_letter_core``). Examples: ``Clarington Orioles``
# vs ``Clarington orioles`` (casing), ``BlackJaQ`` vs ``Black Jaq``
# (spacing+casing), ``#covid`` vs ``covid`` (punctuation).
#
# PURELY NUMERIC groups are EXCLUDED on purpose: ``_letter_core`` strips the
# decimal point and minus sign, so genuinely different numbers collide
# (``2.5`` -> ``25``, ``-0.3`` / ``0.3`` -> ``03``). Those are NOT
# inconsistencies, so a group whose normalised key is all digits is dropped.
#
# Shared by the trivial-inconsistency tagger (``routers/propagation.py``) so
# the rule has a single, unit-tested definition.
# ---------------------------------------------------------------------------

# A normalised key is "numeric" (and so excluded) when it is all ASCII
# digits -- every member of the group is then a number differing only by
# stripped punctuation / spacing (a different VALUE, not an inconsistency).
_INCON_NUMERIC_RE = re.compile(r"[0-9]+")


def _is_numeric_norm_key(key: str) -> bool:
    """True iff ``key`` (a ``_letter_core`` result) is all ASCII digits."""
    return bool(key) and _INCON_NUMERIC_RE.fullmatch(key) is not None


def _trivial_inconsistency_kind(raws: Iterable[str]) -> str:
    """Classify a trivial-inconsistency group by how its members differ.

    Returns a :class:`~src.labeling.schema.ClusterKind` *value* string:
      * ``"casing"``      -- members are equal once case-folded (case only);
      * ``"punctuation"`` -- otherwise (whitespace / punctuation differences,
        the closest taxonomy tag for mechanical surface variance).
    """
    items = list(raws)
    if len({r.casefold() for r in items}) == 1:
        return "casing"
    return "punctuation"


def _trivial_inconsistency_groups(
    values: Iterable[Any],
) -> List[Tuple[str, List[str], str]]:
    """Group a column's values into trivial-inconsistency clusters.

    Two distinct raw values join the same group when their
    :func:`_letter_core` normalisation is equal. Returns one entry per
    group that has >= 2 distinct raw values AND whose normalised key is
    not purely numeric:

        ``[(norm_key, sorted_distinct_raw_values, kind), ...]``

    sorted by ``norm_key`` for a stable, reproducible preview order.
    Empty-normalising values (all punctuation / whitespace) are skipped.
    """
    return [
        (key, raws, kind)
        for key, raws, kind, is_numeric in _inconsistency_collision_groups(values)
        if not is_numeric
    ]


def _inconsistency_collision_groups(
    values: Iterable[Any],
) -> List[Tuple[str, List[str], str, bool]]:
    """All normalisation-collision groups in a column, numeric flag included.

    Like :func:`_trivial_inconsistency_groups` but KEEPS the purely
    numeric groups, tagging each tuple with a trailing ``is_numeric``
    bool so the caller can both act on the non-numeric groups and report
    how many numeric ones were skipped:

        ``[(norm_key, sorted_distinct_raw_values, kind, is_numeric), ...]``
    """
    by_norm: "defaultdict[str, List[str]]" = defaultdict(list)
    seen: "defaultdict[str, set]" = defaultdict(set)
    for v in values:
        s = _scalar(v)
        key = _letter_core(s)
        if not key:
            continue
        if s not in seen[key]:
            seen[key].add(s)
            by_norm[key].append(s)
    out: List[Tuple[str, List[str], str, bool]] = []
    for key in sorted(by_norm):
        raws = by_norm[key]
        if len(raws) < 2:
            continue
        raws_sorted = sorted(raws)
        out.append((
            key,
            raws_sorted,
            _trivial_inconsistency_kind(raws_sorted),
            _is_numeric_norm_key(key),
        ))
    return out


def _validate_source_name(name: str) -> None:
    """Reject source names that look like paths (traversal guard)."""
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise HTTPException(400, f"invalid source name: {name!r}")
