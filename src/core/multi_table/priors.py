"""Cheap structural detectors that surface candidate cross-sheet
relations for multi-table discovery.

The multi-table error type ``inter_table_inconsistency`` is an
umbrella that covers FK orphans, synonym/homonym headers, and
units / scales / encoding mismatches across sibling sheets. To help
the LLM, we pre-compute three candidate structures and surface them
to the discovery prompt:

    1. Foreign-key candidates - ``(parent.col_p, child.col_c)``
       pairs where the child's value set is mostly contained in
       the parent's controlled set, with up to 5 example orphan
       row indices in the child.
    2. Naming-conflict candidates - synonyms (high value-set
       overlap, different headers) and homonyms (same header,
       disjoint value sets or mismatched dtype kinds).
    3. Representation-conflict candidates - same-named columns
       across sibling sheets whose dtype kind / numeric range /
       small-vocabulary token set diverges.

The detectors are pure, deterministic, and capped to keep the
quadratic-in-sheets cost bounded on long-tail workbooks (max 142
sheets observed in the bing21 corpus). They reuse identity-column
hints from :mod:`src.core.table_digest` where available.

The output is a small dataclass with both a structured form (for
tests / future tooling) and a Markdown-rendered form for the
discovery prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..table_digest import TableDigest

logger = logging.getLogger("multi_table_priors")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FKCandidate:
    """A candidate foreign-key dependency. ``child[col_c]`` references
    the controlled set in ``parent[col_p]``; values in ``child[col_c]``
    that lie outside the parent's set are ``orphan_row_indices``.

    ``coverage`` is computed against the *raw* parent set. When the
    optional ``normalized_coverage`` is materially higher than
    ``coverage``, the orphans are likely surface-form mismatches
    ("US" vs "United States"), not true referential-integrity
    breaks, and ``best_normalization_step`` names the rung that
    closed the gap. The LLM verifier sees both numbers + the rung
    label as evidence.
    """

    parent: str
    parent_column: str
    child: str
    child_column: str
    n_parent_distinct: int
    n_child_distinct: int
    overlap: int          # number of distinct child values found in parent (raw)
    coverage: float       # overlap / n_child_distinct (raw)
    orphan_row_indices: Tuple[int, ...]
    orphan_values: Tuple[str, ...]
    normalized_coverage: float = 0.0
    best_normalization_step: str = "raw"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent": self.parent,
            "parent_column": self.parent_column,
            "child": self.child,
            "child_column": self.child_column,
            "n_parent_distinct": int(self.n_parent_distinct),
            "n_child_distinct": int(self.n_child_distinct),
            "overlap": int(self.overlap),
            "coverage": float(self.coverage),
            "orphan_row_indices": list(self.orphan_row_indices),
            "orphan_values": list(self.orphan_values),
            "normalized_coverage": float(self.normalized_coverage),
            "best_normalization_step": str(self.best_normalization_step),
        }


@dataclass(frozen=True)
class NamingCandidate:
    """A candidate cross-sheet naming conflict.

    ``relation`` is one of ``"synonym"`` (different headers, same
    concept) or ``"homonym"`` (same header, different concepts).
    ``jaccard`` is computed over the columns' raw value sets;
    ``normalized_jaccard`` and ``best_normalization_step`` add
    surface-form-aware evidence.
    """

    relation: str  # "synonym" | "homonym"
    sheet_a: str
    column_a: str
    sheet_b: str
    column_b: str
    jaccard: float
    n_distinct_a: int
    n_distinct_b: int
    normalized_jaccard: float = 0.0
    best_normalization_step: str = "raw"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation": self.relation,
            "sheet_a": self.sheet_a,
            "column_a": self.column_a,
            "sheet_b": self.sheet_b,
            "column_b": self.column_b,
            "jaccard": float(self.jaccard),
            "n_distinct_a": int(self.n_distinct_a),
            "n_distinct_b": int(self.n_distinct_b),
            "normalized_jaccard": float(self.normalized_jaccard),
            "best_normalization_step": str(self.best_normalization_step),
        }


@dataclass(frozen=True)
class RepresentationCandidate:
    """A candidate value-representation conflict on a header that
    appears in two sibling sheets but disagrees in dtype kind,
    numeric magnitude, or token vocabulary."""

    column: str
    sheet_a: str
    sheet_b: str
    reason: str   # "dtype" | "magnitude" | "vocabulary"
    detail_a: str
    detail_b: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "sheet_a": self.sheet_a,
            "sheet_b": self.sheet_b,
            "reason": self.reason,
            "detail_a": self.detail_a,
            "detail_b": self.detail_b,
        }


# ---------------------------------------------------------------------------
# New candidate dataclasses (Gap F)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributeAgreementCandidate:
    """Two sibling sheets share an identity key, but rows that match
    on that key disagree on a descriptive attribute.

    Example: ``customers.csv`` and ``orders.csv`` both carry
    ``customer_id`` plus a denormalised ``country`` column, and for
    customer ``C001`` one sheet says ``"US"`` while the other says
    ``"Canada"``.
    """

    sheet_a: str
    sheet_b: str
    key_columns: Tuple[str, ...]
    attribute_column: str
    n_shared_keys: int
    n_conflicts: int
    sample_conflicts: Tuple[Tuple[str, str, str], ...]  # (key_repr, val_a, val_b)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_a": self.sheet_a,
            "sheet_b": self.sheet_b,
            "key_columns": list(self.key_columns),
            "attribute_column": self.attribute_column,
            "n_shared_keys": int(self.n_shared_keys),
            "n_conflicts": int(self.n_conflicts),
            "sample_conflicts": [list(t) for t in self.sample_conflicts],
        }


@dataclass(frozen=True)
class CompositeFKCandidate:
    """Like ``FKCandidate`` but the join key is a tuple of columns.

    Surfaces multi-column FKs that single-column scoring misses
    (e.g. ``(country, postal_code)`` or ``(year, quarter)``).
    """

    parent: str
    parent_columns: Tuple[str, ...]
    child: str
    child_columns: Tuple[str, ...]
    n_parent_distinct: int
    n_child_distinct: int
    overlap: int
    coverage: float
    orphan_row_indices: Tuple[int, ...]
    orphan_keys: Tuple[Tuple[str, ...], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent": self.parent,
            "parent_columns": list(self.parent_columns),
            "child": self.child,
            "child_columns": list(self.child_columns),
            "n_parent_distinct": int(self.n_parent_distinct),
            "n_child_distinct": int(self.n_child_distinct),
            "overlap": int(self.overlap),
            "coverage": float(self.coverage),
            "orphan_row_indices": list(self.orphan_row_indices),
            "orphan_keys": [list(k) for k in self.orphan_keys],
        }


@dataclass(frozen=True)
class ContainmentCandidate:
    """Asymmetric value-set containment between two columns from
    different sheets. ``coverage`` is ``|A ∩_norm B| / |A|`` (NOT
    Jaccard). Surfaces small dimension tables joining into large
    fact tables, where symmetric Jaccard is tiny but the dimension
    side is mostly contained in the fact side (or vice versa)."""

    sheet_a: str
    column_a: str
    sheet_b: str
    column_b: str
    n_distinct_a: int
    n_distinct_b: int
    coverage_a_in_b: float   # fraction of A that appears in B (normalized)
    coverage_b_in_a: float   # fraction of B that appears in A (normalized)
    best_normalization_step: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_a": self.sheet_a,
            "column_a": self.column_a,
            "sheet_b": self.sheet_b,
            "column_b": self.column_b,
            "n_distinct_a": int(self.n_distinct_a),
            "n_distinct_b": int(self.n_distinct_b),
            "coverage_a_in_b": float(self.coverage_a_in_b),
            "coverage_b_in_a": float(self.coverage_b_in_a),
            "best_normalization_step": str(self.best_normalization_step),
        }


@dataclass(frozen=True)
class TimeOverlapCandidate:
    """Two sibling sheets each carry a date column whose ranges
    overlap, suggesting union-style stacking with potential
    duplicate snapshots over the overlapping window."""

    sheet_a: str
    column_a: str
    sheet_b: str
    column_b: str
    range_a: Tuple[str, str]   # (min_iso, max_iso)
    range_b: Tuple[str, str]
    overlap: Tuple[str, str]
    n_rows_a_in_overlap: int
    n_rows_b_in_overlap: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_a": self.sheet_a,
            "column_a": self.column_a,
            "sheet_b": self.sheet_b,
            "column_b": self.column_b,
            "range_a": list(self.range_a),
            "range_b": list(self.range_b),
            "overlap": list(self.overlap),
            "n_rows_a_in_overlap": int(self.n_rows_a_in_overlap),
            "n_rows_b_in_overlap": int(self.n_rows_b_in_overlap),
        }


@dataclass(frozen=True)
class NullSentinelCandidate:
    """Two sibling sheets use different sentinel tokens for the
    same conceptual missing value on the same column.

    Example: sheet A has ``"NA"`` as the dominant missing-like
    token in ``Status``; sheet B has ``"-"``. Both are within-domain
    encodings of "missing" but the concatenation pollutes the
    column's apparent vocabulary.
    """

    column: str
    sheet_a: str
    sheet_b: str
    sentinel_a: str
    sentinel_b: str
    freq_a: int
    freq_b: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "sheet_a": self.sheet_a,
            "sheet_b": self.sheet_b,
            "sentinel_a": self.sentinel_a,
            "sentinel_b": self.sentinel_b,
            "freq_a": int(self.freq_a),
            "freq_b": int(self.freq_b),
        }


@dataclass(frozen=True)
class EmbeddingMatchCandidate:
    """A pair of values from two different sheets / columns that
    have near-zero string overlap but high embedding similarity.

    Surfaces aliases that all string-normalization rungs miss
    (e.g. ``"Germany"`` ↔ ``"Deutschland"``).
    """

    sheet_a: str
    column_a: str
    sheet_b: str
    column_b: str
    matches: Tuple[Tuple[str, str, float], ...]   # (val_a, val_b, cosine)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_a": self.sheet_a,
            "column_a": self.column_a,
            "sheet_b": self.sheet_b,
            "column_b": self.column_b,
            "matches": [list(t) for t in self.matches],
        }


@dataclass(frozen=True)
class UnionableGroupCandidate:
    """A set of sibling sheets whose headers overlap enough that they
    can be treated as a UNION, possibly with column-rename or with
    NaN-fills for partially-shared columns.

    The group is built deterministically by clustering sheets whose
    pairwise normalized-header Jaccard exceeds ``min_jaccard``. This
    is a *candidate* — the LLM downstream confirms it and supplies a
    canonical column mapping (so e.g. ``"Reading Quiz 1 %"`` and
    ``"Reading Quiz 1 (%)"`` collapse to the same canonical key).

    ``sheet_headers`` preserves the per-sheet raw header order so
    the downstream prompt can show the LLM the alignment problem
    concretely.
    """

    group_id: str
    sheets: Tuple[str, ...]
    n_sheets: int
    n_canonical_headers: int   # |union of normalized headers|
    n_shared_headers: int      # |intersection of normalized headers|
    avg_jaccard: float         # mean pairwise normalized-header Jaccard
    sheet_headers: Tuple[Tuple[str, Tuple[str, ...]], ...]   # ((sheet, headers), ...)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "sheets": list(self.sheets),
            "n_sheets": int(self.n_sheets),
            "n_canonical_headers": int(self.n_canonical_headers),
            "n_shared_headers": int(self.n_shared_headers),
            "avg_jaccard": float(self.avg_jaccard),
            "sheet_headers": [
                {"sheet": s, "headers": list(hs)}
                for s, hs in self.sheet_headers
            ],
        }


@dataclass(frozen=True)
class MultiTablePriors:
    """Structural priors for one (workbook, error_type) discovery
    call. Empty fields signal that the heuristic detector found
    nothing for that type."""

    error_type: str
    workbook: str
    sheets: Tuple[str, ...] = field(default_factory=tuple)
    fk_candidates: Tuple[FKCandidate, ...] = field(default_factory=tuple)
    naming_candidates: Tuple[NamingCandidate, ...] = field(default_factory=tuple)
    representation_candidates: Tuple[RepresentationCandidate, ...] = field(
        default_factory=tuple
    )
    attribute_agreement_candidates: Tuple[AttributeAgreementCandidate, ...] = field(
        default_factory=tuple
    )
    composite_fk_candidates: Tuple[CompositeFKCandidate, ...] = field(
        default_factory=tuple
    )
    containment_candidates: Tuple[ContainmentCandidate, ...] = field(
        default_factory=tuple
    )
    time_overlap_candidates: Tuple[TimeOverlapCandidate, ...] = field(
        default_factory=tuple
    )
    null_sentinel_candidates: Tuple[NullSentinelCandidate, ...] = field(
        default_factory=tuple
    )
    embedding_match_candidates: Tuple[EmbeddingMatchCandidate, ...] = field(
        default_factory=tuple
    )
    unionable_groups: Tuple[UnionableGroupCandidate, ...] = field(
        default_factory=tuple
    )

    def is_empty(self) -> bool:
        return not (
            self.fk_candidates
            or self.naming_candidates
            or self.representation_candidates
            or self.attribute_agreement_candidates
            or self.composite_fk_candidates
            or self.containment_candidates
            or self.time_overlap_candidates
            or self.null_sentinel_candidates
            or self.embedding_match_candidates
            or self.unionable_groups
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "workbook": self.workbook,
            "sheets": list(self.sheets),
            "fk_candidates": [c.to_dict() for c in self.fk_candidates],
            "naming_candidates": [c.to_dict() for c in self.naming_candidates],
            "representation_candidates": [
                c.to_dict() for c in self.representation_candidates
            ],
            "attribute_agreement_candidates": [
                c.to_dict() for c in self.attribute_agreement_candidates
            ],
            "composite_fk_candidates": [
                c.to_dict() for c in self.composite_fk_candidates
            ],
            "containment_candidates": [
                c.to_dict() for c in self.containment_candidates
            ],
            "time_overlap_candidates": [
                c.to_dict() for c in self.time_overlap_candidates
            ],
            "null_sentinel_candidates": [
                c.to_dict() for c in self.null_sentinel_candidates
            ],
            "embedding_match_candidates": [
                c.to_dict() for c in self.embedding_match_candidates
            ],
            "unionable_groups": [
                c.to_dict() for c in self.unionable_groups
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _column_value_set(df: pd.DataFrame, col: str, *, cap: int = 200) -> List[Any]:
    """Return up to ``cap`` distinct non-null values from ``df[col]``."""
    if col not in df.columns:
        return []
    try:
        return list(df[col].dropna().astype(str).unique()[:cap])
    except Exception:  # noqa: BLE001
        logger.debug("_column_value_set failed col=%s", col, exc_info=True)
        return []


def _jaccard(a: Sequence[Any], b: Sequence[Any]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return (inter / union) if union else 0.0


# ---------------------------------------------------------------------------
# Normalization ladder + alias tables (Gap B)
# ---------------------------------------------------------------------------
#
# The motivating example is "US" / "U.S." / "USA" / "United States" being
# the same entity but having raw-string Jaccard zero or near-zero. The
# ladder below collapses surface-form variants progressively; each rung
# reveals a different *kind* of inconsistency. We emit the rung that
# produced the largest overlap delta as evidence so the LLM verifier
# (and the human reviewer) sees both raw_overlap and normalized_overlap
# along with which transform closed the gap.
#
# Aliases are intentionally small and deterministic. They cover the
# very-common surface variants (countries, US states, currency codes,
# common units) that show up across spreadsheet families. A pluggable
# extension point - ``register_alias_dictionary`` - lets callers stack
# project-specific dictionaries without forking this module.

import re as _re
import unicodedata as _unicodedata


_PUNCT_RE = _re.compile(r"[^\w\s]")
_WS_RE = _re.compile(r"\s+")


# (canonical_lowercase_form, [aliases]). Kept short on purpose;
# project-specific dictionaries can be merged via register_alias_dictionary.
_DEFAULT_ALIAS_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    # Countries (subset of common surface variants).
    ("united states", ("us", "usa", "u.s.", "u.s.a.", "u s", "u s a", "u s of a", "united states of america", "america", "the united states")),
    ("united kingdom", ("uk", "u.k.", "u k", "britain", "great britain", "gb")),
    ("germany", ("deutschland", "ger", "de")),
    ("france", ("fr",)),
    ("china", ("prc", "p.r.c.", "p r c", "people's republic of china", "peoples republic of china", "cn")),
    ("japan", ("jp", "jpn")),
    ("south korea", ("korea", "republic of korea", "kr", "kor")),
    ("russia", ("russian federation", "ru")),
    ("netherlands", ("the netherlands", "holland", "nl")),
    # US states (a few common ones; full ISO list can be registered).
    ("california", ("ca", "calif")),
    ("new york", ("ny",)),
    ("texas", ("tx",)),
    ("massachusetts", ("ma", "mass")),
    ("washington", ("wa",)),  # state; "DC" is separate
    # Currencies.
    ("usd", ("us dollar", "us dollars", "dollar", "dollars", "$")),
    ("eur", ("euro", "euros", "€")),
    ("gbp", ("pound", "pounds", "british pound", "£")),
    ("jpy", ("yen", "japanese yen", "¥")),
    ("cny", ("yuan", "rmb", "renminbi")),
    # Common units.
    ("kg", ("kilogram", "kilograms", "kgs")),
    ("lb", ("pound", "pounds", "lbs")),
    ("kilometer", ("km", "kilometre", "kilometres", "kilometers")),
    ("mile", ("mi", "miles")),
    ("celsius", ("c", "deg c", "°c")),
    ("fahrenheit", ("f", "deg f", "°f")),
    # Booleans / yes-no encodings.
    ("yes", ("y", "true", "t", "1")),
    ("no", ("n", "false", "f", "0")),
    # Gender encodings.
    ("male", ("m",)),
    ("female", ("f",)),
)


def _build_alias_index(
    groups: Sequence[Tuple[str, Sequence[str]]],
) -> Dict[str, str]:
    """Map every surface form (lowercased) to its canonical token.

    Both the canonical and each alias map to the same canonical key.
    Conflicting aliases (the same surface form mapped under two
    canonical tokens) are resolved last-write-wins; callers should
    register their dictionaries in priority order.
    """
    idx: Dict[str, str] = {}
    for canon, aliases in groups:
        canon_low = canon.strip().lower()
        if not canon_low:
            continue
        idx[canon_low] = canon_low
        for a in aliases:
            a_low = a.strip().lower()
            if a_low:
                idx[a_low] = canon_low
    return idx


_ALIAS_INDEX: Dict[str, str] = _build_alias_index(_DEFAULT_ALIAS_GROUPS)


def register_alias_dictionary(
    groups: Sequence[Tuple[str, Sequence[str]]],
) -> None:
    """Register a project-specific alias dictionary.

    Each entry is ``(canonical_form, (alias_1, alias_2, ...))``.
    Surface forms are case-folded before lookup. Later registrations
    override earlier ones on conflict (last-write-wins).
    """
    global _ALIAS_INDEX
    extra = _build_alias_index(groups)
    _ALIAS_INDEX = {**_ALIAS_INDEX, **extra}


def _norm_step_lower(s: str) -> str:
    return s.lower()


def _norm_step_unidecode(s: str) -> str:
    """NFKD + drop combining marks. ASCII-fold without external dep."""
    nfkd = _unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not _unicodedata.combining(c))


def _norm_step_strip_punct(s: str) -> str:
    return _PUNCT_RE.sub(" ", s)


def _norm_step_collapse_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _norm_step_alias(s: str) -> str:
    """Map known surface form to its canonical token, else passthrough."""
    return _ALIAS_INDEX.get(s, s)


#: The normalization ladder. Each entry is ``(name, fn)``. Applied
#: cumulatively (later rungs see the output of earlier rungs).
NORMALIZATION_LADDER: Tuple[Tuple[str, Any], ...] = (
    ("lower", _norm_step_lower),
    ("unidecode", _norm_step_unidecode),
    ("strip_punct", _norm_step_strip_punct),
    ("collapse_ws", _norm_step_collapse_ws),
    ("alias", _norm_step_alias),
)


def normalize_value(
    value: Any,
    *,
    up_to: Optional[str] = None,
) -> str:
    """Apply the normalization ladder up to and including ``up_to``.

    ``up_to=None`` applies every rung. Non-string inputs are coerced
    via ``str(...)`` (so numeric ``"007"`` and ``7`` collapse via the
    integer cast in the caller's pre-pass; this function only handles
    surface-form folding).
    """
    s = "" if value is None else str(value)
    if not s:
        return s
    for name, fn in NORMALIZATION_LADDER:
        s = fn(s)
        if up_to is not None and name == up_to:
            break
    return s


def _normalize_set(values: Sequence[Any]) -> set:
    """Lowercased + alias-folded set, dropping empties."""
    out: set = set()
    for v in values:
        n = normalize_value(v)
        if n:
            out.add(n)
    return out


def _normalized_overlap_signals(
    values_a: Sequence[Any],
    values_b: Sequence[Any],
) -> Dict[str, float]:
    """Return per-rung Jaccard so callers can pick the rung that
    revealed the largest overlap delta.

    Keys: ``raw``, ``lower``, ``unidecode``, ``strip_punct``,
    ``collapse_ws``, ``alias``. Empty inputs yield 0.0 for every rung.
    """
    if not values_a or not values_b:
        return {"raw": 0.0, "lower": 0.0, "unidecode": 0.0,
                "strip_punct": 0.0, "collapse_ws": 0.0, "alias": 0.0}
    out: Dict[str, float] = {"raw": _jaccard(list(values_a), list(values_b))}
    for name, _fn in NORMALIZATION_LADDER:
        sa = {normalize_value(v, up_to=name) for v in values_a}
        sb = {normalize_value(v, up_to=name) for v in values_b}
        sa.discard("")
        sb.discard("")
        if not sa or not sb:
            out[name] = 0.0
            continue
        inter = len(sa & sb)
        union = len(sa | sb)
        out[name] = (inter / union) if union else 0.0
    return out


def _best_normalization(
    signals: Mapping[str, float],
) -> Tuple[str, float, float]:
    """Return ``(rung_name, best_jaccard, delta_over_raw)``.

    ``delta_over_raw`` is ``best_jaccard - raw_jaccard``; positive
    values mean normalization revealed overlap that raw matching
    missed (the surface-form-inconsistency signal we care about).
    """
    raw = float(signals.get("raw", 0.0))
    best_name = "raw"
    best_val = raw
    for name in ("lower", "unidecode", "strip_punct", "collapse_ws", "alias"):
        v = float(signals.get(name, 0.0))
        if v > best_val:
            best_val = v
            best_name = name
    return best_name, best_val, best_val - raw


# ---------------------------------------------------------------------------
# Embedding-residual ANN hook (Gap C)
# ---------------------------------------------------------------------------
#
# Some surface-disjoint aliases ("Germany" vs "Deutschland") have zero
# overlap on every string-normalization rung but cluster under any
# decent text embedding. Rather than hard-coding an embedder, we expose
# a callable that callers can wire to their preferred backend (OpenAI,
# sentence-transformers, etc.). The default is a no-op so the priors
# module stays dependency-free; callers wire it via
# ``set_embedding_matcher(...)`` from a runner that has the model
# loaded.

EmbeddingMatcher = Any  # Callable[[Sequence[str], Sequence[str]], List[Tuple[str, str, float]]]

_EMBEDDING_MATCHER: Optional[Any] = None


def set_embedding_matcher(matcher: Optional[Any]) -> None:
    """Register an embedding-based matcher for surface-disjoint values.

    The matcher is a callable
    ``matcher(values_a, values_b) -> List[(value_a, value_b, cosine)]``
    returning *only* pairs above the caller's similarity threshold.
    Pass ``None`` to disable.
    """
    global _EMBEDDING_MATCHER
    _EMBEDDING_MATCHER = matcher


def _embedding_residual_matches(
    values_a: Sequence[str],
    values_b: Sequence[str],
    *,
    max_pairs: int = 10,
) -> List[Tuple[str, str, float]]:
    """Run the registered embedding matcher on residual (un-aliased)
    values. Returns at most ``max_pairs`` cross-set pairs."""
    if _EMBEDDING_MATCHER is None or not values_a or not values_b:
        return []
    try:
        pairs = _EMBEDDING_MATCHER(list(values_a), list(values_b))
    except Exception:  # noqa: BLE001
        logger.warning("embedding matcher failed; falling back to no-op")
        return []
    return list(pairs)[:max_pairs]


def _select_representative_sheets(
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    *,
    cap: int = 12,
) -> List[str]:
    """Pick up to ``cap`` sheets ranked by row count, descending.

    Long-tail workbooks have up to 142 sheets in the bing21 corpus;
    a quadratic scan over all of them would dominate the runner.
    Top-N by row count keeps the most informative sheets.
    """
    if len(sheets) <= cap:
        return list(sheets)
    scored = sorted(
        sheets,
        key=lambda s: -int(len(frames[s])) if s in frames else 0,
    )
    return scored[:cap]


# ---------------------------------------------------------------------------
# Detector 1: foreign-key candidates
# ---------------------------------------------------------------------------


def find_fk_candidates(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_candidates: int = 50,
    min_coverage: float = 0.1,
    min_child_distinct: int = 3,
    max_orphan_examples: int = 5,
    parent_value_cap: int = 500,
) -> List[FKCandidate]:
    """Find ``(parent.col_p, child.col_c)`` pairs where the child
    column's value set is mostly contained in the parent column's
    value set.

    Strategy:
      * Treat any digest-flagged identity-like column in ``parent``
        as a controlled-set candidate.
      * For each child column, compute raw coverage =
        ``|child ∩ parent| / |child|`` AND a normalized coverage
        across the surface-form ladder (lower / unidecode / strip
        punct / collapse ws / alias). Candidates pass when EITHER
        crosses ``min_coverage``; the rung that maximises coverage
        is recorded as evidence (``best_normalization_step``).
      * Surface-form-only matches (raw coverage low, normalized
        coverage high) directly catch the "US" vs "United States"
        family of cross-sheet inconsistencies.
      * Skip degenerate cases (parent <2 distinct values).

    The default ``max_candidates=50`` is intentionally permissive:
    recall is the primary objective, FP suppression is delegated to
    the LLM verifier + reviewer queue.
    """
    sheets_list = list(sheets)
    if len(sheets_list) < 2:
        return []

    candidates: List[FKCandidate] = []

    # Build per-sheet identity-column lists.
    identity_cols: Dict[str, List[str]] = {}
    for s in sheets_list:
        d = digests.get(s)
        if d is not None and d.candidate_identity_columns:
            identity_cols[s] = list(d.candidate_identity_columns)
        else:
            identity_cols[s] = []

    for parent in sheets_list:
        p_df = frames.get(parent)
        if p_df is None or p_df.empty:
            continue
        parent_id_cols = identity_cols.get(parent, [])
        if not parent_id_cols:
            continue
        for col_p in parent_id_cols:
            if col_p not in p_df.columns:
                continue
            p_values = _column_value_set(p_df, col_p, cap=parent_value_cap)
            if len(p_values) < 2:
                continue
            p_set = set(p_values)
            p_set_norm = _normalize_set(p_values)

            for child in sheets_list:
                if child == parent:
                    continue
                c_df = frames.get(child)
                if c_df is None or c_df.empty:
                    continue
                # Prefer same-named column. Fallback: scan ALL
                # string/int columns of the child. Previously capped
                # at 20 - that cap was a recall hazard on wide tables.
                child_cols_to_try: List[str] = []
                if col_p in c_df.columns:
                    child_cols_to_try.append(col_p)
                if not child_cols_to_try:
                    for cc in list(c_df.columns):
                        if pd.api.types.is_string_dtype(c_df[cc]) \
                                or pd.api.types.is_integer_dtype(c_df[cc]):
                            child_cols_to_try.append(str(cc))

                for col_c in child_cols_to_try:
                    if col_c not in c_df.columns:
                        continue
                    try:
                        child_series = c_df[col_c].dropna().astype(str)
                    except Exception:  # noqa: BLE001
                        logger.debug("FK child cast failed col=%s", col_c, exc_info=True)
                        continue
                    if child_series.empty:
                        continue
                    n_child_distinct = int(child_series.nunique())
                    if n_child_distinct < min_child_distinct:
                        continue
                    in_set = child_series.isin(p_set)
                    coverage = float(in_set.sum()) / max(1, len(child_series))

                    # Normalized coverage across the ladder.
                    child_distinct_vals = list(child_series.unique())
                    best_step = "raw"
                    best_norm_cov = coverage
                    for step_name, _fn in NORMALIZATION_LADDER:
                        norm_set = {
                            normalize_value(v, up_to=step_name)
                            for v in p_values
                        }
                        norm_set.discard("")
                        if not norm_set:
                            continue
                        norm_child = [
                            normalize_value(v, up_to=step_name)
                            for v in child_distinct_vals
                        ]
                        n_norm_in = sum(1 for v in norm_child if v and v in norm_set)
                        n_norm_total = sum(1 for v in norm_child if v)
                        if n_norm_total == 0:
                            continue
                        cov_step = n_norm_in / n_norm_total
                        if cov_step > best_norm_cov:
                            best_norm_cov = cov_step
                            best_step = step_name

                    # Pass if either raw or normalized coverage is
                    # interesting. Skip 100% on BOTH (no orphans on
                    # either pass -> nothing to flag).
                    interesting = (
                        coverage >= min_coverage
                        or best_norm_cov >= min_coverage
                    )
                    fully_clean = coverage >= 1.0 and best_norm_cov >= 1.0
                    if not interesting or fully_clean:
                        continue

                    orphan_mask = ~in_set
                    orphan_idx = list(child_series[orphan_mask].index)[:max_orphan_examples]
                    orphan_vals = list(
                        child_series[orphan_mask].unique()
                    )[:max_orphan_examples]
                    overlap_distinct = int(
                        child_series[in_set].nunique()
                    )
                    candidates.append(
                        FKCandidate(
                            parent=parent,
                            parent_column=col_p,
                            child=child,
                            child_column=col_c,
                            n_parent_distinct=len(p_values),
                            n_child_distinct=n_child_distinct,
                            overlap=overlap_distinct,
                            coverage=coverage,
                            orphan_row_indices=tuple(int(i) for i in orphan_idx),
                            orphan_values=tuple(str(v) for v in orphan_vals),
                            normalized_coverage=float(best_norm_cov),
                            best_normalization_step=best_step,
                        )
                    )

    # Rank by max(raw, normalized) coverage. Tiebreak by parent / child.
    candidates.sort(
        key=lambda c: (
            -max(c.coverage, c.normalized_coverage),
            c.parent, c.child, c.parent_column,
        )
    )
    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# Detector 2: naming conflicts (synonym / homonym)
# ---------------------------------------------------------------------------


def find_naming_conflicts(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_candidates: int = 50,
    synonym_jaccard: float = 0.4,
    homonym_jaccard: float = 0.05,
    min_distinct: int = 3,
    column_cap: int = 30,
) -> List[NamingCandidate]:
    """Find synonym (different name, same values) and homonym
    (same name, disjoint values) header pairs across sibling sheets.

    Recall-first defaults:
      * ``max_candidates=50`` (was 6) - rely on the LLM verifier and
        review queue to filter, not heuristic top-K truncation.
      * ``synonym_jaccard=0.4`` (was 0.6) on either RAW or NORMALIZED
        Jaccard - catches surface-form-divergent synonyms like
        ``"USA"`` vs ``"United States"`` that the raw Jaccard misses.
      * ``column_cap=30`` (was 12) per sheet.
    """
    sheets_list = list(sheets)
    if len(sheets_list) < 2:
        return []

    candidates: List[NamingCandidate] = []

    for i in range(len(sheets_list)):
        for j in range(i + 1, len(sheets_list)):
            a = sheets_list[i]
            b = sheets_list[j]
            df_a = frames.get(a)
            df_b = frames.get(b)
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue
            cols_a = [str(c) for c in list(df_a.columns)[:column_cap]]
            cols_b = [str(c) for c in list(df_b.columns)[:column_cap]]

            # --- Synonyms ---
            for ca in cols_a:
                va = _column_value_set(df_a, ca, cap=200)
                if len(va) < min_distinct:
                    continue
                kind_a = df_a[ca].dtype.kind
                for cb in cols_b:
                    if ca == cb:
                        continue
                    vb = _column_value_set(df_b, cb, cap=200)
                    if len(vb) < min_distinct:
                        continue
                    if df_b[cb].dtype.kind != kind_a:
                        continue
                    j_raw = _jaccard(va, vb)
                    signals = _normalized_overlap_signals(va, vb)
                    best_step, best_jacc, _delta = _best_normalization(signals)
                    if max(j_raw, best_jacc) >= synonym_jaccard:
                        candidates.append(
                            NamingCandidate(
                                relation="synonym",
                                sheet_a=a, column_a=ca,
                                sheet_b=b, column_b=cb,
                                jaccard=j_raw,
                                n_distinct_a=len(va),
                                n_distinct_b=len(vb),
                                normalized_jaccard=float(best_jacc),
                                best_normalization_step=best_step,
                            )
                        )

            # --- Homonyms ---
            shared = set(cols_a) & set(cols_b)
            for col in shared:
                va = _column_value_set(df_a, col, cap=200)
                vb = _column_value_set(df_b, col, cap=200)
                if len(va) < min_distinct or len(vb) < min_distinct:
                    continue
                kind_a = df_a[col].dtype.kind
                kind_b = df_b[col].dtype.kind
                j_raw = _jaccard(va, vb)
                signals = _normalized_overlap_signals(va, vb)
                best_step, best_jacc, _delta = _best_normalization(signals)
                # Homonym requires BOTH raw AND normalized Jaccard to
                # be low (so a surface-form synonym like "USA" vs
                # "United States" doesn't get mis-flagged as a
                # homonym just because raw Jaccard is 0).
                disjoint = (j_raw <= homonym_jaccard
                            and best_jacc <= homonym_jaccard)
                if (disjoint and len(va) >= min_distinct
                        and len(vb) >= min_distinct) or kind_a != kind_b:
                    candidates.append(
                        NamingCandidate(
                            relation="homonym",
                            sheet_a=a, column_a=col,
                            sheet_b=b, column_b=col,
                            jaccard=j_raw,
                            n_distinct_a=len(va),
                            n_distinct_b=len(vb),
                            normalized_jaccard=float(best_jacc),
                            best_normalization_step=best_step,
                        )
                    )

    # Stable order: synonyms first (more actionable), then homonyms,
    # both ranked by max(raw, normalized) Jaccard.
    syns = sorted(
        [c for c in candidates if c.relation == "synonym"],
        key=lambda c: (-max(c.jaccard, c.normalized_jaccard),
                       c.sheet_a, c.column_a),
    )
    homs = sorted(
        [c for c in candidates if c.relation == "homonym"],
        key=lambda c: (max(c.jaccard, c.normalized_jaccard),
                       c.sheet_a, c.column_a),
    )
    return (syns + homs)[:max_candidates]


# ---------------------------------------------------------------------------
# Detector 3: representation conflicts
# ---------------------------------------------------------------------------


def find_representation_conflicts(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_candidates: int = 50,
    magnitude_ratio: float = 10.0,
    suspect_ratios: Tuple[float, ...] = (
        # Common unit-conversion / scale-flip ratios. A pair whose
        # magnitude ratio lands near one of these is a strong unit-
        # drift candidate even if the absolute ratio is well below
        # ``magnitude_ratio``.
        2.20462,   # kg <-> lb
        1.60934,   # km <-> mi
        1000.0,    # cm <-> km, m <-> km, etc.
        100.0,     # cents <-> dollars
        0.01,
        12.0,      # in <-> ft
        3.28084,   # m <-> ft
    ),
    suspect_ratio_tolerance: float = 0.05,
) -> List[RepresentationCandidate]:
    """Find same-named columns whose representation differs across
    sibling sheets.

    Flag families:
      * ``dtype``: dtype kind differs (string vs numeric vs datetime).
      * ``magnitude``: numeric ratio >= ``magnitude_ratio`` OR within
        ``suspect_ratio_tolerance`` of a known unit-conversion ratio.
      * ``vocabulary``: both string, small enums, disjoint vocabs.
    """
    sheets_list = list(sheets)
    if len(sheets_list) < 2:
        return []

    candidates: List[RepresentationCandidate] = []
    seen_keys: set = set()

    for i in range(len(sheets_list)):
        for j in range(i + 1, len(sheets_list)):
            a = sheets_list[i]
            b = sheets_list[j]
            df_a = frames.get(a)
            df_b = frames.get(b)
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue
            shared = set(df_a.columns) & set(df_b.columns)
            for col in shared:
                col_str = str(col)
                key = (col_str, a, b)
                if key in seen_keys:
                    continue
                sa = df_a[col]
                sb = df_b[col]

                kind_a = sa.dtype.kind
                kind_b = sb.dtype.kind
                if kind_a != kind_b:
                    candidates.append(
                        RepresentationCandidate(
                            column=col_str, sheet_a=a, sheet_b=b,
                            reason="dtype",
                            detail_a=str(sa.dtype),
                            detail_b=str(sb.dtype),
                        )
                    )
                    seen_keys.add(key)
                    continue

                if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
                    try:
                        amax = float(sa.abs().max())
                        bmax = float(sb.abs().max())
                    except Exception:  # noqa: BLE001
                        logger.debug("unit-ratio max failed", exc_info=True)
                        amax, bmax = 0.0, 0.0
                    if amax > 0 and bmax > 0:
                        ratio = max(amax, bmax) / max(min(amax, bmax), 1e-9)
                        # Compare against suspect unit-conversion
                        # ratios via median/median to be robust to
                        # outliers; the max/max above is a coarse
                        # gate.
                        try:
                            amed = float(sa.abs().median())
                            bmed = float(sb.abs().median())
                        except Exception:  # noqa: BLE001
                            logger.debug("unit-ratio median failed", exc_info=True)
                            amed, bmed = 0.0, 0.0
                        med_ratio = (
                            max(amed, bmed) / max(min(amed, bmed), 1e-9)
                            if amed > 0 and bmed > 0 else 0.0
                        )
                        is_suspect_ratio = any(
                            abs(med_ratio - r) / r <= suspect_ratio_tolerance
                            for r in suspect_ratios
                        )
                        if ratio >= magnitude_ratio or is_suspect_ratio:
                            extra = (
                                f" (~{med_ratio:.4g}x, suspect unit ratio)"
                                if is_suspect_ratio else ""
                            )
                            candidates.append(
                                RepresentationCandidate(
                                    column=col_str, sheet_a=a, sheet_b=b,
                                    reason="magnitude",
                                    detail_a=f"max|x|={amax:.4g}{extra}",
                                    detail_b=f"max|x|={bmax:.4g}",
                                )
                            )
                            seen_keys.add(key)
                    continue

                if pd.api.types.is_string_dtype(sa) and pd.api.types.is_string_dtype(sb):
                    va = _column_value_set(df_a, col_str, cap=20)
                    vb = _column_value_set(df_b, col_str, cap=20)
                    if 0 < len(va) <= 6 and 0 < len(vb) <= 6:
                        sa_set, sb_set = set(va), set(vb)
                        if sa_set.isdisjoint(sb_set):
                            candidates.append(
                                RepresentationCandidate(
                                    column=col_str, sheet_a=a, sheet_b=b,
                                    reason="vocabulary",
                                    detail_a="{" + ", ".join(sorted(sa_set)) + "}",
                                    detail_b="{" + ", ".join(sorted(sb_set)) + "}",
                                )
                            )
                            seen_keys.add(key)

    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# Detector 4: attribute-agreement conflicts (Gap F)
# ---------------------------------------------------------------------------


def find_attribute_agreement_conflicts(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_candidates: int = 50,
    max_sample_conflicts: int = 5,
    min_shared_keys: int = 2,
) -> List[AttributeAgreementCandidate]:
    """Find rows in two sheets that share an identity-key value but
    disagree on a denormalised descriptive attribute.

    Pseudo-code:
        for each (sheet_a, sheet_b):
            for each shared identity-like column k present in both:
                join on k, identify rows that match
                for each other column c shared between both sheets:
                    flag rows where a[c] != b[c]

    Surfaces the ``customer_id`` / ``country`` mismatch family of
    cross-table inconsistencies (an entity exists in two tables and
    its attributes disagree).
    """
    sheets_list = list(sheets)
    if len(sheets_list) < 2:
        return []

    out: List[AttributeAgreementCandidate] = []
    for i in range(len(sheets_list)):
        for j in range(i + 1, len(sheets_list)):
            a = sheets_list[i]
            b = sheets_list[j]
            df_a = frames.get(a)
            df_b = frames.get(b)
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue
            d_a = digests.get(a)
            d_b = digests.get(b)
            id_cols_a = list(d_a.candidate_identity_columns) if d_a else []
            id_cols_b = list(d_b.candidate_identity_columns) if d_b else []
            shared_ids = [c for c in id_cols_a if c in id_cols_b
                          and c in df_a.columns and c in df_b.columns]
            if not shared_ids:
                continue
            shared_cols = [
                c for c in df_a.columns
                if c in df_b.columns and c not in shared_ids
            ]
            if not shared_cols:
                continue

            for key_col in shared_ids:
                try:
                    a_keyed = (
                        df_a[[key_col] + shared_cols]
                        .dropna(subset=[key_col])
                        .drop_duplicates(subset=[key_col])
                        .set_index(key_col)
                    )
                    b_keyed = (
                        df_b[[key_col] + shared_cols]
                        .dropna(subset=[key_col])
                        .drop_duplicates(subset=[key_col])
                        .set_index(key_col)
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("attr-agreement keying failed key=%s", key_col, exc_info=True)
                    continue
                # Align on common keys.
                common = a_keyed.index.intersection(b_keyed.index)
                if len(common) < min_shared_keys:
                    continue
                a_sub = a_keyed.loc[common]
                b_sub = b_keyed.loc[common]
                for attr_col in shared_cols:
                    if attr_col not in a_sub.columns or attr_col not in b_sub.columns:
                        continue
                    # Compare as strings to handle dtype quirks.
                    aa = a_sub[attr_col].astype(str)
                    bb = b_sub[attr_col].astype(str)
                    # Normalize for the comparison so trivial casing
                    # / whitespace differences don't dominate the
                    # signal; expose RAW differences as evidence.
                    aa_norm = aa.map(normalize_value)
                    bb_norm = bb.map(normalize_value)
                    diff = (aa_norm.values != bb_norm.values)
                    n_conflicts = int(diff.sum())
                    if n_conflicts == 0:
                        continue
                    sample: List[Tuple[str, str, str]] = []
                    for k_idx, is_diff in zip(common, diff):
                        if not is_diff:
                            continue
                        sample.append(
                            (str(k_idx), str(aa.loc[k_idx]), str(bb.loc[k_idx]))
                        )
                        if len(sample) >= max_sample_conflicts:
                            break
                    out.append(
                        AttributeAgreementCandidate(
                            sheet_a=a, sheet_b=b,
                            key_columns=(key_col,),
                            attribute_column=str(attr_col),
                            n_shared_keys=int(len(common)),
                            n_conflicts=n_conflicts,
                            sample_conflicts=tuple(sample),
                        )
                    )

    out.sort(key=lambda c: -c.n_conflicts)
    return out[:max_candidates]


# ---------------------------------------------------------------------------
# Detector 5: composite-key foreign-key candidates (Gap F)
# ---------------------------------------------------------------------------


def find_composite_fk_candidates(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_candidates: int = 30,
    min_coverage: float = 0.5,
    max_subset_size: int = 3,
    max_orphan_examples: int = 5,
    parent_value_cap: int = 1000,
) -> List[CompositeFKCandidate]:
    """Find multi-column FK relationships across sheets.

    Strategy: enumerate column subsets up to ``max_subset_size`` from
    each sheet's identity-like columns. For each subset present in
    both sheets, treat the parent's tuple set as the controlled set
    and compute coverage of the child's tuples.

    Single-column FKs are skipped here (handled by
    :func:`find_fk_candidates`).
    """
    sheets_list = list(sheets)
    if len(sheets_list) < 2:
        return []

    from itertools import combinations

    out: List[CompositeFKCandidate] = []
    for parent in sheets_list:
        p_df = frames.get(parent)
        if p_df is None or p_df.empty:
            continue
        d_p = digests.get(parent)
        p_ids = list(d_p.candidate_identity_columns) if d_p else []
        if len(p_ids) < 2:
            continue
        # Enumerate subsets of size 2..max_subset_size.
        for k in range(2, max_subset_size + 1):
            for subset in combinations(p_ids, k):
                subset_t = tuple(str(c) for c in subset)
                # Need all columns present in parent.
                if not all(c in p_df.columns for c in subset_t):
                    continue
                try:
                    p_tuples = (
                        p_df[list(subset_t)]
                        .dropna()
                        .astype(str)
                        .apply(tuple, axis=1)
                        .unique()
                        .tolist()
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("composite-FK parent tuples failed subset=%s", subset_t, exc_info=True)
                    continue
                if len(p_tuples) < 3:
                    continue
                p_tuples = p_tuples[:parent_value_cap]
                p_set = set(p_tuples)

                for child in sheets_list:
                    if child == parent:
                        continue
                    c_df = frames.get(child)
                    if c_df is None or c_df.empty:
                        continue
                    if not all(c in c_df.columns for c in subset_t):
                        continue
                    try:
                        c_pairs = (
                            c_df[list(subset_t)]
                            .dropna()
                            .astype(str)
                        )
                        c_tuples = c_pairs.apply(tuple, axis=1)
                    except Exception:  # noqa: BLE001
                        logger.debug("composite-FK child tuples failed subset=%s", subset_t, exc_info=True)
                        continue
                    if c_tuples.empty:
                        continue
                    n_distinct_c = int(c_tuples.nunique())
                    if n_distinct_c < 3:
                        continue
                    in_set = c_tuples.isin(p_set)
                    cov = float(in_set.sum()) / max(1, len(c_tuples))
                    if cov < min_coverage or cov >= 1.0:
                        continue
                    orphan_mask = ~in_set
                    orphan_idx = list(c_tuples[orphan_mask].index)[:max_orphan_examples]
                    orphan_keys = (
                        c_tuples[orphan_mask]
                        .drop_duplicates()
                        .head(max_orphan_examples)
                        .tolist()
                    )
                    out.append(
                        CompositeFKCandidate(
                            parent=parent,
                            parent_columns=subset_t,
                            child=child,
                            child_columns=subset_t,
                            n_parent_distinct=len(p_tuples),
                            n_child_distinct=n_distinct_c,
                            overlap=int(c_tuples[in_set].nunique()),
                            coverage=cov,
                            orphan_row_indices=tuple(int(i) for i in orphan_idx),
                            orphan_keys=tuple(tuple(k) for k in orphan_keys),
                        )
                    )

    out.sort(key=lambda c: (-c.coverage, c.parent, c.child))
    return out[:max_candidates]


# ---------------------------------------------------------------------------
# Detector 6: asymmetric containment (Gap F)
# ---------------------------------------------------------------------------


def find_containment_candidates(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_candidates: int = 50,
    min_one_sided: float = 0.3,
    min_distinct: int = 5,
    column_cap: int = 30,
) -> List[ContainmentCandidate]:
    """Find column pairs where one side is mostly contained in the
    other after normalization, but symmetric Jaccard is small.

    Surfaces small dimension tables joining into large fact tables
    (``|dim ∩ fact| / |dim| ≈ 1`` but Jaccard tiny because fact has
    many other values). Different from ``find_fk_candidates`` because
    it runs over ALL string/int column pairs, not just digest-flagged
    identity columns.
    """
    sheets_list = list(sheets)
    if len(sheets_list) < 2:
        return []

    out: List[ContainmentCandidate] = []
    for i in range(len(sheets_list)):
        for j in range(i + 1, len(sheets_list)):
            a = sheets_list[i]
            b = sheets_list[j]
            df_a = frames.get(a)
            df_b = frames.get(b)
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue
            cols_a = [str(c) for c in list(df_a.columns)[:column_cap]]
            cols_b = [str(c) for c in list(df_b.columns)[:column_cap]]
            for ca in cols_a:
                if not (pd.api.types.is_string_dtype(df_a[ca])
                        or pd.api.types.is_integer_dtype(df_a[ca])):
                    continue
                va = _column_value_set(df_a, ca, cap=500)
                if len(va) < min_distinct:
                    continue
                for cb in cols_b:
                    if not (pd.api.types.is_string_dtype(df_b[cb])
                            or pd.api.types.is_integer_dtype(df_b[cb])):
                        continue
                    vb = _column_value_set(df_b, cb, cap=500)
                    if len(vb) < min_distinct:
                        continue
                    # Pick the rung that maximises one-sided coverage
                    # (in either direction) and report both directions.
                    best_step = "raw"
                    best_one_sided = 0.0
                    best_a_in_b = 0.0
                    best_b_in_a = 0.0
                    rungs = ["raw"] + [n for n, _ in NORMALIZATION_LADDER]
                    for step in rungs:
                        if step == "raw":
                            sa = set(str(v) for v in va)
                            sb = set(str(v) for v in vb)
                        else:
                            sa = {normalize_value(v, up_to=step) for v in va}
                            sb = {normalize_value(v, up_to=step) for v in vb}
                            sa.discard("")
                            sb.discard("")
                        if not sa or not sb:
                            continue
                        a_in_b = len(sa & sb) / len(sa)
                        b_in_a = len(sa & sb) / len(sb)
                        one_sided = max(a_in_b, b_in_a)
                        if one_sided > best_one_sided:
                            best_one_sided = one_sided
                            best_step = step
                            best_a_in_b = a_in_b
                            best_b_in_a = b_in_a
                    if best_one_sided < min_one_sided:
                        continue
                    # Skip pairs with full bidirectional containment
                    # (raw or normalized) on identically-named columns
                    # since those are symmetric and uninteresting.
                    if (best_a_in_b >= 0.999 and best_b_in_a >= 0.999
                            and ca == cb):
                        continue
                    out.append(
                        ContainmentCandidate(
                            sheet_a=a, column_a=ca,
                            sheet_b=b, column_b=cb,
                            n_distinct_a=len(va),
                            n_distinct_b=len(vb),
                            coverage_a_in_b=float(best_a_in_b),
                            coverage_b_in_a=float(best_b_in_a),
                            best_normalization_step=best_step,
                        )
                    )

    out.sort(
        key=lambda c: -max(c.coverage_a_in_b, c.coverage_b_in_a)
    )
    return out[:max_candidates]


# ---------------------------------------------------------------------------
# Detector 7: time-window overlap (Gap F)
# ---------------------------------------------------------------------------


def find_time_overlap_candidates(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_candidates: int = 50,
) -> List[TimeOverlapCandidate]:
    """Find date-column pairs across sibling sheets whose ranges
    overlap, suggesting union-style stacking with potential
    duplicate snapshots over the overlap window.
    """
    sheets_list = list(sheets)
    if len(sheets_list) < 2:
        return []

    out: List[TimeOverlapCandidate] = []
    # Pre-parse date columns per sheet.
    parsed: Dict[str, Dict[str, pd.Series]] = {}
    for s in sheets_list:
        df = frames.get(s)
        if df is None or df.empty:
            continue
        per_col: Dict[str, pd.Series] = {}
        for c in df.columns:
            col = df[c]
            if pd.api.types.is_datetime64_any_dtype(col):
                per_col[str(c)] = col.dropna()
                continue
            if pd.api.types.is_string_dtype(col):
                # Try a cheap parse on a sample; only commit if a
                # meaningful fraction parses.
                sample = col.dropna().astype(str).head(50)
                if sample.empty:
                    continue
                # ``format="mixed"`` (pandas 2.0+) tells pandas we
                # KNOW the column may contain heterogeneous date
                # formats, so it shouldn't emit the "Could not infer
                # format, falling back to dateutil" UserWarning. We
                # used to wrap the call in ``catch_warnings()`` but
                # that context manager mutates the process-wide
                # warning filter list and is not thread-safe; the
                # priors run inside a ThreadPoolExecutor in
                # ``run_relation_discovery``, so the filter window
                # could close before the warning fired and the
                # message would leak to stderr. ``format="mixed"``
                # silences the warning at its source, which is both
                # cheaper and thread-safe.
                parsed_sample = pd.to_datetime(
                    sample, errors="coerce", format="mixed",
                )
                if parsed_sample.notna().mean() < 0.7:
                    continue
                full = pd.to_datetime(
                    col, errors="coerce", format="mixed",
                ).dropna()
                if not full.empty:
                    per_col[str(c)] = full
        if per_col:
            parsed[s] = per_col

    for i in range(len(sheets_list)):
        for j in range(i + 1, len(sheets_list)):
            a = sheets_list[i]
            b = sheets_list[j]
            cols_a = parsed.get(a, {})
            cols_b = parsed.get(b, {})
            if not cols_a or not cols_b:
                continue
            for ca, sa in cols_a.items():
                for cb, sb in cols_b.items():
                    if sa.empty or sb.empty:
                        continue
                    a_min, a_max = sa.min(), sa.max()
                    b_min, b_max = sb.min(), sb.max()
                    overlap_lo = max(a_min, b_min)
                    overlap_hi = min(a_max, b_max)
                    if overlap_lo > overlap_hi:
                        continue
                    in_a = sa[(sa >= overlap_lo) & (sa <= overlap_hi)]
                    in_b = sb[(sb >= overlap_lo) & (sb <= overlap_hi)]
                    out.append(
                        TimeOverlapCandidate(
                            sheet_a=a, column_a=ca,
                            sheet_b=b, column_b=cb,
                            range_a=(str(a_min)[:10], str(a_max)[:10]),
                            range_b=(str(b_min)[:10], str(b_max)[:10]),
                            overlap=(str(overlap_lo)[:10], str(overlap_hi)[:10]),
                            n_rows_a_in_overlap=int(len(in_a)),
                            n_rows_b_in_overlap=int(len(in_b)),
                        )
                    )

    out.sort(
        key=lambda c: -(c.n_rows_a_in_overlap + c.n_rows_b_in_overlap)
    )
    return out[:max_candidates]


# ---------------------------------------------------------------------------
# Detector 8: null-sentinel drift (Gap F)
# ---------------------------------------------------------------------------


# Tokens that frequently encode "missing" outside of pandas's own
# NaN handling. Lowercased; comparison is case-insensitive.
_SENTINEL_CANDIDATES: Tuple[str, ...] = (
    "na", "n/a", "n.a.", "n/a.", "null", "none", "nil",
    "-", "--", "?", "??", ".", "..",
    "unknown", "unk", "tbd", "missing", "n/k", "nk",
    "0", "-1", "9999",  # numeric sentinels (string-cast)
)


def find_null_sentinel_candidates(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_candidates: int = 50,
    min_freq: int = 2,
) -> List[NullSentinelCandidate]:
    """Find shared columns where two sheets use different sentinel
    tokens for missing values.

    For each shared column, look for the most-frequent sentinel-like
    token in each sheet (above ``min_freq``). If the two sheets pick
    *different* sentinels, flag a candidate.
    """
    sheets_list = list(sheets)
    if len(sheets_list) < 2:
        return []
    sent_set = set(_SENTINEL_CANDIDATES)

    def dominant_sentinel(series: pd.Series) -> Optional[Tuple[str, int]]:
        try:
            counts = (
                series.dropna().astype(str).str.strip().str.lower()
                .value_counts()
            )
        except Exception:  # noqa: BLE001
            return None
        for token, freq in counts.items():
            if token in sent_set and freq >= min_freq:
                return (str(token), int(freq))
        return None

    out: List[NullSentinelCandidate] = []
    for i in range(len(sheets_list)):
        for j in range(i + 1, len(sheets_list)):
            a = sheets_list[i]
            b = sheets_list[j]
            df_a = frames.get(a)
            df_b = frames.get(b)
            if df_a is None or df_b is None:
                continue
            shared = set(df_a.columns) & set(df_b.columns)
            for col in shared:
                col_str = str(col)
                d_a = dominant_sentinel(df_a[col])
                d_b = dominant_sentinel(df_b[col])
                if d_a is None or d_b is None:
                    continue
                if d_a[0] == d_b[0]:
                    continue
                out.append(
                    NullSentinelCandidate(
                        column=col_str, sheet_a=a, sheet_b=b,
                        sentinel_a=d_a[0], freq_a=d_a[1],
                        sentinel_b=d_b[0], freq_b=d_b[1],
                    )
                )

    out.sort(key=lambda c: -(c.freq_a + c.freq_b))
    return out[:max_candidates]


# ---------------------------------------------------------------------------
# Detector 9: embedding-residual matches (Gap C)
# ---------------------------------------------------------------------------


def find_embedding_match_candidates(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_candidates: int = 30,
    max_residual_per_column: int = 50,
) -> List[EmbeddingMatchCandidate]:
    """Run the registered embedding matcher on residual values
    (those that survived all string-normalization rungs without
    matching). Returns empty when no matcher is registered.
    """
    if _EMBEDDING_MATCHER is None:
        return []
    sheets_list = list(sheets)
    if len(sheets_list) < 2:
        return []

    out: List[EmbeddingMatchCandidate] = []
    for i in range(len(sheets_list)):
        for j in range(i + 1, len(sheets_list)):
            a = sheets_list[i]
            b = sheets_list[j]
            df_a = frames.get(a)
            df_b = frames.get(b)
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue
            shared = set(df_a.columns) & set(df_b.columns)
            for col in shared:
                col_str = str(col)
                if not pd.api.types.is_string_dtype(df_a[col]):
                    continue
                if not pd.api.types.is_string_dtype(df_b[col]):
                    continue
                va = _column_value_set(df_a, col_str, cap=200)
                vb = _column_value_set(df_b, col_str, cap=200)
                if not va or not vb:
                    continue
                # Residual = values that are NOT matched by any
                # normalization rung. Saves embedding cost on the
                # easy cases.
                norm_a = _normalize_set(va)
                norm_b = _normalize_set(vb)
                shared_norm = norm_a & norm_b
                resid_a = [v for v in va if normalize_value(v) not in shared_norm]
                resid_b = [v for v in vb if normalize_value(v) not in shared_norm]
                if not resid_a or not resid_b:
                    continue
                resid_a = resid_a[:max_residual_per_column]
                resid_b = resid_b[:max_residual_per_column]
                matches = _embedding_residual_matches(resid_a, resid_b)
                if not matches:
                    continue
                out.append(
                    EmbeddingMatchCandidate(
                        sheet_a=a, column_a=col_str,
                        sheet_b=b, column_b=col_str,
                        matches=tuple(matches),
                    )
                )
    return out[:max_candidates]


# ---------------------------------------------------------------------------
# Detector 10: unionable group candidates
# ---------------------------------------------------------------------------


def _normalize_header(h: Any) -> str:
    """Header normalization for cross-sheet alignment.

    Lowercases, ASCII-folds, drops punctuation, and collapses
    whitespace. Two headers normalize to the same string when they
    are likely surface variants of the same concept (e.g.
    ``"Reading Quiz 1 %"``, ``"Reading Quiz 1 (%)"``,
    ``"Reading Quiz 1 (/10%)2"`` -> the first two collapse, the
    third doesn't because the trailing ``2`` survives).
    """
    if h is None:
        return ""
    s = str(h)
    for _name, fn in NORMALIZATION_LADDER:
        # Skip the alias rung - it's tuned for value-level surface
        # forms (US/USA), not header tokens, and would mis-fold
        # things like "kg" -> a unit canonical when the column
        # carries kg-vs-lb data.
        if _name == "alias":
            continue
        s = fn(s)
    return s


def find_unionable_groups(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    min_jaccard: float = 0.25,
    min_group_size: int = 2,
    max_groups: int = 16,
) -> List[UnionableGroupCandidate]:
    """Cluster sibling sheets that share enough headers to be
    candidates for a UNION.

    Two sheets land in the same group when their normalized-header
    Jaccard exceeds ``min_jaccard``. Clustering uses a simple
    union-find over the pairwise edge graph; transitive closure
    keeps the loop O(n^2) which is fine at our sheet cap.

    The Jaccard cutoff is intentionally permissive (``0.25``) so the
    LLM can still see partial-overlap groups that need NaN-fills,
    including sheets where multi-line / parenthetical header drift
    pushes raw-header Jaccard well below 0.5. The downstream LLM
    resolver is the actual gatekeeper: it sees the candidate group
    plus row samples and may mark it ``confirmed=false``.
    """
    if len(sheets) < 2:
        return []

    headers_norm: Dict[str, set] = {}
    headers_raw: Dict[str, Tuple[str, ...]] = {}
    for s in sheets:
        df = frames.get(s)
        if df is None or df.empty:
            continue
        raw = tuple(str(c) for c in df.columns)
        norm = {_normalize_header(c) for c in raw}
        norm.discard("")
        if not norm:
            continue
        headers_raw[s] = raw
        headers_norm[s] = norm

    sheet_list = [s for s in sheets if s in headers_norm]
    n = len(sheet_list)
    if n < min_group_size:
        return []

    # Pairwise Jaccard + union-find.
    parent_idx = list(range(n))

    def _find(i: int) -> int:
        while parent_idx[i] != i:
            parent_idx[i] = parent_idx[parent_idx[i]]
            i = parent_idx[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent_idx[ri] = rj

    pair_jaccard: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        si = sheet_list[i]
        for j in range(i + 1, n):
            sj = sheet_list[j]
            a, b = headers_norm[si], headers_norm[sj]
            inter = len(a & b)
            union = len(a | b)
            jac = (inter / union) if union else 0.0
            pair_jaccard[(i, j)] = jac
            if jac >= min_jaccard:
                _union(i, j)

    # Bucket sheets by root.
    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        buckets.setdefault(_find(i), []).append(i)

    out: List[UnionableGroupCandidate] = []
    gid_counter = 0
    for _root, members in buckets.items():
        if len(members) < min_group_size:
            continue
        members_sorted = sorted(members, key=lambda k: sheet_list[k])
        member_sheets = tuple(sheet_list[k] for k in members_sorted)
        norm_sets = [headers_norm[s] for s in member_sheets]
        canonical_union: set = set()
        for ns in norm_sets:
            canonical_union |= ns
        if not norm_sets:
            continue
        canonical_inter: set = set(norm_sets[0])
        for ns in norm_sets[1:]:
            canonical_inter &= ns
        # Average pairwise Jaccard over members in this group.
        pair_vals: List[float] = []
        for ai in range(len(members_sorted)):
            for bi in range(ai + 1, len(members_sorted)):
                a_idx = members_sorted[ai]
                b_idx = members_sorted[bi]
                key = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
                pair_vals.append(pair_jaccard.get(key, 0.0))
        avg_jac = sum(pair_vals) / len(pair_vals) if pair_vals else 0.0
        gid_counter += 1
        out.append(
            UnionableGroupCandidate(
                group_id=f"u{gid_counter}",
                sheets=member_sheets,
                n_sheets=len(member_sheets),
                n_canonical_headers=len(canonical_union),
                n_shared_headers=len(canonical_inter),
                avg_jaccard=avg_jac,
                sheet_headers=tuple(
                    (s, headers_raw[s]) for s in member_sheets
                ),
            )
        )
        if len(out) >= max_groups:
            break

    # Sort: larger groups first, then higher Jaccard, then sheet-name.
    out.sort(
        key=lambda c: (-c.n_sheets, -c.avg_jaccard, c.sheets),
    )
    return out[:max_groups]


# ---------------------------------------------------------------------------
# Top-level orchestrator + Markdown renderer
# ---------------------------------------------------------------------------


def compute_table_priors(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    error_type: str,
    workbook: str,
    sheet_cap: int = 24,
) -> MultiTablePriors:
    """Run every detector that is useful for ``error_type``.

    Sheets are ranked by row count; only the top ``sheet_cap`` are
    used in the pairwise scan. ``sheet_cap`` was widened from 12 to
    24 to reduce the chance that a relevant sibling sheet is dropped
    by the pre-pairing truncation (recall-first).
    """
    if len(sheets) < 2:
        return MultiTablePriors(error_type=error_type, workbook=workbook)

    chosen = _select_representative_sheets(sheets, frames, cap=sheet_cap)

    fks: List[FKCandidate] = []
    naming: List[NamingCandidate] = []
    repres: List[RepresentationCandidate] = []
    attr_agree: List[AttributeAgreementCandidate] = []
    comp_fk: List[CompositeFKCandidate] = []
    contain: List[ContainmentCandidate] = []
    time_overlap: List[TimeOverlapCandidate] = []
    null_sent: List[NullSentinelCandidate] = []
    emb_match: List[EmbeddingMatchCandidate] = []
    union_groups: List[UnionableGroupCandidate] = []

    if error_type == "inter_table_inconsistency":
        fks = find_fk_candidates(
            sheets=chosen, frames=frames, digests=digests,
        )
        naming = find_naming_conflicts(
            sheets=chosen, frames=frames, digests=digests,
        )
        repres = find_representation_conflicts(
            sheets=chosen, frames=frames, digests=digests,
        )
        attr_agree = find_attribute_agreement_conflicts(
            sheets=chosen, frames=frames, digests=digests,
        )
        comp_fk = find_composite_fk_candidates(
            sheets=chosen, frames=frames, digests=digests,
        )
        contain = find_containment_candidates(
            sheets=chosen, frames=frames, digests=digests,
        )
        time_overlap = find_time_overlap_candidates(
            sheets=chosen, frames=frames, digests=digests,
        )
        null_sent = find_null_sentinel_candidates(
            sheets=chosen, frames=frames, digests=digests,
        )
        emb_match = find_embedding_match_candidates(
            sheets=chosen, frames=frames, digests=digests,
        )
        union_groups = find_unionable_groups(
            sheets=chosen, frames=frames,
        )

    return MultiTablePriors(
        error_type=error_type,
        workbook=workbook,
        sheets=tuple(chosen),
        fk_candidates=tuple(fks),
        naming_candidates=tuple(naming),
        representation_candidates=tuple(repres),
        attribute_agreement_candidates=tuple(attr_agree),
        composite_fk_candidates=tuple(comp_fk),
        containment_candidates=tuple(contain),
        time_overlap_candidates=tuple(time_overlap),
        null_sentinel_candidates=tuple(null_sent),
        embedding_match_candidates=tuple(emb_match),
        unionable_groups=tuple(union_groups),
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_workbook_context(
    *,
    workbook: str,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    max_cols_per_sheet: int = 12,
) -> str:
    """Render the workbook-level overview shown to the LLM.

    For each sheet, list shape, headers (capped), and dtypes. The
    full per-sheet digest is omitted - the LLM doesn't need every
    column's value distribution to reason about cross-sheet
    relations; just enough to identify candidate join columns.
    """
    parts: List[str] = []
    parts.append(
        f"WORKBOOK CONTEXT  workbook={workbook!r}  sheets={len(sheets)}"
    )
    for s in sheets:
        df = frames.get(s)
        if df is None:
            continue
        n_rows, n_cols = df.shape
        cols = list(df.columns)[:max_cols_per_sheet]
        dtypes = [str(df[c].dtype) for c in cols]
        more = (
            f"  (+{n_cols - len(cols)} more)"
            if n_cols > len(cols) else ""
        )
        parts.append(f"\n  --- sheet {s!r}: {n_rows} rows x {n_cols} cols ---")
        # Pair each header with its dtype.
        for col, dt in zip(cols, dtypes):
            parts.append(f"    - {col!r}: {dt}")
        if more:
            parts.append(f"    {more}")

        # Identity-column hint from digest (cross-sheet FK candidates
        # most often live on these).
        d = digests.get(s)
        if d is not None and d.candidate_identity_columns:
            parts.append(
                "    identity-like columns: "
                + ", ".join(repr(c) for c in d.candidate_identity_columns[:5])
            )
    return "\n".join(parts) + "\n"


def render_table_priors_block(priors: MultiTablePriors) -> str:
    """Render the cross-table structural candidates as a Markdown
    section for the discovery prompt."""
    if priors.is_empty():
        return (
            "STRUCTURAL CROSS-TABLE CANDIDATES (pre-computed; you may use, "
            "refine, or reject)\n"
            "  (none found by the heuristic detector)\n"
        )

    parts: List[str] = [
        "STRUCTURAL CROSS-TABLE CANDIDATES (pre-computed; you may use, "
        "refine, or reject)"
    ]

    if priors.fk_candidates:
        parts.append("\nForeign-key candidates (child references parent):")
        for fk in priors.fk_candidates:
            orphan_idx = (
                ", ".join(str(i) for i in fk.orphan_row_indices)
                if fk.orphan_row_indices else "(none)"
            )
            orphan_vals = (
                ", ".join(repr(v) for v in fk.orphan_values)
                if fk.orphan_values else "(none)"
            )
            norm_note = ""
            if (fk.normalized_coverage > fk.coverage + 0.01
                    and fk.best_normalization_step != "raw"):
                norm_note = (
                    f"  [normalized coverage={fk.normalized_coverage:.0%}"
                    f" via '{fk.best_normalization_step}' rung;"
                    f" surface-form mismatch likely]"
                )
            parts.append(
                f"  - parent={fk.parent!r}.{fk.parent_column!r} "
                f"({fk.n_parent_distinct} distinct)  <-  "
                f"child={fk.child!r}.{fk.child_column!r} "
                f"({fk.n_child_distinct} distinct, "
                f"coverage={fk.coverage:.0%}){norm_note}"
            )
            parts.append(
                f"    orphan rows in child: [{orphan_idx}]"
            )
            parts.append(
                f"    orphan values: {{{orphan_vals}}}"
            )

    if priors.naming_candidates:
        parts.append("\nNaming-conflict candidates:")
        for nc in priors.naming_candidates:
            norm_note = ""
            if (nc.normalized_jaccard > nc.jaccard + 0.01
                    and nc.best_normalization_step != "raw"):
                norm_note = (
                    f" [normalized={nc.normalized_jaccard:.2f}"
                    f" via '{nc.best_normalization_step}']"
                )
            if nc.relation == "synonym":
                parts.append(
                    f"  - synonym?  {nc.sheet_a!r}.{nc.column_a!r}  ~  "
                    f"{nc.sheet_b!r}.{nc.column_b!r}  "
                    f"(value-set Jaccard={nc.jaccard:.2f}){norm_note}"
                )
            else:
                parts.append(
                    f"  - homonym?  header {nc.column_a!r} appears in "
                    f"{nc.sheet_a!r} and {nc.sheet_b!r} with disjoint "
                    f"vocabularies (Jaccard={nc.jaccard:.2f}){norm_note}"
                )

    if priors.representation_candidates:
        parts.append("\nRepresentation-conflict candidates:")
        for rc in priors.representation_candidates:
            parts.append(
                f"  - column {rc.column!r}: "
                f"{rc.sheet_a!r} {rc.detail_a}  vs  "
                f"{rc.sheet_b!r} {rc.detail_b}  ({rc.reason})"
            )

    if priors.attribute_agreement_candidates:
        parts.append(
            "\nAttribute-agreement candidates (rows in two sheets with "
            "the same identity disagree on a descriptive attribute):"
        )
        for ac in priors.attribute_agreement_candidates:
            keys = "+".join(ac.key_columns)
            sample = "; ".join(
                f"{k!r}: {a!r} vs {b!r}"
                for k, a, b in ac.sample_conflicts[:3]
            )
            parts.append(
                f"  - key={keys}  attr={ac.attribute_column!r}  "
                f"{ac.sheet_a!r} vs {ac.sheet_b!r}  "
                f"({ac.n_conflicts}/{ac.n_shared_keys} shared keys conflict)"
            )
            if sample:
                parts.append(f"    examples: {sample}")

    if priors.composite_fk_candidates:
        parts.append("\nComposite-key foreign-key candidates:")
        for cfk in priors.composite_fk_candidates:
            cols = "+".join(cfk.parent_columns)
            parts.append(
                f"  - parent={cfk.parent!r}.[{cols}] "
                f"({cfk.n_parent_distinct} distinct tuples)  <-  "
                f"child={cfk.child!r}.[{cols}] "
                f"({cfk.n_child_distinct} distinct, coverage={cfk.coverage:.0%})"
            )
            if cfk.orphan_keys:
                ex = "; ".join("(" + ", ".join(repr(v) for v in k) + ")"
                               for k in cfk.orphan_keys[:3])
                parts.append(f"    orphan tuples: {ex}")

    if priors.containment_candidates:
        parts.append(
            "\nAsymmetric-containment candidates (one column mostly "
            "contained in the other after normalization):"
        )
        for cc in priors.containment_candidates:
            parts.append(
                f"  - {cc.sheet_a!r}.{cc.column_a!r} "
                f"({cc.n_distinct_a} distinct)  ~  "
                f"{cc.sheet_b!r}.{cc.column_b!r} "
                f"({cc.n_distinct_b} distinct)  "
                f"A->B={cc.coverage_a_in_b:.0%}, "
                f"B->A={cc.coverage_b_in_a:.0%} "
                f"[via '{cc.best_normalization_step}']"
            )

    if priors.time_overlap_candidates:
        parts.append(
            "\nTime-window overlap candidates (sibling sheets covering "
            "overlapping date ranges; possible duplicate snapshots):"
        )
        for tc in priors.time_overlap_candidates:
            parts.append(
                f"  - {tc.sheet_a!r}.{tc.column_a!r} {tc.range_a}  &  "
                f"{tc.sheet_b!r}.{tc.column_b!r} {tc.range_b}  "
                f"overlap={tc.overlap}  "
                f"({tc.n_rows_a_in_overlap} + {tc.n_rows_b_in_overlap} rows)"
            )

    if priors.null_sentinel_candidates:
        parts.append(
            "\nNull-sentinel-drift candidates (different missing-value "
            "tokens across sibling sheets):"
        )
        for nsc in priors.null_sentinel_candidates:
            parts.append(
                f"  - column {nsc.column!r}: "
                f"{nsc.sheet_a!r} uses {nsc.sentinel_a!r} "
                f"(freq={nsc.freq_a})  vs  "
                f"{nsc.sheet_b!r} uses {nsc.sentinel_b!r} "
                f"(freq={nsc.freq_b})"
            )

    if priors.embedding_match_candidates:
        parts.append(
            "\nEmbedding-residual matches (surface-disjoint values "
            "with high embedding similarity):"
        )
        for ec in priors.embedding_match_candidates:
            sample = "; ".join(
                f"{a!r}~{b!r} ({s:.2f})"
                for a, b, s in ec.matches[:3]
            )
            parts.append(
                f"  - {ec.sheet_a!r}.{ec.column_a!r}  ~  "
                f"{ec.sheet_b!r}.{ec.column_b!r}  examples: {sample}"
            )

    return "\n".join(parts) + "\n"
