"""Cheap structural detectors that surface candidate groupings for
multi-row relation discovery.

Multi-row error types - currently just ``row_duplication`` - assert
predicates over *sets* of rows that share a key. To cut the LLM's job
from "find the keys AND name the predicate" down to "given these
keys, name the predicate", we precompute a small set of candidate
group structures and present them to the discovery prompt:

    1. Duplicate-prone subsets - column subsets ``K`` where
       ``df.duplicated(subset=K, keep=False)`` flags real groups,
       with two duplicate groups shown verbatim.

Legacy detectors for total-row blocks and identity / descriptive
conflicts are retained in this module so historical relations stay
parseable, but they are no longer wired into ``compute_priors`` for
the active taxonomy.

The detectors are pure, deterministic, and run in a few hundred
milliseconds on tables up to ~5k rows. They reuse the structural
hints already computed by :mod:`src.core.table_digest`
(``total_row_indices``, ``candidate_identity_columns``,
``candidate_fds``) and add the duplicate / conflict scans that the
digest does not perform.

The output is a small dataclass with both a structured form (for
tests / future tooling) and a Markdown-rendered form (for the
discovery prompt).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..table_digest import TableDigest

logger = logging.getLogger("multi_row_priors")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TotalRowBlock:
    """A candidate aggregate group: one total/summary row + the
    immediately preceding non-summary rows that plausibly aggregate
    into it."""

    total_row: int
    label: str
    detail_rows: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_row": int(self.total_row),
            "label": self.label,
            "detail_rows": list(self.detail_rows),
        }


@dataclass(frozen=True)
class DuplicateSubset:
    """A column subset on which exact duplicates exist, with two
    example groups shown verbatim."""

    columns: Tuple[str, ...]
    n_duplicate_rows: int
    n_groups: int
    dup_rate: float
    sample_groups: Tuple[Tuple[int, ...], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": list(self.columns),
            "n_duplicate_rows": int(self.n_duplicate_rows),
            "n_groups": int(self.n_groups),
            "dup_rate": float(self.dup_rate),
            "sample_groups": [list(g) for g in self.sample_groups],
        }


@dataclass(frozen=True)
class IdentityConflict:
    """An identity-key candidate that has at least one group whose
    rows disagree on one or more descriptive columns."""

    key_columns: Tuple[str, ...]
    descriptive_columns: Tuple[str, ...]
    n_conflict_groups: int
    sample_groups: Tuple[Tuple[int, ...], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_columns": list(self.key_columns),
            "descriptive_columns": list(self.descriptive_columns),
            "n_conflict_groups": int(self.n_conflict_groups),
            "sample_groups": [list(g) for g in self.sample_groups],
        }


@dataclass(frozen=True)
class MultiRowPriors:
    """All structural priors for one (source, error_type) discovery
    call. Some lists may be empty: an empty priors block is a
    legitimate signal to the LLM that the heuristic detector found
    nothing."""

    error_type: str
    source: str
    total_row_blocks: Tuple[TotalRowBlock, ...] = field(default_factory=tuple)
    duplicate_subsets: Tuple[DuplicateSubset, ...] = field(default_factory=tuple)
    identity_conflicts: Tuple[IdentityConflict, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not (
            self.total_row_blocks
            or self.duplicate_subsets
            or self.identity_conflicts
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "source": self.source,
            "total_row_blocks": [b.to_dict() for b in self.total_row_blocks],
            "duplicate_subsets": [s.to_dict() for s in self.duplicate_subsets],
            "identity_conflicts": [c.to_dict() for c in self.identity_conflicts],
        }


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


# Same regex the digest uses for total-row labels, kept private so
# we can tighten it independently of the digest if needed.
_TOTAL_LABEL_RE = re.compile(
    r"\b(total|totals|subtotal|sub[- ]?total|grand[- ]?total|"
    r"sum|sums|summary|aggregate|fy|year[- ]?to[- ]?date|ytd)\b",
    re.IGNORECASE,
)


def _label_for_total_row(df: pd.DataFrame, row_idx: int) -> str:
    """Return the leading-text-column value of ``row_idx`` (or an
    empty string when the row is purely numeric)."""
    if row_idx < 0 or row_idx >= len(df):
        return ""
    for col in df.columns:
        v = df.iloc[row_idx][col]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        # First non-empty cell - typically the row label.
        return s
    return ""


def find_total_row_blocks(
    df: pd.DataFrame,
    *,
    digest: Optional[TableDigest] = None,
    max_blocks: int = 6,
    max_detail_rows: int = 6,
) -> List[TotalRowBlock]:
    """Pair each total/summary row with the contiguous run of
    non-summary rows immediately preceding it.

    The digest already detects total rows (``digest.total_row_indices``);
    we reuse that list when available and otherwise fall back to a
    regex scan over the leading text column.
    """
    if df.empty:
        return []

    if digest is not None and digest.total_row_indices:
        total_rows = list(digest.total_row_indices)
    else:
        total_rows = []
        leading_text_col = next(
            (c for c in df.columns if pd.api.types.is_string_dtype(df[c])),
            None,
        )
        if leading_text_col is not None:
            for i in range(len(df)):
                v = df.iloc[i][leading_text_col]
                if pd.isna(v):
                    continue
                if _TOTAL_LABEL_RE.search(str(v)):
                    total_rows.append(i)

    total_set = set(total_rows)
    blocks: List[TotalRowBlock] = []
    for tr in total_rows[:max_blocks]:
        # Walk backwards from tr-1 collecting consecutive non-total
        # rows up to ``max_detail_rows``.
        details: List[int] = []
        i = tr - 1
        while i >= 0 and len(details) < max_detail_rows and i not in total_set:
            details.append(i)
            i -= 1
        details.reverse()
        if not details:
            # A total row with nothing to aggregate is still useful
            # context (the LLM may flag it as orphaned), so keep it
            # but with empty detail rows.
            pass
        blocks.append(
            TotalRowBlock(
                total_row=int(tr),
                label=_label_for_total_row(df, tr),
                detail_rows=tuple(details),
            )
        )
    return blocks


def find_duplicate_subsets(
    df: pd.DataFrame,
    *,
    digest: Optional[TableDigest] = None,
    max_subsets: int = 4,
    max_subset_size: int = 2,
    min_dup_rows: int = 2,
    max_sample_groups: int = 2,
) -> List[DuplicateSubset]:
    """Find column subsets with non-trivial duplicate row groups.

    Strategy:
      * Start with single columns whose dtype is object/int (string
        identifiers / numeric IDs). Skip floats and datetimes.
      * Limit the search to identity-like columns from the digest
        when available, otherwise to the top-K by uniqueness ratio
        among non-float columns. This keeps the scan O(n_cols).
      * For each candidate, compute ``df.duplicated(subset=K, keep=False)``;
        keep subsets where ``n_duplicate_rows >= min_dup_rows``.
      * Sort by descending ``dup_rate`` and emit ``max_subsets``.
    """
    if df.empty or len(df) < min_dup_rows:
        return []

    # Candidate columns: string-like or integer dtype, low NaN ratio.
    # Skip floats (precision issues) and datetimes.
    candidates: List[str] = []
    for c in df.columns:
        s = df[c]
        try:
            non_null = int(s.notna().sum())
        except Exception:  # noqa: BLE001
            non_null = 0
        if non_null < min_dup_rows:
            continue
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s) \
                or pd.api.types.is_bool_dtype(s):
            candidates.append(str(c))

    # Prefer the digest's identity-like columns; they're the most
    # plausible identifying keys.
    if digest is not None and digest.candidate_identity_columns:
        ordered = [c for c in digest.candidate_identity_columns if c in candidates]
        # Append non-identity candidates afterwards so we still
        # discover full-row duplicates on tables with no obvious key.
        ordered += [c for c in candidates if c not in ordered]
        candidates = ordered

    # Cap the search; high-cardinality keys dominate the signal.
    candidates = candidates[:8]

    results: List[DuplicateSubset] = []

    # Single-column subsets first.
    for col in candidates:
        try:
            mask = df.duplicated(subset=[col], keep=False) & df[col].notna()
        except Exception:  # noqa: BLE001
            continue
        n_dup = int(mask.sum())
        if n_dup < min_dup_rows:
            continue
        n_groups = int(df.loc[mask, col].nunique(dropna=True))
        sample_groups = _sample_groups_by_subset(df, [col], mask, max_sample_groups)
        results.append(
            DuplicateSubset(
                columns=(col,),
                n_duplicate_rows=n_dup,
                n_groups=n_groups,
                dup_rate=n_dup / max(1, len(df)),
                sample_groups=sample_groups,
            )
        )

    # Optional 2-column subsets among top candidates (cheap quadratic
    # scan with a hard cap).
    if max_subset_size >= 2 and len(candidates) >= 2:
        for i in range(min(len(candidates), 4)):
            for j in range(i + 1, min(len(candidates), 4)):
                cols = [candidates[i], candidates[j]]
                try:
                    mask = df.duplicated(subset=cols, keep=False) & (
                        df[cols].notna().all(axis=1)
                    )
                except Exception:  # noqa: BLE001
                    continue
                n_dup = int(mask.sum())
                if n_dup < min_dup_rows:
                    continue
                # Skip if the same duplicates are already captured by
                # a single-column subset (i.e. the 2-col subset is
                # less informative).
                if any(set(r.columns) <= set(cols) and r.n_duplicate_rows >= n_dup
                       for r in results):
                    continue
                n_groups = int(
                    df.loc[mask, cols].drop_duplicates().shape[0]
                )
                sample_groups = _sample_groups_by_subset(
                    df, cols, mask, max_sample_groups
                )
                results.append(
                    DuplicateSubset(
                        columns=tuple(cols),
                        n_duplicate_rows=n_dup,
                        n_groups=n_groups,
                        dup_rate=n_dup / max(1, len(df)),
                        sample_groups=sample_groups,
                    )
                )

    results.sort(key=lambda r: (-r.dup_rate, len(r.columns), r.columns))
    return results[:max_subsets]


_PUNCT_RE = re.compile(r"[^0-9A-Za-z]+")


def normalize_for_dedup(value: object) -> str:
    """Canonicalise a cell for duplicate-key matching.

    Light-touch normalisation that catches the common
    near-duplicate cases without changing semantics:

      * ``None`` / ``NaN`` -> empty string.
      * Strings: NFKC implicit via ``str()``, lower-cased, internal
        punctuation removed, whitespace collapsed.
      * Numerics: rendered with no thousands separator and trailing
        zeros stripped (e.g. ``1,000.00`` and ``1000`` both -> ``1000``).

    The result is deterministic and safe to ``groupby`` on.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if not pd.notna(value):
                return ""
            if value == int(value):
                return str(int(value))
            return ("%g" % value).strip()
        return str(int(value))
    s = str(value).strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def normalize_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame restricted to ``cols`` with each cell
    normalised by :func:`normalize_for_dedup`. Index is preserved so
    downstream group lookups still resolve to original row numbers."""
    sub = df[list(cols)].copy()
    for c in cols:
        sub[c] = sub[c].map(normalize_for_dedup)
    return sub


def find_duplicate_subsets_normalized(
    df: pd.DataFrame,
    *,
    digest: Optional[TableDigest] = None,
    max_subsets: int = 4,
    max_subset_size: int = 2,
    min_dup_rows: int = 2,
    max_sample_groups: int = 2,
) -> List[DuplicateSubset]:
    """Variant of :func:`find_duplicate_subsets` that compares cells
    after :func:`normalize_for_dedup`.

    Catches near-duplicates on identifier columns where the only
    difference is case, punctuation, or numeric formatting (e.g.\
    ``"Acme Corp."`` vs ``"ACME CORPORATION"``).

    Internally builds a normalised view of the candidate columns and
    runs the existing duplicate-subset logic on that view; row indices
    in the returned ``DuplicateSubset.sample_groups`` are the original
    row indices in ``df``.
    """
    if df.empty or len(df) < min_dup_rows:
        return []
    candidates: List[str] = []
    for c in df.columns:
        s = df[c]
        try:
            non_null = int(s.notna().sum())
        except Exception:  # noqa: BLE001
            non_null = 0
        if non_null < min_dup_rows:
            continue
        # After normalisation everything reduces to a string, so the
        # dtype filter from ``find_duplicate_subsets`` is unnecessary.
        # We deliberately do NOT filter by raw cardinality here: the
        # whole point of the normalised scan is to catch collisions
        # on columns whose raw values are all distinct (e.g.\ "Acme
        # Corp." vs "ACME CORP").
        candidates.append(str(c))
    if digest is not None and digest.candidate_identity_columns:
        ordered = [c for c in digest.candidate_identity_columns if c in candidates]
        ordered += [c for c in candidates if c not in ordered]
        candidates = ordered
    candidates = candidates[:8]
    if not candidates:
        return []
    norm = normalize_columns(df, candidates)
    # Re-use the standard finder against the normalised frame; row
    # indices flow through unchanged because ``normalize_columns``
    # preserves ``df.index``.
    return find_duplicate_subsets(
        norm,
        digest=digest,
        max_subsets=max_subsets,
        max_subset_size=max_subset_size,
        min_dup_rows=min_dup_rows,
        max_sample_groups=max_sample_groups,
    )


def enumerate_collision_groups(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    min_group_size: int = 2,
    normalize: bool = False,
    canonical_maps: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> List[Tuple[int, ...]]:
    """Return every collision group on ``cols``.

    A group is a tuple of original row indices that share the same
    (optionally normalised) value tuple on ``cols`` and has size
    >= ``min_group_size``. Rows where any key column is null are
    dropped before grouping.

    Unlike :func:`find_duplicate_subsets`, this enumerates ALL groups
    (no sample cap) - it is the primary input to the row-duplication
    adjudicator, which sends one LLM call per group.

    ``canonical_maps`` (optional) is a per-column ``{value: canonical}``
    map, applied AFTER lexical normalisation, that lets callers fold
    semantically-equivalent surface forms (e.g. ``"US"`` /
    ``"United States"``) into a single group key. Values not present
    in the map fall through unchanged.
    """
    cols = list(cols)
    if not cols or df.empty:
        return []
    keep = df[cols].notna().all(axis=1)
    sub = df.loc[keep, cols]
    if sub.empty:
        return []
    if normalize or canonical_maps:
        sub = sub.copy()
        for c in cols:
            if normalize:
                sub[c] = sub[c].map(normalize_for_dedup)
            cmap = (canonical_maps or {}).get(c)
            if cmap:
                # Apply canonical map AFTER lexical normalisation so the
                # caller can build the map against either raw or
                # normalised value space.
                sub[c] = sub[c].map(lambda v, m=cmap: m.get(v, v))
        # After normalisation, treat empty strings as null so we don't
        # group spurious blanks together.
        keep_nb = (sub != "").all(axis=1)
        sub = sub.loc[keep_nb]
    groups: List[Tuple[int, ...]] = []
    for _, g in sub.groupby(cols, dropna=False, sort=False):
        if len(g) >= min_group_size:
            groups.append(tuple(int(i) for i in g.index))
    return groups


def _sample_groups_by_subset(
    df: pd.DataFrame,
    cols: Sequence[str],
    mask: "pd.Series[bool]",
    max_groups: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Return up to ``max_groups`` row-index tuples, one per distinct
    duplicate-key value, ordered by appearance."""
    if not mask.any():
        return ()
    sub = df.loc[mask].copy()
    seen: List[Tuple[int, ...]] = []
    for _, group in sub.groupby(list(cols), dropna=False, sort=False):
        rows = tuple(int(i) for i in group.index)
        if len(rows) >= 2:
            seen.append(rows)
        if len(seen) >= max_groups:
            break
    return tuple(seen)


def find_identity_conflicts(
    df: pd.DataFrame,
    *,
    digest: Optional[TableDigest] = None,
    max_conflicts: int = 4,
    min_conflict_groups: int = 1,
    max_sample_groups: int = 2,
) -> List[IdentityConflict]:
    """Find identity-key candidates whose groups disagree on
    descriptive columns.

    For each candidate key ``K`` (single column today), partition
    the frame by ``K`` and look for groups of size >= 2 where some
    other column has multiple distinct non-null values. The
    disagreeing columns become the rule's RHS.
    """
    if df.empty or len(df) < 2:
        return []

    # Candidate keys: identity-like columns from the digest, or fall
    # back to columns with high but not perfect uniqueness.
    if digest is not None and digest.candidate_identity_columns:
        keys = list(digest.candidate_identity_columns)
    else:
        keys = []
        for c in df.columns:
            try:
                n_non_null = int(df[c].notna().sum())
                n_distinct = int(df[c].nunique(dropna=True))
            except Exception:  # noqa: BLE001
                continue
            if n_non_null < 2 or n_distinct < 2:
                continue
            ratio = n_distinct / max(1, n_non_null)
            # Identity-ish: at least 30% of values are distinct, but
            # not perfectly unique (a perfect key has no groups).
            if 0.30 <= ratio < 1.0:
                keys.append(str(c))

    keys = keys[:6]
    results: List[IdentityConflict] = []

    for key in keys:
        if key not in df.columns:
            continue
        # Find groups of size >= 2 keyed on `key`.
        try:
            sizes = df.groupby(key, dropna=True).size()
        except Exception:  # noqa: BLE001
            continue
        multi_groups = sizes[sizes >= 2]
        if multi_groups.empty:
            continue

        # For each other column, count groups whose non-null values
        # disagree.
        descriptive_cols: List[str] = []
        for c in df.columns:
            if c == key:
                continue
            try:
                nunique_per_group = df.groupby(key, dropna=True)[c].apply(
                    lambda s: int(s.dropna().nunique())
                )
            except Exception:  # noqa: BLE001
                continue
            n_conflicting = int((nunique_per_group >= 2).sum())
            if n_conflicting >= min_conflict_groups:
                descriptive_cols.append((c, n_conflicting))  # type: ignore[arg-type]

        if not descriptive_cols:
            continue

        # Order descriptive columns by number of conflicting groups
        # (most informative first).
        descriptive_cols.sort(key=lambda x: -x[1])  # type: ignore[arg-type,index]
        cols = [c for c, _ in descriptive_cols]  # type: ignore[misc]
        max_n_conflicts = max(n for _, n in descriptive_cols)  # type: ignore[misc]

        # Pick up to ``max_sample_groups`` group keys with at least
        # one disagreement and emit their row indices.
        sample_groups = _sample_conflict_groups(
            df, key, cols, max_sample_groups
        )

        results.append(
            IdentityConflict(
                key_columns=(key,),
                descriptive_columns=tuple(cols),
                n_conflict_groups=int(max_n_conflicts),
                sample_groups=sample_groups,
            )
        )

    # Order by number of conflicting groups, then key name for
    # stability.
    results.sort(key=lambda r: (-r.n_conflict_groups, r.key_columns))
    return results[:max_conflicts]


def _sample_conflict_groups(
    df: pd.DataFrame,
    key: str,
    descriptive_cols: Sequence[str],
    max_groups: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Pick up to ``max_groups`` (key-value) groups where at least
    one descriptive column has >=2 distinct non-null values."""
    out: List[Tuple[int, ...]] = []
    for kv, group in df.groupby(key, dropna=True, sort=False):
        if len(group) < 2:
            continue
        has_conflict = False
        for c in descriptive_cols:
            if int(group[c].dropna().nunique()) >= 2:
                has_conflict = True
                break
        if has_conflict:
            out.append(tuple(int(i) for i in group.index))
        if len(out) >= max_groups:
            break
    return tuple(out)


# ---------------------------------------------------------------------------
# Top-level orchestrator + Markdown renderer
# ---------------------------------------------------------------------------


def compute_priors(
    *,
    df: pd.DataFrame,
    digest: TableDigest,
    error_type: str,
    source: str,
) -> MultiRowPriors:
    """Run the per-error-type detectors that are useful for this
    error type.

    Each detector is independent and cheap; we only skip the ones
    that would never apply to the named error type. This is purely
    a cost optimisation - the LLM still receives only the priors
    that match the type it is asked about.
    """
    total_blocks: List[TotalRowBlock] = []
    dup_subsets: List[DuplicateSubset] = []
    id_conflicts: List[IdentityConflict] = []

    if error_type == "row_duplication":
        dup_subsets = find_duplicate_subsets(df, digest=digest)
    else:
        # Future-proofing: callers may pass other types defensively.
        # Return an empty priors object so the prompt branch shows
        # "(no structural candidates)" rather than crashing.
        pass

    return MultiRowPriors(
        error_type=error_type,
        source=source,
        total_row_blocks=tuple(total_blocks),
        duplicate_subsets=tuple(dup_subsets),
        identity_conflicts=tuple(id_conflicts),
    )


# ---------------------------------------------------------------------------
# Markdown renderer for the prompt
# ---------------------------------------------------------------------------


def _truncate(value: object, max_chars: int = 60) -> str:
    s = "" if pd.isna(value) else str(value)
    if len(s) > max_chars:
        return s[: max_chars - 1] + "\u2026"
    return s


def _render_row_block(
    df: pd.DataFrame,
    rows: Sequence[int],
    *,
    cols: Optional[Sequence[str]] = None,
    max_cols: int = 8,
) -> str:
    """Render a small Markdown table for the listed row indices."""
    if not rows:
        return "(empty)"
    cols_to_show = list(cols) if cols else list(df.columns)
    cols_to_show = cols_to_show[:max_cols]
    sub = df.iloc[list(rows)][cols_to_show].copy()
    sub = sub.astype(object).map(_truncate)
    sub.insert(0, "__row__", list(rows))
    try:
        return sub.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        # to_markdown depends on tabulate; fall back to a simple
        # pipe-separated render if tabulate is missing.
        header = " | ".join(["__row__"] + cols_to_show)
        body_rows: List[str] = []
        for i, r in enumerate(rows):
            row_vals = [str(r)] + [
                _truncate(df.iloc[r][c]) for c in cols_to_show
            ]
            body_rows.append(" | ".join(row_vals))
        return "\n".join([header, "-" * len(header), *body_rows])


def render_priors_block(
    df: pd.DataFrame,
    priors: MultiRowPriors,
) -> str:
    """Render the priors as a discovery-prompt Markdown section.

    When ``priors.is_empty()`` returns ``True``, emit a short note
    so the LLM understands that the structural detector found no
    candidates rather than that the section was forgotten.
    """
    if priors.is_empty():
        return (
            "STRUCTURAL CANDIDATES (pre-computed; you may use, "
            "refine, or reject)\n"
            "  (none found by the heuristic detector)\n"
        )

    parts: List[str] = []
    parts.append(
        "STRUCTURAL CANDIDATES (pre-computed; you may use, refine, "
        "or reject)"
    )

    if priors.total_row_blocks:
        parts.append("\nTotal-row blocks:")
        for b in priors.total_row_blocks:
            label = b.label or "(no label)"
            details = (
                ", ".join(str(i) for i in b.detail_rows)
                if b.detail_rows
                else "(no preceding detail rows)"
            )
            parts.append(
                f"  - row {b.total_row}: label={label!r}; "
                f"plausible detail rows = [{details}]"
            )

    if priors.duplicate_subsets:
        parts.append("\nDuplicate-prone column subsets:")
        for s in priors.duplicate_subsets:
            cols = ", ".join(s.columns)
            parts.append(
                f"  - columns=({cols}): {s.n_duplicate_rows} rows in "
                f"{s.n_groups} duplicate groups (dup_rate="
                f"{s.dup_rate:.2%})"
            )

    if priors.identity_conflicts:
        parts.append(
            "\nIdentity / descriptive-column conflicts (rows agree "
            "on the key but disagree on the listed columns):"
        )
        for c in priors.identity_conflicts:
            keys = ", ".join(c.key_columns)
            descs = ", ".join(c.descriptive_columns[:6])
            extra = (
                f" (+{len(c.descriptive_columns) - 6} more)"
                if len(c.descriptive_columns) > 6 else ""
            )
            parts.append(
                f"  - key=({keys}); conflicting in {{{descs}}}{extra}"
                f"; {c.n_conflict_groups} group(s) disagree"
            )

    return "\n".join(parts) + "\n"


def render_group_blocks(
    df: pd.DataFrame,
    priors: MultiRowPriors,
    *,
    max_blocks: int = 6,
    max_cols: int = 10,
) -> str:
    """Render verbatim row blocks for the candidates in ``priors``.

    Each block shows actual row indices + cell values so the LLM
    can see what "the same group" looks like for this table.
    """
    if priors.is_empty() or df.empty:
        return "GROUP SAMPLES\n  (none)\n"

    parts: List[str] = ["GROUP SAMPLES (verbatim row blocks)"]
    blocks_emitted = 0

    for b in priors.total_row_blocks:
        if blocks_emitted >= max_blocks:
            break
        rows = list(b.detail_rows) + [b.total_row]
        parts.append(
            f"\n--- aggregate block: total row {b.total_row} "
            f"({b.label!r}) preceded by {len(b.detail_rows)} detail row(s) ---"
        )
        parts.append(_render_row_block(df, rows, max_cols=max_cols))
        blocks_emitted += 1

    for s in priors.duplicate_subsets:
        for group_rows in s.sample_groups:
            if blocks_emitted >= max_blocks:
                break
            cols_str = ", ".join(s.columns)
            parts.append(
                f"\n--- duplicate group on ({cols_str}): "
                f"rows {list(group_rows)} ---"
            )
            parts.append(_render_row_block(df, group_rows, max_cols=max_cols))
            blocks_emitted += 1

    for c in priors.identity_conflicts:
        for group_rows in c.sample_groups:
            if blocks_emitted >= max_blocks:
                break
            keys_str = ", ".join(c.key_columns)
            parts.append(
                f"\n--- identity-conflict group on key ({keys_str}): "
                f"rows {list(group_rows)} ---"
            )
            # Show key cols + descriptive cols only.
            cols = list(c.key_columns) + list(c.descriptive_columns)
            parts.append(
                _render_row_block(df, group_rows, cols=cols, max_cols=max_cols)
            )
            blocks_emitted += 1

    if blocks_emitted == 0:
        parts.append("  (no blocks renderable - sample groups empty)")

    return "\n".join(parts) + "\n"
