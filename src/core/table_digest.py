"""Compact whole-table summary used as schema context for the LLM.

The per-type detector chunks a CSV into row windows for each LLM call,
which is fine for cell-local error types but starves the
multi-column / multi-row types of the global structure they need to
reason about (column dtypes, candidate identity columns, total-row
positions, approximate functional dependencies).

A :class:`TableDigest` summarises a CSV in O(columns) tokens. It is
attached once per ``source`` and rendered into the user prompt as a
``TABLE OVERVIEW`` block via
:meth:`ErrorTaxonomy.render_user_prompt`. The digest is intentionally
small (top-K most frequent values per column, a handful of candidate
FDs, etc.) so prepending it to every chunk prompt does not blow the
context window even for tall CSVs.

Design notes
------------

* Pure pandas / stdlib. No LLM calls.
* Conservative thresholds: a "candidate FD" only emits when the
  determinant has a meaningful number of distinct values AND the
  modal mapping covers a large fraction of rows. False positives
  here would mislead the LLM, so we err on the side of an empty
  list.
* The rendered block is human-readable (so prompt logs stay
  inspectable) and keyed by column name so the LLM can cross-
  reference it against the chunk it sees.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Label-column total-row regex. Mirrors the wording in
# `prompts/error_types.yaml` for column_aggregation_inconsistency so
# the digest highlights the same rows the LLM is told to look at.
# No capture groups so ``Series.str.contains`` does not warn.
_TOTAL_LABEL_RE = re.compile(
    r"\b(?:total|totals|subtotal|sub[- ]?total|grand[- ]?total|"
    r"sum|sums|aggregate|fy|year[- ]?to[- ]?date|ytd)\b",
    flags=re.IGNORECASE,
)

# Anchored numeric pattern (allows leading +/-, optional thousands
# separators, optional decimal, optional %, optional currency symbol).
_NUMERIC_RE = re.compile(
    r"^\s*[\$\u20ac\u00a3\u00a5]?\s*[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?\s*%?\s*$"
)


# ---------------------------------------------------------------------------
# Per-column summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnDigest:
    """Per-column statistics packed into the table overview."""

    name: str
    n_non_empty: int
    n_distinct: int
    dtype_kind: str  # "numeric" | "text" | "empty"
    top_values: List[Tuple[str, int]] = field(default_factory=list)
    numeric_summary: Optional[Dict[str, float]] = None
    """For numeric columns: ``min, p25, median, mean, p75, max, n_zeros,
    n_negatives``. ``None`` for text/empty columns."""
    sample_values: List[str] = field(default_factory=list)
    is_identity_candidate: bool = False
    mean_length: float = 0.0
    length_summary: Optional[Dict[str, int]] = None
    """For text columns: ``min, median, max`` cell-character lengths.
    Lets the LLM detect free-text vs short-code columns and reason
    about whether outlier rows have unusually long/short cells."""
    rare_values: List[str] = field(default_factory=list)
    """Up to N values from the long tail (frequency 1) - useful when
    `top_values` is uninformative (every value appears once) or the
    column has a fat head and a long tail of one-offs."""


# ---------------------------------------------------------------------------
# Table-level summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableDigest:
    """Whole-table summary attached as schema context to LLM prompts."""

    source: str
    n_rows: int
    n_columns: int
    headers: List[str]
    columns: Dict[str, ColumnDigest]
    total_row_indices: List[int] = field(default_factory=list)
    candidate_identity_columns: List[str] = field(default_factory=list)
    candidate_fds: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def to_prompt_block(
        self,
        *,
        max_top_values: int = 5,
        max_sample_values: int = 3,
        max_fds: int = 8,
        max_value_chars: int = 40,
    ) -> str:
        """Render the digest as a compact TABLE OVERVIEW block.

        The text is keyed by column name and uses short, structured
        lines so the LLM can quickly look up "what does this column
        look like across the WHOLE table" while reasoning about the
        slice in front of it.
        """
        if self.n_rows == 0:
            return (
                f"TABLE OVERVIEW\n"
                f"  source : {self.source}\n"
                f"  shape  : 0 rows x {self.n_columns} columns (empty table)\n"
            )

        def _trunc(v: str) -> str:
            if len(v) > max_value_chars:
                return v[: max_value_chars - 1] + "\u2026"
            return v

        lines: List[str] = []
        lines.append("TABLE OVERVIEW")
        lines.append(f"  source : {self.source}")
        lines.append(
            f"  shape  : {self.n_rows} rows x {self.n_columns} columns"
        )
        if self.total_row_indices:
            preview = ", ".join(str(i) for i in self.total_row_indices[:10])
            extra = (
                f" (and {len(self.total_row_indices) - 10} more)"
                if len(self.total_row_indices) > 10
                else ""
            )
            lines.append(
                f"  total/summary rows detected at __row__ indices: "
                f"[{preview}]{extra}"
            )
        if self.candidate_identity_columns:
            lines.append(
                "  identity-like columns (high uniqueness): "
                + ", ".join(self.candidate_identity_columns)
            )
        if self.candidate_fds:
            lines.append("  approximate functional dependencies (A -> B):")
            for fd in self.candidate_fds[:max_fds]:
                lines.append(
                    f"    - {fd['lhs']!s} -> {fd['rhs']!s}  "
                    f"(support {fd['support']:.0%}, "
                    f"{fd['n_groups']} groups, "
                    f"{fd['n_rows']} rows)"
                )
                # Surface up to 3 counter-examples for approximate
                # FDs (support < 100%). The LLM should decide whether
                # the FD is real and the violators are defects, OR
                # the FD is spurious and should be rejected.
                examples = fd.get("violation_examples") or []
                if examples and fd["support"] < 1.0:
                    for ex in examples[:3]:
                        lines.append(
                            f"        violation at row {ex['row']}: "
                            f"{fd['lhs']}={_trunc(str(ex['lhs_value']))!r} "
                            f"-> {fd['rhs']}="
                            f"{_trunc(str(ex['rhs_value']))!r} "
                            f"(modal mapping says "
                            f"{_trunc(str(ex['expected_rhs']))!r})"
                        )

        lines.append("  per-column summary:")
        for name in self.headers:
            col = self.columns[name]
            head = (
                f"    - {name!r}: "
                f"{col.dtype_kind}, "
                f"{col.n_non_empty}/{self.n_rows} non-empty, "
                f"{col.n_distinct} distinct"
            )
            if col.is_identity_candidate:
                head += ", identity-like"
            lines.append(head)
            if col.dtype_kind == "numeric" and col.numeric_summary:
                ns = col.numeric_summary
                # Five-number summary for shape, plus zero / negative
                # counts so the LLM can reason about implicit
                # constraints (a count column with n_negatives=0 IS a
                # bound; a percentage column staying inside [0,100]
                # might be definitional).
                p25 = ns.get("p25")
                p75 = ns.get("p75")
                quart_part = (
                    f" p25={p25:g} p75={p75:g}"
                    if p25 is not None and p75 is not None
                    else ""
                )
                lines.append(
                    f"        numeric: min={ns['min']:g}"
                    f"{quart_part} median={ns['median']:g} "
                    f"mean={ns['mean']:g} max={ns['max']:g}"
                )
                n_zeros = ns.get("n_zeros")
                n_negatives = ns.get("n_negatives")
                tags: List[str] = []
                if n_zeros:
                    tags.append(f"zeros={n_zeros}")
                if n_negatives:
                    tags.append(f"negatives={n_negatives}")
                if tags:
                    lines.append(f"        numeric flags: {', '.join(tags)}")
            if col.length_summary:
                ls = col.length_summary
                lines.append(
                    f"        cell length (chars): "
                    f"min={ls['min']} median={ls['median']} max={ls['max']}"
                )
            if col.top_values:
                top = ", ".join(
                    f"{_trunc(v)!r} x{c}"
                    for v, c in col.top_values[:max_top_values]
                )
                lines.append(f"        top values: {top}")
            elif col.sample_values:
                samples = ", ".join(
                    repr(_trunc(v)) for v in col.sample_values[:max_sample_values]
                )
                lines.append(f"        samples: {samples}")
            # Rare values - up to 3 from the long tail. Shown only
            # when the head + tail together carry signal: distinct
            # count exceeds top_values length.
            if col.rare_values and col.n_distinct > len(col.top_values):
                rare = ", ".join(repr(_trunc(v)) for v in col.rare_values[:3])
                lines.append(f"        rare values (long tail): {rare}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view suitable for ``json.dumps``.

        Used by ``experiments.run_relation_discovery`` to persist a
        compact ``sources.json`` artifact next to ``relations.jsonl``,
        which the report template reads to show per-CSV context
        (shape, columns, candidate FDs, total rows) alongside each
        source's relations. Top values are capped to keep the file
        small for runs with hundreds of sources.
        """
        def _col(col: ColumnDigest) -> Dict[str, Any]:
            return {
                "name": col.name,
                "n_non_empty": col.n_non_empty,
                "n_distinct": col.n_distinct,
                "dtype_kind": col.dtype_kind,
                "is_identity_candidate": col.is_identity_candidate,
                "mean_length": col.mean_length,
                "top_values": [
                    {"value": v, "count": c}
                    for v, c in col.top_values[:8]
                ],
                "sample_values": list(col.sample_values[:8]),
                "numeric_summary": dict(col.numeric_summary)
                if col.numeric_summary else None,
                "length_summary": dict(col.length_summary)
                if col.length_summary else None,
                "rare_values": list(col.rare_values[:3]),
            }

        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "headers": list(self.headers),
            "total_row_indices": list(self.total_row_indices),
            "candidate_identity_columns": list(self.candidate_identity_columns),
            "candidate_fds": [dict(fd) for fd in self.candidate_fds],
            "columns": {name: _col(col) for name, col in self.columns.items()},
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _classify_dtype(values: pd.Series) -> str:
    """Return ``"numeric"`` if >=80% of non-empty values look numeric,
    ``"text"`` otherwise. ``"empty"`` if no non-empty values."""
    non_empty = values[values != ""]
    if non_empty.empty:
        return "empty"
    sample = non_empty if len(non_empty) <= 500 else non_empty.sample(500, random_state=0)
    matches = sample.map(lambda v: bool(_NUMERIC_RE.match(v))).sum()
    return "numeric" if matches / max(1, len(sample)) >= 0.8 else "text"


def _to_float(v: str) -> Optional[float]:
    """Best-effort coercion for numeric_summary stats.

    Mirrors the surface forms accepted by :data:`_NUMERIC_RE` so the
    dtype classifier and the per-cell coercion stay in sync (otherwise
    a column could classify as numeric and still leave its
    ``numeric_summary`` empty because every cell rejected the parse).
    Accepts: optional leading ``+``/``-``, accounting-style
    parenthesised negatives ``(50)``, currency prefix
    (``$``/``\u20ac``/``\u00a3``/``\u00a5``) -- with the sign allowed BEFORE or
    AFTER the symbol -- thousand separators (``1,000``), decimals,
    and a trailing ``%``.
    """
    if not isinstance(v, str) or not v.strip():
        return None
    s = v.strip()

    # Accounting-style negative: "(50)" or "(1,234.5)" \u2192 -50, -1234.5.
    paren_negative = False
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        paren_negative = True
        s = s[1:-1].strip()

    # Trailing "%" \u2192 divide by 100 at the end.
    has_pct = s.endswith("%")
    if has_pct:
        s = s[:-1].strip()

    # Sign may appear before OR after the currency symbol
    # ("-$100" and "$-100" both occur in the wild).
    sign = 1
    if s.startswith("+"):
        s = s[1:].strip()
    elif s.startswith("-"):
        sign = -1
        s = s[1:].strip()
    for sym in ("$", "\u20ac", "\u00a3", "\u00a5"):
        if s.startswith(sym):
            s = s[1:].strip()
            break
    if s.startswith("+"):
        s = s[1:].strip()
    elif s.startswith("-"):
        sign = -sign
        s = s[1:].strip()

    s = s.replace(",", "")
    if not s:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    x = sign * x
    if paren_negative:
        x = -abs(x)
    return x / 100.0 if has_pct else x


def _build_column_digest(
    name: str,
    series: pd.Series,
    *,
    n_rows: int,
    top_k: int,
    sample_k: int,
    identity_threshold: float,
) -> ColumnDigest:
    s = series.where(series.notna(), "").astype(str)
    s = s.str.strip()
    non_empty = s[s != ""]
    n_non_empty = int(len(non_empty))
    n_distinct = int(non_empty.nunique()) if n_non_empty else 0
    dtype_kind = _classify_dtype(s)
    # value_counts once; reused for top_values and rare_values below.
    vc = non_empty.value_counts() if n_non_empty else None

    top_values: List[Tuple[str, int]] = []
    sample_values: List[str] = []
    if n_non_empty:
        top_values = [(str(idx), int(cnt)) for idx, cnt in vc.head(top_k).items()]
        # Distinct samples (in encounter order) are useful when the
        # column is high-cardinality and ``top_values`` carries no
        # signal (every value appears once).
        seen: set = set()
        for v in non_empty:
            if v not in seen:
                seen.add(v)
                sample_values.append(str(v))
                if len(sample_values) >= sample_k:
                    break

    numeric_summary: Optional[Dict[str, float]] = None
    if dtype_kind == "numeric" and n_non_empty:
        floats = [x for x in (_to_float(v) for v in non_empty) if x is not None]
        if floats:
            ser = pd.Series(floats, dtype=float)
            numeric_summary = {
                "min": float(ser.min()),
                "p25": float(ser.quantile(0.25)),
                "median": float(ser.median()),
                "mean": float(ser.mean()),
                "p75": float(ser.quantile(0.75)),
                "max": float(ser.max()),
                "n_zeros": int((ser == 0).sum()),
                "n_negatives": int((ser < 0).sum()),
            }

    # Text-length summary - lets the LLM tell free-text columns from
    # short-code columns at a glance, and gives outlier rows a
    # reference for "this cell is unusually long/short".
    length_summary: Optional[Dict[str, int]] = None
    if dtype_kind == "text" and n_non_empty:
        lens = non_empty.str.len()
        length_summary = {
            "min": int(lens.min()),
            "median": int(lens.median()),
            "max": int(lens.max()),
        }

    # Rare values: up to a few values from the long tail (frequency 1).
    # These are *additive* to top_values - they don't replace it.
    # Useful for fat-head/long-tail distributions where the top-K
    # masks the diversity of the rest.
    rare_values: List[str] = []
    if n_non_empty:
        singletons = [str(v) for v, c in vc.items() if c == 1]
        rare_values = singletons[-3:]  # arbitrary 3 from the tail

    is_identity = (
        n_non_empty >= max(5, int(0.5 * n_rows))
        and n_distinct >= int(identity_threshold * n_non_empty)
    )

    mean_length = (
        float(non_empty.str.len().mean()) if n_non_empty else 0.0
    )

    return ColumnDigest(
        name=name,
        n_non_empty=n_non_empty,
        n_distinct=n_distinct,
        dtype_kind=dtype_kind,
        top_values=top_values,
        numeric_summary=numeric_summary,
        sample_values=sample_values,
        is_identity_candidate=is_identity,
        mean_length=mean_length,
        length_summary=length_summary,
        rare_values=rare_values,
    )


def _detect_total_rows(df: pd.DataFrame) -> List[int]:
    """Row indices whose label-like leading cell matches a total/sum
    cue. We probe up to the first three columns to catch tables where
    the label sits one or two columns in (very common in financial
    spreadsheets exported to CSV)."""
    if df.empty:
        return []
    probe_cols = list(df.columns[: min(3, df.shape[1])])
    matches: set[int] = set()
    for col in probe_cols:
        ser = df[col].where(df[col].notna(), "").astype(str).str.strip()
        hit = ser.str.contains(_TOTAL_LABEL_RE, regex=True, na=False)
        for idx in df.index[hit]:
            matches.add(int(idx))
    return sorted(matches)


def _candidate_fds(
    df: pd.DataFrame,
    *,
    max_codomain: int,
    min_support: float,
    min_group_size: int,
    rule_modal_share: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Approximate functional dependency candidates.

    Mirrors the offline computation in
    ``src.heuristics.fd_violation`` but only KEEPS the strong pairs;
    we do not flag rows here, just hint to the LLM that the pair is
    approximately functional.
    """
    cols = list(df.columns)
    n_rows = len(df)
    if n_rows < min_group_size:
        return []

    stripped: Dict[str, pd.Series] = {}
    distinct_counts: Dict[str, int] = {}
    for col in cols:
        ser = df[col].where(df[col].notna(), "").astype(str).str.strip()
        ser.index = pd.RangeIndex(n_rows)
        stripped[col] = ser
        distinct_counts[col] = int(ser[ser != ""].nunique())

    candidates: List[Dict[str, Any]] = []
    for b in cols:
        if not (2 <= distinct_counts[b] <= max_codomain):
            continue
        b_ser = stripped[b]
        for a in cols:
            if a == b:
                continue
            a_distinct = distinct_counts[a]
            if not (2 <= a_distinct <= n_rows - 1):
                continue
            a_ser = stripped[a]
            mask = (a_ser != "") & (b_ser != "")
            if not mask.any():
                continue
            sub = pd.DataFrame({"_a": a_ser[mask], "_b": b_ser[mask]})
            group_sizes = sub.groupby("_a", sort=False).size()
            qualifying = group_sizes[group_sizes >= min_group_size]
            if qualifying.empty:
                continue
            sub_q = sub[sub["_a"].isin(qualifying.index)]
            pair_counts = (
                sub_q.groupby(["_a", "_b"], sort=False)
                .size()
                .rename("c")
                .reset_index()
                .sort_values(["_a", "c", "_b"], ascending=[True, False, True])
            )
            modals = pair_counts.drop_duplicates(subset=["_a"], keep="first")
            total = int(qualifying.sum())
            modal_total = int(modals["c"].sum())
            if total == 0 or modal_total / total < min_support:
                continue
            # Optional extra bar: at least one A-group must be
            # genuinely consistent (modal share >= rule_modal_share).
            modals_with_size = modals.merge(
                qualifying.rename("group_size").reset_index(),
                left_on="_a",
                right_on="_a",
            )
            consistent = modals_with_size[
                modals_with_size["c"] / modals_with_size["group_size"]
                >= rule_modal_share
            ]
            if consistent.empty:
                continue
            # Counter-examples: rows that violate the modal mapping.
            # When the FD is approximate (support < 1) these are
            # exactly the rows the LLM should consider when deciding
            # whether to emit the FD as a hard rule. We persist up
            # to 3 examples in (row_idx, lhs_value, rhs_value,
            # expected_modal_rhs) form so the prompt can ground the
            # FD candidate in concrete conflicts.
            violations: List[Dict[str, Any]] = []
            modal_map = dict(zip(modals["_a"], modals["_b"]))
            mismatches = sub_q[sub_q["_b"] != sub_q["_a"].map(modal_map)]
            if not mismatches.empty:
                for idx, row in mismatches.head(3).iterrows():
                    violations.append({
                        "row": int(idx),
                        "lhs_value": str(row["_a"]),
                        "rhs_value": str(row["_b"]),
                        "expected_rhs": str(modal_map.get(row["_a"], "")),
                    })
            candidates.append(
                {
                    "lhs": a,
                    "rhs": b,
                    "support": modal_total / total,
                    "n_groups": int(len(qualifying)),
                    "n_rows": int(total),
                    "violation_examples": violations,
                }
            )

    candidates.sort(key=lambda d: (-d["support"], -d["n_rows"]))
    return candidates[:top_k]


def build_table_digest(
    df: pd.DataFrame,
    *,
    source: str | Path,
    top_k_values: int = 5,
    sample_k_values: int = 3,
    identity_threshold: float = 0.95,
    fd_max_codomain: int = 50,
    fd_min_support: float = 0.9,
    fd_min_group_size: int = 3,
    fd_rule_modal_share: float = 0.8,
    fd_top_k: int = 12,
) -> TableDigest:
    """Build a :class:`TableDigest` from an in-memory dataframe.

    The dataframe is the one returned by
    :func:`src.core.data_loader.load_csv` (string-typed cells, header
    de-duplicated). The digest is computed once per CSV and shared
    across every (chunk, error_type) call against that source.
    """
    headers = list(df.columns)
    n_rows = len(df)

    # Reset index so total-row indices align with the chunk's
    # ``__row__`` numbering (which is also derived from this same
    # 0-based positional index).
    if not df.empty and not df.index.equals(pd.RangeIndex(n_rows)):
        df = df.reset_index(drop=True)

    columns: Dict[str, ColumnDigest] = {}
    for name in headers:
        columns[name] = _build_column_digest(
            name=name,
            series=df[name],
            n_rows=n_rows,
            top_k=top_k_values,
            sample_k=sample_k_values,
            identity_threshold=identity_threshold,
        )

    total_rows = _detect_total_rows(df)
    identity_cols = [n for n, c in columns.items() if c.is_identity_candidate]
    fds = _candidate_fds(
        df,
        max_codomain=fd_max_codomain,
        min_support=fd_min_support,
        min_group_size=fd_min_group_size,
        rule_modal_share=fd_rule_modal_share,
        top_k=fd_top_k,
    )

    return TableDigest(
        source=str(source),
        n_rows=n_rows,
        n_columns=len(headers),
        headers=headers,
        columns=columns,
        total_row_indices=total_rows,
        candidate_identity_columns=identity_cols,
        candidate_fds=fds,
    )
