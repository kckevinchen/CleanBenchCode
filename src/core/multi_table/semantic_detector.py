"""Step 3 (Shape-1, "LLM as detector") of the multi-table
inconsistency pipeline.

This module REPLACES the deterministic compiler in
:mod:`src.core.multi_table.violations` and the per-violation
verifier in :mod:`src.core.multi_table.verifier` for groups where
the user selects ``--multi-table-strategy semantic``.

The pipeline becomes:

    1. Structural priors (Python, multi_table_priors.py)
    2. Group + canonical-mapping confirmation (LLM,
       prompts/confirm_union_join.yaml)
    3. Semantic inconsistency detection (LLM,
       prompts/detect_multi_table_inconsistency.yaml)  <-- here

Why: the deterministic compiler can only detect
``attribute_conflict`` / ``duplicate_row`` cases that survive a
``==`` after surface-stripping. Semantic variants - ``"USA"`` /
``"United States"``, ``$1.50`` / ``150 cents``, ``"2024-01-15"`` /
``"15/01/24"`` - do not collapse and so are invisible. The
verifier in step 4 cannot find what step 3 never raised. Letting
the LLM see the row samples directly lets it pair entities by
meaning rather than by surface equality.

Output shape (one record per finding-row, just like
:class:`~multi_table_violations.MultiTableViolation` so the
downstream report renderer is shared). When a finding spans
multiple (sheet, row) locations, we emit ONE record per location
linked by ``violation_id``.

Failure modes are recorded but do not crash the pipeline: if the
LLM call errors or the response cannot be parsed, the group
contributes zero findings and the error surfaces in
``summary.json::multi_table.semantic.n_errored``.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd
import yaml

from ..jsonl_io import JSON_FENCE_RE, JSON_OBJECT_RE
from ..llm_client import LLMClient, LLMResponse
from .resolver import ResolvedGroup, ResolvedWorkbook
from .violations import (
    MultiTableViolation,
    _coerce_value,
    _vid,
    compile_violations,
)
from .verifier import Verification

logger = logging.getLogger(__name__)


# Kinds the prompt is allowed to emit. We accept the legacy
# compiler kinds (so the dashboard's existing colour pills work)
# plus the new semantic kinds.
SEMANTIC_KINDS = (
    "duplicate_row",
    "attribute_conflict",
    "surface_drift",
    "unit_drift",
    "format_drift",
    "null_placeholder",
    "snapshot_aggregation",
    "fk_orphan",
)

VERDICTS = (
    "confirmed_inconsistency",
    "benign_variation",
    "uncertain",
)
CONFIDENCES = ("high", "medium", "low")


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectPromptConfig:
    """Loaded prompt config for ``detect_multi_table_inconsistency.yaml``."""

    model: str
    reasoning_effort: str
    reasoning_summary: str
    system_prompt: str
    user_template: str


def load_detect_prompt(path: Path) -> DetectPromptConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"detect prompt must be a mapping: {path}")
    return DetectPromptConfig(
        model=str(raw.get("model") or "gpt-5.2"),
        reasoning_effort=str(raw.get("reasoning_effort") or "medium"),
        reasoning_summary=str(raw.get("reasoning_summary") or "detailed"),
        system_prompt=str(raw.get("system") or "").strip(),
        user_template=str(raw.get("user_template") or "").strip(),
    )


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PageSlice:
    """One contiguous page of rows for one sheet on one LLM call.

    ``row_offsets`` maps the local 0-based row index INSIDE this
    page back to the global 0-based row index of the underlying
    DataFrame (so we can rewrite row-numbers in the LLM's output
    back to the workbook's positional row indices).
    """

    sheet: str
    row_offsets: Tuple[int, ...]   # local idx -> global idx


def _build_pages(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    n_sample_rows: int,
    seed: int,
) -> List[List[_PageSlice]]:
    """Slice each sheet into pages of at most ``n_sample_rows`` rows.

    Returns a list of pages; each page is a list of ``_PageSlice``
    (one slice per sheet).

    For sheets shorter than ``n_sample_rows`` we emit one page that
    contains the whole sheet. For longer sheets we deterministically
    shuffle (seeded) and chunk; this gives the LLM a representative
    cross-section without scanning every row on huge sheets, while
    still letting paging cover the full sheet if budget allows.

    The page count is the max over sheets so each page contains a
    chunk of every sheet (the LLM needs to compare across sheets,
    so we never "exhaust" one sheet before starting another).
    """
    sheet_chunks: Dict[str, List[Tuple[int, ...]]] = {}
    max_pages = 1
    for sheet in sheets:
        df = frames.get(sheet)
        if df is None or df.empty:
            sheet_chunks[sheet] = [tuple()]
            continue
        n = len(df)
        if n <= n_sample_rows:
            sheet_chunks[sheet] = [tuple(range(n))]
            continue
        rng = random.Random(seed + hash(sheet))
        idxs = list(range(n))
        rng.shuffle(idxs)
        chunks: List[Tuple[int, ...]] = []
        for start in range(0, n, n_sample_rows):
            chunks.append(tuple(idxs[start:start + n_sample_rows]))
        sheet_chunks[sheet] = chunks
        if len(chunks) > max_pages:
            max_pages = len(chunks)

    pages: List[List[_PageSlice]] = []
    for p in range(max_pages):
        page: List[_PageSlice] = []
        for sheet in sheets:
            chunks = sheet_chunks.get(sheet) or [tuple()]
            # Cycle through chunks if a sheet has fewer pages than
            # the longest one. This keeps every page populated.
            chunk = chunks[p % len(chunks)] if chunks else tuple()
            page.append(_PageSlice(sheet=sheet, row_offsets=chunk))
        pages.append(page)
    return pages


# ---------------------------------------------------------------------------
# Key-aligned pagination (recall-first): co-locate rows that share a
# normalized key so the LLM sees both sides of a cross-sheet conflict
# on the SAME page, instead of relying on an independent random shuffle
# happening to land them together. Falls back to ``_build_pages`` when
# the group exposes no usable key.
# ---------------------------------------------------------------------------


def _normalize_key_value(val: Any) -> Optional[str]:
    """Fold a single key cell to a comparable token.

    Returns ``None`` for nulls / empties so rows with a missing key
    are treated as un-alignable (they get a singleton bucket) rather
    than colliding on an empty-string key.
    """
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip().casefold()
    return s or None


def _resolve_key_columns(group: ResolvedGroup) -> Dict[str, Tuple[str, ...]]:
    """Map each sheet to the raw column(s) forming its alignment key.

    * Groups with a ``primary_key`` (canonical names): resolve each
      canonical key to the sheet's raw column via ``column_mapping``.
      A sheet missing any key column is omitted (its rows can't be
      key-aligned). Requires the key on >= 2 sheets to be useful.
    * Confirmed ``join`` groups with no PK: use the join spec - the
      parent sheet keys on ``parent_columns`` and the child on
      ``child_columns`` (they align by value, i.e. the FK).

    Returns ``{}`` when no usable key exists.
    """
    out: Dict[str, Tuple[str, ...]] = {}
    if group.primary_key:
        for sheet in group.sheets:
            raw_cols: List[str] = []
            ok = True
            for canonical in group.primary_key:
                entries = group.column_mapping.get(canonical) or ()
                raw = next(
                    (e.column for e in entries if e.sheet == sheet), None,
                )
                if raw is None:
                    ok = False
                    break
                raw_cols.append(raw)
            if ok and raw_cols:
                out[sheet] = tuple(raw_cols)
        if len(out) >= 2:
            return out
        out = {}
    if group.kind == "join" and group.join is not None:
        j = group.join
        if j.parent_columns and j.child_columns:
            out[j.parent_sheet] = tuple(j.parent_columns)
            out[j.child_sheet] = tuple(j.child_columns)
            if len(out) >= 2:
                return out
    return {}


def _row_key(
    df: pd.DataFrame, ridx: int, raw_cols: Sequence[str],
) -> Optional[Tuple[str, ...]]:
    """Normalized key tuple for one row, or ``None`` if any key cell is
    null / empty / missing."""
    parts: List[str] = []
    for col in raw_cols:
        if col not in df.columns:
            return None
        try:
            val = df.iat[ridx, df.columns.get_loc(col)]
        except (KeyError, IndexError):
            return None
        norm = _normalize_key_value(val)
        if norm is None:
            return None
        parts.append(norm)
    return tuple(parts) if parts else None


# Cap on rows scanned when measuring key cardinality. Uniqueness is a
# GLOBAL property the LLM cannot see from a single page sample, so we
# compute it deterministically over the whole sheet (capped so a
# pathologically large sheet can't blow up the pre-pass).
_KEY_CARD_SCAN_CAP = 20000


def _key_cardinality(
    df: pd.DataFrame, raw_cols: Sequence[str],
) -> Tuple[int, int, int, int]:
    """Measure how unique ``raw_cols`` is over the full frame.

    Returns ``(n_rows, n_keyed, n_distinct, max_multiplicity)``:
    ``n_keyed`` counts rows with a non-null key and ``max_multiplicity``
    is the largest number of rows sharing one key value. A unique key
    has ``n_distinct == n_keyed`` and ``max_multiplicity == 1``.
    """
    n_rows = len(df)
    counts: Dict[Tuple[str, ...], int] = {}
    n_keyed = 0
    for ridx in range(min(n_rows, _KEY_CARD_SCAN_CAP)):
        k = _row_key(df, ridx, raw_cols)
        if k is None:
            continue
        n_keyed += 1
        counts[k] = counts.get(k, 0) + 1
    n_distinct = len(counts)
    max_mult = max(counts.values()) if counts else 0
    return n_rows, n_keyed, n_distinct, max_mult


def _uniqueness_label(n_keyed: int, n_distinct: int, max_mult: int) -> str:
    if n_keyed == 0 or n_distinct == 0:
        return "no usable key values"
    if max_mult <= 1:
        return "UNIQUE (one row per key)"
    return f"NON-UNIQUE (up to {max_mult} rows share one key value)"


def _render_key_uniqueness_block(
    group: ResolvedGroup, frames: Mapping[str, pd.DataFrame],
) -> str:
    """Render a deterministic per-sheet key-cardinality summary.

    Uniqueness can only be judged over the WHOLE sheet, not the page
    sample the LLM sees, so we compute it here and hand the model the
    verdict. Join groups are rendered role-aware: the parent key is
    expected to be unique (a lookup / PK) while the child key is
    expected to recur (the many side of a 1:N foreign key), so a child
    value that matches the parent is a satisfied reference, NOT a
    duplicate.
    """
    # Join: label parent vs child and spell out the FK expectation,
    # using the join spec directly (independent of primary_key).
    if group.kind == "join" and group.join is not None:
        j = group.join
        lines: List[str] = ["  JOIN key (parent 1 -- N child):"]
        for role, sheet, cols in (
            ("parent", j.parent_sheet, j.parent_columns),
            ("child", j.child_sheet, j.child_columns),
        ):
            df = frames.get(sheet)
            if df is None:
                continue
            n_rows, n_keyed, n_distinct, max_mult = _key_cardinality(df, cols)
            lines.append(
                f"    {role} {sheet}.{list(cols)}: {n_rows} rows, "
                f"{n_distinct} distinct key value(s) -> "
                f"{_uniqueness_label(n_keyed, n_distinct, max_mult)}"
            )
        lines.append(
            "    NOTE: a child key that matches a parent is a satisfied "
            "foreign key (correct), NOT a duplicate; repeated child keys "
            "are the expected many-to-one. The only cross-table finding "
            "here is fk_orphan (child key with NO matching parent)."
        )
        return "\n".join(lines)

    # Union: per-sheet uniqueness of the resolved primary key.
    key_cols = _resolve_key_columns(group)
    if not key_cols:
        return (
            "  (no key resolved -- pair only on an obvious natural "
            "identifier, and be conservative)"
        )
    lines = ["  PRIMARY key cardinality per sheet:"]
    for sheet in group.sheets:
        cols = key_cols.get(sheet)
        df = frames.get(sheet)
        if not cols or df is None:
            continue
        n_rows, n_keyed, n_distinct, max_mult = _key_cardinality(df, cols)
        note = ""
        if max_mult > 1:
            note = (
                " -- repeats look like event/log grain; rows sharing a "
                "key are SEPARATE records, not duplicates/conflicts"
            )
        lines.append(
            f"    {sheet}.{list(cols)}: {n_rows} rows, {n_distinct} "
            f"distinct -> {_uniqueness_label(n_keyed, n_distinct, max_mult)}"
            f"{note}"
        )
    return "\n".join(lines)


def _build_key_aligned_pages(
    *,
    group: ResolvedGroup,
    frames: Mapping[str, pd.DataFrame],
    n_sample_rows: int,
    key_targeted: bool = False,
) -> Optional[List[List[_PageSlice]]]:
    """Bucket rows by normalized key so matching entities co-locate.

    Returns ``None`` when the group has no usable key (caller falls
    back to :func:`_build_pages`). Every row is assigned to a bucket:
    keyed rows join the bucket for their normalized key (across all
    sheets); rows with a missing key get a unique singleton bucket so
    they are still shown. Buckets are greedily packed into pages so
    each sheet shows at most ``n_sample_rows`` rows per page (a single
    oversized bucket is never split, to keep matching rows together).

    When ``key_targeted`` is set, only buckets spanning >= 2 sheets
    survive - focusing LLM spend on rows that can actually conflict
    across sheets (the budget mode). Returns ``[]`` (no pages) when
    targeting leaves nothing.
    """
    key_cols = _resolve_key_columns(group)
    if not key_cols:
        return None

    sheets = list(group.sheets)
    buckets: Dict[Any, Dict[str, List[int]]] = {}
    order: List[Any] = []

    def _bucket(bid: Any) -> Dict[str, List[int]]:
        b = buckets.get(bid)
        if b is None:
            b = {}
            buckets[bid] = b
            order.append(bid)
        return b

    for sheet in sheets:
        df = frames.get(sheet)
        if df is None or df.empty:
            continue
        raw_cols = key_cols.get(sheet)
        for ridx in range(len(df)):
            kv = _row_key(df, ridx, raw_cols) if raw_cols else None
            bid: Any = ("k", kv) if kv is not None else ("nk", sheet, ridx)
            _bucket(bid).setdefault(sheet, []).append(ridx)

    # Deterministic ordering: keyed buckets (sorted by key text) first,
    # then singleton no-key buckets in discovery order.
    keyed = sorted(
        (b for b in order if b[0] == "k"), key=lambda b: tuple(b[1]),
    )
    no_key = [b for b in order if b[0] == "nk"]
    ordered = keyed + no_key

    if key_targeted:
        ordered = [
            b for b in ordered
            if sum(1 for s in buckets[b] if buckets[b][s]) >= 2
        ]
    if not ordered:
        return []

    # Greedy bin-packing by per-sheet row budget.
    pages_buckets: List[List[Any]] = []
    cur: List[Any] = []
    cur_counts: Dict[str, int] = {}
    for bid in ordered:
        occ = buckets[bid]
        exceeds = any(
            cur_counts.get(s, 0) > 0
            and cur_counts.get(s, 0) + len(rows) > n_sample_rows
            for s, rows in occ.items()
        )
        if exceeds and cur:
            pages_buckets.append(cur)
            cur = []
            cur_counts = {}
        cur.append(bid)
        for s, rows in occ.items():
            cur_counts[s] = cur_counts.get(s, 0) + len(rows)
    if cur:
        pages_buckets.append(cur)

    pages: List[List[_PageSlice]] = []
    for bucket_ids in pages_buckets:
        per_sheet: Dict[str, List[int]] = {}
        for bid in bucket_ids:
            for s, rows in buckets[bid].items():
                per_sheet.setdefault(s, []).extend(rows)
        page = [
            _PageSlice(
                sheet=s, row_offsets=tuple(sorted(per_sheet.get(s, []))),
            )
            for s in sheets
        ]
        pages.append(page)
    return pages


def _pages_for_group(
    *,
    group: ResolvedGroup,
    frames: Mapping[str, pd.DataFrame],
    n_sample_rows: int,
    seed: int,
    key_align: bool,
    key_targeted: bool,
) -> List[List[_PageSlice]]:
    """Pick the page layout for one confirmed group.

    Key-aligned paging is used when requested AND the group exposes a
    usable key; otherwise we fall back to the legacy seeded-shuffle
    paging. ``key_targeted`` implies key alignment and additionally
    drops single-sheet buckets; when the group has no key it yields no
    pages (nothing can be cross-checked under the budget mode).
    """
    if key_align or key_targeted:
        keyed = _build_key_aligned_pages(
            group=group, frames=frames,
            n_sample_rows=n_sample_rows, key_targeted=key_targeted,
        )
        if keyed is not None:
            return keyed
        if key_targeted:
            return []
    return _build_pages(
        sheets=list(group.sheets), frames=frames,
        n_sample_rows=n_sample_rows, seed=seed,
    )


# ---------------------------------------------------------------------------
# Structural priors rendered into the detect prompt (recall-first):
# value-level hints only (alias pairs, orphan VALUES, sentinel tokens,
# synonym/homonym column pairs) - never raw global row indices, which
# would clash with the prompt's LOCAL-index contract.
# ---------------------------------------------------------------------------


def _render_priors_block(
    priors: Any, group: ResolvedGroup, *, max_per_kind: int = 8,
) -> str:
    """Render the structural priors relevant to ``group`` as candidate
    findings the detector can confirm or reject.

    Returns a ``(none)`` block when no priors apply (e.g. after a lossy
    ``--multi-table-from`` round-trip, where priors come back empty).
    """
    none_block = "  (no structural priors available for this group)"
    if priors is None:
        return none_block
    sheets = set(group.sheets)
    lines: List[str] = []

    fk = [
        c for c in getattr(priors, "fk_candidates", ())
        if c.parent in sheets and c.child in sheets
    ][:max_per_kind]
    for c in fk:
        vals = ", ".join(repr(v) for v in c.orphan_values[:5])
        lines.append(
            f"  - possible FK: {c.child}.{c.child_column!r} references "
            f"{c.parent}.{c.parent_column!r} (raw coverage {c.coverage:.2f}, "
            f"normalized {c.normalized_coverage:.2f}); example child values "
            f"with no parent match: {vals or '(none)'}"
        )

    naming = [
        c for c in getattr(priors, "naming_candidates", ())
        if c.sheet_a in sheets and c.sheet_b in sheets
    ][:max_per_kind]
    for c in naming:
        lines.append(
            f"  - {c.relation}: {c.sheet_a}.{c.column_a!r} vs "
            f"{c.sheet_b}.{c.column_b!r} (value-set Jaccard {c.jaccard:.2f})"
        )

    emb = [
        c for c in getattr(priors, "embedding_match_candidates", ())
        if c.sheet_a in sheets and c.sheet_b in sheets
    ][:max_per_kind]
    for c in emb:
        pairs = "; ".join(f"{a!r}~{b!r}" for a, b, _cos in c.matches[:3])
        lines.append(
            f"  - semantic alias: {c.sheet_a}.{c.column_a!r} <-> "
            f"{c.sheet_b}.{c.column_b!r}: {pairs}"
        )

    nulls = [
        c for c in getattr(priors, "null_sentinel_candidates", ())
        if c.sheet_a in sheets and c.sheet_b in sheets
    ][:max_per_kind]
    for c in nulls:
        lines.append(
            f"  - null-placeholder: column {c.column!r} uses "
            f"{c.sentinel_a!r} on {c.sheet_a} vs {c.sentinel_b!r} on "
            f"{c.sheet_b}"
        )

    if not lines:
        return none_block
    return "\n".join(lines)


def _render_column_mapping(group: ResolvedGroup) -> str:
    if not group.column_mapping:
        return "  (none)"
    parts: List[str] = []
    for canonical, entries in group.column_mapping.items():
        variants = ", ".join(
            f"{e.sheet}:{e.column!r}" for e in entries
        )
        parts.append(f"  - {canonical!r}: {variants}")
    return "\n".join(parts)


def _render_page(
    *,
    page: Sequence[_PageSlice],
    frames: Mapping[str, pd.DataFrame],
    group: ResolvedGroup,
    max_cell_chars: int = 80,
) -> str:
    """Render one page of rows. Each row is prefixed with its
    LOCAL 0-based index (matching the prompt contract).

    For each sheet we project to the canonical-named columns
    declared by the group's mapping so the LLM sees a unified
    schema across sheets.
    """
    parts: List[str] = []
    # Build canonical -> raw column lookup per sheet from the
    # group's column_mapping.
    raw_per_sheet: Dict[str, Dict[str, str]] = {}
    for canonical, entries in (group.column_mapping or {}).items():
        for e in entries:
            raw_per_sheet.setdefault(e.sheet, {})[canonical] = e.column

    canonicals = list((group.column_mapping or {}).keys())

    for slc in page:
        sheet = slc.sheet
        df = frames.get(sheet)
        if df is None or df.empty or not slc.row_offsets:
            parts.append(f"=== {sheet!r} (no rows on this page) ===")
            parts.append("")
            continue
        col_lookup = raw_per_sheet.get(sheet, {})
        present = [c for c in canonicals if col_lookup.get(c) in df.columns]
        if not present:
            # Fall back to the sheet's raw columns; the LLM can
            # still triangulate.
            present = [str(c) for c in df.columns]
            col_lookup = {c: c for c in present}
        parts.append(f"=== {sheet!r} ({len(slc.row_offsets)} row(s)) ===")
        # Header line: canonical names, row index column first.
        parts.append("  row_idx | " + " | ".join(present))
        for local_idx, global_row in enumerate(slc.row_offsets):
            cells: List[str] = [str(local_idx)]
            for c in present:
                raw_col = col_lookup.get(c, c)
                try:
                    val = df.iat[global_row, df.columns.get_loc(raw_col)]
                except (KeyError, IndexError):
                    cells.append("")
                    continue
                s = "" if (val is None or pd.isna(val)) else str(val)
                if len(s) > max_cell_chars:
                    s = s[:max_cell_chars] + "…"
                cells.append(s)
            parts.append("  " + " | ".join(cells))
        parts.append("")
    return "\n".join(parts).rstrip() or "(no samples)"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(
    text: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Pull ``findings`` array out of the LLM's JSON response."""
    if not text:
        return [], "empty response"
    body = text.strip()
    fence = JSON_FENCE_RE.search(body)
    if fence:
        body = fence.group(1)
    obj_match = JSON_OBJECT_RE.search(body)
    if not obj_match:
        return [], f"no JSON object in response: {text[:200]!r}"
    try:
        obj = json.loads(obj_match.group(0))
    except json.JSONDecodeError as exc:
        return [], f"JSON decode failed: {exc}"
    if not isinstance(obj, dict):
        return [], "top-level JSON is not an object"
    findings = obj.get("findings")
    if findings is None:
        # Permissive: some models may emit the array at the top level
        # under a different key.
        return [], "'findings' key missing"
    if not isinstance(findings, list):
        return [], "'findings' is not a list"
    return findings, None


def _normalize_kind(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in SEMANTIC_KINDS:
        return s
    # Permissive fallbacks.
    if s in {"duplicate", "dup"}:
        return "duplicate_row"
    if s in {"conflict", "mismatch", "disagreement"}:
        return "attribute_conflict"
    if s in {"format", "format_diff", "rendering"}:
        return "format_drift"
    if s in {"unit", "scale"}:
        return "unit_drift"
    if s in {"surface", "casing", "abbreviation"}:
        return "surface_drift"
    if s in {"null", "missing", "placeholder"}:
        return "null_placeholder"
    if s in {"snapshot", "rollup", "aggregation", "aggregate"}:
        return "snapshot_aggregation"
    if s in {"fk", "orphan", "fk_missing"}:
        return "fk_orphan"
    return "attribute_conflict"


def _normalize_verdict(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in VERDICTS:
        return s
    if s in {"confirmed", "yes", "true", "real"}:
        return "confirmed_inconsistency"
    if s in {"benign", "ok", "expected", "false_positive_benign"}:
        return "benign_variation"
    if s in {"false", "no", "spurious", "not_a_violation"}:
        # Map "not a violation" to "benign_variation" for the
        # semantic detector - if it's reporting nothing went wrong,
        # we don't want to surface it as an inconsistency at all.
        # The downstream filter can still display these.
        return "benign_variation"
    return "uncertain"


def _normalize_confidence(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in CONFIDENCES:
        return s
    return "medium"


def _normalize_locations(
    raw: Any,
    *,
    page: Sequence[_PageSlice],
) -> List[Tuple[str, int]]:
    """Map the LLM's local ``(sheet, row)`` references back to
    global row indices using ``page.row_offsets``.

    Drops references the LLM hallucinated (sheet not in this page,
    row outside the rendered window). Out-of-range indices simply
    become an empty list - the caller skips findings with no valid
    locations.
    """
    if not isinstance(raw, list):
        return []
    by_sheet: Dict[str, _PageSlice] = {s.sheet: s for s in page}
    out: List[Tuple[str, int]] = []
    seen: set = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sheet = str(entry.get("sheet") or "").strip()
        try:
            local_row = int(entry.get("row"))
        except (TypeError, ValueError):
            continue
        slc = by_sheet.get(sheet)
        if slc is None:
            continue
        if local_row < 0 or local_row >= len(slc.row_offsets):
            continue
        global_row = slc.row_offsets[local_row]
        key = (sheet, global_row)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class SemanticPassResult:
    """Bundle of (violations, verifications, stats) returned by
    :func:`detect_workbook_inconsistencies`.

    The caller persists ``violations`` to ``multi_table_violations.jsonl``
    and ``verifications`` to ``multi_table_verifications.jsonl`` so
    the dashboard renders them through the same code path used by
    the compiler+verifier strategy.
    """

    violations: List[MultiTableViolation]
    verifications: List[Verification]
    n_calls: int
    n_errored: int
    n_parse_errored: int


def expected_semantic_call_count(
    *,
    resolved: ResolvedWorkbook,
    frames: Mapping[str, pd.DataFrame],
    n_sample_rows: int,
    max_pages: Optional[int] = None,
    key_align: bool = True,
    key_targeted: bool = False,
) -> int:
    """Pre-compute how many LLM calls a semantic pass over ``resolved``
    will make.

    Mirrors the loop in :func:`detect_workbook_inconsistencies`
    exactly (including the key-aligned vs. shuffle page layout) so the
    runner can render a call-accurate ``tqdm`` bar without waiting for
    the calls to actually fire. Cheap: only inspects frame shapes /
    key columns; no LLM I/O.
    """
    total = 0
    for group in resolved.groups:
        if not group.confirmed:
            continue
        # ``seed`` only affects WHICH rows go in which page, not the
        # page COUNT, so any value works for counting.
        pages = _pages_for_group(
            group=group,
            frames=frames,
            n_sample_rows=n_sample_rows,
            seed=0,
            key_align=key_align,
            key_targeted=key_targeted,
        )
        n = len(pages)
        if max_pages is not None:
            n = min(n, max_pages)
        total += n
    return total


def detect_workbook_inconsistencies(
    *,
    resolved: ResolvedWorkbook,
    frames: Mapping[str, pd.DataFrame],
    llm: LLMClient,
    model: str,
    prompt_cfg: DetectPromptConfig,
    n_sample_rows: int = 50,
    max_pages: Optional[int] = None,
    seed: int = 1234,
    key_align: bool = True,
    key_targeted: bool = False,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> SemanticPassResult:
    """Run step-3 LLM detection for every confirmed group.

    For each confirmed group, slice its sheets into pages of at
    most ``n_sample_rows`` rows, fire one LLM call per page, parse
    the ``findings`` array, and translate local row indices back
    to global ones. Findings spanning >1 location emit one
    ``MultiTableViolation`` per location, all linked by the same
    ``violation_id``.

    Pages are key-aligned by default (``key_align=True``): rows that
    share a normalized primary/join key co-locate on the same page so
    the model sees both sides of a cross-sheet conflict together.
    ``key_targeted`` additionally restricts each call to rows whose
    key appears on >= 2 sheets (the budget mode). Both fall back to
    the legacy seeded-shuffle layout for keyless groups.

    A stub :class:`Verification` is emitted per logical finding so
    the dashboard's verdict column lights up uniformly with the
    compiler-strategy code path.
    """
    violations: List[MultiTableViolation] = []
    verifications: List[Verification] = []
    n_calls = 0
    n_errored = 0
    n_parse_errored = 0

    # Dedupe across pages: a finding (kind, sorted locations,
    # columns) keyed once stays once even if multiple pages hit
    # the same row pair. Last-write-wins for verdict (newer page
    # may have more context, so prefer its rationale).
    seen_finding_ids: set = set()

    for group in resolved.groups:
        if not group.confirmed:
            continue
        sheets = list(group.sheets)
        pages = _pages_for_group(
            group=group, frames=frames,
            n_sample_rows=n_sample_rows, seed=seed,
            key_align=key_align, key_targeted=key_targeted,
        )
        if max_pages is not None:
            pages = pages[:max_pages]
        n_pages = len(pages)
        column_mapping_block = _render_column_mapping(group)
        priors_block = _render_priors_block(
            getattr(resolved, "priors", None), group,
        )
        key_uniqueness_block = _render_key_uniqueness_block(group, frames)

        for page_idx, page in enumerate(pages):
            row_samples_block = _render_page(
                page=page, frames=frames, group=group,
            )
            prompt = prompt_cfg.user_template.format(
                workbook=resolved.workbook,
                group_id=group.group_id,
                group_kind=group.kind,
                sheets=sheets,
                primary_key=list(group.primary_key) or "(none)",
                column_mapping_block=column_mapping_block,
                key_uniqueness_block=key_uniqueness_block,
                priors_block=priors_block,
                row_samples_block=row_samples_block,
                page_index=page_idx + 1,
                n_pages=n_pages,
                n_sample_rows=n_sample_rows,
            )
            n_calls += 1
            response: LLMResponse = llm.query(
                model,
                prompt,
                reasoning_effort=prompt_cfg.reasoning_effort,
                reasoning_summary=prompt_cfg.reasoning_summary,
                system_prompt=prompt_cfg.system_prompt,
            )
            # Tick the optional progress callback after the call has
            # completed (success or failure - both consumed budget).
            # Wrapped so a buggy callback can never crash the pass.
            if progress_cb is not None:
                try:
                    progress_cb(1)
                except Exception:  # noqa: BLE001 - progress is decorative
                    logger.debug("progress_cb raised; ignored", exc_info=True)
            if not response.ok:
                n_errored += 1
                logger.warning(
                    "semantic detector LLM error workbook=%s group=%s "
                    "page=%d/%d: %s",
                    resolved.workbook, group.group_id,
                    page_idx + 1, n_pages, response.error,
                )
                continue
            findings, parse_err = _parse_response(response.text)
            if parse_err:
                n_parse_errored += 1
                logger.warning(
                    "semantic detector parse error workbook=%s "
                    "group=%s page=%d/%d: %s",
                    resolved.workbook, group.group_id,
                    page_idx + 1, n_pages, parse_err,
                )
                continue

            for f in findings:
                if not isinstance(f, dict):
                    continue
                kind = _normalize_kind(f.get("kind"))
                verdict = _normalize_verdict(f.get("verdict"))
                confidence = _normalize_confidence(f.get("confidence"))
                locs = _normalize_locations(
                    f.get("locations"), page=page,
                )
                if not locs:
                    continue
                cols_raw = f.get("columns") or []
                columns: Tuple[str, ...] = tuple(
                    str(c) for c in cols_raw if isinstance(c, (str, int))
                )
                evidence = str(f.get("evidence") or "").strip()
                rationale = str(f.get("rationale") or "").strip()

                # Stable id across pages: hash of canonical
                # locations + columns + kind. Collisions across
                # different findings on the same row pair are fine -
                # we keep the last (last-write-wins).
                sorted_locs = sorted(locs)
                vid = _vid(
                    "semantic", resolved.workbook,
                    group.group_id, kind,
                    *(f"{s}:{r}" for s, r in sorted_locs),
                    *columns,
                )
                if vid in seen_finding_ids:
                    continue
                seen_finding_ids.add(vid)

                # Build values payload by snapshotting the cells we
                # claim to flag (using the first column of `columns`
                # if multiple). Falls back to whole-row snapshot.
                values_payload: Dict[str, Any] = {
                    "evidence": evidence,
                    "kind_detail": kind,
                }

                # Emit one MultiTableViolation per (sheet, row)
                # location. Linked locations share the violation_id.
                # If we have only one location, "linked" is empty.
                for primary_idx, (sheet, row) in enumerate(sorted_locs):
                    linked = tuple(
                        (s2, r2) for j, (s2, r2) in enumerate(sorted_locs)
                        if j != primary_idx
                    )
                    violations.append(MultiTableViolation(
                        violation_id=vid,
                        kind=kind,
                        workbook=resolved.workbook,
                        group_id=group.group_id,
                        group_kind=group.kind,
                        sheet=sheet,
                        row=int(row),
                        columns=columns,
                        values={
                            **values_payload,
                            "row_snapshot": _snapshot_row(
                                frames, sheet, row, group, columns,
                            ),
                        },
                        linked=linked,
                        rationale=rationale or evidence,
                    ))

                # Stub verification record so the dashboard verdict
                # column renders consistently with compiler-strategy.
                verifications.append(Verification(
                    violation_id=vid,
                    workbook=resolved.workbook,
                    group_id=group.group_id,
                    kind=kind,
                    verdict=verdict,
                    confidence=confidence,
                    rationale=rationale or evidence,
                    endpoint=response.endpoint,
                    attempts=response.attempts,
                ))

    return SemanticPassResult(
        violations=violations,
        verifications=verifications,
        n_calls=n_calls,
        n_errored=n_errored,
        n_parse_errored=n_parse_errored,
    )


def _snapshot_row(
    frames: Mapping[str, pd.DataFrame],
    sheet: str,
    row: int,
    group: ResolvedGroup,
    columns: Sequence[str],
) -> Dict[str, Any]:
    """Capture the canonical-named cells the finding touches.

    Used for the dashboard tooltip / detail expansion. Resolves
    canonical -> raw column via the group's mapping. Returns
    ``{}`` for an out-of-range row.
    """
    df = frames.get(sheet)
    if df is None or row < 0 or row >= len(df):
        return {}
    raw_lookup: Dict[str, str] = {}
    for canonical, entries in (group.column_mapping or {}).items():
        for e in entries:
            if e.sheet == sheet:
                raw_lookup[canonical] = e.column
                break
    out: Dict[str, Any] = {}
    cols_to_render = list(columns) or list(raw_lookup.keys())
    for c in cols_to_render:
        raw_col = raw_lookup.get(c, c)
        if raw_col not in df.columns:
            continue
        try:
            val = df.iat[row, df.columns.get_loc(raw_col)]
        except (KeyError, IndexError):
            continue
        out[c] = _coerce_value(val)
    return out


def summarize(result: SemanticPassResult) -> Dict[str, Any]:
    """Roll up a :class:`SemanticPassResult` into ``summary.json``
    shape (mirrors :func:`multi_table_verifier.summarize` so the
    overview dashboard's headline numbers work for either
    strategy)."""
    by_kind: Dict[str, int] = {}
    by_verdict: Dict[str, int] = {}
    by_kind_verdict: Dict[str, Dict[str, int]] = {}
    # Index verifications by violation_id for the verdict roll-up
    # (one verification per logical finding).
    ver_by_id = {v.violation_id: v for v in result.verifications}
    seen_ids: set = set()
    for vio in result.violations:
        if vio.violation_id in seen_ids:
            continue
        seen_ids.add(vio.violation_id)
        ver = ver_by_id.get(vio.violation_id)
        verdict = ver.verdict if ver else "uncertain"
        by_kind[vio.kind] = by_kind.get(vio.kind, 0) + 1
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
        bk = by_kind_verdict.setdefault(vio.kind, {})
        bk[verdict] = bk.get(verdict, 0) + 1
    return {
        "strategy": "semantic",
        "n_calls": result.n_calls,
        "n_errored": result.n_errored,
        "n_parse_errored": result.n_parse_errored,
        "n_findings": len(seen_ids),
        "n_violation_records": len(result.violations),
        "by_kind": by_kind,
        "by_verdict": by_verdict,
        "by_kind_verdict": by_kind_verdict,
    }


# ---------------------------------------------------------------------------
# Structural complement (always-on deterministic detectors under the
# semantic strategy). The compiler finds exact fk_orphan / pk_collision
# / dtype_drift / header_synonym cases cheaply; we merge any the
# semantic pass did not already surface so a confirmed group whose
# conflicting rows happened to miss a page still reports the structural
# certainty.
# ---------------------------------------------------------------------------


def _violation_signature(
    kind: str,
    locations: Iterable[Tuple[str, int]],
    columns: Iterable[str],
) -> Tuple[str, Tuple[Tuple[str, int], ...], Tuple[str, ...]]:
    return (
        kind,
        tuple(sorted(locations)),
        tuple(sorted(str(c) for c in columns)),
    )


def _finding_signatures(
    violations: Sequence[MultiTableViolation],
) -> set:
    """Collapse per-row violations back to per-finding signatures keyed
    by ``violation_id`` (kind, full location set, columns) so a
    compiler finding covering the same cells can be de-duplicated."""
    by_vid: Dict[str, Dict[str, Any]] = {}
    for v in violations:
        slot = by_vid.setdefault(
            v.violation_id,
            {"kind": v.kind, "columns": tuple(v.columns), "locs": set()},
        )
        slot["locs"].add((v.sheet, int(v.row)))
        for s, r in v.linked:
            slot["locs"].add((s, int(r)))
    return {
        _violation_signature(slot["kind"], slot["locs"], slot["columns"])
        for slot in by_vid.values()
    }


def structural_complement(
    *,
    resolved_list: Sequence[ResolvedWorkbook],
    frames: Mapping[str, pd.DataFrame],
    semantic_violations: Sequence[MultiTableViolation],
) -> Tuple[List[MultiTableViolation], List[Verification]]:
    """Run the deterministic compiler over every workbook and return
    the violations (plus stub verifications) the semantic pass did not
    already produce.

    De-duplication is by (kind, sorted locations, sorted columns) so a
    compiler finding matching a semantic finding's full location set is
    dropped. Surviving compiler violations get a stub
    :class:`Verification` with verdict ``"uncertain"`` (recall-first:
    the human labeler adjudicates; there is no downstream LLM verifier
    under the semantic strategy).
    """
    have = _finding_signatures(semantic_violations)
    extra_violations: List[MultiTableViolation] = []
    extra_verifications: List[Verification] = []
    for resolved in resolved_list:
        try:
            compiled = compile_violations(resolved=resolved, frames=frames)
        except Exception:  # noqa: BLE001
            logger.exception(
                "structural complement compile failed for workbook=%s",
                resolved.workbook,
            )
            continue
        by_vid: Dict[str, List[MultiTableViolation]] = defaultdict(list)
        for v in compiled:
            by_vid[v.violation_id].append(v)
        for vid, rows in by_vid.items():
            locs: set = set()
            for v in rows:
                locs.add((v.sheet, int(v.row)))
                for s, r in v.linked:
                    locs.add((s, int(r)))
            sig = _violation_signature(rows[0].kind, locs, rows[0].columns)
            if sig in have:
                continue
            have.add(sig)  # guard against duplicate compiler findings too
            extra_violations.extend(rows)
            extra_verifications.append(Verification(
                violation_id=vid,
                workbook=rows[0].workbook,
                group_id=rows[0].group_id,
                kind=rows[0].kind,
                verdict="uncertain",
                confidence="medium",
                rationale=(
                    "deterministic structural candidate "
                    "(compiler complement)"
                ),
            ))
    return extra_violations, extra_verifications
