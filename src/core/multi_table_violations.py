"""Deterministic compiler from confirmed cross-sheet groups to per-row
violations.

Consumes :class:`~src.core.multi_table_resolver.ResolvedWorkbook`
and emits :class:`MultiTableViolation` records of four kinds:

    * ``header_synonym``     - a canonical column has >=2 distinct
                               raw header variants across sheets.
                               One record per (sheet, canonical) so
                               every variant gets logged.
    * ``duplicate_row``      - PK appears in >=2 sheets and every
                               attribute column agrees across the
                               sheets that carry it. One record per
                               (sheet, row) location, all linked by
                               the same ``violation_id``.
    * ``attribute_conflict`` - PK appears in >=2 sheets and at
                               least one attribute disagrees. One
                               record per (sheet, row), all linked
                               by the same ``violation_id``. The
                               offending columns are recorded in
                               ``columns``.
    * ``dtype_drift``        - canonical column has dtype kind
                               mismatch across sheet variants. One
                               record per (sheet, variant column).

Output shape (the user picked "one record per affected (sheet, row);
link via shared violation_id"):

.. code-block:: json

    {
      "violation_id": "<workbook>::<group>::<kind>::<idx>",
      "kind": "duplicate_row",
      "workbook": "...",
      "group_id": "u1",
      "group_kind": "union",
      "sheet": "sheet_b.csv",
      "row": 12,
      "columns": ["Name", "School Year"],
      "values": {"Name": "...", "School Year": 2022},
      "linked": [{"sheet": "sheet_a.csv", "row": 5}],
      "rationale": "..."
    }

Pure-Python; no LLM, no sandboxing, no codegen. Tests assert each
emitted record points at a real (sheet, row) pair.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .multi_table_resolver import ColumnMappingEntry, ResolvedGroup, ResolvedWorkbook

logger = logging.getLogger(__name__)


VIOLATION_KINDS = (
    "header_synonym",
    "duplicate_row",
    "attribute_conflict",
    "dtype_drift",
    "fk_orphan",
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiTableViolation:
    """One violation record. Multi-row violations (duplicate_row,
    attribute_conflict) emit ONE instance per affected (sheet, row);
    they share a ``violation_id`` so the report can link them.
    """

    violation_id: str
    kind: str
    workbook: str
    group_id: str
    group_kind: str
    sheet: str
    row: int
    columns: Tuple[str, ...]
    values: Mapping[str, Any]
    linked: Tuple[Tuple[str, int], ...]   # other (sheet, row) locations
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "kind": self.kind,
            "workbook": self.workbook,
            "group_id": self.group_id,
            "group_kind": self.group_kind,
            "sheet": self.sheet,
            "row": int(self.row),
            "columns": list(self.columns),
            "values": dict(self.values),
            "linked": [
                {"sheet": s, "row": int(r)} for s, r in self.linked
            ],
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vid(*parts: Any) -> str:
    """Stable, short violation id."""
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def _coerce_value(v: Any) -> Any:
    """JSON-safe coercion for the report payload."""
    if v is None:
        return None
    if isinstance(v, float):
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(float(v)) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    if isinstance(v, (int, bool)):
        return v
    return str(v)


def _dtype_kind(series: pd.Series) -> str:
    """Coarse dtype family used for dtype-drift detection."""
    dt = series.dtype
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    if pd.api.types.is_integer_dtype(dt):
        return "int"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime"
    if pd.api.types.is_object_dtype(dt):
        # Disambiguate string-like vs mixed by sampling.
        non_null = series.dropna().head(50)
        if non_null.empty:
            return "object"
        if all(isinstance(v, str) for v in non_null):
            return "str"
        return "mixed"
    return str(dt)


def _normalize_pk_value(v: Any) -> Any:
    """Trim/casefold strings for PK matching; pass numerics through."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        return s or None
    if pd.isna(v):
        return None
    return v


def _build_pk_index(
    *,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    primary_key: Sequence[str],
    column_mapping: Mapping[str, Sequence[ColumnMappingEntry]],
) -> Dict[Tuple[Any, ...], List[Tuple[str, int]]]:
    """Map normalized PK tuple -> [(sheet, row_index), ...].

    A sheet contributes only when it carries every PK canonical
    (per the column_mapping). PK normalisation strips whitespace and
    case-folds strings so "C001" and " c001 " collapse.
    """
    pk_index: Dict[Tuple[Any, ...], List[Tuple[str, int]]] = {}
    for sheet in sheets:
        df = frames.get(sheet)
        if df is None or df.empty:
            continue
        # Resolve the per-sheet column for each PK canonical.
        pk_cols: List[str] = []
        ok = True
        for canonical in primary_key:
            entries = column_mapping.get(canonical, ())
            col = next((e.column for e in entries if e.sheet == sheet), None)
            if col is None or col not in df.columns:
                ok = False
                break
            pk_cols.append(col)
        if not ok:
            continue
        for idx, row in df[pk_cols].iterrows():
            tup = tuple(_normalize_pk_value(row[c]) for c in pk_cols)
            if any(t is None for t in tup):
                continue
            pk_index.setdefault(tup, []).append((sheet, int(idx)))
    return pk_index


def _attribute_columns_per_sheet(
    *,
    sheet: str,
    column_mapping: Mapping[str, Sequence[ColumnMappingEntry]],
    primary_key: Sequence[str],
) -> Dict[str, str]:
    """Return ``{canonical: raw_column_in_this_sheet}`` for non-PK
    canonicals that this sheet carries."""
    pk_set = set(primary_key)
    out: Dict[str, str] = {}
    for canonical, entries in column_mapping.items():
        if canonical in pk_set:
            continue
        col = next((e.column for e in entries if e.sheet == sheet), None)
        if col is not None:
            out[canonical] = col
    return out


# ---------------------------------------------------------------------------
# Violation kind compilers
# ---------------------------------------------------------------------------


def _compile_header_synonyms(
    group: ResolvedGroup,
) -> List[MultiTableViolation]:
    """One record per (sheet, raw_column) when a canonical has >=2
    distinct raw header variants across the group. Each record's
    ``linked`` lists the other variants."""
    out: List[MultiTableViolation] = []
    for canonical, entries in group.column_mapping.items():
        # Skip canonicals where all variants share the same raw name.
        raw_names = {e.column for e in entries}
        if len(raw_names) < 2:
            continue
        all_locations = [(e.sheet, e.column) for e in entries]
        for i, e in enumerate(entries):
            others = [
                (s, c) for j, (s, c) in enumerate(all_locations) if j != i
            ]
            vid = _vid(group.workbook, group.group_id, "header_synonym",
                       canonical, e.sheet, e.column)
            # Use row=-1 to indicate "this is a header-level finding,
            # not row-bound"; the report renderer treats negative rows
            # specially.
            out.append(MultiTableViolation(
                violation_id=vid,
                kind="header_synonym",
                workbook=group.workbook,
                group_id=group.group_id,
                group_kind=group.kind,
                sheet=e.sheet,
                row=-1,
                columns=(e.column,),
                values={"raw_header": e.column, "canonical": canonical},
                linked=tuple((s, -1) for s, _ in others),
                rationale=(
                    f"Canonical {canonical!r} appears under "
                    f"{sorted(raw_names)} across sheets in this group; "
                    f"this sheet uses {e.column!r}."
                ),
            ))
    return out


def _compile_dtype_drift(
    group: ResolvedGroup,
    frames: Mapping[str, pd.DataFrame],
) -> List[MultiTableViolation]:
    """One record per (sheet, raw_column) when a canonical's dtype
    kind disagrees across sheets."""
    out: List[MultiTableViolation] = []
    for canonical, entries in group.column_mapping.items():
        kinds: Dict[Tuple[str, str], str] = {}
        for e in entries:
            df = frames.get(e.sheet)
            if df is None or e.column not in df.columns:
                continue
            kinds[(e.sheet, e.column)] = _dtype_kind(df[e.column])
        if len(set(kinds.values())) < 2:
            continue
        # Only the variants that differ from the dominant kind get a
        # finding. With multiple disagreeing kinds, every variant is
        # surfaced.
        kind_counts: Dict[str, int] = {}
        for k in kinds.values():
            kind_counts[k] = kind_counts.get(k, 0) + 1
        all_keys = list(kinds.items())
        for (sheet, col), k in all_keys:
            others = [
                (s, c, kk) for (s, c), kk in all_keys
                if (s, c) != (sheet, col)
            ]
            vid = _vid(group.workbook, group.group_id, "dtype_drift",
                       canonical, sheet, col)
            out.append(MultiTableViolation(
                violation_id=vid,
                kind="dtype_drift",
                workbook=group.workbook,
                group_id=group.group_id,
                group_kind=group.kind,
                sheet=sheet,
                row=-1,
                columns=(col,),
                values={
                    "raw_header": col,
                    "canonical": canonical,
                    "dtype_kind": k,
                },
                linked=tuple((s, -1) for s, _, _ in others),
                rationale=(
                    f"Canonical {canonical!r} has dtype-kind {k!r} on "
                    f"{sheet!r}.{col!r} but "
                    f"{ {kk for _, _, kk in others} } elsewhere "
                    f"(value-encoding drift across sheets)."
                ),
            ))
    return out


def _compile_pk_collisions(
    group: ResolvedGroup,
    frames: Mapping[str, pd.DataFrame],
) -> Tuple[List[MultiTableViolation], List[MultiTableViolation]]:
    """Compute duplicate_row + attribute_conflict findings.

    Returns ``(duplicates, conflicts)``. A PK shared across >=2
    sheets is a "collision" - all attribute columns that BOTH sheets
    carry are compared (after normalisation). If every shared
    attribute agrees -> duplicate_row; if at least one disagrees ->
    attribute_conflict, and the disagreeing columns are recorded in
    each emitted record's ``columns`` field.
    """
    duplicates: List[MultiTableViolation] = []
    conflicts: List[MultiTableViolation] = []
    if not group.primary_key:
        return duplicates, conflicts

    pk_index = _build_pk_index(
        sheets=group.sheets, frames=frames,
        primary_key=group.primary_key,
        column_mapping=group.column_mapping,
    )
    if not pk_index:
        return duplicates, conflicts

    for pk_tuple, locations in pk_index.items():
        if len(locations) < 2:
            continue
        # Resolve each location's attribute snapshot (canonical -> value).
        snapshots: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for sheet, idx in locations:
            df = frames.get(sheet)
            if df is None:
                continue
            attr_cols = _attribute_columns_per_sheet(
                sheet=sheet,
                column_mapping=group.column_mapping,
                primary_key=group.primary_key,
            )
            row = df.loc[idx] if idx in df.index else None
            if row is None:
                continue
            snap: Dict[str, Any] = {}
            for canonical, raw_col in attr_cols.items():
                if raw_col not in df.columns:
                    continue
                snap[canonical] = _normalize_pk_value(row[raw_col])
            snapshots[(sheet, idx)] = snap

        # Find conflicting canonicals: any canonical present in >=2
        # snapshots whose values differ.
        conflicting: List[str] = []
        canonical_values: Dict[str, set] = {}
        for snap in snapshots.values():
            for canonical, val in snap.items():
                canonical_values.setdefault(canonical, set()).add(val)
        for canonical, vals in canonical_values.items():
            non_null = {v for v in vals if v is not None}
            if len(non_null) >= 2:
                conflicting.append(canonical)

        all_locs = sorted(snapshots.keys())
        if not all_locs:
            continue
        if conflicting:
            for sheet, idx in all_locs:
                others = [(s, r) for s, r in all_locs if (s, r) != (sheet, idx)]
                vid = _vid(group.workbook, group.group_id,
                           "attribute_conflict", *pk_tuple)
                # Record raw values for the conflicting columns from
                # the actual DF, not the normalised snapshot.
                df = frames.get(sheet)
                values_payload: Dict[str, Any] = {
                    "primary_key": list(pk_tuple),
                }
                if df is not None and idx in df.index:
                    for canonical in conflicting:
                        col = next(
                            (e.column for e in group.column_mapping[canonical]
                             if e.sheet == sheet),
                            None,
                        )
                        if col is not None and col in df.columns:
                            values_payload[canonical] = _coerce_value(
                                df.at[idx, col]
                            )
                conflicts.append(MultiTableViolation(
                    violation_id=vid,
                    kind="attribute_conflict",
                    workbook=group.workbook,
                    group_id=group.group_id,
                    group_kind=group.kind,
                    sheet=sheet,
                    row=int(idx),
                    columns=tuple(conflicting),
                    values=values_payload,
                    linked=tuple(others),
                    rationale=(
                        f"Primary key {list(pk_tuple)!r} appears in "
                        f"{len(all_locs)} sheets; the columns "
                        f"{conflicting!r} disagree across them."
                    ),
                ))
        else:
            # All shared attributes agree -> exact duplicate.
            for sheet, idx in all_locs:
                others = [(s, r) for s, r in all_locs if (s, r) != (sheet, idx)]
                vid = _vid(group.workbook, group.group_id,
                           "duplicate_row", *pk_tuple)
                duplicates.append(MultiTableViolation(
                    violation_id=vid,
                    kind="duplicate_row",
                    workbook=group.workbook,
                    group_id=group.group_id,
                    group_kind=group.kind,
                    sheet=sheet,
                    row=int(idx),
                    columns=tuple(group.primary_key),
                    values={"primary_key": list(pk_tuple)},
                    linked=tuple(others),
                    rationale=(
                        f"Primary key {list(pk_tuple)!r} appears in "
                        f"{len(all_locs)} sheets with identical "
                        f"attribute values - cross-sheet duplicate."
                    ),
                ))
    return duplicates, conflicts


def _compile_fk_orphans(
    group: ResolvedGroup,
    frames: Mapping[str, pd.DataFrame],
) -> List[MultiTableViolation]:
    """For confirmed JOIN groups, list child rows whose join key is
    not in the parent's value set (after string normalisation)."""
    if group.kind != "join" or group.join is None:
        return []
    spec = group.join
    parent = frames.get(spec.parent_sheet)
    child = frames.get(spec.child_sheet)
    if parent is None or child is None:
        return []
    if not all(c in parent.columns for c in spec.parent_columns):
        return []
    if not all(c in child.columns for c in spec.child_columns):
        return []

    def _key(row: pd.Series, cols: Sequence[str]) -> Tuple[Any, ...]:
        return tuple(_normalize_pk_value(row[c]) for c in cols)

    parent_keys: set = set()
    for _, prow in parent[list(spec.parent_columns)].iterrows():
        k = _key(prow, spec.parent_columns)
        if all(v is not None for v in k):
            parent_keys.add(k)

    out: List[MultiTableViolation] = []
    for idx, crow in child[list(spec.child_columns)].iterrows():
        k = _key(crow, spec.child_columns)
        if any(v is None for v in k):
            continue
        if k in parent_keys:
            continue
        vid = _vid(group.workbook, group.group_id, "fk_orphan",
                   spec.child_sheet, idx)
        out.append(MultiTableViolation(
            violation_id=vid,
            kind="fk_orphan",
            workbook=group.workbook,
            group_id=group.group_id,
            group_kind=group.kind,
            sheet=spec.child_sheet,
            row=int(idx),
            columns=tuple(spec.child_columns),
            values={
                "child_key": list(k),
                "parent_sheet": spec.parent_sheet,
                "parent_columns": list(spec.parent_columns),
            },
            linked=(),
            rationale=(
                f"Child row references {list(k)!r} in "
                f"{spec.child_sheet!r}.{list(spec.child_columns)!r} "
                f"but no matching parent row exists in "
                f"{spec.parent_sheet!r}.{list(spec.parent_columns)!r}."
            ),
        ))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compile_violations(
    *,
    resolved: ResolvedWorkbook,
    frames: Mapping[str, pd.DataFrame],
) -> List[MultiTableViolation]:
    """Turn one ``ResolvedWorkbook`` into a flat list of violations.

    Iterates only over confirmed groups. Order is stable: groups in
    declaration order, then header_synonym -> dtype_drift ->
    duplicate_row -> attribute_conflict -> fk_orphan inside each.
    """
    out: List[MultiTableViolation] = []
    for group in resolved.groups:
        if not group.confirmed:
            continue
        try:
            out.extend(_compile_header_synonyms(group))
        except Exception:  # noqa: BLE001
            logger.exception(
                "header_synonym compile failed for group=%s", group.group_id,
            )
        try:
            out.extend(_compile_dtype_drift(group, frames))
        except Exception:  # noqa: BLE001
            logger.exception(
                "dtype_drift compile failed for group=%s", group.group_id,
            )
        try:
            dups, conflicts = _compile_pk_collisions(group, frames)
            out.extend(dups)
            out.extend(conflicts)
        except Exception:  # noqa: BLE001
            logger.exception(
                "pk_collision compile failed for group=%s", group.group_id,
            )
        try:
            out.extend(_compile_fk_orphans(group, frames))
        except Exception:  # noqa: BLE001
            logger.exception(
                "fk_orphan compile failed for group=%s", group.group_id,
            )
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_violations_jsonl(
    violations: Iterable[MultiTableViolation],
    path: Path,
) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for v in violations:
            fh.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
