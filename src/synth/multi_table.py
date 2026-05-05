"""Cross-sheet (workbook-scope) synthetic injectors.

These complement the per-cell :class:`Injector` family with
perturbations that only make sense across two or more sheets in
the same workbook. They are the ground-truth source for the
``inter_table_inconsistency`` benchmark axis.

Each injector takes a mapping ``{sheet: DataFrame}`` (already a
copy provided by the harness) and returns a new mapping plus a
list of :class:`~src.synth.base.InjectionRecord` describing every
cell that was touched.

Currently implemented:

* :class:`CrossSheetAliasSwap` - rewrite a value to a known alias
  ("United States" -> "USA") on one side of a shared column.
* :class:`CrossSheetUnitFlip` - multiply a numeric column by a
  unit-conversion ratio on one sheet (e.g. lbs->kg = 2.20462).
* :class:`CrossSheetOrphanInject` - insert a foreign-key value into
  a child sheet that does not exist on the parent side.
* :class:`CrossSheetAttributeConflict` - flip a descriptive
  attribute on one sheet for a key that also appears in another
  sheet, creating an inter-table inconsistency.
* :class:`CrossSheetRenameHeader` - rename one sheet's column so it
  no longer matches its sibling, simulating naming drift.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from .base import (
    InjectionRecord,
    WorkbookInjector,
    register_workbook_injector,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _shared_columns(frames: Mapping[str, pd.DataFrame]) -> List[Tuple[str, str, str]]:
    """Return all ``(sheet_a, sheet_b, column)`` triples where the
    same column header appears in two different sheets."""
    out: List[Tuple[str, str, str]] = []
    sheets = list(frames.keys())
    for i, a in enumerate(sheets):
        cols_a = set(map(str, frames[a].columns))
        for b in sheets[i + 1:]:
            cols_b = set(map(str, frames[b].columns))
            for c in sorted(cols_a & cols_b):
                out.append((a, b, c))
    return out


def _ensure_copy(frames: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {k: v.copy() for k, v in frames.items()}


# ---------------------------------------------------------------------------
# CrossSheetAliasSwap
# ---------------------------------------------------------------------------


_ALIAS_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("united states", "USA"),
    ("united states", "U.S."),
    ("united states", "America"),
    ("united kingdom", "UK"),
    ("china", "PRC"),
    ("germany", "Deutschland"),
    ("california", "CA"),
    ("new york", "NY"),
    ("texas", "TX"),
    ("massachusetts", "MA"),
)


@register_workbook_injector
class CrossSheetAliasSwap(WorkbookInjector):
    name = "cross_sheet_alias_swap"
    error_type = "inter_table_inconsistency"

    def try_inject(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        rng: random.Random,
    ) -> Optional[Tuple[Dict[str, pd.DataFrame], List[InjectionRecord]]]:
        shared = _shared_columns(frames)
        if not shared:
            return None
        rng.shuffle(shared)
        for sheet_a, sheet_b, col in shared:
            df_a = frames[sheet_a]
            # Find a value in sheet_a whose lowercase form matches the
            # left side of any alias pair.
            for canonical, alias in _ALIAS_PAIRS:
                mask = df_a[col].astype(str).str.strip().str.lower() == canonical
                rows = list(df_a.index[mask])
                if not rows:
                    continue
                row = rng.choice(rows)
                out = _ensure_copy(frames)
                old = str(out[sheet_a].at[row, col])
                out[sheet_a].at[row, col] = alias
                rec = InjectionRecord(
                    source=sheet_a,
                    row=int(row),
                    column=col,
                    error_type=self.error_type,
                    injector=self.name,
                    original_value=old,
                    injected_value=alias,
                    params={"sibling_sheet": sheet_b, "canonical": canonical},
                )
                return out, [rec]
        return None


# ---------------------------------------------------------------------------
# CrossSheetUnitFlip
# ---------------------------------------------------------------------------


_UNIT_RATIOS: Tuple[Tuple[str, float], ...] = (
    ("lb_to_kg", 0.453592),
    ("kg_to_lb", 2.20462),
    ("mi_to_km", 1.60934),
    ("km_to_mi", 0.621371),
    ("scale_x100", 100.0),
    ("scale_x1000", 1000.0),
)


@register_workbook_injector
class CrossSheetUnitFlip(WorkbookInjector):
    name = "cross_sheet_unit_flip"
    error_type = "inter_table_inconsistency"

    def try_inject(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        rng: random.Random,
    ) -> Optional[Tuple[Dict[str, pd.DataFrame], List[InjectionRecord]]]:
        shared = _shared_columns(frames)
        # Filter to numeric-looking columns on both sides.
        numeric: List[Tuple[str, str, str]] = []
        for a, b, c in shared:
            try:
                pd.to_numeric(frames[a][c], errors="raise")
                pd.to_numeric(frames[b][c], errors="raise")
            except (ValueError, TypeError):
                continue
            numeric.append((a, b, c))
        if not numeric:
            return None
        rng.shuffle(numeric)
        sheet_a, sheet_b, col = numeric[0]
        ratio_name, ratio = rng.choice(list(_UNIT_RATIOS))
        out = _ensure_copy(frames)
        records: List[InjectionRecord] = []
        series_a = pd.to_numeric(out[sheet_a][col], errors="coerce")
        for row in out[sheet_a].index:
            old = out[sheet_a].at[row, col]
            v = series_a.at[row]
            if pd.isna(v):
                continue
            new_val = v * ratio
            out[sheet_a].at[row, col] = new_val
            records.append(InjectionRecord(
                source=sheet_a,
                row=int(row),
                column=col,
                error_type=self.error_type,
                injector=self.name,
                original_value=str(old),
                injected_value=str(new_val),
                params={"sibling_sheet": sheet_b, "ratio_name": ratio_name, "ratio": ratio},
            ))
        if not records:
            return None
        return out, records


# ---------------------------------------------------------------------------
# CrossSheetOrphanInject
# ---------------------------------------------------------------------------


@register_workbook_injector
class CrossSheetOrphanInject(WorkbookInjector):
    """Insert a key into the child sheet that has no parent row."""

    name = "cross_sheet_orphan_inject"
    error_type = "inter_table_inconsistency"

    def try_inject(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        rng: random.Random,
    ) -> Optional[Tuple[Dict[str, pd.DataFrame], List[InjectionRecord]]]:
        shared = _shared_columns(frames)
        if not shared:
            return None
        rng.shuffle(shared)
        for sheet_a, sheet_b, col in shared:
            # Treat the smaller-distinct-cardinality side as the parent
            # and the larger as the child.
            distinct_a = frames[sheet_a][col].dropna().astype(str).nunique()
            distinct_b = frames[sheet_b][col].dropna().astype(str).nunique()
            if distinct_a == 0 or distinct_b == 0:
                continue
            if distinct_a <= distinct_b:
                parent, child = sheet_a, sheet_b
            else:
                parent, child = sheet_b, sheet_a
            child_rows = list(frames[child].index)
            if not child_rows:
                continue
            row = rng.choice(child_rows)
            orphan = f"__SYNTH_ORPHAN_{rng.randrange(10_000_000):07d}__"
            out = _ensure_copy(frames)
            old = str(out[child].at[row, col])
            out[child].at[row, col] = orphan
            rec = InjectionRecord(
                source=child,
                row=int(row),
                column=col,
                error_type=self.error_type,
                injector=self.name,
                original_value=old,
                injected_value=orphan,
                params={"parent_sheet": parent},
            )
            return out, [rec]
        return None


# ---------------------------------------------------------------------------
# CrossSheetAttributeConflict
# ---------------------------------------------------------------------------


@register_workbook_injector
class CrossSheetAttributeConflict(WorkbookInjector):
    """For a key that appears in two sheets with a shared attribute
    column, change the attribute value on one side to disagree."""

    name = "cross_sheet_attribute_conflict"
    error_type = "inter_table_inconsistency"

    def try_inject(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        rng: random.Random,
    ) -> Optional[Tuple[Dict[str, pd.DataFrame], List[InjectionRecord]]]:
        # Need at least two sheets sharing two columns: one as key,
        # one as attribute.
        sheets = list(frames.keys())
        for _ in range(20):
            if len(sheets) < 2:
                return None
            sheet_a, sheet_b = rng.sample(sheets, 2)
            cols_shared = sorted(
                set(map(str, frames[sheet_a].columns))
                & set(map(str, frames[sheet_b].columns))
            )
            if len(cols_shared) < 2:
                continue
            rng.shuffle(cols_shared)
            key_col, attr_col = cols_shared[0], cols_shared[1]
            # Find a key that appears on both sides.
            keys_a = set(frames[sheet_a][key_col].astype(str))
            keys_b = set(frames[sheet_b][key_col].astype(str))
            common = sorted(keys_a & keys_b)
            if not common:
                continue
            key = rng.choice(common)
            mask = frames[sheet_a][key_col].astype(str) == key
            rows = list(frames[sheet_a].index[mask])
            if not rows:
                continue
            row = rng.choice(rows)
            out = _ensure_copy(frames)
            old = str(out[sheet_a].at[row, attr_col])
            new_val = old + "_CONFLICT"
            out[sheet_a].at[row, attr_col] = new_val
            rec = InjectionRecord(
                source=sheet_a,
                row=int(row),
                column=attr_col,
                error_type=self.error_type,
                injector=self.name,
                original_value=old,
                injected_value=new_val,
                params={"sibling_sheet": sheet_b, "key_column": key_col, "key_value": key},
            )
            return out, [rec]
        return None


# ---------------------------------------------------------------------------
# CrossSheetRenameHeader
# ---------------------------------------------------------------------------


@register_workbook_injector
class CrossSheetRenameHeader(WorkbookInjector):
    """Rename a shared column on one sheet so it no longer matches
    its sibling - the canonical naming-drift signal."""

    name = "cross_sheet_rename_header"
    error_type = "inter_table_inconsistency"

    def try_inject(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        rng: random.Random,
    ) -> Optional[Tuple[Dict[str, pd.DataFrame], List[InjectionRecord]]]:
        shared = _shared_columns(frames)
        if not shared:
            return None
        rng.shuffle(shared)
        sheet_a, sheet_b, col = shared[0]
        new_name = f"{col}_renamed"
        out = _ensure_copy(frames)
        out[sheet_a] = out[sheet_a].rename(columns={col: new_name})
        rec = InjectionRecord(
            source=sheet_a,
            row=-1,  # header-level perturbation
            column=col,
            error_type=self.error_type,
            injector=self.name,
            original_value=col,
            injected_value=new_name,
            params={"sibling_sheet": sheet_b},
        )
        return out, [rec]
