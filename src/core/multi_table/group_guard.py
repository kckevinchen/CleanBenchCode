"""Deterministic group-level plausibility guard for multi-table findings.

Real data-quality errors are RARE. When a confirmed cross-sheet group
emits a *flood* of row-pairing findings, the cause is almost always a
BAD KEY or BAD GROUPING - not bad data. Such a flood means there is NO
error and the discovered relation is wrong, so this guard DROPS the
whole flood (no labelable finding is emitted); the rejected groups are
recorded only as a count + summary in ``stats["guard"]``.

Two triggers (either fires):

* **Non-identifying key** - the primary key is an ordinal / positional
  index (consecutive-integer values, or an ordinal-named column like
  ``No.`` / ``#`` / ``Index``). Pairing rows by it compares unrelated
  entities, so every "conflict" is spurious.
* **Implausible density** - the group's row-pairing findings cover more
  than ``density_threshold`` of its comparable keys. Replicas that
  should agree don't disagree on a third of their rows; independent
  periods / snapshots (or a bad key) do.

The guard is self-targeting: a genuine replica pair with a few real
conflicts has LOW density and a real key, so it is left untouched. Only
the implausible floods are dropped. Schema-level kinds (``header_synonym``,
``dtype_drift``) and ``fk_orphan`` are always kept; only the row-pairing
kinds are suppressed.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .resolver import ResolvedGroup, _resolved_group_from_dict
from .semantic_detector import _resolve_key_columns, _row_key

logger = logging.getLogger(__name__)


# Kinds that come from cross-sheet ROW PAIRING (suppressed when a group is
# judged implausible). Everything else is schema-level / structural and
# is always kept.
_PAIRING_KINDS = frozenset({
    "attribute_conflict", "duplicate_row", "surface_drift", "unit_drift",
    "format_drift", "null_placeholder", "snapshot_aggregation",
})

# Header names that denote a positional / ordinal index, not an entity id.
_ORDINAL_NAMES = frozenset({
    "no", "no.", "num", "number", "index", "idx", "row", "rownum", "rowno",
    "sno", "srno", "sr", "sl", "slno", "seq", "sequence", "item", "itemno",
    "serial", "serialno", "sno.", "s.no", "s.no.", "rank", "#",
})


def _norm_header(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def _looks_ordinal(values: Sequence[Any]) -> bool:
    """True when the non-null values are (mostly) a consecutive integer run.

    Tolerates gaps from row dropping / subsampling: the distinct integers
    must densely fill their own ``[min, max]`` range and start near 1 (a
    row ordinal), whereas a real id (account numbers, codes) is sparse
    over a large range or non-integer.
    """
    ints: List[int] = []
    n_total = 0
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        n_total += 1
        try:
            f = float(s)
        except (TypeError, ValueError):
            return False
        if f != int(f):
            return False
        ints.append(int(f))
    if n_total < 5 or len(ints) < 5:
        return False
    distinct = set(ints)
    span = max(distinct) - min(distinct) + 1
    if span <= 0:
        return False
    fill = len(distinct) / span
    return fill >= 0.6 and min(distinct) <= 2


def is_non_identifying_key(
    group: ResolvedGroup, frames: Mapping[str, pd.DataFrame],
) -> Tuple[bool, str]:
    """True when the group's pairing key is a positional / ordinal index.

    The key is non-identifying only when EVERY one of its columns is
    ordinal (named or valued) on the sheets where it appears - a
    composite key with one real identifying part still identifies.
    """
    key_cols = _resolve_key_columns(group)
    if not key_cols:
        return False, ""
    all_ordinal = True
    any_checked = False
    example = ""
    for sheet, cols in key_cols.items():
        df = frames.get(sheet)
        if df is None:
            continue
        for c in cols:
            any_checked = True
            named = _norm_header(c) in _ORDINAL_NAMES
            valued = c in df.columns and _looks_ordinal(df[c].tolist())
            if named or valued:
                if not example:
                    example = f"{sheet}.{c!r}"
            else:
                all_ordinal = False
    if any_checked and all_ordinal and example:
        return True, (
            f"primary key is a positional/ordinal index (e.g. {example}) - "
            "cross-sheet pairing compares unrelated rows"
        )
    return False, ""


def pairable_key_count(
    group: ResolvedGroup, frames: Mapping[str, pd.DataFrame],
) -> int:
    """Number of distinct key values that appear on >= 2 sheets."""
    key_cols = _resolve_key_columns(group)
    if not key_cols:
        return 0
    key_sheets: Dict[Any, set] = defaultdict(set)
    for sheet, cols in key_cols.items():
        df = frames.get(sheet)
        if df is None:
            continue
        seen: set = set()
        for ridx in range(len(df)):
            k = _row_key(df, ridx, cols)
            if k is not None:
                seen.add(k)
        for k in seen:
            key_sheets[k].add(sheet)
    return sum(1 for s in key_sheets.values() if len(s) >= 2)


@dataclass
class GuardDecision:
    collapse: bool
    reason: str
    density: float
    n_pairing: int
    n_pairable_keys: int


def evaluate_group(
    group: ResolvedGroup,
    frames: Mapping[str, pd.DataFrame],
    pairing_finding_count: int,
    *,
    density_threshold: float = 0.4,
    min_findings: int = 10,
) -> GuardDecision:
    """Decide whether a confirmed group's row-pairing findings are an
    implausible flood that should collapse to one note."""
    bad_key, key_reason = is_non_identifying_key(group, frames)
    n_keys = pairable_key_count(group, frames)
    if n_keys > 0:
        density = min(1.0, pairing_finding_count / n_keys)
    else:
        max_rows = max(
            (len(frames[s]) for s in group.sheets if s in frames), default=0,
        )
        density = min(1.0, pairing_finding_count / max_rows) if max_rows else 0.0
    dense = density >= density_threshold
    if pairing_finding_count >= min_findings and (bad_key or dense):
        if bad_key:
            reason = (
                f"{key_reason}; {pairing_finding_count} row-pairing findings "
                "suppressed as unreliable"
            )
        elif group.kind == "join":
            reason = (
                f"join group: {pairing_finding_count} cross-sheet "
                "'duplicate'/'conflict' findings - child keys that match the "
                "parent are satisfied foreign keys and repeated child keys are "
                "the expected many-to-one, not duplicates; suppressed"
            )
        else:
            reason = (
                f"{int(round(density * 100))}% of comparable keys disagree - "
                "the sheets look like independent periods / snapshots, not "
                f"replicas; {pairing_finding_count} row-pairing findings "
                "suppressed"
            )
        return GuardDecision(True, reason, density, pairing_finding_count, n_keys)
    return GuardDecision(False, "", density, pairing_finding_count, n_keys)


def filter_scope_leak_dicts(
    *,
    violation_dicts: Sequence[Mapping[str, Any]],
    verification_dicts: Sequence[Mapping[str, Any]],
    drop_same_sheet_pairs: bool = True,
    drop_fk_orphan: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Drop findings that fall outside ``inter_table_inconsistency`` scope.

    ``inter_table_inconsistency`` is about disagreements *across* sheets.
    Two classes of finding leak in and render as a single CSV:

    * **Same-sheet row pairs** - a row-pairing kind (``duplicate_row``,
      ``attribute_conflict``, ...) whose two rows live in the *same*
      sheet. That is a ``row_duplication`` concern, not a cross-table one.
    * **Orphan foreign keys** - ``fk_orphan`` depends on the resolver
      having inferred a real join between two same-file CSVs, which is
      not reliable; a child key with "no parent" is then unverifiable.

    Returns ``(new_violations, new_verifications, stats)``.
    """
    by_vid: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    order: List[Any] = []
    for r in violation_dicts:
        vid = r.get("violation_id")
        if vid not in by_vid:
            order.append(vid)
        by_vid[vid].append(r)

    kept: List[Dict[str, Any]] = []
    drop_vids: set = set()
    dropped_same_sheet = 0
    dropped_fk_orphan = 0
    for vid in order:
        recs = by_vid[vid]
        kind = recs[0].get("kind") if recs else None
        if drop_fk_orphan and kind == "fk_orphan":
            drop_vids.add(vid)
            dropped_fk_orphan += 1
            continue
        if drop_same_sheet_pairs and kind in _PAIRING_KINDS:
            sheets: set = set()
            for r in recs:
                sheets.add(r.get("sheet"))
                for link in r.get("linked") or []:
                    sheets.add(link.get("sheet"))
            sheets.discard(None)
            if len(sheets) <= 1:
                drop_vids.add(vid)
                dropped_same_sheet += 1
                continue
        kept.extend(dict(r) for r in recs)

    new_verifications = [
        dict(v) for v in verification_dicts
        if v.get("violation_id") not in drop_vids
    ]
    stats = {
        "dropped_same_sheet": dropped_same_sheet,
        "dropped_fk_orphan": dropped_fk_orphan,
        "findings_before": len(order),
        "findings_after": len(order) - len(drop_vids),
    }
    return kept, new_verifications, stats


def guard_finding_dicts(
    *,
    violation_dicts: Sequence[Mapping[str, Any]],
    verification_dicts: Sequence[Mapping[str, Any]],
    group_lookup: Mapping[Tuple[str, str], Mapping[str, Any]],
    load_frames: Callable[[str, Sequence[str]], Dict[str, pd.DataFrame]],
    density_threshold: float = 0.4,
    min_findings: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Apply the guard to violation/verification dict lists.

    Returns ``(new_violations, new_verifications, stats)``. ``group_lookup``
    maps ``(workbook, group_id)`` to that group's resolver dict;
    ``load_frames(workbook, sheets)`` returns the sheet DataFrames.
    """
    by_group_vid: Dict[Tuple[str, str], Dict[str, List[Mapping[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for r in violation_dicts:
        key = (r.get("workbook", ""), r.get("group_id", ""))
        by_group_vid[key][r.get("violation_id")].append(r)

    kept: List[Dict[str, Any]] = []
    suppressed_vids: set = set()
    collapsed: List[Dict[str, Any]] = []

    for (wb, gid), vid_map in by_group_vid.items():
        gdict = group_lookup.get((wb, gid))
        pairing_vids = [
            vid for vid, recs in vid_map.items()
            if recs and recs[0].get("kind") in _PAIRING_KINDS
        ]
        if gdict is None or not pairing_vids:
            for recs in vid_map.values():
                kept.extend(dict(r) for r in recs)
            continue
        try:
            rg = _resolved_group_from_dict({**gdict, "workbook": wb})
            frames = load_frames(wb, list(rg.sheets))
        except Exception as exc:  # noqa: BLE001
            logger.warning("guard: could not evaluate %s/%s: %s", wb, gid, exc)
            for recs in vid_map.values():
                kept.extend(dict(r) for r in recs)
            continue

        dec = evaluate_group(
            rg, frames, len(pairing_vids),
            density_threshold=density_threshold, min_findings=min_findings,
        )
        if not dec.collapse:
            for recs in vid_map.values():
                kept.extend(dict(r) for r in recs)
            continue

        kinds_suppressed: Dict[str, int] = defaultdict(int)
        for vid, recs in vid_map.items():
            kind = recs[0].get("kind") if recs else None
            if kind in _PAIRING_KINDS:
                suppressed_vids.add(vid)
                kinds_suppressed[kind] += 1
            else:
                kept.extend(dict(r) for r in recs)

        collapsed.append({
            "workbook": wb, "group_id": gid, "group_kind": rg.kind,
            "sheets": list(rg.sheets),
            "suppressed": sum(kinds_suppressed.values()),
            "suppressed_by_kind": dict(kinds_suppressed),
            "density": round(dec.density, 3), "reason": dec.reason,
        })

    # A guard-collapsed group is NOT an error: the data is fine and the
    # discovered relation (union / join / key) is wrong. So we drop the
    # whole flood and emit NO labelable finding - only a count + record
    # of the rejected relations in stats["guard"].
    new_violations = kept
    new_verifications = [
        dict(v) for v in verification_dicts
        if v.get("violation_id") not in suppressed_vids
    ]

    stats = {
        "groups_collapsed": len(collapsed),
        "findings_suppressed": len(suppressed_vids),
        "notes_added": 0,
        "collapsed_groups": collapsed,
        "density_threshold": density_threshold,
        "min_findings": min_findings,
    }
    return new_violations, new_verifications, stats


# ---------------------------------------------------------------------------
# Post-processor: apply the guard to an existing run's artifacts
# ---------------------------------------------------------------------------


def _build_run_frame_loader(run_dir: Path) -> Callable[
    [str, Sequence[str]], Dict[str, pd.DataFrame]
]:
    """Build a ``load_frames(workbook, sheets)`` from a run's
    ``summary.json::config.csv_paths`` (``<workbook>/<sheet>.csv``)."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    paths = summary.get("config", {}).get("csv_paths") or []
    by_wb_sheet: Dict[Tuple[str, str], str] = {}
    for p in paths:
        pp = Path(p)
        by_wb_sheet[(pp.parent.name, pp.name)] = str(pp)
    cache: Dict[str, pd.DataFrame] = {}

    def _load(workbook: str, sheets: Sequence[str]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for s in sheets:
            path = by_wb_sheet.get((workbook, s))
            if path is None:
                continue
            df = cache.get(path)
            if df is None:
                try:
                    df = pd.read_csv(path, dtype=str, keep_default_na=False)
                except Exception:  # noqa: BLE001
                    continue
                cache[path] = df
            out[s] = df
        return out

    return _load


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def apply_guard_to_run(
    run_dir: Path | str,
    *,
    density_threshold: float = 0.4,
    min_findings: int = 10,
    backup: bool = True,
) -> Dict[str, Any]:
    """Apply the guard in place to an existing run's multi-table artifacts.

    Backs up the original ``multi_table_violations.jsonl`` /
    ``multi_table_verifications.jsonl`` under ``_pre_guard_backup/`` and
    rewrites them with the flood findings dropped. Returns a stats dict
    with before/after finding counts.
    """
    run_dir = Path(run_dir)
    vpath = run_dir / "multi_table_violations.jsonl"
    verpath = run_dir / "multi_table_verifications.jsonl"
    gpath = run_dir / "multi_table_groups.jsonl"
    if not vpath.exists():
        return {"error": "no multi_table_violations.jsonl"}

    violations = _read_jsonl(vpath)
    verifications = _read_jsonl(verpath)
    group_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for w in _read_jsonl(gpath):
        wb = w.get("workbook", "")
        for g in w.get("groups") or []:
            group_lookup[(wb, g.get("group_id", ""))] = g

    load_frames = _build_run_frame_loader(run_dir)
    before = len({r.get("violation_id") for r in violations})

    new_v, new_ver, stats = guard_finding_dicts(
        violation_dicts=violations, verification_dicts=verifications,
        group_lookup=group_lookup, load_frames=load_frames,
        density_threshold=density_threshold, min_findings=min_findings,
    )

    if backup:
        bdir = run_dir / "_pre_guard_backup"
        bdir.mkdir(exist_ok=True)
        if not (bdir / "multi_table_violations.jsonl").exists():
            (bdir / "multi_table_violations.jsonl").write_text(
                vpath.read_text(encoding="utf-8"), encoding="utf-8",
            )
            if verpath.exists():
                (bdir / "multi_table_verifications.jsonl").write_text(
                    verpath.read_text(encoding="utf-8"), encoding="utf-8",
                )

    vpath.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_v),
        encoding="utf-8",
    )
    verpath.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_ver),
        encoding="utf-8",
    )
    stats["findings_before"] = before
    stats["findings_after"] = len({r.get("violation_id") for r in new_v})
    return stats
