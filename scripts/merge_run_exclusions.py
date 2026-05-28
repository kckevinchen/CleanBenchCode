"""Merge exclusion records across several runs and (optionally)
cascade the unified set into ``status=skip`` cluster verdicts.

Use case: the same dataset has been labelled across several runs,
and the operator has progressively marked CSVs / rows / columns as
excluded in a subset of those runs. We want the runs to converge on
a *shared* exclusion set so downstream benchmark exports stay in
lockstep regardless of which run is the source of truth.

Strategy
--------
The combined exclusion set is the **set-union** of every run's
active exclusions, atomised into singletons:

* csv     -> ``("csv", source)``
* rows    -> ``("rows", source, row_int)`` per row
* columns -> ``("columns", source, column_name)`` per column

The intersection interpretation makes no sense here -- runs whose
labelling pre-dates the exclusion subsystem (e.g.
``20260512T001509Z_gpt5-family``) carry zero records, which would
collapse the intersection to the empty set. Treating each missing
record as "data, not signal" and unioning is the conservative
choice: an item is excluded iff *any* operator on *any* run flagged
it.

For each run we compute ``union − atomics(run)`` and append one new
record per (source, scope) bundle so the on-disk shape stays
compact. Tombstoned (``deleted=True``) records on disk are honoured
via :meth:`ExclusionStore.active`; we never re-add an entry the
operator explicitly retracted on that run.

After the merge the script can also run the auto-skip cascade
(``--cascade-skips``) over both ``clusters.jsonl`` (cell-level) and
``column_cluster_drafts.jsonl`` (column-level inconsistency).
Mirrors :mod:`experiments.apply_exclusion_skips` but writes
directly via :class:`LabelStore` -- no workbench HTTP server
required. The cascade *overwrites* prior verdicts: an exclusion
added later is the operator's stronger statement.

Usage
-----

    python scripts/merge_run_exclusions.py \\
        --run outputs/20260513T234733Z_gpt5-family \\
        --run outputs/20260513T014840Z_gpt5-family \\
        --run outputs/20260512T213238Z_gpt5-family \\
        --run outputs/20260512T210510Z_gpt5-family \\
        --run outputs/20260512T001509Z_gpt5-family \\
        --dry-run

    # Then drop ``--dry-run`` once the report looks right. Add
    # ``--cascade-skips`` to also propagate exclusions into
    # ``status=skip`` cluster verdicts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

# Allow ``python scripts/...`` invocation as well as ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labeling.exclusions_store import ExclusionStore  # noqa: E402
from src.labeling.schema import (  # noqa: E402
    Exclusion,
    ExclusionIn,
    ExclusionScope,
    Label,
    LabelStatus,
    LabelTarget,
    column_cluster_target_id,
)
from src.labeling.store import LabelStore  # noqa: E402


# ---------------------------------------------------------------------------
# Atomic exclusion representation
# ---------------------------------------------------------------------------

# A single excluded slice. Three flavours, all keyed by source:
#   ("csv", source)
#   ("rows", source, row_int)
#   ("columns", source, column_name)
Atomic = Tuple[str, ...]


def _atomise(records: Iterable[Exclusion]) -> Set[Atomic]:
    """Turn :class:`Exclusion` records into a set of atomic tuples."""
    out: Set[Atomic] = set()
    for r in records:
        if r.scope == ExclusionScope.CSV:
            out.add(("csv", r.source))
        elif r.scope == ExclusionScope.ROWS:
            for raw in r.rows or []:
                try:
                    out.add(("rows", r.source, int(raw)))
                except (TypeError, ValueError):
                    continue
        elif r.scope == ExclusionScope.COLUMNS:
            for c in r.columns or []:
                if c is not None:
                    out.add(("columns", r.source, str(c)))
    return out


def _split_by_source_and_kind(
    atomics: Iterable[Atomic],
) -> Tuple[Set[str], Dict[str, Set[int]], Dict[str, Set[str]]]:
    """Group atomics back into (csv-files, rows-by-source, cols-by-source)."""
    csv_set: Set[str] = set()
    rows: Dict[str, Set[int]] = defaultdict(set)
    cols: Dict[str, Set[str]] = defaultdict(set)
    for a in atomics:
        if a[0] == "csv":
            csv_set.add(a[1])
        elif a[0] == "rows":
            rows[a[1]].add(int(a[2]))
        elif a[0] == "columns":
            cols[a[1]].add(str(a[2]))
    return csv_set, rows, cols


# ---------------------------------------------------------------------------
# Phase 1: load, atomise, union
# ---------------------------------------------------------------------------


def _load_run_atomics(
    run_dir: Path,
) -> Tuple[ExclusionStore, Set[Atomic]]:
    store = ExclusionStore(run_dir / "exclusions.jsonl")
    return store, _atomise(store.active())


def _build_union(
    per_run: Dict[Path, Set[Atomic]],
) -> Set[Atomic]:
    union: Set[Atomic] = set()
    for atomics in per_run.values():
        union.update(atomics)
    return union


# ---------------------------------------------------------------------------
# Phase 2: per-run merge writes
# ---------------------------------------------------------------------------


def _missing_grouped(
    have: Set[Atomic], target: Set[Atomic],
) -> Tuple[Set[str], Dict[str, Set[int]], Dict[str, Set[str]]]:
    return _split_by_source_and_kind(target - have)


def _write_missing(
    store: ExclusionStore,
    missing_csv: Set[str],
    missing_rows: Dict[str, Set[int]],
    missing_cols: Dict[str, Set[str]],
    *,
    annotator: str,
    note: str,
    dry_run: bool,
) -> int:
    """Append one record per (source, scope) bundle for the missing
    atomics. Returns the number of records that would be written."""
    n = 0
    for src in sorted(missing_csv):
        n += 1
        if dry_run:
            continue
        store.add(
            ExclusionIn(
                scope=ExclusionScope.CSV,
                source=src,
                note=note,
                annotator=annotator,
            ),
            annotator=annotator,
        )
    for src in sorted(missing_rows):
        rows = sorted(missing_rows[src])
        if not rows:
            continue
        n += 1
        if dry_run:
            continue
        store.add(
            ExclusionIn(
                scope=ExclusionScope.ROWS,
                source=src,
                rows=rows,
                note=note,
                annotator=annotator,
            ),
            annotator=annotator,
        )
    for src in sorted(missing_cols):
        cols = sorted(missing_cols[src])
        if not cols:
            continue
        n += 1
        if dry_run:
            continue
        store.add(
            ExclusionIn(
                scope=ExclusionScope.COLUMNS,
                source=src,
                columns=cols,
                note=note,
                annotator=annotator,
            ),
            annotator=annotator,
        )
    return n


# ---------------------------------------------------------------------------
# Phase 3: skip cascade
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _normalise_values(raw) -> List[str]:
    out: List[str] = []
    if not isinstance(raw, list):
        return out
    for v in raw:
        if isinstance(v, str) and v:
            out.append(v)
        elif isinstance(v, dict) and isinstance(v.get("value"), str) and v["value"]:
            out.append(v["value"])
    return out


def _cluster_in_scope(
    cluster: dict,
    excluded_files: Set[str],
    rows_by_src: Dict[str, Set[int]],
    cols_by_src: Dict[str, Set[str]],
) -> Optional[str]:
    src = cluster.get("source") or ""
    if not src:
        return None
    if src in excluded_files:
        return "csv"
    row = cluster.get("row")
    if isinstance(row, int) and row in rows_by_src.get(src, set()):
        return f"row={row}"
    excluded_cols = cols_by_src.get(src, set())
    for col in cluster.get("columns") or []:
        if col in excluded_cols:
            return f"col={col}"
    return None


def _cascade_skips(
    run_dir: Path,
    *,
    atomics: Set[Atomic],
    annotator: str,
    note_prefix: str,
    dry_run: bool,
) -> Dict[str, int]:
    """Write ``status=skip`` for every cell-cluster and column-cluster
    that falls within ``atomics`` (the union-merged exclusion set).
    Returns a small stats dict. Pass the post-merge atomics so the
    dry-run preview matches what the real run would do, even for
    runs that had zero exclusions on disk to start with.
    """
    excluded_files, rows_by_src, cols_by_src = _split_by_source_and_kind(atomics)

    if not (excluded_files or rows_by_src or cols_by_src):
        return {"clusters_in_scope": 0, "clusters_to_write": 0,
                "colclusters_in_scope": 0, "colclusters_to_write": 0,
                "actives": 0}

    label_store = LabelStore(run_dir / "labels.jsonl")
    existing = label_store.latest_by_target()

    to_write: List[Label] = []

    # Cell-level clusters.
    cl_path = run_dir / "clusters.jsonl"
    cluster_in_scope = 0
    for c in _iter_jsonl(cl_path):
        reason = _cluster_in_scope(c, excluded_files, rows_by_src, cols_by_src)
        if reason is None:
            continue
        cluster_in_scope += 1
        cid = c.get("cluster_id")
        if not cid:
            continue
        prior = existing.get(cid)
        if prior is not None and prior.status == LabelStatus.SKIP:
            continue
        cols = list(c.get("columns") or [])
        row = c.get("row")
        to_write.append(Label(
            target=LabelTarget.CLUSTER,
            target_id=cid,
            source=c.get("source") or "",
            row=int(row) if isinstance(row, int) else None,
            column=cols[0] if cols else None,
            columns=cols,
            status=LabelStatus.SKIP,
            note=f"{note_prefix} ({reason})",
            annotator=annotator,
        ))

    # Column-level inconsistency clusters (drafts seed the local ids).
    cc_drafts = run_dir / "column_cluster_drafts.jsonl"
    colclust_in_scope = 0
    if cc_drafts.exists():
        for rec in _iter_jsonl(cc_drafts):
            src = rec.get("source")
            col = rec.get("column")
            if not isinstance(src, str) or not isinstance(col, str):
                continue
            # Column-clusters live at the (source, column) granularity;
            # rows-scope exclusions don't apply.
            in_csv = src in excluded_files
            in_col = col in cols_by_src.get(src, set())
            if not (in_csv or in_col):
                continue
            reason = "csv" if in_csv else f"col={col}"
            # Replicate the workbench's c1..cN numbering: only drafts
            # with >= 2 string values are surfaced as label targets.
            local_idx = 0
            for dc in rec.get("draft_clusters") or []:
                if not isinstance(dc, dict):
                    continue
                values = _normalise_values(dc.get("values"))
                if len(values) < 2:
                    continue
                local_idx += 1
                local_id = f"c{local_idx}"
                tid = column_cluster_target_id(src, col, local_id)
                colclust_in_scope += 1
                prior = existing.get(tid)
                if prior is not None and prior.status == LabelStatus.SKIP:
                    continue
                to_write.append(Label(
                    target=LabelTarget.COLUMN_CLUSTER,
                    target_id=tid,
                    source=src,
                    row=None,
                    column=col,
                    columns=[col],
                    status=LabelStatus.SKIP,
                    note=f"{note_prefix} ({reason})",
                    annotator=annotator,
                ))

    if not dry_run and to_write:
        label_store.append_many(to_write)

    return {
        "actives": len(atomics),
        "clusters_in_scope": cluster_in_scope,
        "colclusters_in_scope": colclust_in_scope,
        "clusters_to_write": sum(
            1 for l in to_write if l.target == LabelTarget.CLUSTER
        ),
        "colclusters_to_write": sum(
            1 for l in to_write if l.target == LabelTarget.COLUMN_CLUSTER
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run", required=True, action="append", type=Path,
        help="Path to a run output directory. Repeat to add more runs.",
    )
    p.add_argument(
        "--annotator", default="bulk-merge",
        help="Annotator name written to new exclusion records and skip labels.",
    )
    p.add_argument(
        "--note",
        default="auto: union-merge across runs",
        help="Note attached to new exclusion records (cascade gets its own).",
    )
    p.add_argument(
        "--cascade-skips", action="store_true",
        help="After merging, write status=skip labels for every cluster "
             "and column-cluster that falls within the merged exclusions.",
    )
    p.add_argument(
        "--cascade-note",
        default="auto: exclusion -> skip cascade",
        help="Note prefix on skip labels written by the cascade phase.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report the merge / cascade plan without writing anything.",
    )
    args = p.parse_args(argv)

    runs: List[Path] = [Path(r).resolve() for r in args.run]
    for r in runs:
        if not r.is_dir():
            print(f"error: not a directory: {r}", file=sys.stderr)
            return 2

    # Phase 1: load + atomise.
    per_run: Dict[Path, Tuple[ExclusionStore, Set[Atomic]]] = {}
    for r in runs:
        store, atomics = _load_run_atomics(r)
        per_run[r] = (store, atomics)
        n_active = len(store.active())
        print(f"  loaded {r.name}: active_records={n_active}, "
              f"atomic_singletons={len(atomics)}")

    union = _build_union({r: at for r, (_, at) in per_run.items()})
    csv_u, rows_u, cols_u = _split_by_source_and_kind(union)
    n_csv_atoms = len(csv_u)
    n_row_atoms = sum(len(v) for v in rows_u.values())
    n_col_atoms = sum(len(v) for v in cols_u.values())
    print(
        f"\nUNION: {len(union)} atomic singletons "
        f"(csv={n_csv_atoms}, rows={n_row_atoms}, cols={n_col_atoms}) "
        f"across {len({a[1] for a in union})} sources"
    )

    # Phase 2: per-run merge writes.
    print("\n--- merge plan ---")
    total_recs = 0
    for r in runs:
        store, have = per_run[r]
        missing_csv, missing_rows, missing_cols = _missing_grouped(have, union)
        n = _write_missing(
            store, missing_csv, missing_rows, missing_cols,
            annotator=args.annotator, note=args.note, dry_run=args.dry_run,
        )
        total_recs += n
        m_csv = len(missing_csv)
        m_rows = sum(len(v) for v in missing_rows.values())
        m_cols = sum(len(v) for v in missing_cols.values())
        print(
            f"  {r.name}: "
            f"missing_atomics(csv={m_csv}, rows={m_rows}, cols={m_cols}); "
            f"new_records={n}"
        )
    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {total_recs} new exclusion record(s) total.")

    # Phase 3: cascade.
    if args.cascade_skips:
        print("\n--- cascade plan (exclusion -> skip) ---")
        for r in runs:
            stats = _cascade_skips(
                r,
                atomics=union,
                annotator=args.annotator,
                note_prefix=args.cascade_note,
                dry_run=args.dry_run,
            )
            print(
                f"  {r.name}: actives={stats['actives']} "
                f"clusters(in_scope={stats['clusters_in_scope']}, "
                f"to_skip={stats['clusters_to_write']}) "
                f"col_clusters(in_scope={stats['colclusters_in_scope']}, "
                f"to_skip={stats['colclusters_to_write']})"
            )

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
