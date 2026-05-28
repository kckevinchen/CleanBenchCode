"""Bulk-apply a verdict to every column-cluster draft of a given kind.

Use case: the operator wants to triage all "casing"-kind column-cluster
drafts to ``unsure`` so they can be revisited later. Doing this through
the labelling UI is one click per cluster across hundreds of columns;
this script does it in one batch write.

The script targets the **drafts** in ``column_cluster_drafts.jsonl`` --
not saved ``column_clusters.jsonl`` records (this run typically has no
saved records yet). Per the labelling app's promotion rules, a verdict
on a draft is valid as long as the draft has >= 2 string values; the
local ``cluster_id`` follows the ``c{1..N}`` order of the draft list,
matching the convention enforced by ``list_column_clusters`` and
``post_label`` in ``src/labeling/app.py``.

Idempotent: clusters that already have a label of any status are
skipped. Use ``--force`` to overwrite existing labels with the new
verdict (audit trail is preserved by the append-only store).

Usage:

    python -m scripts.bulk_label_column_cluster_drafts \\
        --run outputs/20260513T234733Z_gpt5-family \\
        --kind casing --status unsure --dry-run

    python -m scripts.bulk_label_column_cluster_drafts \\
        --run outputs/20260513T234733Z_gpt5-family \\
        --kind casing --status unsure
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow ``python scripts/...`` invocation as well as ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labeling.schema import (  # noqa: E402
    Label,
    LabelStatus,
    LabelTarget,
    column_cluster_target_id,
)
from src.labeling.store import LabelStore  # noqa: E402


def _latest_drafts_by_column(
    drafts_path: Path,
) -> "OrderedDict[Tuple[str, str], List[Dict[str, Any]]]":
    """Latest-wins per (source, column); preserves insertion order so
    the output report reflects the on-disk order of columns."""
    out: "OrderedDict[Tuple[str, str], List[Dict[str, Any]]]" = OrderedDict()
    with drafts_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            source = rec.get("source")
            column = rec.get("column")
            if not isinstance(source, str) or not isinstance(column, str):
                continue
            out[(source, column)] = rec.get("draft_clusters") or []
    return out


def _normalise_values(raw: Any) -> List[str]:
    """Mirror the labelling app's flexible draft value-extraction:
    each draft entry's ``values`` is usually a list of strings, but
    historically also accepted ``[{"value": "...", ...}, ...]``."""
    out: List[str] = []
    if not isinstance(raw, list):
        return out
    for v in raw:
        if isinstance(v, str) and v:
            out.append(v)
        elif isinstance(v, dict) and isinstance(v.get("value"), str) and v["value"]:
            out.append(v["value"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run", required=True, type=Path,
                   help="Path to the run output directory.")
    p.add_argument("--kind", default="casing",
                   help="ClusterKind tag to match on draft entries (default: casing).")
    p.add_argument("--status", default="unsure",
                   help="LabelStatus value to apply (default: unsure).")
    p.add_argument("--annotator", default="bulk-script",
                   help="Annotator name on the written labels.")
    p.add_argument("--note", default=None,
                   help="Override the auto-generated note.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without touching labels.jsonl.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing labels (otherwise skip).")
    args = p.parse_args()

    drafts_path = args.run / "column_cluster_drafts.jsonl"
    if not drafts_path.exists():
        print(f"error: no drafts file at {drafts_path}", file=sys.stderr)
        return 2

    try:
        status = LabelStatus(args.status.lower())
    except ValueError:
        print(
            f"error: invalid status {args.status!r}; "
            f"expected one of {[s.value for s in LabelStatus]}",
            file=sys.stderr,
        )
        return 2

    note = args.note or f"auto: bulk {args.kind} -> {status.value}"

    drafts_by_column = _latest_drafts_by_column(drafts_path)
    labels_path = args.run / "labels.jsonl"
    store = LabelStore(labels_path)
    existing = store.latest_by_target()

    to_write: List[Label] = []
    skipped_existing: List[Tuple[str, str, str, str]] = []
    skipped_too_few: List[Tuple[str, str, str]] = []
    matched = 0
    columns_with_match = 0

    for (source, column), drafts in drafts_by_column.items():
        col_had_match = False
        for idx, dr in enumerate(drafts, start=1):
            if not isinstance(dr, dict):
                continue
            if (dr.get("kind") or "").strip().lower() != args.kind.lower():
                continue
            matched += 1
            col_had_match = True
            local_id = f"c{idx}"
            tid = column_cluster_target_id(source, column, local_id)
            values = _normalise_values(dr.get("values"))
            if len(values) < 2:
                # Mirrors the labelling app's >=2-string-values rule
                # for drafts to be a valid label target.
                skipped_too_few.append((source, column, local_id))
                continue
            prior = existing.get(tid)
            if prior is not None and not args.force:
                skipped_existing.append(
                    (source, column, local_id, prior.status.value),
                )
                continue
            to_write.append(Label(
                target=LabelTarget.COLUMN_CLUSTER,
                target_id=tid,
                source=source,
                row=None,
                column=column,
                columns=[column],
                status=status,
                note=note,
                annotator=args.annotator,
            ))
        if col_had_match:
            columns_with_match += 1

    print(f"scanned {len(drafts_by_column)} (source, column) pairs")
    print(f"matched {matched} {args.kind!r} draft cluster(s) "
          f"across {columns_with_match} column(s)")
    print(f"would write {len(to_write)} label(s) "
          f"with status={status.value} annotator={args.annotator!r}")
    if skipped_existing:
        print(f"skipping {len(skipped_existing)} cluster(s) with existing label "
              f"(use --force to overwrite); first 5:")
        for src, col, lid, st in skipped_existing[:5]:
            print(f"    {src!r} {col!r} {lid} (current status={st})")
    if skipped_too_few:
        print(f"skipping {len(skipped_too_few)} draft(s) with <2 string values "
              f"(unusable as label target); first 5:")
        for src, col, lid in skipped_too_few[:5]:
            print(f"    {src!r} {col!r} {lid}")

    if args.dry_run:
        print("DRY RUN - nothing written.")
        if to_write:
            print("First 5 labels that would be written:")
            for lab in to_write[:5]:
                print(f"    source={lab.source!r} column={lab.column!r} "
                      f"target_id={lab.target_id!r}")
        return 0

    if not to_write:
        print("nothing to write.")
        return 0

    store.append_many(to_write)
    print(f"wrote {len(to_write)} label(s) to {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
