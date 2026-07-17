"""Reconcile a run's auto-applied TRIVIAL labels with the STRICT
whitespace rule, and bulk-mark pure-casing clusters FP.

Background: the in-page trivial tagger originally tagged a broad set of
"cosmetic" anomalies (comma-spacing ``Madi,Clarkson``, space-before-
bracket, apostrophe-S casing ``REED'S``, ``=`` spacing) in addition to
genuine whitespace. Those reformats are presentation conventions in
tabular data, not verifiable defects, so the rule was narrowed to
whitespace-only (edge / 2+ internal spaces + non-printable whitespace).
This script reconciles a run that was tagged under the OLD broad rule:

  1. RETRACT every auto-created TRIVIAL label (note == the tagger's
     create-note) whose cell no longer matches the strict rule
     (the comma / bracket / 'S / '=' cases) -> tombstone (status
     ``unlabeled``).
  2. RE-FIX an auto-TRIVIAL label that still matches the strict rule but
     whose ``corrected_value`` was over-corrected by the old normaliser
     (e.g. a cell with both a double space and a comma) -> re-append with
     the strict normalised value.
  3. FP every PURE-casing cluster (all flavoured detections are
     ``typo:casing``) that has no human verdict yet -> casing in a
     tabular column is a convention, not a defect.

Idempotent and append-only. Use ``--dry-run`` (default) to preview;
``--apply`` to write. Stop the workbench before ``--apply`` so there is a
single writer, then restart it.

    python -m experiments.reconcile_trivial_labels \\
        --run-id RUN_ID [--apply] [--annotator reviewer]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.labeling._helpers import _scalar
from src.labeling.app import RunContext
from src.labeling.routers.propagation import (
    _TRIVIAL_CREATE_NOTE,
    _has_whitespace_anomaly,
    _normalize_trivial,
)
from src.labeling.schema import Label, LabelStatus, LabelTarget

_FLAV_RE = re.compile(r"typo:(spelling|whitespace|punctuation|casing|spacing)", re.I)
_RETRACT_NOTE = "(auto: retracted - not a whitespace anomaly under strict rule)"
_CASING_FP_NOTE = "(auto: casing is a column/presentation convention, not a typo)"


def _cell_value(ctx: RunContext, source: str, row: Any, column: str) -> Optional[str]:
    """Read a cell exactly as the tagger did, or None if unreadable."""
    try:
        frame = ctx.load_frame(source)
    except (FileNotFoundError, KeyError):
        return None
    if column not in frame.columns:
        return None
    try:
        r = int(row)
    except (TypeError, ValueError):
        return None
    if not (0 <= r < len(frame)):
        return None
    return _scalar(frame.iat[r, frame.columns.get_loc(column)])


def _pure_casing(cluster: Dict[str, Any]) -> bool:
    flavours = set()
    for d in cluster.get("detections") or []:
        m = _FLAV_RE.search(d.get("explanation", "") or "")
        if m:
            flavours.add(m.group(1).lower())
    return flavours == {"casing"}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--annotator", default="auto-reconcile")
    p.add_argument("--apply", action="store_true",
                   help="Write the changes (default is a dry-run preview).")
    p.add_argument("--project-root",
                   default=str(Path(__file__).resolve().parents[1]))
    args = p.parse_args(argv)

    run_dir = Path(args.project_root).resolve() / "outputs" / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 2
    ctx = RunContext(run_dir)
    latest = ctx.store.latest_by_target()
    ann = args.annotator

    batch: List[Label] = []
    retracted: List[str] = []
    refixed: List[str] = []
    cluster_by_id = {c.get("cluster_id"): c for c in ctx.clusters}

    # --- 1 & 2: reconcile auto-created TRIVIAL labels to the strict rule.
    for tid, lab in latest.items():
        if (lab.note or "") != _TRIVIAL_CREATE_NOTE:
            continue
        if lab.status not in (LabelStatus.TP, LabelStatus.FN, LabelStatus.PARTIAL):
            continue
        val = _cell_value(ctx, lab.source or "", lab.row, lab.column or "")
        if val is None:
            continue
        if not _has_whitespace_anomaly(val):
            # Cosmetic-only (comma / bracket / 'S / '=') -> retract.
            batch.append(Label(
                target=lab.target, target_id=lab.target_id, source=lab.source,
                row=lab.row, column=lab.column,
                columns=list(lab.columns) if lab.columns else [],
                status=LabelStatus.UNLABELED, note=_RETRACT_NOTE, annotator=ann,
            ))
            retracted.append(f"{lab.source} r{lab.row} {lab.column}={val!r}")
        else:
            new_fix = _normalize_trivial(val)
            if new_fix != (lab.corrected_value or ""):
                batch.append(Label(
                    target=lab.target, target_id=lab.target_id, source=lab.source,
                    row=lab.row, column=lab.column,
                    columns=list(lab.columns) if lab.columns else [],
                    status=lab.status, corrected_value=new_fix,
                    note=_TRIVIAL_CREATE_NOTE, annotator=ann,
                ))
                refixed.append(f"{lab.source} r{lab.row} {lab.column}: "
                               f"{lab.corrected_value!r} -> {new_fix!r}")

    # --- 3: FP pure-casing clusters with no existing verdict.
    casing_fp: List[str] = []
    casing_skipped_labelled = 0
    casing_total = 0
    for cid, c in cluster_by_id.items():
        if not isinstance(cid, str) or not _pure_casing(c):
            continue
        casing_total += 1
        if latest.get(cid) is not None:
            casing_skipped_labelled += 1
            continue
        cols = [x for x in (c.get("columns") or []) if isinstance(x, str)]
        r = c.get("row")
        batch.append(Label(
            target=LabelTarget.CLUSTER, target_id=cid, source=c.get("source"),
            row=int(r) if isinstance(r, int) else None,
            column=cols[0] if cols else None, columns=cols,
            status=LabelStatus.FP, note=_CASING_FP_NOTE, annotator=ann,
        ))
        casing_fp.append(cid)

    print(f"Run: {args.run_id}")
    print(f"  auto-TRIVIAL labels retracted (cosmetic-only): {len(retracted)}")
    print(f"  auto-TRIVIAL labels re-fixed (strip over-correction): {len(refixed)}")
    print(f"  pure-casing clusters: {casing_total}  "
          f"-> FP: {len(casing_fp)}  (skipped, already labelled: {casing_skipped_labelled})")
    print(f"  total records to append: {len(batch)}")
    for label, rows in (("retract", retracted), ("re-fix", refixed)):
        for s in rows[:8]:
            print(f"    [{label}] {s}")
        if len(rows) > 8:
            print(f"    [{label}] ... and {len(rows) - 8} more")

    if not args.apply:
        print("\n[dry-run] no records written. Re-run with --apply to write.")
        return 0
    if batch:
        ctx.store.append_many(batch)
    print(f"\nApplied: appended {len(batch)} records to {run_dir / 'labels.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
