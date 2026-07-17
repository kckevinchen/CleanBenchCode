"""SKIP redundant clusters whose cells are already captured by a
TRIVIAL true-positive cluster.

The LLM sometimes emits BOTH a broad multi-row cluster (its
``affected_rows`` span a whole messy column) AND separate per-row
clusters for the same cells. When the trivial whitespace tagger tags the
broad cluster TP+TRIVIAL, those per-row clusters are left UNLABELLED and
clutter the queue even though every cell they cover is already an error
in the benchmark.

This pass marks such a cluster ``skip`` (deferred, out of the active
queue) when EVERY (row, column) cell it covers is also covered by a
DIFFERENT cluster that carries a TP verdict with a ``TRIVIAL`` note.
Append-only and idempotent (an already-labelled cluster is left alone).

    python -m experiments.skip_redundant_clusters \\
        --run-id RUN_ID [--apply] [--annotator reviewer]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.labeling.app import RunContext
from src.labeling.schema import Label, LabelStatus, LabelTarget

_SKIP_NOTE = "(auto: redundant - cells already captured by a TRIVIAL cluster)"
Cell = Tuple[int, str]


def _cluster_cells(c: Dict[str, Any]) -> Set[Cell]:
    """Every (row, column) a cluster covers: its anchor row's columns plus
    every detection's affected_rows x detection columns."""
    cells: Set[Cell] = set()
    cols = [x for x in (c.get("columns") or []) if isinstance(x, str)]
    r = c.get("row")
    if isinstance(r, int):
        for col in cols:
            cells.add((r, col))
    for det in c.get("detections") or ():
        dcols = [x for x in (det.get("columns") or cols) if isinstance(x, str)]
        for ar in det.get("affected_rows") or ():
            if isinstance(ar, int):
                for col in dcols:
                    cells.add((ar, col))
    return cells


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--annotator", default="auto-reconcile")
    p.add_argument("--apply", action="store_true",
                   help="Write the SKIP labels (default is a dry-run preview).")
    p.add_argument("--project-root",
                   default=str(Path(__file__).resolve().parents[1]))
    args = p.parse_args(argv)

    run_dir = Path(args.project_root).resolve() / "outputs" / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 2
    ctx = RunContext(run_dir)
    latest = ctx.store.latest_by_target()

    # Cells covered by a TP + TRIVIAL cluster -> the set of cids covering
    # each (so we can require a DIFFERENT cluster covers a redundant one).
    triv_cover: Dict[Cell, Set[str]] = {}
    cells_by_cid: Dict[str, Set[Cell]] = {}
    for c in ctx.clusters:
        cid = c.get("cluster_id")
        if not isinstance(cid, str):
            continue
        cells = _cluster_cells(c)
        cells_by_cid[cid] = cells
        lab = latest.get(cid)
        if (lab is not None and lab.target == LabelTarget.CLUSTER
                and lab.status == LabelStatus.TP
                and "TRIVIAL" in (lab.note or "").upper()):
            for cell in cells:
                triv_cover.setdefault(cell, set()).add(cid)

    redundant: List[str] = []
    for c in ctx.clusters:
        cid = c.get("cluster_id")
        if not isinstance(cid, str) or latest.get(cid) is not None:
            continue  # only UNLABELLED clusters
        cells = cells_by_cid.get(cid) or set()
        if not cells:
            continue
        # Redundant iff every cell is covered by some OTHER TP+TRIVIAL cluster.
        if all(any(other != cid for other in triv_cover.get(cell, ()))
               for cell in cells):
            redundant.append(cid)

    print(f"Run: {args.run_id}")
    print(f"  clusters: {len(cells_by_cid)}")
    print(f"  redundant UNLABELLED clusters -> SKIP: {len(redundant)}")
    for cid in redundant[:8]:
        print(f"    [skip] {cid}")
    if len(redundant) > 8:
        print(f"    [skip] ... and {len(redundant) - 8} more")

    if not args.apply:
        print("\n[dry-run] no records written. Re-run with --apply to write.")
        return 0

    batch: List[Label] = []
    by_id = {c.get("cluster_id"): c for c in ctx.clusters}
    for cid in redundant:
        c = by_id.get(cid, {})
        cols = [x for x in (c.get("columns") or []) if isinstance(x, str)]
        r = c.get("row")
        batch.append(Label(
            target=LabelTarget.CLUSTER, target_id=cid, source=c.get("source"),
            row=int(r) if isinstance(r, int) else None,
            column=cols[0] if cols else None, columns=cols,
            status=LabelStatus.SKIP, note=_SKIP_NOTE, annotator=args.annotator,
        ))
    if batch:
        ctx.store.append_many(batch)
    print(f"\nApplied: appended {len(batch)} SKIP records to "
          f"{run_dir / 'labels.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
