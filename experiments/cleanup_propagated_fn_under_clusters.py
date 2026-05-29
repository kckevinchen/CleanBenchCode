"""Remove auto-propagated FN cell labels that fall inside a cluster.

``propagate-same-value`` used to FN-label same-value cells that were
already flagged by a cluster: before detection-coverage existed it FN'd a
cluster's OWN multi-row siblings, and even after that it FN'd cells
belonging to still-unlabelled *sibling* clusters. A cell any cluster
flagged is NOT a false negative (the LLM caught it), so these labels are
artifacts that pollute the benchmark export.

This one-shot walks each run, finds auto-propagated FN cell labels whose
``(source, row, column)`` lies inside any cluster's coverage (anchor row +
each detection's ``affected_rows``), and retracts them. It does not delete
history -- it appends an ``UNLABELED`` tombstone per offending cell, so
``latest_by_target`` treats the cell as un-labelled while the full audit
trail (the original FN record) stays in ``labels.jsonl``.

Genuine FN cells -- same-value cells that NO cluster flagged -- are left
untouched. Operator-authored FN cells (no ``same-value`` propagation
marker in the note) are also left untouched.

A running workbench picks the tombstones up automatically: ``LabelStore``
reloads when ``labels.jsonl`` changes on disk.

Usage::

    python -m experiments.cleanup_propagated_fn_under_clusters            # dry-run, all runs
    python -m experiments.cleanup_propagated_fn_under_clusters --apply     # write tombstones
    python -m experiments.cleanup_propagated_fn_under_clusters <run_id>    # one run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.labeling.schema import Label, LabelStatus, LabelTarget
from src.labeling.store import LabelStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CellKey = Tuple[str, int, str]  # (source, row, column)


def _cluster_coverage(clusters: List[Dict[str, Any]]) -> Set[CellKey]:
    """Every cell any cluster flagged: anchor (row, col) + each
    detection's ``affected_rows`` x columns. Mirrors the coverage set in
    ``propagation.propagate_same_value``."""
    covered: Set[CellKey] = set()
    for c in clusters:
        source = c.get("source")
        if not source:
            continue
        cols = list(c.get("columns") or [])
        r = c.get("row")
        if isinstance(r, int):
            for col in cols:
                covered.add((source, r, col))
        for det in (c.get("detections") or ()):
            det_cols = list(det.get("columns") or cols)
            for ar in (det.get("affected_rows") or ()):
                if isinstance(ar, int):
                    for col in det_cols:
                        covered.add((source, ar, col))
    return covered


def _load_clusters(run_dir: Path) -> List[Dict[str, Any]]:
    cj = run_dir / "clusters.json"
    if not cj.exists():
        return []
    try:
        data = json.loads(cj.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    return data if isinstance(data, list) else data.get("clusters", []) or []


def _is_auto_propagated_fn(lab: Label) -> bool:
    """An FN cell label written by same-value propagation (not a
    deliberate operator FN). The propagation note always carries the
    ``same-value`` marker (client: '(auto: same-value as <id>)';
    server default: '(auto: same-value as labelled TP cell)')."""
    return (
        lab.target == LabelTarget.CELL
        and lab.status == LabelStatus.FN
        and "same-value" in (lab.note or "")
    )


def process_run(run_dir: Path, *, apply: bool) -> Dict[str, Any]:
    clusters = _load_clusters(run_dir)
    labels_path = run_dir / "labels.jsonl"
    if not labels_path.exists() or not clusters:
        return {"run": run_dir.name, "offending": 0, "examples": []}

    covered = _cluster_coverage(clusters)
    store = LabelStore(labels_path)

    offending: List[Label] = []
    for lab in store.latest_by_target().values():
        if not _is_auto_propagated_fn(lab):
            continue
        if lab.row is None or not lab.column:
            continue
        if (lab.source, int(lab.row), lab.column) in covered:
            offending.append(lab)

    examples = [
        {"source": lab.source, "row": lab.row, "column": lab.column,
         "note": (lab.note or "")[:60]}
        for lab in offending[:5]
    ]

    if apply and offending:
        tombstones = [
            Label(
                target=LabelTarget.CELL,
                target_id=lab.target_id,
                source=lab.source,
                row=lab.row,
                column=lab.column,
                status=LabelStatus.UNLABELED,
                note="(cleanup: cell flagged by a cluster -> not an FN; "
                     "retracted same-value propagation artifact)",
                annotator="cleanup",
            )
            for lab in offending
        ]
        store.append_many(tombstones)

    return {"run": run_dir.name, "offending": len(offending), "examples": examples}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id", nargs="?", help="Single run id (default: all runs).")
    ap.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    ap.add_argument("--apply", action="store_true",
                    help="Write UNLABELED tombstones (default: dry-run report).")
    args = ap.parse_args(argv)

    outputs = args.outputs_dir
    if args.run_id:
        run_dirs = [outputs / args.run_id]
    else:
        run_dirs = sorted(d for d in outputs.iterdir() if d.is_dir()) if outputs.is_dir() else []

    total = 0
    affected_runs = 0
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"  (skip) run not found: {run_dir}", file=sys.stderr)
            continue
        res = process_run(run_dir, apply=args.apply)
        if res["offending"]:
            affected_runs += 1
            total += res["offending"]
            verb = "retracted" if args.apply else "would retract"
            print(f"  {res['run']:60} {verb} {res['offending']:5} auto-FN cells")
            for ex in res["examples"]:
                print(f"      - {ex['source']} row={ex['row']} col={ex['column']!r}")

    mode = "APPLIED" if args.apply else "DRY-RUN (re-run with --apply to write)"
    print(f"\n[{mode}] {total} auto-FN-under-cluster cells across {affected_runs} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
