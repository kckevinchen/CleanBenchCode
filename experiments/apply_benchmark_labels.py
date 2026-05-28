"""Apply TP/ambiguous labels from a canonical benchmark JSONL to a run.

Reads ``label_result/<benchmark>.jsonl`` (per-cell ground truth produced
by :mod:`src.labeling.export`) and writes one cluster-level label into
the target run for each row whose ``label`` is ``error`` (-> ``tp``)
or ``ambiguous`` (-> ``unsure``). ``clean`` rows are deliberately
ignored: the user wants to lift the human-curated positives/uncertain
calls onto a fresh detection pass so the new run can be re-exported
into an "enhanced" benchmark with broader coverage.

Matching strategy (per benchmark cell):
  1. ``cluster_ids`` recorded in the benchmark row -- preferred since
     it carries the original cluster_id format (``<src>__r{row}-cNNN``)
     and is exact when the new run reuses the same cluster ids.
  2. Fallback to ``(source, row, column)`` against the new run's
     cluster index (most clusters in this codebase are single-column,
     row-anchored, so this picks up cluster_id renames).

Cells that match no cluster in the target run are skipped and logged
(the new detector simply did not flag them; preserving them would
require cell-level FN labels which the user did not ask for).

Usage::

    python -m experiments.apply_benchmark_labels \\
        --run-id 20260512T213238Z_gpt5-family \\
        --benchmark label_result/benchmark_typo_20260507T215150Z_gpt5-family.canonical.jsonl \\
        --annotator kaiwen \\
        [--dry-run] [--overwrite] [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import requests


# benchmark.label -> LabelStatus enum value
_STATUS_MAP: Dict[str, str] = {
    "error": "tp",
    "ambiguous": "unsure",
}


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_run_clusters(
    cl_path: Path,
) -> Tuple[Dict[str, dict], Dict[Tuple[str, int, str], str]]:
    """Return ``(by_id, by_loc)`` indexes for the target run's clusters.

    ``by_id`` maps ``cluster_id -> cluster``. ``by_loc`` maps
    ``(source, row, column) -> cluster_id`` for primary (row, column)
    pairs. Locations seen on more than one cluster are dropped from
    ``by_loc`` (caller must fall back to cluster_id matching).
    """
    by_id: Dict[str, dict] = {}
    loc_owners: Dict[Tuple[str, int, str], List[str]] = {}
    for c in _iter_jsonl(cl_path):
        cid = c.get("cluster_id")
        if not cid:
            continue
        by_id[cid] = c
        src = c.get("source") or ""
        row = c.get("row")
        if not src or row is None:
            continue
        for col in c.get("columns") or []:
            loc_owners.setdefault((src, int(row), col), []).append(cid)
    by_loc: Dict[Tuple[str, int, str], str] = {
        k: v[0] for k, v in loc_owners.items() if len(v) == 1
    }
    return by_id, by_loc


def fetch_existing_labels(base: str, run_id: str) -> Dict[str, str]:
    r = requests.get(f"{base}/api/runs/{run_id}/labels", timeout=30)
    r.raise_for_status()
    data = r.json()
    return {tid: rec.get("status", "") for tid, rec in (data.get("latest") or {}).items()}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--benchmark",
        required=True,
        help="Path to canonical benchmark JSONL (per-cell ground truth).",
    )
    p.add_argument("--annotator", default="kaiwen")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Workbench project root (used to locate the run's clusters.jsonl).",
    )
    p.add_argument(
        "--note",
        default="(auto: lifted from canonical benchmark)",
        help="Note attached to every label written.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="POST even when the target cluster already has a label "
             "(default: skip already-labelled clusters).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and a sample but do not POST.",
    )
    p.add_argument(
        "--include-clean",
        action="store_true",
        help="Also POST clean->fp labels (off by default; user asked "
             "for TP/ambiguous only).",
    )
    args = p.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    bench_path = Path(args.benchmark)
    if not bench_path.is_absolute():
        bench_path = project_root / bench_path
    if not bench_path.exists():
        print(f"ERROR: benchmark not found: {bench_path}", file=sys.stderr)
        return 2

    run_dir = project_root / "outputs" / args.run_id
    cl_path = run_dir / "clusters.jsonl"
    if not cl_path.exists():
        print(f"ERROR: clusters.jsonl not found: {cl_path}", file=sys.stderr)
        return 2

    status_map = dict(_STATUS_MAP)
    if args.include_clean:
        status_map["clean"] = "fp"

    base = f"http://{args.host}:{args.port}"

    try:
        existing = fetch_existing_labels(base, args.run_id)
    except requests.RequestException as exc:
        print(f"ERROR: cannot reach workbench at {base}: {exc}", file=sys.stderr)
        return 2
    print(f"Loaded {len(existing)} existing labels from {base}")

    by_id, by_loc = _load_run_clusters(cl_path)
    print(f"Indexed {len(by_id)} clusters from {cl_path.name} "
          f"({len(by_loc)} unique (src,row,col) keys)")

    total = 0
    targets: List[dict] = []
    skipped_label = 0           # benchmark label not in status_map (e.g. clean)
    skipped_unmatched = 0       # no cluster in new run
    skipped_existing = 0        # cluster already labelled (no --overwrite)
    skipped_dupe = 0            # cluster already in targets via another bench cell
    unmatched_samples: List[Tuple] = []
    seen_cluster_ids: set = set()

    for rec in _iter_jsonl(bench_path):
        total += 1
        label = (rec.get("label") or "").strip().lower()
        if label not in status_map:
            skipped_label += 1
            continue
        src = rec.get("source") or ""
        row = rec.get("row")
        col = rec.get("column") or ""
        if not src or row is None:
            skipped_unmatched += 1
            continue
        row_i = int(row)

        # 1) cluster_ids from the benchmark (preferred).
        matched_cid: Optional[str] = None
        for cid in rec.get("cluster_ids") or []:
            if cid in by_id:
                matched_cid = cid
                break
        # 2) fallback: (source, row, column).
        if matched_cid is None:
            matched_cid = by_loc.get((src, row_i, col))
        if matched_cid is None:
            skipped_unmatched += 1
            if len(unmatched_samples) < 20:
                unmatched_samples.append((src, row_i, col, label,
                                          rec.get("original_value")))
            continue

        if matched_cid in seen_cluster_ids:
            skipped_dupe += 1
            continue
        seen_cluster_ids.add(matched_cid)

        if (matched_cid in existing) and not args.overwrite:
            skipped_existing += 1
            continue

        cluster = by_id[matched_cid]
        cols = list(cluster.get("columns") or [])
        cluster_row = cluster.get("row")
        targets.append({
            "target": "cluster",
            "target_id": matched_cid,
            "source": cluster.get("source") or src,
            "row": int(cluster_row) if isinstance(cluster_row, int) else row_i,
            "column": cols[0] if cols else col,
            "columns": cols if cols else [col],
            "status": status_map[label],
            "note": args.note,
            "corrected_value": rec.get("corrected_value"),
        })

    print(
        f"Benchmark rows: total={total} "
        f"skipped_label={skipped_label} (not in {sorted(status_map)}) "
        f"skipped_unmatched={skipped_unmatched} "
        f"skipped_dupe={skipped_dupe} "
        f"skipped_existing={skipped_existing} "
        f"to_post={len(targets)}"
    )
    if unmatched_samples:
        print("First unmatched benchmark cells (source,row,column,label,value):")
        for u in unmatched_samples:
            print(f"  - {u}")

    if args.dry_run:
        print("[dry-run] sample of first 5 targets:")
        for t in targets[:5]:
            print(f"  - {t['target_id']} -> {t['status']} "
                  f"(corr={t.get('corrected_value')!r})")
        return 0

    if not targets:
        print("Nothing to do.")
        return 0

    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})
    url = f"{base}/api/runs/{args.run_id}/labels"
    ok = 0
    failed = 0
    failures: List[str] = []
    for i, t in enumerate(targets, 1):
        payload = dict(t)
        payload["annotator"] = args.annotator
        try:
            r = sess.post(url, data=json.dumps(payload), timeout=60)
            if r.status_code >= 300:
                failed += 1
                failures.append(f"{t['target_id']}: {r.status_code} {r.text[:160]}")
            else:
                ok += 1
        except requests.RequestException as exc:
            failed += 1
            failures.append(f"{t['target_id']}: {exc}")
        if i % 200 == 0:
            print(f"  ... {i}/{len(targets)} (ok={ok} failed={failed})")

    print(f"Done: posted={ok} failed={failed} of {len(targets)}")
    if failures:
        print("First failures:")
        for line in failures[:10]:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
