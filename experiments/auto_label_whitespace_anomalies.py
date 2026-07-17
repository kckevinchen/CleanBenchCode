"""Auto-label whitespace-anomaly cells in a run as ``tp`` (or FN).

The labelling workbench's CSV viewer paints a red dotted underline on
every cell whose value contains:

  * leading or trailing ASCII space,
  * an internal run of 2+ ASCII spaces, or
  * any tab / newline / NBSP / zero-width / soft-hyphen / BOM
    character.

These are extremely high-precision signals that the cell value is
malformed (copy-paste mishaps, formula export glitches), but the LLM
detector pipeline frequently misses them because they are visually
invisible -- the operator can see them only because the workbench
escapes the special chars. This script lifts every such cell into a
verdict so the canonical benchmark gains coverage:

  * If a cluster in the target run covers ``(source, row, column)``
    and is currently unlabelled, the cluster receives ``status=tp``
    with ``corrected_value`` = whitespace-normalised value.
  * Otherwise, a cell-level FN label is posted (the LLM missed the
    cell entirely; the benchmark exporter renders it as an "error"
    row with empty ``cluster_ids``).

Already-labelled clusters are left alone unless ``--overwrite`` is
passed; existing cell FN labels at the same key are also respected.

Usage::

    python -m experiments.auto_label_whitespace_anomalies \\
        --run-id 20260512T213238Z_gpt5-family \\
        --annotator reviewer \\
        [--dry-run] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import requests


# Mirror EXACTLY what frontend/labeling/label.js paints with .has-ws.
# Two parts (any match -> anomaly):
#   1. /^[ ]|[ ]$|[ ]{2,}/  -- edge or internal ASCII space run.
#   2. /[\t\u00A0\n\r\u2028\u2029\u200B\u200C\u200D\u00AD\uFEFF]/
_WS_DETECT = re.compile(
    r"^ | $|  +|[\t\u00A0\n\r\u2028\u2029\u200B\u200C\u200D\u00AD\uFEFF]"
)

# Normalisation matching the detection set:
#   - drop zero-width / soft hyphen / BOM
#   - convert tab / CR / LF / line / paragraph separator to a space
#   - convert NBSP to a space
#   - collapse 2+ ASCII spaces -> single space
#   - strip edges
_ZW_DROP = re.compile(r"[\u200B\u200C\u200D\u00AD\uFEFF]")
_CONTROL_TO_SPACE = re.compile(r"[\t\n\r\u2028\u2029]")
_NBSP_TO_SPACE = re.compile(r"[\u00A0]")
_MULTI_SPACE = re.compile(r" {2,}")


def normalize_whitespace(value: str) -> str:
    s = _ZW_DROP.sub("", value)
    s = _CONTROL_TO_SPACE.sub(" ", s)
    s = _NBSP_TO_SPACE.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s)
    return s.strip()


def has_whitespace_anomaly(value: str) -> bool:
    if not value:
        return False
    return _WS_DETECT.search(value) is not None


def cell_target_id(source: str, row: int, column: str) -> str:
    """Match :func:`src.labeling.schema.cell_target_id`. The server
    rebuilds this anyway but sending a sensible value keeps the wire
    payload self-describing."""
    return f"cell\x00{source}\x00{int(row)}\x00{column}"


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
) -> Tuple[Dict[str, dict], Dict[Tuple[str, int, str], List[str]]]:
    by_id: Dict[str, dict] = {}
    by_loc: Dict[Tuple[str, int, str], List[str]] = {}
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
            by_loc.setdefault((src, int(row), col), []).append(cid)
    return by_id, by_loc


def fetch_run(base: str, run_id: str) -> dict:
    r = requests.get(f"{base}/api/runs/{run_id}", timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_source_rows(base: str, run_id: str, name: str) -> dict:
    r = requests.get(
        f"{base}/api/runs/{run_id}/sources/{name}",
        params={"start": 0, "limit": 5000},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def fetch_existing_labels(base: str, run_id: str) -> Dict[str, dict]:
    r = requests.get(f"{base}/api/runs/{run_id}/labels", timeout=30)
    r.raise_for_status()
    data = r.json()
    return dict(data.get("latest") or {})


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--annotator", default="reviewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Workbench project root (used to locate the run's clusters.jsonl).",
    )
    p.add_argument(
        "--note",
        default="(auto: whitespace anomaly)",
        help="Note attached to every label written.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="POST even when the target cluster/cell already has a label.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and a sample but do not POST.",
    )
    p.add_argument(
        "--cluster-only",
        action="store_true",
        help="Only label cells that fall inside an existing cluster "
             "(skip cell-level FN posting for un-clustered cells).",
    )
    p.add_argument(
        "--fn-only",
        action="store_true",
        help="Only post cell-level FN labels (skip cluster TP upgrades).",
    )
    args = p.parse_args(argv)

    if args.cluster_only and args.fn_only:
        print("ERROR: --cluster-only and --fn-only are mutually exclusive",
              file=sys.stderr)
        return 2

    project_root = Path(args.project_root).resolve()
    run_dir = project_root / "outputs" / args.run_id
    cl_path = run_dir / "clusters.jsonl"
    if not cl_path.exists():
        print(f"ERROR: clusters.jsonl not found: {cl_path}", file=sys.stderr)
        return 2

    base = f"http://{args.host}:{args.port}"

    try:
        run_info = fetch_run(base, args.run_id)
    except requests.RequestException as exc:
        print(f"ERROR: cannot reach workbench at {base}: {exc}", file=sys.stderr)
        return 2
    sources = [s.get("name") for s in (run_info.get("sources") or []) if s.get("name")]
    print(f"Sources: {len(sources)}")

    existing = fetch_existing_labels(base, args.run_id)
    print(f"Existing labels: {len(existing)}")

    by_id, by_loc = _load_run_clusters(cl_path)
    print(f"Indexed {len(by_id)} clusters from {cl_path.name}")

    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})
    url = f"{base}/api/runs/{args.run_id}/labels"

    cluster_targets: List[dict] = []
    cell_targets: List[dict] = []
    skipped_existing_cluster = 0
    skipped_existing_cell = 0
    skipped_dupe_cluster: set = set()
    total_anomalies = 0
    scan_errors: List[str] = []

    for name in sources:
        try:
            data = fetch_source_rows(base, args.run_id, name)
        except requests.RequestException as exc:
            scan_errors.append(f"{name}: {exc}")
            continue
        columns = list(data.get("columns") or [])
        for r in data.get("rows") or []:
            row = r.get("row")
            if row is None:
                continue
            values = r.get("values") or {}
            for col in columns:
                v = values.get(col)
                if v is None:
                    continue
                if not isinstance(v, str):
                    # Numeric / bool cells -> coerce; pandas keeps NaN
                    # as None already by way of the API, so anything
                    # non-string here is a stringified scalar that
                    # can't carry whitespace anomalies.
                    continue
                if not has_whitespace_anomaly(v):
                    continue
                total_anomalies += 1
                fixed = normalize_whitespace(v)
                cids = by_loc.get((name, int(row), col), [])

                if cids and not args.fn_only:
                    # Pick the first cluster as the verdict target.
                    cid = cids[0]
                    if cid in skipped_dupe_cluster:
                        continue
                    skipped_dupe_cluster.add(cid)
                    if cid in existing and not args.overwrite:
                        skipped_existing_cluster += 1
                        continue
                    cluster = by_id[cid]
                    cols = list(cluster.get("columns") or [])
                    cluster_row = cluster.get("row")
                    cluster_targets.append({
                        "target": "cluster",
                        "target_id": cid,
                        "source": cluster.get("source") or name,
                        "row": int(cluster_row) if isinstance(cluster_row, int) else int(row),
                        "column": cols[0] if cols else col,
                        "columns": cols if cols else [col],
                        "status": "tp",
                        "note": args.note,
                        "corrected_value": fixed,
                        "_original_value": v,
                    })
                    continue

                # No cluster (or --fn-only): emit a cell-level FN label.
                if args.cluster_only:
                    continue
                tid = cell_target_id(name, int(row), col)
                if tid in existing and not args.overwrite:
                    skipped_existing_cell += 1
                    continue
                cell_targets.append({
                    "target": "cell",
                    "target_id": tid,
                    "source": name,
                    "row": int(row),
                    "column": col,
                    "columns": [col],
                    "status": "fn",
                    "note": args.note,
                    "corrected_value": fixed,
                    "_original_value": v,
                })

    print(
        f"Anomaly cells: {total_anomalies} "
        f"-> cluster TP queued: {len(cluster_targets)} "
        f"(skipped already-labelled: {skipped_existing_cluster}), "
        f"cell FN queued: {len(cell_targets)} "
        f"(skipped already-labelled: {skipped_existing_cell})"
    )
    if scan_errors:
        print("Scan errors:")
        for e in scan_errors[:5]:
            print(f"  - {e}")
        if len(scan_errors) > 5:
            print(f"  ... and {len(scan_errors) - 5} more")

    if args.dry_run:
        print("\n[dry-run] sample cluster targets:")
        for t in cluster_targets[:5]:
            print(f"  TP {t['target_id']!s:48s} "
                  f"{t['_original_value']!r} -> {t['corrected_value']!r}")
        print("\n[dry-run] sample cell-FN targets:")
        for t in cell_targets[:5]:
            print(f"  FN {t['source']}:{t['row']}:{t['column']:20s} "
                  f"{t['_original_value']!r} -> {t['corrected_value']!r}")
        return 0

    targets = cluster_targets + cell_targets
    if not targets:
        print("Nothing to do.")
        return 0

    ok = 0
    failed = 0
    failures: List[str] = []
    for i, t in enumerate(targets, 1):
        payload = {k: v for k, v in t.items() if not k.startswith("_")}
        payload["annotator"] = args.annotator
        try:
            r = sess.post(url, data=json.dumps(payload), timeout=60)
            if r.status_code >= 300:
                failed += 1
                failures.append(
                    f"{t['target_id']}: {r.status_code} {r.text[:160]}"
                )
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
