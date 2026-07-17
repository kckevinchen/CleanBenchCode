"""Auto-label every "abbreviation typo" cluster in a run as ``unsure``.

A typo cluster is treated as an *abbreviation typo* when:

  * its ``error_types`` list contains ``"typo"``, and
  * at least one of its ``typo`` detections has an ``explanation``
    that mentions ``abbreviation`` / ``acronym`` / ``initialism``
    / ``contraction`` (case-insensitive).

The error-type prompt (``prompts/error_types.yaml``) tells the model
to include those keywords when it flags a short value as a possible
typo even though the value is a recognisable abbreviation, so this
filter is the canonical "the only reason this was flagged is because
it looks like an abbreviation" signal.

The script writes labels through the running workbench's HTTP API
(default ``http://127.0.0.1:8000``) so the cached ``LabelStore`` stays
in sync with disk. Already-labelled clusters are left untouched.

Usage::

    python -m experiments.auto_label_abbreviation_typos \\
        --run-id 20260512T001509Z_gpt5-family \\
        --annotator reviewer \\
        [--dry-run] [--status unsure] [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests


ABBR_PATTERN = re.compile(r"abbrev|acronym|initialism|\bcontraction\b", re.I)


def iter_clusters(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def is_abbreviation_typo(cluster: dict) -> bool:
    ets = cluster.get("error_types") or []
    if "typo" not in ets:
        return False
    for det in cluster.get("detections") or []:
        if det.get("error_type") != "typo":
            continue
        expl = det.get("explanation") or ""
        if ABBR_PATTERN.search(expl):
            return True
    return False


def fetch_existing_labels(base: str, run_id: str) -> Dict[str, str]:
    r = requests.get(f"{base}/api/runs/{run_id}/labels", timeout=30)
    r.raise_for_status()
    data = r.json()
    return {tid: rec.get("status", "") for tid, rec in (data.get("latest") or {}).items()}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--annotator", default="reviewer")
    p.add_argument("--status", default="unsure",
                   choices=["unsure", "tp", "fp", "partial", "skip"])
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Workbench project root (used to locate clusters.jsonl).",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print counts but do not POST labels.")
    p.add_argument("--note",
                   default="(auto: abbreviation typo)",
                   help="Note attached to every label written.")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N labels (0 = no limit).")
    args = p.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    run_dir = project_root / "outputs" / args.run_id
    cl_path = run_dir / "clusters.jsonl"
    if not cl_path.exists():
        print(f"ERROR: clusters.jsonl not found: {cl_path}", file=sys.stderr)
        return 2

    base = f"http://{args.host}:{args.port}"

    # Confirm workbench is alive and pre-load already-labelled cluster ids.
    try:
        existing = fetch_existing_labels(base, args.run_id)
    except requests.RequestException as exc:
        print(f"ERROR: cannot reach workbench at {base}: {exc}", file=sys.stderr)
        return 2
    print(f"Loaded {len(existing)} existing labels from {base}")

    # Scan clusters and build the work list.
    total = 0
    typo = 0
    abbr = 0
    skipped_existing = 0
    targets: List[dict] = []
    seen_ids: set = set()
    for c in iter_clusters(cl_path):
        total += 1
        ets = c.get("error_types") or []
        if "typo" in ets:
            typo += 1
        if not is_abbreviation_typo(c):
            continue
        abbr += 1
        cid = c.get("cluster_id")
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        if cid in existing:
            skipped_existing += 1
            continue
        cols = c.get("columns") or []
        row = c.get("row")
        targets.append({
            "target": "cluster",
            "target_id": cid,
            "source": c.get("source", ""),
            "row": row if isinstance(row, int) else None,
            "column": cols[0] if cols else None,
            "columns": list(cols),
            "status": args.status,
            "note": args.note,
        })

    print(
        f"Scanned: total={total} typo_clusters={typo} "
        f"abbreviation_typo={abbr} already_labelled={skipped_existing} "
        f"to_label={len(targets)}"
    )

    if args.limit and len(targets) > args.limit:
        print(f"Truncating to first --limit={args.limit}")
        targets = targets[: args.limit]

    if args.dry_run:
        print("[dry-run] not posting labels. Sample of first 5 targets:")
        for t in targets[:5]:
            print(f"  - {t['target_id']} src={t['source']!r} col={t['column']!r}")
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
                failures.append(f"{t['target_id']}: {r.status_code} {r.text[:120]}")
            else:
                ok += 1
        except requests.RequestException as exc:
            failed += 1
            failures.append(f"{t['target_id']}: {exc}")
        if i % 500 == 0:
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
