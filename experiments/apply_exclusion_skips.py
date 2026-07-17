"""Apply the documented "exclusion -> skip" rule to a run.

The labelling workbench's schema and frontend both reference an
auto-skip-from-exclusion cascade: when an operator marks a CSV (or
rows / columns inside one) as excluded, every cluster that falls
within that scope is supposed to receive a ``status=skip`` cluster
label so the queue advances past it. The server-side hook
(``_auto_skip_for_exclusion`` in ``app.py``, per the frontend
comment) does not exist in the current codebase, so exclusions
added after labels are recorded leave stale verdicts in place.

This script reconciles a run in-place:

  1. Read ``<run_dir>/exclusions.jsonl`` (active records only -
     tombstoned ids dropped).
  2. Bucket scopes the same way ``_build_exclusion_lookup`` in
     ``app.py`` does: ``excluded_files``, ``per_source_rows``,
     ``per_source_cols``.
  3. For every cluster in ``clusters.jsonl``, check whether
     ``(source, row, columns)`` falls inside any active exclusion.
  4. If yes and the current verdict is not already ``skip``, POST
     a fresh ``skip`` cluster label (overwriting tp/fp/partial/
     unsure/unlabeled). ``--dry-run`` reports counts only.

Unlike :mod:`experiments.apply_benchmark_labels` /
:mod:`experiments.auto_label_abbreviation_typos`, this script
WILL overwrite prior decisions by design - the exclusion is a
later, stronger statement ("don't touch this slice") so the older
verdicts are obsolete.

Usage::

    python -m experiments.apply_exclusion_skips \\
        --run-id 20260512T213238Z_gpt5-family \\
        --annotator reviewer \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import requests


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


def fetch_exclusions(base: str, run_id: str) -> List[dict]:
    r = requests.get(f"{base}/api/runs/{run_id}/exclusions", timeout=30)
    r.raise_for_status()
    return list((r.json() or {}).get("items") or [])


def fetch_existing_labels(base: str, run_id: str) -> Dict[str, dict]:
    r = requests.get(f"{base}/api/runs/{run_id}/labels", timeout=30)
    r.raise_for_status()
    data = r.json()
    return dict(data.get("latest") or {})


def build_lookup(
    actives: List[dict],
) -> Tuple[Set[str], Dict[str, Set[int]], Dict[str, Set[str]]]:
    """Mirror :func:`src.labeling.app._build_exclusion_lookup`."""
    excluded_files: Set[str] = set()
    per_source_rows: Dict[str, Set[int]] = {}
    per_source_cols: Dict[str, Set[str]] = {}
    for e in actives:
        scope = e.get("scope")
        src = e.get("source") or ""
        if not src:
            continue
        if scope == "csv":
            excluded_files.add(src)
        elif scope == "rows":
            bucket = per_source_rows.setdefault(src, set())
            for r in e.get("rows") or []:
                try:
                    bucket.add(int(r))
                except (TypeError, ValueError):
                    continue
        elif scope == "columns":
            bucket = per_source_cols.setdefault(src, set())
            for c in e.get("columns") or []:
                if c is not None:
                    bucket.add(str(c))
    return excluded_files, per_source_rows, per_source_cols


def cluster_in_exclusion(
    cluster: dict,
    excluded_files: Set[str],
    per_source_rows: Dict[str, Set[int]],
    per_source_cols: Dict[str, Set[str]],
) -> Optional[str]:
    """Return a reason string if ``cluster`` falls within any active
    exclusion, else ``None``."""
    src = cluster.get("source") or ""
    if not src:
        return None
    if src in excluded_files:
        return "csv"
    row = cluster.get("row")
    if isinstance(row, int) and row in per_source_rows.get(src, set()):
        return f"row={row}"
    cols = cluster.get("columns") or []
    excluded_cols = per_source_cols.get(src, set())
    for col in cols:
        if col in excluded_cols:
            return f"col={col}"
    return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--annotator", default="reviewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Workbench project root (used to locate clusters.jsonl).",
    )
    p.add_argument(
        "--note",
        default="(auto: exclusion -> skip cascade)",
        help="Note attached to every skip label written.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and a sample but do not POST.",
    )
    args = p.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    run_dir = project_root / "outputs" / args.run_id
    cl_path = run_dir / "clusters.jsonl"
    if not cl_path.exists():
        print(f"ERROR: clusters.jsonl not found: {cl_path}", file=sys.stderr)
        return 2

    base = f"http://{args.host}:{args.port}"
    try:
        actives = fetch_exclusions(base, args.run_id)
    except requests.RequestException as exc:
        print(f"ERROR: cannot reach workbench at {base}: {exc}", file=sys.stderr)
        return 2

    if not actives:
        print("No active exclusions; nothing to do.")
        return 0

    excluded_files, per_source_rows, per_source_cols = build_lookup(actives)
    print(f"Active exclusions: {len(actives)} "
          f"(files={len(excluded_files)} "
          f"row-scopes={sum(len(v) for v in per_source_rows.values())} "
          f"col-scopes={sum(len(v) for v in per_source_cols.values())})")

    try:
        existing = fetch_existing_labels(base, args.run_id)
    except requests.RequestException as exc:
        print(f"ERROR: fetching labels: {exc}", file=sys.stderr)
        return 2
    print(f"Existing labels: {len(existing)}")

    skip_targets: List[dict] = []
    already_skip = 0
    examined = 0
    for c in _iter_jsonl(cl_path):
        examined += 1
        reason = cluster_in_exclusion(
            c, excluded_files, per_source_rows, per_source_cols,
        )
        if reason is None:
            continue
        cid = c.get("cluster_id")
        if not cid:
            continue
        cur = existing.get(cid) or {}
        if cur.get("status") == "skip":
            already_skip += 1
            continue
        cols = list(c.get("columns") or [])
        row = c.get("row")
        skip_targets.append({
            "target": "cluster",
            "target_id": cid,
            "source": c.get("source") or "",
            "row": int(row) if isinstance(row, int) else None,
            "column": cols[0] if cols else None,
            "columns": cols,
            "status": "skip",
            "note": f"{args.note} ({reason})",
            "_prior_status": cur.get("status") or "<unlabeled>",
        })

    print(
        f"Clusters scanned: {examined}; "
        f"in-scope: {len(skip_targets) + already_skip}; "
        f"already skip: {already_skip}; "
        f"to POST: {len(skip_targets)}"
    )

    if args.dry_run:
        print("[dry-run] sample of first 10 targets:")
        for t in skip_targets[:10]:
            print(f"  - {t['target_id']:48s}  "
                  f"prior={t['_prior_status']:10s}  reason={t['note']}")
        # show distribution of prior status to be overwritten
        from collections import Counter
        c = Counter(t["_prior_status"] for t in skip_targets)
        print(f"prior-status distribution: {dict(c)}")
        return 0

    if not skip_targets:
        print("Nothing to do.")
        return 0

    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})
    url = f"{base}/api/runs/{args.run_id}/labels"
    ok = 0
    failed = 0
    failures: List[str] = []
    for i, t in enumerate(skip_targets, 1):
        payload = {k: v for k, v in t.items() if not k.startswith("_")}
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
            print(f"  ... {i}/{len(skip_targets)} (ok={ok} failed={failed})")

    print(f"Done: posted={ok} failed={failed} of {len(skip_targets)}")
    if failures:
        print("First failures:")
        for line in failures[:10]:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
