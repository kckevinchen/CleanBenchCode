"""One-off helper: count cells covered by TP/unsure column-cluster verdicts.

Used to estimate LLM call volume before running the full benchmark
export.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RUN = "outputs/20260513T014840Z_gpt5-family"


def main() -> None:
    labels_path = f"{RUN}/labels.jsonl"
    drafts_path = f"{RUN}/column_cluster_drafts.jsonl"

    latest: dict[str, dict] = {}
    with open(labels_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("target_id")
            if tid and tid.startswith("colclust"):
                latest[tid] = rec

    by_status: dict[str, list[str]] = defaultdict(list)
    for tid, rec in latest.items():
        by_status[rec.get("status", "?")].append(tid)
    print(
        f"latest col-clust labels: tp={len(by_status['tp'])} "
        f"unsure={len(by_status['unsure'])} fp={len(by_status['fp'])}"
    )

    def parse_tid(tid: str):
        parts = tid.split("\x00")
        if len(parts) != 4:
            return None
        return parts[1], parts[2], parts[3]

    draft_by_sc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(drafts_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            s = doc.get("source")
            c = doc.get("column")
            if not s or not c:
                continue
            draft_by_sc[(s, c)].append(doc)

    interesting = by_status["tp"] + by_status["unsure"]
    total_clusters_with_values = 0
    unique_values: set[tuple[str, str, str]] = set()
    for tid in interesting:
        p = parse_tid(tid)
        if not p:
            continue
        s, c, local = p
        docs = draft_by_sc.get((s, c), [])
        if not docs:
            continue
        doc = docs[-1]
        drafts = doc.get("draft_clusters") or []
        try:
            idx = int(local.lstrip("c")) - 1
        except Exception:  # noqa: BLE001
            continue
        if not (0 <= idx < len(drafts)):
            continue
        cl = drafts[idx]
        vals: list[str] = []
        for v in cl.get("values") or []:
            if isinstance(v, dict):
                vv = v.get("value")
            else:
                vv = v
            if isinstance(vv, str) and vv:
                vals.append(vv)
        if len(vals) >= 2:
            total_clusters_with_values += 1
            for vv in vals:
                unique_values.add((s, c, vv))
    print(
        f"clusters resolved with >=2 values: {total_clusters_with_values}"
    )
    print(
        f"unique (source,column,value) tuples involved: "
        f"{len(unique_values)}"
    )

    summary = json.load(open(f"{RUN}/summary.json"))
    paths = summary["config"]["csv_paths"]
    src_to_path = {Path(p).name: p for p in paths}
    cells = 0
    by_src: Counter[str] = Counter()
    for (s, c, v) in unique_values:
        p = src_to_path.get(s)
        if not p or not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception:  # noqa: BLE001
            continue
        if c not in df.columns:
            continue
        cnt = int((df[c].astype(str) == v).sum())
        cells += cnt
        by_src[s] += cnt
    print(f"estimated cells (sum of CSV row hits across unique values): {cells}")
    print("top sources by cell count:")
    for s, n in by_src.most_common(10):
        print(f"  {n:5d}  {s}")


if __name__ == "__main__":
    main()
