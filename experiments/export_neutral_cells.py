"""Export the NEUTRAL cell set (skipped + excluded) for the cell benchmark.

The cell benchmark (``benchmark/merged_v2``) exports only ``error`` and
``ambiguous`` cells; every other cell is treated as clean. But the labeling
pipeline also produces **skipped** cells and **exclusion** sentinels (whole
CSV / rows / columns the reviewer set aside). For a fair "evaluate over the
entire CSV" protocol those must be *neutral* -- never scored as clean / FP.

This script reuses the canonical ``_materialize_run`` (same code path that built
the benchmark) over the same ``REGISTRY`` pools and writes one JSONL of neutral
records to ``benchmark/merged_v2/neutral_cells.jsonl``:

    {"kind": "skip",         "source": ..., "row": i, "column": name}
    {"kind": "exclude_csv",  "source": ...}
    {"kind": "exclude_rows", "source": ..., "rows": [..]}
    {"kind": "exclude_cols", "source": ..., "columns": [..]}

``score.py`` consumes this file and neutralizes any predicted cell that matches.

Run:
    python -m experiments.export_neutral_cells
    python -m experiments.export_neutral_cells --pools pbi_500 --out /tmp/neutral.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from experiments.export_merged_benchmark import (
    DATASET_OF,
    OUTPUTS,
    PROJECT_ROOT,
    REGISTRY,
    _materialize_run,
)
from src.labeling.schema import ExclusionScope


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=list(REGISTRY), choices=list(REGISTRY))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "benchmark" / "merged_v2" / "neutral_cells.jsonl"))
    args = ap.parse_args()

    skip_keys: set[tuple[str, object, str]] = set()
    excl_csv: set[str] = set()
    excl_rows: dict[str, set[int]] = collections.defaultdict(set)
    excl_cols: dict[str, set[str]] = collections.defaultdict(set)
    stats = collections.Counter()

    for pool in args.pools:
        for run_label, _types, run_id in REGISTRY[pool]:
            run_dir = OUTPUTS / run_id
            if not run_dir.is_dir():
                print(f"  WARN missing run dir: {pool}/{run_label} {run_id}")
                continue
            _cells, skipped, exclusions = _materialize_run(run_dir)
            stats["runs"] += 1
            for s in skipped:
                skip_keys.add((s.source, s.row, s.column))
            for e in exclusions:
                src = e.source or ""
                if e.scope == ExclusionScope.CSV:
                    excl_csv.add(src)
                elif e.scope == ExclusionScope.ROWS:
                    for r in e.rows or []:
                        try:
                            excl_rows[src].add(int(r))
                        except (TypeError, ValueError):
                            continue
                elif e.scope == ExclusionScope.COLUMNS:
                    for col in e.columns or []:
                        if col is not None:
                            excl_cols[src].add(str(col))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for (source, row, column) in sorted(skip_keys, key=lambda k: (k[0], str(k[1]), str(k[2]))):
            f.write(json.dumps({"kind": "skip", "source": source, "row": row, "column": column}) + "\n")
            n += 1
        for source in sorted(excl_csv):
            f.write(json.dumps({"kind": "exclude_csv", "source": source}) + "\n")
            n += 1
        for source in sorted(excl_rows):
            f.write(json.dumps({"kind": "exclude_rows", "source": source,
                                "rows": sorted(excl_rows[source])}) + "\n")
            n += 1
        for source in sorted(excl_cols):
            f.write(json.dumps({"kind": "exclude_cols", "source": source,
                                "columns": sorted(excl_cols[source])}) + "\n")
            n += 1

    print("=== export_neutral_cells summary ===")
    print(f"runs materialized   : {stats['runs']}")
    print(f"skip cells          : {len(skip_keys)}")
    print(f"exclude CSV (files) : {len(excl_csv)}")
    print(f"exclude-rows sources: {len(excl_rows)} ({sum(len(v) for v in excl_rows.values())} rows)")
    print(f"exclude-cols sources: {len(excl_cols)} ({sum(len(v) for v in excl_cols.values())} cols)")
    print(f"records written     : {n} -> {out_path}")


if __name__ == "__main__":
    main()
