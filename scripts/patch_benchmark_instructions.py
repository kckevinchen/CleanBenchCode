#!/usr/bin/env python3
"""Rewrite edit-INSTRUCTION prose that leaked into a cell benchmark's
``corrected_value`` into the concrete VALUE shown in the "Original / Fix" panel.

Some labelled cells carry an edit instruction ("Remove the trailing comma",
"Add closing parenthesis") in ``corrected_value`` instead of the literal
replacement value. That happens when the operator left the prefilled fix as an
instruction sentence (or an older, pre-fix label was captured). A benchmark
exists to teach ``original -> corrected`` pairs, so an instruction there is
unusable.

The concrete value the operator actually saw is the "Fix" the workbench renders
in the ``Original / Fix`` panel -- ``src.labeling.export._llm_suggestion_for_cluster``
(``synthesis.canonical_record`` / per-detection ``corrected_value`` literals,
with free-form prose only as a last resort). This one-off migration walks a
merged cell benchmark (``benchmark_cell.json`` JSONL + ``benchmark_cell.csv``)
and, for every ``error`` cell whose ``corrected_value`` reads as an instruction
(``_looks_like_instruction``), re-resolves it, in order:

  1. FIX PANEL -- the concrete ``_llm_suggestion_for_cluster`` value recovered
     from the source runs' clusters (the registry that built the benchmark);
  2. DERIVED -- the literal result of a trivial, unambiguous instruction applied
     to the original (``_derive_trivial_fix``, faithful by construction), used
     only when the panel Fix is itself prose;
  3. NULL -- ``corrected_value = null`` / empty CSV cell, when no concrete value
     exists anywhere (e.g. "set X to null"). The cell stays a confirmed
     ``error``; only its (unavailable) fix is dropped.

Every affected cell is written to a review CSV (with the source of the new
value) so the change can be audited. The rewrite mirrors
``experiments.export_cell_benchmark`` serialisation exactly. Idempotent: a
second run finds nothing to change.

Examples:

    # Dry-run: report what would change, write no files
    python -m scripts.patch_benchmark_instructions --dir benchmark/merged_v6 --dry-run

    # Apply in place + emit the review CSV alongside
    python -m scripts.patch_benchmark_instructions --dir benchmark/merged_v6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.export_merged_benchmark import OUTPUTS, REGISTRY  # noqa: E402
from src.labeling._helpers import (  # noqa: E402
    _derive_trivial_fix,
    _looks_like_instruction,
)
from src.labeling.export import _llm_suggestion_for_cluster  # noqa: E402

_CSV_FIELDS = [
    "source",
    "row",
    "column",
    "original_value",
    "status",
    "error_types",
    "corrected_value",
    "explanation",
    "dataset",
]

CellKey = tuple[str, Any, str]


def _csv_corrected(cv: Any) -> str:
    """Flatten a corrected_value for CSV: scalar verbatim, bi-directional
    list as a JSON array, null as empty. (Mirrors export_cell_benchmark.)"""
    if cv is None:
        return ""
    if isinstance(cv, list):
        return json.dumps(cv, ensure_ascii=False)
    return str(cv)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_FIELDS)
        for r in records:
            w.writerow(
                [
                    r["source"],
                    r["row"],
                    r["column"],
                    r["original_value"],
                    r["status"],
                    ";".join(r.get("error_types") or []),
                    _csv_corrected(r.get("corrected_value")),
                    r.get("explanation") or "",
                    r.get("dataset", ""),
                ]
            )
    tmp.replace(path)


def _fix_panel_index(keys: set[CellKey]) -> dict[CellKey, str]:
    """Recover the concrete "Original / Fix" panel value for each cell in
    ``keys`` from the source runs that built the benchmark.

    Scans every registry run's ``clusters.jsonl``; for each cluster matching a
    wanted ``(source, row, column)`` it computes
    ``_llm_suggestion_for_cluster`` and keeps the CONCRETE (non-instruction)
    candidate. When several runs offer a value, the most common wins (ties ->
    longest, then lexicographic) for determinism.
    """
    candidates: dict[CellKey, Counter] = {}
    run_ids = {run_id for pool in REGISTRY.values() for _, _, run_id in pool}
    for run_id in sorted(run_ids):
        path = OUTPUTS / run_id / "clusters.jsonl"
        if not path.is_file():
            continue
        for raw in path.open(encoding="utf-8"):
            if '"source"' not in raw:
                continue
            try:
                cluster = json.loads(raw)
            except json.JSONDecodeError:
                continue
            src = cluster.get("source")
            row = cluster.get("row")
            cols = cluster.get("columns") or []
            wanted_here = [c for c in cols if (src, row, c) in keys]
            if not wanted_here:
                continue
            suggestion = _llm_suggestion_for_cluster(cluster)
            for col in wanted_here:
                val = suggestion.get(col)
                if not isinstance(val, str):
                    continue
                val = val.strip()
                if not val or _looks_like_instruction(val):
                    continue
                candidates.setdefault((src, row, col), Counter())[val] += 1
    resolved: dict[CellKey, str] = {}
    for key, counter in candidates.items():
        best = max(counter.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))
        resolved[key] = best[0]
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        default=str(PROJECT_ROOT / "benchmark" / "merged_v6"),
        help="merged benchmark directory holding benchmark_cell.{json,csv}",
    )
    ap.add_argument(
        "--review-csv",
        default=None,
        help="path for the audit CSV (default: <dir>/instruction_fix_review.csv)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report changes but write nothing",
    )
    args = ap.parse_args()

    bdir = Path(args.dir)
    json_path = bdir / "benchmark_cell.json"
    csv_path = bdir / "benchmark_cell.csv"
    if not json_path.is_file():
        ap.error(f"missing {json_path}")
    review_path = Path(args.review_csv) if args.review_csv else bdir / "instruction_fix_review.csv"

    records: list[dict[str, Any]] = []
    for raw in json_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            records.append(json.loads(line))

    # Identify leaked cells first, then resolve their Fix-panel values in one
    # pass over the source runs.
    leaked_keys: set[CellKey] = set()
    for r in records:
        if r.get("status") != "error":
            continue
        cv = r.get("corrected_value")
        if isinstance(cv, str) and _looks_like_instruction(cv):
            leaked_keys.add((r["source"], r["row"], r["column"]))
    fix_panel = _fix_panel_index(leaked_keys) if leaked_keys else {}

    review_rows: list[dict[str, Any]] = []
    counts = {"FIX_PANEL": 0, "DERIVED": 0, "NULL": 0}
    for r in records:
        key: CellKey = (r.get("source"), r.get("row"), r.get("column"))
        if r.get("status") != "error" or key not in leaked_keys:
            continue
        cv = r["corrected_value"]
        original = r.get("original_value") or ""
        new_value = fix_panel.get(key)
        source_of_fix = "FIX_PANEL"
        if new_value is None:
            derived = _derive_trivial_fix(original, cv)
            if derived is not None:
                new_value, source_of_fix = derived, "DERIVED"
            else:
                source_of_fix = "NULL"
        counts[source_of_fix] += 1
        review_rows.append(
            {
                "source": r["source"],
                "row": r["row"],
                "column": r["column"],
                "original_value": original,
                "old_corrected_value": cv,
                "new_corrected_value": "" if new_value is None else new_value,
                "source_of_fix": source_of_fix,
                "error_types": ";".join(r.get("error_types") or []),
            }
        )
        if not args.dry_run:
            r["corrected_value"] = new_value

    print(f"scanned {len(records)} cells in {bdir}")
    print(
        f"instruction cells: {len(review_rows)}  "
        f"(FIX_PANEL={counts['FIX_PANEL']}, DERIVED={counts['DERIVED']}, "
        f"NULL={counts['NULL']})"
    )
    for row in review_rows:
        print(
            f"  [{row['source_of_fix']:9}] {row['source']} r{row['row']} "
            f"{row['column']!r}: {row['old_corrected_value']!r} -> "
            f"{row['new_corrected_value']!r}"
        )

    if args.dry_run:
        print("\n(dry-run: no files written)")
        return

    review_fields = [
        "source",
        "row",
        "column",
        "original_value",
        "old_corrected_value",
        "new_corrected_value",
        "source_of_fix",
        "error_types",
    ]
    with review_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=review_fields)
        w.writeheader()
        for row in review_rows:
            w.writerow(row)
    print(f"\nwrote review CSV: {review_path}")

    if review_rows:
        _write_jsonl(json_path, records)
        _write_csv(csv_path, records)
        print(f"rewrote {json_path.name} + {csv_path.name} ({len(records)} cells)")
    else:
        print("no benchmark changes needed")


if __name__ == "__main__":
    main()
