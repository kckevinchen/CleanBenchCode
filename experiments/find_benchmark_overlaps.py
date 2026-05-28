"""Find cells labeled for more than one error type across all benchmarks.

Each benchmark JSONL in ``benchmark/`` contributes exactly one error
type (``typo``, ``inconsistency``, ``numeric_outlier``,
``semantic_incompatible``, ``fd_violation``) and labels every record as
``"clean"``, ``"error"``, or ``"ambiguous"``. A cell, identified by
``(source, row, column)``, "overlaps" when it appears in two or more
benchmark files.

This script:

  * Joins all five benchmark JSONLs on ``(source, row, column)``.
  * Reports how many cells appear in 2, 3, 4, or 5 benchmarks.
  * Breaks the overlaps down by which error types co-occur and by the
    cross-product of per-benchmark labels.
  * Writes the full list of overlapping cells (with their per-error-type
    labels and ``original_value``) to a JSONL and CSV for follow-up.

Usage
-----

::

    python -m experiments.find_benchmark_overlaps                     \\
        --benchmark-dir benchmark                                     \\
        --output-dir outputs/benchmark_overlaps_<run_id>

If ``--output-dir`` is omitted the script writes to
``outputs/benchmark_overlaps_<UTC timestamp>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("find_benchmark_overlaps")


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------


CellKey = Tuple[str, int, str]


def _discover_benchmarks(benchmark_dir: Path) -> Dict[str, Path]:
    """Return ``{error_type: path}`` for every JSONL in ``benchmark_dir``.

    The error type is read from the first record so the script does not
    depend on filename conventions.
    """
    out: Dict[str, Path] = {}
    for path in sorted(benchmark_dir.glob("benchmark_*.jsonl")):
        with path.open() as fh:
            first = fh.readline().strip()
        if not first:
            logger.warning("skipping empty benchmark %s", path)
            continue
        rec = json.loads(first)
        ets = rec.get("error_types") or []
        if len(ets) != 1:
            raise ValueError(
                f"{path.name}: expected exactly one error type per record, "
                f"got {ets!r}"
            )
        et = ets[0]
        if et in out:
            raise ValueError(
                f"two benchmark files claim error type {et!r}: "
                f"{out[et].name} and {path.name}"
            )
        out[et] = path
    return out


def _load_benchmark(
    path: Path,
    include_clean: bool = False,
) -> Dict[CellKey, Dict[str, Any]]:
    """Index a benchmark by ``(source, row, column)``.

    By default ``clean`` records are dropped, so the index only contains
    cells with a positive label (``error`` or ``ambiguous``). Pass
    ``include_clean=True`` to retain everything.

    On the rare collision (same cell labeled twice in one file) the
    record with the strongest label wins, with ``error`` beating
    ``ambiguous`` beating ``clean``.
    """
    label_strength = {"error": 2, "ambiguous": 1, "clean": 0}
    out: Dict[CellKey, Dict[str, Any]] = {}
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not include_clean and rec["label"] == "clean":
                continue
            key = (rec["source"], int(rec["row"]), rec.get("column", ""))
            existing = out.get(key)
            if existing is None or label_strength.get(
                rec["label"], -1
            ) > label_strength.get(existing["label"], -1):
                out[key] = rec
    return out


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------


POSITIVE_LABELS = ("error", "ambiguous")


def _build_overlap_index(
    benchmarks: Dict[str, Dict[CellKey, Dict[str, Any]]],
) -> Dict[CellKey, Dict[str, Dict[str, Any]]]:
    """Return ``{cell_key: {error_type: record}}`` for cells in 2+ benchmarks."""
    all_keys: set[CellKey] = set()
    for idx in benchmarks.values():
        all_keys.update(idx.keys())

    overlaps: Dict[CellKey, Dict[str, Dict[str, Any]]] = {}
    for key in all_keys:
        per_type = {
            et: idx[key]
            for et, idx in benchmarks.items()
            if key in idx
        }
        if len(per_type) >= 2:
            overlaps[key] = per_type
    return overlaps


def _label_combo(
    per_type: Dict[str, Dict[str, Any]],
    error_types: List[str],
) -> Tuple[str, ...]:
    """Return labels in ``error_types`` order; missing slots become ``-``."""
    return tuple(
        per_type[et]["label"] if et in per_type else "-"
        for et in error_types
    )


def _summarise(
    overlaps: Dict[CellKey, Dict[str, Dict[str, Any]]],
    error_types: List[str],
) -> Dict[str, Any]:
    by_count = Counter()
    by_typeset = Counter()
    by_labelcombo: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

    for key, per_type in overlaps.items():
        by_count[len(per_type)] += 1
        ts = tuple(sorted(per_type.keys()))
        by_typeset[ts] += 1
        by_labelcombo[ts][_label_combo(per_type, error_types)] += 1

    return {
        "n_overlap_cells": len(overlaps),
        "by_count": dict(by_count),
        "by_typeset": by_typeset,
        "by_labelcombo": by_labelcombo,
    }


def _summarise_by_source(
    benchmarks: Dict[str, Dict[CellKey, Dict[str, Any]]],
    overlaps: Dict[CellKey, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Per-source totals for the inputs and the overlap set.

    Returns ``{source: {"positive_total": N, "positive_by_type": {et: n}, 
    "overlap_total": N, "overlap_by_count": {k: n}, 
    "overlap_by_typeset": {ts: n}}}``.
    """
    by_source: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "positive_total_keys": set(),
            "positive_by_type": Counter(),
            "overlap_total": 0,
            "overlap_by_count": Counter(),
            "overlap_by_typeset": Counter(),
        }
    )

    for et, idx in benchmarks.items():
        for (source, row, col) in idx:
            entry = by_source[source]
            entry["positive_total_keys"].add((row, col))
            entry["positive_by_type"][et] += 1

    for (source, row, col), per_type in overlaps.items():
        entry = by_source[source]
        entry["overlap_total"] += 1
        entry["overlap_by_count"][len(per_type)] += 1
        ts = tuple(sorted(per_type.keys()))
        entry["overlap_by_typeset"][ts] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for source, entry in by_source.items():
        out[source] = {
            "positive_total": len(entry["positive_total_keys"]),
            "positive_by_type": dict(entry["positive_by_type"]),
            "overlap_total": entry["overlap_total"],
            "overlap_by_count": dict(entry["overlap_by_count"]),
            "overlap_by_typeset": dict(entry["overlap_by_typeset"]),
        }
    return out


# ----------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------


def _fmt_typeset(ts: Tuple[str, ...]) -> str:
    return " + ".join(ts)


def _render_markdown(
    summary: Dict[str, Any],
    error_types: List[str],
    benchmark_paths: Dict[str, Path],
    benchmark_counts: Dict[str, int],
    include_clean: bool,
) -> str:
    lines: List[str] = []
    lines.append("# Benchmark overlap report")
    lines.append("")
    if include_clean:
        scope_sentence = (
            "Cells (`source`, `row`, `column`) that appear in more than "
            "one labeled benchmark with any label (`error`, `ambiguous`, "
            "or `clean`), and have therefore been labeled for more than "
            "one error type."
        )
    else:
        scope_sentence = (
            "Cells (`source`, `row`, `column`) that have a positive "
            "label (`error` or `ambiguous`) in more than one benchmark. "
            "`clean` records are excluded."
        )
    lines.append(scope_sentence)
    lines.append("")

    # Inputs ------------------------------------------------------------
    lines.append("## Inputs")
    lines.append("")
    header_count = (
        "Records (positive only)" if not include_clean else "Records"
    )
    lines.append(f"| Error type | File | {header_count} |")
    lines.append("| --- | --- | --- |")
    for et in error_types:
        path = benchmark_paths[et]
        n = benchmark_counts[et]
        lines.append(f"| `{et}` | `{path.name}` | {n} |")
    lines.append("")

    # Headline ----------------------------------------------------------
    lines.append("## Headline")
    lines.append("")
    n_overlap = summary["n_overlap_cells"]
    by_count = summary["by_count"]
    lines.append(
        f"- **{n_overlap}** cells have a positive label "
        f"(`error` or `ambiguous`) in **2 or more** error types."
    )
    if by_count:
        parts = [f"{n} cells in {k} benchmarks" for k, n in sorted(by_count.items())]
        lines.append(f"- Distribution: {', '.join(parts)}.")
    lines.append("")

    # Overlap by error-type set -----------------------------------------
    lines.append("## Overlaps by error-type combination")
    lines.append("")
    lines.append("| Error types | Cells |")
    lines.append("| --- | ---: |")
    typesets = sorted(
        summary["by_typeset"].keys(),
        key=lambda ts: (-summary["by_typeset"][ts], ts),
    )
    for ts in typesets:
        any_n = summary["by_typeset"][ts]
        lines.append(f"| {_fmt_typeset(ts)} | {any_n} |")
    lines.append("")

    # Label-combination breakdown ---------------------------------------
    lines.append("## Label-combination breakdown")
    lines.append("")
    lines.append(
        "For each pair (or larger set) of error types, how often each "
        "combination of per-benchmark labels appears. Columns follow the "
        "order: " + ", ".join(f"`{et}`" for et in error_types) + ". "
        "`-` means the cell is not in that benchmark."
    )
    lines.append("")
    for ts in typesets:
        lines.append(f"### {_fmt_typeset(ts)}")
        lines.append("")
        header = "| " + " | ".join(f"`{et}`" for et in error_types) + " | Cells |"
        sep = "| " + " | ".join("---" for _ in error_types) + " | ---: |"
        lines.append(header)
        lines.append(sep)
        combo_counter: Counter = summary["by_labelcombo"][ts]
        for combo, n in combo_counter.most_common():
            row = "| " + " | ".join(combo) + f" | {n} |"
            lines.append(row)
        lines.append("")

    # Files emitted -----------------------------------------------------
    lines.append("## Artifacts")
    lines.append("")
    lines.append(
        "- `overlapping_cells.jsonl` — one record per overlapping cell "
        "with the full per-benchmark labels and original values."
    )
    lines.append(
        "- `overlapping_cells.csv` — flattened CSV with one column per "
        "error type (`label_<error_type>`)."
    )
    lines.append(
        "- `overlap_by_source.md` / `overlap_by_source.csv` \u2014 "
        "per-`source` CSV breakdown of overlap counts by error-type "
        "combination."
    )
    lines.append("")
    return "\n".join(lines)


def _render_per_source_markdown(
    per_source: Dict[str, Dict[str, Any]],
    error_types: List[str],
    include_clean: bool,
) -> str:
    lines: List[str] = []
    lines.append("# Benchmark overlap \u2014 per-source breakdown")
    lines.append("")
    if include_clean:
        scope = (
            "All records are counted regardless of label "
            "(`error`, `ambiguous`, or `clean`)."
        )
    else:
        scope = (
            "Only positive records (`error` or `ambiguous`) are counted; "
            "`clean` records are excluded."
        )
    lines.append(scope)
    lines.append("")
    lines.append(
        "For each `source` CSV, this report shows how many positively "
        "labeled cells appear in more than one benchmark, broken down "
        "by which error-type combinations they participate in."
    )
    lines.append("")

    # Sort: sources with any overlap first (by overlap_total desc), 
    # then alphabetically.
    items = sorted(
        per_source.items(),
        key=lambda kv: (-kv[1]["overlap_total"], kv[0]),
    )

    n_total_sources = len(items)
    n_sources_with_overlap = sum(
        1 for _, v in items if v["overlap_total"] > 0
    )
    n_overlap_cells = sum(v["overlap_total"] for _, v in items)
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- **{n_sources_with_overlap}** of **{n_total_sources}** "
        f"source CSVs have at least one overlapping cell."
    )
    lines.append(
        f"- **{n_overlap_cells}** overlapping cells in total "
        "(sum across all sources)."
    )
    lines.append("")

    # Per-source summary table -----------------------------------------
    lines.append("## Per-source summary")
    lines.append("")
    header_cols = (
        ["source", "positive cells", "overlap cells", "top combo", "top combo cells"]
    )
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append(
        "| --- | ---: | ---: | --- | ---: |"
    )
    for source, info in items:
        if info["overlap_by_typeset"]:
            top_ts, top_n = max(
                info["overlap_by_typeset"].items(),
                key=lambda kv: (kv[1], kv[0]),
            )
            top_combo = _fmt_typeset(top_ts)
        else:
            top_combo = "\u2013"
            top_n = 0
        lines.append(
            f"| `{source}` | {info['positive_total']} | "
            f"{info['overlap_total']} | {top_combo} | {top_n} |"
        )
    lines.append("")

    # Per-source combo detail ------------------------------------------
    lines.append("## Per-source overlap by error-type combination")
    lines.append("")
    lines.append(
        "Only sources with at least one overlapping cell are listed."
    )
    lines.append("")
    for source, info in items:
        if info["overlap_total"] == 0:
            continue
        lines.append(f"### `{source}`")
        lines.append("")
        lines.append("| Error types | Cells |")
        lines.append("| --- | ---: |")
        combos = sorted(
            info["overlap_by_typeset"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for ts, n in combos:
            lines.append(f"| {_fmt_typeset(ts)} | {n} |")
        # Per-error-type positive counts for context.
        pos_parts = [
            f"{et}={n}"
            for et, n in sorted(info["positive_by_type"].items())
        ]
        lines.append("")
        lines.append(
            f"Positive-label totals in this source: {', '.join(pos_parts)} "
            f"(unique positive cells: {info['positive_total']})."
        )
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------


def _write_overlap_files(
    overlaps: Dict[CellKey, Dict[str, Dict[str, Any]]],
    error_types: List[str],
    out_dir: Path,
) -> None:
    jsonl_path = out_dir / "overlapping_cells.jsonl"
    csv_path = out_dir / "overlapping_cells.csv"

    # Sort for deterministic output.
    sorted_items = sorted(
        overlaps.items(),
        key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]),
    )

    with jsonl_path.open("w") as fh:
        for key, per_type in sorted_items:
            source, row, column = key
            originals = {
                et: rec.get("original_value")
                for et, rec in per_type.items()
            }
            distinct_orig = {v for v in originals.values() if v is not None}
            rec_out = {
                "source": source,
                "row": row,
                "column": column,
                "original_value": (
                    next(iter(distinct_orig)) if len(distinct_orig) == 1 else None
                ),
                "original_values_per_type": originals,
                "labels": {
                    et: per_type[et]["label"] for et in per_type
                },
                "error_types": sorted(per_type.keys()),
                "n_positive": sum(
                    1
                    for r in per_type.values()
                    if r["label"] in POSITIVE_LABELS
                ),
            }
            fh.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    fieldnames = [
        "source",
        "row",
        "column",
        "n_error_types",
        "n_positive",
        "error_types",
        "original_value",
    ] + [f"label_{et}" for et in error_types]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key, per_type in sorted_items:
            source, row, column = key
            originals = {
                et: rec.get("original_value")
                for et, rec in per_type.items()
            }
            distinct_orig = {v for v in originals.values() if v is not None}
            row_out = {
                "source": source,
                "row": row,
                "column": column,
                "n_error_types": len(per_type),
                "n_positive": sum(
                    1
                    for r in per_type.values()
                    if r["label"] in POSITIVE_LABELS
                ),
                "error_types": "+".join(sorted(per_type.keys())),
                "original_value": (
                    next(iter(distinct_orig)) if len(distinct_orig) == 1 else ""
                ),
            }
            for et in error_types:
                row_out[f"label_{et}"] = (
                    per_type[et]["label"] if et in per_type else ""
                )
            writer.writerow(row_out)


def _write_per_source_csv(
    per_source: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Write a long-form per-source x per-combo CSV.

    One row per (source, error_type_combo). Sources with zero overlap
    still get a single row with empty ``error_types_combo`` so they
    appear in the file.
    """
    path = out_dir / "overlap_by_source.csv"
    fieldnames = [
        "source",
        "positive_total",
        "overlap_total",
        "error_types_combo",
        "n_error_types_in_combo",
        "cells_in_combo",
    ]
    items = sorted(
        per_source.items(),
        key=lambda kv: (-kv[1]["overlap_total"], kv[0]),
    )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for source, info in items:
            if not info["overlap_by_typeset"]:
                writer.writerow(
                    {
                        "source": source,
                        "positive_total": info["positive_total"],
                        "overlap_total": info["overlap_total"],
                        "error_types_combo": "",
                        "n_error_types_in_combo": 0,
                        "cells_in_combo": 0,
                    }
                )
                continue
            combos = sorted(
                info["overlap_by_typeset"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
            for ts, n in combos:
                writer.writerow(
                    {
                        "source": source,
                        "positive_total": info["positive_total"],
                        "overlap_total": info["overlap_total"],
                        "error_types_combo": "+".join(ts),
                        "n_error_types_in_combo": len(ts),
                        "cells_in_combo": n,
                    }
                )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmark",
        help="Directory containing benchmark_*.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write report and artifacts to. Default: "
             "outputs/benchmark_overlaps_<UTC timestamp>.",
    )
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Include `clean` records in the overlap analysis. By "
             "default only `error` and `ambiguous` records count.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir = args.output_dir
    if out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = PROJECT_ROOT / "outputs" / f"benchmark_overlaps_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output directory: %s", out_dir)

    benchmark_paths = _discover_benchmarks(args.benchmark_dir)
    if not benchmark_paths:
        logger.error("no benchmark files found under %s", args.benchmark_dir)
        return 1
    error_types = sorted(benchmark_paths.keys())
    logger.info(
        "loaded %d benchmark files: %s",
        len(error_types),
        ", ".join(error_types),
    )

    benchmarks: Dict[str, Dict[CellKey, Dict[str, Any]]] = {}
    benchmark_counts: Dict[str, int] = {}
    for et in error_types:
        idx = _load_benchmark(
            benchmark_paths[et], include_clean=args.include_clean,
        )
        benchmarks[et] = idx
        benchmark_counts[et] = len(idx)
        logger.info("  %s: %d unique cells", et, len(idx))

    overlaps = _build_overlap_index(benchmarks)
    logger.info("%d cells overlap (appear in >= 2 benchmarks)", len(overlaps))

    summary = _summarise(overlaps, error_types)

    md = _render_markdown(
        summary,
        error_types,
        benchmark_paths,
        benchmark_counts,
        include_clean=args.include_clean,
    )
    (out_dir / "overlap_report.md").write_text(md)
    _write_overlap_files(overlaps, error_types, out_dir)

    per_source = _summarise_by_source(benchmarks, overlaps)
    per_source_md = _render_per_source_markdown(
        per_source, error_types, include_clean=args.include_clean
    )
    (out_dir / "overlap_by_source.md").write_text(per_source_md)
    _write_per_source_csv(per_source, out_dir)

    print(f"wrote {out_dir / 'overlap_report.md'}")
    print(f"wrote {out_dir / 'overlapping_cells.jsonl'}")
    print(f"wrote {out_dir / 'overlapping_cells.csv'}")
    print(f"wrote {out_dir / 'overlap_by_source.md'}")
    print(f"wrote {out_dir / 'overlap_by_source.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
