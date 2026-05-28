"""Score column-clusters labels into per-cell inconsistency benchmark.

Reads ``column_clusters.jsonl`` (the labelling artifact for the
"inconsistency" error type; produced by Slice 1's store) and joins it
against the source CSVs to emit one
:class:`src.labeling.schema.BenchmarkCell` per matched
``(source, row, column)`` cell.

Mapping rule:
  * ``status=labeled`` -> emit ``label="error"`` for every row whose
    cell value appears in any cluster of that (source, column).
    Direction is encoded by the polymorphic ``corrected_value``:
    ``List[str]`` (sorted distinct values) when the cluster has no
    operator-confirmed canonical (bi-directional); scalar ``str``
    (the canonical) when the operator confirmed one. The LLM's
    proposed canonical is preserved as ``llm_suggested_value``.
  * ``status=no_inconsistency`` / ``dropped_*`` / ``unlabeled`` ->
    emit nothing.

The optional threshold-driven flip (``count(canonical)/total >= T``)
lives in ``src.labeling.column_cluster_export``; this scorer is the
bi-directional reference output that the exporter post-processes.

Usage::

    python -m experiments.score_column_clusters \\
        --clusters-jsonl <run_dir>/column_clusters.jsonl \\
        --data-dir <data_dir> \\
        --output benchmark_inconsistency.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.core.data_loader import load_csv  # noqa: E402
from src.labeling.column_cluster_store import ColumnClusterStore  # noqa: E402
from src.labeling.schema import (  # noqa: E402
    BenchmarkCell,
    ColumnClusters,
    ColumnLabelStatus,
)

logger = logging.getLogger("score_column_clusters")


# ---------------------------------------------------------------------
# Library API
# ---------------------------------------------------------------------


def _resolve_source_path(data_dir: Path, source: str) -> Optional[Path]:
    """Look up the CSV for ``source`` under ``data_dir``.

    Accepts both ``"foo"`` and ``"foo.csv"`` and tries the literal
    filename first, then the ``.csv`` suffix. Returns ``None`` when
    the file isn't found so the caller can warn-and-skip cleanly.
    """
    candidates = [source, f"{source}.csv"] if not source.endswith(".csv") else [source]
    for cand in candidates:
        path = data_dir / cand
        if path.is_file():
            return path
    return None


def _global_cluster_id(source: str, column: str, cluster_id: str) -> str:
    """Globally-unique cluster identifier for the emitted benchmark.

    A bare ``cluster_id`` (``"c1"``) is record-local; downstream
    cleaners need a stable handle that includes the (source, column)
    context. Using ``::`` as the separator matches the workspace's
    existing ``source::column`` convention in run dashboards.
    """
    return f"{source}::{column}::{cluster_id}"


def materialize_inconsistency_benchmark(
    records: Iterable[ColumnClusters],
    data_dir: Path,
    *,
    source_filter: Optional[str] = None,
    column_filter: Optional[str] = None,
    csv_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[BenchmarkCell]:
    """Return one :class:`BenchmarkCell` per matched cell.

    ``records`` should already be the latest-wins-reduced set per
    (source, column); pass ``store.latest_by_column().values()`` from
    a freshly-loaded :class:`ColumnClusterStore`.

    ``csv_cache`` (optional) lets a caller re-use loaded dataframes
    across calls. We never mutate the dataframes.

    Output shape per cell:
      * ``label="error"``
      * ``error_types=["inconsistency"]``
      * ``cluster_ids=[gid]`` and ``cluster_kinds=[cluster.kind]``
        (single-element lists — the unified schema uses lists for
        consistency with multi-cluster cells emitted by other
        detectors).
      * ``corrected_value`` is polymorphic: scalar canonical when
        the cluster has an operator-confirmed canonical (=>
        one-directional fix); sorted list of all distinct cluster
        values otherwise (=> bi-directional, downstream may map to
        any value including ``original_value``).
      * ``llm_suggested_value`` carries the LLM's proposed canonical
        (the cluster's ``llm_proposed_canonical``) as a hint,
        independent of whether the operator confirmed it.
    """
    cells: List[BenchmarkCell] = []
    cache: Dict[str, pd.DataFrame] = csv_cache if csv_cache is not None else {}
    stats = Counter()

    for rec in records:
        if rec.status != ColumnLabelStatus.LABELED:
            stats[f"skip_status_{rec.status.value}"] += 1
            continue
        if source_filter is not None and rec.source != source_filter:
            continue
        if column_filter is not None and rec.column != column_filter:
            continue

        # Load CSV once per source.
        if rec.source not in cache:
            path = _resolve_source_path(data_dir, rec.source)
            if path is None:
                logger.warning(
                    "Source CSV not found for %r under %s; skipping record.",
                    rec.source, data_dir,
                )
                stats["skip_missing_csv"] += 1
                continue
            try:
                cache[rec.source] = load_csv(path)
            except Exception as exc:  # noqa: BLE001 - report-and-skip
                logger.warning(
                    "Failed to load CSV %s: %s; skipping record.", path, exc,
                )
                stats["skip_csv_load_error"] += 1
                continue
        df = cache[rec.source]

        if rec.column not in df.columns:
            logger.warning(
                "Column %r not present in %s; skipping record.",
                rec.column, rec.source,
            )
            stats["skip_missing_column"] += 1
            continue

        col_series = df[rec.column].astype(str)

        for cluster in rec.clusters:
            gid = _global_cluster_id(rec.source, rec.column, cluster.cluster_id)
            distinct_values = sorted(v.value for v in cluster.values)
            cluster_value_set = set(distinct_values)
            # Polymorphic ``corrected_value``: scalar canonical when
            # confirmed (=> one-directional); sorted list of distinct
            # surface forms otherwise (=> bi-directional).
            corrected: Any
            if cluster.canonical_value is not None:
                corrected = cluster.canonical_value
            else:
                corrected = distinct_values
            # Vectorised match: every row in col_series that's a member
            # of the cluster contributes one benchmark cell.
            mask = col_series.isin(cluster_value_set)
            matched_idx = list(col_series.index[mask])
            if not matched_idx:
                logger.warning(
                    "Cluster %s has no matching rows in %s/%s; "
                    "labelling artifact may be stale.",
                    gid, rec.source, rec.column,
                )
                stats["empty_cluster_match"] += 1
                continue
            for ridx in matched_idx:
                original = col_series.at[ridx]
                cells.append(
                    BenchmarkCell(
                        source=rec.source,
                        row=int(ridx),
                        column=rec.column,
                        original_value=str(original),
                        label="error",
                        corrected_value=corrected,
                        llm_suggested_value=cluster.llm_proposed_canonical,
                        error_types=["inconsistency"],
                        cluster_ids=[gid],
                        cluster_kinds=[cluster.kind.value],
                        annotators=[rec.annotator],
                        note=rec.note,
                    )
                )
                stats["emit_cells"] += 1

    logger.info("scoring stats: %s", dict(stats))
    return cells


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _write_jsonl(out_path: Path, cells: List[BenchmarkCell]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for cell in cells:
            fh.write(json.dumps(cell.model_dump(), ensure_ascii=False) + "\n")
    # ``os.replace`` is the portable atomic-rename primitive (works on
    # both POSIX and Windows); the other writers in the column-cluster
    # chain (run_column_clustering._write_drafts_jsonl,
    # save_cluster_variant_records) use the same call.
    os.replace(tmp_path, out_path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score column-clusters labels into a per-cell inconsistency "
            "benchmark."
        )
    )
    parser.add_argument(
        "--clusters-jsonl", type=Path, required=True,
        help="Path to <run_dir>/column_clusters.jsonl.",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory holding the source CSV files referenced by the "
             "labelling artifact.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--source-filter", default=None,
        help="Optional: restrict to a single source CSV basename.",
    )
    parser.add_argument(
        "--column-filter", default=None,
        help="Optional: restrict to a single column header.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    store = ColumnClusterStore(args.clusters_jsonl)
    records = list(store.latest_by_column().values())
    logger.info(
        "Loaded %d latest record(s) from %s.", len(records), args.clusters_jsonl,
    )

    cells = materialize_inconsistency_benchmark(
        records, args.data_dir,
        source_filter=args.source_filter,
        column_filter=args.column_filter,
    )
    _write_jsonl(args.output, cells)
    logger.info("Wrote %d cell(s) to %s.", len(cells), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
