"""CLI: export ``benchmark_inconsistency_<ts>Z_<tag>.jsonl``.

Reads the labelling artifact written by Slice 3
(``<run_dir>/column_clusters.jsonl``) and materialises a per-cell
benchmark under ``label_result/`` with the project's standard naming
convention. Mirrors the workflow of the existing typo benchmark export.

Usage::

    python -m experiments.export_inconsistency_benchmark \\
        --run-dir outputs/<run_id> \\
        --data-dir data/<dataset> \\
        --label-result-dir label_result \\
        --tag gpt5-family \\
        [--threshold 0.6]

Default behaviour is bi-directional emission (every clustered cell is
``label="error"`` with the full cluster as ``acceptable_values``).
Passing ``--threshold T`` flips clusters whose canonical's frequency
share is ``>= T`` to ``direction="one_directional"`` with
``corrected_value`` populated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.column_cluster_export import (  # noqa: E402
    ClusterFlipDecision,
    benchmark_output_filename,
    export_inconsistency_benchmark,
    load_records_for_export,
)
from src.labeling.column_cluster_store import ColumnClusterStore  # noqa: E402
from src.labeling.schema import BenchmarkCell  # noqa: E402

logger = logging.getLogger("export_inconsistency_benchmark")


def _utc_timestamp(now: Optional[datetime] = None) -> str:
    """Return ``YYYYmmddTHHMMSS`` (no separator, no trailing 'Z').

    The trailing ``Z`` is appended by
    :func:`benchmark_output_filename` so the convention stays in one
    place.
    """
    dt = now if now is not None else datetime.now(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%S")


def _write_jsonl(out_path: Path, cells: List[BenchmarkCell]) -> None:
    """Crash-safe atomic write of the JSONL benchmark."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for cell in cells:
            fh.write(json.dumps(cell.model_dump(), ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)


def _write_flip_report(
    out_path: Path,
    decisions: List[ClusterFlipDecision],
    *,
    threshold: Optional[float],
) -> None:
    """Sidecar JSON: per-cluster flip decisions for transparency.

    Written next to the benchmark JSONL with suffix ``.flips.json`` so
    a reviewer can quickly see which clusters were treated as
    one-directional and why.
    """
    body = {
        "threshold": threshold,
        "decisions": [
            {
                "source": d.source,
                "column": d.column,
                "cluster_id": d.cluster_id,
                "canonical_value": d.canonical_value,
                "canonical_count": d.canonical_count,
                "total_count": d.total_count,
                "canonical_fraction": d.canonical_fraction,
                "flipped": d.flipped,
            }
            for d in decisions
        ],
        "summary": {
            "n_clusters": len(decisions),
            "n_flipped": sum(1 for d in decisions if d.flipped),
        },
    }
    out_path.write_text(
        json.dumps(body, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export column-cluster labels as a per-cell "
            "inconsistency benchmark JSONL."
        ),
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to the run directory containing column_clusters.jsonl.",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory holding the source CSV files.",
    )
    parser.add_argument(
        "--label-result-dir", type=Path,
        default=PROJECT_ROOT / "label_result",
        help="Output directory (default: label_result/ in the repo root).",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Optional tag suffix appended to the filename "
             "(e.g. 'gpt5-family' or '<run_id>').",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Optional canonical-fraction threshold in (0, 1]; "
             "clusters at or above this fraction flip to "
             "one_directional. Default off -> bi_directional only.",
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
        "--clusters-jsonl", type=Path, default=None,
        help="Override the column_clusters.jsonl location "
             "(default: <run-dir>/column_clusters.jsonl).",
    )
    parser.add_argument(
        "--timestamp", default=None,
        help="Override the timestamp tag (format YYYYmmddTHHMMSS, "
             "no trailing Z). Default: current UTC.",
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

    clusters_path = (
        args.clusters_jsonl
        if args.clusters_jsonl is not None
        else args.run_dir / "column_clusters.jsonl"
    )
    # Two record sources are supported:
    #   * column_clusters.jsonl  — operator finalised a disjoint
    #     clustering for the column via the column-board UI.
    #   * column_cluster_drafts.jsonl + labels.jsonl — operator
    #     voted TP/Partial on individual LLM-proposed draft
    #     clusters (per-cluster labelling). One single-cluster
    #     synthesised record is emitted per labelled draft so the
    #     scorer can process them unchanged. Columns covered by the
    #     saved record path take precedence (no double-counting).
    use_drafts_fallback = (
        args.clusters_jsonl is None
        and (args.run_dir / "column_cluster_drafts.jsonl").exists()
    )
    if not clusters_path.exists() and not use_drafts_fallback:
        logger.error(
            "neither column_clusters.jsonl nor column_cluster_drafts.jsonl "
            "found under %s",
            args.run_dir,
        )
        return 2

    if use_drafts_fallback:
        records = load_records_for_export(args.run_dir)
        logger.info(
            "Loaded %d record(s) "
            "(saved + synthesized from per-cluster labels) under %s.",
            len(records), args.run_dir,
        )
    else:
        store = ColumnClusterStore(clusters_path)
        records = list(store.latest_by_column().values())
        logger.info(
            "Loaded %d latest record(s) from %s.",
            len(records), clusters_path,
        )

    cells, decisions = export_inconsistency_benchmark(
        records, args.data_dir,
        threshold=args.threshold,
        source_filter=args.source_filter,
        column_filter=args.column_filter,
    )

    timestamp = args.timestamp or _utc_timestamp()
    filename = benchmark_output_filename(timestamp, args.tag)
    out_path = args.label_result_dir / filename
    _write_jsonl(out_path, cells)
    _write_flip_report(
        out_path.with_suffix(".flips.json"),
        decisions,
        threshold=args.threshold,
    )
    n_flipped = sum(1 for d in decisions if d.flipped)
    logger.info(
        "Wrote %d cell(s) to %s "
        "(clusters_considered=%d flipped=%d threshold=%s).",
        len(cells), out_path, len(decisions), n_flipped, args.threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
