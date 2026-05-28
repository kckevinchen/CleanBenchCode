"""Export an inconsistency benchmark with LLM-generated canonical lists.

Thin CLI wrapper around
:func:`src.labeling.inconsistency_export.materialize_inconsistency_benchmark`.

The same materialiser also powers the live "Benchmark" download
button in the labelling UI (cache-only mode), so the offline JSONL
emitted here and the on-demand JSONL streamed by the FastAPI
endpoint are byte-identical for a given (run-dir, cache) snapshot.

Per-cell label rule (union of involved clusters):

    * any involved cluster has verdict ``tp``     -> label = ``error``
    * else any involved cluster has verdict ``unsure`` -> ``ambiguous``

Cells whose value does not appear in ANY ``tp``/``unsure`` cluster
are not emitted.

Usage (typical):

    python -m experiments.export_inconsistency_benchmark_with_llm \\
        --run-dir outputs/20260513T014840Z_gpt5-family \\
        --output  label_result/benchmark_inconsistency_<ts>Z_<tag>.jsonl \\
        --model   gpt-5.2 \\
        --concurrency 8

Use ``--dry-run`` to skip the LLM and emit cells with
``corrected_value=null`` (operator-saved ``value_corrections`` are
still applied as authoritative overrides). Useful for sanity-checking
the cell enumeration and replaying after appending new
``[llm-suggested]`` labels.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

# Make ``src`` importable when run as a script as well as a module.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402
from src.labeling.inconsistency_export import (  # noqa: E402
    materialize_inconsistency_benchmark,
)
from src.labeling.schema import BenchmarkCell  # noqa: E402

logger = logging.getLogger("export_inconsistency_benchmark_with_llm")


def _write_jsonl(out_path: Path, cells: Iterable[BenchmarkCell]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    n = 0
    with tmp_path.open("w", encoding="utf-8") as fh:
        for cell in cells:
            fh.write(json.dumps(cell.model_dump(), ensure_ascii=False))
            fh.write("\n")
            n += 1
    os.replace(tmp_path, out_path)
    return n


def _default_output_path(run_id: str, tag: str) -> Path:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"benchmark_inconsistency_{ts}_{tag}.jsonl"
    return REPO_ROOT / "label_result" / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", required=True, type=Path,
        help="Run directory containing labels.jsonl, "
             "column_cluster_drafts.jsonl, and summary.json.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path (default: label_result/"
             "benchmark_inconsistency_<ts>Z_<tag>.jsonl).",
    )
    parser.add_argument(
        "--tag", default="gpt5-family",
        help="Filename tag suffix when --output is not given.",
    )
    parser.add_argument(
        "--model", default="gpt-5.2",
        help="Model to query for the possible-correct list "
             "(comma-list accepted by LLMClient).",
    )
    parser.add_argument(
        "--reasoning-effort", default="medium",
        choices=["minimal", "low", "medium", "high"],
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Number of parallel LLM calls.",
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of independent LLM samples per cell group; final "
             "canonical list is taken by majority vote across samples.",
    )
    parser.add_argument(
        "--samples-threshold", type=float, default=0.5,
        help="Vote-fraction threshold a canonical must clear to be "
             "included in the consensus list (default 0.5 = majority).",
    )
    parser.add_argument(
        "--retry-samples", type=int, default=4,
        help="For groups where the initial K=samples consensus is "
             "exactly [original_value] (LLM says 'no fix needed'), "
             "draw this many ADDITIONAL samples and re-aggregate. "
             "Reviewer-labeled (TP/unsure) cells are never dropped; "
             "if the retried consensus is still [original_value], "
             "the cell is emitted with corrected_value=null. Set to "
             "0 to disable retries (cells will still be emitted, just "
             "without the extra LLM calls).",
    )
    parser.add_argument(
        "--row-context-rows", type=int, default=3,
        help="Number of sample CSV rows containing the cell value to "
             "include in the LLM prompt (0 disables row context).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Resumable LLM cache JSONL (default: "
             "<run-dir>/inconsistency_llm_cache.jsonl).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N cell groups (debugging aid).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip the LLM and emit cells with corrected_value=null.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir does not exist: {run_dir}")
    run_id = run_dir.name
    output_path = args.output or _default_output_path(run_id, args.tag)

    llm: Optional[LLMClient] = None if args.dry_run else LLMClient()

    stats: Counter = Counter()

    def _progress(phase_label: str, done: int, total: int) -> None:
        if total <= 0:
            return
        if done % 20 == 0 or done == total:
            logger.info(
                "[%s %d/%d] LLM progress "
                "(ok=%d parse_fail=%d err=%d)",
                phase_label, done, total,
                stats.get("llm_ok", 0),
                stats.get("llm_parse_failed", 0),
                stats.get("llm_error", 0),
            )

    cells = materialize_inconsistency_benchmark(
        run_dir,
        llm_client=llm,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        samples=args.samples,
        samples_threshold=args.samples_threshold,
        retry_samples=args.retry_samples,
        row_context_rows=args.row_context_rows,
        cache_path=args.cache,
        concurrency=args.concurrency,
        progress_cb=_progress,
        dry_run=args.dry_run,
        limit=args.limit,
        stats=stats,
    )

    label_counts: Counter = Counter()
    for cell in cells:
        label_counts[cell.label] += 1

    n_written = _write_jsonl(output_path, cells)
    logger.info(
        "wrote %d benchmark cells -> %s", n_written, output_path,
    )
    logger.info("label distribution: %s", dict(label_counts))
    logger.info("stats: %s", dict(stats))


if __name__ == "__main__":
    main()
