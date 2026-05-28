"""Runner CLI for column-clustering adaptive loop.

For each (source, column) candidate in a run dir, queries the LLM in
a **per-run outer loop** (see
:func:`src.core.value_cluster.adaptive.run_sweep`): pass 0 sweeps
every candidate with the seed prompt, pass 1 sweeps every still-
active candidate with variant 1, and so on. Columns drop out of
later passes as their Good-Turing / Chao1 estimator declares them
saturated. The final draft for each (source, column) is written to
``<run_dir>/column_cluster_drafts.jsonl``.

Each output JSONL line is the ``to_dict()`` payload of one
:class:`ColumnClusteringResult` (see
:mod:`src.core.value_cluster.adaptive` for the schema). The labelling
UI (Slice 6) reads this file to pre-populate the column board.

Cardinality filter mirrors the
``/api/runs/{run_id}/inconsistency-columns`` endpoint defaults: only
columns with < ``min_distinct`` distinct values are skipped (no
cluster is mathematically possible). ``max_distinct_ratio`` defaults
to ``1.0`` so high-cardinality columns (phone numbers, emails,
addresses, etc.) are sent to the LLM and surfaced for human review.
Lower the flag if you want the historical PK / free-text guard back.

Per-arm GT/Chao1 stats are appended to
``<run_dir>/good_turing/per_arm_estimates.json`` so the
labelling-effort panel can render the saturation chip per column.

Usage::

    python -m experiments.run_column_clustering \\
        --run-dir outputs/<run_id> \\
        --data-dir data/<dataset>  \\
        --model gpt-5.2 \\
        --reasoning-effort medium \\
        [--source <basename>]  [--column <header>] \\
        [--min-distinct 2]  [--max-distinct-ratio 1.0]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.adaptive.scoped_gt import ArmConfig  # noqa: E402
from src.core.data_loader import load_csv  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402
from src.core.value_cluster.adaptive import (  # noqa: E402
    ColumnClusteringAdaptive,
    ColumnClusteringResult,
    run_sweep,
)
from src.core.value_cluster.proposer import ValueClusterProposer  # noqa: E402

logger = logging.getLogger("run_column_clustering")


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------


def _is_candidate(
    n_rows: int, n_distinct: int,
    *, min_distinct: int, max_distinct_ratio: float,
) -> bool:
    if n_distinct < min_distinct:
        return False
    if n_rows == 0:
        return False
    return (n_distinct / n_rows) <= max_distinct_ratio


def _value_counts(series: pd.Series) -> List[Tuple[str, int]]:
    """Return ``[(value, count), ...]`` for a column, excluding empty
    cells. Order: descending count, ties alphabetical (stable across
    runs)."""
    counts: Dict[str, int] = {}
    for v in series.astype(str):
        if v == "":
            continue
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def enumerate_candidates(
    *,
    data_dir: Path,
    sources: Sequence[str],
    min_distinct: int,
    max_distinct_ratio: float,
    source_filter: Optional[str] = None,
    column_filter: Optional[str] = None,
) -> List[Tuple[str, str, List[Tuple[str, int]]]]:
    """Return ``[(source, column, value_counts), ...]`` for every
    (source, column) that satisfies the cardinality filter."""
    out: List[Tuple[str, str, List[Tuple[str, int]]]] = []
    for src in sources:
        if source_filter is not None and src != source_filter:
            continue
        path = data_dir / src
        if not path.is_file():
            path = data_dir / f"{src}.csv"
            if not path.is_file():
                logger.warning(
                    "source not found under %s: %s", data_dir, src,
                )
                continue
        try:
            df = load_csv(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load %s: %s", path, exc)
            continue
        n_rows = int(len(df))
        for col in df.columns:
            if column_filter is not None and col != column_filter:
                continue
            counts = _value_counts(df[col])
            if not _is_candidate(
                n_rows, len(counts),
                min_distinct=min_distinct,
                max_distinct_ratio=max_distinct_ratio,
            ):
                continue
            out.append((src, col, counts))
    return out


def enumerate_candidates_from_frames(
    *,
    source_frames: Dict[str, pd.DataFrame],
    min_distinct: int,
    max_distinct_ratio: float,
    source_filter: Optional[str] = None,
    column_filter: Optional[str] = None,
) -> List[Tuple[str, str, List[Tuple[str, int]]]]:
    """Variant of :func:`enumerate_candidates` that operates on
    already-loaded frames. Used by the cleaning runner so we don't
    re-read the CSVs from disk for the column-clustering pass.

    Frame keys ``source_frames`` should match the relative source
    identifier used elsewhere in the run (basename or workbook-keyed
    sheet name); this function does not interpret them, it only
    iterates.
    """
    out: List[Tuple[str, str, List[Tuple[str, int]]]] = []
    for src, df in source_frames.items():
        if source_filter is not None and src != source_filter:
            continue
        n_rows = int(len(df))
        for col in df.columns:
            if column_filter is not None and col != column_filter:
                continue
            counts = _value_counts(df[col])
            if not _is_candidate(
                n_rows, len(counts),
                min_distinct=min_distinct,
                max_distinct_ratio=max_distinct_ratio,
            ):
                continue
            out.append((src, col, counts))
    return out


# ---------------------------------------------------------------------------
# GT artifact
# ---------------------------------------------------------------------------


def _append_per_arm_stats(
    run_dir: Path,
    results: Sequence[ColumnClusteringResult],
) -> None:
    """Append per-(source, column) GT/Chao1 stats to the run's
    ``good_turing/per_arm_estimates.json``.

    Scope kind is ``"column_clustering"`` so it doesn't clash with
    existing per-cell arms ("column"/"csv"/"multi_table"/"workbook").
    The labelling-effort panel groups by scope_kind, so column
    clustering stats end up under their own chip without polluting
    the per-cell view.
    """
    gt_dir = run_dir / "good_turing"
    gt_dir.mkdir(parents=True, exist_ok=True)
    path = gt_dir / "per_arm_estimates.json"
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "good_turing/per_arm_estimates.json invalid, recreating: %s",
                exc,
            )
            existing = {}
    arms: List[Dict[str, Any]] = existing.get("arms") or []
    # Drop prior column_clustering arms so a re-run overwrites cleanly
    # (don't accumulate stale entries when the model / settings change).
    arms = [
        a for a in arms
        if (a or {}).get("scope_kind") != "column_clustering"
    ]
    for r in results:
        if r.runs_completed == 0:
            continue
        arms.append({
            "scope_kind": "column_clustering",
            "scope_id": f"{r.source}::{r.column}",
            "error_type": "inconsistency",
            "runs_completed": r.runs_completed,
            "stopped_reason": r.stopped_reason,
            "good_turing_missing_mass": r.gt_missing_mass,
            "chao1_estimated_missed": r.chao1_missed,
            "n_species_observed": len(r.draft_clusters),
        })
    existing["arms"] = arms
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _write_drafts_jsonl(
    out_path: Path,
    results: Sequence[ColumnClusteringResult],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def run_for_candidates(
    *,
    run_dir: Path,
    candidates: Sequence[Tuple[str, str, List[Tuple[str, int]]]],
    model: str,
    reasoning_effort: str,
    reasoning_summary: str,
    arm_config: ArmConfig,
    prompt_path: Path,
    max_workers: int,
    llm: Optional[LLMClient] = None,
    variants: Optional[Sequence[Mapping[str, str]]] = None,
) -> List[ColumnClusteringResult]:
    """Core driver: drives the per-run adaptive sweep over an already
    enumerated candidate list and writes
    ``column_cluster_drafts.jsonl`` + per-arm GT stats. Returns the
    results so callers can log counts.

    The sweep is **per-run, not per-column**: pass 0 calls every
    candidate column once with the seed prompt; pass 1 calls every
    still-active column once with variant 1, and so on. A column
    drops out of subsequent passes when its Good-Turing / Chao1
    estimator declares the arm saturated (or when ``k_max`` runs
    have been spent). This mirrors the chunk-scan adaptive layer in
    :mod:`experiments.run_cleaning_detection` so the operator sees
    one consistent ``Pass i/K`` cadence across error types.

    ``variants`` (optional) is the list of paraphrased prompts
    produced by
    :class:`src.adaptive.prompt_variants.ClusterPromptVariantGenerator`.
    When set, the proposer rotates through them on runs 1..K-1
    (run 0 always uses the seed prompt loaded from ``prompt_path``).
    """
    if llm is None:
        llm = LLMClient()
    # ``proposer`` is shared by every worker thread in ``run_sweep``
    # below. It is intentionally stateless post-init:
    # ``ValueClusterProposer`` only holds the prompt / variants / LLM
    # client (the latter is internally thread-safe), so concurrent
    # ``propose_once`` calls do not contend on any mutable field. The
    # per-arm bookkeeping (``proposals_by_run`` / ``species_to_runs``
    # / ``_stable_rng``) lives inside :class:`_ColumnArm` instances
    # owned by ``run_sweep`` and is mutated only by the dispatching
    # thread, never the workers. If you add caching, counters or
    # any other field that mutates inside ``propose_once``, route
    # them through a lock (or instantiate one proposer per worker) --
    # otherwise the per-thread results will silently corrupt one
    # another.
    proposer = ValueClusterProposer(
        llm=llm, model=model, prompt_path=prompt_path,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        variants=variants,
    )

    out_path = run_dir / "column_cluster_drafts.jsonl"
    if not candidates:
        results: List[ColumnClusteringResult] = []
        _write_drafts_jsonl(out_path, results)
        logger.info("column-clustering: no candidate columns; wrote empty drafts.")
        return results

    # tqdm config tuned for job-log capture: ``ascii=True`` keeps the
    # bar readable in non-TTY log files (which the workbench surfaces
    # to operators via /api/launcher/jobs/.../log), ``mininterval``
    # throttles updates so we don't spam the log file, and ``ncols``
    # keeps the line length stable so the runs.js log viewer (which
    # collapses ``\r`` overwrites onto a single live line) can repaint
    # without scrolling. We re-create a fresh bar per pass so the
    # ``total`` matches the (shrinking) active-arm count.
    _bar: Dict[str, Any] = {"obj": None}

    def _on_pass_start(run_idx: int, active: int, total: int) -> None:
        if _bar["obj"] is not None:
            _bar["obj"].close()
        k_max = max(1, int(arm_config.k_max or 1))
        bar = tqdm(
            total=active,
            desc=f"cluster pass {run_idx + 1}/{k_max}",
            unit="col",
            ascii=True,
            ncols=100,
            mininterval=1.0,
            miniters=1,
            dynamic_ncols=False,
            leave=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
            file=sys.stdout,
        )
        _bar["obj"] = bar
        logger.info(
            "column-clustering pass %d/%d: probing %d active column(s) "
            "(%d already saturated)",
            run_idx + 1, k_max, active, total - active,
        )

    def _on_arm_completed(run_idx: int, arm, completed_this_pass: int) -> None:
        bar = _bar["obj"]
        if bar is not None:
            bar.update(1)

    def _on_pass_end(
        run_idx: int,
        completed: int,
        n_saturated_total: int,
        n_active_next: int,
    ) -> None:
        bar = _bar["obj"]
        if bar is not None:
            bar.close()
            _bar["obj"] = None
        k_max = max(1, int(arm_config.k_max or 1))
        logger.info(
            "column-clustering pass %d/%d done: %d probed, "
            "%d saturated cumulatively, %d active next",
            run_idx + 1, k_max, completed, n_saturated_total, n_active_next,
        )

    with logging_redirect_tqdm():
        results = run_sweep(
            columns=candidates,
            proposer=proposer,
            arm_config=arm_config,
            max_workers=max_workers,
            on_pass_start=_on_pass_start,
            on_arm_completed=_on_arm_completed,
            on_pass_end=_on_pass_end,
        )
    # Defensive: close any bar left dangling if run_sweep exited early.
    if _bar["obj"] is not None:
        _bar["obj"].close()
        _bar["obj"] = None

    _write_drafts_jsonl(out_path, results)
    _append_per_arm_stats(run_dir, results)
    # One concise summary instead of two separate INFO lines so the
    # log isn't dominated by post-pass bookkeeping. The detailed
    # paths land in DEBUG for anyone who needs them.
    _n_with_runs = sum(1 for r in results if r.runs_completed > 0)
    logger.info(
        "column-clustering: %d/%d column(s) probed -> drafts at %s",
        _n_with_runs, len(results),
        out_path.relative_to(PROJECT_ROOT)
        if PROJECT_ROOT in out_path.parents
        else out_path,
    )
    logger.debug(
        "column-clustering: per-arm stats appended to %s",
        run_dir / "good_turing" / "per_arm_estimates.json",
    )
    return results


def run(
    *,
    run_dir: Path,
    data_dir: Path,
    sources: Sequence[str],
    model: str,
    reasoning_effort: str,
    reasoning_summary: str,
    arm_config: ArmConfig,
    prompt_path: Path,
    min_distinct: int,
    max_distinct_ratio: float,
    source_filter: Optional[str],
    column_filter: Optional[str],
    max_workers: int,
    llm: Optional[LLMClient] = None,
) -> List[ColumnClusteringResult]:
    candidates = enumerate_candidates(
        data_dir=data_dir,
        sources=sources,
        min_distinct=min_distinct,
        max_distinct_ratio=max_distinct_ratio,
        source_filter=source_filter,
        column_filter=column_filter,
    )
    return run_for_candidates(
        run_dir=run_dir,
        candidates=candidates,
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        arm_config=arm_config,
        prompt_path=prompt_path,
        max_workers=max_workers,
        llm=llm,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the column-clustering adaptive loop over a run dir."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--source", default=None,
        help="Optional: restrict to one source CSV basename.",
    )
    parser.add_argument(
        "--column", default=None,
        help="Optional: restrict to one column header.",
    )
    parser.add_argument(
        "--sources-file", type=Path, default=None,
        help="Optional: file listing source basenames (one per line). "
             "Defaults to enumerating data-dir for *.csv.",
    )
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument(
        "--reasoning-effort", default="medium",
        choices=["minimal", "low", "medium", "high"],
    )
    parser.add_argument(
        "--reasoning-summary", default="detailed",
        choices=["auto", "concise", "detailed"],
    )
    parser.add_argument(
        "--prompt", type=Path,
        default=PROJECT_ROOT / "prompts" / "cluster_column_values.yaml",
    )
    parser.add_argument("--min-distinct", type=int, default=2)
    # ``max-distinct-ratio=1.0`` disables the cardinality pre-filter
    # by default so every column with >=2 distinct values reaches
    # the LLM proposer and is available for human review. Set lower
    # (e.g. 0.95) to skip PK / free-text columns at intake time.
    parser.add_argument("--max-distinct-ratio", type=float, default=1.0)
    parser.add_argument(
        "--k-min", type=int, default=None,
        help="Override ArmConfig.k_min.",
    )
    parser.add_argument(
        "--k-max", type=int, default=None,
        help="Override ArmConfig.k_max.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="Override ArmConfig.epsilon (Good-Turing missing mass).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Threads for parallel column processing.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.run_dir.is_dir():
        parser.error(f"run dir not found: {args.run_dir}")
    if not args.data_dir.is_dir():
        parser.error(f"data dir not found: {args.data_dir}")

    # Source enumeration.
    if args.sources_file is not None:
        if not args.sources_file.is_file():
            parser.error(f"sources file not found: {args.sources_file}")
        sources = [
            ln.strip()
            for ln in args.sources_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        sources = sorted(
            p.name for p in args.data_dir.iterdir() if p.suffix == ".csv"
        )

    arm_config = ArmConfig(
        k_min=args.k_min, k_max=args.k_max, epsilon=args.epsilon,
    ).merge_into(ArmConfig.default())

    run(
        run_dir=args.run_dir,
        data_dir=args.data_dir,
        sources=sources,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary,
        arm_config=arm_config,
        prompt_path=args.prompt,
        min_distinct=args.min_distinct,
        max_distinct_ratio=args.max_distinct_ratio,
        source_filter=args.source,
        column_filter=args.column,
        max_workers=args.max_workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
