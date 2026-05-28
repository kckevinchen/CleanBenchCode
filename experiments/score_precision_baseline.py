"""Score predictions from ``run_precision_baseline`` against the labeled
benchmark JSONLs in ``benchmark/``.

For each requested single-column error type:

  * Load ``<run_dir>/predictions_<error_type>.jsonl`` (cell-level
    predictions emitted by the baseline runner).
  * Load the matching ``benchmark/benchmark_<error_type>_*.jsonl``
    (cell-level ground-truth labels: ``"error"`` / ``"ambiguous"`` /
    ``"clean"``).
  * Match cells by ``(source, row, column)`` (the benchmark's primary
    key). Three views are computed:

      - **STRICT**: a prediction matches a benchmark "error" cell;
        ambiguous benchmark cells are not counted.
      - **LENIENT**: a prediction matches "error" OR "ambiguous"
        benchmark cells.
      - **PRECISION-OVER-LABELED-UNIVERSE**: restrict predictions to
        cells that *appear in the benchmark at all*. This is the
        cleanest precision read: a predicted cell that exists in the
        benchmark and is labeled "clean" is a true FP; one labeled
        "error" is a TP; one labeled "ambiguous" is soft TP.
        Predictions on cells NOT in the benchmark are unmeasured and
        excluded from the denominator.

Output:
  * ``<run_dir>/scoring/<error_type>.json`` -- raw metrics per type.
  * ``<run_dir>/scoring/per_source_<error_type>.csv`` -- per-source
    breakdown for the precision-over-labeled view.
  * ``<run_dir>/precision_baseline_report.md`` -- human-readable
    cross-type report.

Example::

    python -m experiments.score_precision_baseline \\
        outputs/precision_baseline_20260517T... \\
        --benchmark-dir benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("score_precision_baseline")

DEFAULT_ERROR_TYPES = (
    "typo", "inconsistency", "numeric_outlier", "semantic_incompatible",
)


@dataclass(frozen=True)
class CellKey:
    source: str
    row: int
    column: str


@dataclass
class BenchEntry:
    label: str  # "error" | "ambiguous" | "clean"
    original_value: str
    corrected_value: Optional[str]
    note: str = ""


# ---------------------------------------------------------------------
# Benchmark file resolution
# ---------------------------------------------------------------------


def _find_benchmark_file(bench_dir: Path, error_type: str) -> Optional[Path]:
    """Pick the most recent ``benchmark_<error_type>_*.jsonl`` in dir."""
    candidates = sorted(
        bench_dir.glob(f"benchmark_{error_type}_*.jsonl"),
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_benchmark(path: Path) -> Dict[CellKey, BenchEntry]:
    out: Dict[CellKey, BenchEntry] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = str(rec.get("source") or "")
            row = rec.get("row")
            col = str(rec.get("column") or "")
            if not src or row is None:
                continue
            key = CellKey(source=src, row=int(row), column=col)
            out[key] = BenchEntry(
                label=str(rec.get("label") or "clean"),
                original_value=str(rec.get("original_value") or ""),
                corrected_value=(
                    None if rec.get("corrected_value") is None
                    else str(rec.get("corrected_value"))
                ),
                note=str(rec.get("note") or ""),
            )
    return out


def load_predictions(path: Path) -> List[Dict]:
    out = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def predictions_to_keys(preds: Iterable[Dict]) -> Dict[CellKey, Dict]:
    """Index predictions by cell key. If the same cell is predicted
    twice (e.g. value present in multiple findings), the last write wins
    -- not an issue for metrics since we count predicted-cells, not
    raw prediction records."""
    out: Dict[CellKey, Dict] = {}
    for p in preds:
        src = str(p.get("source") or "")
        row = p.get("row")
        col = str(p.get("column") or "")
        if not src or row is None:
            continue
        key = CellKey(source=src, row=int(row), column=col)
        out[key] = p
    return out


# ---------------------------------------------------------------------
# Per-type scoring
# ---------------------------------------------------------------------


def score_one(
    error_type: str,
    preds: Dict[CellKey, Dict],
    bench: Dict[CellKey, BenchEntry],
) -> Dict:
    """Compute strict / lenient / labeled-universe metrics."""
    bench_keys = set(bench.keys())
    pred_keys = set(preds.keys())

    n_pred = len(pred_keys)
    n_bench_error = sum(1 for e in bench.values() if e.label == "error")
    n_bench_ambig = sum(1 for e in bench.values() if e.label == "ambiguous")
    n_bench_clean = sum(1 for e in bench.values() if e.label == "clean")

    # STRICT view
    tp_strict = sum(
        1 for k in pred_keys
        if k in bench and bench[k].label == "error"
    )
    fp_strict = n_pred - tp_strict
    fn_strict = n_bench_error - tp_strict

    # LENIENT view
    tp_lenient = sum(
        1 for k in pred_keys
        if k in bench and bench[k].label in ("error", "ambiguous")
    )
    fp_lenient = n_pred - tp_lenient
    fn_lenient = (n_bench_error + n_bench_ambig) - tp_lenient

    # LABELED-UNIVERSE view (restrict predictions to cells in bench)
    pred_in_bench = pred_keys & bench_keys
    n_pred_lu = len(pred_in_bench)
    tp_lu_strict = sum(1 for k in pred_in_bench if bench[k].label == "error")
    tp_lu_lenient = sum(
        1 for k in pred_in_bench
        if bench[k].label in ("error", "ambiguous")
    )
    fp_lu_strict = n_pred_lu - tp_lu_strict
    fp_lu_lenient = n_pred_lu - tp_lu_lenient

    def _safe(num, den):
        return (num / den) if den else None

    def _f1(p, r):
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    # Headline metrics
    prec_strict = _safe(tp_strict, n_pred)
    rec_strict = _safe(tp_strict, n_bench_error)
    f1_strict = _f1(prec_strict, rec_strict)

    prec_lenient = _safe(tp_lenient, n_pred)
    rec_lenient = _safe(tp_lenient, n_bench_error + n_bench_ambig)
    f1_lenient = _f1(prec_lenient, rec_lenient)

    prec_lu_strict = _safe(tp_lu_strict, n_pred_lu)
    prec_lu_lenient = _safe(tp_lu_lenient, n_pred_lu)

    # Coverage diagnostics
    coverage = _safe(n_pred_lu, n_pred)  # share of preds that landed on labeled cells

    out = {
        "error_type": error_type,
        "n_predictions": n_pred,
        "n_predictions_in_labeled_universe": n_pred_lu,
        "n_benchmark_cells": len(bench_keys),
        "n_benchmark_error": n_bench_error,
        "n_benchmark_ambiguous": n_bench_ambig,
        "n_benchmark_clean": n_bench_clean,
        "strict": {
            "TP": tp_strict, "FP": fp_strict, "FN": fn_strict,
            "precision": prec_strict, "recall": rec_strict, "f1": f1_strict,
        },
        "lenient": {
            "TP": tp_lenient, "FP": fp_lenient, "FN": fn_lenient,
            "precision": prec_lenient, "recall": rec_lenient, "f1": f1_lenient,
        },
        "labeled_universe": {
            "n_predictions": n_pred_lu,
            "coverage_of_predictions_in_labeled_universe": coverage,
            "strict_precision": prec_lu_strict,
            "lenient_precision": prec_lu_lenient,
            "TP_strict": tp_lu_strict, "FP_strict": fp_lu_strict,
            "TP_lenient": tp_lu_lenient, "FP_lenient": fp_lu_lenient,
        },
    }
    return out


def categorize_examples(
    preds: Dict[CellKey, Dict],
    bench: Dict[CellKey, BenchEntry],
    *,
    max_per_bucket: int = 12,
) -> Dict[str, List[Dict]]:
    """Pick representative cell-level examples in each bucket.

    Buckets emitted (each capped at ``max_per_bucket``):
      * ``tp_strict``           -- prediction matched bench label=error
      * ``tp_lenient_ambig``    -- prediction matched bench label=ambiguous
                                    (TP under lenient, FP under strict)
      * ``fp_in_labeled``       -- prediction landed on a bench cell
                                    labeled "clean" (a true FP)
      * ``fp_outside_labeled``  -- prediction on a cell NOT in the
                                    benchmark (unmeasured -- caller has
                                    to inspect to judge)
      * ``fn_error``            -- bench label=error not predicted
      * ``fn_ambig``            -- bench label=ambiguous not predicted
    """
    tp_strict, tp_lenient_ambig = [], []
    fp_in_labeled, fp_outside_labeled = [], []
    for k, p in preds.items():
        be = bench.get(k)
        sample = {
            "source": k.source, "row": k.row, "column": k.column,
            "original_value": p.get("original_value", ""),
            "predicted_correction": p.get("predicted_correction"),
            "rationale": p.get("rationale", ""),
        }
        if be is None:
            if len(fp_outside_labeled) < max_per_bucket:
                fp_outside_labeled.append(sample)
        elif be.label == "error":
            sample["benchmark_corrected_value"] = be.corrected_value
            if len(tp_strict) < max_per_bucket:
                tp_strict.append(sample)
        elif be.label == "ambiguous":
            sample["benchmark_corrected_value"] = be.corrected_value
            sample["benchmark_note"] = be.note
            if len(tp_lenient_ambig) < max_per_bucket:
                tp_lenient_ambig.append(sample)
        else:  # clean
            sample["benchmark_label"] = "clean"
            if len(fp_in_labeled) < max_per_bucket:
                fp_in_labeled.append(sample)

    fn_error, fn_ambig = [], []
    for k, be in bench.items():
        if k in preds:
            continue
        sample = {
            "source": k.source, "row": k.row, "column": k.column,
            "original_value": be.original_value,
            "benchmark_corrected_value": be.corrected_value,
            "benchmark_note": be.note,
        }
        if be.label == "error" and len(fn_error) < max_per_bucket:
            fn_error.append(sample)
        elif be.label == "ambiguous" and len(fn_ambig) < max_per_bucket:
            fn_ambig.append(sample)

    return {
        "tp_strict": tp_strict,
        "tp_lenient_ambig": tp_lenient_ambig,
        "fp_in_labeled": fp_in_labeled,
        "fp_outside_labeled": fp_outside_labeled,
        "fn_error": fn_error,
        "fn_ambig": fn_ambig,
    }


def coverage_diagnostic(
    preds: Dict[CellKey, Dict],
    bench: Dict[CellKey, BenchEntry],
) -> Dict:
    """Recall attainability: of benchmark errors, how many fell in
    columns the baseline actually saw a prediction for?

    A column the baseline "saw" is any (source, column) pair where the
    baseline emitted at least one prediction. Benchmark errors in
    NEVER-SEEN columns are upstream-missed (recall ceiling). Errors in
    seen columns that were not predicted are LLM-missed.
    """
    seen_cols = {(k.source, k.column) for k in preds.keys()}
    err_in_seen, err_in_unseen = 0, 0
    ambig_in_seen, ambig_in_unseen = 0, 0
    for k, e in bench.items():
        in_seen = (k.source, k.column) in seen_cols
        if e.label == "error":
            if in_seen: err_in_seen += 1
            else: err_in_unseen += 1
        elif e.label == "ambiguous":
            if in_seen: ambig_in_seen += 1
            else: ambig_in_unseen += 1
    return {
        "n_columns_seen_by_baseline": len(seen_cols),
        "errors_in_seen_columns": err_in_seen,
        "errors_in_unseen_columns": err_in_unseen,
        "ambiguous_in_seen_columns": ambig_in_seen,
        "ambiguous_in_unseen_columns": ambig_in_unseen,
    }


def per_source_breakdown(
    preds: Dict[CellKey, Dict],
    bench: Dict[CellKey, BenchEntry],
) -> List[Dict]:
    """Per-source view for the labeled-universe perspective."""
    by_source = defaultdict(lambda: {"n_pred": 0, "n_pred_lu": 0,
                                     "tp_strict": 0, "tp_lenient": 0,
                                     "n_bench_error": 0, "n_bench_ambig": 0,
                                     "n_bench_clean": 0})
    for k in preds:
        s = by_source[k.source]
        s["n_pred"] += 1
        if k in bench:
            s["n_pred_lu"] += 1
            label = bench[k].label
            if label == "error":
                s["tp_strict"] += 1
                s["tp_lenient"] += 1
            elif label == "ambiguous":
                s["tp_lenient"] += 1
    for k, e in bench.items():
        s = by_source[k.source]
        if e.label == "error":
            s["n_bench_error"] += 1
        elif e.label == "ambiguous":
            s["n_bench_ambig"] += 1
        else:
            s["n_bench_clean"] += 1
    rows = []
    for src, s in sorted(by_source.items()):
        prec_strict = (s["tp_strict"] / s["n_pred_lu"]) if s["n_pred_lu"] else None
        prec_lenient = (s["tp_lenient"] / s["n_pred_lu"]) if s["n_pred_lu"] else None
        rec_strict = (s["tp_strict"] / s["n_bench_error"]) if s["n_bench_error"] else None
        rec_lenient = (
            s["tp_lenient"] / (s["n_bench_error"] + s["n_bench_ambig"])
            if (s["n_bench_error"] + s["n_bench_ambig"]) else None
        )
        rows.append({
            "source": src,
            "n_pred": s["n_pred"],
            "n_pred_in_bench": s["n_pred_lu"],
            "n_bench_error": s["n_bench_error"],
            "n_bench_ambiguous": s["n_bench_ambig"],
            "n_bench_clean": s["n_bench_clean"],
            "TP_strict": s["tp_strict"],
            "TP_lenient": s["tp_lenient"],
            "precision_lu_strict": prec_strict,
            "precision_lu_lenient": prec_lenient,
            "recall_strict": rec_strict,
            "recall_lenient": rec_lenient,
        })
    return rows


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _short(s: str, n: int = 80) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _examples_md_block(name: str, items: List[Dict]) -> str:
    if not items:
        return f"\n_None._\n"
    lines = ["", "| source | row | column | value | rationale / corrected |", "| --- | --- | --- | --- | --- |"]
    for it in items:
        rat = it.get("rationale") or it.get("benchmark_corrected_value") or ""
        if "predicted_correction" in it and it.get("predicted_correction"):
            rat = f"→ {it['predicted_correction']} ({it.get('rationale','')})"
        lines.append("| " + " | ".join([
            _short(it.get("source", ""), 28),
            str(it.get("row", "")),
            _short(it.get("column", ""), 28),
            "`" + _short(it.get("original_value", ""), 50).replace("`", "'") + "`",
            _short(rat, 90),
        ]) + " |")
    return "\n" + "\n".join(lines) + "\n"


def write_report(
    run_dir: Path,
    metrics_by_type: Dict[str, Dict],
    *,
    benchmark_paths: Dict[str, Path],
    coverage_by_type: Optional[Dict[str, Dict]] = None,
    examples_by_type: Optional[Dict[str, Dict[str, List[Dict]]]] = None,
) -> Path:
    """Cross-type human-readable Markdown report."""
    lines: List[str] = []
    lines.append("# Precision-baseline experiment report\n")
    lines.append(f"Run dir: `{run_dir.name}`\n")
    lines.append("\n## Configuration\n")
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = summary.get("config", {})
        stats = summary.get("stats", {})
        lines.append(f"- source_dir: `{summary.get('source_dir')}`")
        lines.append(f"- csv_count: {summary.get('csv_count')}")
        lines.append(f"- error_types: {cfg.get('error_types')}")
        lines.append(f"- models: {cfg.get('models')}")
        lines.append(f"- reasoning_effort: {cfg.get('reasoning_effort')}")
        lines.append(f"- concurrency: {cfg.get('concurrency')}")
        lines.append(f"- prompt_version: {cfg.get('prompt_version')}")
        if stats:
            elapsed = stats.get("elapsed_s")
            elapsed_str = f"{elapsed:.1f}" if isinstance(elapsed, (int, float)) else "—"
            lines.append(
                f"- n_jobs: {stats.get('n_jobs')}; "
                f"n_cached: {stats.get('n_cached')}; "
                f"n_errors: {stats.get('n_errors')}; "
                f"elapsed_s: {elapsed_str}"
            )
        lines.append("")
    lines.append("\n## Benchmark sources\n")
    for et in sorted(benchmark_paths):
        lines.append(f"- **{et}**: `{benchmark_paths[et].name}`")
    lines.append("")

    lines.append("\n## Headline metrics per error type\n")
    lines.append(
        "Three views per type. Strict counts only `label='error'` benchmark cells "
        "as ground-truth positives; lenient also accepts `label='ambiguous'`. "
        "Labeled-universe restricts the precision denominator to predictions on "
        "cells that appear in the benchmark at all (excludes unmeasured predictions "
        "outside the benchmark)."
    )
    lines.append("")
    header = (
        "| error_type | n_pred | n_pred ∈ bench | bench(err/ambig/clean) | "
        "strict P | strict R | strict F1 | "
        "lenient P | lenient R | lenient F1 | "
        "LU strict P | LU lenient P |"
    )
    sep = "| " + " | ".join(["---"] * 12) + " |"
    lines.append(header)
    lines.append(sep)
    for et in sorted(metrics_by_type):
        m = metrics_by_type[et]
        lu = m["labeled_universe"]
        lines.append("| " + " | ".join([
            et,
            str(m["n_predictions"]),
            str(m["n_predictions_in_labeled_universe"]),
            f"{m['n_benchmark_error']}/{m['n_benchmark_ambiguous']}/{m['n_benchmark_clean']}",
            _fmt(m["strict"]["precision"]),
            _fmt(m["strict"]["recall"]),
            _fmt(m["strict"]["f1"]),
            _fmt(m["lenient"]["precision"]),
            _fmt(m["lenient"]["recall"]),
            _fmt(m["lenient"]["f1"]),
            _fmt(lu["strict_precision"]),
            _fmt(lu["lenient_precision"]),
        ]) + " |")
    lines.append("")

    lines.append("\n## Confusion counts per error type\n")
    lines.append("| error_type | view | TP | FP | FN |")
    lines.append("| --- | --- | --- | --- | --- |")
    for et in sorted(metrics_by_type):
        m = metrics_by_type[et]
        lines.append(f"| {et} | strict | {m['strict']['TP']} | {m['strict']['FP']} | {m['strict']['FN']} |")
        lines.append(f"| {et} | lenient | {m['lenient']['TP']} | {m['lenient']['FP']} | {m['lenient']['FN']} |")
        lu = m["labeled_universe"]
        lines.append(f"| {et} | LU strict | {lu['TP_strict']} | {lu['FP_strict']} | — |")
        lines.append(f"| {et} | LU lenient | {lu['TP_lenient']} | {lu['FP_lenient']} | — |")
    lines.append("")

    # Coverage diagnostic per type
    if coverage_by_type:
        lines.append("\n## Recall attainability (coverage diagnostic)\n")
        lines.append(
            "Splits the benchmark errors by whether the baseline even "
            "made a prediction for the column. Errors in *unseen* columns "
            "are upstream-missed (the LLM call was either content-filter-"
            "rejected or returned an empty list); errors in *seen* "
            "columns that the LLM did not flag are conservative-prompt "
            "misses."
        )
        lines.append("")
        lines.append(
            "| error_type | n_cols_seen | errors in seen | errors in unseen | ambig in seen | ambig in unseen |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for et in sorted(coverage_by_type):
            c = coverage_by_type[et]
            lines.append("| " + " | ".join([
                et,
                str(c["n_columns_seen_by_baseline"]),
                str(c["errors_in_seen_columns"]),
                str(c["errors_in_unseen_columns"]),
                str(c["ambiguous_in_seen_columns"]),
                str(c["ambiguous_in_unseen_columns"]),
            ]) + " |")
        lines.append("")

    # Sample findings per type
    if examples_by_type:
        for et in sorted(examples_by_type):
            ex = examples_by_type[et]
            lines.append(f"\n## Examples — {et}\n")
            lines.append("### True positives (strict; matched bench label=error)")
            lines.append(_examples_md_block("tp_strict", ex.get("tp_strict", [])))
            lines.append("### Soft TPs (lenient; matched bench label=ambiguous)")
            lines.append(_examples_md_block("tp_lenient_ambig", ex.get("tp_lenient_ambig", [])))
            lines.append("### False positives (predicted but bench labeled clean)")
            lines.append(_examples_md_block("fp_in_labeled", ex.get("fp_in_labeled", [])))
            lines.append("### Unmeasured predictions (not in benchmark — judge manually)")
            lines.append(_examples_md_block("fp_outside_labeled", ex.get("fp_outside_labeled", [])))
            lines.append("### False negatives (bench label=error not predicted)")
            lines.append(_examples_md_block("fn_error", ex.get("fn_error", [])))
            lines.append("### Lenient false negatives (bench label=ambiguous not predicted)")
            lines.append(_examples_md_block("fn_ambig", ex.get("fn_ambig", [])))

    lines.append("\n## How to read the columns\n")
    lines.append("""
- **n_pred**: total cells the baseline flagged for this error type
  across the whole corpus.
- **n_pred ∈ bench**: how many of those flagged cells appear in the
  benchmark JSONL. Predictions outside the benchmark are unmeasured.
- **bench(err/ambig/clean)**: count of benchmark cells by label.
- **strict P / R / F1**: precision and recall over all predictions and
  the benchmark's `label='error'` set only.
- **lenient P / R / F1**: same denominator but counts ambiguous cells
  as positives too (a sanity-floor for precision when the labeler was
  uncertain).
- **LU strict P / LU lenient P**: precision computed over the
  *predictions-in-labeled-universe* subset only. This is the
  cleanest precision read because every denominator cell has a
  ground-truth label. Use this as the primary precision number when
  comparing against benchmarks that are not complete coverage.
""".strip())

    out_path = run_dir / "precision_baseline_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path,
                   help="Output dir from experiments.run_precision_baseline.")
    p.add_argument("--benchmark-dir", type=Path,
                   default=PROJECT_ROOT / "benchmark",
                   help="Dir containing benchmark_<error_type>_*.jsonl files.")
    p.add_argument("--error-types", type=str,
                   default=",".join(DEFAULT_ERROR_TYPES))
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        logger.error("run dir not found: %s", run_dir)
        return 2

    bench_dir = args.benchmark_dir.expanduser().resolve()
    if not bench_dir.is_dir():
        logger.error("benchmark dir not found: %s", bench_dir)
        return 2

    error_types = [s.strip() for s in args.error_types.split(",") if s.strip()]
    metrics_by_type: Dict[str, Dict] = {}
    benchmark_paths: Dict[str, Path] = {}
    coverage_by_type: Dict[str, Dict] = {}
    examples_by_type: Dict[str, Dict[str, List[Dict]]] = {}

    scoring_dir = run_dir / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)

    for et in error_types:
        pred_path = run_dir / f"predictions_{et}.jsonl"
        if not pred_path.is_file():
            logger.warning("no predictions_%s.jsonl in %s; skipping",
                           et, run_dir)
            continue
        bench_path = _find_benchmark_file(bench_dir, et)
        if bench_path is None:
            logger.warning("no benchmark_%s_*.jsonl in %s; skipping",
                           et, bench_dir)
            continue
        benchmark_paths[et] = bench_path
        bench = load_benchmark(bench_path)
        preds = load_predictions(pred_path)
        pred_keys = predictions_to_keys(preds)
        metrics = score_one(et, pred_keys, bench)
        metrics_by_type[et] = metrics
        (scoring_dir / f"{et}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Coverage diagnostic
        coverage_by_type[et] = coverage_diagnostic(pred_keys, bench)
        # Sample TP/FP/FN cells per bucket
        examples_by_type[et] = categorize_examples(pred_keys, bench)
        (scoring_dir / f"examples_{et}.json").write_text(
            json.dumps(examples_by_type[et], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Per-source breakdown CSV
        rows = per_source_breakdown(pred_keys, bench)
        per_src_path = scoring_dir / f"per_source_{et}.csv"
        with per_src_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "source", "n_pred", "n_pred_in_bench",
                "n_bench_error", "n_bench_ambiguous", "n_bench_clean",
                "TP_strict", "TP_lenient",
                "precision_lu_strict", "precision_lu_lenient",
                "recall_strict", "recall_lenient",
            ])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info(
            "Scored %s: n_pred=%d strict_P=%.3f lenient_P=%.3f LU_strict_P=%s LU_lenient_P=%s",
            et,
            metrics["n_predictions"],
            metrics["strict"]["precision"] or 0.0,
            metrics["lenient"]["precision"] or 0.0,
            _fmt(metrics["labeled_universe"]["strict_precision"]),
            _fmt(metrics["labeled_universe"]["lenient_precision"]),
        )

    if not metrics_by_type:
        logger.error("no error types scored; check run_dir and benchmark_dir")
        return 2

    report_path = write_report(run_dir, metrics_by_type,
                               benchmark_paths=benchmark_paths,
                               coverage_by_type=coverage_by_type,
                               examples_by_type=examples_by_type)
    logger.info("Wrote report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
