"""Score a detection run against the synthetic-injection manifest.

Workflow
--------
1. Load ``injection_manifest.jsonl`` (the ground truth).
2. Load ``issues.jsonl`` from the detection run dir.
3. Align by ``(source, row, column)``. Compute per-error-type
   precision / recall / F1 plus a confusion matrix at the error-type
   level (which type the LLM assigned vs. which type was actually
   injected).
4. Emit ``benchmark_score.json`` and ``benchmark_score.md``.

Usage::

    python -m experiments.score_synth_benchmark \\
        --benchmark-dir outputs/synth_271841717 \\
        --run-dir outputs/20260425T120000Z_gpt-5.2 \\
        --output-dir outputs/synth_271841717/score_gpt5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("score_synth_benchmark")

CellKey = Tuple[str, int, str]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _normalize_source(s: str) -> str:
    """Strip a trailing ``.csv`` so manifest (source = stem) and run
    artifacts (source = filename) align."""
    if s.endswith(".csv"):
        return s[:-4]
    return s


def _issue_cells(issue: Mapping[str, Any]) -> List[CellKey]:
    """A single flat issue may carry one column (``column``) or many
    (``columns``). Return one cell key per column so multi-column
    issues credit each touched column."""
    source = issue.get("source") or issue.get("table") or issue.get("file")
    row = issue.get("row")
    if row is None or source is None:
        return []
    try:
        row_i = int(row)
    except (TypeError, ValueError):
        return []
    cols: List[str] = []
    if "column" in issue and issue["column"]:
        cols.append(str(issue["column"]))
    if "columns" in issue and issue["columns"]:
        cols.extend(str(c) for c in issue["columns"])
    if not cols:
        return []
    src = _normalize_source(str(source))
    return [(src, row_i, c) for c in cols]


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def score(
    *,
    benchmark_dir: Path,
    run_dir: Path,
) -> Dict[str, Any]:
    manifest = _load_jsonl(benchmark_dir / "injection_manifest.jsonl")
    issues = _load_jsonl(run_dir / "issues.jsonl")

    # Ground truth: cell -> injected error_type.
    truth_by_cell: Dict[CellKey, str] = {}
    for rec in manifest:
        key = (_normalize_source(rec["source"]),
               int(rec["row"]), rec["column"])
        # If multiple injections collide on a cell, keep the first
        # (builder serialises injectors but a defensive guard is cheap).
        truth_by_cell.setdefault(key, rec["error_type"])
    truth_cells: Set[CellKey] = set(truth_by_cell)

    # Predicted: cell -> set of error_types from the run.
    pred_by_cell: Dict[CellKey, Set[str]] = defaultdict(set)
    pred_cells_total: Set[CellKey] = set()
    for issue in issues:
        for key in _issue_cells(issue):
            pred_cells_total.add(key)
            et = issue.get("error_type") or "unknown"
            pred_by_cell[key].add(et)

    # Per-error-type metrics:
    # tp(t) = #cells where injected==t and predicted has any issue
    #         AND predicted set contains t.
    # fn(t) = #cells where injected==t and predicted has no t.
    # fp(t) = #cells where predicted has t but injected!=t (or
    #         injected nothing - i.e. cell was clean).
    types = sorted({et for et in truth_by_cell.values()} |
                   {et for ets in pred_by_cell.values() for et in ets})
    per_type: Dict[str, Dict[str, Any]] = {}
    for t in types:
        tp = fp = fn = 0
        for cell, injected_t in truth_by_cell.items():
            preds = pred_by_cell.get(cell, set())
            if injected_t == t:
                if t in preds:
                    tp += 1
                else:
                    fn += 1
        # FP: any predicted t on a cell where the truth is not t.
        for cell, preds in pred_by_cell.items():
            if t not in preds:
                continue
            injected_t = truth_by_cell.get(cell)
            if injected_t != t:
                fp += 1
        p, r, f = _prf(tp, fp, fn)
        per_type[t] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "n_injected": sum(1 for v in truth_by_cell.values() if v == t),
        }

    # Cell-level (any-issue-on-injected-cell) recall - useful as a
    # ceiling since the LLM might detect the cell but mis-label it.
    cells_caught_any = sum(
        1 for cell in truth_cells if cell in pred_by_cell
    )
    cell_recall_any = cells_caught_any / len(truth_cells) if truth_cells else 0.0

    # Confusion matrix: injected_type -> predicted_type counts.
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for cell, injected_t in truth_by_cell.items():
        preds = pred_by_cell.get(cell)
        if not preds:
            confusion[injected_t]["__missed__"] += 1
        else:
            for et in preds:
                confusion[injected_t][et] += 1

    return {
        "benchmark_dir": str(benchmark_dir),
        "run_dir": str(run_dir),
        "n_injections": len(manifest),
        "n_unique_injected_cells": len(truth_cells),
        "n_predicted_cells": len(pred_cells_total),
        "cell_level_recall_any_issue": round(cell_recall_any, 4),
        "per_error_type": per_type,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Synthetic benchmark score\n")
    lines.append(f"- Benchmark: `{report['benchmark_dir']}`")
    lines.append(f"- Run: `{report['run_dir']}`")
    lines.append(f"- Injections: **{report['n_injections']}** "
                 f"({report['n_unique_injected_cells']} unique cells)")
    lines.append(f"- Cells flagged by run: **{report['n_predicted_cells']}**")
    lines.append(
        f"- Cell-level recall (any issue on injected cell): "
        f"**{report['cell_level_recall_any_issue']:.2f}**"
    )
    lines.append("")
    lines.append("## Per-error-type precision / recall\n")
    lines.append("| error_type | n_injected | TP | FP | FN | precision | recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t, m in sorted(report["per_error_type"].items()):
        lines.append(
            f"| `{t}` | {m['n_injected']} | {m['tp']} | {m['fp']} | "
            f"{m['fn']} | {m['precision']:.2f} | {m['recall']:.2f} | "
            f"{m['f1']:.2f} |"
        )
    lines.append("")
    lines.append("## Confusion matrix\n")
    lines.append("Rows = injected type, columns = predicted type "
                 "(`__missed__` = cell not flagged at all).\n")
    confusion = report["confusion"]
    if confusion:
        cols = sorted({c for row in confusion.values() for c in row})
        lines.append("| injected \\ predicted | " + " | ".join(f"`{c}`" for c in cols) + " |")
        lines.append("|---" + "|---:" * len(cols) + "|")
        for inj_t in sorted(confusion):
            row = confusion[inj_t]
            cells = " | ".join(str(row.get(c, 0)) for c in cols)
            lines.append(f"| `{inj_t}` | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    report = score(
        benchmark_dir=args.benchmark_dir.resolve(),
        run_dir=args.run_dir.resolve(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark_score.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "benchmark_score.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    logger.info("wrote benchmark score to %s", args.output_dir)


if __name__ == "__main__":
    main()
