"""Compare an LLM run against a heuristic run on the same CSVs.

The heuristic run is treated as a precise-but-narrow oracle. Per
``(source, row, column, error_type_bucket)`` we compute:

* ``H \\ LLM`` - cells the heuristic flagged that the LLM did not.
  These are the **false-negative candidates** of the LLM.
* ``LLM \\ H`` - cells the LLM flagged that the heuristic did not.
  Heuristics are coverage-narrow, so this set is mostly real LLM
  hits the heuristic can't see (semantic / cross-table errors). It
  is **not** a false-positive estimate.
* ``LLM \\u2229 H`` - both agreed. Effectively guaranteed-true positives
  (assuming the heuristic has high precision on its slice).

Per error-type bucket we report:

* heuristic-derived **FN floor** = ``|H \\ LLM| / |H|`` - a lower bound
  on the LLM's FN rate, calibrated by the heuristic's precision (set
  via ``--heuristic-precision``; default 0.9).
* LLM-on-heuristic recall = ``|LLM \\u2229 H| / |H|`` - what fraction of
  the heuristic's hits the LLM also caught.

Outputs ``comparison.json`` (for downstream tooling), a Markdown
summary, and an optional candidates JSONL for hand-labelling FN
candidates in the labeling UI.

Usage::

    python -m experiments.compare_runs \\
        --llm-run outputs/20260424T193234Z_gpt-5.2 \\
        --heuristic-run outputs/20260424T235931Z_heuristic \\
        --output-dir outputs/comparison_154329618
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("compare_runs")


# ---------------------------------------------------------------------------
# Bucketing: collapse the LLM's 9-type taxonomy onto buckets the
# compare-runs panel can usefully diff. The five legacy
# ``*_inconsistency`` keys (text / casing / pattern / punctuation /
# number_format) were merged into a single ``inconsistency`` umbrella
# at the prompt level, so there is no longer a deliberate fold to
# apply here - every emitted ``error_type`` maps to itself. Anything
# not in this map keeps its raw ``error_type`` string and ends up in
# a per-type bucket - which is fine; we just want every error_type
# the detectors can emit to be VISIBLE in the bucket overlap table.
# ---------------------------------------------------------------------------


ERROR_TYPE_BUCKETS = {
    # ---- single-column ---------------------------------------------
    "typo": "typo",
    "inconsistency": "inconsistency",
    "semantic_incompatible": "semantic_incompatible",
    "numeric_outlier": "numeric_outlier",
    # ---- multi-column (intra-row) ----------------------------------
    "fd_violation": "fd_violation",
    "row_calculation_inconsistency": "row_calculation_inconsistency",
    "string_calculation_inconsistency": "string_calculation_inconsistency",
    # ---- multi-row (intra-table) -----------------------------------
    "row_duplication": "row_duplication",
    # ---- multi-table -----------------------------------------------
    "inter_table_inconsistency": "inter_table_inconsistency",
}


def _bucket(error_type: str) -> str:
    return ERROR_TYPE_BUCKETS.get(error_type, error_type or "unknown")


# Error-type buckets that at least one heuristic targets. Used by the
# LLM-vs-heuristic ``compare`` (recall + FN-floor) report; buckets
# outside this set get an ``n/a`` recall row in the markdown table.
# Keep this in sync with the detectors under ``src/heuristics/`` and
# ``src/core/row_duplication_detector.py``.
HEURISTIC_BUCKETS = {
    "typo", "inconsistency",
    "numeric_outlier", "fd_violation",
    "row_duplication",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


from src.core.jsonl_io import read_jsonl as _read_jsonl  # noqa: E402


def _load_run_issues(run_dir: Path) -> List[Dict[str, Any]]:
    """Read flat issues from a run dir.

    Both the LLM runner (``experiments.run_cleaning_detection``) and
    the heuristic runner write ``issues.jsonl`` with a compatible
    flat-issue shape, so a single loader is enough.
    """
    p = run_dir / "issues.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return _read_jsonl(p)


# ---------------------------------------------------------------------------
# Indexing: cell-level keyed sets
# ---------------------------------------------------------------------------


CellKey = Tuple[str, int, str]  # (source, row, folded_column)

# Sentinel column key for whole-row issues (e.g. ``row_duplication``).
# row-level issues have ``columns: []`` in ``issues.jsonl``; rather
# than dropping them silently, we slot each into a virtual cell so
# the per-bucket overlap table and the diff list can show them. The
# label is intentionally non-empty + non-numeric so it never
# collides with a real column header (``_fold_col`` keeps the
# parentheses, which no real column header carries).
WHOLE_ROW_COL = "(whole row)"


def _fold_col(col: str) -> str:
    """Canonical column key for cross-run comparison.

    Folds case and whitespace (incl. embedded ``\\n`` from multi-line
    headers) so a normalized run flagging ``"state"`` aligns with a
    raw run flagging ``"State"`` or ``"State\\nName"``. Without this
    fold the per-cell diff would treat them as different cells and
    every normalized-vs-raw comparison would be 100% disagreement.
    """
    return " ".join(str(col or "").lower().split())


def _index_cells(
    issues: List[Dict[str, Any]],
) -> Tuple[Dict[CellKey, Dict[str, List[Dict[str, Any]]]], Dict[CellKey, str]]:
    """Group issues by ``(source, row, folded_column)``, then by bucket.

    A single cell can carry multiple distinct error types (the LLM
    side does this routinely - missing_value + violated_attribute_dependency
    on the same blank). Bucketing by error type lets us measure
    per-type recall instead of mixing classes.

    Returns
    -------
    (index, display_columns)
        ``index`` is keyed by the case+whitespace-folded column so a
        normalized run aligns with a raw run; ``display_columns`` maps
        each folded key back to the first human-readable column name
        seen, so callers can render readable labels in diffs.
    """
    out: Dict[CellKey, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    display: Dict[CellKey, str] = {}
    for it in issues:
        source = str(it.get("source", "") or "")
        row = it.get("row")
        if not isinstance(row, int):
            continue
        cols = it.get("columns") or []
        bucket_key = _bucket(str(it.get("error_type", "") or ""))
        if not cols:
            # Whole-row issue (e.g. row_duplication). Slot under a
            # sentinel column so the per-bucket overlap and diff list
            # surface it instead of silently dropping. ``row`` is the
            # primary row for this issue; the duplicate group as a
            # whole gets one entry per row in ``affected_rows`` (each
            # of which is emitted as its own issue by the detector,
            # so we don't need to re-expand here).
            key = (source, int(row), _fold_col(WHOLE_ROW_COL))
            out[key][bucket_key].append(it)
            display.setdefault(key, WHOLE_ROW_COL)
            continue
        for col in cols:
            col_s = str(col)
            key = (source, int(row), _fold_col(col_s))
            out[key][bucket_key].append(it)
            display.setdefault(key, col_s)
    return out, display


# ---------------------------------------------------------------------------
# Per-bucket overlap arithmetic
# ---------------------------------------------------------------------------


def _bucket_membership(
    indexed: Dict[CellKey, Dict[str, List[Dict[str, Any]]]],
    bucket: str,
) -> set[CellKey]:
    return {k for k, by_bucket in indexed.items() if bucket in by_bucket}


def _all_buckets(*indexed: Dict[CellKey, Dict[str, List[Dict[str, Any]]]]) -> List[str]:
    seen: set[str] = set()
    for ix in indexed:
        for by_bucket in ix.values():
            seen.update(by_bucket)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def compare(
    llm_run_dir: Path,
    heur_run_dir: Path,
    *,
    heuristic_precision: float = 0.9,
) -> Dict[str, Any]:
    llm_issues = _load_run_issues(llm_run_dir)
    heur_issues = _load_run_issues(heur_run_dir)
    llm_idx, _ = _index_cells(llm_issues)
    heur_idx, _ = _index_cells(heur_issues)

    # Per-bucket recall + FN-floor table.
    rows: List[Dict[str, Any]] = []
    fn_candidates: List[Dict[str, Any]] = []
    for bucket in sorted(set(_all_buckets(llm_idx, heur_idx))):
        H = _bucket_membership(heur_idx, bucket)
        L = _bucket_membership(llm_idx, bucket)
        # Cell-level intersection at the bucket level. We treat
        # "LLM said *anything* in this bucket on this cell" as a hit.
        inter = H & L
        only_h = H - L
        only_l = L - H
        if H:
            recall = len(inter) / len(H)
            fn_floor = (len(only_h) / len(H)) * heuristic_precision
        else:
            recall = None
            fn_floor = None
        rows.append({
            "error_type_bucket": bucket,
            "heuristic_hits": len(H),
            "llm_hits": len(L),
            "intersection": len(inter),
            "heuristic_only": len(only_h),
            "llm_only": len(only_l),
            "llm_recall_on_heuristic": (
                round(recall, 3) if recall is not None else None
            ),
            # Lower bound on the LLM's miss rate AMONG cells of this
            # bucket the heuristic could see. Multiplied by the
            # caller's estimate of the heuristic's precision so we
            # don't penalise the LLM for heuristic FPs.
            "fn_floor_estimate": (
                round(fn_floor, 3) if fn_floor is not None else None
            ),
            "heuristic_in_scope": bucket in HEURISTIC_BUCKETS,
        })
        # Persist the FN candidates for hand-labelling.
        for key in sorted(only_h):
            src, row, col = key
            issues_for_cell = heur_idx[key][bucket]
            primary = issues_for_cell[0]
            fn_candidates.append({
                "source": src,
                "row": row,
                "column": col,
                "error_type_bucket": bucket,
                "heuristic_error_type": primary.get("error_type"),
                "detector": primary.get("detector"),
                "observed": primary.get("observed", ""),
                "explanation": primary.get("explanation", ""),
                "suggested_fix": primary.get("suggested_fix"),
                "severity": primary.get("severity"),
                "n_heuristic_signals": len(issues_for_cell),
            })

    # Top-level aggregates.
    in_scope_rows = [r for r in rows if r["heuristic_in_scope"] and r["heuristic_hits"]]
    avg_recall = (
        sum(r["llm_recall_on_heuristic"] for r in in_scope_rows) / len(in_scope_rows)
        if in_scope_rows else None
    )
    avg_fn_floor = (
        sum(r["fn_floor_estimate"] for r in in_scope_rows) / len(in_scope_rows)
        if in_scope_rows else None
    )

    return {
        "llm_run": str(llm_run_dir),
        "heuristic_run": str(heur_run_dir),
        "heuristic_precision_assumed": heuristic_precision,
        "n_llm_issues": len(llm_issues),
        "n_heuristic_issues": len(heur_issues),
        "by_bucket": rows,
        "summary": {
            "buckets_in_scope": len(in_scope_rows),
            "avg_llm_recall_on_heuristic": (
                round(avg_recall, 3) if avg_recall is not None else None
            ),
            "avg_fn_floor": (
                round(avg_fn_floor, 3) if avg_fn_floor is not None else None
            ),
        },
        "fn_candidates": fn_candidates,
    }


def _issue_summary(it: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact view of a single issue, suitable for the diff API."""
    return {
        "id": it.get("id"),
        "error_type": it.get("error_type"),
        "observed": it.get("observed", ""),
        "explanation": it.get("explanation", ""),
        "suggested_fix": it.get("suggested_fix"),
        "severity": it.get("severity"),
        "detector": it.get("detector"),
    }


def diff_runs(
    run_a_dir: Path,
    run_b_dir: Path,
    *,
    include_both: bool = False,
) -> Dict[str, Any]:
    """Generic cell-level bucket diff between two runs.

    Symmetric counterpart to :func:`compare`: both sides are treated
    equally (no precision-of-A-vs-B framing). Used by the labeling-UI
    ``Compare`` panel to focus the operator on cells where the two
    runs disagree at the bucket level (``a_only`` / ``b_only``).

    Parameters
    ----------
    run_a_dir, run_b_dir
        Paths to the two run directories. Each must have an
        ``issues.jsonl`` file produced by either the LLM detector or
        the heuristic runner.
    include_both
        If ``True``, also emit ``side="both"`` entries in
        ``diff_cells``. Default ``False`` because the per-bucket
        ``intersection`` count already covers agreement and the diff
        view is meant to surface *disagreements*.

    Returns
    -------
    dict with keys: ``run_a``, ``run_b``, ``n_a_issues``, ``n_b_issues``,
    ``sources`` (alignment summary), ``by_bucket`` (per-bucket counts +
    Jaccard), ``diff_cells`` (one entry per disagreeing cell).
    """
    a_issues = _load_run_issues(run_a_dir)
    b_issues = _load_run_issues(run_b_dir)
    a_idx, a_disp = _index_cells(a_issues)
    b_idx, b_disp = _index_cells(b_issues)

    # Source alignment so the UI can warn if the two runs targeted
    # different corpora (in which case the diff is meaningless).
    a_sources = {k[0] for k in a_idx}
    b_sources = {k[0] for k in b_idx}
    common = a_sources & b_sources
    a_total = a_sources | b_sources

    rows: List[Dict[str, Any]] = []
    for bucket in sorted(set(_all_buckets(a_idx, b_idx))):
        A = _bucket_membership(a_idx, bucket)
        B = _bucket_membership(b_idx, bucket)
        inter = A & B
        only_a = A - B
        only_b = B - A
        union = A | B
        rows.append({
            "bucket": bucket,
            "a_hits": len(A),
            "b_hits": len(B),
            "intersection": len(inter),
            "a_only": len(only_a),
            "b_only": len(only_b),
            "jaccard": (round(len(inter) / len(union), 3) if union else 0.0),
        })

    # Cell-level diff (the headline output for the UI). A cell is
    # considered "agreement" if BOTH runs flagged it at all - even if
    # they tagged it with different buckets - because the operator
    # cares about whether the cell was caught, not whether the two
    # runs agreed on the precise taxonomy label. Per-bucket stats
    # remain available in ``by_bucket`` above.
    diff_cells: List[Dict[str, Any]] = []
    all_keys = set(a_idx) | set(b_idx)
    for key in sorted(all_keys):
        src, row, _folded_col = key
        # Prefer the column name from whichever run flagged it (raw run
        # wins over normalized when both did, since raw preserves
        # casing); falls back to the folded key as a last resort.
        col = b_disp.get(key) or a_disp.get(key) or _folded_col
        a_by_bucket = a_idx.get(key, {})
        b_by_bucket = b_idx.get(key, {})
        a_hit = bool(a_by_bucket)
        b_hit = bool(b_by_bucket)
        if a_hit and b_hit:
            side = "both"
            if not include_both:
                continue
        elif a_hit:
            side = "a_only"
        else:
            side = "b_only"
        a_summaries = [
            _issue_summary(it)
            for issues in a_by_bucket.values()
            for it in issues
        ]
        b_summaries = [
            _issue_summary(it)
            for issues in b_by_bucket.values()
            for it in issues
        ]
        buckets = sorted(set(a_by_bucket) | set(b_by_bucket))
        diff_cells.append({
            "source": src,
            "row": row,
            "column": col,
            "buckets": buckets,
            # ``bucket`` kept for back-compat; comma-joined when a cell
            # spans multiple buckets in a single run.
            "bucket": ", ".join(buckets),
            "side": side,
            "a_issues": a_summaries,
            "b_issues": b_summaries,
        })

    n_diff = sum(1 for d in diff_cells if d["side"] != "both")
    return {
        "run_a": str(run_a_dir),
        "run_b": str(run_b_dir),
        "n_a_issues": len(a_issues),
        "n_b_issues": len(b_issues),
        "sources": {
            "a_count": len(a_sources),
            "b_count": len(b_sources),
            "common": sorted(common),
            "a_only": sorted(a_sources - b_sources),
            "b_only": sorted(b_sources - a_sources),
            "alignment_jaccard": (
                round(len(common) / len(a_total), 3) if a_total else 1.0
            ),
        },
        "by_bucket": rows,
        "diff_cells": diff_cells,
        "n_diff_cells": n_diff,
    }


def diff_cell_detail(
    run_a_dir: Path,
    run_b_dir: Path,
    *,
    source: str,
    row: int,
    column: str,
) -> Dict[str, Any]:
    """Return all issues from both runs for a single cell.

    Used by the cell-detail drawer in the Compare panel: surfaces
    every issue (across all buckets) that either run emitted for the
    target cell, so the operator can see the full picture for one
    disagreement at a time.
    """
    a_issues = _load_run_issues(run_a_dir)
    b_issues = _load_run_issues(run_b_dir)
    a_idx, _ = _index_cells(a_issues)
    b_idx, _ = _index_cells(b_issues)
    key = (str(source), int(row), _fold_col(str(column)))
    a_hits: List[Dict[str, Any]] = []
    b_hits: List[Dict[str, Any]] = []
    for bucket, items in a_idx.get(key, {}).items():
        for it in items:
            payload = _issue_summary(it)
            payload["bucket"] = bucket
            a_hits.append(payload)
    for bucket, items in b_idx.get(key, {}).items():
        for it in items:
            payload = _issue_summary(it)
            payload["bucket"] = bucket
            b_hits.append(payload)
    return {
        "source": source,
        "row": row,
        "column": column,
        "a_issues": a_hits,
        "b_issues": b_hits,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# LLM vs heuristic comparison\n")
    lines.append(f"- LLM run: `{report['llm_run']}`")
    lines.append(f"- Heuristic run: `{report['heuristic_run']}`")
    lines.append(
        f"- LLM issues: **{report['n_llm_issues']}**, "
        f"heuristic issues: **{report['n_heuristic_issues']}**"
    )
    lines.append(
        f"- Heuristic precision assumed: **{report['heuristic_precision_assumed']:.2f}** "
        "(adjust with `--heuristic-precision`)"
    )
    s = report["summary"]
    if s["avg_llm_recall_on_heuristic"] is not None:
        lines.append(
            f"- Avg LLM recall over heuristic-in-scope buckets: "
            f"**{s['avg_llm_recall_on_heuristic']:.2f}**"
        )
        lines.append(
            f"- Avg FN-rate floor: **{s['avg_fn_floor']:.2f}** "
            "(`|H\\\\LLM|/|H|` weighted by heuristic precision)"
        )
    lines.append("")
    lines.append("| bucket | H | LLM | inter | H\\\\LLM | LLM\\\\H | recall | FN floor |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in report["by_bucket"]:
        recall = (
            f"{r['llm_recall_on_heuristic']:.2f}"
            if r["llm_recall_on_heuristic"] is not None else "n/a"
        )
        floor = (
            f"{r['fn_floor_estimate']:.2f}"
            if r["fn_floor_estimate"] is not None else "n/a"
        )
        oos = "" if r["heuristic_in_scope"] else " *(out of heuristic scope)*"
        lines.append(
            f"| `{r['error_type_bucket']}`{oos} "
            f"| {r['heuristic_hits']} | {r['llm_hits']} | {r['intersection']} "
            f"| {r['heuristic_only']} | {r['llm_only']} | {recall} | {floor} |"
        )
    lines.append("")
    lines.append(
        f"## FN candidates ({len(report['fn_candidates'])} cells)\n\n"
        "Cells flagged by a heuristic but missed by the LLM. Treat as "
        "hand-labelling backlog: each is either a real false negative "
        "(\u2192 fix the LLM) or a heuristic FP (\u2192 tighten the heuristic).\n"
    )
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm-run", type=Path, required=True)
    p.add_argument("--heuristic-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--heuristic-precision", type=float, default=0.9,
        help=("Calibration constant: assumed precision of the heuristic on "
              "its slice. The FN-floor estimate is multiplied by this. "
              "Default 0.9 reflects the conservative-tuning of the "
              "shipped heuristics."),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    report = compare(
        args.llm_run, args.heuristic_run,
        heuristic_precision=args.heuristic_precision,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )
    (args.output_dir / "comparison.md").write_text(
        _render_markdown(report), encoding="utf-8",
    )
    # FN candidates as JSONL for direct ingestion into the labeling UI.
    cand_path = args.output_dir / "fn_candidates.jsonl"
    with cand_path.open("w", encoding="utf-8") as fh:
        for c in report["fn_candidates"]:
            fh.write(json.dumps(c) + "\n")
    logger.info("wrote comparison report to %s", args.output_dir)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
