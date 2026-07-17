"""Per-cell comparison: GPT-5.5 rulebook findings vs human-labeled gold.

Scores the standalone GPT-5.5 rulebook detector (``run_on_dataset.py``
findings dump) against the human-curated column-cluster labels of a
workbench run, at the **cell** level (one unit per ``(source, row,
column)``).

Eval contract (first-pass, TP-focused)
--------------------------------------
  POSITIVES  = ``benchmark.jsonl`` cells with ``label == "error"``.
  IMMUNE     = excluded from scoring entirely:
                 * ``benchmark.jsonl`` cells with ``label == "ambiguous"``
                 * ``benchmark.skipped.jsonl`` cells (if present)
                 * every cell whose value sits in a human cluster verdict
                   of ``partial`` / ``unsure`` / ``skip``
  CLEAN      = everything else. No explicit labels: a GPT-5.5 flag here
               counts as a false positive.

  A GPT-5.5 finding ``(source, column, value)`` is expanded to every CSV
  row whose cell equals ``value`` (whitespace-normalised).

  TP = error cell that GPT-5.5 flagged
  FN = error cell GPT-5.5 missed
  FP = GPT-5.5-flagged clean cell (not error, not immune)
  precision = TP/(TP+FP)   recall = TP/(TP+FN)   F1 = 2PR/(P+R)

Usage::

    python -m RFT.inconsistency.v1.eval.eval_percell_vs_human \
        --run outputs/20260605T000013Z_model_06e51dd5 \
        --findings outputs/inconsistency_rulebook_gpt55/curated_pbi_500_dedup/pbi500_findings_46b4d9166ec0.jsonl \
        --tag pbi500
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.labeling.correction_suggester import (  # noqa: E402
    build_cluster_index,
    load_drafts,
    load_labels,
    resolve_csv_map,
)

IMMUNE_CLUSTER_STATUSES = {"partial", "unsure", "skip"}


def _norm(s) -> str:
    return " ".join(str(s).split())


def _alnum_fold(s: str) -> str:
    """Keep only alphanumerics, casefolded -- strips casing, punctuation
    and spacing so two strings match iff they differ ONLY by those."""
    return "".join(ch for ch in str(s).casefold() if ch.isalnum())


def _only_casing_punct_space(a: str, b: str) -> bool:
    """True iff a and b differ only by casing / punctuation / spacing
    (their alphanumeric content is identical, case-insensitively)."""
    return _alnum_fold(a) == _alnum_fold(b)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def _is_trivial_edit(value: str, corrected: str, max_dist: int) -> bool:
    """A trivial model edit: corrected differs from value only by
    casing/punctuation/**spacing** AND the edit distance is < ``max_dist``.

    Whitespace-only fixes (trailing/leading space, newline, non-breaking
    space) count as spacing edits -- they are trivial too, even though
    they collapse to equal under whitespace normalisation. Only a missing
    correction (empty string) is exempt: that is a flag with no proposed
    fix, not a trivial edit.
    """
    if not isinstance(corrected, str) or not corrected.strip():
        return False
    return (
        _only_casing_punct_space(value, corrected)
        and _levenshtein(value, corrected) < max_dist
    )


def _load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Workbench run dir (has benchmark.jsonl, labels.jsonl, "
                         "column_cluster_drafts.jsonl, summary.json).")
    ap.add_argument("--findings", required=True,
                    help="GPT-5.5 rulebook findings JSONL "
                         "(run_on_dataset.py output).")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-trivial-edits", action="store_true",
                    help="EVAL 2: drop model findings whose corrected_value "
                         "differs from the original only by casing/punctuation/"
                         "spacing AND has edit distance < --trivial-edit-max. "
                         "Findings that match a gold-error cell are exempt "
                         "(stay TP); only non-TP trivial findings are removed.")
    ap.add_argument("--trivial-edit-max", type=int, default=2,
                    help="Skip a trivial edit when its Levenshtein distance is "
                         "strictly less than this (default 2 -> distance 0/1).")
    ap.add_argument("--out", default="",
                    help="Output dir (default: <run>/percell_eval).")
    args = ap.parse_args()

    run = Path(args.run)
    out_dir = Path(args.out) if args.out else (run / "percell_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_map = resolve_csv_map(run)
    _df_cache: dict[str, pd.DataFrame | None] = {}
    _colmap_cache: dict[str, dict[str, str]] = {}

    def df_for(source: str) -> pd.DataFrame | None:
        if source not in _df_cache:
            p = csv_map.get(source)
            _df_cache[source] = _load_csv(p) if p else None
        return _df_cache[source]

    def resolve_col(source: str, column: str) -> str | None:
        """Map a (whitespace-normalised) column name to the actual CSV
        header, so a gold ``"KPI_Name"`` matches a raw header ``"KPI_Name "``.
        """
        if source not in _colmap_cache:
            df = df_for(source)
            _colmap_cache[source] = (
                {_norm(c): c for c in df.columns} if df is not None else {}
            )
        return _colmap_cache[source].get(_norm(column))

    # Cell keys are (source, row, NORMALISED column) so the gold,
    # findings, and cluster sides always agree on the column string.
    # ---- gold positives + benchmark-immune cells -------------------------
    gold_error: set[tuple[str, int, str]] = set()
    immune: set[tuple[str, int, str]] = set()
    bench_path = run / "benchmark.jsonl"
    for line in bench_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["source"], r.get("row"), _norm(r["column"]))
        if r.get("label") == "error":
            gold_error.add(key)
        elif r.get("label") == "ambiguous":
            immune.add(key)
    skip_path = run / "benchmark.skipped.jsonl"
    if skip_path.exists():
        for line in skip_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            immune.add((r["source"], r.get("row"), _norm(r["column"])))

    # ---- expand partial/unsure/skip clusters to immune cells -------------
    drafts = load_drafts(run / "column_cluster_drafts.jsonl")
    labels = load_labels(run / "labels.jsonl")
    cluster_index = build_cluster_index(drafts, labels)
    for (source, column), ctxs in cluster_index.items():
        immune_vals = {
            _norm(v)
            for ctx in ctxs
            if ctx.status in IMMUNE_CLUSTER_STATUSES
            for v in ctx.values
        }
        if not immune_vals:
            continue
        df = df_for(source)
        actual = resolve_col(source, column)
        if df is None or actual is None:
            continue
        for ridx, cell in enumerate(df[actual]):
            if _norm(cell) in immune_vals:
                immune.add((source, ridx, _norm(column)))
    # An error cell is always scored, even if some sibling cluster is immune.
    immune -= gold_error

    # ---- expand GPT-5.5 findings to flagged cells ------------------------
    flagged: set[tuple[str, int, str]] = set()
    trivial_flagged: set[tuple[str, int, str]] = set()
    findings_by_col: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    trivial_by_col: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    n_trivial_findings = 0
    for line in Path(args.findings).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        v = r.get("value")
        if not isinstance(v, str):
            continue
        nv = _norm(v)
        findings_by_col[(r["source"], r["column"])].add(nv)
        if args.skip_trivial_edits and _is_trivial_edit(
            v, r.get("corrected_value"), args.trivial_edit_max
        ):
            trivial_by_col[(r["source"], r["column"])].add(nv)
            n_trivial_findings += 1
    for (source, column), vals in findings_by_col.items():
        df = df_for(source)
        actual = resolve_col(source, column)
        if df is None or actual is None:
            continue
        tvals = trivial_by_col.get((source, column), set())
        for ridx, cell in enumerate(df[actual]):
            ncell = _norm(cell)
            if ncell in vals:
                key = (source, ridx, _norm(column))
                flagged.add(key)
                if ncell in tvals:
                    trivial_flagged.add(key)

    # ---- score -----------------------------------------------------------
    tp_cells = gold_error & flagged
    fn_cells = gold_error - flagged
    # EVAL 2: drop trivial-edit flags from the FP set (gold matches already
    # excluded by -gold_error, so they stay TP -- non-TP trivials are skipped).
    fp_cells = flagged - gold_error - immune - trivial_flagged

    TP, FN, FP = len(tp_cells), len(fn_cells), len(fp_cells)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0

    print("=" * 64)
    mode = "EVAL 2 (skip trivial edits)" if args.skip_trivial_edits else "EVAL 1"
    print(f"PER-CELL {mode}  run={run.name}  tag={args.tag or '-'}")
    print(f"  gold error cells   : {len(gold_error)}")
    print(f"  immune cells        : {len(immune)} "
          f"(ambiguous + partial/unsure/skip)")
    print(f"  GPT-5.5 flagged cells: {len(flagged)}")
    if args.skip_trivial_edits:
        print(f"  trivial findings dropped: {n_trivial_findings} "
              f"-> {len(trivial_flagged)} cells "
              f"({len(trivial_flagged - gold_error)} removed from FP, "
              f"edit-dist < {args.trivial_edit_max})")
    print("-" * 64)
    print(f"  TP={TP}  FN={FN}  FP={FP}")
    print(f"  recall    = {R:.3f}  ({TP}/{TP + FN})")
    print(f"  precision = {P:.3f}  ({TP}/{TP + FP})")
    print(f"  F1        = {F1:.3f}")
    print("=" * 64)

    tag = (args.tag + "_") if args.tag else ""
    summary = {
        "run": run.name, "tag": args.tag,
        "skip_trivial_edits": args.skip_trivial_edits,
        "trivial_edit_max": args.trivial_edit_max,
        "n_trivial_findings_dropped": n_trivial_findings,
        "n_trivial_fp_cells_removed": len(trivial_flagged - gold_error),
        "findings": str(args.findings),
        "n_gold_error_cells": len(gold_error),
        "n_immune_cells": len(immune),
        "n_flagged_cells": len(flagged),
        "TP": TP, "FN": FN, "FP": FP,
        "recall": R, "precision": P, "f1": F1,
    }
    (out_dir / f"{tag}summary.json").write_text(json.dumps(summary, indent=2))

    def _dump(name, cells):
        with (out_dir / f"{tag}{name}.jsonl").open("w") as f:
            for s, row, c in sorted(cells, key=lambda x: (x[0], x[2], x[1] or -1)):
                f.write(json.dumps({"source": s, "row": row, "column": c}) + "\n")

    _dump("fn", fn_cells)
    _dump("fp", fp_cells)
    _dump("tp", tp_cells)
    print(f"artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
