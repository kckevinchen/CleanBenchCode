"""
Compare external typo-detector run(s) against the labeled typo benchmark
(label_result/benchmark_typo_*.jsonl).

The external TSV is already pre-scored for precision; this script adds recall,
F1, and correction-quality, plus per-source breakdowns and reviewable CSVs of
the true positives / false positives / missed errors.

Cell-level matching uses the same normalization as the existing scoring:
lowercase, strip non-word chars, drop digits/underscores. This matches the
"normalization_key" idea in external/find_errors_by_normalization.py.

``--predictions`` accepts either a single TSV (e.g. external/run_1.scoring.tsv)
or a directory of TSVs (e.g. external/GPT_score/). In directory mode every
matching file is scored independently and a top-level ``summary.tsv`` collates
the per-run metrics, analogous to external/GPT_score/summary.tsv but computed
against our labeled benchmark.

Usage (single file):
    python -m experiments.compare_external_typo_run \\
        --benchmark label_result/benchmark_typo_20260507T215150Z_gpt5-family.jsonl \\
        --predictions external/run_1.scoring.tsv \\
        --out-dir outputs/external_run_1_vs_benchmark

Usage (batch over external/GPT_score):
    python -m experiments.compare_external_typo_run \\
        --benchmark label_result/benchmark_typo_20260507T215150Z_gpt5-family.canonical.jsonl \\
        --predictions external/GPT_score \\
        --out-dir outputs/external_gpt_score_vs_benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Match key
# ---------------------------------------------------------------------------
#
# Cells are matched on ``(source, column, normalized(original_value))``
# where ``normalized`` collapses runs of whitespace to a single space and
# trims the edges (``" ".join(s.split())``). The verbatim value is
# carried by both sides:
#
#   * In the benchmark JSONL: ``original_value``
#   * In the scoring TSV:     ``predicted_original_value``
#
# Whitespace-only normalization keeps the original character content and
# case intact; the change just makes ``'Bradford  Forfiet'`` (double
# space, as seen verbatim in some bench cells) hash the same as
# ``'Bradford Forfiet'`` (single space, the form upstream extractors
# emit after their own whitespace normalization). No casefolding, no
# punctuation stripping -- the previous attempts at heavier
# normalization silently merged distinct cells (e.g. numeric-only
# columns collapsed to one key) and hid real disagreements.
#
# When an exact (whitespace-collapsed) value match fails, we fall back
# to a **substring** match within the same ``(source, column)``: a
# prediction whose normalized value is a substring of a bench cell's
# normalized value links to that bench cell. This recovers two upstream
# behaviours that otherwise leave preds unlinked:
#
#   * Partial-token extractions like ``'exisitng'`` (8 chars) where the
#     bench cell is the full sentence ``'Replacing exisitng water
#     line...'``.
#   * Truncated values where the upstream cut the prediction to a fixed
#     prefix length (e.g. ``'RIPLEY-UNION-LEWIS-HUNTINGTON LO'`` vs the
#     full ``'RIPLEY-UNION-LEWIS-HUNTINGTON LOCAL'``).
#
# Substring fallback only fires when the pred normalizes to at least
# ``_MIN_SUBSTRING_LEN`` characters (default 4) to avoid spurious
# matches from short tokens. The grid surfaces a ``match_method`` flag
# (``"exact"`` or ``"substring"``) on every prediction so reviewers can
# spot fallbacks at a glance.
#
# The upstream-emitted ``benchmark_rows`` field is *not* used as the join
# key (its ``result_row_idx`` cousin is an upstream LLM-call counter, not a
# CSV row index, and ``benchmark_rows`` itself is sometimes empty when the
# upstream's value-matcher couldn't link a partial token). We still carry
# ``benchmark_rows`` through to the UI for transparency, but our own
# join is the value match described above.
#
# Consequences worth noting:
#
#   1. Distinct cells that genuinely share the same string value (e.g.
#      100 NHS rows that all read ``'Bradford Forfiet'``) collapse onto
#      one match key -- a single LLM prediction therefore links to all of
#      them. ``compare`` deliberately counts each prediction as ONE
#      outcome for precision so we don't inflate FP/TP by the number of
#      duplicate cells; recall, however, still credits every matched
#      bench-error row so it isn't artificially deflated.
#   2. The grid view fans each prediction out across every benchmark row
#      sharing its value, so every bench cell is independently visible
#      even when many share a value.
#   3. Predictions whose ``predicted_original_value`` doesn't appear in
#      any bench row (in-scope source/column), either exact or as a
#      substring, show up as ``unmatched_prediction`` in the grid and
#      ``no_value_match`` in the excluded list.


_MIN_SUBSTRING_LEN = 4  # min chars in normalized pred value for substring fallback


def _parse_benchmark_rows(raw: str | None) -> list[int]:
    """Parse the upstream ``benchmark_rows`` field into a list of ints.

    Informational only -- carried through to the UI as a transparency
    signal showing what the upstream's value-matcher claimed; the
    comparator's own join uses :func:`_match_pred` instead.
    """
    if not raw:
        return []
    s = str(raw).strip().strip("[]")
    if not s:
        return []
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


def _norm_for_match(value: str | None) -> str:
    """Collapse whitespace runs to a single space and trim edges. Used as
    the canonical form for the cell-value match key.
    """
    return " ".join((value or "").split())


def _build_bench_indexes(bench_rows: list[dict]) -> tuple[dict, dict]:
    """Return ``(exact_index, by_sc_index)``.

    * ``exact_index[(src, col, norm_value)] -> [bench_record, ...]``
      -- duplicates allowed (same value in multiple rows).
    * ``by_sc_index[(src, col)] -> [(norm_value, bench_record), ...]``
      -- used for the substring fallback (linear scan within one
      (source, column); cheap because each column has at most a few
      hundred labeled rows).
    """
    exact: dict[tuple, list[dict]] = defaultdict(list)
    by_sc: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
    for r in bench_rows:
        src = r.get("source") or ""
        col = r.get("column") or ""
        norm = _norm_for_match(r.get("original_value"))
        exact[(src, col, norm)].append(r)
        by_sc[(src, col)].append((norm, r))
    return exact, by_sc


def _match_pred(
    src: str | None,
    col: str | None,
    pred_value: str | None,
    exact_index: dict,
    by_sc_index: dict,
    *,
    min_substring_len: int = _MIN_SUBSTRING_LEN,
) -> tuple[list[dict], str]:
    """Find bench records linked to a prediction's value.

    Returns ``(matches, method)`` where ``method`` is one of:
      * ``"exact"``     -- whitespace-collapsed equality
      * ``"substring"`` -- normalized pred is a substring of a normalized
                           bench value (only attempted when exact fails
                           and the pred is at least ``min_substring_len``
                           characters)
      * ``"none"``      -- no match
    """
    norm = _norm_for_match(pred_value)
    if not norm:
        return [], "none"
    src = src or ""
    col = col or ""
    exact_matches = exact_index.get((src, col, norm), [])
    if exact_matches:
        return list(exact_matches), "exact"
    if len(norm) < min_substring_len:
        return [], "none"
    subs = [r for bnorm, r in by_sc_index.get((src, col), []) if norm in bnorm]
    return (subs, "substring") if subs else ([], "none")


def _find_csv_matches(
    frame,
    column: str | None,
    pred_value: str | None,
    *,
    min_substring_len: int = _MIN_SUBSTRING_LEN,
    allow_substring: bool = True,
) -> tuple[list[int], str]:
    """Return rows in ``frame[column]`` whose value matches ``pred_value``.

    Mirrors :func:`_match_pred` -- whitespace-collapsed exact equality
    first, substring fallback (``norm_pred in norm_cell``) when the
    pred is long enough. Used to enrich ``unmatched_prediction`` cells
    with the actual CSV row indices a reviewer could go label, turning
    "unmatched" rows into bench-coverage-gap candidates.

    Set ``allow_substring=False`` to disable the substring fallback --
    the bench-gap classifier uses this so a stray prediction like
    ``'ane'`` doesn't get promoted to "in_csv" just because some long
    cell happens to contain that trigram.

    ``frame`` is a pandas DataFrame (we duck-type the few attributes we
    need so we don't have to import pandas at module scope). Returns
    ``([], "none")`` if the column isn't present or the pred is empty.
    """
    if frame is None or not column:
        return [], "none"
    # ``getattr(..., []) or []`` would crash on a pandas Index because
    # its ``__bool__`` is ambiguous. List-coerce defensively first.
    raw_cols = getattr(frame, "columns", None)
    columns = list(raw_cols) if raw_cols is not None else []
    if column not in columns:
        return [], "none"
    norm_pred = _norm_for_match(pred_value)
    if not norm_pred:
        return [], "none"
    try:
        col_idx = columns.index(column)
    except ValueError:
        return [], "none"
    rows_exact: list[int] = []
    rows_substr: list[int] = []
    can_substr = allow_substring and len(norm_pred) >= min_substring_len
    n = len(frame)
    for i in range(n):
        try:
            v = frame.iat[i, col_idx]
        except Exception:  # noqa: BLE001
            continue
        if v is None:
            continue
        # Skip NaN-ish values without pulling pandas in.
        try:
            if v != v:  # pandas NaN
                continue
        except Exception:  # noqa: BLE001
            pass
        norm_cell = _norm_for_match(str(v))
        if not norm_cell:
            continue
        if norm_cell == norm_pred:
            rows_exact.append(i)
        elif can_substr and norm_pred in norm_cell:
            rows_substr.append(i)
    if rows_exact:
        return rows_exact, "exact"
    if rows_substr:
        return rows_substr, "substring"
    return [], "none"


def _safe_norm(v: str) -> str:
    """Trim for correction-quality comparisons (NOT match keys)."""
    return (v or "").strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_benchmark(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(
    bench_rows: list[dict],
    pred_rows: list[dict],
    *,
    include_trivial: bool = False,
    include_ambiguous: bool = False,
) -> dict:
    # ---- Index benchmark by (source, column, value) and (source, column). ----
    # Exact-match index returns every bench record whose normalized
    # value equals the lookup key (duplicates expected when the same
    # value appears in multiple cells -- e.g. 100 NHS rows all reading
    # 'Bradford Forfiet'). The by-(src,col) index supports the
    # substring fallback.
    exact_index, by_sc_index = _build_bench_indexes(bench_rows)

    bench_source_columns = {(r["source"], r["column"]) for r in bench_rows}

    # ---- Filter predictions for fair recall ----
    # Limit to predictions whose (source, column) pair was actually labeled
    # in the benchmark - otherwise we'd ding the model for flagging cells
    # the benchmark never considered.
    in_scope_preds: list[dict] = []
    out_of_scope = 0
    trivial_skipped = 0
    for p in pred_rows:
        if (p["source"], p["column"]) not in bench_source_columns:
            out_of_scope += 1
            continue
        if (not include_trivial) and str(p.get("trivial_prediction", "")).lower() == "true":
            trivial_skipped += 1
            continue
        in_scope_preds.append(p)

    # ---- Cell-level scoring ----
    # Each prediction is keyed on (source, column, predicted_original_value)
    # and looked up against the bench index built above. A single
    # prediction can therefore "fan out" across every bench row that
    # shares its value (e.g. 100 cells of 'Bradford Forfiet').
    error_labels = {"error", "ambiguous"} if include_ambiguous else {"error"}

    tp_cells: list[dict] = []   # benchmark error cells caught
    fp_preds: list[dict] = []   # predictions that hit clean cells
    excluded_preds: list[dict] = []  # ambiguous / unmatched / ...
    correction_exact = 0
    correction_ci = 0
    correction_norm = 0
    correction_total = 0  # over TP cells where benchmark has a corrected_value

    matched_error_keys: set[tuple] = set()
    matched_clean_keys: set[tuple] = set()

    # Per-prediction classification. A prediction that matches multiple
    # bench rows still counts as ONE outcome -- otherwise a single LLM
    # call that flagged a value appearing in 100 cells would inflate
    # FP/TP by 100x. We do, however, record each matched bench error row
    # in ``matched_error_keys`` so recall reflects every cell that the
    # prediction actually rescued.
    for p in in_scope_preds:
        src, col = p["source"], p["column"]
        matches, method = _match_pred(
            src, col, p.get("predicted_original_value"),
            exact_index, by_sc_index,
        )
        if not matches:
            # Predicted value doesn't appear in any bench row of this
            # (source, column), even as a substring -- e.g. a
            # hallucinated value or a token below the min-substring
            # threshold.
            excluded_preds.append({**p, "_reason": "no_value_match"})
            continue

        labels_seen: list[str] = []
        hit_error_rows: list[dict] = []
        hit_clean_rows: list[dict] = []
        for m in matches:
            label = m.get("label")
            labels_seen.append(label or "")
            if label in error_labels:
                hit_error_rows.append(m)
            elif label == "clean":
                hit_clean_rows.append(m)
            # ambiguous (when not include_ambiguous) / other: still
            # recorded in labels_seen but neither hit_error nor hit_clean.

        # Pred touched ONLY ambiguous (or unknown-label) rows -- don't
        # count for precision but keep visible.
        if not hit_error_rows and not hit_clean_rows:
            excluded_preds.append({**p, "_reason": "ambiguous_only"})
            continue

        if hit_error_rows:
            # TP credit: one per pred, but record every matched error
            # row so recall is honest.
            for m in hit_error_rows:
                ek = (m["source"], m["column"], m.get("row"))
                matched_error_keys.add(ek)
            # The primary TP record uses the first matched error row.
            m = hit_error_rows[0]
            tp_cells.append({
                "source": m["source"],
                "row": m["row"],
                "column": m["column"],
                "original_value": m["original_value"],
                "predicted_original_value": p.get("predicted_original_value", ""),
                "benchmark_corrected_value": m.get("corrected_value"),
                "predicted_correction": p["predicted_correction"],
                "pair_confidence": p.get("pair_confidence", ""),
                "match_method": method,
                "extra_matched_error_rows": [m2["row"] for m2 in hit_error_rows[1:]],
                "also_hit_clean_rows": [m2["row"] for m2 in hit_clean_rows],
            })
            gt_corr = (m.get("corrected_value") or "").strip()
            pred_corr = (p.get("predicted_correction") or "").strip()
            if gt_corr and pred_corr:
                correction_total += 1
                if gt_corr == pred_corr:
                    correction_exact += 1
                if gt_corr.lower() == pred_corr.lower():
                    correction_ci += 1
                if gt_corr.strip().lower() == pred_corr.strip().lower():
                    correction_norm += 1
        else:
            # FP: prediction touched only clean bench rows.
            for m in hit_clean_rows:
                ck = (m["source"], m["column"], m.get("row"))
                matched_clean_keys.add(ck)
            m = hit_clean_rows[0]
            fp_preds.append({
                "source": src,
                "row": m.get("row"),
                "column": col,
                "predicted_original_value": p["predicted_original_value"],
                "predicted_correction": p["predicted_correction"],
                "benchmark_rows": p.get("benchmark_rows", ""),
                "match_method": method,
                "n_clean_rows_hit": len(hit_clean_rows),
            })

    # ---- Recall: every bench error cell, was it caught? ----
    benchmark_error_cells = [r for r in bench_rows if r.get("label") in error_labels]
    missed_errors: list[dict] = []
    for r in benchmark_error_cells:
        key = (r["source"], r["column"], r.get("row"))
        if key not in matched_error_keys:
            missed_errors.append({
                "source": r["source"],
                "row": r["row"],
                "column": r["column"],
                "original_value": r.get("original_value"),
                "corrected_value": r.get("corrected_value"),
            })

    # ---- Metrics ----
    n_tp = len(tp_cells)              # benchmark error cells caught
    n_fp = len(fp_preds)              # predictions that flagged clean cells
    n_fn = len(missed_errors)         # benchmark error cells missed
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # ---- Per-source breakdown ----
    per_source = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "bench_errors": 0})
    for r in benchmark_error_cells:
        per_source[r["source"]]["bench_errors"] += 1
    for c in tp_cells:
        per_source[c["source"]]["tp"] += 1
    for c in fp_preds:
        per_source[c["source"]]["fp"] += 1
    for c in missed_errors:
        per_source[c["source"]]["fn"] += 1

    return {
        "summary": {
            "benchmark_total_cells": len(bench_rows),
            "benchmark_error_cells": len(benchmark_error_cells),
            "benchmark_clean_cells": sum(1 for r in bench_rows if r.get("label") == "clean"),
            "benchmark_ambiguous_cells": sum(1 for r in bench_rows if r.get("label") == "ambiguous"),
            "predictions_total": len(pred_rows),
            "predictions_in_scope": len(in_scope_preds),
            "predictions_out_of_scope": out_of_scope,
            "predictions_trivial_skipped": trivial_skipped,
            "predictions_excluded_ambig_or_unmatched": len(excluded_preds),
            "tp_cells": n_tp,
            "fp_predictions": n_fp,
            "fn_cells": n_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "correction_tp_with_gt": correction_total,
            "correction_exact_match": correction_exact,
            "correction_caseinsensitive_match": correction_ci,
            "correction_normalized_match": correction_norm,
            "correction_exact_rate": round(correction_exact / correction_total, 4) if correction_total else None,
            "correction_normalized_rate": round(correction_norm / correction_total, 4) if correction_total else None,
        },
        "tp_cells": tp_cells,
        "fp_predictions": fp_preds,
        "missed_errors": missed_errors,
        "excluded_predictions": excluded_preds,
        "per_source": dict(per_source),
    }


# ---------------------------------------------------------------------------
# Per-cell join (UI-oriented view)
# ---------------------------------------------------------------------------
#
# The :func:`compare` summary above answers "how good is the model" - which
# is what the CLI report wants. For the labelling UI we instead need a
# per-cell view that tells the operator, for every cell either side flagged
# (or labelled), exactly what each source said. ``build_per_cell_view``
# emits one record per cell with a single ``state`` string drawn from:
#
#   * ``true_positive``       benchmark=error AND a prediction matched
#   * ``false_positive``      benchmark=clean AND a prediction matched
#                             (the model flagged a clean cell as a typo)
#   * ``false_negative``      benchmark=error AND no prediction matched
#                             (the model missed the typo)
#   * ``agreed_clean``        benchmark=clean AND no prediction matched
#                             (both sides quietly agree the cell is fine)
#   * ``ambiguous_in_bench``  benchmark=ambiguous (irrespective of pred);
#                             surfaced separately so the operator can
#                             re-adjudicate without confusing P/R numbers.
#   * ``mixed_conflict``      a single (source, column, norm-value) maps to
#                             both clean and error benchmark rows
#   * ``unmatched_prediction`` prediction whose (source, column) is in
#                             scope but whose value has no benchmark row
#                             at all (genuine gap in benchmark coverage)
#
# ``out_of_scope_prediction`` rows are deliberately omitted: they aren't
# actionable from the labelling UI - the (source, column) pair simply isn't
# part of this benchmark, so showing them would invite the operator to add
# labels for cells the curator never intended to cover.


# Priority order for collapsing a (source, column, value) group's labels
# into a single ``summary_label`` that drives the per-cell ``state``.
#
# ``error`` wins: any unambiguous error verdict in the group makes the
# cell an error overall (model flagging it gets TP credit; missing it
# gets FN debit). Ambiguous siblings are downstream cleanup.
#
# ``ambiguous`` outranks ``clean``: if even one row in the group is
# ambiguous, the cell isn't unambiguously clean - claiming
# ``false_positive`` (model flagged a clean cell) would mislead the
# operator because the bench itself is uncertain. The state is
# instead ``ambiguous_in_bench`` so the operator sees the conflict
# and can adjudicate. (This used to be the other way around, which
# produced rows whose ``state=false_positive`` but whose displayed
# bench label was ``ambiguous`` - cosmetically broken.)
#
# The ``error+clean`` co-occurrence is handled as a special-case
# ``mixed_conflict`` ahead of this priority loop and is not affected.
_BENCH_LABEL_PRIORITY = ("error", "ambiguous", "clean")


def _bench_summary_label(matches: list[dict]) -> str:
    """Pick a single representative label for a group of benchmark rows
    sharing the same normalized key. Used to colour-code the UI grid.
    """
    labels = {m.get("label") for m in matches}
    if "error" in labels and "clean" in labels:
        return "mixed_conflict"
    for cand in _BENCH_LABEL_PRIORITY:
        if cand in labels:
            return cand
    # Fall back to whatever string we did see.
    return next(iter(labels), "") or ""


def build_per_cell_view(
    bench_rows: list[dict],
    pred_rows: list[dict],
    *,
    include_trivial: bool = False,
    csv_resolver=None,
) -> dict:
    """Build a per-cell join of benchmark + predictions for UI display.

    Returns ``{"summary": {...}, "cells": [...], "states": {state: count}}``.

    ``csv_resolver`` is optional. When provided it's a callable
    ``csv_resolver(source: str) -> pandas.DataFrame | None`` that
    returns the underlying CSV for a source so each
    ``unmatched_prediction`` row can be enriched with
    ``csv_match_rows`` (CSV rows whose value matches the predicted
    value) and ``csv_match_status`` (one of ``"in_csv"`` --
    bench-coverage gap; ``"not_in_csv"`` -- value isn't in the CSV
    at all, i.e. a hallucination; ``"unknown"`` -- no resolver
    available). This lets the operator flip the diff lens around:
    the unmatched predictions whose value is actually in the CSV
    are the most useful candidates for expanding the benchmark.

    Each cell record has the shape::

        {
          "state":          one of the labels above,
          "source":         "...csv",
          "row":            int | None,
          "column":         "...",
          "original_value": str | None,
          "benchmark": {                # None for unmatched_prediction
            "row":            int,
            "label":          "error" | "clean" | "ambiguous",
            "corrected_value": str | None,
            "corrected_value_raw": str | None,
            "error_types":   [str],
            "annotators":    [str],
            "note":          str,
            "all_rows":      [int],     # kept = [row] for back-compat
            "all_labels":    [str],     # kept = [label] for back-compat
            "record_index":  int,       # 0-based offset into the JSONL
          },
          "prediction": {                # None for FN / agreed_clean
            "predicted_original_value": str,
            "predicted_correction":     str,
            "pair_confidence":          str,
            "trivial_prediction":       bool,
            "gt_status":                str,
            "precision_outcome":        str,
            "notes":                    str,
            "benchmark_rows":           str,
            "match_method":             "exact" | "substring",
          },
        }
    """
    # ---- Index benchmark twice:
    #      * ``bench_by_row_key`` (src, col, row) -> (idx, record) -- used
    #        to iterate one grid row per bench JSONL record.
    #      * the value indexes from :func:`_build_bench_indexes` -- used
    #        by :func:`_match_pred` to link predictions to bench rows
    #        (exact value match, falling back to substring).
    bench_by_row_key: dict[tuple, tuple[int, dict]] = {}
    for idx, r in enumerate(bench_rows):
        rkey = (r.get("source") or "", r.get("column") or "", r.get("row"))
        bench_by_row_key[rkey] = (idx, r)
    exact_index, by_sc_index = _build_bench_indexes(bench_rows)

    bench_source_columns = {(r.get("source") or "", r.get("column") or "") for r in bench_rows}

    # ---- Bucket predictions onto bench row-keys via value match. ----
    # A prediction whose ``predicted_original_value`` appears in multiple
    # bench rows fans out across all of them so every bench cell sees
    # its own prediction record. Predictions whose value matches nothing
    # in scope (exact or substring) land in ``pred_unlinked`` and become
    # ``unmatched_prediction`` rows below.
    pred_by_key: dict[tuple, list[dict]] = defaultdict(list)
    pred_unlinked: list[dict] = []        # in-scope, but value matches no bench row
    out_of_scope_preds: list[dict] = []
    trivial_preds: list[dict] = []
    for p in pred_rows:
        src = p.get("source") or ""
        col = p.get("column") or ""
        if (src, col) not in bench_source_columns:
            out_of_scope_preds.append(p)
            continue
        if (not include_trivial) and str(p.get("trivial_prediction", "")).lower() == "true":
            trivial_preds.append(p)
            continue
        matches, method = _match_pred(
            src, col, p.get("predicted_original_value"),
            exact_index, by_sc_index,
        )
        if not matches:
            pred_unlinked.append(p)
            continue
        # Stash the method on a shallow copy so the grid view can label
        # substring-rescued matches without mutating the caller's dict.
        p_annot = dict(p)
        p_annot["_match_method"] = method
        for m in matches:
            rkey = (m["source"], m["column"], m.get("row"))
            pred_by_key[rkey].append(p_annot)

    cells: list[dict] = []

    def _pred_view(p: dict) -> dict:
        return {
            "predicted_original_value": p.get("predicted_original_value", ""),
            "predicted_correction": p.get("predicted_correction", ""),
            "pair_confidence": p.get("pair_confidence", ""),
            "trivial_prediction": str(p.get("trivial_prediction", "")).lower() == "true",
            "gt_status": p.get("gt_status", ""),
            "precision_outcome": p.get("precision_outcome", ""),
            "notes": p.get("notes", ""),
            "benchmark_rows": p.get("benchmark_rows", ""),
            "match_method": p.get("_match_method", "exact"),
        }

    def _bench_view(idx: int, r: dict) -> dict:
        # Each grid row maps to exactly one bench JSONL record; the
        # ``all_rows`` / ``all_labels`` arrays are kept (singletons) so
        # downstream JS doesn't have to special-case the new shape.
        row = r.get("row")
        return {
            "row": row,
            "label": r.get("label"),
            "corrected_value": r.get("corrected_value"),
            "corrected_value_raw": r.get("corrected_value_raw"),
            "corrected_value_canonicalized": r.get("corrected_value_canonicalized"),
            "corrected_value_canonicalized_by": r.get("corrected_value_canonicalized_by"),
            "llm_suggested_value": r.get("llm_suggested_value"),
            "error_types": r.get("error_types") or [],
            "annotators": r.get("annotators") or [],
            "note": r.get("note") or "",
            "all_rows": [int(row)] if row is not None else [],
            "all_labels": [r.get("label")] if r.get("label") is not None else [],
            "record_index": idx,
            "summary_label": r.get("label") or "",
            # Surface the in-file audit trail so the UI can both render
            # it in the drawer AND count "rows the operator has touched"
            # for the canonicalize button.
            "_label_history": r.get("_label_history") or [],
        }

    # ---- One grid row per benchmark record. ----
    for key, (idx, r) in bench_by_row_key.items():
        src, col, row = key
        preds = pred_by_key.get(key, [])
        bview = _bench_view(idx, r)
        label = bview["label"]
        pview = _pred_view(preds[0]) if preds else None

        if label == "ambiguous":
            state = "ambiguous_in_bench"
        elif label == "error":
            state = "true_positive" if preds else "false_negative"
        elif label == "clean":
            state = "false_positive" if preds else "agreed_clean"
        else:
            state = "ambiguous_in_bench"

        cells.append({
            "state": state,
            "source": src,
            "column": col,
            "row": row,
            "original_value": r.get("original_value"),
            "benchmark": bview,
            "prediction": pview,
            "extra_predictions": max(0, len(preds) - 1),
        })

    # ---- Predictions whose value didn't match any bench row.       ----
    # When a ``csv_resolver`` is supplied we scan the underlying CSV
    # for the predicted value, so the operator can tell the difference
    # between a "hallucination" (value not in the CSV) and a real
    # bench-coverage gap (value in the CSV but unlabeled). The cell's
    # ``row`` is set to the first CSV match so the drawer's CSV
    # context lookup works without any client-side hop.
    bench_gap_count = 0
    csv_seen = csv_resolver is not None
    for p in pred_unlinked:
        src = p.get("source") or ""
        col = p.get("column") or ""
        csv_rows: list[int] = []
        csv_method = "none"
        if csv_seen:
            try:
                frame = csv_resolver(src)
            except Exception:  # noqa: BLE001
                frame = None
            # Exact-only for bench-gap classification: a substring hit
            # (e.g. predicted ``'ane'`` appearing inside a long cell)
            # is not strong enough evidence that the model actually
            # flagged a real CSV cell, so we don't promote those to
            # ``in_csv`` / bench_gap.
            csv_rows, csv_method = _find_csv_matches(
                frame, col, p.get("predicted_original_value"),
                allow_substring=False,
            )
        if csv_seen:
            csv_status = "in_csv" if csv_rows else "not_in_csv"
        else:
            csv_status = "unknown"
        if csv_status == "in_csv":
            bench_gap_count += 1
        cells.append({
            "state": "unmatched_prediction",
            "source": src,
            "column": col,
            # The first CSV hit drives the drawer's CSV context block.
            "row": csv_rows[0] if csv_rows else None,
            "original_value": p.get("predicted_original_value"),
            "benchmark": None,
            "prediction": _pred_view(p),
            "extra_predictions": 0,
            "csv_match_status": csv_status,
            "csv_match_rows": csv_rows,
            "csv_match_method": csv_method,
        })

    # ---- Stable, useful ordering: state-priority, then source, then row. ----
    state_order = {
        "false_negative":     0,  # operator-action ranked highest
        "false_positive":     1,
        "mixed_conflict":     2,
        "ambiguous_in_bench": 3,
        "unmatched_prediction": 4,
        "true_positive":      5,
        "agreed_clean":       6,
    }
    cells.sort(key=lambda c: (
        state_order.get(c["state"], 99),
        c["source"] or "",
        (c["row"] if c["row"] is not None else 10**9),
        c["column"] or "",
    ))

    state_counts: dict[str, int] = Counter(c["state"] for c in cells)
    # Pseudo-state surfaced for the UI's coverage-gap filter: a
    # subset of ``unmatched_prediction`` whose value DOES appear in
    # the source CSV (so the operator could go label it).
    if csv_seen:
        state_counts["bench_gap"] = bench_gap_count
        state_counts["unmatched_hallucination"] = (
            state_counts.get("unmatched_prediction", 0) - bench_gap_count
        )

    # Reuse the existing summary so the UI can show the same headline
    # metrics (precision/recall/F1) without recomputing.
    summary = compare(bench_rows, pred_rows, include_trivial=include_trivial)["summary"]
    summary["out_of_scope_predictions"] = len(out_of_scope_preds)
    summary["trivial_predictions_skipped"] = len(trivial_preds)
    summary["csv_resolver_used"] = csv_seen
    if csv_seen:
        summary["bench_gap_candidates"] = bench_gap_count

    return {
        "summary": summary,
        "states": dict(state_counts),
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict], columns: Iterable[str]) -> None:
    columns = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )

    _write_csv(
        out_dir / "true_positives.csv",
        result["tp_cells"],
        ["source", "row", "column", "original_value",
         "predicted_original_value",
         "benchmark_corrected_value", "predicted_correction", "pair_confidence",
         "match_method"],
    )
    _write_csv(
        out_dir / "false_positives.csv",
        result["fp_predictions"],
        ["source", "column", "predicted_original_value",
         "predicted_correction", "benchmark_rows", "match_method"],
    )
    _write_csv(
        out_dir / "missed_errors.csv",
        result["missed_errors"],
        ["source", "row", "column", "original_value", "corrected_value"],
    )
    _write_csv(
        out_dir / "excluded_predictions.csv",
        result["excluded_predictions"],
        ["source", "column", "predicted_original_value",
         "predicted_correction", "gt_status", "_reason"],
    )

    per_src_rows = [
        {"source": src, **stats,
         "precision": round(stats["tp"] / (stats["tp"] + stats["fp"]), 4)
            if (stats["tp"] + stats["fp"]) else None,
         "recall": round(stats["tp"] / (stats["tp"] + stats["fn"]), 4)
            if (stats["tp"] + stats["fn"]) else None}
        for src, stats in sorted(result["per_source"].items())
    ]
    _write_csv(
        out_dir / "per_source.csv",
        per_src_rows,
        ["source", "bench_errors", "tp", "fp", "fn", "precision", "recall"],
    )


def print_summary(summary: dict, stream=sys.stdout) -> None:
    width = 38
    print("=" * width, file=stream)
    print("External run vs. typo benchmark", file=stream)
    print("=" * width, file=stream)
    for k, v in summary.items():
        print(f"  {k:<42} {v}", file=stream)


# ---------------------------------------------------------------------------
# Batch driver (directory of prediction TSVs)
# ---------------------------------------------------------------------------

# Columns in the per-run summary.tsv emitted by batch mode. Kept stable so it
# can be diffed across invocations and joined with external/GPT_score/summary.tsv.
_BATCH_SUMMARY_COLUMNS = [
    "run_file",
    "predictions_total",
    "predictions_in_scope",
    "predictions_out_of_scope",
    "predictions_trivial_skipped",
    "tp_cells",
    "fp_predictions",
    "fn_cells",
    "precision",
    "recall",
    "f1",
    "correction_tp_with_gt",
    "correction_exact_match",
    "correction_caseinsensitive_match",
    "correction_normalized_match",
    "correction_exact_rate",
    "correction_normalized_rate",
]


def _safe_subdir(stem: str) -> str:
    """Filesystem-safe subdir name derived from a prediction filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "run"


def run_batch(
    bench_rows: list[dict],
    predictions_dir: Path,
    out_dir: Path,
    *,
    pattern: str = "*.scoring.tsv",
    include_trivial: bool = False,
    include_ambiguous: bool = False,
    stream=sys.stdout,
) -> list[dict]:
    """Compare every TSV in ``predictions_dir`` matching ``pattern`` against
    ``bench_rows``. Writes per-run detail CSVs under
    ``out_dir/<run_stem>/`` and a collated ``out_dir/summary.tsv``.
    """
    files = sorted(predictions_dir.glob(pattern))
    if not files:
        print(
            f"WARNING: no files matched {predictions_dir}/{pattern}",
            file=stream,
        )
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for path in files:
        print(f"\n--- {path.name} ---", file=stream)
        pred_rows = load_predictions(path)
        result = compare(
            bench_rows,
            pred_rows,
            include_trivial=include_trivial,
            include_ambiguous=include_ambiguous,
        )

        run_dir = out_dir / _safe_subdir(path.stem)
        write_outputs(result, run_dir)
        print_summary(result["summary"], stream=stream)
        print(f"  details: {run_dir}", file=stream)

        summary_rows.append({
            "run_file": path.name,
            **{k: result["summary"].get(k) for k in _BATCH_SUMMARY_COLUMNS[1:]},
        })

    # ---- Collated summary.tsv (analogous to external/GPT_score/summary.tsv) ----
    summary_path = out_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=_BATCH_SUMMARY_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
        )
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    # Also write the same data as JSON for downstream tooling.
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )

    print(f"\nBatch summary written to: {summary_path}", file=stream)
    return summary_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, type=Path)
    p.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to a single scoring TSV, or a directory of TSVs.",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--predictions-glob",
        default="*.scoring.tsv",
        help="Glob (relative to --predictions) used when --predictions is a "
             "directory. Default: *.scoring.tsv",
    )
    p.add_argument("--include-trivial", action="store_true",
                   help="Count trivial-diff predictions instead of ignoring them.")
    p.add_argument("--include-ambiguous", action="store_true",
                   help="Treat 'ambiguous' benchmark cells as errors.")
    args = p.parse_args(argv)

    bench_rows = load_benchmark(args.benchmark)

    if args.predictions.is_dir():
        run_batch(
            bench_rows,
            args.predictions,
            args.out_dir,
            pattern=args.predictions_glob,
            include_trivial=args.include_trivial,
            include_ambiguous=args.include_ambiguous,
        )
        return 0

    pred_rows = load_predictions(args.predictions)
    result = compare(
        bench_rows,
        pred_rows,
        include_trivial=args.include_trivial,
        include_ambiguous=args.include_ambiguous,
    )
    write_outputs(result, args.out_dir)
    print_summary(result["summary"])
    print(f"\nDetail CSVs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
