"""
Compare external inconsistency-detector run(s) against the labeled
inconsistency benchmark
(benchmark/benchmark_inconsistency_*.jsonl).

The external TSV (e.g. external/model_inconsistency_result/d.txt) emits
one row per *synonym pair* the model proposes on a given (table, column):

    file_id, col_header, synonyms = [[val_a, val_b]], pair_confidence,
    DetectionType ("SYN"), Trivial-diff, ...

Each pair is the model's claim that ``val_a`` and ``val_b`` are two
representations of the same entity (hence one of them is an
inconsistency).

The benchmark JSONL is keyed per cell:

    source, column, row, original_value, corrected_value, label

A *pair-level* match is declared when a predicted pair {V_a, V_b}
links, on the same (source, column), to a benchmark row whose
``original_value`` matches one of {V_a, V_b} and whose
``corrected_value`` matches the other. Whitespace is collapsed for the
match (same rule as ``compare_external_typo_run``); casefolding is not
applied so the model is still penalised for proposing casing
inconsistencies that the benchmark does not endorse.

The script understands ``--predictions`` as either a single TSV or a
directory of TSVs; in the latter case every file is scored and a
``summary.tsv`` collates per-run metrics. It also emits a synthetic
"union" entry that combines all predictions across runs (deduplicated
on the unordered pair) so the user can see how much extra recall the
runs gain in aggregate.

Usage::

    python -m experiments.compare_external_inconsistency_run \\
        --benchmark benchmark/benchmark_inconsistency_20260513T090000Z_gpt5-family.jsonl \\
        --predictions external/model_inconsistency_result \\
        --out-dir outputs/external_inconsistency_vs_benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(value: str | None) -> str:
    """Collapse internal whitespace and trim edges. No casefolding."""
    return " ".join((value or "").split())


def _candidate_corrected_values(raw) -> list[str]:
    """Normalize the benchmark's polymorphic ``corrected_value`` field
    into a list of candidate canonical strings.

    The field appears as one of:
      * ``None`` -- no correction proposed; returns ``[]``.
      * ``str``  -- single canonical form; returns ``[norm(str)]``.
      * ``list[str]`` -- multiple plausible canonical forms (e.g.
        ambiguous correction); returns each element normalized.

    Empty strings and entries equal (after normalization) to the
    original value should be filtered by the caller.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        n = _norm(raw)
        return [n] if n else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            n = _norm(item)
            if n:
                out.append(n)
        return out
    return []


def _parse_pair(raw: str | None) -> tuple[str, str] | None:
    """Parse the external ``synonyms`` cell into a single (val_a, val_b) pair.

    The TSV emits one pair per row (already), but the cell still holds
    the JSON ``[[val_a, val_b]]`` form. Returns ``None`` when the cell
    is empty / malformed / not a 2-element pair.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    first = parsed[0]
    if not isinstance(first, list) or len(first) != 2:
        return None
    a, b = first
    if a is None or b is None:
        return None
    return (str(a), str(b))


def _pair_key(a: str, b: str) -> frozenset[str]:
    """Unordered pair key on whitespace-collapsed values."""
    return frozenset({_norm(a), _norm(b)})


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_benchmark(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_predictions(path: Path) -> list[dict]:
    """Load an external TSV and decorate each row with parsed pair / key."""
    out: list[dict] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            pair = _parse_pair(r.get("synonyms"))
            if pair is None:
                # Defensive: skip rows we cannot interpret.
                continue
            r["_pair_a"], r["_pair_b"] = pair
            r["_pair_key"] = _pair_key(*pair)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Benchmark indexing
# ---------------------------------------------------------------------------

def _build_bench_pair_index(
    bench_rows: list[dict],
    *,
    error_labels: set[str],
) -> tuple[dict, dict]:
    """Return ``(pair_index, by_sc_values)``.

    * ``pair_index[(source, column)] -> list[dict]`` of synthetic
      "pair records" representing each labeled inconsistency as an
      unordered pair { original_value, corrected_value }. Each entry
      carries: ``key`` (frozenset), ``rows`` (every bench row that
      contributes), ``label_set`` (labels of contributing rows),
      ``cluster_kinds``, ``corrected_value``.

    * ``by_sc_values[(source, column)] -> set[str]`` of all
      normalized values appearing in the benchmark for that
      (source, column). Used to classify a predicted pair where neither
      value matches a benchmark cell as ``no_value_match``.
    """
    # Per (src,col) gather contributions per unordered pair.
    pair_buckets: dict[tuple, dict[frozenset, dict]] = defaultdict(dict)
    by_sc_values: dict[tuple, set[str]] = defaultdict(set)
    for r in bench_rows:
        src = r.get("source") or ""
        col = r.get("column") or ""
        orig = _norm(r.get("original_value"))
        candidates = _candidate_corrected_values(r.get("corrected_value"))
        by_sc_values[(src, col)].add(orig)
        for corr in candidates:
            by_sc_values[(src, col)].add(corr)
        if not orig or not candidates:
            # Cells with no proposed correction (label='error' but
            # corrected_value=None, or label='ambiguous' with empty
            # correction) contribute their value to by_sc_values for
            # matching, but cannot define a pair.
            continue
        for corr in candidates:
            if corr == orig:
                continue
            key = frozenset({orig, corr})
            bucket = pair_buckets[(src, col)].setdefault(
                key,
                {
                    "key": key,
                    "original_value": orig,
                    "corrected_value": corr,
                    "rows": [],
                    "labels": set(),
                    "cluster_kinds": set(),
                },
            )
            bucket["rows"].append(r)
            bucket["labels"].add(r.get("label") or "")
            for ck in r.get("cluster_kinds", []) or []:
                bucket["cluster_kinds"].add(ck)

    pair_index: dict[tuple, list[dict]] = {}
    for sc, by_key in pair_buckets.items():
        pair_index[sc] = list(by_key.values())
    return pair_index, by_sc_values


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(
    bench_rows: list[dict],
    pred_rows: list[dict],
    *,
    include_ambiguous: bool = False,
    include_trivial: bool = True,
) -> dict:
    """Score predictions against the benchmark at the unordered-pair level.

    * Each prediction (one TSV row) is one outcome (TP / FP / excluded).
    * Each benchmark *pair* is the recall denominator. A pair is the
      unordered set ``{original_value, corrected_value}`` aggregated
      over all cells on the same (source, column) that share that pair.
    """
    error_labels = {"error", "ambiguous"} if include_ambiguous else {"error"}
    pair_index, by_sc_values = _build_bench_pair_index(
        bench_rows, error_labels=error_labels
    )

    bench_source_columns = set(pair_index.keys()) | set(by_sc_values.keys())

    # Filter predictions in-scope.
    in_scope: list[dict] = []
    out_of_scope: list[dict] = []
    trivial_skipped: list[dict] = []
    for p in pred_rows:
        sc = (p["file_id"], p["col_header"])
        if sc not in bench_source_columns:
            out_of_scope.append(p)
            continue
        if (not include_trivial) and str(p.get("Trivial-diff", "")).strip().upper() == "TRUE":
            trivial_skipped.append(p)
            continue
        in_scope.append(p)

    # Helper to classify whether a benchmark pair counts as ground truth.
    def _bench_pair_is_positive(b: dict) -> bool:
        return bool(b["labels"] & error_labels)

    tp_records: list[dict] = []
    fp_records: list[dict] = []
    excluded: list[dict] = []

    matched_bench_keys: set[tuple] = set()  # (src, col, frozenset)

    for p in in_scope:
        sc = (p["file_id"], p["col_header"])
        key = p["_pair_key"]
        a_norm = _norm(p["_pair_a"])
        b_norm = _norm(p["_pair_b"])

        # Try exact pair match against benchmark pairs first.
        match = None
        for b in pair_index.get(sc, []):
            if b["key"] == key:
                match = b
                break

        if match is not None and _bench_pair_is_positive(match):
            matched_bench_keys.add((sc[0], sc[1], match["key"]))
            tp_records.append({
                "file_id": sc[0],
                "col_header": sc[1],
                "pred_pair_a": p["_pair_a"],
                "pred_pair_b": p["_pair_b"],
                "bench_original_value": match["original_value"],
                "bench_corrected_value": match["corrected_value"],
                "bench_labels": sorted(match["labels"]),
                "bench_cluster_kinds": sorted(match["cluster_kinds"]),
                "bench_rows": [m.get("row") for m in match["rows"]],
                "pair_confidence_min_token": p.get("pair_min_token_probs"),
                "pair_geo_mean_prob": p.get("pair_geo_mean_probs"),
                "trivial_diff": p.get("Trivial-diff"),
            })
            continue

        if match is not None and not _bench_pair_is_positive(match):
            # Pair was labeled but only with "ambiguous"/other (when
            # include_ambiguous is False) -- exclude from precision.
            excluded.append({
                "file_id": sc[0],
                "col_header": sc[1],
                "pred_pair_a": p["_pair_a"],
                "pred_pair_b": p["_pair_b"],
                "reason": "ambiguous_only",
                "bench_labels": sorted(match["labels"]),
                "trivial_diff": p.get("Trivial-diff"),
            })
            continue

        # Fall through: no benchmark pair has this exact pair. Decide
        # FP vs no_value_match based on whether the predicted values
        # appear at all in the benchmark for this (src, col).
        vals_in_bench = by_sc_values.get(sc, set())
        a_in = a_norm in vals_in_bench
        b_in = b_norm in vals_in_bench
        if a_in and b_in:
            # Both values are in the column, but neither side is paired
            # as an inconsistency in the benchmark -- straightforward FP.
            fp_records.append({
                "file_id": sc[0],
                "col_header": sc[1],
                "pred_pair_a": p["_pair_a"],
                "pred_pair_b": p["_pair_b"],
                "reason": "both_values_present_but_not_paired_in_bench",
                "pair_confidence_min_token": p.get("pair_min_token_probs"),
                "pair_geo_mean_prob": p.get("pair_geo_mean_probs"),
                "trivial_diff": p.get("Trivial-diff"),
            })
        elif a_in or b_in:
            fp_records.append({
                "file_id": sc[0],
                "col_header": sc[1],
                "pred_pair_a": p["_pair_a"],
                "pred_pair_b": p["_pair_b"],
                "reason": "one_value_present_only",
                "pair_confidence_min_token": p.get("pair_min_token_probs"),
                "pair_geo_mean_prob": p.get("pair_geo_mean_probs"),
                "trivial_diff": p.get("Trivial-diff"),
            })
        else:
            # Neither value present in the bench column at all.
            excluded.append({
                "file_id": sc[0],
                "col_header": sc[1],
                "pred_pair_a": p["_pair_a"],
                "pred_pair_b": p["_pair_b"],
                "reason": "no_value_match",
                "trivial_diff": p.get("Trivial-diff"),
            })

    # Recall: every positive benchmark pair, was it caught?
    bench_positive_pairs: list[dict] = []
    for sc, pairs in pair_index.items():
        if sc not in bench_source_columns:
            continue
        for b in pairs:
            if _bench_pair_is_positive(b):
                bench_positive_pairs.append({"sc": sc, **b})

    missed: list[dict] = []
    for b in bench_positive_pairs:
        sc = b["sc"]
        if (sc[0], sc[1], b["key"]) not in matched_bench_keys:
            missed.append({
                "file_id": sc[0],
                "col_header": sc[1],
                "bench_original_value": b["original_value"],
                "bench_corrected_value": b["corrected_value"],
                "bench_labels": sorted(b["labels"]),
                "bench_cluster_kinds": sorted(b["cluster_kinds"]),
                "bench_rows": [m.get("row") for m in b["rows"]],
            })

    # Restrict recall to (source, column) pairs the external run actually
    # covered. Predictions for a column are required for that column's
    # bench pairs to be reachable; otherwise we'd dock recall for
    # columns the external never inspected.
    covered_scs = {(p["file_id"], p["col_header"]) for p in in_scope}
    bench_positive_in_scope = [
        b for b in bench_positive_pairs if b["sc"] in covered_scs
    ]
    missed_in_scope = [m for m in missed if (m["file_id"], m["col_header"]) in covered_scs]

    n_tp = len(tp_records)
    n_fp = len(fp_records)
    n_predictions = n_tp + n_fp  # excluded preds not counted for precision

    n_positive_total = len(bench_positive_pairs)
    n_positive_in_scope = len(bench_positive_in_scope)
    n_matched_in_scope = len(
        {(t["file_id"], t["col_header"], frozenset({_norm(t["bench_original_value"]), _norm(t["bench_corrected_value"])}))
         for t in tp_records}
    )

    precision = n_tp / n_predictions if n_predictions else 0.0
    recall_in_scope = n_matched_in_scope / n_positive_in_scope if n_positive_in_scope else 0.0
    recall_overall = n_matched_in_scope / n_positive_total if n_positive_total else 0.0
    f1_in_scope = (
        2 * precision * recall_in_scope / (precision + recall_in_scope)
        if (precision + recall_in_scope) else 0.0
    )

    return {
        "metrics": {
            "include_ambiguous": include_ambiguous,
            "include_trivial": include_trivial,
            "n_predictions_total": len(pred_rows),
            "n_predictions_in_scope": len(in_scope),
            "n_predictions_out_of_scope": len(out_of_scope),
            "n_predictions_trivial_skipped": len(trivial_skipped),
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_excluded": len(excluded),
            "n_positive_bench_pairs_total": n_positive_total,
            "n_positive_bench_pairs_in_scope": n_positive_in_scope,
            "n_matched_bench_pairs": n_matched_in_scope,
            "precision": precision,
            "recall_in_scope": recall_in_scope,
            "recall_overall": recall_overall,
            "f1_in_scope": f1_in_scope,
            "covered_source_columns": len(covered_scs),
        },
        "tp": tp_records,
        "fp": fp_records,
        "excluded": excluded,
        "missed_in_scope": missed_in_scope,
        "missed_out_of_scope": [
            m for m in missed if (m["file_id"], m["col_header"]) not in covered_scs
        ],
        "covered_source_columns": sorted(covered_scs),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            # frozenset / set are not JSON-serializable; normalize to lists.
            safe = {
                k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
                for k, v in r.items()
            }
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")


def _write_metrics_json(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def _print_metrics(label: str, metrics: dict, out: Iterable | None = None) -> None:
    f = out if out is not None else sys.stdout
    print(f"\n=== {label} ===", file=f)
    print(
        f"  predictions: total={metrics['n_predictions_total']} "
        f"in_scope={metrics['n_predictions_in_scope']} "
        f"out_of_scope={metrics['n_predictions_out_of_scope']} "
        f"trivial_skipped={metrics['n_predictions_trivial_skipped']}",
        file=f,
    )
    print(
        f"  pair-level:  TP={metrics['n_tp']} FP={metrics['n_fp']} "
        f"excluded={metrics['n_excluded']}",
        file=f,
    )
    print(
        f"  bench pairs: total={metrics['n_positive_bench_pairs_total']} "
        f"in_scope={metrics['n_positive_bench_pairs_in_scope']} "
        f"matched={metrics['n_matched_bench_pairs']}",
        file=f,
    )
    print(
        f"  precision={metrics['precision']:.3f} "
        f"recall_in_scope={metrics['recall_in_scope']:.3f} "
        f"recall_overall={metrics['recall_overall']:.3f} "
        f"f1_in_scope={metrics['f1_in_scope']:.3f}",
        file=f,
    )
    print(
        f"  covered (source,column) keys: {metrics['covered_source_columns']}",
        file=f,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_prediction_paths(arg: Path) -> list[Path]:
    if arg.is_dir():
        return sorted(p for p in arg.iterdir() if p.is_file() and p.suffix in {".tsv", ".txt"})
    return [arg]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path,
                    help="Either a single TSV/TXT file or a directory of them.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--include-ambiguous", action="store_true",
                    help="Count benchmark rows with label='ambiguous' as positives.")
    ap.add_argument("--exclude-trivial", action="store_true",
                    help="Drop external predictions flagged Trivial-diff=TRUE.")
    args = ap.parse_args(argv)

    bench_rows = load_benchmark(args.benchmark)
    pred_paths = _resolve_prediction_paths(args.predictions)
    if not pred_paths:
        print(f"No prediction files found at {args.predictions}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    union_preds: dict[tuple, dict] = {}

    for pred_path in pred_paths:
        preds = load_predictions(pred_path)
        result = compare(
            bench_rows,
            preds,
            include_ambiguous=args.include_ambiguous,
            include_trivial=not args.exclude_trivial,
        )
        run_dir = args.out_dir / pred_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        _write_metrics_json(run_dir / "metrics.json", result["metrics"])
        _write_jsonl(run_dir / "tp.jsonl", result["tp"])
        _write_jsonl(run_dir / "fp.jsonl", result["fp"])
        _write_jsonl(run_dir / "excluded.jsonl", result["excluded"])
        _write_jsonl(run_dir / "missed_in_scope.jsonl", result["missed_in_scope"])
        _write_jsonl(run_dir / "missed_out_of_scope.jsonl", result["missed_out_of_scope"])

        m = result["metrics"]
        summary_rows.append({"run": pred_path.name, **m})
        _print_metrics(pred_path.name, m)

        # Accumulate into the union pool (dedupe on (sc, pair_key)).
        for p in preds:
            sc = (p["file_id"], p["col_header"])
            key = (sc, frozenset(p["_pair_key"]))
            # Keep the prediction with the highest min-token probability
            # if duplicates appear across runs.
            current = union_preds.get(key)
            try:
                cand_score = float(p.get("pair_min_token_probs") or 0.0)
            except (TypeError, ValueError):
                cand_score = 0.0
            if current is None or cand_score > current.get("_score", 0.0):
                union_preds[key] = {**p, "_score": cand_score}

    # Union pseudo-run
    if len(pred_paths) > 1:
        union_pred_rows = list(union_preds.values())
        result = compare(
            bench_rows,
            union_pred_rows,
            include_ambiguous=args.include_ambiguous,
            include_trivial=not args.exclude_trivial,
        )
        run_dir = args.out_dir / "_union"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_metrics_json(run_dir / "metrics.json", result["metrics"])
        _write_jsonl(run_dir / "tp.jsonl", result["tp"])
        _write_jsonl(run_dir / "fp.jsonl", result["fp"])
        _write_jsonl(run_dir / "excluded.jsonl", result["excluded"])
        _write_jsonl(run_dir / "missed_in_scope.jsonl", result["missed_in_scope"])
        _write_jsonl(run_dir / "missed_out_of_scope.jsonl", result["missed_out_of_scope"])
        summary_rows.append({"run": "_union", **result["metrics"]})
        _print_metrics("_union (dedup across runs)", result["metrics"])

    # Write summary TSV
    summary_path = args.out_dir / "summary.tsv"
    if summary_rows:
        cols = ["run"] + [k for k in summary_rows[0].keys() if k != "run"]
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        print(f"\nWrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
