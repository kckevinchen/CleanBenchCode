"""
Compare the Excel "Clean Data / Ideas" external finding dump
(``external/excel_finding.csv``) against the curated per-cell benchmark
(``benchmark/merged_v3/benchmark_cell.csv``), restricted to the
**inconsistency** error type.

Unlike ``compare_external_inconsistency_run`` (which scores a model that
emits one row per *synonym pair*), the Excel dump is **per cell**: every
row flags a single (source, column, row) cell with a ``type`` and an
example ``suggested_fix``. Its inconsistency-flavoured types are::

    Inconsistent pattern   (e.g. phone "651-602-4702" -> "(651) 602-4702")
    Inconsistent text      (e.g. "keys" -> "Keys")
    Inconsistent order     (e.g. "Spanish, English" -> "English, Spanish")

The Excel ``data_row`` column is the 0-based data-row index and aligns
1:1 with the benchmark ``row``; ``header`` aligns with the benchmark
``column``. Matching is therefore primarily **positional**
``(source, column, row)``, with a whitespace-collapsed **value** fallback
within the same ``(source, column)`` to survive any row renumbering from
the benchmark's source normalisation.

Scope
-----
* An Excel finding is *judgeable* iff its source is part of the benchmark
  coverage (``sources_normalized.zip``); everything else is reported as
  out-of-scope and excluded from precision.
* For recall we additionally report a "scanned" view restricted to the
  sources that appear anywhere in the Excel dump (the files Excel
  demonstrably looked at), alongside the global view over every benchmark
  inconsistency-error cell.

Buckets (per Excel inconsistency finding)
-----------------------------------------
* ``tp``            -- matches a benchmark **inconsistency / error** cell.
* ``ambiguous``     -- matches a benchmark **inconsistency / ambiguous**
                       cell; excluded from precision & recall.
* ``fp_othertype``  -- matches a benchmark error/ambiguous cell of a
                       *different* type (Excel mislabelled the kind).
* ``fp_clean``      -- matches no benchmark error/ambiguous cell (the
                       benchmark considers the cell clean).

Buckets (per benchmark inconsistency-error cell, recall side)
-------------------------------------------------------------
* covered by an Excel inconsistency finding   -> counted as recall TP.
* ``fn_othertype`` -- Excel flagged the cell, but with a non-inconsistency
                      type (e.g. "Typo suggestion").
* ``fn_missed``    -- Excel did not flag the cell at all.

Usage::

    python -m experiments.compare_excel_finding_inconsistency \\
        --benchmark benchmark/merged_v3/benchmark_cell.csv \\
        --predictions external/excel_finding.csv \\
        --coverage-zip benchmark/merged_v3/sources_normalized.zip \\
        --out-dir outputs/excel_finding_vs_merged_v3_inconsistency
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# Excel finding "type" values that express an inconsistency claim.
EXCEL_INCONSISTENCY_TYPES = {
    "Inconsistent pattern",
    "Inconsistent text",
    "Inconsistent order",
}

# Benchmark error-type token that means "inconsistency".
BENCH_INCONSISTENCY = "inconsistency"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm(value: str | None) -> str:
    """Collapse internal whitespace and trim edges. No casefolding."""
    return " ".join((value or "").split())


def _bench_types(row: dict) -> set[str]:
    raw = row.get("error_types") or ""
    return {t for t in raw.split(";") if t}


def _is_inconsistency(row: dict) -> bool:
    return BENCH_INCONSISTENCY in _bench_types(row)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_benchmark_cells(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_excel_findings(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_coverage(zip_path: Path | None, benchmark_rows: list[dict]) -> set[str]:
    """Set of source basenames the benchmark is allowed to judge.

    Prefer the normalized-sources zip (the exact files the benchmark was
    built over); fall back to the sources that appear in the benchmark.
    """
    if zip_path and zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            return {Path(n).name for n in z.namelist() if n.endswith(".csv")}
    return {r["source"] for r in benchmark_rows}


def load_dataset_map(zip_path: Path | None, benchmark_rows: list[dict]) -> dict[str, str]:
    """Map source basename -> dataset label (``pbi_500`` / ``bing21_500``).

    Derived from the normalized-sources zip's top-level folder, with the
    benchmark rows' own ``dataset`` column taking precedence where present.
    """
    out: dict[str, str] = {}
    if zip_path and zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            for n in z.namelist():
                if not n.endswith(".csv"):
                    continue
                prefix = n.split("/")[0]
                if "pbi" in prefix:
                    out[Path(n).name] = "pbi_500"
                elif "bing21" in prefix:
                    out[Path(n).name] = "bing21_500"
    for r in benchmark_rows:
        if r.get("dataset"):
            out[r["source"]] = r["dataset"]
    return out


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _index_benchmark(rows: list[dict]) -> tuple[dict, dict]:
    """``by_cell[(source, column, row)] -> record`` and
    ``by_sc_value[(source, column)][norm_value] -> list[record]``."""
    by_cell: dict[tuple[str, str, str], dict] = {}
    by_sc_value: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        key = (r["source"], r["column"], str(r["row"]))
        by_cell[key] = r
        by_sc_value[(r["source"], r["column"])][_norm(r["original_value"])].append(r)
    return by_cell, by_sc_value


def _resolve_bench(
    finding: dict,
    by_cell: dict,
    by_sc_value: dict,
) -> tuple[dict | None, str | None]:
    """Resolve an Excel finding to the benchmark cell it refers to.

    Returns ``(record, method)`` where method is ``"row"`` (positional),
    ``"value"`` (value fallback), or ``(None, None)`` when nothing in the
    benchmark covers that cell.
    """
    src = finding["source"]
    col = finding["header"]
    row = str(finding["data_row"])
    rec = by_cell.get((src, col, row))
    if rec is not None:
        return rec, "row"
    # Value fallback within the same (source, column).
    cands = by_sc_value.get((src, col), {}).get(_norm(finding["value"]))
    if cands:
        # Prefer the most informative match: inconsistency-error first,
        # then inconsistency-ambiguous, then any flagged cell.
        def rank(r: dict) -> tuple[int, int]:
            inc = _is_inconsistency(r)
            err = r["status"] == "error"
            return (0 if inc else 1, 0 if err else 1)

        return sorted(cands, key=rank)[0], "value"
    return None, None


def _index_excel_all(findings: list[dict]) -> tuple[dict, dict]:
    """Index *all* Excel findings (every type) for the FN-side lookup."""
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_sc_value: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for f in findings:
        by_cell[(f["source"], f["header"], str(f["data_row"]))].append(f)
        by_sc_value[(f["source"], f["header"])][_norm(f["value"])].append(f)
    return by_cell, by_sc_value


def _index_excel_inc_coverage(inc_findings: list[dict]) -> tuple[dict, dict]:
    """Per ``(source, column)`` coverage sets for the **inconsistency**
    findings: the set of flagged rows and the set of flagged values.

    Excel's "Inconsistent pattern/text/order" findings are effectively a
    *value-level* claim about a column, so a benchmark cell counts as
    covered when Excel flagged either its exact row **or** its exact
    value -- otherwise duplicate-valued rows Excel did not bother to
    re-enumerate would look like false negatives.
    """
    rows: dict[tuple[str, str], set[str]] = defaultdict(set)
    values: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in inc_findings:
        sc = (f["source"], f["header"])
        rows[sc].add(str(f["data_row"]))
        values[sc].add(_norm(f["value"]))
    return rows, values


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(
    benchmark_rows: list[dict],
    excel_rows: list[dict],
    coverage: set[str],
    dataset_map: dict[str, str] | None = None,
) -> dict:
    dataset_map = dataset_map or {}
    by_cell, by_sc_value = _index_benchmark(benchmark_rows)
    excel_by_cell, excel_by_sc_value = _index_excel_all(excel_rows)

    excel_scanned_sources = {f["source"] for f in excel_rows}
    excel_inc = [f for f in excel_rows if f["type"] in EXCEL_INCONSISTENCY_TYPES]
    inc_cov_rows, inc_cov_values = _index_excel_inc_coverage(excel_inc)

    tp: list[dict] = []
    ambiguous: list[dict] = []
    fp_othertype: list[dict] = []
    fp_clean: list[dict] = []
    out_of_scope: list[dict] = []

    covered_cells: set[tuple[str, str, str]] = set()

    for f in excel_inc:
        if f["source"] not in coverage:
            out_of_scope.append(f)
            continue
        rec, method = _resolve_bench(f, by_cell, by_sc_value)
        entry = {
            "dataset": (rec.get("dataset") if rec else None) or dataset_map.get(f["source"], ""),
            "source": f["source"],
            "column": f["header"],
            "row": f["data_row"],
            "excel_type": f["type"],
            "excel_value": f["value"],
            "excel_suggested_fix": f["suggested_fix"],
            "match_method": method or "",
            "bench_status": rec["status"] if rec else "",
            "bench_error_types": rec["error_types"] if rec else "",
            "bench_original_value": rec["original_value"] if rec else "",
            "bench_corrected_value": rec["corrected_value"] if rec else "",
        }
        if rec is None:
            fp_clean.append(entry)
        elif _is_inconsistency(rec) and rec["status"] == "error":
            tp.append(entry)
        elif _is_inconsistency(rec) and rec["status"] == "ambiguous":
            ambiguous.append(entry)
        else:
            fp_othertype.append(entry)

    # ---- recall side: benchmark inconsistency-error cells ----
    # A benchmark cell is covered when Excel flagged its exact row OR its
    # exact value as an inconsistency in the same (source, column).
    def _inc_covered(r: dict) -> bool:
        sc = (r["source"], r["column"])
        return str(r["row"]) in inc_cov_rows.get(sc, set()) or _norm(
            r["original_value"]
        ) in inc_cov_values.get(sc, set())

    inc_error_cells = [r for r in benchmark_rows if _is_inconsistency(r) and r["status"] == "error"]
    fn_missed: list[dict] = []
    fn_othertype: list[dict] = []
    n_covered_scanned = 0
    n_covered_global = 0
    n_inc_err_scanned = 0

    for r in inc_error_cells:
        key = (r["source"], r["column"], str(r["row"]))
        scanned = r["source"] in excel_scanned_sources
        if scanned:
            n_inc_err_scanned += 1
        if _inc_covered(r):
            covered_cells.add(key)
            n_covered_global += 1
            if scanned:
                n_covered_scanned += 1
            continue
        # Not covered by an Excel inconsistency finding -> FN. Did Excel
        # flag this cell with a different type (positionally or by value)?
        excel_here = excel_by_cell.get(key) or excel_by_sc_value.get(
            (r["source"], r["column"]), {}
        ).get(_norm(r["original_value"]))
        entry = {
            "dataset": r.get("dataset", ""),
            "source": r["source"],
            "column": r["column"],
            "row": r["row"],
            "bench_original_value": r["original_value"],
            "bench_corrected_value": r["corrected_value"],
            "bench_error_types": r["error_types"],
            "excel_scanned_source": scanned,
            "excel_types_here": ";".join(sorted({e["type"] for e in excel_here}))
            if excel_here
            else "",
        }
        if excel_here:
            fn_othertype.append(entry)
        else:
            fn_missed.append(entry)

    # ---- metrics ----
    n_tp = len(tp)
    n_fp_other = len(fp_othertype)
    n_fp_clean = len(fp_clean)
    n_amb = len(ambiguous)
    judged = n_tp + n_fp_other + n_fp_clean  # excludes ambiguous

    prec_strict = n_tp / judged if judged else 0.0
    prec_any_error = (n_tp + n_fp_other) / judged if judged else 0.0

    recall_scanned = n_covered_scanned / n_inc_err_scanned if n_inc_err_scanned else 0.0
    recall_global = n_covered_global / len(inc_error_cells) if inc_error_cells else 0.0

    def f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if (p + r) else 0.0

    # Precision per Excel finding type (strict inconsistency).
    by_type: dict[str, Counter] = defaultdict(Counter)
    for bucket, name in (
        (tp, "tp"),
        (fp_othertype, "fp_othertype"),
        (fp_clean, "fp_clean"),
        (ambiguous, "ambiguous"),
    ):
        for e in bucket:
            by_type[e["excel_type"]][name] += 1
    precision_by_type: dict[str, dict] = {}
    for t, c in by_type.items():
        judged_t = c["tp"] + c["fp_othertype"] + c["fp_clean"]
        precision_by_type[t] = {
            "tp": c["tp"],
            "fp_othertype": c["fp_othertype"],
            "fp_clean": c["fp_clean"],
            "ambiguous": c["ambiguous"],
            "precision_strict": round(c["tp"] / judged_t, 4) if judged_t else 0.0,
        }

    summary = {
        "excel_inconsistency_findings": len(excel_inc),
        "excel_inconsistency_out_of_scope": len(out_of_scope),
        "benchmark_inconsistency_error_cells": len(inc_error_cells),
        "benchmark_inconsistency_error_cells_in_scanned_sources": n_inc_err_scanned,
        "tp_findings": n_tp,
        "fp_othertype": n_fp_other,
        "fp_clean": n_fp_clean,
        "ambiguous_excluded": n_amb,
        "fn_othertype": len(fn_othertype),
        "fn_missed": len(fn_missed),
        "covered_cells_scanned": n_covered_scanned,
        "covered_cells_global": n_covered_global,
        "precision_strict_inconsistency": round(prec_strict, 4),
        "precision_any_error": round(prec_any_error, 4),
        "recall_scanned_sources": round(recall_scanned, 4),
        "recall_global": round(recall_global, 4),
        "f1_strict_scanned": round(f1(prec_strict, recall_scanned), 4),
        "match_method_counts": dict(Counter(e["match_method"] for e in tp)),
        "excel_inconsistency_type_breakdown": dict(Counter(f["type"] for f in excel_inc)),
        "precision_by_excel_type": precision_by_type,
    }

    return {
        "summary": summary,
        "tp": tp,
        "ambiguous": ambiguous,
        "fp_othertype": fp_othertype,
        "fp_clean": fp_clean,
        "fn_othertype": fn_othertype,
        "fn_missed": fn_missed,
        "out_of_scope": out_of_scope,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _per_dataset(result: dict) -> dict[str, Counter]:
    agg: dict[str, Counter] = defaultdict(Counter)
    for bucket in ("tp", "fp_othertype", "fp_clean", "ambiguous"):
        for e in result[bucket]:
            agg[e.get("dataset") or "?"][bucket] += 1
    for bucket in ("fn_othertype", "fn_missed"):
        for e in result[bucket]:
            agg[e.get("dataset") or "?"][bucket] += 1
    return agg


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--benchmark", type=Path, default=Path("benchmark/merged_v3/benchmark_cell.csv")
    )
    ap.add_argument("--predictions", type=Path, default=Path("external/excel_finding.csv"))
    ap.add_argument(
        "--coverage-zip",
        type=Path,
        default=Path("benchmark/merged_v3/sources_normalized.zip"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/excel_finding_vs_merged_v3_inconsistency"),
    )
    args = ap.parse_args()

    benchmark_rows = load_benchmark_cells(args.benchmark)
    excel_rows = load_excel_findings(args.predictions)
    coverage = load_coverage(args.coverage_zip, benchmark_rows)
    dataset_map = load_dataset_map(args.coverage_zip, benchmark_rows)

    result = compare(benchmark_rows, excel_rows, coverage, dataset_map)
    summary = result["summary"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for bucket in (
        "tp",
        "ambiguous",
        "fp_othertype",
        "fp_clean",
        "fn_othertype",
        "fn_missed",
    ):
        _write_csv(args.out_dir / f"{bucket}.csv", result[bucket])

    lines: list[str] = []
    lines.append("Excel findings vs benchmark/merged_v3 -- INCONSISTENCY")
    lines.append("=" * 60)
    for k, v in summary.items():
        if k == "precision_by_excel_type":
            lines.append("precision_by_excel_type:")
            for t, d in v.items():
                lines.append(f"    {t}: {d}")
            continue
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Per-dataset (dataset: bucket counts)")
    for ds, c in sorted(_per_dataset(result).items()):
        lines.append(f"  {ds}: {dict(c)}")
    report = "\n".join(lines)
    (args.out_dir / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWrote review CSVs + summary.txt to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
