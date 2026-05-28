"""
Investigate whether the inconsistency benchmark is missing instances.

For each predicted pair from
``external/model_inconsistency_result/`` that is NOT scored as a TP
against ``benchmark/benchmark_inconsistency_*.jsonl``, verify each
endpoint value against the actual source CSVs (searching across
``data/source_inconsistency_bing21`` plus the curated/sampled siblings
to absorb any minor version drift) and classify the pair by how the
benchmark currently labels each endpoint.

Categories
----------
``pair_gap_both_unlabeled``
    Both endpoints exist in the source column but neither has any
    benchmark row on (source, column). Strongest candidate for a true
    benchmark gap: the relationship is plausible AND the values
    themselves were never audited.

``pair_gap_one_known_error``
    One endpoint is labeled ``error``, the other is unlabeled. The
    benchmark knew about one side but never paired it with this
    sibling.

``ambiguous_only``
    Both/one endpoint(s) labeled ``ambiguous``. The annotator was
    unsure; worth re-reviewing.

``already_known_both_error_different_canonical``
    Both endpoints labeled ``error`` already, but each is mapped to a
    *third* canonical form. Not a benchmark gap, but informative: the
    model is identifying the link between two known errors.

``column_gap_file_in_bench``
    Benchmark covers other columns of this file but did not audit this
    column. Pair endpoints both exist in source. Clean candidate gap
    at the column level.

``column_gap_file_not_in_bench``
    Whole file is absent from the benchmark.

``not_in_any_source``
    Neither endpoint exists in any available source folder -- usually a
    model hallucination.

Output
------
A single JSON document at ``<out>/benchmark_gap_report.json`` plus a
human-readable Markdown summary at ``<out>/benchmark_gap_report.md``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Folders scanned for source CSVs, in priority order. The first one
# whose copy contains the column/value wins for "primary"; the remaining
# folders are still consulted to confirm the value appears in *some*
# version of the table (the user noted upstream version drift is
# possible).
DEFAULT_DATA_FOLDERS = [
    "data/source_inconsistency_bing21",
    "data/curated_bing21_200",
    "data/sampled_bing200_trim",
    "data/bing21_grouped",
    "data/curated_bing21_min_filtered",
    "data/curated_bing21_min_multi_table",
]


def _norm(v) -> str:
    return " ".join((str(v or "")).split())


def _candidates(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [_norm(x) for x in v if isinstance(x, str) and _norm(x)]
    if isinstance(v, str):
        n = _norm(v)
        return [n] if n else []
    return []


# Label "worst-wins" ranking, used when a single value appears in
# multiple benchmark rows (e.g. as ``original_value`` in some rows and
# ``corrected_value`` in others). ``error`` outranks ``ambiguous`` which
# outranks the implicit ``clean`` we tag corrected_values with.
LABEL_RANK = {"error": 3, "ambiguous": 2, "clean": 1}


def _upgrade(d: dict, key: str, label: str) -> None:
    cur = d.get(key)
    if cur is None or LABEL_RANK.get(label, 0) > LABEL_RANK.get(cur, 0):
        d[key] = label


def _load_predictions(pred_dir: Path) -> dict:
    """Return ``{(file_id, col_header, (raw_a, raw_b)) -> entry}``.

    Verbatim raw values are kept in the key so a whitespace-only pair
    (e.g. ``'X  Y'`` vs ``'X Y'``) is NOT deduplicated with a
    non-trivial pair on the same column.
    """
    preds: dict = {}
    for p in sorted(pred_dir.iterdir()):
        if not p.is_file() or p.suffix not in {".tsv", ".txt"}:
            continue
        with p.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                try:
                    syn = json.loads(r["synonyms"])
                    a, b = syn[0]
                except Exception:  # noqa: BLE001
                    continue
                key = (r["file_id"], r["col_header"], (a, b))
                entry = preds.setdefault(
                    key,
                    {
                        "pair": (a, b),
                        "fid": r["file_id"],
                        "col": r["col_header"],
                        "trivial": r.get("Trivial-diff", ""),
                        "runs": set(),
                        "min_token_prob": float(r.get("pair_min_token_probs") or 0.0),
                    },
                )
                entry["runs"].add(p.name)
    return preds


def _index_benchmark(bench_path: Path) -> tuple[set, dict, dict]:
    """Return ``(bench_sc, bench_pair_keys, bench_value_status)``.

    * ``bench_sc``           -- set of (source, column) keys present in
      the benchmark at all.
    * ``bench_pair_keys``    -- {(source, column) -> set[frozenset(orig, corr)]}
      for ``label='error'`` rows whose corrected_value is non-empty.
      Used to short-circuit prediction pairs that are already TPs.
    * ``bench_value_status`` -- {(source, column) -> {norm_value -> worst_label}}.
      ``corrected_value`` strings are tagged ``'clean'`` unless they
      also appear as an ``original_value`` with a higher label.
    """
    bench_sc: set = set()
    bench_pair_keys: dict = defaultdict(set)
    bench_value_status: dict = defaultdict(dict)
    with bench_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sc = (r["source"], r["column"])
            bench_sc.add(sc)
            label = r.get("label") or "unknown"
            orig = _norm(r.get("original_value"))
            if orig:
                _upgrade(bench_value_status[sc], orig, label)
            for corr in _candidates(r.get("corrected_value")):
                _upgrade(bench_value_status[sc], corr, "clean")
            if label == "error" and orig:
                for corr in _candidates(r.get("corrected_value")):
                    if corr and corr != orig:
                        bench_pair_keys[sc].add(frozenset({orig, corr}))
    return bench_sc, bench_pair_keys, bench_value_status


def _col_values(path: str, col: str) -> list[tuple[int, str]] | None:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            if not header or col not in header:
                return None
            idx = header.index(col)
            return [
                (i, row[idx] if idx < len(row) else "") for i, row in enumerate(rdr)
            ]
    except Exception:  # noqa: BLE001
        return None


def _classify(pair_info: dict, status_a: str, status_b: str, file_in_bench: bool) -> str:
    statuses = {status_a, status_b}
    if "col_not_in_bench" in statuses:
        return "column_gap_file_in_bench" if file_in_bench else "column_gap_file_not_in_bench"
    if statuses == {"error"}:
        return "already_known_both_error_different_canonical"
    if "error" in statuses:
        return "pair_gap_one_known_error"
    if "ambiguous" in statuses:
        return "ambiguous_only"
    return "pair_gap_both_unlabeled"


def investigate(
    benchmark: Path,
    predictions: Path,
    data_folders: list[str],
) -> dict:
    preds = _load_predictions(predictions)
    bench_sc, bench_pair_keys, bench_value_status = _index_benchmark(benchmark)
    bench_files = {sc[0] for sc in bench_sc}

    gaps: list[dict] = []
    not_in_any_source: list[dict] = []
    already_tp: list[dict] = []

    for key, p in preds.items():
        fid, col, (raw_a, raw_b) = key
        a_n, b_n = _norm(raw_a), _norm(raw_b)

        # Already a TP?
        if frozenset({a_n, b_n}) in bench_pair_keys.get((fid, col), set()):
            already_tp.append({"fid": fid, "col": col, "pair": (raw_a, raw_b),
                                "runs": sorted(p["runs"])})
            continue

        # Scan source folders.
        found_a: list = []
        found_b: list = []
        primary_path: str | None = None
        for root in data_folders:
            for path in sorted(glob.glob(f"{root}/*/{fid}")):
                rows = _col_values(path, col)
                if rows is None:
                    continue
                ra = [i for i, v in rows if _norm(v) == a_n]
                rb = [i for i, v in rows if _norm(v) == b_n]
                if ra or rb:
                    if primary_path is None:
                        primary_path = path
                    if ra:
                        found_a.append({"folder": Path(root).name, "count": len(ra), "row_sample": ra[:6]})
                    if rb:
                        found_b.append({"folder": Path(root).name, "count": len(rb), "row_sample": rb[:6]})

        if not found_a and not found_b:
            not_in_any_source.append({
                "fid": fid, "col": col, "pair": [raw_a, raw_b],
                "trivial": p["trivial"], "runs": sorted(p["runs"]),
                "min_token_prob": p["min_token_prob"],
            })
            continue

        bvs = bench_value_status.get((fid, col), {})
        col_in_bench = (fid, col) in bench_sc
        status_a = bvs.get(a_n, "unlabeled" if col_in_bench else "col_not_in_bench")
        status_b = bvs.get(b_n, "unlabeled" if col_in_bench else "col_not_in_bench")

        gaps.append({
            "fid": fid, "col": col, "pair": [raw_a, raw_b],
            "pair_norm": [a_n, b_n],
            "status_a": status_a, "status_b": status_b,
            "category": _classify(p, status_a, status_b, fid in bench_files),
            "trivial": p["trivial"], "runs": sorted(p["runs"]),
            "min_token_prob": p["min_token_prob"],
            "a_in_folders": found_a, "b_in_folders": found_b,
            "primary_path": primary_path,
            "a_in_source": bool(found_a),
            "b_in_source": bool(found_b),
        })

    cat_counter = Counter(g["category"] for g in gaps)
    return {
        "predictions_total_unique": len(preds),
        "already_tp": already_tp,
        "gaps": gaps,
        "not_in_any_source": not_in_any_source,
        "category_counts": dict(cat_counter),
    }


def write_markdown(report: dict, out_path: Path, benchmark: Path, predictions: Path) -> None:
    lines: list[str] = []
    push = lines.append

    push(f"# Benchmark gap investigation\n")
    push(f"Benchmark: `{benchmark}`")
    push(f"Predictions dir: `{predictions}`\n")

    n_total = report["predictions_total_unique"]
    n_tp = len(report["already_tp"])
    n_gap = len(report["gaps"])
    n_hallu = len(report["not_in_any_source"])
    push(f"Total unique predicted pairs (verbatim): **{n_total}**  ")
    push(f"  - Already TP (matched a bench error pair): {n_tp}  ")
    push(f"  - Verified against source (gap candidates): {n_gap}  ")
    push(f"  - Not in any source folder (likely hallucination): {n_hallu}\n")

    push(f"## Gap-candidate category counts\n")
    for c, n in sorted(report["category_counts"].items(), key=lambda kv: -kv[1]):
        push(f"- `{c}`: **{n}**")
    push("")

    order = [
        ("pair_gap_both_unlabeled",
         "Strongest gap candidates: both pair values exist in the source column "
         "but neither has any benchmark row."),
        ("pair_gap_one_known_error",
         "One value is a known error in the bench, the other is unlabeled. "
         "The benchmark may have missed the relationship."),
        ("column_gap_file_in_bench",
         "Benchmark covers other columns of this file, but did not audit this "
         "column. Both pair values exist in the source."),
        ("column_gap_file_not_in_bench",
         "Entire file is absent from the benchmark."),
        ("ambiguous_only",
         "Bench labeled the values 'ambiguous'. Worth re-reviewing."),
        ("already_known_both_error_different_canonical",
         "Both values are already bench errors, but each maps to a *third* "
         "canonical form. Not a gap; informational."),
    ]
    for cat, descr in order:
        items = [g for g in report["gaps"] if g["category"] == cat]
        if not items:
            continue
        push(f"\n## {cat}  ({len(items)})\n")
        push(f"{descr}\n")
        items.sort(key=lambda g: (g["fid"], g["col"]))
        for g in items:
            push(
                f"- `{g['fid']}` / `{g['col']}` -- "
                f"`{g['pair_norm'][0]}` ({g['status_a']}) vs "
                f"`{g['pair_norm'][1]}` ({g['status_b']})  "
            )
            a_sum = ", ".join(f"{f['folder']}({f['count']})" for f in g["a_in_folders"])
            b_sum = ", ".join(f"{f['folder']}({f['count']})" for f in g["b_in_folders"])
            push(
                f"  - Trivial-diff: `{g['trivial']}` -- runs: {g['runs']} -- "
                f"min_token_prob: {g['min_token_prob']:.3f}"
            )
            push(f"  - a in: {a_sum or '(none)'}")
            push(f"  - b in: {b_sum or '(none)'}")
            primary = g.get("primary_path")
            if primary:
                push(f"  - primary CSV: `{primary}`")

    if report["not_in_any_source"]:
        push(f"\n## not_in_any_source  ({len(report['not_in_any_source'])})\n")
        push(
            "Neither value exists in any inspected source-folder copy. "
            "Most likely a model hallucination -- safe to ignore for "
            "benchmark-gap purposes.\n"
        )
        for g in report["not_in_any_source"]:
            push(
                f"- `{g['fid']}` / `{g['col']}` -- pair=`{g['pair']!r}` -- "
                f"runs={g['runs']} -- trivial={g['trivial']}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path,
                    help="Directory of external prediction TSVs.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--data-folder", action="append", default=None,
                    help="Source-CSV folder to scan. May be passed multiple "
                         "times. Defaults to the standard bing21 set.")
    args = ap.parse_args(argv)

    folders = args.data_folder or DEFAULT_DATA_FOLDERS
    report = investigate(args.benchmark, args.predictions, folders)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "benchmark_gap_report.json"
    md_path = args.out_dir / "benchmark_gap_report.md"

    # JSON dump with sorted keys for stable diffs.
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)

    write_markdown(report, md_path, args.benchmark, args.predictions)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print()
    print("Summary:")
    print(f"  unique predicted pairs   : {report['predictions_total_unique']}")
    print(f"  already TP               : {len(report['already_tp'])}")
    print(f"  verified gap candidates  : {len(report['gaps'])}")
    print(f"  not in any source folder : {len(report['not_in_any_source'])}")
    print()
    print("By category:")
    for c, n in sorted(report["category_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:55} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
