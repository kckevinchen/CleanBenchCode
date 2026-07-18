"""Adapter: materialize CleanBench into REIN dataset directories.

For every benchmark *source* CSV that has at least one labeled cell, this writes
``datasets/<ds>/`` containing:

* ``dirty.csv``          the source table (read the REIN way, rewritten verbatim).
* ``clean.csv``          dirty with ``corrected_value`` applied at ``error`` cells
                         (used only as raha's labeling oracle).
* ``actual_errors.csv``  ``i,j,DUMMY`` rows for ``error`` cells -- REIN reads this
                         as ground truth, so detectors are scored against OUR
                         labels (no dirty/clean diff, no PostgreSQL needed).
* ``error_rate.json``    ``len(actual_errors) / df.size`` (REIN-native companion).
* ``gt_cells.jsonl``     ``{i,j,status,error_types,original_value}`` for every
                         benchmark cell of the source (richer GT for our scorer).

A top-level ``datasets/_manifest.json`` records the source<->dataset mapping and
per-dataset stats (including value/column mismatches and missing sources).

Run (from repo root, main venv):
    python3 external/rein/prepare_datasets.py            # full benchmark
    python3 external/rein/prepare_datasets.py --limit 5  # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

# Allow ``python3 external/rein/prepare_datasets.py`` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def build_dataset(
    ds_name: str,
    source: str,
    source_path: Path,
    records: list[dict],
    out_dir: Path,
) -> dict:
    """Write one REIN dataset directory; return a manifest entry."""
    df = C.load_rein_df(source_path)
    n_rows, n_cols = df.shape
    col_index = {c: i for i, c in enumerate(df.columns)}
    # Whitespace-tolerant fallback: the benchmark stores trimmed column names,
    # but a few sanitized source headers retain leading/trailing spaces (e.g.
    # "Result "). Map a stripped name to its column index only when that stripped
    # form is unambiguous. Scoring is positional (row, col int index), so using
    # the correct index recovers these cells with no risk of mismatch.
    _stripped_counts = Counter(str(c).strip() for c in df.columns)
    stripped_index = {
        str(c).strip(): i
        for i, c in enumerate(df.columns)
        if _stripped_counts[str(c).strip()] == 1
    }

    clean = df.copy()
    error_cells: list[tuple[int, int]] = []
    gt_lines: list[str] = []
    n_value_mismatch = 0
    n_oob = 0  # row/column out of bounds

    for rec in records:
        row = rec["row"]
        col = rec["column"]
        status = rec.get("status", rec.get("label"))
        if col in col_index:
            j = col_index[col]
        elif str(col).strip() in stripped_index:
            j = stripped_index[str(col).strip()]
        else:
            j = None
        if j is None or row < 0 or row >= n_rows:
            n_oob += 1
            continue
        # Sanity-check the cell value against the benchmark's record.
        if str(df.iat[row, j]) != str(rec.get("original_value", df.iat[row, j])):
            n_value_mismatch += 1

        gt_lines.append(
            json.dumps(
                {
                    "i": row,
                    "j": j,
                    "status": status,
                    "error_types": rec.get("error_types", []),
                    "original_value": rec.get("original_value"),
                },
                ensure_ascii=False,
            )
        )

        if status == C.STATUS_ERROR:
            error_cells.append((row, j))
            corrected = rec.get("corrected_value")
            if C.is_usable_correction(corrected):
                clean.iat[row, j] = str(corrected)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop any stale raha/Baran strategy caches from a prior materialization:
    # those .dictionary files are keyed by strategy config, NOT data content, so
    # raha would silently reuse feature vectors computed on the OLD dirty.csv.
    for stale in out_dir.glob("raha-baran-results-*"):
        if stale.is_dir():
            shutil.rmtree(stale)
    df.to_csv(out_dir / "dirty.csv", index=False, encoding="utf-8")
    clean.to_csv(out_dir / "clean.csv", index=False, encoding="utf-8")

    with open(out_dir / "actual_errors.csv", "w", encoding="utf-8") as f:
        for (i, j) in error_cells:
            f.write(f"{i},{j},DUMMY VALUE\n")

    error_rate = (len(error_cells) / df.size) if df.size else 0.0
    with open(out_dir / "error_rate.json", "w", encoding="utf-8") as f:
        json.dump(error_rate, f)

    with open(out_dir / "gt_cells.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(gt_lines) + ("\n" if gt_lines else ""))

    return {
        "dataset": ds_name,
        "source": source,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_cells_labeled": len(records),
        "n_error_cells": len(error_cells),
        "n_ambiguous_cells": len(records) - len(error_cells) - n_oob,
        "n_value_mismatch": n_value_mismatch,
        "n_out_of_bounds": n_oob,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-dir", default=str(C.DEFAULT_BENCHMARK_DIR))
    ap.add_argument("--sources-dir", default=str(C.DEFAULT_SOURCES_DIR))
    ap.add_argument("--out", default=str(C.DATASETS_DIR), help="datasets output dir")
    ap.add_argument("--limit", type=int, default=0, help="only first N sources (0=all)")
    ap.add_argument("--include-clean-universe", action="store_true",
                    help="also materialize every OTHER CSV in --sources-dir as a clean "
                         "dataset (no labeled errors). Required to score precision over "
                         "the full benchmark universe (flags on clean tables = FPs).")
    args = ap.parse_args()

    sources_dir = Path(args.sources_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cells = C.read_benchmark_cells(args.benchmark_dir)
    grouped = C.cells_by_source(cells)
    sources = sorted(grouped)
    if args.limit:
        sources = sources[: args.limit]

    manifest: dict[str, dict] = {}
    missing: list[str] = []
    seen_slugs: dict[str, str] = {}
    totals = {"datasets": 0, "error_cells": 0, "ambiguous": 0, "mismatch": 0, "oob": 0,
              "clean_datasets": 0}

    for source in sources:
        src_path = sources_dir / source
        if not src_path.exists():
            missing.append(source)
            continue
        ds_name = C.slugify_dataset(source)
        if ds_name in seen_slugs:
            raise SystemExit(
                f"Dataset slug collision: {ds_name!r} from {source!r} and {seen_slugs[ds_name]!r}"
            )
        seen_slugs[ds_name] = source

        entry = build_dataset(ds_name, source, src_path, grouped[source], out_root / ds_name)
        manifest[ds_name] = entry
        totals["datasets"] += 1
        totals["error_cells"] += entry["n_error_cells"]
        totals["ambiguous"] += entry["n_ambiguous_cells"]
        totals["mismatch"] += entry["n_value_mismatch"]
        totals["oob"] += entry["n_out_of_bounds"]

    # Optionally materialize every remaining CSV in the universe as a CLEAN
    # dataset (no labeled errors) so detectors can be scored for false positives
    # over the whole benchmark, not just the error-bearing tables.
    if args.include_clean_universe and not args.limit:
        labeled_sources = set(grouped)
        for src_path in sorted(sources_dir.glob("*.csv")):
            source = src_path.name
            if source in labeled_sources:
                continue
            ds_name = C.slugify_dataset(source)
            if ds_name in seen_slugs:
                continue
            seen_slugs[ds_name] = source
            entry = build_dataset(ds_name, source, src_path, [], out_root / ds_name)
            entry["clean_universe"] = True
            manifest[ds_name] = entry
            totals["datasets"] += 1
            totals["clean_datasets"] += 1

    # A full materialization defines the complete output universe. Remove
    # directories left by an older benchmark so ad-hoc scans cannot include
    # datasets that are absent from the new manifest.
    if not args.limit:
        for child in out_root.iterdir():
            if child.is_dir() and child.name not in manifest:
                shutil.rmtree(child)

    with open(out_root / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"datasets": manifest, "missing_sources": missing}, f, indent=2)

    print("=== prepare_datasets summary ===")
    print(f"datasets prepared : {totals['datasets']}  "
          f"(labeled {totals['datasets'] - totals['clean_datasets']}, "
          f"clean-universe {totals['clean_datasets']})")
    print(f"error cells (GT)  : {totals['error_cells']}")
    print(f"ambiguous cells   : {totals['ambiguous']}")
    print(f"value mismatches  : {totals['mismatch']} (benchmark value != source cell)")
    print(f"out-of-bounds     : {totals['oob']} (row/col not found in source)")
    print(f"missing sources   : {len(missing)} {missing if missing else ''}")
    print(f"manifest          : {out_root / '_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
