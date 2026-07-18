"""Emit a SANITISED copy of the CleanBench source corpus.

Produces the source CSVs paired with ``benchmark/CleanBench``
with every operator EXCLUSION applied, so a downstream data-cleaning step
sees exactly the slices the benchmark actually scores:

  * ``scope=csv``     -> the whole file is OMITTED from the output.
  * ``scope=rows``    -> the listed 0-based positional rows are dropped.
  * ``scope=columns`` -> the listed columns are dropped.

Exclusions are gathered as the UNION across every ``(pool x error_type)``
run in :data:`experiments.export_cleanbench.REGISTRY` (the same runs
that feed ``benchmark_cell.json`` / ``benchmark_row.json``), so a cell
excluded under ANY error type is removed once, run-agnostic.

The registered PBI pool currently contains 504 CSVs and writes 500 after four
whole-file exclusions. If a future exclusion drops it below 500, the exporter
can deterministically backfill with readable, non-empty CSVs from
``data/curated_pbi_mt100_dedup`` that are unique by basename and SHA-256.
The Excel/Bing21 pool writes 500 after 17 whole-file exclusions, including seven
reviewed-clean corpus-balance files with no benchmark cell or row records.

The inter-table sheets carry no exclusions, but CleanBench's
``inter_table_inconsistency`` findings reference a handful of them. To keep
the corpus self-contained for that task, the FULL workbooks involved in any
confirmed/unsure inter-table finding (from
:data:`experiments.export_row_benchmark.INTER_TABLE_REGISTRY`) are copied
verbatim. They belong to the SAME dataset as the rest of that corpus
(``bing21_500``) and are written by their normal sheet filename alongside the
pool's flat CSVs -- there is no separate ``mt100`` subdir.

Layout (top-level dir per ``dataset`` label, matching the benchmark)::

    <out>/pbi_500/<basename>.csv
    <out>/bing21_500/<basename>.csv          (incl. the inter-table sheets)
    <out>/_ROW_MAP.json                      original rows for filtered CSVs
    <out>/MANIFEST.json         what was omitted / row-dropped / col-dropped
    <out>.zip                   the same tree, zipped

Untouched files are copied as RAW BYTES so quirky encodings / quoting
survive; files with row/column drops are re-emitted via the same
``src.core.data_loader.load_csv`` frame the benchmark's ``original_value``
is read from, so the surviving cells still line up.

Read-only w.r.t. the runs and the source corpora. Run::

    python -m experiments.export_sanitized_sources \
        --out benchmark/CleanBench/sanitized_sources
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from experiments.export_cleanbench import DATASET_OF, OUTPUTS, REGISTRY
from experiments.export_row_benchmark import INTER_TABLE_REGISTRY, _materialize_inter_table
from src.core.data_loader import load_csv
from src.labeling.app import RunContext
from src.labeling.schema import ExclusionScope

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The two per-cell / per-row pools carry all the exclusions; the inter-table
# corpus is handled separately (whole involved workbooks, verbatim).
SANITISE_POOLS = ["pbi_500", "excel_500"]
PBI_BACKFILL_ROOT = PROJECT_ROOT / "data" / "curated_pbi_mt100_dedup"
PBI_MINIMUM_WRITTEN = 500


def _gather_pool(
    pool: str,
) -> Tuple[Dict[str, Path], Set[str], Dict[str, Set[int]], Dict[str, Set[str]]]:
    """Union a pool's sources + exclusions across all its REGISTRY runs.

    Returns ``(sources, excl_files, excl_rows, excl_cols)`` where
    ``sources`` maps each source basename to its on-disk path (first seen).
    """
    sources: Dict[str, Path] = {}
    excl_files: Set[str] = set()
    excl_rows: Dict[str, Set[int]] = defaultdict(set)
    excl_cols: Dict[str, Set[str]] = defaultdict(set)
    for _label, _types, run_id in REGISTRY[pool]:
        run_dir = OUTPUTS / run_id
        if not run_dir.is_dir():
            print(f"  WARN missing run dir: {pool}/{run_id}")
            continue
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        for p in summary.get("config", {}).get("csv_paths") or []:
            sources.setdefault(Path(p).name, Path(p))
        ctx = RunContext(run_dir)
        for e in ctx.exclusions_store.active():
            src = e.source or ""
            if e.scope == ExclusionScope.CSV:
                excl_files.add(src)
            elif e.scope == ExclusionScope.ROWS:
                for r in e.rows or []:
                    try:
                        excl_rows[src].add(int(r))
                    except (TypeError, ValueError):
                        continue
            elif e.scope == ExclusionScope.COLUMNS:
                for c in e.columns or []:
                    if c is not None:
                        excl_cols[src].add(str(c))
    return sources, excl_files, excl_rows, excl_cols


def _write_sanitised_csv(
    src_path: Path,
    out_path: Path,
    drop_rows: Set[int],
    drop_cols: Set[str],
) -> Tuple[str, Optional[List[int]]]:
    """Emit one source and return its action plus any exported-row mapping."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not drop_rows and not drop_cols:
        shutil.copy2(src_path, out_path)
        return "copy", None
    frame = load_csv(src_path)
    row_map: Optional[List[int]] = None
    if drop_cols:
        keep = [c for c in frame.columns if c not in drop_cols]
        frame = frame.loc[:, keep]
    if drop_rows:
        keep_idx = [i for i in range(len(frame)) if i not in drop_rows]
        frame = frame.iloc[keep_idx]
        row_map = keep_idx
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out_path)
    return "rewrite", row_map


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_pbi_backfill(
    sources: Dict[str, Path],
    needed: int,
) -> List[Tuple[Path, str]]:
    """Select readable MT100 CSVs unique from the registered PBI corpus."""
    if needed <= 0:
        return []
    if not PBI_BACKFILL_ROOT.is_dir():
        raise FileNotFoundError(f"PBI backfill corpus not found: {PBI_BACKFILL_ROOT}")

    missing_sources = [path for path in sources.values() if not path.is_file()]
    if missing_sources:
        sample = ", ".join(str(path) for path in missing_sources[:3])
        raise FileNotFoundError(f"registered PBI source(s) missing: {sample}")

    used_names = set(sources)
    used_hashes = {_sha256(path) for path in sources.values()}
    used_workbooks: Set[str] = set()
    selected: List[Tuple[Path, str]] = []
    deferred: List[Tuple[Path, str]] = []

    candidates = sorted(
        PBI_BACKFILL_ROOT.rglob("*.csv"),
        key=lambda path: path.relative_to(PBI_BACKFILL_ROOT).as_posix(),
    )
    for candidate in candidates:
        if candidate.name in used_names:
            continue
        file_hash = _sha256(candidate)
        if file_hash in used_hashes:
            continue

        frame = load_csv(candidate)
        if frame.empty or not len(frame.columns):
            continue

        used_names.add(candidate.name)
        used_hashes.add(file_hash)
        item = (candidate, file_hash)
        workbook = candidate.relative_to(PBI_BACKFILL_ROOT).parts[0]
        if workbook in used_workbooks:
            deferred.append(item)
        else:
            used_workbooks.add(workbook)
            selected.append(item)
        if len(selected) >= needed:
            break

    if len(selected) < needed:
        selected.extend(deferred[: needed - len(selected)])
    if len(selected) < needed:
        raise RuntimeError(
            f"only {len(selected)} unique PBI backfill CSVs available; need {needed}"
        )
    return selected[:needed]


def _inter_table_workbooks() -> Dict[str, Path]:
    """Map each workbook (dir name) involved in a confirmed/unsure
    inter-table finding to its ``source_root`` on disk."""
    records = _materialize_inter_table()
    wb_root: Dict[str, Path] = {}
    root_by_dataset = {spec["dataset"]: Path(spec["source_root"]) for spec in INTER_TABLE_REGISTRY}
    for rec in records:
        wb = rec.get("workbook")
        root = root_by_dataset.get(rec.get("dataset"))
        if wb and root is not None:
            wb_root[wb] = root
    return wb_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "benchmark" / "CleanBench" / "sanitized_sources"),
        help="output directory (a sibling .zip is written too)",
    )
    ap.add_argument(
        "--no-zip",
        action="store_true",
        help="skip the .zip bundle",
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "pools": {},
        "inter_table_workbooks": {},
        "totals": {},
    }
    n_written = 0
    n_omitted = 0
    n_rewritten = 0
    row_maps: List[Dict[str, Any]] = []

    # ---- the two exclusion-bearing pools -----------------------------
    for pool in SANITISE_POOLS:
        dataset = DATASET_OF[pool]
        sources, excl_files, excl_rows, excl_cols = _gather_pool(pool)
        pool_dir = out_dir / dataset
        omitted: List[str] = []
        rewritten: Dict[str, Dict[str, List]] = {}
        written = 0
        for basename, src_path in sorted(sources.items()):
            if basename in excl_files:
                omitted.append(basename)
                continue
            if not src_path.exists():
                print(f"  WARN missing source on disk: {src_path}")
                continue
            drop_rows = excl_rows.get(basename, set())
            drop_cols = excl_cols.get(basename, set())
            action, row_map = _write_sanitised_csv(
                src_path,
                pool_dir / basename,
                drop_rows,
                drop_cols,
            )
            written += 1
            if action == "rewrite":
                n_rewritten += 1
                rewritten[basename] = {
                    "dropped_rows": sorted(drop_rows),
                    "dropped_columns": sorted(drop_cols),
                }
            if row_map is not None:
                row_maps.append(
                    {
                        "source": basename,
                        "path": f"{dataset}/{basename}",
                        "exported_to_source": row_map,
                    }
                )

        registered_written = written
        backfilled: List[Dict[str, Any]] = []
        if pool == "pbi_500":
            needed = max(0, PBI_MINIMUM_WRITTEN - written)
            for src_path, file_hash in _select_pbi_backfill(sources, needed):
                _write_sanitised_csv(
                    src_path,
                    pool_dir / src_path.name,
                    set(),
                    set(),
                )
                written += 1
                backfilled.append(
                    {
                        "source": src_path.relative_to(PBI_BACKFILL_ROOT).as_posix(),
                        "output": src_path.name,
                        "sha256": file_hash,
                    }
                )

        n_written += written
        n_omitted += len(omitted)
        manifest["pools"][pool] = {
            "dataset": dataset,
            "source_total": len(sources),
            "registered_written": registered_written,
            "written": written,
            "omitted_csvs": sorted(omitted),
            "rewritten": rewritten,
        }
        if pool == "pbi_500":
            manifest["pools"][pool]["backfill"] = {
                "source_root": str(PBI_BACKFILL_ROOT),
                "minimum_written": PBI_MINIMUM_WRITTEN,
                "selection": (
                    "readable non-empty CSVs unique by basename and SHA-256; "
                    "lexicographic order, preferring distinct workbooks"
                ),
                "role": "source-only corpus backfill; no gold cell/row labels",
                "files": backfilled,
            }
        print(
            f"{pool} -> {dataset}/: {written} written "
            f"({len(omitted)} CSVs omitted, {len(rewritten)} row/col-trimmed, "
            f"{len(backfilled)} backfilled)"
        )

    # ---- inter-table sheets: backfill any not already in the pool -----
    # The involved workbooks' sheets belong to the SAME dataset as the rest
    # of that corpus (bing21_500). Since these sheets' single-column/relation
    # detections are merged into the excel_500 (=bing21_500) runs, the pool
    # loop above already writes them by their normal filename. This block is
    # now a defensive BACKFILL: it adds any involved-workbook sheet the pool
    # did NOT already write (e.g. if the merge hasn't happened yet), keeping
    # the corpus self-contained for the inter_table task. Sheets already
    # present (from the pool) are skipped -- the pool copy (exclusion-applied)
    # is canonical -- so there is never a duplicate or a separate mt100 dir.
    it_dataset = INTER_TABLE_REGISTRY[0]["dataset"] if INTER_TABLE_REGISTRY else "bing21_500"
    wb_root = _inter_table_workbooks()
    it_dir = out_dir / it_dataset
    it_backfilled = 0
    it_skipped = 0
    it_sheets: Dict[str, List[str]] = {}
    # Basenames already written to this dataset dir by the pool loop above.
    seen_names: Set[str] = set()
    if it_dir.is_dir():
        seen_names.update(p.name for p in it_dir.glob("*.csv"))
    for wb, root in sorted(wb_root.items()):
        wb_src = root / wb
        if not wb_src.is_dir():
            print(f"  WARN inter-table workbook missing: {wb_src}")
            continue
        sheets = sorted(p.name for p in wb_src.glob("*.csv"))
        for sheet in sheets:
            if sheet in seen_names:
                it_skipped += 1
                continue
            _write_sanitised_csv(wb_src / sheet, it_dir / sheet, set(), set())
            seen_names.add(sheet)
            it_backfilled += 1
        it_sheets[wb] = sheets
    n_written += it_backfilled
    manifest["inter_table_workbooks"] = {
        "dataset": it_dataset,
        "workbooks": it_sheets,
        "sheets_backfilled": it_backfilled,
        "sheets_from_pool": it_skipped,
        "note": (
            f"folded into the {it_dataset} corpus by sheet filename "
            f"(no separate mt100 subdir); sheets already written by the pool "
            f"are not duplicated"
        ),
    }
    print(
        f"inter-table -> {it_dataset}/: {it_backfilled} backfilled, "
        f"{it_skipped} already from pool ({len(it_sheets)} workbook(s))"
    )

    manifest["totals"] = {
        "csvs_written": n_written,
        "csvs_omitted_fully_excluded": n_omitted,
        "csvs_row_or_col_trimmed": n_rewritten,
    }
    if row_maps:
        (out_dir / "_ROW_MAP.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "description": (
                        "For each row-filtered CSV, exported_to_source[i] is "
                        "the original zero-based source row for exported data row i."
                    ),
                    "sources": row_maps,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- zip -------------------------------------------------------
    if not args.no_zip:
        zip_path = out_dir.with_suffix(".zip")
        tmp_zip = zip_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(out_dir).as_posix())
        tmp_zip.replace(zip_path)
        print(f"\nzip: {zip_path}")

    print(f"\nwrote {out_dir}/  ({n_written} CSVs total)")
    print(f"  omitted (fully excluded): {n_omitted}  |  " f"row/col-trimmed: {n_rewritten}")


if __name__ == "__main__":
    main()
