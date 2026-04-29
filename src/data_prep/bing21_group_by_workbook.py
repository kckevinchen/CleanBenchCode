"""Group flat Bing21 CSVs into per-workbook subdirectories.

The deduped Bing21 corpus
(``data/s2.deduped_processed_csv_bing21_463G_non_final``) lives as a
single flat directory of ~243k files named

    <wb_sha40>.xlsx_<sheet_sha8>.csv

Sheets that came from the same source workbook share the leading
``<wb_sha40>.xlsx`` prefix. About 35% of workbooks contributed >=2
sheets, so grouping is the difference between treating every CSV as
an island and being able to study multi-sheet workbooks.

This script walks the flat dir, parses the prefix, and emits

    <output-dir>/<wb_sha40>.xlsx/<sheet_sha8>.csv

The default mode is **symlink** so we don't duplicate ~28-463 GB of
data; pass ``--mode copy`` or ``--mode hardlink`` if you need a
self-contained tree. CSVs that don't match the convention are
collected under ``<output-dir>/_unmatched/`` (and reported on stdout)
so nothing is silently dropped.

Usage
-----

::

    python -m src.data_prep.bing21_group_by_workbook \\
        --source-dir data/s2.deduped_processed_csv_bing21_463G_non_final \\
        --output-dir data/bing21_grouped \\
        --mode symlink

After this runs, ``data/bing21_grouped/<wb_sha40>.xlsx/`` is a
ready-to-curate multi-table corpus that ``src.data_prep.curate`` (or any
``--csv-dir`` consumer) can walk like ``curated_pbi_full``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger("bing21_group_by_workbook")

# Filename convention. The wb prefix is variable-length in practice
# (sha1=40 hex chars in this corpus, but be generous: 8-64).
_FNAME_RE = re.compile(
    r"^(?P<wb>[0-9a-f]{8,64}\.xlsx)_(?P<sheet>[0-9a-f]+)\.csv$",
    re.IGNORECASE,
)

LinkMode = str  # "symlink" | "hardlink" | "copy"


def _link_one(src: Path, dst: Path, mode: LinkMode, *, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "symlink":
        # Use absolute target so the link works regardless of where
        # callers cd to.
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:  # defensive - argparse already restricts choices
        raise ValueError(f"unknown mode: {mode!r}")


def group_directory(
    *,
    source_dir: Path,
    output_dir: Path,
    mode: LinkMode = "symlink",
    overwrite: bool = False,
    progress_every: int = 10_000,
) -> Dict[str, int]:
    if not source_dir.is_dir():
        raise SystemExit(f"source-dir not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    unmatched_dir = output_dir / "_unmatched"

    counts = Counter()
    per_wb: Counter = Counter()
    manifest_path = output_dir / "grouping_manifest.jsonl"
    # Stream the manifest out so we don't hold 250k records in RAM.
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, entry in enumerate(_iter_csvs(source_dir), start=1):
            m = _FNAME_RE.match(entry.name)
            if m is None:
                counts["unmatched"] += 1
                unmatched_dir.mkdir(exist_ok=True)
                _link_one(entry, unmatched_dir / entry.name, mode,
                          overwrite=overwrite)
                continue
            wb = m.group("wb")
            sheet = m.group("sheet")
            wb_dir = output_dir / wb
            wb_dir.mkdir(exist_ok=True)
            dst = wb_dir / f"{sheet}.csv"
            _link_one(entry, dst, mode, overwrite=overwrite)
            counts["linked"] += 1
            per_wb[wb] += 1
            mf.write(json.dumps({
                "source": str(entry),
                "workbook": wb,
                "sheet": sheet,
                "dst": str(dst),
            }, ensure_ascii=False) + "\n")
            if i % progress_every == 0:
                logger.info("processed %d files (linked=%d, unmatched=%d, "
                            "workbooks=%d)",
                            i, counts["linked"], counts["unmatched"],
                            len(per_wb))

    # Distribution stats - useful for picking curation parameters.
    sheet_counts = Counter(per_wb.values())
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "files_total": counts["linked"] + counts["unmatched"],
        "files_linked": counts["linked"],
        "files_unmatched": counts["unmatched"],
        "workbooks": len(per_wb),
        "workbooks_with_1_sheet": sheet_counts.get(1, 0),
        "workbooks_with_ge_2_sheets": sum(
            n for k, n in sheet_counts.items() if k >= 2
        ),
        "workbooks_with_ge_5_sheets": sum(
            n for k, n in sheet_counts.items() if k >= 5
        ),
        "max_sheets_in_workbook": max(per_wb.values()) if per_wb else 0,
    }
    (output_dir / "grouping_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("done: %s", json.dumps(summary, indent=2))
    return summary


def _iter_csvs(source_dir: Path) -> Iterator[Path]:
    """Yield every ``*.csv`` directly under ``source_dir`` without
    materialising the full list (the dir has 200k+ entries)."""
    with os.scandir(source_dir) as it:
        for de in it:
            if de.is_file() and de.name.lower().endswith(".csv"):
                yield Path(de.path)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Group a flat Bing21 CSV directory into per-workbook "
            "subdirectories using the '<wb>.xlsx_<sheet>.csv' filename "
            "convention. Defaults to symlinks so the original 463 GB "
            "corpus isn't duplicated."
        )
    )
    p.add_argument("--source-dir", type=Path, required=True,
                   help="flat directory of *.csv files")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="where to write the grouped tree")
    p.add_argument("--mode", choices=("symlink", "hardlink", "copy"),
                   default="symlink",
                   help="how to materialise grouped files (default: symlink)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace existing entries in the output tree")
    p.add_argument("--progress-every", type=int, default=10_000)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    group_directory(
        source_dir=args.source_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        mode=args.mode,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
