"""Reconstruct per-table CSVs from the Bing21-CtrlT column-level TSV.

The upstream file (``Bing21-CtrlT-processed.tsv``) stores **one column per
TSV row**. Six tab-separated fields, verified against the full 295k-row
file shipped by the team:

    0. spreadsheet-file-name-id  e.g. ``corpus\\Bing21-subset1-xlsx\\<sha1>.xlsx``
    1. table-name-id             e.g. ``tid0_0.json_line1_<sheet>_<A1:range>``
    2. (empty reserved field)
    3. table-col-number          0-indexed
    4. table-col-header
    5. table-vals-concatenated-using-"___"

This module groups rows by ``(spreadsheet, table)``, sorts each group by
``col-number``, splits the concatenated values on the literal ``___``
separator, and emits one CSV per table plus a ``manifest.jsonl`` that
preserves the original ``spreadsheet-file-name-id`` so multi-table
analyses can still identify which tables came from the same source
workbook.

Usage
-----

::

    python -m src.bing21_reconstruct \\
        --input /datadrive/yeye/data/Bing21-CtrlT/Bing21-CtrlT-processed.tsv \\
        --output-dir data/Bing21 \\
        [--limit-spreadsheets N]

Layout
------

One subdirectory per source ``.xlsx`` (so downstream ``--csv-dir`` sees
each spreadsheet as a self-contained multi-table group)::

    <output-dir>/
      <xlsx-stem>/                    # e.g. 05c19c7293d95c...
        <safe-table-id>.csv
        ...
      manifest.jsonl                  # one record per emitted CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The TSV uses CRLF line endings and a literal "___" separator between
# consecutive cell values within a single column.
VALUE_SEP = "___"
EXPECTED_FIELDS = 6
# Keep filenames portable (Windows-safe). Table-ids contain ':' and
# occasionally other shell-unfriendly characters.
_UNSAFE = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def _safe_filename(name: str, max_len: int = 120) -> str:
    cleaned = _UNSAFE.sub("_", name).strip(" ._")
    return cleaned[:max_len] or "table"


def _xlsx_stem(spreadsheet_id: str) -> str:
    """Extract the bare hash stem from a path like
    ``corpus\\Bing21-subset1-xlsx\\<hash>.xlsx``.
    """
    tail = spreadsheet_id.replace("\\", "/").rsplit("/", 1)[-1]
    if tail.lower().endswith(".xlsx"):
        tail = tail[: -len(".xlsx")]
    return _safe_filename(tail)


def _iter_rows(tsv_path: Path) -> Iterator[Tuple[str, str, int, str, str]]:
    """Yield ``(spreadsheet, table, col_num, header, raw_values)`` tuples.

    Skips malformed lines (wrong field count or non-integer col-number)
    with a warning so a single bad row doesn't abort a 295k-row run.
    """
    skipped = 0
    # newline='' so embedded \r\n inside a TSV row doesn't confuse the
    # Python iterator; the file's line terminator is CRLF.
    with tsv_path.open("r", encoding="utf-8", newline="") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != EXPECTED_FIELDS:
                skipped += 1
                if skipped <= 5:
                    logger.warning(
                        "line %d: expected %d fields, got %d - skipping",
                        lineno, EXPECTED_FIELDS, len(parts),
                    )
                continue
            spreadsheet, table, _unused, col_num_str, header, values = parts
            try:
                col_num = int(col_num_str)
            except ValueError:
                skipped += 1
                continue
            yield spreadsheet, table, col_num, header, values
    if skipped:
        logger.warning("skipped %d malformed line(s)", skipped)


def _assemble_table(
    columns: Dict[int, Tuple[str, List[str]]],
) -> Tuple[List[str], List[List[str]]]:
    """Turn a ``{col_num: (header, values)}`` dict into (headers, rows).

    Rows are padded to the longest column with empty strings. Headers
    are emitted in ascending ``col_num`` order.
    """
    ordered = sorted(columns.items(), key=lambda kv: kv[0])
    headers = [header for _, (header, _) in ordered]
    value_lists = [vals for _, (_, vals) in ordered]
    n_rows = max((len(v) for v in value_lists), default=0)
    rows: List[List[str]] = []
    for r in range(n_rows):
        rows.append([v[r] if r < len(v) else "" for v in value_lists])
    return headers, rows


def reconstruct(
    tsv_path: Path,
    output_dir: Path,
    *,
    limit_spreadsheets: Optional[int] = None,
) -> Dict[str, int]:
    """Stream ``tsv_path`` into per-table CSVs under ``output_dir``.

    Returns a small stats dict: ``spreadsheets``, ``tables``, ``columns``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    # (spreadsheet, table) -> {col_num: (header, [values])}. The TSV is
    # ~99.8% sorted by (spreadsheet, table), but ~500 pairs appear out
    # of order, so we hold the full dict in memory (~1-2 GB peak for the
    # full 726 MB file). This is the simplest correct approach.
    buckets: Dict[Tuple[str, str], Dict[int, Tuple[str, List[str]]]] = (
        defaultdict(dict)
    )
    for spreadsheet, table, col_num, header, values in _iter_rows(tsv_path):
        buckets[(spreadsheet, table)][col_num] = (
            header,
            values.split(VALUE_SEP) if values else [],
        )

    # Group by spreadsheet so we can honour --limit-spreadsheets and
    # emit each workbook's tables into its own subdirectory.
    by_spreadsheet: Dict[str, List[str]] = defaultdict(list)
    for spreadsheet, table in buckets.keys():
        by_spreadsheet[spreadsheet].append(table)

    spreadsheets = sorted(by_spreadsheet.keys())
    if limit_spreadsheets is not None:
        spreadsheets = spreadsheets[:limit_spreadsheets]

    stats = {"spreadsheets": 0, "tables": 0, "columns": 0}
    with manifest_path.open("w", encoding="utf-8") as mf:
        for spreadsheet in spreadsheets:
            stem = _xlsx_stem(spreadsheet)
            sub = output_dir / stem
            sub.mkdir(parents=True, exist_ok=True)
            tables = sorted(by_spreadsheet[spreadsheet])
            stats["spreadsheets"] += 1
            for table in tables:
                columns = buckets[(spreadsheet, table)]
                headers, rows = _assemble_table(columns)
                if not headers:
                    continue
                out_path = sub / f"{_safe_filename(table)}.csv"
                with out_path.open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(headers)
                    writer.writerows(rows)
                stats["tables"] += 1
                stats["columns"] += len(headers)
                mf.write(json.dumps({
                    "spreadsheet_file_name_id": spreadsheet,
                    "table_name_id": table,
                    "output_path": str(out_path.relative_to(output_dir)),
                    "n_columns": len(headers),
                    "n_rows": len(rows),
                }, ensure_ascii=False) + "\n")
    return stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct per-table CSVs from Bing21-CtrlT-processed.tsv"
    )
    p.add_argument(
        "--input", type=Path, required=True,
        help="Path to Bing21-CtrlT-processed.tsv",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write <xlsx-stem>/<table>.csv files into.",
    )
    p.add_argument(
        "--limit-spreadsheets", type=int, default=None,
        help="Only emit the first N spreadsheets (alphabetical). "
             "Useful for smoke-testing.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    if not args.input.is_file():
        logger.error("input TSV not found: %s", args.input)
        return 2
    stats = reconstruct(
        args.input, args.output_dir,
        limit_spreadsheets=args.limit_spreadsheets,
    )
    logger.info(
        "done: %d spreadsheets, %d tables, %d columns -> %s",
        stats["spreadsheets"], stats["tables"], stats["columns"],
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
