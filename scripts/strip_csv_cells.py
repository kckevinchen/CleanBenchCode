"""One-shot utility: strip leading/trailing whitespace from every cell of
every CSV under a given directory.

Mirrors the in-memory normalisation done by
:func:`src.core.data_loader.load_csv`, but persists the change on disk so the
saved benchmark CSVs match what every downstream loader sees.

Usage::

    python -m scripts.strip_csv_cells data/curated_bing21_200            # dry-run
    python -m scripts.strip_csv_cells data/curated_bing21_200 --apply    # rewrite

Behaviour:
* Cells: leading/trailing whitespace stripped. INTERNAL whitespace
  (repeated spaces, embedded newlines, tabs) is preserved verbatim.
* Headers: leading/trailing whitespace stripped. If stripping causes
  two headers to collide (e.g. ``"Total"`` and ``"Total "`` exist
  side-by-side), the FIRST occurrence is kept and later duplicate
  columns are dropped from the file. This matches the runtime policy
  in :func:`src.core.data_loader.load_csv`.
* Empty / whitespace-only cells collapse to ``""``.
* Files with no changes are left untouched.
* CSVs are read with the same ``dtype=str, keep_default_na=False`` flags
  used by ``load_csv``, then written back with ``index=False,
  lineterminator='\\n'``.

The script is idempotent: a second run is a no-op.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd


logger = logging.getLogger("strip_csv_cells")


def _scan_file(
    path: Path,
) -> Tuple[int, int, int, List[str], pd.DataFrame]:
    """Return ``(headers_changed, cells_changed, cols_dropped, dup_headers, new_df)``.

    ``new_df`` is the cleaned frame ready to write. If stripping headers
    causes collisions, the FIRST occurrence of each name is kept and the
    later duplicate columns are dropped from the frame. ``dup_headers``
    reports which names collided.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    raw_cols = list(df.columns)
    stripped_cols = [c.strip() if isinstance(c, str) else c for c in raw_cols]

    seen: set[str] = set()
    keep_indices: List[int] = []
    final_cols: List[str] = []
    dup_set: set[str] = set()
    for i, name in enumerate(stripped_cols):
        if name in seen:
            dup_set.add(name)
            continue
        seen.add(name)
        keep_indices.append(i)
        final_cols.append(name)

    dup_headers = sorted(dup_set)
    cols_dropped = len(raw_cols) - len(keep_indices)

    # Headers changed iff: any kept column was renamed, OR any column dropped.
    headers_changed = cols_dropped + sum(
        1 for i, name in zip(keep_indices, final_cols) if raw_cols[i] != name
    )

    if cols_dropped:
        df = df.iloc[:, keep_indices].copy()
    df.columns = final_cols

    stripped_df = df.map(lambda s: s.strip() if isinstance(s, str) else s)
    cells_changed = int(df.ne(stripped_df).to_numpy().sum())
    return headers_changed, cells_changed, cols_dropped, dup_headers, stripped_df


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory to walk for *.csv files",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite files. Without this flag the script reports "
             "what WOULD change.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    root: Path = args.root
    if not root.is_dir():
        logger.error("Not a directory: %s", root)
        return 2

    csv_paths = sorted(root.rglob("*.csv"))
    logger.info("Found %d CSV file(s) under %s", len(csv_paths), root)

    total_cells = 0
    total_headers = 0
    total_dropped = 0
    files_changed = 0
    files_skipped: List[Path] = []
    sample_changes: List[Tuple[Path, int, int, int]] = []

    for path in csv_paths:
        try:
            (
                headers_changed,
                cells_changed,
                cols_dropped,
                dup_headers,
                new_df,
            ) = _scan_file(path)
        except Exception as exc:  # noqa: BLE001 - we want to log + continue
            logger.error("Failed to read %s: %s", path, exc)
            files_skipped.append(path)
            continue

        if dup_headers:
            logger.warning(
                "%s: dropping %d duplicate column(s) after header-strip "
                "(collided names: %s; first occurrence kept)",
                path, cols_dropped, dup_headers,
            )

        if cells_changed == 0 and headers_changed == 0:
            continue

        total_cells += cells_changed
        total_headers += headers_changed
        total_dropped += cols_dropped
        files_changed += 1
        sample_changes.append((path, headers_changed, cells_changed, cols_dropped))

        if args.apply:
            new_df.to_csv(path, index=False, lineterminator="\n")

    action = "Wrote" if args.apply else "Would change"
    logger.info("=" * 60)
    logger.info(
        "%s %d file(s): %d header(s) stripped, %d cell(s) stripped, "
        "%d duplicate column(s) dropped",
        action, files_changed, total_headers, total_cells, total_dropped,
    )
    if files_skipped:
        logger.info("Skipped %d file(s); see warnings above.", len(files_skipped))
    for path, h, c, d in sample_changes[:10]:
        logger.info(
            "  %s  headers=%d cells=%d dropped_cols=%d",
            path.relative_to(root.parent if root.parent else root),
            h, c, d,
        )
    if len(sample_changes) > 10:
        logger.info("  ... (%d more)", len(sample_changes) - 10)
    if not args.apply:
        logger.info("Dry-run; re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
