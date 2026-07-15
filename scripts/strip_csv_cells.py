"""One-shot utility: strip leading/trailing whitespace from every cell of
every CSV under a given directory.

Mirrors the in-memory normalisation done by
:func:`src.core.data_loader.load_csv`, but persists the change on disk so the
saved benchmark CSVs match what every downstream loader sees.

Usage::

    python -m scripts.strip_csv_cells data/curated_bing21_200            # dry-run
    python -m scripts.strip_csv_cells data/curated_bing21_200 --apply    # rewrite
    python -m scripts.strip_csv_cells data/mt100_x --collapse-internal --apply

Behaviour:
* Cells: leading/trailing whitespace stripped. INTERNAL whitespace
  (repeated spaces, embedded newlines, tabs) is preserved verbatim -- this
  keeps ``typo:whitespace`` findings (a double space, an embedded newline)
  intact, matching ``load_csv``'s runtime policy.
* ``--collapse-internal`` (opt-in): additionally collapse every run of
  whitespace *inside* a cell to a single space (``\\s+`` -> ``" "``), then
  strip. Use this to positively CLEAN internal-whitespace defects on disk.
  NOTE: this NEUTRALISES whitespace-typo findings (the defect is gone), so it
  is deliberately off by default and does NOT mirror ``load_csv``. Applied to
  cell VALUES only -- headers keep the strip-only policy so column identity
  (and cluster/column matching in the labelling UI) is never altered.
* Headers: leading/trailing whitespace stripped. If stripping causes
  two headers to collide (e.g. ``"Total"`` and ``"Total "`` exist
  side-by-side), the FIRST occurrence is kept and later duplicate
  columns are dropped from the file. This matches the runtime policy
  in :func:`src.core.data_loader.load_csv`.
* Empty / whitespace-only cells collapse to ``""``.
* Files with no changes are left untouched.
* CSVs are read with the same ``dtype=str, keep_default_na=False`` flags
  used by ``load_csv``, then written back atomically with ``index=False,
  lineterminator='\\n'``.

The script is idempotent: a second run is a no-op.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd


logger = logging.getLogger("strip_csv_cells")

# Any run of whitespace (spaces, tabs, newlines, unicode separators) used by
# ``--collapse-internal`` to fold internal whitespace to a single space.
_WS_RUN = re.compile(r"\s+")


def _clean_cell(s: str, collapse_internal: bool) -> str:
    """Normalise one cell value.

    Default: strip leading/trailing whitespace only (internal preserved).
    ``collapse_internal``: also fold every internal whitespace run to a single
    space (then strip) -- i.e. positively clean internal-whitespace defects.
    """
    if collapse_internal:
        return _WS_RUN.sub(" ", s).strip()
    return s.strip()


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically.

    Writes to a temp file in the same directory, then ``os.replace``s it into
    place (an atomic rename on POSIX). A concurrent reader (e.g. a detection
    run loading the CSV) therefore always sees either the complete old file or
    the complete new file - never a truncated/partial one. The replace also
    bumps the inode/mtime, which is exactly what the labeling UI's
    ``load_frame`` signature check watches to reload an edited source.
    """
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False, lineterminator="\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _scan_file(
    path: Path,
    collapse_internal: bool = False,
) -> Tuple[int, int, int, List[str], pd.DataFrame]:
    """Return ``(headers_changed, cells_changed, cols_dropped, dup_headers, new_df)``.

    ``new_df`` is the cleaned frame ready to write. If stripping headers
    causes collisions, the FIRST occurrence of each name is kept and the
    later duplicate columns are dropped from the frame. ``dup_headers``
    reports which names collided.

    ``collapse_internal`` only affects CELL values (see ``_clean_cell``);
    headers are always strip-only so column identity is never altered.
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

    stripped_df = df.map(
        lambda s: _clean_cell(s, collapse_internal) if isinstance(s, str) else s
    )
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
    parser.add_argument(
        "--collapse-internal",
        action="store_true",
        help="Also collapse internal whitespace runs inside each cell to a "
             "single space (then strip). CLEANS internal-whitespace defects "
             "on disk but NEUTRALISES whitespace-typo findings. Off by "
             "default; applies to cell values only, never headers.",
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
    logger.info(
        "Found %d CSV file(s) under %s%s",
        len(csv_paths),
        root,
        "  [--collapse-internal: folding internal whitespace to single spaces]"
        if args.collapse_internal else "",
    )

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
            ) = _scan_file(path, collapse_internal=args.collapse_internal)
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
            _atomic_write_csv(new_df, path)

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
