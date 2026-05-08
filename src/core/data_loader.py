"""CSV loading and chunking utilities for data-quality experiments.

A *chunk* is the unit of work given to one LLM call. It always carries:

* a contiguous slice of CSV rows ``[start_row, end_row)``
* a subset of CSV columns (one or more) rendered into the prompt as a
  Markdown table

Two `column_mode` strategies are supported:

* ``"all"``     - one chunk per row-slice, containing **all** CSV columns.
                  Best when error detection requires cross-column context
                  (e.g. attribute-dependency violations, semantic_pattern
                  violations across the row).
* ``"single"``  - one chunk per (row-slice, column) pair, containing a
                  single CSV column at a time. Best for *single-column*
                  error types (typo, inconsistency,
                  semantic_incompatible) where the
                  detector only needs to look at a single column's
                  distribution to decide.

In both modes the same :class:`CsvChunk` dataclass is yielded; downstream
code treats them uniformly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# Column-mode literal kept loose (str) so that yaml-loaded configs work
# without extra coercion. Validated in :func:`iter_chunks`.
COLUMN_MODE_ALL = "all"
COLUMN_MODE_SINGLE = "single"
# Dispatcher-only mode: the chunker never sees ``auto`` (it raises if
# fed to :func:`iter_chunks`); instead, the run-cleaning-detection
# entry point splits selected error types by scope and invokes the
# detector twice - single-column-scope types against ``single`` chunks
# and the rest against ``all`` chunks.
COLUMN_MODE_AUTO = "auto"
VALID_COLUMN_MODES = {COLUMN_MODE_ALL, COLUMN_MODE_SINGLE}
VALID_DISPATCHER_COLUMN_MODES = VALID_COLUMN_MODES | {COLUMN_MODE_AUTO}


@dataclass
class CsvChunk:
    """One unit of work for a single LLM call.

    Attributes
    ----------
    source
        Path to the originating CSV (informational; used in metadata).
    start_row, end_row
        ``[start_row, end_row)`` row slice from the source CSV
        (0-indexed; ``end_row`` is exclusive).
    headers
        Column names included in this chunk (in CSV order).
    frame
        DataFrame restricted to ``headers`` and the row slice, with the
        index reset.
    column_mode
        ``"all"`` or ``"single"``; informational.
    """

    source: Path
    start_row: int
    end_row: int
    headers: List[str]
    frame: pd.DataFrame
    column_mode: str = COLUMN_MODE_ALL

    @property
    def row_count(self) -> int:
        return len(self.frame)

    @property
    def column_count(self) -> int:
        return len(self.headers)

    @property
    def is_single_column(self) -> bool:
        return self.column_mode == COLUMN_MODE_SINGLE

    def to_markdown(self, max_cell_chars: int = 120) -> str:
        """Return a compact Markdown table representation suitable for prompts.

        The ``__row__`` column uses the frame's INDEX as the source of
        row numbers. For unshuffled data the index is ``range(start_row,
        end_row)`` so behaviour is unchanged. For perturbation experiments
        (see :mod:`src.perturbations`) rows may be shuffled before
        chunking - we still want the LLM to see the ORIGINAL CSV row
        numbers so downstream canonicalisation keeps working, which the
        preserved index gives us.
        """
        frame = self.frame.copy()

        def _format(v: object) -> str:
            s = "" if pd.isna(v) else str(v)
            s = _visible_whitespace(s)
            if len(s) > max_cell_chars:
                return s[: max_cell_chars - 1] + "\u2026"
            return s

        frame = frame.astype(object).map(_format)
        frame.insert(0, "__row__", list(self.frame.index))
        return frame.to_markdown(index=False)


# ---------------------------------------------------------------------------
# Whitespace markers for LLM-facing renderings
# ---------------------------------------------------------------------------
# ``tabulate`` (used by ``DataFrame.to_markdown``) right-pads cells with
# regular spaces to align the column - which is visually identical to a
# trailing-space run inside a cell value. A cell of ``"Centerville "``
# and one of ``"Centerville"`` therefore round-trip to the EXACT same
# rendered line, so every downstream detector (typo, inconsistency,
# ...) is structurally blind to leading/trailing whitespace defects.
# Internal newlines, tabs, NBSPs, and zero-width characters are
# similarly invisible in a Markdown table.
#
# Substituting these characters with visible sentinels right before
# rendering preserves their PRESENCE in the LLM's view while keeping
# the markdown structurally correct. The convention is documented in
# the ``base.user_template`` block of ``prompts/error_types.yaml`` so
# every per-type prompt knows what the markers mean.
_WS_NEWLINE = "\u21B5"   # downwards arrow with corner leftwards
_WS_TAB = "\u2192"       # rightwards arrow
_WS_SPACE = "\u2423"     # open box (visible space)
_WS_ZW = "\u00B7"        # middle dot (zero-width characters)

_NEWLINE_CHARS = ("\n", "\r", "\u2028", "\u2029")
_ZW_CHARS = ("\u200B", "\u200C", "\u200D", "\u00AD", "\uFEFF")


def _visible_whitespace(s: str) -> str:
    """Substitute invisible / structurally hidden whitespace with sentinels.

    See module-level note above ``_WS_NEWLINE`` for why we do this.

    Rules:
    * Leading / trailing runs of regular ``" "`` -> one ``␣`` per space.
    * ``\\t`` -> ``→``.
    * Newlines (``\\n``, ``\\r``, U+2028, U+2029) -> ``↵``.
    * NBSP (U+00A0) -> ``␣`` (it functions as a space).
    * Zero-width characters (ZWSP, ZWNJ, ZWJ, soft hyphen, BOM) -> ``·``.

    Internal multi-space runs are LEFT ALONE because tabulate preserves
    them verbatim - they remain visible in the rendered table.
    """
    if not s:
        return s
    # Mark zero-width and NBSP first so they don't interfere with the
    # leading/trailing-space detection (NBSP is not "whitespace" for
    # ``str.lstrip()`` with default argument anyway, but mark it for
    # the LLM regardless).
    for ch in _ZW_CHARS:
        if ch in s:
            s = s.replace(ch, _WS_ZW)
    if "\u00A0" in s:
        s = s.replace("\u00A0", _WS_SPACE)
    if "\t" in s:
        s = s.replace("\t", _WS_TAB)
    for ch in _NEWLINE_CHARS:
        if ch in s:
            s = s.replace(ch, _WS_NEWLINE)
    # Leading / trailing regular-space runs.
    n = len(s)
    lead = 0
    while lead < n and s[lead] == " ":
        lead += 1
    trail = 0
    while trail < n - lead and s[n - 1 - trail] == " ":
        trail += 1
    if lead or trail:
        s = (_WS_SPACE * lead) + s[lead : n - trail] + (_WS_SPACE * trail)
    return s


def _dedupe_columns(columns: Sequence[str]) -> List[str]:
    """Return ``columns`` with duplicates suffixed ``.1``, ``.2``, ...

    Mirrors pandas' own default behaviour when the on-disk CSV has
    repeated header labels. We apply it ourselves AFTER whitespace
    stripping, because two columns that look distinct in the raw file
    (e.g. ``"Numeric"`` and ``" Numeric"``) can collide after
    normalisation and break every downstream helper that assumes
    ``frame[col]`` returns a Series (they would instead get a
    DataFrame).
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in columns:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out


def load_csv(path: Path | str, **read_csv_kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    logger.info("Loading CSV: %s", path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, **read_csv_kwargs)
    stripped = [c.strip() for c in df.columns]
    deduped = _dedupe_columns(stripped)
    if deduped != stripped:
        dup_labels = sorted({s for s in stripped if stripped.count(s) > 1})
        logger.warning(
            "Duplicate column header(s) %s in %s; renaming to disambiguate.",
            dup_labels, path,
        )
    df.columns = deduped
    return df


def iter_chunks(
    df: pd.DataFrame,
    source: Path,
    chunk_size: int,
    max_chunks: Optional[int] = None,
    start_row: int = 0,
    column_mode: str = COLUMN_MODE_ALL,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[CsvChunk]:
    """Yield :class:`CsvChunk` objects covering the dataframe.

    Parameters
    ----------
    df
        Source dataframe (already loaded via :func:`load_csv`).
    source
        Path to the CSV (carried into each chunk for metadata).
    chunk_size
        Number of rows per chunk.
    max_chunks
        If set, stops after emitting that many chunks **total** (counting
        every (row-slice, column) pair in single-column mode).
    start_row
        First CSV row index to include.
    column_mode
        Either ``"all"`` (default) or ``"single"``; see module docstring.
    columns
        Optional explicit column subset (CSV column names). Defaults to
        every column in ``df`` (in DataFrame order). Useful for limiting
        single-column mode to a handful of columns of interest.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if column_mode not in VALID_COLUMN_MODES:
        raise ValueError(
            f"column_mode must be one of {sorted(VALID_COLUMN_MODES)}, "
            f"got {column_mode!r}"
        )

    if columns is None:
        active_columns: List[str] = list(df.columns)
    else:
        active_columns = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"Requested columns not present in CSV: {missing}. "
                f"Available: {list(df.columns)}"
            )
    if not active_columns:
        raise ValueError("No active columns to chunk over")

    if column_mode == COLUMN_MODE_SINGLE:
        column_groups: List[List[str]] = [[c] for c in active_columns]
    else:
        column_groups = [active_columns]

    emitted = 0
    for begin in range(start_row, len(df), chunk_size):
        if max_chunks is not None and emitted >= max_chunks:
            return
        end = min(begin + chunk_size, len(df))
        row_frame = df.iloc[begin:end]
        for group in column_groups:
            if max_chunks is not None and emitted >= max_chunks:
                return
            # Keep the frame's index so downstream code can recover the
            # ORIGINAL CSV row number (which is what the LLM's ``row``
            # output references). Resetting here would break perturbation
            # experiments that shuffle rows before chunking.
            sub = row_frame[group].copy()
            yield CsvChunk(
                source=source,
                start_row=begin,
                end_row=end,
                headers=list(group),
                frame=sub,
                column_mode=column_mode,
            )
            emitted += 1
