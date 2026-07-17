"""Collapse embedded newlines inside DATA cells of the curated pbi_500 /
bing21 CSV datasets, and (optionally) rewrite the matching value snapshots
in the labeled runs that reference those datasets so labels stay consistent.

Why this exists
===============
Some curated cells contain hard line breaks inside a single quoted field
(e.g. ``"27/06/2017\\n-Matin"``). They are noise for the detectors yet are
preserved verbatim by :func:`src.core.data_loader.load_csv` (which only
strips *leading/trailing* whitespace). This one-shot utility removes the
internal line breaks on disk.

Newline policy (default)
========================
Any maximal run of whitespace that contains a line break
(``\\n``, ``\\r``, U+2028, U+2029) collapses to a SINGLE space; the cell is
then stripped. Cells WITHOUT a line break are left byte-for-byte untouched.

Full-whitespace policy (``--collapse-spaces``)
==============================================
Collapses EVERY kind of excessive whitespace -- internal multi-space runs,
tabs, NBSP, zero-width characters, and line breaks -- to a single space and
strips the edges (a superset of the newline policy). The result equals the
loader's in-memory view (``src.core.data_loader.load_csv`` +
``_normalize_trivial``), so on-disk cells stop drifting from what the
detector / labeling UI render. Still BODY cells only; headers untouched.

Carefully preserved
===================
* The HEADER row (row 0) is never modified, so the column names that run
  labels reference stay identical. (Hundreds of labels store a ``column``
  field whose name literally contains a newline; renaming headers would
  orphan them.)
* Only cells that actually contain a line break change; everything else is
  byte-identical.
* After an ``--apply`` write the file is re-parsed and verified: the header
  must be unchanged and every body cell must equal its expected cleaned
  value, otherwise the script aborts.

Run-snapshot rewrite (``--rewrite-runs``)
=========================================
The same collapse is applied to value-bearing snapshot fields in the runs
that reference these datasets, so a cleaned cell still matches the label /
issue / benchmark record that points at it. Only an allow-list of
cell-value fields is touched (never prose like ``explanation`` / ``note`` /
``raw_text``), and only strings that actually contain a line break change.

Dry-run by default. Pass ``--apply`` to write.

Usage::

    python -m scripts.collapse_cell_newlines                      # dry-run, CSVs only
    python -m scripts.collapse_cell_newlines --apply              # write CSVs
    python -m scripts.collapse_cell_newlines --rewrite-runs       # dry-run incl. runs
    python -m scripts.collapse_cell_newlines --apply --rewrite-runs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("collapse_cell_newlines")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = [
    PROJECT_ROOT / "data" / "curated_pbi_500_dedup",
    PROJECT_ROOT / "data" / "curated_bing21_500_dedup",
]

# Run artifact files we rewrite (jsonl = one object per line, json = single doc).
RUN_ARTIFACTS = [
    "issues.jsonl",
    "issues.json",
    "clusters.jsonl",
    "clusters.json",
    "chunk_reports.jsonl",
    "chunk_reports.json",
    "labels.jsonl",
    "benchmark.jsonl",
    "benchmark.skipped.jsonl",
    "benchmark.low_value.jsonl",
    "row_verdicts.jsonl",
]

# JSON keys whose value is a verbatim copy of a data cell (a scalar string or
# a {column: value} map). These are the ONLY keys we rewrite. Prose / raw /
# identifier keys (explanation, note, rationale, raw_text, reviewer_note,
# root_cause, prompt, parse_error, source, column, ...) are deliberately
# excluded so we never touch human- or model-authored narrative text.
CELL_VALUE_KEYS = frozenset(
    {
        # scalar cell mirrors
        "observed",
        "suggested_fix",
        "original_value",
        "llm_suggested_value",
        "canonical_fix",
        # polymorphic: scalar in benchmark/labels, {col: val} map elsewhere
        "corrected_value",
        # {column: value} maps
        "row_values",
        "full_row_values",
        "corrected_values",
        "value_corrections",
        "canonical_record",
    }
)

_NL_CHARS = "\r\n\u2028\u2029"
# A run of optional spaces/tabs, at least one line break, then any spaces /
# tabs / line breaks. Greedy, so consecutive blank lines collapse to one space.
_NL_RUN = re.compile(r"[ \t]*[" + _NL_CHARS + r"][ \t" + _NL_CHARS + r"]*")

# Full-whitespace normalisation primitives. These MIRROR
# ``src/labeling/_helpers.py::_normalize_trivial`` -- the canonical
# "trivial whitespace fix" the typo tagger and benchmark exporter use --
# so a cell cleaned here is byte-identical to what the pipeline would
# consider the de-whitespaced form. Replicated (not imported) to keep this
# data-prep script free of any labeling-layer dependency.
_ZW_DROP = re.compile(r"[\u200B\u200C\u200D\u00AD\uFEFF]")
_CONTROL_TO_SPACE = re.compile(r"[\t\n\r\u2028\u2029]")
_NBSP_TO_SPACE = re.compile(r"[\u00A0]")
_MULTI_SPACE = re.compile(r" {2,}")


def collapse_newlines(s: str) -> str:
    """Collapse whitespace runs containing a line break to one space; strip.

    Strings without a line break are returned unchanged (identity), so the
    transform is a no-op for the overwhelming majority of cells and is
    idempotent.
    """
    if not isinstance(s, str):
        return s
    if not any(ch in s for ch in _NL_CHARS):
        return s
    return _NL_RUN.sub(" ", s).strip()


def normalize_whitespace(s: str) -> str:
    """Collapse EVERY kind of excessive whitespace to a single space; strip.

    Superset of :func:`collapse_newlines`: drops zero-width characters,
    folds tab / NBSP / line breaks to a space, collapses internal
    multi-space runs to one, and strips the edges. Matches
    ``_normalize_trivial`` so the on-disk value equals the loader's view.
    Idempotent; a string with no excessive whitespace is returned
    unchanged.
    """
    if not isinstance(s, str):
        return s
    out = _ZW_DROP.sub("", s)
    out = _CONTROL_TO_SPACE.sub(" ", out)
    out = _NBSP_TO_SPACE.sub(" ", out)
    out = _MULTI_SPACE.sub(" ", out)
    out = out.strip()
    return out


# --------------------------------------------------------------------------- CSV

def clean_csv_file(
    path: Path, apply: bool, transform: Callable[[str], str]
) -> Tuple[int, List[Tuple[int, int, str, str]]]:
    """Clean body-cell whitespace in one CSV. Returns ``(n_changed, samples)``.

    ``transform`` is applied to every body cell (header row untouched).
    ``samples`` holds up to 3 ``(row, col, old, new)`` tuples for reporting.
    When ``apply`` is set and at least one cell changes, the file is
    rewritten and re-verified.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return 0, []

    n_changed = 0
    samples: List[Tuple[int, int, str, str]] = []
    # Row 0 is the header - leave it untouched.
    for r in range(1, len(rows)):
        row = rows[r]
        for c, cell in enumerate(row):
            new = transform(cell)
            if new != cell:
                row[c] = new
                n_changed += 1
                if len(samples) < 3:
                    samples.append((r, c, cell, new))

    if n_changed and apply:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerows(rows)
        _verify_csv(path, tmp, rows)
        os.replace(tmp, path)

    return n_changed, samples


def _verify_csv(original: Path, written: Path, expected_rows: List[List[str]]) -> None:
    """Re-parse the freshly written CSV and assert it matches ``expected_rows``.

    Guards against any quoting / dialect surprise: the parsed values must be
    exactly what we intended to write (header included, byte-for-byte per
    field). Raises ``RuntimeError`` on any mismatch so ``--apply`` aborts
    before the temp file replaces the original.
    """
    with written.open("r", encoding="utf-8", newline="") as fh:
        got = list(csv.reader(fh))
    if len(got) != len(expected_rows):
        raise RuntimeError(
            f"{original}: row count changed on write "
            f"({len(expected_rows)} -> {len(got)}); aborting"
        )
    for i, (a, b) in enumerate(zip(expected_rows, got)):
        if a != b:
            raise RuntimeError(
                f"{original}: row {i} differs after re-parse; aborting "
                f"(expected {a!r} got {b!r})"
            )


# -------------------------------------------------------------------------- JSON

def transform_json(obj: Any, transform: Callable[[str], str]) -> int:
    """Apply ``transform`` to allow-listed cell-value fields, in place.

    Returns the number of string values changed. Recurses through every
    dict / list, but only rewrites strings sitting directly under a key in
    :data:`CELL_VALUE_KEYS` (scalar) or the string values of a map under
    such a key. All other strings (prose, ids) are left untouched.
    """
    changed = 0
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in CELL_VALUE_KEYS and isinstance(val, str):
                new = transform(val)
                if new != val:
                    obj[key] = new
                    changed += 1
            elif key in CELL_VALUE_KEYS and isinstance(val, dict):
                for ck, cv in val.items():
                    if isinstance(cv, str):
                        new = transform(cv)
                        if new != cv:
                            val[ck] = new
                            changed += 1
            else:
                changed += transform_json(val, transform)
    elif isinstance(obj, list):
        for item in obj:
            changed += transform_json(item, transform)
    return changed


def clean_jsonl_file(path: Path, apply: bool, transform: Callable[[str], str]) -> int:
    """Rewrite a ``.jsonl`` artifact in place. Returns string-values changed."""
    out_lines: List[str] = []
    total = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                out_lines.append(line.rstrip("\n"))
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                out_lines.append(line.rstrip("\n"))
                continue
            n = transform_json(rec, transform)
            total += n
            out_lines.append(json.dumps(rec, ensure_ascii=False))
    if total and apply:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return total


def clean_json_file(path: Path, apply: bool, transform: Callable[[str], str]) -> int:
    """Rewrite a single-document ``.json`` artifact. Returns values changed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    total = transform_json(data, transform)
    if total and apply:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, path)
    return total


# --------------------------------------------------------------------------- run discovery

def discover_affected_runs(
    outputs_dir: Path, dataset_dirs: List[Path]
) -> List[Path]:
    """Return run dirs whose ``summary.json`` references one of the datasets."""
    names = {d.name for d in dataset_dirs}
    hits: List[Path] = []
    for summary in sorted(outputs_dir.glob("*/summary.json")):
        try:
            cfg = json.loads(summary.read_text(encoding="utf-8")).get("config", {})
        except Exception:  # noqa: BLE001
            continue
        csv_path = str(cfg.get("csv_path") or "").rstrip("/")
        csv_paths = cfg.get("csv_paths") or []
        ref = os.path.basename(csv_path) in names or any(
            os.path.basename(d) in names
            for p in csv_paths
            for d in [os.path.dirname(str(p).rstrip("/"))]
        )
        if ref:
            hits.append(summary.parent)
    return hits


# --------------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets", nargs="*", type=Path, default=None,
        help="Dataset dirs to clean (default: curated_pbi_500_dedup + "
             "curated_bing21_500_dedup).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes. Without it the script only reports.",
    )
    parser.add_argument(
        "--rewrite-runs", action="store_true",
        help="Also rewrite value snapshots in runs referencing these datasets.",
    )
    parser.add_argument(
        "--collapse-spaces", action="store_true",
        help="Collapse EVERY kind of excessive whitespace (internal "
             "multi-space runs, tabs, NBSP, zero-width, line breaks) to a "
             "single space and strip edges, matching the loader's view. "
             "Without this flag only newline-containing runs are collapsed.",
    )
    parser.add_argument(
        "--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs",
        help="Where run dirs live (default: <root>/outputs).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    datasets = args.datasets or DEFAULT_DATASETS

    # Pick the cell transform: full whitespace normalisation (superset) or
    # the original newline-only collapse.
    transform = normalize_whitespace if args.collapse_spaces else collapse_newlines
    mode_label = (
        "collapse ALL excessive whitespace"
        if args.collapse_spaces
        else "collapse newlines"
    )

    # ----- Phase 1: CSV data cells -------------------------------------------
    logger.info("=" * 68)
    logger.info("PHASE 1: %s in DATA cells (headers preserved)", mode_label)
    logger.info("=" * 68)
    grand_cells = 0
    grand_files = 0
    for dataset in datasets:
        if not dataset.is_dir():
            logger.error("Not a directory: %s", dataset)
            return 2
        csv_paths = sorted(dataset.rglob("*.csv"))
        ds_cells = 0
        ds_files = 0
        first_samples: List[str] = []
        for path in csv_paths:
            try:
                n, samples = clean_csv_file(path, args.apply, transform)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed on %s: %s", path, exc)
                return 3
            if n:
                ds_cells += n
                ds_files += 1
                if len(first_samples) < 5:
                    rel = path.relative_to(dataset.parent)
                    r, c, old, new = samples[0]
                    first_samples.append(
                        f"      {rel}  (row {r}, col {c}): {old!r} -> {new!r}"
                    )
        logger.info(
            "  %-32s %4d cell(s) in %3d/%d file(s)",
            dataset.name, ds_cells, ds_files, len(csv_paths),
        )
        for s in first_samples:
            logger.info(s)
        grand_cells += ds_cells
        grand_files += ds_files
    logger.info(
        "  TOTAL: %d cell(s) across %d file(s)", grand_cells, grand_files
    )

    # ----- Phase 2: run snapshots --------------------------------------------
    if args.rewrite_runs:
        logger.info("=" * 68)
        logger.info("PHASE 2: rewrite value snapshots in referencing runs")
        logger.info("=" * 68)
        runs = discover_affected_runs(args.outputs_dir, datasets)
        logger.info("  %d run(s) reference these datasets", len(runs))
        grand_snap = 0
        grand_run_files = 0
        for run in runs:
            run_changed = 0
            file_hits: List[str] = []
            for name in RUN_ARTIFACTS:
                fp = run / name
                if not fp.is_file():
                    continue
                try:
                    if name.endswith(".jsonl"):
                        n = clean_jsonl_file(fp, args.apply, transform)
                    else:
                        n = clean_json_file(fp, args.apply, transform)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed on %s: %s", fp, exc)
                    return 4
                if n:
                    run_changed += n
                    grand_run_files += 1
                    file_hits.append(f"{name}={n}")
            if run_changed:
                grand_snap += run_changed
                logger.info("  %-46s %5d value(s)  [%s]",
                            run.name[:46], run_changed, ", ".join(file_hits))
        logger.info(
            "  TOTAL: %d snapshot value(s) across %d file(s) in %d run(s)",
            grand_snap, grand_run_files, len(runs),
        )

    if not args.apply:
        logger.info("")
        logger.info("DRY-RUN. Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
