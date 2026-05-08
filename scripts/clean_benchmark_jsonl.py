#!/usr/bin/env python3
"""Strip a labelled benchmark JSONL down to a clean, shareable form.

Filters and rewrites benchmark records exported by the labelling
workbench (``benchmark_<error_type>_<run_id>.jsonl``) so the output
is safe to hand off to a downstream team:

* keeps only rows whose ``label`` is ``"error"`` or ``"ambiguous"``
  (``"clean"`` rows, which are confirmed non-errors, are dropped),
* removes provenance / annotator fields:
  ``detection_ids``, ``cluster_ids``, ``annotators``, ``note``,
* with ``--exclusions`` set, also drops records that point at a
  CSV / row / column the run excluded, so the cleaned file lines
  up exactly with the sanitised ``sources_<run_id>.zip``.

Every other field is preserved verbatim so column ordering /
``original_value`` / ``corrected_value`` / ``llm_suggested_value``
still line up with the source CSV.

Examples:

    # Single file -> sibling ``*.clean.jsonl``
    python scripts/clean_benchmark_jsonl.py \\
        benchmark/benchmark_typo_20260507T215150Z_gpt5-family.jsonl

    # Drop records that reference excluded files / rows / columns
    python scripts/clean_benchmark_jsonl.py \\
        --exclusions outputs/<run_id>/exclusions.jsonl \\
        benchmark/benchmark_typo_<run_id>.jsonl

    # Glob -> explicit output directory
    python scripts/clean_benchmark_jsonl.py \\
        --out-dir benchmark/clean \\
        benchmark/benchmark_*.jsonl

    # Stdin -> stdout
    cat in.jsonl | python scripts/clean_benchmark_jsonl.py - > out.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, TextIO

# Verdict values to keep. Drop "clean" (confirmed non-errors are
# noise for a downstream cleaner) and any unexpected statuses.
KEEP_LABELS = frozenset({"error", "ambiguous"})

# Provenance / annotator fields stripped from every emitted record.
DROP_FIELDS = ("detection_ids", "cluster_ids", "annotators", "note")


class ExclusionFilter:
    """Resolve a run's ``exclusions.jsonl`` into per-scope lookups.

    The store is append-only with tombstones: a follow-up record
    with the same ``id`` and ``deleted=True`` retracts an earlier
    exclusion. We replay the file keeping the latest record per id
    (matching ``ExclusionStore.active``) and bucket the survivors
    by scope into:

    * ``csvs`` - sources to drop entirely;
    * ``rows`` - ``source -> {row_idx}`` rows to drop;
    * ``cols`` - ``source -> {column_name}`` columns to drop.
    """

    def __init__(
        self,
        csvs: set[str],
        rows: dict[str, set[int]],
        cols: dict[str, set[str]],
    ) -> None:
        self.csvs = csvs
        self.rows = rows
        self.cols = cols

    @classmethod
    def from_path(cls, path: Path) -> "ExclusionFilter":
        latest: dict[str, dict] = {}
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                eid = rec.get("id")
                if eid:
                    latest[eid] = rec
        csvs: set[str] = set()
        rows: dict[str, set[int]] = {}
        cols: dict[str, set[str]] = {}
        for rec in latest.values():
            if rec.get("deleted"):
                continue
            scope = rec.get("scope")
            source = rec.get("source") or ""
            if not source:
                continue
            if scope == "csv":
                csvs.add(source)
            elif scope == "rows":
                bucket = rows.setdefault(source, set())
                for r in rec.get("rows") or []:
                    try:
                        bucket.add(int(r))
                    except (TypeError, ValueError):
                        continue
            elif scope == "columns":
                bucket = cols.setdefault(source, set())
                for c in rec.get("columns") or []:
                    if c is not None:
                        bucket.add(str(c))
        return cls(csvs, rows, cols)

    def excludes(self, rec: dict) -> bool:
        source = rec.get("source") or ""
        if source in self.csvs:
            return True
        try:
            row_i = int(rec.get("row"))
        except (TypeError, ValueError):
            row_i = None
        if row_i is not None and row_i in self.rows.get(source, ()):
            return True
        column = rec.get("column") or ""
        if column and column in self.cols.get(source, ()):
            return True
        return False


def _clean_record(
    rec: dict, exclusions: ExclusionFilter | None = None,
) -> dict | None:
    """Return ``rec`` with provenance fields removed, or ``None`` if
    the record should be dropped (label not in :data:`KEEP_LABELS`,
    or excluded by the run's exclusions ledger when one was loaded)."""
    if rec.get("label") not in KEEP_LABELS:
        return None
    if exclusions is not None and exclusions.excludes(rec):
        return None
    return {k: v for k, v in rec.items() if k not in DROP_FIELDS}


def _iter_clean(
    src: Iterable[str],
    exclusions: ExclusionFilter | None = None,
) -> Iterator[tuple[dict, int]]:
    """Yield ``(cleaned_record, source_lineno)`` for usable lines.

    Skips blank lines silently. Logs a warning on malformed JSON
    (and continues) so a single corrupt line doesn't poison the run.
    """
    for lineno, raw in enumerate(src, 1):
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                f"warning: line {lineno}: invalid JSON ({exc}); skipped",
                file=sys.stderr,
            )
            continue
        cleaned = _clean_record(rec, exclusions=exclusions)
        if cleaned is not None:
            yield cleaned, lineno


def _write(records: Iterable[dict], out: TextIO) -> int:
    n = 0
    for rec in records:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    return n


def _process_file(
    in_path: Path,
    out_path: Path,
    exclusions: ExclusionFilter | None = None,
) -> tuple[int, int]:
    """Clean ``in_path`` -> ``out_path``. Returns ``(kept, total)``."""
    kept = 0
    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with in_path.open("r", encoding="utf-8") as src, tmp.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
        # Re-iterate cleanly: count above used a separate pass so
        # ``total`` reflects all attempted records (including dropped
        # ones), giving a meaningful "kept N/M" summary line below.
        src.seek(0)
        for cleaned, _ in _iter_clean(src, exclusions=exclusions):
            dst.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1
    tmp.replace(out_path)
    return kept, total


def _default_out_path(in_path: Path) -> Path:
    """``foo.jsonl`` -> ``foo.clean.jsonl`` (preserves directory)."""
    if in_path.suffixes[-1:] == [".jsonl"]:
        return in_path.with_suffix(".clean.jsonl")
    return in_path.with_name(in_path.name + ".clean.jsonl")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Strip a labelled benchmark JSONL to error+ambiguous "
            "rows and remove provenance fields."
        )
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Input JSONL paths. Use '-' to read from stdin and "
            "write to stdout."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Write cleaned files into this directory using the "
            "same basenames. Default: alongside each input as "
            "<name>.clean.jsonl."
        ),
    )
    p.add_argument(
        "--suffix",
        default=".clean.jsonl",
        help=(
            "Suffix appended (replacing '.jsonl') when writing next "
            "to the input. Ignored when --out-dir is set or when "
            "reading from stdin. Default: .clean.jsonl"
        ),
    )
    p.add_argument(
        "--exclusions",
        type=Path,
        default=None,
        help=(
            "Path to a run's exclusions.jsonl. When set, records "
            "pointing at an excluded CSV / row / column are dropped "
            "too, so the cleaned file aligns with sources_<run_id>.zip."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    exclusions: ExclusionFilter | None = None
    if args.exclusions is not None:
        if not args.exclusions.is_file():
            print(
                f"error: --exclusions path not found: {args.exclusions}",
                file=sys.stderr,
            )
            return 2
        exclusions = ExclusionFilter.from_path(args.exclusions)
        print(
            f"loaded exclusions from {args.exclusions}: "
            f"{len(exclusions.csvs)} csv-scope, "
            f"{sum(len(v) for v in exclusions.rows.values())} row-scope, "
            f"{sum(len(v) for v in exclusions.cols.values())} column-scope",
            file=sys.stderr,
        )

    # Stdin/stdout streaming mode: no .tmp dance, no summary.
    if args.inputs == ["-"]:
        kept = _write(
            (rec for rec, _ in _iter_clean(sys.stdin, exclusions=exclusions)),
            sys.stdout,
        )
        print(f"kept {kept} record(s) from stdin", file=sys.stderr)
        return 0

    grand_kept = 0
    grand_total = 0
    for raw in args.inputs:
        in_path = Path(raw)
        if not in_path.is_file():
            print(f"error: not a file: {in_path}", file=sys.stderr)
            return 2
        if args.out_dir is not None:
            out_path = args.out_dir / in_path.name
        else:
            if in_path.suffix == ".jsonl":
                out_path = in_path.with_suffix(args.suffix)
            else:
                out_path = in_path.with_name(in_path.name + args.suffix)
        kept, total = _process_file(
            in_path, out_path, exclusions=exclusions,
        )
        grand_kept += kept
        grand_total += total
        print(f"{in_path}  ->  {out_path}  ({kept}/{total} kept)")

    if len(args.inputs) > 1:
        print(f"total: {grand_kept}/{grand_total} kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
