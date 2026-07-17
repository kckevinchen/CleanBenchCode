"""Gold loading, CSV resolution, and the by-source train/test split.

Reads the per-cell inconsistency benchmark
(``benchmark/merged_v2/benchmark_inconsistency_cells.jsonl`` — fields
``source,row,column,original_value,label,corrected_value,dataset``) into the
``(source, column)`` gold shape the scorer needs, and splits **by source**
(stratified by ``dataset`` and has-error) so no CSV straddles dev and test —
zero leakage. ``dataset`` attribution is kept so the final test report can be
broken down pbi vs bing.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any

from external.baseline.inconsistency._task import norm

# Repo-relative defaults (verified present at build time).
DEFAULT_BENCHMARK = "benchmark/merged_v2/benchmark_inconsistency_cells.jsonl"
DEFAULT_DATA_DIRS = ("data/curated_pbi_500_dedup", "data/curated_bing21_500_dedup")
# Canonical immunity set produced alongside the benchmark by
# experiments/export_merged_benchmark.py ("skip + exclusion sentinels"):
# whole CSVs / columns / rows the curators ignored, plus individual skipped
# cells. Flagging any of these is neither a true nor a false positive.
DEFAULT_EXCLUSIONS = "benchmark/merged_v1/ignored.jsonl"


class Exclusions:
    """Curator-ignored CSVs / columns / rows / cells (FP-immune, dropped from eval).

    * ``sources``  — whole CSVs to drop (``exclusion`` scope ``csv``).
    * ``columns``  — ``(source, column)`` to drop (scope ``columns``).
    * ``rows``     — ``source -> {row_index}`` to drop (scope ``rows``).
    * ``cells``    — ``(source, column) -> {norm(value)}`` skipped cells (and any
                     value whose every occurrence is in an excluded row) — made
                     FP-immune rather than dropping the whole column.
    """

    def __init__(self) -> None:
        self.sources: set[str] = set()
        self.columns: set[tuple[str, str]] = set()
        self.rows: dict[str, set[int]] = collections.defaultdict(set)
        self.cells: dict[tuple[str, str], set[str]] = collections.defaultdict(set)

    def __bool__(self) -> bool:
        return bool(self.sources or self.columns or self.rows or self.cells)

    def drops_column(self, source: str, column: str) -> bool:
        return source in self.sources or (source, column) in self.columns

    def immune_values(self, source: str, column: str) -> set[str]:
        return self.cells.get((source, column), set())

    def stats(self) -> dict[str, int]:
        return {
            "sources": len(self.sources),
            "columns": len(self.columns),
            "rows": sum(len(v) for v in self.rows.values()),
            "skip_cells": sum(len(v) for v in self.cells.values()),
        }


def load_exclusions(
    path: str | Path = DEFAULT_EXCLUSIONS,
    *,
    csv_map: dict[str, Path] | None = None,
) -> Exclusions:
    """Parse ``ignored.jsonl`` into an :class:`Exclusions`.

    Row-scope exclusions are resolved to immune ``(column, value)`` pairs via
    ``csv_map`` (a value is immunised only when *every* one of its occurrences in
    the column falls in an excluded row, so a frequent value is never wrongly
    excluded). Without ``csv_map`` row exclusions are recorded but not resolved.
    """
    exc = Exclusions()
    p = Path(path)
    if not p.exists():
        return exc
    for raw_line in p.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        r = json.loads(line)
        src = r.get("source")
        if not src:
            continue
        reason, scope = r.get("reason"), r.get("scope")
        if reason == "exclusion" and scope == "csv":
            exc.sources.add(src)
        elif reason == "exclusion" and scope == "columns":
            exc.columns.add((src, r.get("column")))
        elif reason == "exclusion" and scope == "rows":
            if r.get("row") is not None:
                exc.rows[src].add(int(r["row"]))
        elif reason == "skip":
            v = r.get("original_value")
            if v is not None and r.get("column") is not None:
                exc.cells[(src, r["column"])].add(norm(v))

    if csv_map:
        _resolve_row_exclusions(exc, csv_map)
    return exc


def _resolve_row_exclusions(exc: Exclusions, csv_map: dict[str, Path]) -> None:
    """Turn ``source -> {row}`` exclusions into immune ``(source, col) -> values``.

    Only values entirely contained in excluded rows are immunised (a value that
    also appears in a kept row stays scorable).
    """
    import pandas as pd

    for src, rows in exc.rows.items():
        if src in exc.sources:
            continue
        path = csv_map.get(src)
        if path is None:
            continue
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception:
            continue
        for col in df.columns:
            series = df[col].astype(str).map(norm)
            for ridx in rows:
                if 0 <= ridx < len(series):
                    val = series.iloc[ridx]
                    # Immune only if every occurrence of this value is excluded.
                    if set(series.index[series == val]).issubset(rows):
                        exc.cells[(src, str(col))].add(val)


def acceptable_set(rec: dict[str, Any]) -> set[str]:
    """Normalised acceptable corrections for a gold-error record.

    ``corrected_value`` may be a string, a list, or ``None`` (merged_v2 has all
    three). An empty acceptable set means "any flag counts as a TP, but no
    specific correction is graded as right".
    """
    out: set[str] = set()
    cv = rec.get("corrected_value")
    items = cv if isinstance(cv, list) else [cv]
    for x in items:
        if isinstance(x, str) and x.strip():
            out.add(norm(x))
    return out


def load_gold(
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    """Return ``(cols, source_dataset)``.

    ``cols[(source, column)] = {"error": {nvalue: set(acceptable)}, "amb": {nvalue}}``
    ``source_dataset[source] = dataset``  (e.g. ``"pbi_500"`` / ``"bing21_500"``).
    """
    bench = Path(benchmark_path)
    err: dict[tuple[str, str], dict[str, set[str]]] = collections.defaultdict(dict)
    amb: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    source_dataset: dict[str, str] = {}
    for line in bench.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        src = r["source"]
        if r.get("dataset"):
            source_dataset[src] = r["dataset"]
        key = (src, r["column"])
        nv = norm(r["original_value"])
        label = r.get("label")
        if label == "error":
            err[key].setdefault(nv, set()).update(acceptable_set(r))
        elif label == "ambiguous":
            amb[key].add(nv)
    cols: dict[tuple[str, str], dict[str, Any]] = {}
    for key in set(err) | set(amb):
        e = dict(err.get(key, {}))
        a = {v for v in amb.get(key, set()) if v not in e}
        cols[key] = {"error": e, "amb": a}
    return cols, source_dataset


def csv_lookup(data_dirs: list[str] | tuple[str, ...] = DEFAULT_DATA_DIRS) -> dict[str, Path]:
    """Map ``<source basename>.csv -> Path`` across the dataset dirs (recursive)."""
    out: dict[str, Path] = {}
    for d in data_dirs:
        for p in Path(d).rglob("*.csv"):
            out.setdefault(p.name, p)
    return out


def error_bearing_sources(cols: dict[tuple[str, str], dict[str, Any]]) -> set[str]:
    """Sources with >=1 gold-error value."""
    return {s for (s, _c), g in cols.items() if g["error"]}


def split_sources(
    source_dataset: dict[str, str],
    cols: dict[tuple[str, str], dict[str, Any]],
    *,
    dev_frac: float = 0.34,
    val_frac: float = 0.16,
    seed: int = 13,
    dev_dataset: str | None = None,
) -> dict[str, Any]:
    """Split sources into dev/val/test, by source, stratified by dataset and has-error.

    Three-way split for overfit-resistant optimization:
      * **dev**  — the reviewer sees these errors (the "train" signal).
      * **val**  — used ONLY to accept/reject candidate prompts (selection); the
        reviewer never sees its individual errors, so a val improvement is a real
        generalization signal, not memorization of dev.
      * **test** — scored once at the very end; never touched during the loop.

    Deterministic (seeded shuffle over *sorted* sources). When ``dev_dataset`` is
    given, dev+val are sampled only from that dataset and every other dataset goes
    entirely to test (out-of-distribution test).
    """
    eb = error_bearing_sources(cols)
    by_ds: dict[str, list[str]] = collections.defaultdict(list)
    for src, ds in source_dataset.items():
        by_ds[ds].append(src)

    dev: set[str] = set()
    val: set[str] = set()
    test: set[str] = set()
    rng = random.Random(seed)
    for ds in sorted(by_ds):
        if dev_dataset is not None and ds != dev_dataset:
            test.update(by_ds[ds])
            continue
        # Stratify within the dataset: error-bearing and clean sampled separately
        # so every split gets gold signal proportional to its fraction.
        for stratum in (
            sorted(s for s in by_ds[ds] if s in eb),
            sorted(s for s in by_ds[ds] if s not in eb),
        ):
            shuffled = list(stratum)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_dev = round(dev_frac * n)
            n_val = round(val_frac * n)
            dev.update(shuffled[:n_dev])
            val.update(shuffled[n_dev : n_dev + n_val])
            test.update(shuffled[n_dev + n_val :])

    def _stats(srcs: set[str]) -> dict[str, Any]:
        per_ds = collections.Counter(source_dataset.get(s, "?") for s in srcs)
        n_err = sum(len(g["error"]) for (s, _c), g in cols.items() if s in srcs)
        return {
            "n_sources": len(srcs),
            "n_error_bearing": len(srcs & eb),
            "n_error_values": n_err,
            "by_dataset": dict(per_ds),
        }

    return {
        "dev_frac": dev_frac,
        "val_frac": val_frac,
        "seed": seed,
        "dev_dataset": dev_dataset,
        "dev": sorted(dev),
        "val": sorted(val),
        "test": sorted(test),
        "dev_stats": _stats(dev),
        "val_stats": _stats(val),
        "test_stats": _stats(test),
    }
