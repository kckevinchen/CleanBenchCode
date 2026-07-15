"""Aggregate the v4 fixed cell-level typo benchmark into per-column v4 records.

This script reads ``benchmark/benchmark_typo_*.jsonl`` (the post-
cluster-contamination-fix export) and groups cells by ``(source, column)``,
producing one v4 RFT JSONL record per column.

Trivial-typo filtering
----------------------
v4 has its own trivial-correction filter
(``experiments._typo_v4_task.is_v4_trivial_correction``) that is
STRICTER than v3's: it only excludes pure-padding-whitespace and
pure-casing edits. Spacing-tokenisation edits (``DuPage`` <-> ``Du Page``)
are kept because they're substantive ``spacing``-flavour typos under
v4 \u00a71.

Per the user's instruction ("exclude trivial cases when sampling or
ignore them during evaluation"), this script applies the v4-trivial
filter at SPLIT time, so the v4 grader never sees them.

Output (under ``RFT/typo/v4/data/``):

* ``eval.jsonl``    -- per-column records (the v4 evaluation set).
* ``demo.jsonl``    -- small stratified held-out subset for synth
                       exemplars (mirrors v3's demo split).
* ``manifest.json`` -- provenance + counts.

Usage
-----
::

    python -m RFT.typo.v4.scripts.split_typo_benchmark_v4 \\
        --benchmark benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl \\
        --csv-root data/source_inconsistency_bing21 \\
        --output-dir RFT/typo/v4/data \\
        --demo-size 20
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v4 import _task as _typo_v4_task  # noqa: E402

logger = logging.getLogger("split_typo_benchmark_v4")


DEFAULT_BENCHMARK = Path(
    "benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl"
)
DEFAULT_CSV_ROOT = Path("data/source_inconsistency_bing21")
DEFAULT_OUTPUT_DIR = Path("RFT/typo/v4/data")
DEFAULT_DEMO_SIZE = 20
DEFAULT_SEED = 0


def _build_csv_index(csv_root: Path) -> dict[str, Path]:
    """Basename -> absolute path, scanning ``csv_root`` recursively."""
    return {p.name: p for p in csv_root.rglob("*.csv")}


def _load_column_distinct_values(
    csv_path: Path, column: str,
) -> list[tuple[str, int]]:
    """Read every value in ``column`` from ``csv_path`` and return a
    sorted ``(value, count)`` list (most-frequent first).
    """
    counts: collections.Counter[str] = collections.Counter()
    try:
        with csv_path.open(encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or column not in reader.fieldnames:
                logger.warning(
                    "Column %r not in %s; available=%s",
                    column, csv_path.name, reader.fieldnames,
                )
                return []
            for row in reader:
                val = row.get(column)
                if val is None:
                    continue
                val = str(val)
                if not val or val.strip() == "":
                    continue
                counts[val] += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading %s: %s", csv_path, exc)
        return []
    return counts.most_common()


def _dedupe_gold_typos(
    rows: list[dict[str, Any]],
    *,
    stats: dict[str, int],
) -> list[dict[str, str]]:
    """Collapse benchmark 'error' rows for the same value into one gold entry.

    Applies the v4-trivial filter on the way through: a (value,
    correction) pair whose edit is pure padding/casing is dropped and
    counted in ``stats['trivial_dropped']`` so the manifest records
    what was excluded.
    """
    by_value: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for r in rows:
        if r.get("label") != "error":
            continue
        v = r.get("original_value")
        c = r.get("corrected_value")
        if not isinstance(v, str) or not v:
            continue
        if not isinstance(c, str) or not c:
            continue
        if _typo_v4_task.is_v4_trivial_correction(v, c):
            stats["trivial_dropped"] += 1
            continue
        by_value[v][c] += 1

    out: list[dict[str, str]] = []
    for value, counter in by_value.items():
        correction, _count = max(
            counter.items(), key=lambda kv: (kv[1], -ord(kv[0][:1] or "z"))
        )
        out.append({"value": value, "correction": correction})
    return sorted(out, key=lambda d: d["value"])


def _distinct_ambiguous_values(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        if r.get("label") != "ambiguous":
            continue
        v = r.get("original_value")
        if isinstance(v, str) and v and v not in seen:
            seen.add(v)
            out.append(v)
    return sorted(out)


def _aggregate(
    rows: list[dict[str, Any]],
    csv_index: dict[str, Path],
    *,
    max_distinct: int,
    english_only: bool,
    english_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_col: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for r in rows:
        src = r.get("source")
        col = r.get("column")
        if not src or not col:
            continue
        by_col[(src, col)].append(r)

    stats: dict[str, int] = {
        "benchmark_rows_in": len(rows),
        "columns_total": len(by_col),
        "columns_skipped_no_csv": 0,
        "columns_skipped_no_distinct": 0,
        "columns_skipped_non_english": 0,
        "columns_with_gold_typos": 0,
        "columns_with_only_ambiguous": 0,
        "columns_pure_clean": 0,
        "columns_emitted": 0,
        "gold_typos_total": 0,
        "ambiguous_values_total": 0,
        "values_with_counts_total": 0,
        "trivial_dropped": 0,
        "gold_values_dropped_not_in_csv": 0,
    }

    items: list[dict[str, Any]] = []
    for (src, col), grouped in sorted(by_col.items()):
        csv_path = csv_index.get(src)
        if csv_path is None:
            stats["columns_skipped_no_csv"] += 1
            continue
        values_with_counts = _load_column_distinct_values(csv_path, col)
        if not values_with_counts:
            stats["columns_skipped_no_distinct"] += 1
            continue
        if english_only and not _typo_v4_task.is_english_column(
            col,
            (v for v, _ in values_with_counts),
            value_threshold=english_threshold,
        ):
            stats["columns_skipped_non_english"] += 1
            continue
        gold_typos = _dedupe_gold_typos(grouped, stats=stats)
        ambiguous_values = _distinct_ambiguous_values(grouped)

        # Sanity: every gold typo value must appear in the column's
        # distinct values; otherwise the trainee can't be expected to
        # spot it (and build_record would reject it).
        distinct_set = {v for v, _ in values_with_counts}
        gold_before = len(gold_typos)
        gold_typos = [g for g in gold_typos if g["value"] in distinct_set]
        stats["gold_values_dropped_not_in_csv"] += (
            gold_before - len(gold_typos)
        )
        ambiguous_values = [
            v for v in ambiguous_values if v in distinct_set
        ]

        if gold_typos:
            stats["columns_with_gold_typos"] += 1
        elif ambiguous_values:
            stats["columns_with_only_ambiguous"] += 1
        else:
            stats["columns_pure_clean"] += 1

        stats["gold_typos_total"] += len(gold_typos)
        stats["ambiguous_values_total"] += len(ambiguous_values)
        stats["values_with_counts_total"] += len(values_with_counts)
        stats["columns_emitted"] += 1

        items.append({
            "source": src,
            "column": col,
            "values_with_counts": values_with_counts,
            "gold_typos": gold_typos,
            "ambiguous_values": ambiguous_values,
            "max_distinct": max_distinct,
        })
    return items, stats


def _content_hash(items: list[dict[str, Any]]) -> str:
    keys = sorted(
        hashlib.sha256(
            json.dumps(
                {"source": it["source"], "column": it["column"]},
                sort_keys=True, ensure_ascii=False,
            ).encode()
        ).hexdigest()[:16]
        for it in items
    )
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _stratify_items(items: list[dict[str, Any]]) -> dict[
    str, list[dict[str, Any]]
]:
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for it in items:
        if it["gold_typos"]:
            buckets["has_typo"].append(it)
        elif it["ambiguous_values"]:
            buckets["ambiguous_only"].append(it)
        else:
            buckets["clean"].append(it)
    return buckets


def _proportional_demo_sizes(
    buckets: dict[str, list[dict[str, Any]]], demo_size: int,
) -> dict[str, int]:
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return {}
    sizes = {
        k: max(1, len(v) * demo_size // total)
        for k, v in buckets.items() if v
    }
    assigned = sum(sizes.values())
    if assigned > demo_size:
        excess = assigned - demo_size
        for k, _ in sorted(sizes.items(), key=lambda kv: -kv[1]):
            cut = min(sizes[k] - 1, excess)
            sizes[k] -= cut
            excess -= cut
            if excess == 0:
                break
    elif assigned < demo_size:
        rem = demo_size - assigned
        for k, _ in sorted(sizes.items(), key=lambda kv: -len(buckets[kv[0]])):
            headroom = len(buckets[k]) - sizes[k]
            add = min(headroom, rem)
            sizes[k] += add
            rem -= add
            if rem == 0:
                break
    return sizes


def split_v4(
    *,
    benchmark_path: Path,
    csv_root: Path,
    output_dir: Path,
    demo_size: int = DEFAULT_DEMO_SIZE,
    seed: int = DEFAULT_SEED,
    max_distinct: int = _typo_v4_task.TYPO_V4_MAX_DISTINCT,
    english_only: bool = True,
    english_threshold: float = (
        _typo_v4_task.TYPO_V4_ENGLISH_VALUE_THRESHOLD
    ),
) -> dict[str, Any]:
    if not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)
    if not csv_root.exists():
        raise FileNotFoundError(csv_root)

    csv_index = _build_csv_index(csv_root)
    if not csv_index:
        raise RuntimeError(
            f"No CSVs found under {csv_root}; aborting before grouping."
        )

    rows: list[dict[str, Any]] = []
    with benchmark_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("Loaded %d benchmark rows from %s", len(rows), benchmark_path)

    items, stats = _aggregate(
        rows, csv_index,
        max_distinct=max_distinct,
        english_only=english_only,
        english_threshold=english_threshold,
    )
    logger.info(
        "Aggregated %d columns (%d with typos, %d ambiguous-only, "
        "%d pure clean; %d non-English skipped; %d trivial corrections "
        "dropped at split time).",
        stats["columns_emitted"], stats["columns_with_gold_typos"],
        stats["columns_with_only_ambiguous"], stats["columns_pure_clean"],
        stats["columns_skipped_non_english"], stats["trivial_dropped"],
    )

    buckets = _stratify_items(items)
    sizes = _proportional_demo_sizes(buckets, demo_size)
    rng = random.Random(seed)
    demo_items: list[dict[str, Any]] = []
    eval_items: list[dict[str, Any]] = []
    for label, bucket in buckets.items():
        n = sizes.get(label, 0)
        shuffled = bucket[:]
        rng.shuffle(shuffled)
        demo_items.extend(shuffled[:n])
        eval_items.extend(shuffled[n:])

    def _sort_key(it: dict[str, Any]) -> tuple[str, str]:
        return (it["source"], it["column"])

    demo_items.sort(key=_sort_key)
    eval_items.sort(key=_sort_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    demo_path = output_dir / "demo.jsonl"
    eval_path = output_dir / "eval.jsonl"
    manifest_path = output_dir / "manifest.json"

    def _write(path: Path, items_: list[dict[str, Any]]) -> int:
        n = 0
        with path.open("w", encoding="utf-8") as fh:
            for it in items_:
                rec = _typo_v4_task.build_record(it)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        return n

    n_demo_w = _write(demo_path, demo_items)
    n_eval_w = _write(eval_path, eval_items)

    def _label_counts(items_: list[dict[str, Any]]) -> dict[str, int]:
        c: collections.Counter[str] = collections.Counter()
        for it in items_:
            if it["gold_typos"]:
                c["has_typo"] += 1
            elif it["ambiguous_values"]:
                c["ambiguous_only"] += 1
            else:
                c["clean"] += 1
        return dict(c)

    manifest = {
        "benchmark_path": str(benchmark_path),
        "csv_root": str(csv_root),
        "csv_index_size": len(csv_index),
        "max_distinct": max_distinct,
        "seed": seed,
        "english_only": english_only,
        "english_threshold": english_threshold,
        "demo_size_requested": demo_size,
        "demo_size_actual": n_demo_w,
        "eval_size_actual": n_eval_w,
        "stats": stats,
        "demo_label_counts": _label_counts(demo_items),
        "eval_label_counts": _label_counts(eval_items),
        "demo_content_hash": _content_hash(demo_items),
        "eval_content_hash": _content_hash(eval_items),
        "demo_sources": sorted({it["source"] for it in demo_items}),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "Wrote %d demo + %d eval columns to %s",
        n_demo_w, n_eval_w, output_dir,
    )
    logger.info("Demo label counts: %s", manifest["demo_label_counts"])
    logger.info("Eval label counts: %s", manifest["eval_label_counts"])
    return manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demo-size", type=int, default=DEFAULT_DEMO_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-distinct", type=int,
        default=_typo_v4_task.TYPO_V4_MAX_DISTINCT,
    )
    parser.add_argument("--no-english-only", action="store_true")
    parser.add_argument(
        "--english-threshold", type=float,
        default=_typo_v4_task.TYPO_V4_ENGLISH_VALUE_THRESHOLD,
    )
    args = parser.parse_args(argv)

    manifest = split_v4(
        benchmark_path=args.benchmark,
        csv_root=args.csv_root,
        output_dir=args.output_dir,
        demo_size=args.demo_size,
        seed=args.seed,
        max_distinct=args.max_distinct,
        english_only=not args.no_english_only,
        english_threshold=args.english_threshold,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
