"""Aggregate the v2 cell-level typo benchmark into the v3 per-column shape.

Reads the v2 split output (cell-level rows from
``experiments.split_typo_benchmark`` with ``--exclude-trivial-typos``)
and re-shapes it for the v3 column-audit RFT task:

* Groups rows by ``(source, column)``.
* For each (source, column), loads the source CSV from the corpus
  root and computes the distinct-value frequency map (with the
  ``TYPO_V3_MAX_DISTINCT`` cap matching the trainee's view).
* Builds the per-column gold typo set: distinct values whose v2
  ``label`` was ``error``, with their ``corrected_value`` as the
  intended canonical form. Repeated identical typos collapse to
  one entry; a value with conflicting gold corrections (rare)
  takes the most common one.
* Builds the per-column ambiguous-values set: distinct values whose
  v2 ``label`` was ``ambiguous``. The v3 grader credits predictions
  on these values as neither TP nor FP.
* Emits one v3 RFT JSONL row per (source, column) pair into
  ``eval.jsonl`` (always) and ``demo.jsonl`` (a small held-back
  subset that synthesis can use as exemplars).

Why aggregate v2's split rather than the raw benchmark?

* v2's split already applied the trivial-typo filter and dropped
  rows the binary task can't handle. Aggregating after that filter
  keeps v2 and v3 evaluating exactly the same set of substantive
  typos -- a fair head-to-head.

Usage
-----
::

    python -m experiments.split_typo_benchmark_v3 \\
        --v2-eval RFT/typo/v2/data/eval.jsonl \\
        --v2-demo RFT/typo/v2/data/demo.jsonl \\
        --csv-root data/source_inconsistency_bing21 \\
        --output-dir outputs/typo_rft/benchmark_split_v3 \\
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common import _typo_v3_task
from RFT._common._typo_task import is_trivial_typo_correction  # noqa: E402

logger = logging.getLogger("split_typo_benchmark_v3")


DEFAULT_V2_EVAL = Path("RFT/typo/v2/data/eval.jsonl")
DEFAULT_V2_DEMO = Path("RFT/typo/v2/data/demo.jsonl")
DEFAULT_CSV_ROOT = Path("data/source_inconsistency_bing21")
DEFAULT_OUTPUT_DIR = Path("outputs/typo_rft/benchmark_split_v3")
DEFAULT_DEMO_SIZE = 20
DEFAULT_SEED = 0


def _build_csv_index(csv_root: Path) -> dict[str, Path]:
    """basename -> absolute path, scanning ``csv_root`` recursively."""
    return {p.name: p for p in csv_root.rglob("*.csv")}


def _load_column_distinct_values(
    csv_path: Path, column: str,
) -> list[tuple[str, int]]:
    """Read every value in ``column`` from ``csv_path`` and return
    a sorted ``(value, count)`` list (most-frequent first).

    Skips null / empty cells; the v3 task treats them as absent.
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
) -> list[dict[str, str]]:
    """Collapse multiple v2 'error' rows for the same value -> one gold entry.

    When the same ``original_value`` appears in several rows of the
    same column with different ``corrected_value`` strings (rare), we
    pick the most common; ties broken by lexical order so the result
    is deterministic.
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
        if is_trivial_typo_correction(v, c):
            # Defence-in-depth: v2 should have filtered these already.
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
    english_only: bool = True,
    english_threshold: float = (
        _typo_v3_task.TYPO_V3_ENGLISH_VALUE_THRESHOLD
    ),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Group cell-level rows into per-column v3 items.

    Returns ``(items, stats)`` where ``items`` is the list of dicts
    suitable for :func:`_typo_v3_task.build_record`, and ``stats`` is
    a small per-stage row-count dict for the manifest.

    With ``english_only=True`` (the v3 default), columns whose
    header has any non-ASCII alphabetic character, or whose values
    are predominantly non-ASCII, are skipped. This keeps the v3
    eval set aligned with the English-only training corpus.
    """
    by_col: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for r in rows:
        src = r.get("source")
        col = r.get("column")
        if not src or not col:
            continue
        by_col[(src, col)].append(r)

    stats = {
        "v2_rows_in": len(rows),
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
    }

    items: list[dict[str, Any]] = []
    for (src, col), grouped in sorted(by_col.items()):
        csv_path = csv_index.get(src)
        if csv_path is None:
            stats["columns_skipped_no_csv"] += 1
            logger.warning("No CSV for source=%s (column=%s)", src, col)
            continue
        values_with_counts = _load_column_distinct_values(csv_path, col)
        if not values_with_counts:
            stats["columns_skipped_no_distinct"] += 1
            continue
        if english_only and not _typo_v3_task.is_english_column(
            col,
            (v for v, _ in values_with_counts),
            value_threshold=english_threshold,
        ):
            stats["columns_skipped_non_english"] += 1
            continue
        gold_typos = _dedupe_gold_typos(grouped)
        ambiguous_values = _distinct_ambiguous_values(grouped)
        # Sanity: every gold typo value MUST appear in the column's
        # distinct values; otherwise the trainee can't be expected to
        # spot it (and build_record would reject it).
        distinct_set = {v for v, _ in values_with_counts}
        gold_typos = [g for g in gold_typos if g["value"] in distinct_set]
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
    """Stratify columns by their gold composition for demo selection.

    Three strata, matched to v2's row-level strata so the demo set
    has a useful spread:

    * ``has_typo``      -- at least one substantive gold typo.
    * ``ambiguous_only`` -- no typos but at least one ambiguous value.
    * ``clean``         -- no typos, no ambiguous; pure clean column.
    """
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
    """Same allocator as v2: floor=1 per bucket, then largest-fraction.

    Guarantees the demo has at least one column from each bucket so
    synthesis can draw exemplars across all three patterns.
    """
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return {}
    sizes = {k: max(1, len(v) * demo_size // total) for k, v in buckets.items()
             if v}
    assigned = sum(sizes.values())
    if assigned > demo_size:
        # Trim excess from the biggest bucket first.
        excess = assigned - demo_size
        for k, _ in sorted(sizes.items(), key=lambda kv: -kv[1]):
            cut = min(sizes[k] - 1, excess)
            sizes[k] -= cut
            excess -= cut
            if excess == 0:
                break
    elif assigned < demo_size:
        # Distribute remainder by bucket size (largest gets first).
        rem = demo_size - assigned
        for k, _ in sorted(sizes.items(), key=lambda kv: -len(buckets[kv[0]])):
            headroom = len(buckets[k]) - sizes[k]
            add = min(headroom, rem)
            sizes[k] += add
            rem -= add
            if rem == 0:
                break
    return sizes


def split_v3(
    *,
    v2_eval_path: Path,
    v2_demo_path: Path | None,
    csv_root: Path,
    output_dir: Path,
    demo_size: int = DEFAULT_DEMO_SIZE,
    seed: int = DEFAULT_SEED,
    max_distinct: int = _typo_v3_task.TYPO_V3_MAX_DISTINCT,
    english_only: bool = True,
    english_threshold: float = (
        _typo_v3_task.TYPO_V3_ENGLISH_VALUE_THRESHOLD
    ),
) -> dict[str, Any]:
    """Build v3 eval + demo splits from the v2 cell-level data.

    The ``v2_demo`` rows are also folded in (they were held out from
    eval, so for v3 we still include them as eval-data when their
    column wasn't already covered by ``v2_eval``).
    """
    if not v2_eval_path.exists():
        raise FileNotFoundError(v2_eval_path)
    if not csv_root.exists():
        raise FileNotFoundError(csv_root)

    csv_index = _build_csv_index(csv_root)
    if not csv_index:
        raise RuntimeError(
            f"No CSVs found under {csv_root}; aborting before grouping."
        )

    rows: list[dict[str, Any]] = []
    with v2_eval_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n_eval = len(rows)
    n_demo = 0
    if v2_demo_path and v2_demo_path.exists():
        seen_keys = {(r["source"], r["column"], r["row"]) for r in rows}
        with v2_demo_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["source"], r["column"], r["row"])
                if key not in seen_keys:
                    rows.append(r)
                    n_demo += 1
    logger.info("Loaded %d v2 rows (%d eval + %d demo)", len(rows),
                n_eval, n_demo)

    items, stats = _aggregate(
        rows, csv_index,
        max_distinct=max_distinct,
        english_only=english_only,
        english_threshold=english_threshold,
    )
    logger.info(
        "Aggregated %d columns (%d with typos, %d ambiguous-only, "
        "%d pure clean; %d skipped non-English).",
        stats["columns_emitted"], stats["columns_with_gold_typos"],
        stats["columns_with_only_ambiguous"], stats["columns_pure_clean"],
        stats["columns_skipped_non_english"],
    )

    # Stratified demo selection.
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

    # Deterministic sort for diff stability.
    def _sort_key(it: dict[str, Any]) -> tuple[str, str]:
        return (it["source"], it["column"])

    demo_items.sort(key=_sort_key)
    eval_items.sort(key=_sort_key)

    # Build RFT records and dump.
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_path = output_dir / "demo.jsonl"
    eval_path = output_dir / "eval.jsonl"
    manifest_path = output_dir / "manifest.json"

    def _write(path: Path, items_: list[dict[str, Any]]) -> int:
        n = 0
        with path.open("w", encoding="utf-8") as fh:
            for it in items_:
                rec = _typo_v3_task.build_record(it)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        return n

    n_demo_w = _write(demo_path, demo_items)
    n_eval_w = _write(eval_path, eval_items)

    def _label_counts(items_: list[dict[str, Any]]) -> dict[str, int]:
        c = collections.Counter()
        for it in items_:
            if it["gold_typos"]:
                c["has_typo"] += 1
            elif it["ambiguous_values"]:
                c["ambiguous_only"] += 1
            else:
                c["clean"] += 1
        return dict(c)

    manifest = {
        "v2_eval_path": str(v2_eval_path),
        "v2_demo_path": str(v2_demo_path) if v2_demo_path else None,
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
    parser.add_argument("--v2-eval", type=Path, default=DEFAULT_V2_EVAL)
    parser.add_argument("--v2-demo", type=Path, default=DEFAULT_V2_DEMO)
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demo-size", type=int, default=DEFAULT_DEMO_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-distinct", type=int,
        default=_typo_v3_task.TYPO_V3_MAX_DISTINCT,
    )
    parser.add_argument(
        "--no-english-only", action="store_true",
        help=(
            "Disable the English-only filter. v3 default is to drop "
            "columns whose header or values are not predominantly "
            "ASCII Latin alphabet."
        ),
    )
    parser.add_argument(
        "--english-threshold", type=float,
        default=_typo_v3_task.TYPO_V3_ENGLISH_VALUE_THRESHOLD,
        help=(
            "Minimum fraction of ASCII alphabetic characters across "
            "a column's distinct values for it to be considered "
            "English. Default: %(default).2f."
        ),
    )
    args = parser.parse_args(argv)

    manifest = split_v3(
        v2_eval_path=args.v2_eval,
        v2_demo_path=args.v2_demo,
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
