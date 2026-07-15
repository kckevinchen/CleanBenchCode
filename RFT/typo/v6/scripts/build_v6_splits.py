"""Build v6 train/valid/eval splits directly from the human-labelled benchmark.

Why v6
------
v5 train/valid were LLM-synthesised typo injections (see
``RFT/typo/v5/scripts/synth_typo_llm_v5.py``). Inspection (May 2026)
showed those synthetic labels include many frequency-anchored and
proper-noun calls that violate the v4 per-cell-judgement rulebook
(e.g. ``AAY2020 -> AY2020``, ``DD-21073 -> D-21073``, ``Bhmao -> Bhamo``).
The fine-tuned models learned to fire on frequency anchors, hurting
recall on the human-labelled eval set.

v6 throws out the synthetic data and rebuilds all three splits from the
benchmark itself:

* Partition by SOURCE CSV (no column from one CSV appears in two splits).
* train / valid / eval = 60% / 10% / 30% of source CSVs.
* TRAIN balance: keep all has-typo columns from train-assigned CSVs, then
  undersample non-typo columns (clean + ambiguous-only) to a target
  has_typo : no_typo ratio (default 1:1).
* VALID / EVAL: keep all columns from their assigned CSVs (natural ratio).

Output (under ``RFT/typo/v6/data/``):

* ``train.jsonl``    -- balanced training set
* ``valid.jsonl``    -- held-out validation
* ``eval.jsonl``     -- held-out evaluation
* ``demo.jsonl``     -- small stratified sample drawn from eval (for prompts)
* ``manifest.json``  -- provenance + counts + source-CSV assignments

Usage
-----
::

    python -m RFT.typo.v6.scripts.build_v6_splits \\
        --benchmark benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl \\
        --csv-root  data/benchmark_sources_v6 \\
        --output-dir RFT/typo/v6/data \\
        --train-frac 0.60 --valid-frac 0.10 --eval-frac 0.30 \\
        --train-balance-ratio 1.0 --demo-size 10
"""

from __future__ import annotations

import argparse
import collections
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

# Reuse v5's aggregation primitives + record builder + filters.
from RFT.typo.v5 import _task as _typo_v5_task  # noqa: E402
from RFT.typo.v5.scripts.split_typo_benchmark_v5 import (  # noqa: E402
    _aggregate,
    _build_csv_index,
    _content_hash,
)

logger = logging.getLogger("build_v6_splits")

DEFAULT_BENCHMARK = Path(
    "benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl"
)
DEFAULT_CSV_ROOT = Path("data/benchmark_sources_v6")
DEFAULT_OUTPUT_DIR = Path("RFT/typo/v6/data")
DEFAULT_SEED = 0
DEFAULT_TRAIN_FRAC = 0.60
DEFAULT_VALID_FRAC = 0.10
DEFAULT_EVAL_FRAC = 0.30
DEFAULT_BALANCE_RATIO = 1.0  # has_typo : no_typo target in train
DEFAULT_DEMO_SIZE = 10


def _bucket(it: dict[str, Any]) -> str:
    if it["gold_typos"]:
        return "has_typo"
    if it["ambiguous_values"]:
        return "ambiguous_only"
    return "clean"


def _partition_sources(
    items: list[dict[str, Any]],
    *,
    train_frac: float,
    valid_frac: float,
    eval_frac: float,
    seed: int,
) -> dict[str, set[str]]:
    """Assign each source CSV to exactly one of train/valid/eval.

    Strategy: stratify CSVs by whether they contain any has-typo column,
    then shuffle within each stratum and slice by fraction. This keeps
    has-typo CSVs roughly proportional across splits.
    """
    by_source: dict[str, list[str]] = collections.defaultdict(list)
    for it in items:
        by_source[it["source"]].append(_bucket(it))

    has_typo_sources, no_typo_sources = [], []
    for src, kinds in by_source.items():
        if any(k == "has_typo" for k in kinds):
            has_typo_sources.append(src)
        else:
            no_typo_sources.append(src)

    rng = random.Random(seed)
    rng.shuffle(has_typo_sources)
    rng.shuffle(no_typo_sources)

    def _slice(srcs: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(srcs)
        n_train = int(round(n * train_frac))
        n_valid = int(round(n * valid_frac))
        # Whatever's left goes to eval (handles rounding drift).
        return (
            srcs[:n_train],
            srcs[n_train : n_train + n_valid],
            srcs[n_train + n_valid :],
        )

    t1, v1, e1 = _slice(has_typo_sources)
    t2, v2, e2 = _slice(no_typo_sources)

    return {
        "train": set(t1) | set(t2),
        "valid": set(v1) | set(v2),
        "eval": set(e1) | set(e2),
    }


def _balance_train(
    train_items: list[dict[str, Any]],
    *,
    balance_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Undersample non-typo columns to achieve target has_typo : no_typo ratio.

    Always keeps ALL has-typo columns. Randomly subsamples non-typo
    columns (clean + ambiguous_only combined) to ``len(has_typo) /
    balance_ratio``.
    """
    has_typo = [it for it in train_items if _bucket(it) == "has_typo"]
    no_typo = [it for it in train_items if _bucket(it) != "has_typo"]
    target_no_typo = int(round(len(has_typo) / balance_ratio))
    if target_no_typo >= len(no_typo):
        return train_items  # nothing to drop
    rng = random.Random(seed)
    sampled_no_typo = rng.sample(no_typo, target_no_typo)
    return has_typo + sampled_no_typo


def _label_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for it in items:
        c[_bucket(it)] += 1
    return dict(c)


def _stratified_demo(
    eval_items: list[dict[str, Any]],
    demo_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Pick a small stratified sample from eval for prompt few-shot use."""
    if demo_size <= 0 or not eval_items:
        return []
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for it in eval_items:
        buckets[_bucket(it)].append(it)
    total = sum(len(b) for b in buckets.values())
    sizes = {
        k: max(1, len(v) * demo_size // total)
        for k, v in buckets.items() if v
    }
    # Trim to exactly demo_size.
    assigned = sum(sizes.values())
    while assigned > demo_size:
        # Trim from largest bucket first.
        k = max(sizes, key=lambda k_: sizes[k_])
        sizes[k] -= 1
        assigned -= 1
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for k, n in sizes.items():
        pool = buckets[k][:]
        rng.shuffle(pool)
        out.extend(pool[:n])
    return out


def build(
    *,
    benchmark_path: Path,
    csv_root: Path,
    output_dir: Path,
    train_frac: float,
    valid_frac: float,
    eval_frac: float,
    train_balance_ratio: float,
    demo_size: int,
    seed: int,
    max_distinct: int,
    english_only: bool,
    english_threshold: float,
    pin_assignments: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if abs((train_frac + valid_frac + eval_frac) - 1.0) > 1e-6:
        raise ValueError(
            f"fractions must sum to 1.0 (got {train_frac + valid_frac + eval_frac})"
        )
    if not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)
    if not csv_root.exists():
        raise FileNotFoundError(csv_root)

    csv_index = _build_csv_index(csv_root)
    if not csv_index:
        raise RuntimeError(f"No CSVs found under {csv_root}")

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
        "Aggregated %d columns (%d has_typo, %d ambiguous_only, %d clean)",
        stats["columns_emitted"],
        stats["columns_with_gold_typos"],
        stats["columns_with_only_ambiguous"],
        stats["columns_pure_clean"],
    )

    if pin_assignments is not None:
        # Reuse a frozen source->split partition (e.g. to keep the eval set
        # stable across benchmark relabels) instead of re-deriving it.
        src_assignments = {
            sp: set(pin_assignments.get(sp, []))
            for sp in ("train", "valid", "eval")
        }
        assigned = set().union(*src_assignments.values())
        item_sources = {it["source"] for it in items}
        missing = item_sources - assigned
        if missing:
            raise ValueError(
                f"{len(missing)} aggregated source(s) absent from pinned "
                f"assignment (cannot place them): {sorted(missing)[:10]}"
            )
        logger.info("Using PINNED source assignment (frozen split).")
    else:
        src_assignments = _partition_sources(
            items,
            train_frac=train_frac,
            valid_frac=valid_frac,
            eval_frac=eval_frac,
            seed=seed,
        )

    split_items: dict[str, list[dict[str, Any]]] = {
        "train": [], "valid": [], "eval": [],
    }
    for it in items:
        for split, srcs in src_assignments.items():
            if it["source"] in srcs:
                split_items[split].append(it)
                break

    pre_balance_counts = {
        sp: _label_counts(its) for sp, its in split_items.items()
    }

    # Balance train (undersample non-typo columns).
    split_items["train"] = _balance_train(
        split_items["train"],
        balance_ratio=train_balance_ratio,
        seed=seed,
    )

    # Stratified demo from eval.
    demo_items = _stratified_demo(split_items["eval"], demo_size, seed)

    # Sort within each split for deterministic on-disk ordering.
    def _sort_key(it: dict[str, Any]) -> tuple[str, str]:
        return (it["source"], it["column"])

    for sp in split_items:
        split_items[sp].sort(key=_sort_key)
    demo_items.sort(key=_sort_key)

    # Write.
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, items_: list[dict[str, Any]]) -> int:
        n = 0
        with path.open("w", encoding="utf-8") as fh:
            for it in items_:
                rec = _typo_v5_task.build_record(it)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        return n

    counts_written: dict[str, int] = {}
    for sp, its in split_items.items():
        counts_written[sp] = _write(output_dir / f"{sp}.jsonl", its)
    counts_written["demo"] = _write(output_dir / "demo.jsonl", demo_items)

    manifest = {
        "benchmark_path": str(benchmark_path),
        "csv_root": str(csv_root),
        "csv_index_size": len(csv_index),
        "max_distinct": max_distinct,
        "seed": seed,
        "english_only": english_only,
        "english_threshold": english_threshold,
        "train_frac": train_frac,
        "valid_frac": valid_frac,
        "eval_frac": eval_frac,
        "train_balance_ratio": train_balance_ratio,
        "demo_size_requested": demo_size,
        "stats_aggregate": stats,
        "counts_written": counts_written,
        "label_counts_pre_balance": pre_balance_counts,
        "label_counts_post_balance": {
            sp: _label_counts(its) for sp, its in split_items.items()
        },
        "demo_label_counts": _label_counts(demo_items),
        "source_assignments": {
            sp: sorted(srcs) for sp, srcs in src_assignments.items()
        },
        "split_content_hashes": {
            sp: _content_hash(its) for sp, its in split_items.items()
        },
        "demo_content_hash": _content_hash(demo_items),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    logger.info("Wrote splits: %s", counts_written)
    for sp, its in split_items.items():
        logger.info(
            "  %s post-balance label counts: %s",
            sp, _label_counts(its),
        )
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
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--valid-frac", type=float, default=DEFAULT_VALID_FRAC)
    parser.add_argument("--eval-frac", type=float, default=DEFAULT_EVAL_FRAC)
    parser.add_argument(
        "--train-balance-ratio", type=float, default=DEFAULT_BALANCE_RATIO,
        help=(
            "Target has_typo : no_typo column ratio in train. 1.0 = 50/50. "
            "Non-typo columns are undersampled to hit this ratio; has-typo "
            "columns are always kept."
        ),
    )
    parser.add_argument("--demo-size", type=int, default=DEFAULT_DEMO_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-distinct", type=int,
        default=_typo_v5_task.TYPO_V5_MAX_DISTINCT,
    )
    parser.add_argument("--no-english-only", action="store_true")
    parser.add_argument(
        "--english-threshold", type=float,
        default=_typo_v5_task.TYPO_V5_ENGLISH_VALUE_THRESHOLD,
    )
    parser.add_argument(
        "--pin-assignments", type=Path, default=None,
        help=(
            "Path to a manifest.json whose 'source_assignments' is reused "
            "verbatim as the train/valid/eval source partition (freezes the "
            "split across benchmark relabels instead of re-deriving it)."
        ),
    )
    args = parser.parse_args(argv)

    pin = None
    if args.pin_assignments:
        pin = json.loads(args.pin_assignments.read_text())["source_assignments"]

    build(
        benchmark_path=args.benchmark,
        csv_root=args.csv_root,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        eval_frac=args.eval_frac,
        train_balance_ratio=args.train_balance_ratio,
        demo_size=args.demo_size,
        seed=args.seed,
        max_distinct=args.max_distinct,
        english_only=not args.no_english_only,
        english_threshold=args.english_threshold,
        pin_assignments=pin,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
