"""Deterministic stratified split of the typo benchmark into demo + eval.

The typo RFT pipeline needs two non-overlapping slices of the benchmark:

* ``demo`` -- a small (~30 row) hand-pickable subset that becomes the
  source of few-shot exemplars for synthesis and the seed of the
  calibration set for the grader. We hold it *back* from eval so the
  RFT loop never sees it indirectly.
* ``eval`` -- everything else; the held-out benchmark we score the
  trained model against in :mod:`experiments.run_precision_baseline`
  and used to calibrate the typo grader (``demo_rft calibrate-grader``).

Why a separate script (instead of inline in synthesis)?

* Reproducibility -- the manifest pins a content hash so any downstream
  consumer can verify it's reading the *same* split the last run wrote.
* Auditability -- demo selection drives which exemplars steer synthesis,
  which steers the model. Having that selection as a stand-alone
  artifact makes it inspectable.

Usage
-----
::

    python -m experiments.split_typo_benchmark \\
        --benchmark benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl \\
        --output-dir outputs/typo_rft/benchmark_split

Stratification
--------------
We stratify by the ``label`` field so the demo and eval sets keep the
same {error, ambiguous, clean} ratio as the full benchmark. Without
stratification a 30-row demo at the benchmark's natural distribution
(12% error / 23% amb / 65% clean) could come out with only 1-2 error
examples, which is too thin to use as few-shot.
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("split_typo_benchmark")


DEFAULT_BENCHMARK = Path(
    "benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/typo_rft/benchmark_split")
DEFAULT_DEMO_SIZE = 30
DEFAULT_SEED = 0

# Strata to preserve. Anything else (NULL/missing/typo'd values in the
# label column) falls into a single "other" bucket so we don't silently
# drop rows.
_KNOWN_LABELS = ("error", "ambiguous", "clean")


def _record_key(rec: dict[str, Any]) -> str:
    """Stable identity for a benchmark row, robust to file reordering.

    We deliberately avoid using just (source, row, column) because the
    benchmark sometimes carries the same logical cell as multiple rows
    (one per detector hit). Hashing the full payload + index gives us a
    de-dupe-friendly key.
    """
    payload = {
        "source": rec.get("source"),
        "row": rec.get("row"),
        "column": rec.get("column"),
        "original_value": rec.get("original_value"),
        "label": rec.get("label"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def _content_hash(rows: list[dict[str, Any]]) -> str:
    """Order-independent hash of a row set, used in the manifest."""
    keys = sorted(_record_key(r) for r in rows)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _stratify(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        label = r.get("label")
        bucket = label if label in _KNOWN_LABELS else "other"
        buckets[bucket].append(r)
    return buckets


def _proportional_demo_sizes(
    buckets: dict[str, list[dict[str, Any]]], demo_size: int
) -> dict[str, int]:
    """Allocate demo_size across strata proportionally with floor=1 per bucket.

    Guarantees every non-empty stratum contributes at least one demo row
    (so the few-shot pool isn't accidentally single-class), then
    distributes the remainder by largest-fractional-part.
    """
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return {}

    # Compute proportional ideal sizes.
    ideal = {k: demo_size * len(v) / total for k, v in buckets.items() if v}
    # Floor = 1 for every non-empty bucket.
    sizes = {k: max(1, int(v)) for k, v in ideal.items()}
    # Cap each bucket at its actual size (small strata).
    sizes = {k: min(sizes[k], len(buckets[k])) for k in sizes}

    # Distribute / reclaim remainder.
    diff = demo_size - sum(sizes.values())
    if diff > 0:
        # Need to add: prefer buckets with largest fractional remainder
        # AND remaining headroom.
        remainders = sorted(
            ((ideal[k] - sizes[k], k) for k in sizes if sizes[k] < len(buckets[k])),
            reverse=True,
        )
        for _, k in remainders:
            if diff == 0:
                break
            sizes[k] += 1
            diff -= 1
    elif diff < 0:
        # Over-allocated (floor=1 in many tiny buckets); trim from
        # largest buckets first.
        for k in sorted(sizes, key=lambda k: -sizes[k]):
            if diff == 0:
                break
            if sizes[k] > 1:
                sizes[k] -= 1
                diff += 1

    return sizes


def split_benchmark(
    benchmark_path: Path,
    output_dir: Path,
    *,
    demo_size: int = DEFAULT_DEMO_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Read benchmark JSONL, write {demo, eval, manifest} under output_dir."""
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    rows: list[dict[str, Any]] = []
    with benchmark_path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {ln} of {benchmark_path}: {exc}"
                ) from exc

    if not rows:
        raise ValueError(f"Benchmark file {benchmark_path} is empty.")

    buckets = _stratify(rows)
    sizes = _proportional_demo_sizes(buckets, demo_size)

    rng = random.Random(seed)
    demo_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for label, bucket in buckets.items():
        n = sizes.get(label, 0)
        # Deterministic shuffle, then take first n.
        shuffled = bucket[:]
        rng.shuffle(shuffled)
        demo_rows.extend(shuffled[:n])
        eval_rows.extend(shuffled[n:])

    # Sort outputs deterministically (by content hash) so the file is
    # diff-stable across runs with the same seed.
    demo_rows.sort(key=_record_key)
    eval_rows.sort(key=_record_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    demo_path = output_dir / "demo.jsonl"
    eval_path = output_dir / "eval.jsonl"
    manifest_path = output_dir / "manifest.json"

    for path, recs in [(demo_path, demo_rows), (eval_path, eval_rows)]:
        with path.open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "benchmark_path": str(benchmark_path),
        "benchmark_total": len(rows),
        "seed": seed,
        "demo_size_requested": demo_size,
        "demo_size_actual": len(demo_rows),
        "eval_size_actual": len(eval_rows),
        "benchmark_label_counts": {
            k: len(v) for k, v in sorted(buckets.items())
        },
        "demo_label_counts": dict(
            collections.Counter(r.get("label") for r in demo_rows)
        ),
        "eval_label_counts": dict(
            collections.Counter(r.get("label") for r in eval_rows)
        ),
        "demo_content_hash": _content_hash(demo_rows),
        "eval_content_hash": _content_hash(eval_rows),
        # Source CSVs that appear in either split -- the synth script
        # uses this as a deny-list to avoid training on a column whose
        # values are already in the evaluation set.
        "benchmark_sources": sorted(
            {r.get("source") for r in rows if r.get("source")}
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "Wrote %d demo + %d eval rows to %s",
        len(demo_rows), len(eval_rows), output_dir,
    )
    logger.info("Demo label counts: %s", manifest["demo_label_counts"])
    logger.info("Eval label counts: %s", manifest["eval_label_counts"])

    return manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demo-size", type=int, default=DEFAULT_DEMO_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    manifest = split_benchmark(
        args.benchmark, args.output_dir,
        demo_size=args.demo_size, seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
