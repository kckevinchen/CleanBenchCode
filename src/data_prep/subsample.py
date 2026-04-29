"""Produce a reproducible subsample of a CSV for manual review.

Why: running the per-type detector on the full 1195-row CSV produces far too
many issues to hand-verify. A deterministic subsample (random with a fixed
seed, then resorted into original order) gives a tractable slice while still
exercising every error type.

Usage::

    python -m src.subsample \
        --input data/0a0a2ebffde940e2a232f26ba3ceb327.csv \
        --output data/sample_100.csv \
        --rows 100 \
        --seed 42

The output CSV:

* has its index reset (0..N-1), so row numbers emitted by the detector map
  directly to line (row + 2) of the output file (1 for the header, 1 for
  0-indexing);
* preserves the original ``Serial`` column (when present) so every sampled
  row remains traceable back to the full dataset;
* is re-sorted by original row index so it reads top-to-bottom in source
  order, which is what manual verification expects.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def subsample_csv(
    input_path: Path,
    output_path: Path,
    rows: int,
    seed: int = 42,
    strategy: str = "random",
) -> pd.DataFrame:
    """Write a reproducible subsample and return the sampled DataFrame."""
    if rows <= 0:
        raise ValueError(f"--rows must be positive, got {rows}")

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    total = len(df)
    logger.info("Loaded %d rows x %d columns from %s", total, len(df.columns), input_path)

    if rows >= total:
        logger.warning(
            "--rows=%d >= total rows (%d); copying the full CSV as the sample",
            rows,
            total,
        )
        sample = df.copy()
    elif strategy == "random":
        sample = df.sample(n=rows, random_state=seed).sort_index()
    elif strategy == "head":
        sample = df.head(rows).copy()
    else:
        raise ValueError(f"Unknown strategy: {strategy!r} (use 'random' or 'head')")

    sample = sample.reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)

    logger.info(
        "Wrote %d sampled rows (strategy=%s, seed=%s) to %s",
        len(sample),
        strategy,
        seed if strategy == "random" else "n/a",
        output_path,
    )
    return sample


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subsample a CSV file.")
    parser.add_argument("--input", type=Path, required=True, help="Source CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV.")
    parser.add_argument("--rows", type=int, default=100, help="Sample size (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--strategy",
        choices=("random", "head"),
        default="random",
        help=(
            "Sampling strategy. 'random' (default) uses a seeded random "
            "sample; 'head' takes the first N rows."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    subsample_csv(
        input_path=args.input,
        output_path=args.output,
        rows=args.rows,
        seed=args.seed,
        strategy=args.strategy,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
