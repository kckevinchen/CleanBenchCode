"""Generate small, reproducible perturbations of a CSV for repeat-run
experiments.

The downstream Good-Turing / Chao1 false-negative estimator
(:mod:`src.good_turing`) needs multiple runs of the same detection
pipeline under *slightly different* inputs, so that "found by k of K
runs" counts form a meaningful frequency distribution. The standard
trick is to present the data in a shuffled order each time - the LLM
sees the same facts but in a different arrangement, so any defect
whose detectability depends on ordering (e.g. "this row looks odd
compared to its neighbours") will flip in and out of the result set.

Perturbation knobs
------------------

* ``shuffle_rows``    - permute the row order (default on).
* ``shuffle_columns`` - permute the column order; only observable by the
  model in ``column_mode="all"`` since single-column calls always see
  one header at a time (default off).

The original CSV row index is PRESERVED in the returned DataFrames so
that :meth:`src.data_loader.CsvChunk.to_markdown` emits the correct
``__row__`` labels. That way every run's detections reference the
same row-id space, which the species canonicaliser (in
:mod:`src.good_turing`) relies on to match findings across runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Mapping

import pandas as pd


@dataclass
class PerturbedView:
    """One perturbed view of a source DataFrame.

    ``frame`` keeps the ORIGINAL row index (so row numbers survive
    round-tripping through the LLM prompt) and has its columns
    re-ordered according to ``column_permutation``.
    """

    run_id: int
    base_seed: int
    effective_seed: int
    source: str
    frame: pd.DataFrame
    row_permutation: List[int] = field(default_factory=list)
    column_permutation: List[str] = field(default_factory=list)
    shuffled_rows: bool = False
    shuffled_columns: bool = False

    @property
    def description(self) -> str:
        bits = []
        if self.shuffled_rows:
            bits.append("rows")
        if self.shuffled_columns:
            bits.append("cols")
        if not bits:
            return "identity"
        return "shuffle(" + "+".join(bits) + f", seed={self.effective_seed})"


def _effective_seed(base_seed: int, run_id: int) -> int:
    """Derive a deterministic per-run seed from ``(base_seed, run_id)``.

    Keeping this explicit (vs. Python's ``random.Random(...).random()``
    chaining) makes the resulting per-run seeds easy to recompute for
    auditing or rerunning a single run in isolation.
    """
    # Cheap mixer that's deterministic across Python versions.
    return (base_seed * 1_000_003 + run_id * 97 + 31) & 0xFFFFFFFF


def perturb_frame(
    frame: pd.DataFrame,
    *,
    run_id: int,
    base_seed: int,
    source: str,
    shuffle_rows: bool = True,
    shuffle_columns: bool = False,
) -> PerturbedView:
    """Return a single :class:`PerturbedView` of ``frame``.

    The frame's original index is preserved (so each row carries its
    CSV row number through the shuffle). ``run_id == 0`` by convention
    returns the IDENTITY view (no shuffling) so that Run 0 matches what
    a plain single-pass experiment would see - handy for debugging
    without having to re-run.
    """
    seed = _effective_seed(base_seed, run_id)
    rng = random.Random(seed)

    if run_id == 0:
        # Deterministic identity baseline.
        return PerturbedView(
            run_id=run_id,
            base_seed=base_seed,
            effective_seed=seed,
            source=source,
            frame=frame.copy(),
            row_permutation=list(frame.index),
            column_permutation=list(frame.columns),
            shuffled_rows=False,
            shuffled_columns=False,
        )

    row_order = list(frame.index)
    col_order = list(frame.columns)

    if shuffle_rows and len(row_order) > 1:
        rng.shuffle(row_order)
    if shuffle_columns and len(col_order) > 1:
        rng.shuffle(col_order)

    # ``loc`` keeps the permuted index in place (no reset) so
    # downstream chunking can recover original CSV row numbers via
    # ``frame.index``.
    permuted = frame.loc[row_order, col_order].copy()

    return PerturbedView(
        run_id=run_id,
        base_seed=base_seed,
        effective_seed=seed,
        source=source,
        frame=permuted,
        row_permutation=row_order,
        column_permutation=col_order,
        shuffled_rows=bool(shuffle_rows and len(row_order) > 1),
        shuffled_columns=bool(shuffle_columns and len(col_order) > 1),
    )


def perturb_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    run_id: int,
    base_seed: int,
    shuffle_rows: bool = True,
    shuffle_columns: bool = False,
) -> List[PerturbedView]:
    """Perturb every DataFrame in ``frames`` for a single run.

    The per-file effective seed is :func:`_effective_seed` mixed with
    the file name, so different files get different row orders on the
    same run (otherwise run 1 would shuffle every file with the same
    permutation indices - unhelpful for variance).
    """
    out: List[PerturbedView] = []
    for source, frame in frames.items():
        # Fold the source name into the seed so inter-file shuffles are
        # independent. Use a stable hash (hash() is salted per-process
        # in CPython; we want determinism).
        file_offset = 0
        for ch in source:
            file_offset = (file_offset * 131 + ord(ch)) & 0xFFFFFFFF
        per_file_seed = (base_seed ^ file_offset) & 0xFFFFFFFF
        out.append(
            perturb_frame(
                frame,
                run_id=run_id,
                base_seed=per_file_seed,
                source=source,
                shuffle_rows=shuffle_rows,
                shuffle_columns=shuffle_columns,
            )
        )
    return out


def generate_runs(
    frames: Mapping[str, pd.DataFrame],
    *,
    num_runs: int,
    base_seed: int,
    shuffle_rows: bool = True,
    shuffle_columns: bool = False,
    include_identity_run: bool = True,
) -> List[List[PerturbedView]]:
    """Generate ``num_runs`` perturbed views per source CSV.

    Returns a list of length ``num_runs`` where each entry is the list
    of :class:`PerturbedView` objects (one per input frame) for that
    run.

    If ``include_identity_run`` is True (default), run 0 uses the
    original unshuffled frames; runs 1..K-1 are perturbed. This is the
    usual setup for the Good-Turing experiment: the baseline result
    you'd get without perturbation is kept as one of the K samples.
    """
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    runs: List[List[PerturbedView]] = []
    for i in range(num_runs):
        # Run 0 is identity only when include_identity_run is True;
        # otherwise every run is perturbed (useful if the baseline was
        # already captured by a prior experiment).
        rid = i if include_identity_run else i + 1
        views = perturb_frames(
            frames,
            run_id=rid,
            base_seed=base_seed,
            shuffle_rows=shuffle_rows,
            shuffle_columns=shuffle_columns,
        )
        runs.append(views)
    return runs


__all__ = [
    "PerturbedView",
    "generate_runs",
    "perturb_frame",
    "perturb_frames",
]
