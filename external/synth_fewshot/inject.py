"""Fixed synthetic-error injection recipe for honest few-shot training.

The few-shot baselines (raha, Baran, TabClean) are normally handed the **real**
benchmark errors as their labeled examples, which lets them *memorize* the exact
(dirty -> clean) value pairs and re-apply them to identical values elsewhere in
the table (we measured 97% of Baran's held-out hits this way). That inflates
precision and measures re-application, not detection.

This module instead builds a **synthetic** training signal: into a fixed set of
"train rows" of each real table, it injects a deterministic set of synthetic
errors at currently-**clean** cells (never the real error/ambiguous cells), using
the repo's ``src/synth`` injectors. A method trained on these synthetic examples
never sees a real error label, so it can be evaluated on **all** the real errors
(full universe, no held-out leakage) and the score reflects genuine
generalization from synthetic demonstrations to real defects.

The recipe is **global and fixed** (same injector rotation, per-row budget, and
RNG seed for every table and every method), so the synthetic-error distribution
is identical across baselines -- the comparability the experiment needs.

Key invariant: the synthetic "clean" oracle is just the **original** dirty table.
We hand a method ``dirty = df_syn`` (real table + synthetic injections) and
``clean = df`` (the original). Diffing them reveals *only* the synthetic
injections, so:
  * synthetic cells  -> labeled ERROR (the training signal),
  * real error cells -> labeled CLEAN (never revealed),
  * everything else  -> CLEAN.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.synth  # noqa: F401  (registers injectors via side-effect)
from src.synth.base import INJECTOR_REGISTRY

# Global, fixed recipe -------------------------------------------------------
SEED = 20240601
DEFAULT_ROWS_FRACTION = 0.5  # inject into ~half the rows (realistic ~5% cell rate)
DEFAULT_MAX_PER_ROW = 1      # one synthetic error per injected row
DEFAULT_BUDGET = 20          # TabClean dev-set size (== few-shot labeling budget)

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

# Injectors grouped by the cell kind they apply to. Order = try priority.
_TEXT_INJECTORS = ("typo", "punctuation", "casing", "pattern")
_NUM_INJECTORS = ("numeric_outlier", "number_format")


def _injector(name):
    return INJECTOR_REGISTRY[name]()


def load_protected_cells(gt_cells_path: Path) -> set:
    """Real error + ambiguous cells (``(i, j)``) that must NOT be perturbed."""
    protected: set = set()
    if not gt_cells_path.exists():
        return protected
    for line in gt_cells_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        protected.add((int(r["i"]), int(r["j"])))
    return protected


def count_real_errors(gt_cells_path: Path) -> int:
    """Number of real ``error`` cells in a table (drives the realistic density:
    we inject the same COUNT of synthetic errors, never their values/positions)."""
    n = 0
    if not gt_cells_path.exists():
        return 0
    for line in gt_cells_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("status") == "error":
            n += 1
    return n


def select_train_rows(df: pd.DataFrame, budget: int = DEFAULT_BUDGET) -> list:
    """The ordered rows used as few-shot training/dev examples.

    Uses ``df.sample(n=budget, random_state=42)`` -- the SAME sampling TabClean's
    ``profile_node`` uses to build its dev set -- and **preserves that order**, so
    the synthetic-error ``dev_row`` numbering matches the ``dev_dirty.csv`` the
    pipeline reindexes (rows ``0..b-1`` in sample order). ``sample`` with a fixed
    seed depends only on the row count, so this is identical for ``df`` and any
    same-shape ``df_syn``. Capped to leave the majority of rows for testing.
    """
    n = len(df)
    b = max(1, min(budget, n // 3)) if n >= 6 else max(1, n // 2)
    idx = df.sample(n=min(b, n), random_state=42).index
    return [int(i) for i in idx]


def _try_inject_cell(df_syn, i, j, protected, rng, cols, injections) -> bool:
    """Try to inject one synthetic error at cell (i, j). Returns True on success."""
    if (i, j) in protected:
        return False
    val = str(df_syn.iat[i, j])
    if not val.strip():
        return False
    is_num = bool(_NUMERIC_RE.match(val.strip()))
    names = list(_NUM_INJECTORS if is_num else _TEXT_INJECTORS)
    rng.shuffle(names)
    for name in names:
        inj = _injector(name)
        if not inj.applicable(val):
            continue
        res = inj.try_inject(val, rng=rng)
        if not res:
            continue
        new_val = str(res[0])
        if new_val == val:
            continue
        df_syn.iat[i, j] = new_val
        injections.append({
            "i": int(i), "j": int(j), "column": cols[j],
            "original_value": val, "injected_value": new_val,
            "error_type": inj.error_type,
        })
        return True
    return False


def build_synthetic_table(
    df: pd.DataFrame,
    protected: set,
    *,
    n_errors: int | None = None,
    rows_fraction: float = DEFAULT_ROWS_FRACTION,
    max_per_row: int = DEFAULT_MAX_PER_ROW,
    seed: int = SEED,
):
    """Inject the fixed synthetic recipe into ``df``'s clean cells.

    Two density modes (both never touch ``protected`` = real error/ambiguous
    cells, so the real defects stay pristine as the test target):

    * **realistic** (``n_errors`` given) -- inject exactly ``n_errors`` synthetic
      errors (one per distinct random row, cycling if needed). Pass each table's
      *real* error count to mirror the benchmark's sparse density (~4-5 per
      labeled table, 0 on clean tables). Only the count is borrowed, never the
      error values/positions, so there is no value-leak.
    * **dense** (``n_errors=None``) -- inject into a deterministic
      ``rows_fraction`` of rows (default half), one error each (~5% of cells).
      Generous signal: even so the few-shot learners fail.

    Returns ``(df_syn, injections)`` where ``injections`` is a list of
    ``{i, j, column, original_value, injected_value, error_type}``.
    """
    rng = random.Random(seed)
    df_syn = df.copy()
    cols = list(df.columns)
    n_cols = len(cols)
    n = len(df)
    injections: list = []

    if n_errors is not None:
        # Realistic: place exactly n_errors errors, one per distinct random row.
        target = max(0, min(int(n_errors), n * n_cols))
        order = list(range(n))
        rng.shuffle(order)
        placed = 0
        passes = 0
        while placed < target and passes < max_per_row + 1:
            for i in order:
                if placed >= target:
                    break
                cand = list(range(n_cols))
                rng.shuffle(cand)
                for j in cand:
                    if _try_inject_cell(df_syn, i, j, protected | {
                        (inj["i"], inj["j"]) for inj in injections
                    }, rng, cols, injections):
                        placed += 1
                        break
            passes += 1
        return df_syn, injections

    # Dense: a deterministic ~rows_fraction of rows, one error each.
    order = list(range(n))
    rng.shuffle(order)
    inject_rows = set(order[: max(1, round(rows_fraction * n))])
    for i in range(n):
        if i not in inject_rows:
            continue
        col_order = list(range(n_cols))
        rng.shuffle(col_order)
        row_count = 0
        for j in col_order:
            if row_count >= max_per_row:
                break
            if _try_inject_cell(df_syn, i, j, protected, rng, cols, injections):
                row_count += 1
    return df_syn, injections


def dev_gt_errors_text(injections: list, dev_rows: list) -> str:
    """Render the injections that fall in ``dev_rows`` as TabClean's
    ``dev_row | column | dirty | correct`` text.

    ``dev_row`` is the position of the real row index within ``dev_rows`` -- the
    same order TabClean's ``profile_node`` reindexes its ``dev_dirty.csv`` to (it
    samples the same rows in the same order via ``random_state=42``)."""
    pos = {r: k for k, r in enumerate(dev_rows)}
    lines = ["# dev_row | column | dirty_value | correct_value"]
    for inj in injections:
        if inj["i"] in pos:
            lines.append(
                f"{pos[inj['i']]} | {inj['column']} | {inj['injected_value']} | {inj['original_value']}"
            )
    return "\n".join(lines)
