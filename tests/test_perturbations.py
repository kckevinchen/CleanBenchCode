"""Tests for ``src.adaptive.perturbations``: deterministic CSV reshuffles.

Pinned by Pass 6c HIGH coverage gap. The Good-Turing estimator depends
on these views being:

* deterministic for a given (base_seed, run_id, source) tuple
* identity on run_id=0
* preserving the original DataFrame index (so LLM output references
  the canonical CSV row numbers, not the shuffled positions)
* per-source-different on the same run_id (otherwise every CSV would
  shuffle to the same permutation, which defeats the purpose)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.adaptive.perturbations import (
    PerturbedView,
    _effective_seed,
    generate_runs,
    perturb_frame,
    perturb_frames,
)


def _frame(rows: int = 10, cols: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        f"c{i}": [f"r{r}c{i}" for r in range(rows)]
        for i in range(cols)
    })


# ---------------------------------------------------------------------------
# _effective_seed
# ---------------------------------------------------------------------------


class TestEffectiveSeed:
    def test_deterministic(self) -> None:
        assert _effective_seed(42, 1) == _effective_seed(42, 1)

    def test_run_id_changes_seed(self) -> None:
        assert _effective_seed(42, 1) != _effective_seed(42, 2)

    def test_base_seed_changes_seed(self) -> None:
        assert _effective_seed(42, 1) != _effective_seed(43, 1)


# ---------------------------------------------------------------------------
# perturb_frame
# ---------------------------------------------------------------------------


class TestPerturbFrame:
    def test_run_id_zero_is_identity(self) -> None:
        df = _frame(rows=8, cols=3)
        view = perturb_frame(df, run_id=0, base_seed=42, source="a.csv")
        assert view.shuffled_rows is False
        assert view.shuffled_columns is False
        assert list(view.frame.index) == list(df.index)
        assert list(view.frame.columns) == list(df.columns)
        # Frame is a copy, not the same object.
        assert view.frame is not df

    def test_run_id_zero_identity_even_with_shuffle_flags(self) -> None:
        # The identity-baseline rule overrides shuffle_rows / shuffle_columns.
        df = _frame(rows=8, cols=3)
        view = perturb_frame(
            df, run_id=0, base_seed=42, source="a.csv",
            shuffle_rows=True, shuffle_columns=True,
        )
        assert view.shuffled_rows is False
        assert view.shuffled_columns is False
        assert list(view.frame.index) == list(df.index)

    def test_perturbation_preserves_original_index(self) -> None:
        df = _frame(rows=8, cols=3)
        view = perturb_frame(df, run_id=1, base_seed=42, source="a.csv")
        # Row order is shuffled, but each value carries its ORIGINAL
        # DataFrame index — that's what to_markdown's __row__ relies on.
        assert sorted(view.frame.index) == list(range(8))
        # ``loc`` preserves index, doesn't reset.
        for orig_idx, row in view.frame.iterrows():
            assert row["c0"] == f"r{orig_idx}c0"

    def test_deterministic_across_runs(self) -> None:
        df = _frame(rows=8, cols=3)
        v1 = perturb_frame(df, run_id=3, base_seed=42, source="a.csv")
        v2 = perturb_frame(df, run_id=3, base_seed=42, source="a.csv")
        assert v1.row_permutation == v2.row_permutation
        assert v1.column_permutation == v2.column_permutation
        assert list(v1.frame.index) == list(v2.frame.index)

    def test_different_run_ids_give_different_permutations(self) -> None:
        df = _frame(rows=10, cols=3)
        v1 = perturb_frame(df, run_id=1, base_seed=42, source="a.csv")
        v2 = perturb_frame(df, run_id=2, base_seed=42, source="a.csv")
        assert v1.row_permutation != v2.row_permutation

    def test_shuffle_columns_disabled_by_default(self) -> None:
        df = _frame(rows=8, cols=4)
        view = perturb_frame(df, run_id=1, base_seed=42, source="a.csv")
        assert view.shuffled_columns is False
        assert list(view.frame.columns) == list(df.columns)

    def test_shuffle_columns_when_requested(self) -> None:
        df = _frame(rows=8, cols=4)
        view = perturb_frame(
            df, run_id=1, base_seed=42, source="a.csv",
            shuffle_columns=True,
        )
        assert view.shuffled_columns is True
        # Same set, different order.
        assert sorted(view.frame.columns) == sorted(df.columns)

    def test_singleton_row_or_column_skips_shuffle(self) -> None:
        # Single-row frame: row order can't actually permute.
        df = pd.DataFrame({"a": [1], "b": [2]})
        view = perturb_frame(df, run_id=1, base_seed=42, source="x.csv")
        assert view.shuffled_rows is False
        # Single-column frame with shuffle_columns=True: skipped too.
        df = pd.DataFrame({"only": [1, 2, 3]})
        view = perturb_frame(
            df, run_id=1, base_seed=42, source="x.csv",
            shuffle_columns=True,
        )
        assert view.shuffled_columns is False

    def test_description_string(self) -> None:
        df = _frame(rows=8, cols=3)
        v_id = perturb_frame(df, run_id=0, base_seed=42, source="a.csv")
        assert v_id.description == "identity"
        v_rows = perturb_frame(df, run_id=1, base_seed=42, source="a.csv")
        assert "rows" in v_rows.description
        assert "seed=" in v_rows.description


# ---------------------------------------------------------------------------
# perturb_frames — multi-source determinism + per-file independence
# ---------------------------------------------------------------------------


class TestPerturbFrames:
    def test_per_file_independence_on_same_run(self) -> None:
        # Both files share the same base_seed AND run_id, but their
        # row permutations MUST differ (otherwise inter-file variance
        # collapses, defeating the perturbation purpose).
        a = _frame(rows=12, cols=2)
        b = _frame(rows=12, cols=2)
        views = perturb_frames(
            {"a.csv": a, "b.csv": b},
            run_id=1, base_seed=42,
        )
        assert len(views) == 2
        assert views[0].source == "a.csv"
        assert views[1].source == "b.csv"
        # Different file names → different folded seeds → different
        # row permutations.
        assert views[0].row_permutation != views[1].row_permutation

    def test_deterministic_across_calls(self) -> None:
        a = _frame(rows=12, cols=2)
        out1 = perturb_frames({"a.csv": a}, run_id=2, base_seed=42)
        out2 = perturb_frames({"a.csv": a}, run_id=2, base_seed=42)
        assert out1[0].row_permutation == out2[0].row_permutation


# ---------------------------------------------------------------------------
# generate_runs — top-level entry point
# ---------------------------------------------------------------------------


class TestGenerateRuns:
    def test_run_zero_is_identity_when_included(self) -> None:
        a = _frame(rows=8, cols=2)
        runs = generate_runs(
            {"a.csv": a}, num_runs=3, base_seed=42,
            include_identity_run=True,
        )
        assert len(runs) == 3
        # Run 0 is identity (perturbed at run_id=0).
        assert runs[0][0].shuffled_rows is False
        assert list(runs[0][0].frame.index) == list(a.index)
        # Runs 1 and 2 are perturbed.
        assert runs[1][0].shuffled_rows is True
        assert runs[2][0].shuffled_rows is True

    def test_skip_identity_run(self) -> None:
        a = _frame(rows=8, cols=2)
        runs = generate_runs(
            {"a.csv": a}, num_runs=2, base_seed=42,
            include_identity_run=False,
        )
        # All runs are perturbed (run_id starts at 1 internally).
        for views in runs:
            assert views[0].shuffled_rows is True

    def test_zero_runs_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_runs"):
            generate_runs({"a.csv": _frame()}, num_runs=0, base_seed=42)
