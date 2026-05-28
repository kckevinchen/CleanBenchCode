"""Tests for ``experiments.run_column_clustering`` helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_column_clustering import (  # noqa: E402
    _append_per_arm_stats,
    _is_candidate,
    _value_counts,
    enumerate_candidates,
    enumerate_candidates_from_frames,
    run_for_candidates,
)
from src.adaptive.scoped_gt import ArmConfig  # noqa: E402
from src.core.value_cluster.adaptive import ColumnClusteringResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestValueCounts:
    def test_excludes_empty_and_sorts(self) -> None:
        s = pd.Series(["BBB", "BBB", "", "ABC HS", "ABC HS", "BBB"])
        out = _value_counts(s)
        assert out == [("BBB", 3), ("ABC HS", 2)]

    def test_ties_alphabetical(self) -> None:
        s = pd.Series(["a", "b", "a", "b"])
        out = _value_counts(s)
        assert out == [("a", 2), ("b", 2)]


class TestIsCandidate:
    @pytest.mark.parametrize("n_rows,n_distinct,min_distinct,max_ratio,expected", [
        (10, 1, 2, 0.95, False),     # too few distinct
        (10, 2, 2, 0.95, True),
        (10, 9, 2, 0.95, False),     # ratio 0.9 -- wait, allowed
        (10, 10, 2, 0.95, False),    # all distinct -> too high
        (10, 5, 2, 0.5, True),       # exactly at ratio
        (10, 6, 2, 0.5, False),      # 0.6 > 0.5
        (0, 5, 2, 0.95, False),      # empty source
    ])
    def test_threshold_logic(self, n_rows, n_distinct, min_distinct,
                             max_ratio, expected) -> None:
        # tweak: 10 rows / 9 distinct = 0.9 which is < 0.95 so it
        # would actually pass; let me re-check the test_cases.
        if (n_rows, n_distinct, max_ratio) == (10, 9, 0.95):
            expected = True
        assert _is_candidate(
            n_rows, n_distinct,
            min_distinct=min_distinct,
            max_distinct_ratio=max_ratio,
        ) is expected


class TestEnumerateCandidates:
    def _build(self, tmp_path: Path) -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # smoke.csv: School (inconsistent), State (inconsistent), Id (unique)
        rows = [
            {"School": "BBB",    "State": "CA",     "Id": f"i{i}"}
            for i in range(5)
        ] + [
            {"School": "ABC HS", "State": "CA",     "Id": f"j{i}"}
            for i in range(2)
        ] + [
            {"School": "EEE",    "State": "Calif.", "Id": f"k{i}"}
            for i in range(3)
        ]
        pd.DataFrame(rows).to_csv(data_dir / "smoke.csv", index=False)
        return data_dir

    def test_returns_low_cardinality_only(self, tmp_path: Path) -> None:
        data_dir = self._build(tmp_path)
        cands = enumerate_candidates(
            data_dir=data_dir,
            sources=["smoke.csv"],
            min_distinct=2,
            max_distinct_ratio=0.5,
        )
        cols = {(src, col) for src, col, _ in cands}
        assert cols == {("smoke.csv", "School"), ("smoke.csv", "State")}
        # Id has 10/10 distinct, excluded.

    def test_source_filter(self, tmp_path: Path) -> None:
        data_dir = self._build(tmp_path)
        # Build a second source.
        pd.DataFrame({"X": ["a", "a", "b"]}).to_csv(
            data_dir / "other.csv", index=False,
        )
        cands = enumerate_candidates(
            data_dir=data_dir,
            sources=["smoke.csv", "other.csv"],
            min_distinct=2,
            max_distinct_ratio=0.95,
            source_filter="other.csv",
        )
        assert {(s, c) for s, c, _ in cands} == {("other.csv", "X")}

    def test_column_filter(self, tmp_path: Path) -> None:
        data_dir = self._build(tmp_path)
        cands = enumerate_candidates(
            data_dir=data_dir,
            sources=["smoke.csv"],
            min_distinct=2,
            max_distinct_ratio=0.95,
            column_filter="School",
        )
        assert {(s, c) for s, c, _ in cands} == {("smoke.csv", "School")}

    def test_missing_source_logged_and_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        data_dir = self._build(tmp_path)
        with caplog.at_level("WARNING"):
            cands = enumerate_candidates(
                data_dir=data_dir,
                sources=["nope.csv"],
                min_distinct=2,
                max_distinct_ratio=0.95,
            )
        assert cands == []
        assert any("not found" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# GT artifact append
# ---------------------------------------------------------------------------


class TestAppendPerArmStats:
    def _make_result(self, src: str, col: str, *,
                     runs: int = 3,
                     gt: float = 0.02, chao1: float = 0.01,
                     ) -> ColumnClusteringResult:
        return ColumnClusteringResult(
            source=src, column=col,
            runs_completed=runs,
            stopped_reason="gt",
            gt_missing_mass=gt,
            chao1_missed=chao1,
            draft_clusters=[],
        )

    def test_writes_fresh_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _append_per_arm_stats(run_dir, [
            self._make_result("a.csv", "X"),
            self._make_result("a.csv", "Y"),
        ])
        path = run_dir / "good_turing" / "per_arm_estimates.json"
        body = json.loads(path.read_text(encoding="utf-8"))
        assert len(body["arms"]) == 2
        for a in body["arms"]:
            assert a["scope_kind"] == "column_clustering"
            assert a["error_type"] == "inconsistency"

    def test_merges_with_existing_non_column_arms(
        self, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        gt_dir = run_dir / "good_turing"
        gt_dir.mkdir(parents=True)
        (gt_dir / "per_arm_estimates.json").write_text(json.dumps({
            "arms": [
                {"scope_kind": "column", "scope_id": "a.csv::X",
                 "error_type": "typo", "runs_completed": 2,
                 "stopped_reason": "gt",
                 "good_turing_missing_mass": 0.01,
                 "chao1_estimated_missed": 0.0,
                 "n_species_observed": 1},
            ],
        }))
        _append_per_arm_stats(run_dir, [
            self._make_result("a.csv", "X"),
        ])
        body = json.loads(
            (gt_dir / "per_arm_estimates.json").read_text(encoding="utf-8"),
        )
        arms = body["arms"]
        kinds = sorted(a["scope_kind"] for a in arms)
        assert kinds == ["column", "column_clustering"]

    def test_replaces_stale_column_clustering_arms(
        self, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        gt_dir = run_dir / "good_turing"
        gt_dir.mkdir(parents=True)
        (gt_dir / "per_arm_estimates.json").write_text(json.dumps({
            "arms": [
                {"scope_kind": "column_clustering",
                 "scope_id": "a.csv::OLD",
                 "error_type": "inconsistency",
                 "runs_completed": 5,
                 "stopped_reason": "gt",
                 "good_turing_missing_mass": 0.10,
                 "chao1_estimated_missed": 0.05,
                 "n_species_observed": 0},
            ],
        }))
        _append_per_arm_stats(run_dir, [
            self._make_result("a.csv", "NEW"),
        ])
        body = json.loads(
            (gt_dir / "per_arm_estimates.json").read_text(encoding="utf-8"),
        )
        scope_ids = [a["scope_id"] for a in body["arms"]]
        assert scope_ids == ["a.csv::NEW"]  # OLD was dropped


# ---------------------------------------------------------------------------
# enumerate_candidates_from_frames
# ---------------------------------------------------------------------------


class TestEnumerateCandidatesFromFrames:
    def test_works_on_in_memory_frames(self) -> None:
        frames = {
            "a.csv": pd.DataFrame({
                "Cat": ["x", "x", "y", "y", "z"],
                "Free": ["aa", "bb", "cc", "dd", "ee"],  # too unique
            }),
            "b.csv": pd.DataFrame({
                "Only": ["k", "k", "k"],  # only 1 distinct -> skipped
            }),
        }
        cands = enumerate_candidates_from_frames(
            source_frames=frames,
            min_distinct=2,
            max_distinct_ratio=0.95,
        )
        names = sorted((s, c) for s, c, _ in cands)
        assert names == [("a.csv", "Cat")]

    def test_source_filter(self) -> None:
        frames = {
            "a.csv": pd.DataFrame({"X": ["a", "b", "c"]}),
            "b.csv": pd.DataFrame({"X": ["a", "b", "c"]}),
        }
        cands = enumerate_candidates_from_frames(
            source_frames=frames,
            min_distinct=2,
            max_distinct_ratio=1.0,
            source_filter="b.csv",
        )
        assert [s for s, _, _ in cands] == ["b.csv"]


# ---------------------------------------------------------------------------
# run_for_candidates: driver using a fake sweep
# ---------------------------------------------------------------------------


class _StubProposer:
    def propose_once(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError(
            "proposer should not be called when run_sweep is faked"
        )


def _fake_sweep_factory(stopped_reason: str = "k_max"):
    """Build a ``run_sweep`` stand-in that returns one
    :class:`ColumnClusteringResult` per candidate without making any
    LLM call. The factory captures stop-reason / runs so individual
    tests can tune what the artifact writers see."""

    captured: dict = {"calls": []}

    def _fake(*, columns, proposer, arm_config, max_workers,
              on_pass_start=None, on_arm_completed=None,
              on_pass_end=None):
        captured["calls"].append({
            "columns": [(s, c) for s, c, _ in columns],
            "max_workers": max_workers,
            "arm_config": arm_config,
        })
        results = [
            ColumnClusteringResult(
                source=src, column=col,
                runs_completed=1,
                stopped_reason=stopped_reason,
                gt_missing_mass=0.0,
                chao1_missed=0.0,
                draft_clusters=[],
            )
            for src, col, _ in columns
        ]
        results.sort(key=lambda r: (r.source, r.column))
        return results

    return _fake, captured


class TestRunForCandidates:
    def test_writes_jsonl_and_per_arm_stats(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Patch the heavy LLM call + the per-run sweep inside the module
        # so the driver runs without touching the network or doing any
        # real GT/Chao1 estimation.
        import experiments.run_column_clustering as rcc

        monkeypatch.setattr(
            rcc, "LLMClient", lambda *a, **k: object(),
        )
        monkeypatch.setattr(
            rcc, "ValueClusterProposer", lambda *a, **k: _StubProposer(),
        )
        fake_sweep, _ = _fake_sweep_factory()
        monkeypatch.setattr(rcc, "run_sweep", fake_sweep)

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        prompt_path = tmp_path / "fake.yaml"  # never opened by the fake
        prompt_path.write_text("system: x\nuser: y\n", encoding="utf-8")

        results = run_for_candidates(
            run_dir=run_dir,
            candidates=[
                ("a.csv", "X", [("k", 2), ("v", 1)]),
                ("a.csv", "Y", [("p", 3)]),
            ],
            model="gpt-5.2",
            reasoning_effort="low",
            reasoning_summary="short",
            arm_config=ArmConfig.default(),
            prompt_path=prompt_path,
            max_workers=1,
        )
        assert {(r.source, r.column) for r in results} == {
            ("a.csv", "X"), ("a.csv", "Y"),
        }
        drafts = (run_dir / "column_cluster_drafts.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        assert len(drafts) == 2
        for line in drafts:
            payload = json.loads(line)
            assert payload["source"] == "a.csv"
            assert "column" in payload

    def test_empty_candidates_still_writes_empty_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import experiments.run_column_clustering as rcc

        monkeypatch.setattr(
            rcc, "LLMClient", lambda *a, **k: object(),
        )
        monkeypatch.setattr(
            rcc, "ValueClusterProposer", lambda *a, **k: _StubProposer(),
        )
        fake_sweep, captured = _fake_sweep_factory()
        monkeypatch.setattr(rcc, "run_sweep", fake_sweep)

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        prompt_path = tmp_path / "fake.yaml"
        prompt_path.write_text("system: x\nuser: y\n", encoding="utf-8")

        results = run_for_candidates(
            run_dir=run_dir,
            candidates=[],
            model="gpt-5.2",
            reasoning_effort="low",
            reasoning_summary="short",
            arm_config=ArmConfig.default(),
            prompt_path=prompt_path,
            max_workers=1,
        )
        assert results == []
        # An empty drafts file is still written so the labelling UI's
        # "Column board (0)" check is unambiguous (file exists -> the
        # pass ran; missing -> the pass never ran).
        assert (run_dir / "column_cluster_drafts.jsonl").is_file()
        assert (
            run_dir / "column_cluster_drafts.jsonl"
        ).read_text(encoding="utf-8") == ""
