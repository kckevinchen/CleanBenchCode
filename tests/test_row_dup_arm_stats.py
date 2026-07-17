"""Tests for ``_append_row_dup_arm_stats`` -- the writer that merges
whole_table row_duplication per-CSV Good-Turing arms into the run's
``good_turing/per_arm_estimates.json`` without clobbering other arm
kinds.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.run_cleaning_detection import _append_row_dup_arm_stats


def _arm(scope_kind: str, scope_id: str, status: str = "saturated") -> dict:
    return {
        "scope_kind": scope_kind,
        "scope_id": scope_id,
        "error_type": scope_kind,
        "runs_completed": 3,
        "num_runs": 3,
        "status": status,
        "stopped_reason": "gt" if status == "saturated" else "k_max",
        "n_observed": 1,
        "n_species_observed": 1,
        "good_turing_missing_mass": 0.0,
        "chao1_estimated_missed": 0.0,
    }


def _read_arms(tmp_path: Path) -> list:
    return json.loads(
        (tmp_path / "good_turing" / "per_arm_estimates.json").read_text()
    )["arms"]


def test_preserves_other_scope_kinds(tmp_path: Path) -> None:
    p = tmp_path / "good_turing" / "per_arm_estimates.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"arms": [
        _arm("column", "a.csv::col"),
        _arm("column_clustering", "b.csv::x"),
    ]}))
    _append_row_dup_arm_stats(tmp_path, [_arm("row_duplication", "c.csv")])
    kinds = sorted(a["scope_kind"] for a in _read_arms(tmp_path))
    assert kinds == ["column", "column_clustering", "row_duplication"]


def test_idempotent_re_run_drops_prior_row_dup_arms(tmp_path: Path) -> None:
    _append_row_dup_arm_stats(tmp_path, [_arm("row_duplication", "c.csv", "saturated")])
    _append_row_dup_arm_stats(tmp_path, [_arm("row_duplication", "c.csv", "exhausted")])
    rd = [a for a in _read_arms(tmp_path) if a["scope_kind"] == "row_duplication"]
    assert len(rd) == 1
    assert rd[0]["status"] == "exhausted"


def test_creates_file_when_missing(tmp_path: Path) -> None:
    _append_row_dup_arm_stats(tmp_path, [_arm("row_duplication", "c.csv")])
    arms = _read_arms(tmp_path)
    assert len(arms) == 1 and arms[0]["scope_kind"] == "row_duplication"


def test_skips_zero_run_arms(tmp_path: Path) -> None:
    zero = _arm("row_duplication", "c.csv")
    zero["runs_completed"] = 0
    _append_row_dup_arm_stats(tmp_path, [zero])
    assert _read_arms(tmp_path) == []


def test_tolerates_invalid_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "good_turing" / "per_arm_estimates.json"
    p.parent.mkdir(parents=True)
    p.write_text("{ not valid json")
    _append_row_dup_arm_stats(tmp_path, [_arm("row_duplication", "c.csv")])
    arms = _read_arms(tmp_path)
    assert len(arms) == 1 and arms[0]["scope_kind"] == "row_duplication"
