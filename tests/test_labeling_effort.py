"""Regression tests for the per-(source, column) labelling-effort
endpoint surfaced from the per-arm Good-Turing artifact.

Endpoint: ``GET /api/runs/{run_id}/labeling-effort``.

Source artifact: ``<run_dir>/good_turing/per_arm_estimates.json``
written by adaptive runs at ``stop_scope=column`` (one record per
(source, column, error_type) arm). The endpoint groups arms by
(source, column), aggregates Chao1 and Good-Turing estimates, and
cross-references the labelling store to expose per-cell remaining
effort.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app


def _bootstrap_run(project: Path, run_id: str = "20260101T000000Z_test") -> Path:
    data_dir = project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "a.csv").write_text(
        "name,email\nalice,a@x\nbob,b@x\n", encoding="utf-8"
    )
    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    run_dir = outputs / run_id
    run_dir.mkdir()
    summary = {
        "config": {"csv_paths": [str(data_dir / "a.csv")], "sources": ["a.csv"]},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")
    return run_dir


def _arm(source: str, column: str, etype: str, *, n_obs: int,
         est_total: float, missing_mass: float, status: str = "active"):
    return {
        "scope_kind": "column",
        "scope_id": [source, column],
        "error_type": etype,
        "num_runs": 5,
        "n_observed": n_obs,
        "frequency_counts": {"1": n_obs},
        "total_hits": n_obs,
        "good_turing_missing_mass": missing_mass,
        "chao1": {
            "estimated_missed": max(est_total - n_obs, 0.0),
            "estimated_total": est_total,
            "variance": 0.5,
            "ci_95": [est_total, est_total + 1.0],
        },
        "estimated_fn_rate": {"point": 0.0, "ci_95": [0.0, 0.0]},
        "status": status,
        "saturated_streak": 0,
    }


@pytest.fixture
def client_project(tmp_path: Path):
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    run_dir = _bootstrap_run(project)
    app = create_app(project, annotator="tester")
    with TestClient(app) as client:
        yield client, project, run_dir


class TestLabelingEffortEndpoint:
    def test_returns_unavailable_when_artifact_missing(self, client_project):
        client, _project, _run_dir = client_project
        # Use the same run_id created by the fixture.
        run_id = _run_dir.name
        r = client.get(f"/api/runs/{run_id}/labeling-effort")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False
        assert "reason" in body

    def test_returns_unavailable_for_non_column_scope(self, client_project):
        client, _project, run_dir = client_project
        gt_dir = run_dir / "good_turing"
        gt_dir.mkdir()
        (gt_dir / "per_arm_estimates.json").write_text(json.dumps({
            "scope_kind": "csv",
            "num_runs_completed": 5,
            "arms": [],
        }))
        r = client.get(f"/api/runs/{run_dir.name}/labeling-effort")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False

    def test_aggregates_by_source_and_column(self, client_project):
        client, _project, run_dir = client_project
        gt_dir = run_dir / "good_turing"
        gt_dir.mkdir()
        arms = [
            # (a.csv, name) - two error types
            _arm("a.csv", "name", "et1", n_obs=4, est_total=10.0,
                 missing_mass=0.30, status="active"),
            _arm("a.csv", "name", "et2", n_obs=2, est_total=3.0,
                 missing_mass=0.05, status="saturated"),
            # (a.csv, email) - one arm, all saturated
            _arm("a.csv", "email", "et1", n_obs=8, est_total=8.0,
                 missing_mass=0.01, status="saturated"),
        ]
        (gt_dir / "per_arm_estimates.json").write_text(json.dumps({
            "scope_kind": "column",
            "num_runs_completed": 7,
            "arms": arms,
        }))

        r = client.get(f"/api/runs/{run_dir.name}/labeling-effort")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["available"] is True
        assert body["scope_kind"] == "column"
        assert body["num_runs_completed"] == 7
        assert body["totals"]["cells"] == 2

        cells = {(c["source"], c["column"]): c for c in body["cells"]}
        assert set(cells) == {("a.csv", "name"), ("a.csv", "email")}

        name_cell = cells[("a.csv", "name")]
        # n_observed sums across error types (4 + 2)
        assert name_cell["n_observed"] == pytest.approx(6.0)
        # chao1_estimated_total sums (10 + 3)
        assert name_cell["chao1_estimated_total"] == pytest.approx(13.0)
        # estimated_missed sums (6 + 1)
        assert name_cell["chao1_estimated_missed"] == pytest.approx(7.0)
        # 2 arms total: 1 active + 1 saturated
        assert name_cell["arms_total"] == 2
        assert name_cell["arms_saturated"] == 1
        assert name_cell["arms_active"] == 1
        # Hit-weighted missing mass = (0.30*4 + 0.05*2)/(4+2) = 1.30/6
        assert name_cell["good_turing_missing_mass"] == pytest.approx(1.30 / 6.0)
        # by_error_type sorted by estimated_missed desc
        breakdown = name_cell["by_error_type"]
        assert [e["error_type"] for e in breakdown] == ["et1", "et2"]
        # No labels logged yet => effort_remaining == chao1_estimated_total
        assert name_cell["labelled_so_far"] == 0
        assert name_cell["effort_remaining"] == pytest.approx(13.0)

        email_cell = cells[("a.csv", "email")]
        assert email_cell["arms_total"] == 1
        assert email_cell["arms_saturated"] == 1
        assert email_cell["chao1_estimated_missed"] == pytest.approx(0.0)

        # Default sort: largest projected missed first
        assert body["cells"][0]["column"] == "name"

    def test_skips_malformed_arms(self, client_project):
        client, _project, run_dir = client_project
        gt_dir = run_dir / "good_turing"
        gt_dir.mkdir()
        arms = [
            _arm("a.csv", "name", "et1", n_obs=1, est_total=2.0,
                 missing_mass=0.1),
            # malformed: missing scope_id second element
            {"scope_kind": "column", "scope_id": ["a.csv"], "error_type": "et2",
             "n_observed": 0, "chao1": {"estimated_total": 0.0,
             "estimated_missed": 0.0, "ci_95": [0.0, 0.0]},
             "good_turing_missing_mass": 0.0, "status": "active"},
        ]
        (gt_dir / "per_arm_estimates.json").write_text(json.dumps({
            "scope_kind": "column",
            "arms": arms,
        }))
        r = client.get(f"/api/runs/{run_dir.name}/labeling-effort")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["totals"]["cells"] == 1
