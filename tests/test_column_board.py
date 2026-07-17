"""Tests for Slice 6 backend additions (the column-board page is
covered by manual smoke + JS code review; we test the API contract
here so the frontend can rely on the shapes we ship)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.app import create_app  # noqa: E402
from src.labeling.runs import _parse_run_dir  # noqa: E402


# ---------------------------------------------------------------------------
# Workspace builder
# ---------------------------------------------------------------------------


def _build_workspace(tmp_path: Path) -> Path:
    """Build a minimal workspace with one source CSV + one run dir."""
    ws = tmp_path
    (ws / "frontend" / "labeling").mkdir(parents=True)
    (ws / "outputs").mkdir()

    data_dir = ws / "data"
    data_dir.mkdir()
    pd.DataFrame({
        "State": ["CA"] * 5 + ["Calif."] * 3 + ["California"] * 2,
        "Year": ["2024"] * 10,
    }).to_csv(data_dir / "smoke.csv", index=False)

    run_dir = ws / "outputs" / "r1"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "config": {
            "csv_paths": [str(data_dir / "smoke.csv")],
            "sources": ["smoke.csv"],
        },
    }))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")
    return ws


# ---------------------------------------------------------------------------
# RunInfo
# ---------------------------------------------------------------------------


class TestRunInfoFields:
    def test_no_artifacts_yields_zero_counts(self, tmp_path: Path) -> None:
        ws = _build_workspace(tmp_path)
        info = _parse_run_dir(ws / "outputs" / "r1").to_dict()
        assert info["has_column_cluster_drafts"] is False
        assert info["n_column_cluster_drafts"] == 0
        assert info["has_column_cluster_labels"] is False
        assert info["n_column_cluster_labels"] == 0

    def test_drafts_dedup_by_source_column(self, tmp_path: Path) -> None:
        ws = _build_workspace(tmp_path)
        run_dir = ws / "outputs" / "r1"
        (run_dir / "column_cluster_drafts.jsonl").write_text(
            json.dumps({"source": "smoke.csv", "column": "State"}) + "\n" +
            json.dumps({"source": "smoke.csv", "column": "State"}) + "\n" +
            json.dumps({"source": "smoke.csv", "column": "Year"}) + "\n",
        )
        info = _parse_run_dir(run_dir).to_dict()
        assert info["has_column_cluster_drafts"] is True
        assert info["n_column_cluster_drafts"] == 2

    def test_labels_dedup_by_source_column(self, tmp_path: Path) -> None:
        ws = _build_workspace(tmp_path)
        run_dir = ws / "outputs" / "r1"
        (run_dir / "column_clusters.jsonl").write_text(
            json.dumps({"source": "smoke.csv", "column": "State"}) + "\n" +
            json.dumps({"source": "smoke.csv", "column": "State"}) + "\n",
        )
        info = _parse_run_dir(run_dir).to_dict()
        assert info["has_column_cluster_labels"] is True
        assert info["n_column_cluster_labels"] == 1

    def test_malformed_lines_are_skipped(self, tmp_path: Path) -> None:
        ws = _build_workspace(tmp_path)
        run_dir = ws / "outputs" / "r1"
        (run_dir / "column_cluster_drafts.jsonl").write_text(
            "not json\n"
            + json.dumps({"source": "smoke.csv", "column": "State"}) + "\n"
            + "\n",  # blank line
        )
        info = _parse_run_dir(run_dir).to_dict()
        assert info["n_column_cluster_drafts"] == 1


# ---------------------------------------------------------------------------
# /draft endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    ws = _build_workspace(tmp_path)
    # Point at the real frontend dir so static-file tests can see the
    # column_board.* files we ship in the repo.
    app = create_app(
        ws, annotator="reviewer",
        static_dir=PROJECT_ROOT / "frontend" / "labeling",
    )
    return TestClient(app)


class TestDraftEndpoint:
    def test_no_drafts_file_returns_null(self, client: TestClient) -> None:
        r = client.get(
            "/api/runs/r1/columns/smoke.csv/State/draft",
        )
        assert r.status_code == 200
        assert r.json() == {"draft": None}

    def test_unknown_run_returns_404(self, tmp_path: Path) -> None:
        ws = _build_workspace(tmp_path)
        app = create_app(ws, annotator="reviewer")
        with TestClient(app) as client:
            r = client.get(
                "/api/runs/no_such_run/columns/smoke.csv/State/draft",
            )
            assert r.status_code == 404

    def test_returns_matching_draft(self, tmp_path: Path) -> None:
        ws = _build_workspace(tmp_path)
        run_dir = ws / "outputs" / "r1"
        (run_dir / "column_cluster_drafts.jsonl").write_text(
            json.dumps({
                "source": "smoke.csv", "column": "State",
                "runs_completed": 3, "stopped_reason": "two_consecutive",
                "gt_missing_mass": 0.0, "chao1_missed": 0.0,
                "draft_clusters": [{"values": ["CA", "Calif.", "California"],
                                    "kind": "text",
                                    "canonical": "California",
                                    "support_runs": [0, 1, 2],
                                    "agreement": 1.0}],
            }) + "\n",
        )
        app = create_app(ws, annotator="reviewer")
        with TestClient(app) as client:
            r = client.get(
                "/api/runs/r1/columns/smoke.csv/State/draft",
            )
            assert r.status_code == 200
            draft = r.json()["draft"]
            assert draft is not None
            assert draft["column"] == "State"
            assert len(draft["draft_clusters"]) == 1
            assert draft["draft_clusters"][0]["canonical"] == "California"

    def test_latest_wins_for_repeated_column(self, tmp_path: Path) -> None:
        ws = _build_workspace(tmp_path)
        run_dir = ws / "outputs" / "r1"
        (run_dir / "column_cluster_drafts.jsonl").write_text(
            json.dumps({"source": "smoke.csv", "column": "State",
                        "runs_completed": 1, "draft_clusters": []}) + "\n"
            + json.dumps({"source": "smoke.csv", "column": "State",
                          "runs_completed": 5, "draft_clusters": []}) + "\n",
        )
        app = create_app(ws, annotator="reviewer")
        with TestClient(app) as client:
            r = client.get(
                "/api/runs/r1/columns/smoke.csv/State/draft",
            )
            assert r.json()["draft"]["runs_completed"] == 5

    def test_non_matching_column_returns_null(self, tmp_path: Path) -> None:
        ws = _build_workspace(tmp_path)
        run_dir = ws / "outputs" / "r1"
        (run_dir / "column_cluster_drafts.jsonl").write_text(
            json.dumps({"source": "smoke.csv", "column": "Year",
                        "draft_clusters": []}) + "\n",
        )
        app = create_app(ws, annotator="reviewer")
        with TestClient(app) as client:
            r = client.get(
                "/api/runs/r1/columns/smoke.csv/State/draft",
            )
            assert r.json() == {"draft": None}

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        ws = _build_workspace(tmp_path)
        run_dir = ws / "outputs" / "r1"
        (run_dir / "column_cluster_drafts.jsonl").write_text(
            "garbage\n"
            + json.dumps({"source": "smoke.csv", "column": "State",
                          "runs_completed": 2,
                          "draft_clusters": []}) + "\n",
        )
        app = create_app(ws, annotator="reviewer")
        with TestClient(app) as client:
            r = client.get(
                "/api/runs/r1/columns/smoke.csv/State/draft",
            )
            assert r.json()["draft"]["runs_completed"] == 2


# ---------------------------------------------------------------------------
# Static file serving (column_board.html / .js / .css)
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_column_board_html_served(self, client: TestClient) -> None:
        r = client.get("/ui/column_board.html")
        assert r.status_code == 200
        assert "column board" in r.text.lower()

    def test_column_board_js_served(self, client: TestClient) -> None:
        r = client.get("/ui/column_board.js")
        assert r.status_code == 200
        assert "openColumn" in r.text

    def test_column_board_js_supports_deep_link(
        self, client: TestClient,
    ) -> None:
        """The report HTML deep-links to ``column_board.html?run=...&source=...
        &column=...``; verify the JS still parses these params and
        auto-opens the column. A regression here would silently
        downgrade the report→board hop to a plain landing page."""
        r = client.get("/ui/column_board.js")
        assert r.status_code == 200
        # Both URL params must be read.
        assert 'get("source")' in r.text
        assert 'get("column")' in r.text
        # And both must drive the auto-open call site.
        assert "INITIAL_SOURCE" in r.text
        assert "INITIAL_COLUMN" in r.text
        assert "openColumn(INITIAL_SOURCE, INITIAL_COLUMN)" in r.text

    def test_column_board_js_handles_deep_link_outside_queue(
        self, client: TestClient,
    ) -> None:
        """Columns with n_distinct/n_rows > 0.95 are filtered out of
        the standard queue. The deep-link path must still open them
        by injecting a synthetic queue item; otherwise the report
        link silently dead-ends on near-unique columns."""
        r = client.get("/ui/column_board.js")
        assert r.status_code == 200
        # The bootstrap injects a synthetic queue item with this marker.
        assert "_outside_default_queue" in r.text
        # And calls openColumn unconditionally once the deep-link
        # params are present (no longer gated on `state.queue.find`).
        assert "state.queue.unshift" in r.text

    def test_column_board_css_served(self, client: TestClient) -> None:
        r = client.get("/ui/column_board.css")
        assert r.status_code == 200
        assert "cb-cluster-card" in r.text


# ---------------------------------------------------------------------------
# Slash-in-column-name routing (alias endpoints).
#
# Some columns in real CSVs have ``/`` in the header (e.g.
# ``"Project Name/Description"``). Starlette decodes ``%2F`` to ``/``
# before path matching, so the original ``…/{column}/<action>`` shape
# 404s on such columns. The alias routes mount the same handlers
# under ``…/<action>/{column:path}`` to fix this.
# ---------------------------------------------------------------------------


def _build_workspace_with_slash(tmp_path: Path) -> Path:
    """Workspace whose source CSV has a column with ``/`` in the name."""
    ws = tmp_path
    (ws / "frontend" / "labeling").mkdir(parents=True)
    (ws / "outputs").mkdir()

    data_dir = ws / "data"
    data_dir.mkdir()
    pd.DataFrame({
        "Project Name/Description": [
            "Road Inventory", "Road Inventory", "Culvert",
        ],
        "Year": ["2024", "2025", "2024"],
    }).to_csv(data_dir / "smoke.csv", index=False)

    run_dir = ws / "outputs" / "r1"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "config": {
            "csv_paths": [str(data_dir / "smoke.csv")],
            "sources": ["smoke.csv"],
        },
    }))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")
    return ws


class TestSlashInColumnName:
    def test_values_alias_with_slash(self, tmp_path: Path) -> None:
        ws = _build_workspace_with_slash(tmp_path)
        app = create_app(ws, annotator="reviewer")
        with TestClient(app) as client:
            r = client.get(
                "/api/runs/r1/columns/smoke.csv/values/"
                "Project%20Name/Description",
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["column"] == "Project Name/Description"
            assert body["n_distinct"] == 2

    def test_clusters_alias_get_with_slash(self, tmp_path: Path) -> None:
        ws = _build_workspace_with_slash(tmp_path)
        app = create_app(ws, annotator="reviewer")
        with TestClient(app) as client:
            r = client.get(
                "/api/runs/r1/columns/smoke.csv/clusters/"
                "Project%20Name/Description",
            )
            assert r.status_code == 200
            assert r.json() == {"record": None, "history": []}

    def test_clusters_alias_post_with_slash(self, tmp_path: Path) -> None:
        ws = _build_workspace_with_slash(tmp_path)
        app = create_app(ws, annotator="reviewer")
        with TestClient(app) as client:
            # Use status=no_inconsistency so we don't have to satisfy
            # the cluster-content invariants (>=2 values etc.) -- this
            # test is purely about the slash-routing alias.
            payload = {
                "source": "smoke.csv",
                "column": "Project Name/Description",
                "status": "no_inconsistency",
                "drop_reason": "",
                "clusters": [],
                "note": "",
            }
            r = client.post(
                "/api/runs/r1/columns/smoke.csv/clusters/"
                "Project%20Name/Description",
                json=payload,
            )
            assert r.status_code == 200, r.text
            rec = r.json()["record"]
            assert rec["source"] == "smoke.csv"
            assert rec["column"] == "Project Name/Description"
            assert rec["status"] == "no_inconsistency"

    def test_draft_alias_with_slash(self, tmp_path: Path) -> None:
        ws = _build_workspace_with_slash(tmp_path)
        run_dir = ws / "outputs" / "r1"
        (run_dir / "column_cluster_drafts.jsonl").write_text(
            json.dumps({
                "source": "smoke.csv",
                "column": "Project Name/Description",
                "runs_completed": 5,
                "stopped_reason": "k_max",
                "gt_missing_mass": 0.5,
                "chao1_missed": 0.3,
                "draft_clusters": [{
                    "values": ["Road Inventory"],
                    "kind": "text",
                    "canonical": "Road Inventory",
                    "support_runs": [0, 1],
                    "agreement": 1.0,
                }],
            }) + "\n",
        )
        app = create_app(ws, annotator="reviewer")
        with TestClient(app) as client:
            r = client.get(
                "/api/runs/r1/columns/smoke.csv/draft/"
                "Project%20Name/Description",
            )
            assert r.status_code == 200
            draft = r.json()["draft"]
            assert draft is not None
            assert draft["column"] == "Project Name/Description"
            assert draft["draft_clusters"][0]["canonical"] == "Road Inventory"
