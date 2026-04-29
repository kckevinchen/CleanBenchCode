"""End-to-end tests for the FN (false-negative) labelling flow.

Covers the four invariants the user asked for:

1. Operator can mark an unflagged cell as FN.
2. While reviewing the CSV, the operator can re-label a cluster's
   verdict via the cell popover (handled at the cluster endpoint -
   exercised here as ``POST /labels`` with target=cluster).
3. Double-labelling: an FN cell that ALSO has a TP cluster covering it
   is silently superseded by the cluster (TP wins). The cell-state
   endpoint surfaces this via ``superseded_fn_by`` so the UI can warn,
   and ``stats.fn.shadowed`` counts them.
4. Cells already covered by a TP cluster cannot themselves become an
   FN error in the benchmark - the benchmark exporter merges the FN
   into the existing TP record instead of producing two cells.

These tests use ``fastapi.testclient`` against an in-memory run dir.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.schema import LabelStatus, LabelTarget, cell_target_id


def _make_run(project: Path, run_id: str = "20260101T000000Z_test") -> Path:
    """Create a minimal but valid run dir with one TP-eligible cluster
    on (a.csv, row 1, col=name) and an unflagged cell at (a.csv, row 2,
    col=email). Returns the run dir."""
    # Create the on-disk CSV so the labeling server can resolve sources.
    data_dir = project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "a.csv").write_text(
        "name,email\nalice,a@x\nbob,b@x\ncarol,c@x\n", encoding="utf-8",
    )

    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    run_dir = outputs / run_id
    run_dir.mkdir()

    csv_paths = [str(data_dir / "a.csv")]
    summary = {"config": {"csv_paths": csv_paths, "sources": ["a.csv"]}}
    (run_dir / "summary.json").write_text(json.dumps(summary))
    clusters = [
        {
            "cluster_id": "c1",
            "source": "a.csv",
            "row": 1,
            "columns": ["name"],
            "error_types": ["typo"],
            "evidence_count": 1,
            "synthesis": None,
        },
    ]
    (run_dir / "clusters.json").write_text(json.dumps(clusters))
    (run_dir / "issues.json").write_text("[]")
    return run_dir


@pytest.fixture
def client_and_run():
    with tempfile.TemporaryDirectory() as td:
        project = Path(td)
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        app = create_app(project, annotator="tester")
        client = TestClient(app)
        yield client, run_dir.name


# ---------------------------------------------------------------------------
# Cell target_id helper
# ---------------------------------------------------------------------------


def test_cell_target_id_is_stable() -> None:
    a = cell_target_id("a.csv", 5, "name")
    b = cell_target_id("a.csv", 5, "name")
    c = cell_target_id("a.csv", 5, "email")
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# Posting FN cell labels
# ---------------------------------------------------------------------------


class TestPostFNLabel:
    def test_mark_unflagged_cell_as_fn(self, client_and_run) -> None:
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell",
                "target_id": "ignored-server-canonicalises",
                "source": "a.csv",
                "row": 2,
                "column": "email",
                "status": "fn",
                "note": "missed",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "fn"
        assert body["target"] == "cell"
        # Server canonicalised the target_id.
        assert body["target_id"] == cell_target_id("a.csv", 2, "email")

    def test_cell_label_requires_source_row_column(self, client_and_run) -> None:
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell",
                "target_id": "x",
                "source": "a.csv",
                "row": None,
                "column": "email",
                "status": "fn",
            },
        )
        assert r.status_code == 400

    def test_cell_label_status_constrained(self, client_and_run) -> None:
        client, run_id = client_and_run
        # tp is not allowed on cell target - the cluster path owns TP.
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell",
                "target_id": "x",
                "source": "a.csv",
                "row": 2,
                "column": "email",
                "status": "tp",
            },
        )
        assert r.status_code == 400

    def test_cell_label_unknown_source_404(self, client_and_run) -> None:
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell",
                "target_id": "x",
                "source": "missing.csv",
                "row": 1,
                "column": "email",
                "status": "fn",
            },
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Cell-state derivation + double-labelling shadowing
# ---------------------------------------------------------------------------


class TestCellState:
    def _post_cluster_status(
        self, client: TestClient, run_id: str, cid: str, status: str,
    ) -> None:
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster",
                "target_id": cid,
                "source": "a.csv",
                "row": 1,
                "column": "name",
                "status": status,
            },
        )
        assert r.status_code == 200, r.text

    def test_unlabelled_run_has_empty_cell_state(self, client_and_run) -> None:
        client, run_id = client_and_run
        r = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()
        assert r["cells"] == []
        assert r["shadowed_fn_count"] == 0

    def test_tp_cluster_paints_its_cells(self, client_and_run) -> None:
        client, run_id = client_and_run
        self._post_cluster_status(client, run_id, "c1", "tp")
        cells = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        # row=1, col=name should now be tp.
        match = [c for c in cells if c["row"] == 1 and c["column"] == "name"]
        assert match and match[0]["state"] == "tp"
        assert "c1" in match[0]["cluster_ids"]

    def test_fn_cell_appears_in_state(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": "fn",
            },
        )
        cells = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        match = [c for c in cells if c["row"] == 2 and c["column"] == "email"]
        assert match and match[0]["state"] == "fn"
        assert match[0]["label_source"] == "cell"

    def test_tp_cluster_supersedes_pre_existing_fn_on_same_cell(
        self, client_and_run,
    ) -> None:
        """Operator labels (a.csv, row=1, col=name) as FN, then later
        promotes cluster c1 (which covers the same cell) to TP. The
        cell-state endpoint must report ``state=tp`` AND surface
        ``superseded_fn_by`` so the UI can warn about the
        double-labelling.
        """
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 1, "column": "name", "status": "fn", "note": "missed",
            },
        )
        self._post_cluster_status(client, run_id, "c1", "tp")
        body = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()
        match = [c for c in body["cells"] if c["row"] == 1 and c["column"] == "name"]
        assert match
        assert match[0]["state"] == "tp"
        assert match[0]["superseded_fn_by"] == ["c1"]
        assert body["shadowed_fn_count"] == 1

    def test_fp_clears_prior_fn(self, client_and_run) -> None:
        client, run_id = client_and_run
        # FN, then change of heart -> FP on the same cell.
        for status in ("fn", "fp"):
            client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "cell", "target_id": "x", "source": "a.csv",
                    "row": 2, "column": "email", "status": status,
                },
            )
        cells = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        match = [c for c in cells if c["row"] == 2 and c["column"] == "email"]
        # Latest verdict on the cell key is FP, so FN shouldn't surface.
        assert not match or match[0]["state"] != "fn"


# ---------------------------------------------------------------------------
# Stats include FN + shadowed count
# ---------------------------------------------------------------------------


class TestStatsFN:
    def test_stats_carry_fn_block(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": "fn",
            },
        )
        s = client.get(f"/api/runs/{run_id}/stats").json()
        assert s["fn"]["total"] == 1
        assert s["fn"]["shadowed"] == 0
        assert s["fn"]["by_source"]["a.csv"] == 1

    def test_shadowed_when_tp_covers_fn(self, client_and_run) -> None:
        client, run_id = client_and_run
        # FN on the same cell c1 covers, then c1 -> TP.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 1, "column": "name", "status": "fn",
            },
        )
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp",
            },
        )
        s = client.get(f"/api/runs/{run_id}/stats").json()
        assert s["fn"]["total"] == 1
        assert s["fn"]["shadowed"] == 1


# ---------------------------------------------------------------------------
# Sibling-tables surface
# ---------------------------------------------------------------------------


def test_get_source_returns_siblings_for_workbook(tmp_path: Path) -> None:
    project = tmp_path
    (project / "frontend" / "labeling").mkdir(parents=True)
    workbook = project / "data" / "wb1"
    workbook.mkdir(parents=True)
    for name in ("a.csv", "b.csv", "c.csv"):
        (workbook / name).write_text("col\nv\n", encoding="utf-8")

    outputs = project / "outputs"
    outputs.mkdir()
    run_dir = outputs / "20260101T000000Z_test"
    run_dir.mkdir()
    csv_paths = [str(workbook / n) for n in ("a.csv", "b.csv", "c.csv")]
    (run_dir / "summary.json").write_text(json.dumps({
        "config": {"csv_paths": csv_paths, "sources": ["a.csv", "b.csv", "c.csv"]},
    }))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")

    client = TestClient(create_app(project, annotator="tester"))
    r = client.get(f"/api/runs/{run_dir.name}/sources/a.csv").json()
    assert r["workbook"] == "wb1"
    assert r["siblings"] == ["b.csv", "c.csv"]


# ---------------------------------------------------------------------------
# Benchmark export folds in FN cells
# ---------------------------------------------------------------------------


class TestExportWithFN:
    def test_fn_cell_emits_error_record_with_empty_cluster_ids(
        self, client_and_run,
    ) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": "fn",
                "note": "missed", "corrected_value": "b@example.com",
            },
        )
        r = client.get(f"/api/runs/{run_id}/export/benchmark")
        assert r.status_code == 200
        rows: List[Dict[str, Any]] = [
            json.loads(line) for line in r.text.splitlines() if line.strip()
        ]
        # No TP clusters labelled, so the only benchmark row comes from
        # the FN cell.
        assert len(rows) == 1
        rec = rows[0]
        assert rec["source"] == "a.csv"
        assert rec["row"] == 2
        assert rec["column"] == "email"
        assert rec["label"] == "error"
        assert rec["cluster_ids"] == []
        assert rec["corrected_value"] == "b@example.com"

    def test_fn_superseded_by_tp_cluster_is_merged_not_duplicated(
        self, client_and_run,
    ) -> None:
        client, run_id = client_and_run
        # FN on the same cell c1 covers ...
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 1, "column": "name", "status": "fn",
                "note": "alice should be Alice",
            },
        )
        # ... then cluster c1 -> TP, which should swallow the FN.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp", "corrected_value": "Alice",
            },
        )
        rows = [
            json.loads(line)
            for line in client.get(f"/api/runs/{run_id}/export/benchmark").text.splitlines()
            if line.strip()
        ]
        # Exactly one cell record, owned by the TP cluster.
        cells = [r for r in rows if r["row"] == 1 and r["column"] == "name"]
        assert len(cells) == 1
        rec = cells[0]
        assert rec["label"] == "error"
        assert rec["cluster_ids"] == ["c1"]
        assert rec["corrected_value"] == "Alice"


# ---------------------------------------------------------------------------
# Per-source coverage surface (checked / labelled_clusters / fn_cells)
# ---------------------------------------------------------------------------


class TestSourceCoverage:
    def test_unlabelled_source_is_not_checked(self, client_and_run) -> None:
        client, run_id = client_and_run
        run = client.get(f"/api/runs/{run_id}").json()
        meta = next(s for s in run["sources"] if s["name"] == "a.csv")
        assert meta["checked"] is False
        assert meta["labelled_clusters"] == 0
        assert meta["fn_cells"] == 0

    def test_cluster_label_marks_source_checked(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp",
            },
        )
        meta = next(
            s for s in client.get(f"/api/runs/{run_id}").json()["sources"]
            if s["name"] == "a.csv"
        )
        assert meta["checked"] is True
        assert meta["labelled_clusters"] == 1
        assert meta["fn_cells"] == 0

    def test_fn_cell_marks_source_checked(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": "fn",
            },
        )
        meta = next(
            s for s in client.get(f"/api/runs/{run_id}").json()["sources"]
            if s["name"] == "a.csv"
        )
        assert meta["checked"] is True
        assert meta["labelled_clusters"] == 0
        assert meta["fn_cells"] == 1

    def test_unflagged_csv_is_visible_and_browsable(self, tmp_path: Path) -> None:
        """A CSV that the LLM never produced clusters for must still
        appear in /api/runs/{run_id} (so the UI can list it) and be
        readable via /sources/{name} (so the operator can browse it
        for FN candidates).
        """
        project = tmp_path
        (project / "frontend" / "labeling").mkdir(parents=True)
        data = project / "data"
        data.mkdir()
        (data / "untouched.csv").write_text("col\nfoo\nbar\n", encoding="utf-8")
        (data / "flagged.csv").write_text("col\nx\ny\n", encoding="utf-8")

        out = project / "outputs" / "20260101T000000Z_test"
        out.mkdir(parents=True)
        csv_paths = [str(data / "untouched.csv"), str(data / "flagged.csv")]
        (out / "summary.json").write_text(json.dumps({
            "config": {"csv_paths": csv_paths,
                       "sources": ["untouched.csv", "flagged.csv"]},
        }))
        # Only flagged.csv has a cluster; untouched.csv has none.
        (out / "clusters.json").write_text(json.dumps([{
            "cluster_id": "c1", "source": "flagged.csv",
            "row": 0, "columns": ["col"], "error_types": ["typo"],
            "evidence_count": 1, "synthesis": None,
        }]))
        (out / "issues.json").write_text("[]")

        from src.labeling.app import create_app
        client = TestClient(create_app(project, annotator="t"))
        run = client.get(f"/api/runs/{out.name}").json()
        names = [s["name"] for s in run["sources"]]
        assert "untouched.csv" in names
        meta = next(s for s in run["sources"] if s["name"] == "untouched.csv")
        assert meta["clusters"] == 0
        assert meta["checked"] is False

        # The unflagged CSV is browsable for FN labelling.
        slice_ = client.get(f"/api/runs/{out.name}/sources/untouched.csv").json()
        assert slice_["total_rows"] == 2
        cells = client.get(
            f"/api/runs/{out.name}/sources/untouched.csv/cells"
        ).json()
        assert cells["cells"] == []

        # And FN labelling on it works end-to-end.
        r = client.post(
            f"/api/runs/{out.name}/labels",
            json={
                "target": "cell", "target_id": "x",
                "source": "untouched.csv", "row": 1, "column": "col",
                "status": "fn", "note": "should be Bar",
            },
        )
        assert r.status_code == 200
        # ``checked`` flips after the FN.
        meta = next(
            s for s in client.get(f"/api/runs/{out.name}").json()["sources"]
            if s["name"] == "untouched.csv"
        )
        assert meta["checked"] is True
        assert meta["fn_cells"] == 1
