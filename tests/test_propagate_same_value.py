"""Same-value FN propagation: when an operator marks a cluster as TP
and supplies a corrected value, every OTHER row in the same column
holding the identical erroneous value should be auto-labelled FN with
the same correction - unless that cell is already covered by another
cluster verdict or has its own pre-existing cell label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.schema import LabelStatus, LabelTarget


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    """Tiny run with one CSV that has the typo "AcmeCo" repeated in
    several rows of the ``contractor`` column. Two existing clusters
    flag the typo on rows 1 and 5 only - rows 3 and 7 are also typos
    but the LLM missed them (= the FN candidates we expect to auto-
    propagate to)."""
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "contracts.csv"
    csv.write_text(
        "id,contractor,amount\n"
        "1,AcmeCo,100\n"      # row 0 - typo, NOT flagged
        "2,AcmeCo,150\n"      # row 1 - typo, FLAGGED (cluster c000)
        "3,Acme Corp,200\n"   # row 2 - clean
        "4,AcmeCo,175\n"      # row 3 - typo, NOT flagged
        "5,Beta Inc,250\n"    # row 4 - clean
        "6,AcmeCo,300\n"      # row 5 - typo, FLAGGED (cluster c001)
        "7,Acme Corp,225\n"   # row 6 - clean
        "8,AcmeCo,400\n",     # row 7 - typo, NOT flagged
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260428T000000Z_propagate"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "contracts.csv__r00001-c000", "source": "contracts.csv",
         "row": 1, "columns": ["contractor"],
         "row_values": {"id": "2", "contractor": "AcmeCo", "amount": "150"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
        {"cluster_id": "contracts.csv__r00005-c000", "source": "contracts.csv",
         "row": 5, "columns": ["contractor"],
         "row_values": {"id": "6", "contractor": "AcmeCo", "amount": "300"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_propagate_writes_fn_for_uncovered_same_value_rows(tmp_path: Path):
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/labels/propagate-same-value", json={
            "source": "contracts.csv",
            "row": 1,
            "column": "contractor",
            "corrected_value": "Acme Corp",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        # Anchor row=1 with value "AcmeCo" matches rows 0, 3, 5, 7.
        assert body["matched"] == 4
        # Row 5 is covered by cluster c001 (no verdict yet, but the
        # cluster covers it so we don't pre-empt the operator's call).
        # Wait: only TP/FP/PARTIAL/UNSURE clusters block. An UNLABELLED
        # cluster does NOT block - the operator hasn't ruled, so we
        # are free to FN siblings whose only "coverage" is a still-
        # untouched cluster.
        assert body["propagated"] == 4
        assert body["skipped_covered"] == 0
        assert body["skipped_existing"] == 0
        rows_written = sorted(c["row"] for c in body["cells"])
        assert rows_written == [0, 3, 5, 7]


def test_propagate_skips_rows_already_covered_by_labelled_cluster(tmp_path: Path):
    """A row whose only-covering cluster carries a TP/FP/PARTIAL/UNSURE
    verdict must NOT be overwritten - the operator already ruled."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Pre-label the *other* AcmeCo cluster (row 5) as FP. Now when
        # we propagate from row 1, row 5 must be skipped because it's
        # actively claimed by a verdict.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "contracts.csv__r00005-c000",
            "source": "contracts.csv",
            "row": 5, "column": "contractor",
            "status": "fp",
        })
        assert r.status_code == 200, r.text

        r = client.post(f"/api/runs/{run_id}/labels/propagate-same-value", json={
            "source": "contracts.csv",
            "row": 1,
            "column": "contractor",
            "corrected_value": "Acme Corp",
        })
        body = r.json()
        assert body["matched"] == 4
        assert body["skipped_covered"] == 1     # row 5
        assert body["propagated"] == 3          # rows 0, 3, 7
        assert sorted(c["row"] for c in body["cells"]) == [0, 3, 7]


def test_propagate_skips_existing_cell_label(tmp_path: Path):
    """A cell that already carries an explicit cell label (from a prior
    FN action) must be left alone - clobbering an annotator's prior
    correction would silently lose ground truth."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Operator manually FN'd row 3 earlier with a different fix.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell",
            "target_id": "ignored-rebuilt",
            "source": "contracts.csv",
            "row": 3, "column": "contractor",
            "status": "fn",
            "corrected_value": "AcmeCorp Holdings",
        })
        assert r.status_code == 200, r.text

        r = client.post(f"/api/runs/{run_id}/labels/propagate-same-value", json={
            "source": "contracts.csv",
            "row": 1,
            "column": "contractor",
            "corrected_value": "Acme Corp",
        })
        body = r.json()
        assert body["skipped_existing"] == 1    # row 3
        # Rows 0, 5, 7 still get propagated.
        assert body["propagated"] == 3
        assert 3 not in {c["row"] for c in body["cells"]}


def test_propagate_requires_corrected_value(tmp_path: Path):
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/labels/propagate-same-value", json={
            "source": "contracts.csv",
            "row": 1,
            "column": "contractor",
            # corrected_value omitted
        })
        assert r.status_code == 400


def test_propagate_validates_source_row_column(tmp_path: Path):
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Bad source.
        r = client.post(f"/api/runs/{run_id}/labels/propagate-same-value", json={
            "source": "nope.csv", "row": 1, "column": "contractor",
            "corrected_value": "x",
        })
        assert r.status_code == 404
        # Bad column.
        r = client.post(f"/api/runs/{run_id}/labels/propagate-same-value", json={
            "source": "contracts.csv", "row": 1, "column": "nope",
            "corrected_value": "x",
        })
        assert r.status_code == 404
        # Out-of-range row.
        r = client.post(f"/api/runs/{run_id}/labels/propagate-same-value", json={
            "source": "contracts.csv", "row": 99, "column": "contractor",
            "corrected_value": "x",
        })
        assert r.status_code == 400


def test_propagated_cells_show_up_in_cell_state(tmp_path: Path):
    """After propagation the per-cell state endpoint must report the
    new FN cells with the supplied corrected_value (so the labeling
    UI paints them and the benchmark export picks them up)."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels/propagate-same-value", json={
            "source": "contracts.csv", "row": 1, "column": "contractor",
            "corrected_value": "Acme Corp",
        })
        cells = client.get(
            f"/api/runs/{run_id}/sources/contracts.csv/cells"
        ).json()["cells"]
        by_key = {(c["row"], c["column"]): c for c in cells}
        for r in (0, 3, 5, 7):
            entry = by_key.get((r, "contractor"))
            assert entry is not None, f"row {r} missing from cell state"
            assert entry["state"] == "fn"
            assert entry["corrected_value"] == "Acme Corp"
