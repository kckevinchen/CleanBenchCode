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


# ---------------------------------------------------------------------
# Cluster-status propagation: when the operator marks ONE cluster as
# FP (the LLM was wrong), every OTHER cluster on the same (source,
# column) whose anchor cell value matches should also get FP'd. No
# corrected_value is involved - "this value is fine" is a global call.
# ---------------------------------------------------------------------


def test_cluster_propagate_fp_marks_sibling_clusters(tmp_path: Path):
    """Two clusters on column ``contractor`` both flag value
    ``AcmeCo``. Marking the row=1 cluster FP and asking for
    propagation must FP the row=5 cluster too."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Seed verdict on cluster c000 (row 1).
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "contracts.csv__r00001-c000",
            "source": "contracts.csv",
            "row": 1, "column": "contractor",
            "status": "fp",
        })
        assert r.status_code == 200, r.text

        r = client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
            "source": "contracts.csv",
            "row": 1,
            "column": "contractor",
            "status": "fp",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        # Cluster c001 is at row 5 with the same value -> must propagate.
        assert body["matched"] == 1
        assert body["propagated"] == 1
        assert body["skipped_existing"] == 0
        assert body["clusters"][0]["cluster_id"] == "contracts.csv__r00005-c000"


def test_cluster_propagate_overwrites_existing_verdicts(tmp_path: Path):
    """Same-value cluster propagation now overrides prior verdicts on
    sibling clusters - the operator clicked propagation precisely
    because the value's verdict should be uniform, and silently
    leaving the old (often contradictory) label in place produced
    confusing UIs where the same value carried mixed states."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Pre-label sibling cluster c001 as TP so it has a non-skip
        # verdict already on file (by a different annotator, to also
        # confirm we don't refuse to overwrite someone else's label).
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "contracts.csv__r00005-c000",
            "source": "contracts.csv",
            "row": 5, "column": "contractor",
            "status": "tp",
            "annotator": "other",
        })
        r = client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
            "source": "contracts.csv",
            "row": 1,
            "column": "contractor",
            "status": "fp",
        })
        body = r.json()
        assert body["matched"] == 1
        assert body["propagated"] == 1
        assert body["overridden_existing"] == 1
        # Back-compat key: kept around so older callers don't KeyError.
        assert body["skipped_existing"] == 0
        # The sibling's effective verdict must now be FP (latest-wins).
        bundle = client.get(
            f"/api/runs/{run_id}/clusters/contracts.csv__r00005-c000"
        ).json()
        assert bundle["label"]["status"] == "fp"


def test_cluster_propagate_noop_when_same_status_same_annotator(tmp_path: Path):
    """If a sibling already carries the SAME status from the SAME
    annotator, propagation should be a no-op (avoids polluting the
    audit log with redundant identical records)."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Pre-label sibling with the exact same verdict the propagation
        # will try to apply.
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "contracts.csv__r00005-c000",
            "source": "contracts.csv",
            "row": 5, "column": "contractor",
            "status": "fp",
        })
        r = client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
            "source": "contracts.csv",
            "row": 1, "column": "contractor",
            "status": "fp",
        })
        body = r.json()
        # Match still counts (we found a sibling); but no record was
        # appended and no override happened.
        assert body["matched"] == 1
        assert body["propagated"] == 0
        assert body["overridden_existing"] == 0


def test_cluster_propagate_skips_seed_and_non_matching_values(tmp_path: Path):
    """Only clusters whose anchor cell value EQUALS the seed's must
    propagate. The seed cluster itself is always skipped."""
    project, run_id = _build_run(tmp_path)
    # Add a third cluster on a DIFFERENT contractor value so we can
    # confirm it's left alone.
    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    clusters.append({
        "cluster_id": "contracts.csv__r00002-c000", "source": "contracts.csv",
        "row": 2, "columns": ["contractor"],
        "row_values": {"id": "3", "contractor": "Acme Corp", "amount": "200"},
        "error_types": ["typo"], "error_type_categories": [],
        "evidence_count": 1, "detections": [],
    })
    (rd / "clusters.json").write_text(json.dumps(clusters))
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
            "source": "contracts.csv",
            "row": 1,                     # value=AcmeCo
            "column": "contractor",
            "status": "fp",
        })
        body = r.json()
        # Only c001 (row 5, AcmeCo) should match. Row 2 = "Acme Corp"
        # is filtered out; row 1 itself is the seed and skipped.
        assert body["matched"] == 1
        assert body["propagated"] == 1
        assert body["clusters"][0]["row"] == 5


def test_cluster_propagate_validates_status(tmp_path: Path):
    """FN is invalid for cluster propagation (cells, not clusters,
    carry FN). SKIP is allowed - it spreads the skip verdict to other
    clusters on the column carrying the same anchor value (separate
    from the row-cascade skip path)."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        for bad in ("fn", "garbage", ""):
            r = client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
                "source": "contracts.csv",
                "row": 1, "column": "contractor",
                "status": bad,
            })
            assert r.status_code == 400, f"expected 400 for status={bad!r}"


def test_cluster_propagate_validates_source_row_column(tmp_path: Path):
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Bad source.
        r = client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
            "source": "nope.csv", "row": 1, "column": "contractor",
            "status": "fp",
        })
        assert r.status_code == 404
        # Bad column.
        r = client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
            "source": "contracts.csv", "row": 1, "column": "nope",
            "status": "fp",
        })
        assert r.status_code == 404
        # Out-of-range row.
        r = client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
            "source": "contracts.csv", "row": 99, "column": "contractor",
            "status": "fp",
        })
        assert r.status_code == 400


def test_cluster_propagate_fp_visible_in_cell_state(tmp_path: Path):
    """After FP propagation the per-cell state for the sibling cluster's
    cell must report ``fp`` so the labeling UI repaints it."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Seed verdict on c000.
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "contracts.csv__r00001-c000",
            "source": "contracts.csv",
            "row": 1, "column": "contractor",
            "status": "fp",
        })
        client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
            "source": "contracts.csv",
            "row": 1, "column": "contractor",
            "status": "fp",
        })
        cells = client.get(
            f"/api/runs/{run_id}/sources/contracts.csv/cells"
        ).json()["cells"]
        by_key = {(c["row"], c["column"]): c for c in cells}
        # Both seed (row 1) and propagated sibling (row 5) should be FP.
        assert by_key[(1, "contractor")]["state"] == "fp"
        assert by_key[(5, "contractor")]["state"] == "fp"
        # Other AcmeCo rows (0, 3, 7) are NOT clusters - cluster
        # propagation only touches existing clusters, never creates
        # new ones. They remain absent from cell state.
        assert (0, "contractor") not in by_key
        assert (3, "contractor") not in by_key
        assert (7, "contractor") not in by_key


def test_cluster_propagate_with_partial_status(tmp_path: Path):
    """``partial`` is also valid for cluster propagation (in case the
    operator wants to mark all sibling clusters as partially correct)."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/clusters/propagate-status", json={
            "source": "contracts.csv",
            "row": 1, "column": "contractor",
            "status": "partial",
        })
        assert r.status_code == 200
        assert r.json()["propagated"] == 1
        # Sibling cluster's verdict must reflect the new status.
        bundle = client.get(
            f"/api/runs/{run_id}/clusters/contracts.csv__r00005-c000"
        ).json()
        assert bundle["label"]["status"] == "partial"

