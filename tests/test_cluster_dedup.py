"""Cluster-queue dedup: the labeling list endpoint must collapse
clusters that resolve to the SAME canonical (source, row, columns)
cell — even when ``clusters.json`` was persisted with raw, differently-
mangled column names.

This guards against the regression where the LLM returned the same
multi-line header in different shortened forms, the aggregator
clustered on raw strings (so they didn't merge), and the labeling UI
- which canonicalizes at serve time - showed the same cell as two
separate queue items.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core.columns import canonicalize_columns
from src.labeling.app import create_app


def test_canonicalize_columns_resolves_multiline_headers():
    headers = ["County\nCode", "County\nTotal", "Withdrawal\nby parent"]
    # All three of these should resolve to "Withdrawal\nby parent".
    assert canonicalize_columns(["Withdrawal"], headers) == ["Withdrawal\nby parent"]
    assert canonicalize_columns(["withdrawal by parent"], headers) == ["Withdrawal\nby parent"]
    assert canonicalize_columns(["by parent"], headers) == ["Withdrawal\nby parent"]
    # Exact match passes through.
    assert canonicalize_columns(["County\nCode"], headers) == ["County\nCode"]
    # Unmatched gets dropped (keeps the labeling UI from silently failing).
    assert canonicalize_columns(["NoSuchColumn"], headers) == []


@pytest.fixture
def run_with_duplicate_clusters(tmp_path: Path):
    """Build a minimal run where ``clusters.json`` carries two clusters
    flagging the same cell via differently-mangled column names. The
    cluster list endpoint should return only one item."""
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)

    data_dir = project / "data"
    data_dir.mkdir()
    csv_path = data_dir / "demo.csv"
    # Header has an embedded newline (a real-world LLM-mangling source).
    csv_path.write_text(
        'id,"Withdrawal\nby parent",notes\n'
        '1,yes,first\n'
        '2,no,second\n',
        encoding="utf-8",
    )

    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_dup"
    rd = outputs / run_id
    rd.mkdir()

    summary = {"config": {"csv_paths": [str(csv_path)]}}
    (rd / "summary.json").write_text(json.dumps(summary))

    # Two clusters on the same source / row / cell, but with
    # differently-mangled column strings. Without dedup these would
    # show as two queue items pointing at the same cell.
    clusters = [
        {
            "cluster_id": "demo.csv__r00000-c000",
            "source": "demo.csv",
            "row": 0,
            "columns": ["Withdrawal"],
            "row_values": {"id": "1"},
            "error_types": ["scmr"],
            "error_type_categories": [],
            "evidence_count": 2,
            "detections": [],
        },
        {
            "cluster_id": "demo.csv__r00000-c001",
            "source": "demo.csv",
            "row": 0,
            "columns": ["by parent"],
            "row_values": {"id": "1"},
            "error_types": ["scmr"],
            "error_type_categories": [],
            "evidence_count": 1,
            "detections": [],
        },
    ]
    (rd / "clusters.json").write_text(json.dumps(clusters))
    (rd / "issues.json").write_text("[]")

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        yield client, run_id


def test_queue_collapses_canonical_duplicates(run_with_duplicate_clusters):
    client, run_id = run_with_duplicate_clusters
    r = client.get(f"/api/runs/{run_id}/clusters")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1, f"expected 1 deduped item, got {body['total']}"
    item = body["items"][0]
    # The kept primary should be the higher-evidence cluster (c000) and
    # carry the absorbed sibling's id in ``merged_cluster_ids``.
    assert item["cluster_id"] == "demo.csv__r00000-c000"
    assert item["merged_cluster_ids"] == ["demo.csv__r00000-c001"]
    # Both clusters resolved their column to the canonical form.
    assert item["columns"] == ["Withdrawal\nby parent"]


def test_absorbed_cluster_is_still_reachable_by_id(run_with_duplicate_clusters):
    """Direct ``/clusters/{cid}`` lookup must still return the absorbed
    sibling so any labels stored against its id remain visible."""
    client, run_id = run_with_duplicate_clusters
    r = client.get(f"/api/runs/{run_id}/clusters/demo.csv__r00000-c001")
    assert r.status_code == 200, r.text
    assert r.json()["cluster"]["cluster_id"] == "demo.csv__r00000-c001"


def test_labelled_cluster_wins_over_unlabelled(tmp_path: Path):
    """If the LOWER-evidence cluster has been labelled, it must remain
    the primary so the annotator's prior work is preserved."""
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv_path = data_dir / "demo.csv"
    csv_path.write_text(
        'id,"Withdrawal\nby parent",notes\n1,yes,x\n',
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_dup2"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv_path)]}})
    )
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "demo.csv__r00000-c000", "source": "demo.csv",
         "row": 0, "columns": ["Withdrawal"], "row_values": {"id": "1"},
         "error_types": ["scmr"], "error_type_categories": [],
         "evidence_count": 5, "detections": []},
        {"cluster_id": "demo.csv__r00000-c001", "source": "demo.csv",
         "row": 0, "columns": ["by parent"], "row_values": {"id": "1"},
         "error_types": ["scmr"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Label the LOWER-evidence cluster.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "demo.csv__r00000-c001",
            "source": "demo.csv",
            "status": "tp",
            "annotator": "t",
        })
        assert r.status_code in (200, 201), r.text

        body = client.get(f"/api/runs/{run_id}/clusters").json()
        assert body["total"] == 1
        item = body["items"][0]
        # The labelled cluster must remain the primary, even though it
        # has fewer detections.
        assert item["cluster_id"] == "demo.csv__r00000-c001"
        assert item["label_status"] == "tp"


def test_clusters_without_columns_are_not_deduped(tmp_path: Path):
    """Row-less or column-less defects (dataset-wide flags) are
    legitimately separate clusters and must NOT be collapsed."""
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv_path = data_dir / "demo.csv"
    csv_path.write_text("id,name\n1,a\n2,b\n", encoding="utf-8")
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_dup3"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv_path)]}})
    )
    # Two row-less, column-less clusters - both should survive.
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "demo.csv__rNA-c0000", "source": "demo.csv",
         "row": None, "columns": [], "row_values": {},
         "error_types": ["dup_rows"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
        {"cluster_id": "demo.csv__rNA-c0001", "source": "demo.csv",
         "row": None, "columns": [], "row_values": {},
         "error_types": ["scmr"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        body = client.get(f"/api/runs/{run_id}/clusters").json()
        assert body["total"] == 2
