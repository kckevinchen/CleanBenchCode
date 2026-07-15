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


def _typo_run_with_fanned_clusters(tmp_path: Path):
    """Build a run whose ``clusters.json`` carries the fanned output of
    one typo issue: a single bad value ("Partpaymnt") repeated in one
    column across rows 0, 2, 4 - one primary cluster plus two inferred
    siblings, exactly as ``cluster_issues`` emits. A second distinct
    value gives a second, independent group."""
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv_path = data_dir / "demo.csv"
    csv_path.write_text(
        "id,Status\n"
        "1,Partpaymnt\n"     # row 0 - bad value A
        "2,Paid\n"            # row 1 - clean
        "3,Partpaymnt\n"     # row 2 - bad value A
        "4,Refundd\n"        # row 3 - bad value B
        "5,Partpaymnt\n",    # row 4 - bad value A
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_typo"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv_path)]}})
    )

    def _cl(cid, row, inferred):
        return {
            "cluster_id": cid, "source": "demo.csv", "row": row,
            "columns": ["Status"], "row_values": {"id": str(row + 1)},
            "error_types": ["typo"], "error_type_categories": ["typo"],
            "evidence_count": 1, "inferred": inferred,
            "detections": [{
                "error_type": "typo", "observed": "Partpaymnt",
                "suggested_fix": "Part payment", "columns": ["Status"],
                "affected_rows": [0, 2, 4],
            }],
        }

    (rd / "clusters.json").write_text(json.dumps([
        _cl("demo.csv__r00000-c000", 0, False),   # primary (value A)
        _cl("demo.csv__r00002-c000", 2, True),    # inferred sibling (A)
        _cl("demo.csv__r00004-c000", 4, True),    # inferred sibling (A)
        {  # independent group: value B on row 3
            "cluster_id": "demo.csv__r00003-c000", "source": "demo.csv",
            "row": 3, "columns": ["Status"], "row_values": {"id": "4"},
            "error_types": ["typo"], "error_type_categories": ["typo"],
            "evidence_count": 1, "inferred": False,
            "detections": [{
                "error_type": "typo", "observed": "Refundd",
                "suggested_fix": "Refund", "columns": ["Status"],
                "affected_rows": [3],
            }],
        },
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_fanned_same_value_clusters_group_into_one_item(tmp_path: Path):
    """Three fanned clusters on the same (source, column, value) must
    collapse into ONE labellable queue item that spans every row, while
    a distinct value remains its own item."""
    project, run_id = _typo_run_with_fanned_clusters(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        body = client.get(f"/api/runs/{run_id}/clusters").json()
        # 3 fanned (value A) -> 1 item; value B -> 1 item.
        assert body["total"] == 2, body
        by_id = {i["cluster_id"]: i for i in body["items"]}
        # The primary (non-inferred) cluster anchors the group.
        rep = by_id["demo.csv__r00000-c000"]
        assert rep["group_size"] == 3
        assert rep["group_rows"] == [0, 2, 4]
        assert rep["affected_rows"] == [0, 2, 4]
        assert sorted(rep["group_member_ids"]) == [
            "demo.csv__r00000-c000",
            "demo.csv__r00002-c000",
            "demo.csv__r00004-c000",
        ]
        # The distinct value stays a separate single-row item (no group
        # fields are added for single-member groups).
        other = by_id["demo.csv__r00003-c000"]
        assert other.get("group_size", 1) == 1
        assert other.get("affected_rows") in ([], [3], None)


def test_grouped_item_label_status_reflects_labelled_member(tmp_path: Path):
    """Labelling ANY member of the group makes the collapsed item read
    as labelled (count-once semantics)."""
    project, run_id = _typo_run_with_fanned_clusters(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Label one inferred sibling tp; the group should surface tp.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "demo.csv__r00002-c000",
            "source": "demo.csv", "row": 2, "column": "Status",
            "columns": ["Status"], "status": "tp", "annotator": "t",
        })
        assert r.status_code in (200, 201), r.text
        body = client.get(f"/api/runs/{run_id}/clusters").json()
        assert body["total"] == 2
        rep = next(
            i for i in body["items"]
            if i["group_size"] == 3 if i.get("group_size")
        )
        assert rep["label_status"] == "tp"


def test_stats_counts_grouped_issues_once(tmp_path: Path):
    """The /stats bar must count PER ISSUE: the 3 fanned clusters on one
    value collapse to a single tally, so total == grouped item count and
    labelling one member counts the group once (not three times)."""
    project, run_id = _typo_run_with_fanned_clusters(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        stats = client.get(f"/api/runs/{run_id}/stats").json()
        # 3 fanned (value A) -> 1 + value B -> 1 == 2 issues, not 4.
        assert stats["overall"]["total"] == 2, stats["overall"]
        assert stats["overall"]["unlabeled"] == 2, stats["overall"]

        # Label one member of the 3-fanned group tp; the whole group is
        # counted once as tp, leaving exactly one unlabeled issue.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "demo.csv__r00002-c000",
            "source": "demo.csv", "row": 2, "column": "Status",
            "columns": ["Status"], "status": "tp", "annotator": "t",
        })
        assert r.status_code in (200, 201), r.text
        stats = client.get(f"/api/runs/{run_id}/stats").json()
        assert stats["overall"]["total"] == 2, stats["overall"]
        assert stats["overall"]["tp"] == 1, stats["overall"]
        assert stats["overall"]["unlabeled"] == 1, stats["overall"]


def test_duplicate_run_clones_into_new_run(tmp_path: Path):
    """POST /duplicate copies the run dir under a fresh, unique run-id,
    carries labels by default, unstars the clone, and names it
    'Copy of <name>'."""
    project, run_id = _typo_run_with_fanned_clusters(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Label one cluster so we can assert labels are carried over.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "demo.csv__r00000-c000",
            "source": "demo.csv", "row": 0, "column": "Status",
            "columns": ["Status"], "status": "tp", "annotator": "t",
        })
        assert r.status_code in (200, 201), r.text

        res = client.post(f"/api/runs/{run_id}/duplicate", json={})
        assert res.status_code == 200, res.text
        body = res.json()
        new_id = body["run_id"]
        assert new_id != run_id
        assert new_id.endswith("_typo")
        assert body["source_run_id"] == run_id
        assert body["run"]["display_name"].startswith("Copy of ")
        assert body["run"]["starred"] is False

        # The clone shows up in the run list and keeps the label.
        runs = {x["run_id"]: x for x in client.get("/api/runs").json()["runs"]}
        assert new_id in runs
        clone_stats = client.get(f"/api/runs/{new_id}/stats").json()
        assert clone_stats["overall"]["tp"] == 1, clone_stats["overall"]


def test_duplicate_run_reset_labels_drops_labels(tmp_path: Path):
    """reset_labels=true gives the clone a clean labelling slate."""
    project, run_id = _typo_run_with_fanned_clusters(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster",
            "target_id": "demo.csv__r00000-c000",
            "source": "demo.csv", "row": 0, "column": "Status",
            "columns": ["Status"], "status": "tp", "annotator": "t",
        })
        res = client.post(
            f"/api/runs/{run_id}/duplicate", json={"reset_labels": True}
        )
        assert res.status_code == 200, res.text
        new_id = res.json()["run_id"]
        clone_stats = client.get(f"/api/runs/{new_id}/stats").json()
        assert clone_stats["overall"]["tp"] == 0, clone_stats["overall"]
        # Original is untouched.
        orig_stats = client.get(f"/api/runs/{run_id}/stats").json()
        assert orig_stats["overall"]["tp"] == 1, orig_stats["overall"]
