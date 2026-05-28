"""Tests for the column-clusters labelling endpoints in
``src.labeling.app``.

Slice 3 of the column-clusters labelling subsystem. Covers:
  * ``GET /api/runs/{run_id}/inconsistency-columns``
  * ``GET /api/runs/{run_id}/columns/{source}/{column}/values``
  * ``GET /api/runs/{run_id}/columns/{source}/{column}/clusters``
  * ``POST /api/runs/{run_id}/columns/{source}/{column}/clusters``
  * ``GET /api/runs/{run_id}/column-cluster-stats``
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.schema import ColumnLabelStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SMOKE_RUN_ID = "20260101T000000Z_cctest"


def _build_smoke_csv(path: Path) -> None:
    rows = [
        {"School": "BBB",    "State": "CA"},
        {"School": "BBB",    "State": "CA"},
        {"School": "BBB",    "State": "CA"},
        {"School": "BBB",    "State": "CA"},
        {"School": "ABC HS", "State": "CA"},
        {"School": "ABC HS", "State": "CA"},
        {"School": "EEE",    "State": "Calif."},
        {"School": "EEE",    "State": "Calif."},
        {"School": "DDD",    "State": "Cali"},
        {"School": "BBB",    "State": "CA"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_run(project: Path, run_id: str = SMOKE_RUN_ID) -> Path:
    data_dir = project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _build_smoke_csv(data_dir / "smoke.csv")
    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    run_dir = outputs / run_id
    run_dir.mkdir()
    summary = {
        "config": {
            "csv_paths": [str(data_dir / "smoke.csv")],
            "sources": ["smoke.csv"],
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")
    return run_dir


@pytest.fixture
def client_project(tmp_path: Path):
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    run_dir = _make_run(project)
    app = create_app(project, annotator="kaiwen")
    with TestClient(app) as client:
        yield client, project, run_dir.name


# ---------------------------------------------------------------------------
# /inconsistency-columns
# ---------------------------------------------------------------------------


class TestInconsistencyColumns:
    def test_lists_smoke_columns(self, client_project) -> None:
        client, _, run_id = client_project
        r = client.get(f"/api/runs/{run_id}/inconsistency-columns")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 2
        cols = {(i["source"], i["column"]) for i in body["items"]}
        assert cols == {("smoke.csv", "School"), ("smoke.csv", "State")}
        for item in body["items"]:
            assert item["status"] == "unlabeled"
            assert item["n_clusters"] == 0
            assert item["n_rows"] == 10
            assert item["n_distinct"] >= 2

    def test_status_promoted_after_label(self, client_project) -> None:
        client, _, run_id = client_project
        # Post a labelled record for smoke.csv / School.
        payload = {
            "source": "smoke.csv",
            "column": "School",
            "status": "labeled",
            "clusters": [
                {
                    "cluster_id": "c1",
                    "kind": "text",
                    "canonical_value": "BBB",
                    "values": [
                        {"value": "BBB", "count": 5},
                        {"value": "ABC HS", "count": 2},
                    ],
                },
            ],
        }
        r = client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json=payload,
        )
        assert r.status_code == 200, r.text
        # Listing now shows labeled status and 1 cluster.
        r = client.get(f"/api/runs/{run_id}/inconsistency-columns")
        assert r.status_code == 200
        body = r.json()
        labelled = [i for i in body["items"] if i["column"] == "School"]
        assert len(labelled) == 1
        assert labelled[0]["status"] == "labeled"
        assert labelled[0]["n_clusters"] == 1
        assert labelled[0]["annotator"] == "kaiwen"

    def test_cardinality_filter(self, client_project, tmp_path: Path) -> None:
        client, project, run_id = client_project
        # Add a high-cardinality column to verify the filter.
        path = project / "data" / "smoke.csv"
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df["UniqueId"] = [f"id-{i}" for i in range(len(df))]
        df.to_csv(path, index=False)
        # max_distinct_ratio=0.5 keeps School (4/10) but drops UniqueId (10/10).
        r = client.get(
            f"/api/runs/{run_id}/inconsistency-columns",
            params={"max_distinct_ratio": 0.5},
        )
        assert r.status_code == 200, r.text
        cols = {i["column"] for i in r.json()["items"]}
        assert "UniqueId" not in cols
        assert "School" in cols

    def test_min_distinct_filter(self, client_project, tmp_path: Path) -> None:
        client, project, run_id = client_project
        # Add a single-value column.
        path = project / "data" / "smoke.csv"
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df["AllSame"] = "Y"
        df.to_csv(path, index=False)
        r = client.get(
            f"/api/runs/{run_id}/inconsistency-columns",
            params={"min_distinct": 2},
        )
        cols = {i["column"] for i in r.json()["items"]}
        assert "AllSame" not in cols


# ---------------------------------------------------------------------------
# /columns/{source}/{column}/values
# ---------------------------------------------------------------------------


class TestColumnValues:
    def test_returns_value_counts(self, client_project) -> None:
        client, _, run_id = client_project
        r = client.get(
            f"/api/runs/{run_id}/columns/smoke.csv/School/values"
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "smoke.csv"
        assert body["column"] == "School"
        assert body["n_rows"] == 10
        assert body["n_empty"] == 0
        assert body["n_distinct"] == 4
        values = {v["value"]: v["count"] for v in body["values"]}
        assert values == {"BBB": 5, "ABC HS": 2, "EEE": 2, "DDD": 1}
        # Sort: descending count, ties alphabetical (ABC HS before EEE).
        ordered = [v["value"] for v in body["values"]]
        assert ordered == ["BBB", "ABC HS", "EEE", "DDD"]

    def test_missing_source_404(self, client_project) -> None:
        client, _, run_id = client_project
        r = client.get(
            f"/api/runs/{run_id}/columns/nope.csv/School/values"
        )
        assert r.status_code == 404

    def test_missing_column_404(self, client_project) -> None:
        client, _, run_id = client_project
        r = client.get(
            f"/api/runs/{run_id}/columns/smoke.csv/NoSuchCol/values"
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET/POST /columns/{source}/{column}/clusters
# ---------------------------------------------------------------------------


class TestPostColumnClusters:
    def _valid_payload(self) -> dict:
        return {
            "source": "smoke.csv",
            "column": "School",
            "status": "labeled",
            "clusters": [
                {
                    "cluster_id": "c1",
                    "kind": "text",
                    "canonical_value": "BBB",
                    "values": [
                        {"value": "BBB", "count": 5},
                        {"value": "ABC HS", "count": 2},
                    ],
                },
                {
                    "cluster_id": "c2",
                    "kind": "untagged",
                    "values": [
                        {"value": "EEE", "count": 2},
                        {"value": "DDD", "count": 1},
                    ],
                },
            ],
            "note": "first pass",
        }

    def test_post_and_get_round_trip(self, client_project) -> None:
        client, _, run_id = client_project
        r = client.get(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters"
        )
        assert r.status_code == 200
        assert r.json()["record"] is None
        # POST.
        r = client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json=self._valid_payload(),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        rec = body["record"]
        assert rec["source"] == "smoke.csv"
        assert rec["column"] == "School"
        assert rec["status"] == "labeled"
        assert rec["annotator"] == "kaiwen"
        assert len(rec["clusters"]) == 2
        # Subsequent GET sees the record.
        r = client.get(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["record"] is not None
        assert body["record"]["status"] == "labeled"
        assert len(body["history"]) == 1

    def test_latest_wins_history_grows(self, client_project) -> None:
        client, _, run_id = client_project
        client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json=self._valid_payload(),
        )
        # Second write with a different note.
        p2 = self._valid_payload()
        p2["note"] = "second pass"
        r = client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json=p2,
        )
        assert r.status_code == 200, r.text
        r = client.get(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters"
        )
        body = r.json()
        assert body["record"]["note"] == "second pass"
        assert len(body["history"]) == 2

    def test_unknown_value_rejected(self, client_project) -> None:
        client, _, run_id = client_project
        payload = self._valid_payload()
        # Insert a value that's not in the column.
        payload["clusters"][0]["values"].append(
            {"value": "GHOST", "count": 1}
        )
        r = client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json=payload,
        )
        assert r.status_code == 400, r.text
        assert "GHOST" in r.text

    def test_path_overrides_payload_source_column(self, client_project) -> None:
        """If frontend sends mismatched source/column in body, path wins."""
        client, _, run_id = client_project
        payload = self._valid_payload()
        payload["source"] = "x.csv"        # mismatch
        payload["column"] = "Y"            # mismatch
        r = client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json=payload,
        )
        assert r.status_code == 200, r.text
        rec = r.json()["record"]
        assert rec["source"] == "smoke.csv"
        assert rec["column"] == "School"

    def test_invariant_violation_rejected_by_pydantic(self, client_project) -> None:
        """A single-value cluster fails Pydantic validation -> 422."""
        client, _, run_id = client_project
        payload = self._valid_payload()
        payload["clusters"][0]["values"] = [{"value": "BBB", "count": 5}]
        r = client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json=payload,
        )
        assert r.status_code == 422, r.text

    def test_no_inconsistency_status(self, client_project) -> None:
        client, _, run_id = client_project
        payload = {
            "source": "smoke.csv",
            "column": "School",
            "status": "no_inconsistency",
            "clusters": [],
            "note": "verified clean",
        }
        r = client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json=payload,
        )
        assert r.status_code == 200, r.text
        assert r.json()["record"]["status"] == "no_inconsistency"

    def test_missing_source_404(self, client_project) -> None:
        client, _, run_id = client_project
        r = client.post(
            f"/api/runs/{run_id}/columns/nope.csv/School/clusters",
            json=self._valid_payload(),
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# /column-cluster-stats
# ---------------------------------------------------------------------------


class TestColumnClusterStats:
    def test_empty_run(self, client_project) -> None:
        client, _, run_id = client_project
        r = client.get(f"/api/runs/{run_id}/column-cluster-stats")
        assert r.status_code == 200
        body = r.json()
        assert body["labelled_columns"] == 0
        assert body["by_status"] == {}
        assert body["n_clusters_total"] == 0

    def test_counts_after_label(self, client_project) -> None:
        client, _, run_id = client_project
        client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json={
                "source": "smoke.csv", "column": "School",
                "status": "labeled",
                "clusters": [{
                    "cluster_id": "c1",
                    "values": [
                        {"value": "BBB", "count": 5},
                        {"value": "ABC HS", "count": 2},
                    ],
                }],
            },
        )
        client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/State/clusters",
            json={
                "source": "smoke.csv", "column": "State",
                "status": "no_inconsistency",
                "clusters": [],
            },
        )
        r = client.get(f"/api/runs/{run_id}/column-cluster-stats")
        body = r.json()
        assert body["labelled_columns"] == 2
        assert body["by_status"] == {"labeled": 1, "no_inconsistency": 1}
        assert body["n_clusters_total"] == 1


# ---------------------------------------------------------------------------
# /column-clusters (queue items + label flow)
# ---------------------------------------------------------------------------


def _post_column_clusters(client, run_id: str) -> None:
    """Helper: save two clusters on ``smoke.csv/School`` so the
    list-clusters endpoint has something to surface."""
    client.post(
        f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
        json={
            "source": "smoke.csv", "column": "School",
            "status": "labeled",
            "clusters": [
                {
                    "cluster_id": "c1",
                    "kind": "text",
                    "canonical_value": "BBB",
                    "values": [
                        {"value": "BBB", "count": 5},
                        {"value": "ABC HS", "count": 2},
                    ],
                },
                {
                    "cluster_id": "c2",
                    "kind": "casing",
                    "canonical_value": "EEE",
                    "values": [
                        {"value": "EEE", "count": 2},
                        {"value": "DDD", "count": 1},
                    ],
                },
            ],
        },
    ).raise_for_status()


class TestListColumnClusters:
    def test_empty_when_no_records(self, client_project) -> None:
        client, _, run_id = client_project
        r = client.get(f"/api/runs/{run_id}/column-clusters")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_lists_saved_clusters(self, client_project) -> None:
        client, _, run_id = client_project
        _post_column_clusters(client, run_id)
        r = client.get(f"/api/runs/{run_id}/column-clusters")
        body = r.json()
        assert body["total"] == 2
        items = body["items"]
        assert {it["cluster_local_id"] for it in items} == {"c1", "c2"}
        # Both clusters expose the full value list + canonical so
        # the frontend can paint the CSV without an extra fetch.
        by_id = {it["cluster_local_id"]: it for it in items}
        assert by_id["c1"]["canonical_value"] == "BBB"
        assert by_id["c1"]["kind"] == "text"
        assert {v["value"] for v in by_id["c1"]["values"]} == {"BBB", "ABC HS"}
        # All items default to ``unlabeled`` until a verdict is posted.
        assert all(it["label_status"] == "unlabeled" for it in items)
        # target_id uses NUL separators (canonical column_cluster_target_id).
        assert by_id["c1"]["target_id"] == "colclust\x00smoke.csv\x00School\x00c1"
        # ``is_col_cluster`` brand surfaces so the queue merge in the
        # frontend can identify these items without extra heuristics.
        assert all(it["is_col_cluster"] is True for it in items)
        assert all(it["error_types"] == ["inconsistency"] for it in items)

    def test_filter_by_source(self, client_project) -> None:
        client, _, run_id = client_project
        _post_column_clusters(client, run_id)
        r = client.get(f"/api/runs/{run_id}/column-clusters?source=other.csv")
        assert r.json()["total"] == 0
        r = client.get(f"/api/runs/{run_id}/column-clusters?source=smoke.csv")
        assert r.json()["total"] == 2

    def test_no_inconsistency_columns_excluded(self, client_project) -> None:
        """Columns whose record has status != labeled don't surface
        any clusters (they have none anyway, but defensive)."""
        client, _, run_id = client_project
        client.post(
            f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
            json={
                "source": "smoke.csv", "column": "School",
                "status": "no_inconsistency",
                "clusters": [],
            },
        ).raise_for_status()
        r = client.get(f"/api/runs/{run_id}/column-clusters")
        assert r.json()["total"] == 0


class TestColumnClusterLabel:
    """The /labels POST accepts target=column_cluster and joins the
    verdict back into /column-clusters' ``label_status`` field."""

    def test_post_label_and_join(self, client_project) -> None:
        client, _, run_id = client_project
        _post_column_clusters(client, run_id)
        tid = "colclust\x00smoke.csv\x00School\x00c1"
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster",
                "target_id": tid,
                "source": "smoke.csv",
                "column": "School",
                "status": "tp",
                "note": "real inconsistency",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "tp"
        # The endpoint canonicalises target_id from (source, column,
        # local_id) parts so two clients can't drift on key format.
        assert body["target_id"] == tid
        # The verdict surfaces back through /column-clusters.
        listing = client.get(f"/api/runs/{run_id}/column-clusters").json()
        statuses = {it["cluster_local_id"]: it["label_status"] for it in listing["items"]}
        assert statuses["c1"] == "tp"
        assert statuses["c2"] == "unlabeled"

    def test_post_label_rejects_unknown_cluster(self, client_project) -> None:
        client, _, run_id = client_project
        _post_column_clusters(client, run_id)
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster",
                "target_id": "colclust\x00smoke.csv\x00School\x00cZZ",
                "source": "smoke.csv",
                "column": "School",
                "status": "tp",
            },
        )
        assert r.status_code == 404

    def test_post_label_rejects_missing_record(self, client_project) -> None:
        client, _, run_id = client_project
        # No column_clusters record exists yet.
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster",
                "target_id": "colclust\x00smoke.csv\x00School\x00c1",
                "source": "smoke.csv",
                "column": "School",
                "status": "tp",
            },
        )
        assert r.status_code == 404

    def test_label_status_filter(self, client_project) -> None:
        client, _, run_id = client_project
        _post_column_clusters(client, run_id)
        tid = "colclust\x00smoke.csv\x00School\x00c1"
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster", "target_id": tid,
                "source": "smoke.csv", "column": "School", "status": "fp",
            },
        ).raise_for_status()
        r = client.get(f"/api/runs/{run_id}/column-clusters?status=fp")
        items = r.json()["items"]
        assert len(items) == 1
        assert items[0]["cluster_local_id"] == "c1"
        r = client.get(f"/api/runs/{run_id}/column-clusters?status=unlabeled")
        items = r.json()["items"]
        assert len(items) == 1
        assert items[0]["cluster_local_id"] == "c2"

    def test_post_label_accepts_confirmed_canonical(
        self, client_project,
    ) -> None:
        """Operator endorses one of the cluster's surface forms as
        the canonical via ``confirmed_canonical``. The verdict
        round-trips through /column-clusters as
        ``label_confirmed_canonical``."""
        client, _, run_id = client_project
        _post_column_clusters(client, run_id)
        tid = "colclust\x00smoke.csv\x00School\x00c1"
        # Pick one of c1's actual values (BBB is the canonical in
        # the fixture; ABC HS is also a member).
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster", "target_id": tid,
                "source": "smoke.csv", "column": "School",
                "status": "tp",
                "confirmed_canonical": "BBB",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "tp"
        assert body["confirmed_canonical"] == "BBB"
        # Surfaces back through the listing for UI pre-fill.
        listing = client.get(
            f"/api/runs/{run_id}/column-clusters",
        ).json()
        by_local = {it["cluster_local_id"]: it for it in listing["items"]}
        assert by_local["c1"]["label_confirmed_canonical"] == "BBB"
        # c2 (no label) has None.
        assert by_local["c2"]["label_confirmed_canonical"] is None

    def test_post_label_rejects_off_cluster_confirmed_canonical(
        self, client_project,
    ) -> None:
        """Operator endorses a value that's not in the cluster's
        ``values`` -- server rejects with 400 (the synthesised
        ``ColumnClusters`` record's schema would later reject the
        same canonical, so it's better to fail at write time)."""
        client, _, run_id = client_project
        _post_column_clusters(client, run_id)
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster",
                "target_id": "colclust\x00smoke.csv\x00School\x00c1",
                "source": "smoke.csv", "column": "School",
                "status": "tp",
                # Not a value of c1 (c1 = {BBB, ABC HS}).
                "confirmed_canonical": "California",
            },
        )
        assert r.status_code == 400, r.text
        assert "confirmed_canonical" in r.text

    def test_post_label_clears_confirmed_canonical_with_null(
        self, client_project,
    ) -> None:
        """Operator changes their mind: re-POST with
        ``confirmed_canonical=null`` clears a previous endorsement.
        Latest-line-wins semantics."""
        client, _, run_id = client_project
        _post_column_clusters(client, run_id)
        tid = "colclust\x00smoke.csv\x00School\x00c1"
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster", "target_id": tid,
                "source": "smoke.csv", "column": "School",
                "status": "tp", "confirmed_canonical": "BBB",
            },
        ).raise_for_status()
        # Re-POST without the field => server defaults to None.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster", "target_id": tid,
                "source": "smoke.csv", "column": "School",
                "status": "tp",
                "confirmed_canonical": None,
            },
        ).raise_for_status()
        listing = client.get(
            f"/api/runs/{run_id}/column-clusters",
        ).json()
        by_local = {it["cluster_local_id"]: it for it in listing["items"]}
        assert by_local["c1"]["label_confirmed_canonical"] is None


# ---------------------------------------------------------------------------
# /column-clusters + post_label draft surfacing & auto-promotion
# ---------------------------------------------------------------------------


def _write_drafts(run_dir: Path) -> None:
    """Write a 2-cluster draft for ``smoke.csv/School`` so the
    /column-clusters endpoint has unsaved drafts to surface.

    Matches the production format written by
    ``experiments.run_column_clustering`` (``values`` is a list of
    raw strings, not ``{value, count}`` dicts -- the count is
    backfilled on the read path).
    """
    drafts = {
        "source": "smoke.csv",
        "column": "School",
        "draft_clusters": [
            {
                "kind": "text",
                "canonical": "BBB",
                "agreement": 0.85,
                "values": ["BBB", "ABC HS"],
                # type-2 hint -- exercises the llm_target_style
                # passthrough in /column-clusters and the column-
                # board JS join-by-value-set.
                "target_style": "spelled-out form, as in 'Tipton High School'",
            },
            {
                "kind": "casing",
                "canonical": "EEE",
                "agreement": 0.66,
                "values": ["EEE", "DDD"],
                # type-1 alias group -- ``target_style`` intentionally
                # absent so the API exposes ``None`` for this draft.
            },
        ],
    }
    with (run_dir / "column_cluster_drafts.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(drafts) + "\n")


class TestDraftSurfacing:
    def test_drafts_surface_with_no_record(self, tmp_path: Path) -> None:
        project = tmp_path / "ws"
        project.mkdir()
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        _write_drafts(run_dir)
        app = create_app(project, annotator="kaiwen")
        with TestClient(app) as client:
            r = client.get(f"/api/runs/{run_dir.name}/column-clusters")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["total"] == 2
            items = sorted(body["items"], key=lambda it: it["cluster_local_id"])
            assert items[0]["cluster_local_id"] == "c1"
            assert items[0]["draft_pending"] is True
            assert items[0]["from_draft"] is True
            assert items[0]["label_status"] == "unlabeled"
            assert items[0]["llm_canonical"] == "BBB"
            assert items[0]["llm_agreement"] == 0.85
            assert items[0]["llm_target_style"] == (
                "spelled-out form, as in 'Tipton High School'"
            )
            assert items[0]["kind"] == "text"
            assert items[1]["cluster_local_id"] == "c2"
            assert items[1]["kind"] == "casing"
            # Type-1 alias draft has no ``target_style`` -> exposed
            # as ``None`` so the column-board UI skips the hint row.
            assert items[1]["llm_target_style"] is None
            # ``target_id`` matches the canonical helper so a verdict
            # posted on the draft can be looked up after promotion.
            assert items[0]["target_id"] == (
                "colclust\x00smoke.csv\x00School\x00c1"
            )

    def test_include_drafts_false_hides_them(self, tmp_path: Path) -> None:
        project = tmp_path / "ws"
        project.mkdir()
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        _write_drafts(run_dir)
        app = create_app(project, annotator="kaiwen")
        with TestClient(app) as client:
            r = client.get(
                f"/api/runs/{run_dir.name}/column-clusters?include_drafts=false"
            )
            assert r.json()["total"] == 0

    def test_drafts_hidden_when_column_already_labelled(
        self, tmp_path: Path,
    ) -> None:
        project = tmp_path / "ws"
        project.mkdir()
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        _write_drafts(run_dir)
        app = create_app(project, annotator="kaiwen")
        with TestClient(app) as client:
            # Save a different cluster (1 cluster) - the column is now
            # labelled, so draft items should NOT surface anymore.
            client.post(
                f"/api/runs/{run_dir.name}/columns/smoke.csv/School/clusters",
                json={
                    "source": "smoke.csv", "column": "School",
                    "status": "labeled",
                    "clusters": [{
                        "cluster_id": "c1",
                        "values": [
                            {"value": "BBB", "count": 5},
                            {"value": "ABC HS", "count": 2},
                        ],
                    }],
                },
            ).raise_for_status()
            r = client.get(f"/api/runs/{run_dir.name}/column-clusters")
            items = r.json()["items"]
            # Only the saved cluster, not the drafts.
            assert len(items) == 1
            assert items[0]["draft_pending"] is False


class TestPerClusterLabelling:
    """Per-cluster labelling on LLM drafts.

    Inconsistency clusters are voted on individually: the operator
    assigns TP/FP/Partial/Unsure/Skip to one cluster at a time. The
    verdict is recorded in the ``Label`` store keyed by the synthetic
    ``target_id``; the underlying cluster definition stays in the
    draft JSONL.

    No ``ColumnClusters`` record is created by a verdict. Drafts that
    haven't been voted on stay visible in the queue so the operator
    can keep working through the column's proposals. Overlapping
    drafts (e.g. LLM emits a coarse superset + finer subsets) are
    fine: each gets its own verdict; the column board UI is the
    only path that materializes a disjoint partition.
    """

    def test_first_verdict_records_label_without_promoting(
        self, tmp_path: Path,
    ) -> None:
        """Posting a verdict on a draft cluster records the label
        but does NOT create a ``ColumnClusters`` record. The other
        drafts in the column remain visible in the queue."""
        project = tmp_path / "ws"
        project.mkdir()
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        _write_drafts(run_dir)
        app = create_app(project, annotator="kaiwen")
        with TestClient(app) as client:
            run_id = run_dir.name
            tid_c2 = "colclust\x00smoke.csv\x00School\x00c2"
            r = client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "column_cluster",
                    "target_id": tid_c2,
                    "source": "smoke.csv",
                    "column": "School",
                    "status": "tp",
                    "note": "casing - EEE",
                },
            )
            assert r.status_code == 200, r.text
            assert r.json()["target_id"] == tid_c2
            # No ColumnClusters record was created -- drafts still
            # surface as queue items, c2 carries the TP verdict via
            # the label join.
            assert not (run_dir / "column_clusters.jsonl").exists(), (
                "auto-promote should be gone; no record file should "
                "be written by a per-cluster verdict"
            )
            listing = client.get(
                f"/api/runs/{run_id}/column-clusters"
            ).json()
            by_id = {it["cluster_local_id"]: it for it in listing["items"]}
            assert set(by_id.keys()) == {"c1", "c2"}
            # Both surface as drafts: c2 has the TP verdict, c1 is
            # still unlabeled and ready for the next vote.
            assert by_id["c1"]["draft_pending"] is True
            assert by_id["c2"]["draft_pending"] is True
            assert by_id["c1"]["label_status"] == "unlabeled"
            assert by_id["c2"]["label_status"] == "tp"
            # Cluster definitions come straight from the draft JSONL.
            assert by_id["c1"]["kind"] == "text"
            assert by_id["c1"]["llm_canonical"] == "BBB"
            assert by_id["c2"]["kind"] == "casing"
            assert by_id["c2"]["llm_canonical"] == "EEE"
            # Type-2 hint is still recovered.
            assert by_id["c1"]["llm_target_style"] == (
                "spelled-out form, as in 'Tipton High School'"
            )
            assert by_id["c2"]["llm_target_style"] is None

    def test_second_verdict_lands_on_another_draft(
        self, tmp_path: Path,
    ) -> None:
        """After voting on c2 the operator can still vote on c1.
        Both verdicts land; no record materializes."""
        project = tmp_path / "ws"
        project.mkdir()
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        _write_drafts(run_dir)
        app = create_app(project, annotator="kaiwen")
        with TestClient(app) as client:
            run_id = run_dir.name
            for local_id, status_ in (("c2", "tp"), ("c1", "fp")):
                tid = f"colclust\x00smoke.csv\x00School\x00{local_id}"
                r = client.post(
                    f"/api/runs/{run_id}/labels",
                    json={
                        "target": "column_cluster",
                        "target_id": tid,
                        "source": "smoke.csv",
                        "column": "School",
                        "status": status_,
                    },
                )
                assert r.status_code == 200, r.text
            listing = client.get(
                f"/api/runs/{run_id}/column-clusters"
            ).json()
            by_id = {it["cluster_local_id"]: it for it in listing["items"]}
            assert by_id["c1"]["label_status"] == "fp"
            assert by_id["c2"]["label_status"] == "tp"
            assert not (run_dir / "column_clusters.jsonl").exists()

    def test_overlapping_drafts_each_get_independent_verdict(
        self, tmp_path: Path,
    ) -> None:
        """Regression for the 500 the operator hit when the LLM
        emitted overlapping drafts (a superset cluster + two subset
        clusters). Each draft is voted on independently; nothing is
        dropped from the queue."""
        project = tmp_path / "ws"
        project.mkdir()
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        _write_overlapping_drafts(run_dir)
        app = create_app(project, annotator="kaiwen")
        with TestClient(app) as client:
            run_id = run_dir.name
            # Operator votes TP on c1 (the superset).
            tid_c1 = "colclust\x00smoke.csv\x00School\x00c1"
            r = client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "column_cluster",
                    "target_id": tid_c1,
                    "source": "smoke.csv",
                    "column": "School",
                    "status": "tp",
                },
            )
            assert r.status_code == 200, r.text
            # c2 and c3 (the subsets) remain available for labelling.
            listing = client.get(
                f"/api/runs/{run_id}/column-clusters"
            ).json()
            by_id = {it["cluster_local_id"]: it for it in listing["items"]}
            assert set(by_id.keys()) == {"c1", "c2", "c3"}
            assert by_id["c1"]["label_status"] == "tp"
            assert by_id["c2"]["label_status"] == "unlabeled"
            assert by_id["c3"]["label_status"] == "unlabeled"
            # Vote FP on c2 -- still no record materialized.
            tid_c2 = "colclust\x00smoke.csv\x00School\x00c2"
            r = client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "column_cluster",
                    "target_id": tid_c2,
                    "source": "smoke.csv",
                    "column": "School",
                    "status": "fp",
                },
            )
            assert r.status_code == 200, r.text
            listing = client.get(
                f"/api/runs/{run_id}/column-clusters"
            ).json()
            by_id = {it["cluster_local_id"]: it for it in listing["items"]}
            assert by_id["c1"]["label_status"] == "tp"
            assert by_id["c2"]["label_status"] == "fp"
            assert by_id["c3"]["label_status"] == "unlabeled"
            assert not (run_dir / "column_clusters.jsonl").exists()

    def test_unknown_cluster_id_404s(self, tmp_path: Path) -> None:
        """A target_id that doesn't match any draft (or saved record)
        returns 404 rather than silently accepting a phantom verdict."""
        project = tmp_path / "ws"
        project.mkdir()
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        _write_drafts(run_dir)
        app = create_app(project, annotator="kaiwen")
        with TestClient(app) as client:
            run_id = run_dir.name
            r = client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "column_cluster",
                    "target_id": "colclust\x00smoke.csv\x00School\x00c99",
                    "source": "smoke.csv",
                    "column": "School",
                    "status": "tp",
                },
            )
            assert r.status_code == 404

    def test_saved_record_still_accepts_verdicts(
        self, tmp_path: Path,
    ) -> None:
        """When the column board has saved a ``ColumnClusters`` record
        (the disjoint-partition path), verdicts on its cluster_ids
        still go through. This exercises the ``in_record`` branch of
        the validator."""
        project = tmp_path / "ws"
        project.mkdir()
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        # No drafts; just save a clustering through the board endpoint.
        app = create_app(project, annotator="kaiwen")
        with TestClient(app) as client:
            run_id = run_dir.name
            save = client.post(
                f"/api/runs/{run_id}/columns/smoke.csv/School/clusters",
                json={
                    "source": "smoke.csv", "column": "School",
                    "status": "labeled",
                    "clusters": [{
                        "cluster_id": "c1",
                        "values": [
                            {"value": "BBB", "count": 5},
                            {"value": "ABC HS", "count": 2},
                        ],
                    }],
                },
            )
            save.raise_for_status()
            r = client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "column_cluster",
                    "target_id": "colclust\x00smoke.csv\x00School\x00c1",
                    "source": "smoke.csv",
                    "column": "School",
                    "status": "tp",
                },
            )
            assert r.status_code == 200, r.text
            listing = client.get(
                f"/api/runs/{run_id}/column-clusters"
            ).json()
            by_id = {it["cluster_local_id"]: it for it in listing["items"]}
            assert by_id["c1"]["label_status"] == "tp"


def _write_overlapping_drafts(run_dir: Path) -> None:
    """Write three drafts for ``smoke.csv/School`` that overlap.

    Mirrors the production failure mode (LLM proposer emits a coarse
    super-set view + finer subset views):

      c1 = {BBB, ABC HS, EEE, DDD}            (4 values, superset)
      c2 = {BBB, ABC HS}                       (2 values, subset of c1)
      c3 = {EEE, DDD}                          (2 values, subset of c1)

    The persisted ``ColumnClusters`` schema rejects overlap, so the
    auto-promote path must dedupe before validation. The cluster the
    operator targets is given priority so the verdict can always
    land.
    """
    drafts = {
        "source": "smoke.csv",
        "column": "School",
        "draft_clusters": [
            {
                "kind": "casing",
                "canonical": "BBB",
                "agreement": 0.6,
                "values": ["BBB", "ABC HS", "EEE", "DDD"],
                "target_style": "uppercase canonical",
            },
            {
                "kind": "casing",
                "canonical": "BBB",
                "agreement": 0.8,
                "values": ["BBB", "ABC HS"],
            },
            {
                "kind": "casing",
                "canonical": "EEE",
                "agreement": 0.7,
                "values": ["EEE", "DDD"],
            },
        ],
    }
    with (run_dir / "column_cluster_drafts.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        fh.write(json.dumps(drafts) + "\n")

