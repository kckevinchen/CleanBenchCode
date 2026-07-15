"""Tests for column-cluster contribution to the ``/stats`` endpoint.

When a run is column-cluster-only (``clusters.json`` empty,
``column_clusters.jsonl`` and/or ``column_cluster_drafts.jsonl`` non-
empty), the label page's progress pills used to show all zeros because
the stats endpoint only walked ``ctx.clusters``. After the fix, the
endpoint also enumerates column-cluster queue targets via
:func:`src.labeling.column_cluster_queue.iter_column_cluster_targets`
and folds their verdict counts into ``overall`` and ``by_source``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.schema import (
    ClusterKind, ClusterValue, ColumnClusters, ColumnLabelStatus, StdCluster,
    column_cluster_target_id,
)


def _make_run(project: Path, run_id: str = "20260101T000000Z_ccstats") -> Path:
    """Run dir with **no cell clusters** but two column-cluster
    sources: a saved labelled record on a.csv and an LLM draft on
    b.csv. Mirrors the live ``20260512T210510Z_gpt5-family`` shape
    (column-cluster-only run)."""
    data_dir = project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "a.csv").write_text(
        "name,email\nalice,a@x\nAlice,a@x\nbob,b@y\nBob,b@y\n",
        encoding="utf-8",
    )
    (data_dir / "b.csv").write_text(
        "name,email\nemma,e@x\nEmma,e@x\n", encoding="utf-8",
    )

    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    run_dir = outputs / run_id
    run_dir.mkdir()

    summary = {
        "config": {
            "csv_paths": [
                str(data_dir / "a.csv"),
                str(data_dir / "b.csv"),
            ],
            "sources": ["a.csv", "b.csv"],
        }
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")

    # Saved record: a.csv / name -> c1 (2 vals), c2 (2 vals).
    rec = ColumnClusters(
        source="a.csv", column="name",
        status=ColumnLabelStatus.LABELED,
        clusters=[
            StdCluster(
                cluster_id="c1", kind=ClusterKind.UNTAGGED,
                canonical_value="alice",
                values=[
                    ClusterValue(value="alice", count=2),
                    ClusterValue(value="Alice", count=2),
                ],
            ),
            StdCluster(
                cluster_id="c2", kind=ClusterKind.UNTAGGED,
                canonical_value="bob",
                values=[
                    ClusterValue(value="bob", count=2),
                    ClusterValue(value="Bob", count=2),
                ],
            ),
        ],
        annotator="tester",
    )
    (run_dir / "column_clusters.jsonl").write_text(
        rec.model_dump_json() + "\n", encoding="utf-8",
    )

    # Drafts: b.csv / name has one ≥2-value draft cluster.
    drafts = [
        {
            "source": "b.csv", "column": "name",
            "runs_completed": 1, "stopped_reason": "two_consecutive",
            "draft_clusters": [{
                "values": ["emma", "Emma"], "kind": "untagged",
                "canonical": "emma", "agreement": 0.5,
            }],
        },
    ]
    (run_dir / "column_cluster_drafts.jsonl").write_text(
        "\n".join(json.dumps(d) for d in drafts) + "\n",
        encoding="utf-8",
    )

    _make_run.targets = {  # type: ignore[attr-defined]
        "a_name_c1": column_cluster_target_id("a.csv", "name", "c1"),
        "a_name_c2": column_cluster_target_id("a.csv", "name", "c2"),
        "b_name_d1": column_cluster_target_id("b.csv", "name", "c1"),
    }
    return run_dir


@pytest.fixture
def cc_client_and_run():
    with tempfile.TemporaryDirectory() as td:
        project = Path(td)
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(project)
        app = create_app(project, annotator="tester")
        client = TestClient(app)
        yield client, run_dir.name, _make_run.targets  # type: ignore[attr-defined]


class TestStatsIncludeColumnClusters:
    """The progress pills on the label page (driven by ``/stats``)
    must include column-cluster queue items, not just cell-clusters."""

    def test_overall_total_includes_column_cluster_targets(
        self, cc_client_and_run,
    ) -> None:
        client, run_id, _targets = cc_client_and_run
        s = client.get(f"/api/runs/{run_id}/stats").json()
        # 2 saved clusters + 1 draft cluster = 3 unlabeled targets.
        assert s["overall"]["total"] == 3
        assert s["overall"]["unlabeled"] == 3
        assert s["overall"]["tp"] == 0
        assert s["overall"]["fp"] == 0

    def test_by_source_split_includes_column_clusters(
        self, cc_client_and_run,
    ) -> None:
        client, run_id, _targets = cc_client_and_run
        s = client.get(f"/api/runs/{run_id}/stats").json()
        # a.csv: 2 saved clusters; b.csv: 1 draft cluster.
        assert s["by_source"]["a.csv"]["total"] == 2
        assert s["by_source"]["a.csv"]["unlabeled"] == 2
        assert s["by_source"]["b.csv"]["total"] == 1
        assert s["by_source"]["b.csv"]["unlabeled"] == 1

    def test_verdict_on_column_cluster_target_updates_stats(
        self, cc_client_and_run,
    ) -> None:
        """TP / FP / Skip verdicts on a column-cluster target must move
        the corresponding counter in /stats."""
        client, run_id, targets = cc_client_and_run
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster",
                "target_id": targets["a_name_c1"],
                "source": "a.csv", "column": "name",
                "columns": ["name"], "status": "tp",
            },
        )
        assert r.status_code == 200, r.text
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster",
                "target_id": targets["a_name_c2"],
                "source": "a.csv", "column": "name",
                "columns": ["name"], "status": "fp",
            },
        )
        assert r.status_code == 200, r.text
        s = client.get(f"/api/runs/{run_id}/stats").json()
        assert s["overall"]["total"] == 3
        assert s["overall"]["tp"] == 1
        assert s["overall"]["fp"] == 1
        assert s["overall"]["unlabeled"] == 1
        # by_source slice picks up the same counts.
        assert s["by_source"]["a.csv"]["tp"] == 1
        assert s["by_source"]["a.csv"]["fp"] == 1
        assert s["by_source"]["a.csv"]["unlabeled"] == 0
        assert s["by_source"]["b.csv"]["unlabeled"] == 1

    def test_exclusion_cascade_moves_stats_to_skip(
        self, cc_client_and_run,
    ) -> None:
        """The exclusion -> skip cascade already covers column-clusters
        (see test_exclusion_cascade.py); confirm the stats pill flips
        accordingly so the operator sees real progress."""
        client, run_id, _targets = cc_client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "columns", "source": "a.csv", "columns": ["name"]},
        )
        assert r.status_code == 200, r.text
        s = client.get(f"/api/runs/{run_id}/stats").json()
        assert s["overall"]["total"] == 3
        # 2 column-clusters on a.csv/name moved to skip.
        assert s["overall"]["skip"] == 2
        assert s["overall"]["unlabeled"] == 1  # b.csv/name draft still raw

    def test_empty_run_yields_zero_total(self, tmp_path: Path) -> None:
        """Sanity: a run with no cell clusters and no column-clusters
        still returns a valid stats payload with all zeros (no
        regression)."""
        project = tmp_path
        (project / "frontend" / "labeling").mkdir(parents=True)
        data_dir = project / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "a.csv").write_text("name\nalice\n", encoding="utf-8")
        outputs = project / "outputs"
        outputs.mkdir(parents=True)
        run_dir = outputs / "r"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps({
            "config": {"csv_paths": [str(data_dir / "a.csv")]}
        }))
        (run_dir / "clusters.json").write_text("[]")
        (run_dir / "issues.json").write_text("[]")
        app = create_app(project, annotator="t")
        client = TestClient(app)
        s = client.get("/api/runs/r/stats").json()
        assert s["overall"]["total"] == 0
        assert s["overall"]["unlabeled"] == 0
        assert s["fn"]["total"] == 0


# ---------------------------------------------------------------------------
# Multi-table-only runs must populate the stats bar too
# ---------------------------------------------------------------------------


def _make_mt_run(project: Path, run_id: str = "20260101T000000Z_mtstats") -> Path:
    """Run dir with NO cell/column clusters but three multi-table
    findings (one spanning two sheets). Mirrors a multi-table-only run
    (e.g. inter_table_inconsistency over a curated workbook corpus)."""
    data_dir = project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "a.csv").write_text("K\n1\n", encoding="utf-8")
    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    run_dir = outputs / run_id
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "config": {"csv_paths": [str(data_dir / "a.csv")]}
    }))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")
    # 3 distinct findings; vid1 spans two (sheet,row) rows.
    rows = [
        {"violation_id": "vid1", "kind": "attribute_conflict",
         "workbook": "wb1", "group_id": "u1",
         "sheet": "a.csv", "row": 0, "columns": ["x"], "values": {},
         "linked": [{"sheet": "b.csv", "row": 0}]},
        {"violation_id": "vid1", "kind": "attribute_conflict",
         "workbook": "wb1", "group_id": "u1",
         "sheet": "b.csv", "row": 0, "columns": ["x"], "values": {},
         "linked": [{"sheet": "a.csv", "row": 0}]},
        {"violation_id": "vid2", "kind": "duplicate_row",
         "workbook": "wb1", "group_id": "u1",
         "sheet": "a.csv", "row": 1, "columns": [], "values": {}, "linked": []},
        {"violation_id": "vid3", "kind": "fk_orphan",
         "workbook": "wb1", "group_id": "j1",
         "sheet": "b.csv", "row": 2, "columns": [], "values": {}, "linked": []},
    ]
    (run_dir / "multi_table_violations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )
    # Resolver output: one workbook with a confirmed union, a confirmed
    # join, and an unconfirmed union candidate (filtered out by default).
    groups = [{
        "workbook": "wb1",
        "sheets": ["a.csv", "b.csv"],
        "groups": [
            {"group_id": "u1", "kind": "union", "confirmed": True,
             "sheets": ["a.csv", "b.csv"], "primary_key": ["K"],
             "rationale": "same schema"},
            {"group_id": "j1", "kind": "join", "confirmed": True,
             "sheets": ["a.csv", "b.csv"], "primary_key": [],
             "join": {"parent_sheet": "a.csv", "parent_columns": ["K"],
                      "child_sheet": "b.csv", "child_columns": ["K"]},
             "rationale": "fk"},
            {"group_id": "u2", "kind": "union", "confirmed": False,
             "sheets": ["a.csv", "b.csv"], "primary_key": ["K"],
             "rationale": "maybe"},
        ],
    }]
    (run_dir / "multi_table_groups.jsonl").write_text(
        "\n".join(json.dumps(g) for g in groups) + "\n", encoding="utf-8",
    )
    return run_dir


@pytest.fixture
def mt_client_and_run():
    with tempfile.TemporaryDirectory() as td:
        project = Path(td)
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_mt_run(project)
        app = create_app(project, annotator="tester")
        yield TestClient(app), run_dir.name


class TestStatsIncludeMultiTable:
    def test_overall_total_includes_mt_findings(self, mt_client_and_run) -> None:
        client, run_id = mt_client_and_run
        s = client.get(f"/api/runs/{run_id}/stats").json()
        # 3 distinct violation_ids -> 3 unlabeled findings (NOT 0).
        assert s["overall"]["total"] == 3
        assert s["overall"]["unlabeled"] == 3

    def test_mt_label_moves_the_counter(self, mt_client_and_run) -> None:
        client, run_id = mt_client_and_run
        r = client.post(
            f"/api/runs/{run_id}/multi-table-labels",
            json={"violation_id": "vid1",
                  "status": "confirmed_inconsistency"},
        )
        assert r.status_code == 200, r.text
        r = client.post(
            f"/api/runs/{run_id}/multi-table-labels",
            json={"violation_id": "vid2", "status": "not_a_violation"},
        )
        assert r.status_code == 200, r.text
        s = client.get(f"/api/runs/{run_id}/stats").json()
        assert s["overall"]["total"] == 3
        assert s["overall"]["tp"] == 1      # confirmed_inconsistency
        assert s["overall"]["fp"] == 1      # not_a_violation
        assert s["overall"]["unlabeled"] == 1


class TestMultiTableGroupsEndpoint:
    def test_confirmed_only_by_default(self, mt_client_and_run) -> None:
        client, run_id = mt_client_and_run
        d = client.get(f"/api/runs/{run_id}/multi-table-groups").json()
        t = d["totals"]
        assert t["workbooks"] == 1
        assert t["union"] == 1            # only the confirmed union
        assert t["join"] == 1
        assert t["groups"] == 2
        assert t["unconfirmed"] == 1      # u2 filtered out, still counted
        gids = {g["group_id"] for g in d["workbooks"][0]["groups"]}
        assert gids == {"u1", "j1"}

    def test_finding_counts_and_join_spec(self, mt_client_and_run) -> None:
        client, run_id = mt_client_and_run
        d = client.get(f"/api/runs/{run_id}/multi-table-groups").json()
        groups = {g["group_id"]: g for g in d["workbooks"][0]["groups"]}
        # u1 carries vid1 + vid2; j1 carries vid3.
        assert groups["u1"]["n_findings"] == 2
        assert groups["u1"]["kind"] == "union"
        assert groups["u1"]["primary_key"] == ["K"]
        assert groups["j1"]["n_findings"] == 1
        assert groups["j1"]["kind"] == "join"
        assert groups["j1"]["join"]["parent_sheet"] == "a.csv"
        assert groups["j1"]["join"]["child_columns"] == ["K"]

    def test_include_unconfirmed(self, mt_client_and_run) -> None:
        client, run_id = mt_client_and_run
        d = client.get(
            f"/api/runs/{run_id}/multi-table-groups",
            params={"include_unconfirmed": "true"},
        ).json()
        assert d["totals"]["union"] == 2   # confirmed u1 + candidate u2
        gids = {g["group_id"] for g in d["workbooks"][0]["groups"]}
        assert gids == {"u1", "j1", "u2"}
        u2 = next(g for g in d["workbooks"][0]["groups"] if g["group_id"] == "u2")
        assert u2["confirmed"] is False

