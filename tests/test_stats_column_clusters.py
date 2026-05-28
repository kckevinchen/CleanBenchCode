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
