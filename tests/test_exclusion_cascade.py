"""Tests for the exclusion -> skip cascade.

When an exclusion is added via POST /api/runs/{run_id}/exclusions,
every cluster that falls inside the new scope should immediately
receive a ``skip`` cluster label (matches the frontend's
``_auto_skip_for_exclusion`` contract). DELETE-ing the exclusion
should revert the auto-skip labels via ``UNLABELED`` tombstones so
the targets go back to "unlabeled".
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app


def _make_run(project: Path, run_id: str = "20260101T000000Z_test") -> Path:
    """Run dir with clusters spanning multiple sources / rows / columns
    so each scope variant can be exercised independently.

    Layout::

        a.csv: c1 row=1 col=name, c2 row=2 col=email, c3 row=3 col=name
        b.csv: c4 row=0 col=name
    """
    data_dir = project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "a.csv").write_text(
        "name,email\nalice,a@x\nbob,b@x\ncarol,c@x\ndave,d@x\n",
        encoding="utf-8",
    )
    (data_dir / "b.csv").write_text(
        "name,email\nemma,e@x\nfrank,f@x\n", encoding="utf-8",
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
    clusters = [
        {
            "cluster_id": "c1", "source": "a.csv", "row": 1,
            "columns": ["name"], "error_types": ["typo"],
            "evidence_count": 1, "synthesis": None,
        },
        {
            "cluster_id": "c2", "source": "a.csv", "row": 2,
            "columns": ["email"], "error_types": ["typo"],
            "evidence_count": 1, "synthesis": None,
        },
        {
            "cluster_id": "c3", "source": "a.csv", "row": 3,
            "columns": ["name"], "error_types": ["typo"],
            "evidence_count": 1, "synthesis": None,
        },
        {
            "cluster_id": "c4", "source": "b.csv", "row": 0,
            "columns": ["name"], "error_types": ["typo"],
            "evidence_count": 1, "synthesis": None,
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


def _label_status(client: TestClient, run_id: str, target_id: str) -> str | None:
    r = client.get(f"/api/runs/{run_id}/labels")
    assert r.status_code == 200, r.text
    latest = (r.json() or {}).get("latest") or {}
    rec = latest.get(target_id)
    return rec.get("status") if rec else None


class TestExclusionCascade:
    """POST /exclusions should auto-write skip labels for every
    cluster inside the new scope."""

    def test_column_exclusion_skips_matching_clusters(self, client_and_run):
        client, run_id = client_and_run
        # c1 and c3 are on column "name"; c2 is on "email"; both in a.csv.
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={
                "scope": "columns",
                "source": "a.csv",
                "columns": ["name"],
            },
        )
        assert r.status_code == 200, r.text
        assert _label_status(client, run_id, "c1") == "skip"
        assert _label_status(client, run_id, "c3") == "skip"
        # Column doesn't match -> untouched.
        assert _label_status(client, run_id, "c2") is None
        # Other source untouched.
        assert _label_status(client, run_id, "c4") is None

    def test_row_exclusion_skips_matching_clusters(self, client_and_run):
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "rows", "source": "a.csv", "rows": [1, 2]},
        )
        assert r.status_code == 200, r.text
        assert _label_status(client, run_id, "c1") == "skip"  # row=1
        assert _label_status(client, run_id, "c2") == "skip"  # row=2
        assert _label_status(client, run_id, "c3") is None    # row=3
        assert _label_status(client, run_id, "c4") is None    # b.csv

    def test_csv_exclusion_skips_every_cluster_in_source(self, client_and_run):
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "csv", "source": "a.csv"},
        )
        assert r.status_code == 200, r.text
        for cid in ("c1", "c2", "c3"):
            assert _label_status(client, run_id, cid) == "skip", cid
        # b.csv untouched.
        assert _label_status(client, run_id, "c4") is None

    def test_cascade_overwrites_prior_verdict(self, client_and_run):
        """A column exclusion should win over a prior TP cluster
        verdict so the queue advances past stale labels."""
        client, run_id = client_and_run
        # Operator labels c1 as TP first.
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "columns": ["name"], "status": "tp",
            },
        )
        assert r.status_code == 200, r.text
        assert _label_status(client, run_id, "c1") == "tp"
        # Then excludes the column.
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "columns", "source": "a.csv", "columns": ["name"]},
        )
        assert _label_status(client, run_id, "c1") == "skip"

    def test_cascade_idempotent_when_already_skip(self, client_and_run):
        """Re-issuing the same exclusion (or excluding twice) should not
        write a redundant skip label for a cluster that's already
        ``skip``."""
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "columns", "source": "a.csv", "columns": ["name"]},
        )
        before = client.get(f"/api/runs/{run_id}/labels").json()
        before_count = (before.get("history_count") or 0)
        # Same scope, fresh exclusion -> targets are already skip, so
        # no new labels should be appended.
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "columns", "source": "a.csv", "columns": ["name"]},
        )
        after = client.get(f"/api/runs/{run_id}/labels").json()
        assert after.get("history_count") == before_count

    def test_delete_exclusion_reverts_auto_skip(self, client_and_run):
        """Removing the exclusion should un-do the auto-skip via
        ``UNLABELED`` tombstones - matching the schema's documented
        contract."""
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "columns", "source": "a.csv", "columns": ["name"]},
        )
        exclusion_id = r.json()["id"]
        assert _label_status(client, run_id, "c1") == "skip"
        d = client.delete(
            f"/api/runs/{run_id}/exclusions/{exclusion_id}",
        )
        assert d.status_code == 200, d.text
        # ``latest_by_target`` filters UNLABELED tombstones out -> the
        # cluster is unlabeled again.
        assert _label_status(client, run_id, "c1") is None
        assert _label_status(client, run_id, "c3") is None

    def test_delete_preserves_manual_relabel_after_auto_skip(self, client_and_run):
        """If the operator manually relabels a target AFTER an auto-skip,
        un-doing the exclusion must NOT revert the manual label."""
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "columns", "source": "a.csv", "columns": ["name"]},
        )
        exclusion_id = r.json()["id"]
        # Operator overrides to TP after the auto-skip.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "columns": ["name"], "status": "tp",
            },
        )
        assert _label_status(client, run_id, "c1") == "tp"
        # Now undo the exclusion: c1 should stay TP, c3 (still auto-skip)
        # should revert to unlabeled.
        client.delete(f"/api/runs/{run_id}/exclusions/{exclusion_id}")
        assert _label_status(client, run_id, "c1") == "tp"
        assert _label_status(client, run_id, "c3") is None

    def test_post_succeeds_when_no_clusters_match(self, client_and_run):
        """Excluding a column with no clusters in scope should still
        succeed and simply write nothing."""
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={
                "scope": "columns", "source": "b.csv",
                "columns": ["email"],
            },
        )
        assert r.status_code == 200, r.text
        # No cluster in b.csv touches "email" so no labels should
        # have been written.
        labels = client.get(f"/api/runs/{run_id}/labels").json()
        assert (labels.get("history_count") or 0) == 0


# ---------------------------------------------------------------------------
# Column-cluster cascade (inconsistency error type)
# ---------------------------------------------------------------------------


def _make_run_with_column_clusters(
    project: Path, run_id: str = "20260101T000000Z_cctest",
) -> Path:
    """Run dir wired up so both saved column-cluster records *and* LLM
    drafts are present.

    Layout::

        a.csv:
          saved record on column "name"  -> clusters c1, c2
          LLM draft on column "email"    -> draft cluster c1 (2 values)
        b.csv:
          LLM draft on column "name"     -> draft cluster c1 (2 values)

    Cell-cluster ``clusters.json`` is empty: the column-cluster
    cascade must work even when no cell-clusters exist.
    """
    from src.labeling.schema import (
        ClusterKind, ClusterValue, ColumnClusters, ColumnLabelStatus,
        StdCluster, column_cluster_target_id,
    )

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

    # Saved column-cluster record on a.csv / name (two clusters).
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

    # LLM drafts: one on a.csv / email (a column with no saved record)
    # and one on b.csv / name. Both columns become queue items.
    drafts = [
        {
            "source": "a.csv", "column": "email",
            "runs_completed": 1, "stopped_reason": "two_consecutive",
            "draft_clusters": [{
                "values": ["a@x", "A@X"], "kind": "untagged",
                "canonical": "a@x", "agreement": 0.5,
            }],
        },
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

    # Stash the expected target_ids on the function for the tests.
    _make_run_with_column_clusters.targets = {  # type: ignore[attr-defined]
        "a_name_c1": column_cluster_target_id("a.csv", "name", "c1"),
        "a_name_c2": column_cluster_target_id("a.csv", "name", "c2"),
        "a_email_d1": column_cluster_target_id("a.csv", "email", "c1"),
        "b_name_d1": column_cluster_target_id("b.csv", "name", "c1"),
    }
    return run_dir


@pytest.fixture
def cc_client_and_run():
    with tempfile.TemporaryDirectory() as td:
        project = Path(td)
        (project / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run_with_column_clusters(project)
        app = create_app(project, annotator="tester")
        client = TestClient(app)
        yield client, run_dir.name, (
            _make_run_with_column_clusters.targets  # type: ignore[attr-defined]
        )


class TestColumnClusterCascade:
    """The same exclusion -> skip cascade should apply to column-cluster
    queue items (saved records and LLM drafts), not just cell-cluster
    targets. ROWS-scope exclusions are deliberately a no-op on
    column-clusters (which span the whole column)."""

    def test_csv_exclusion_skips_every_column_cluster_in_source(
        self, cc_client_and_run,
    ) -> None:
        client, run_id, targets = cc_client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "csv", "source": "a.csv"},
        )
        assert r.status_code == 200, r.text
        # a.csv: saved record's c1+c2 *and* the email draft all skip.
        assert _label_status(client, run_id, targets["a_name_c1"]) == "skip"
        assert _label_status(client, run_id, targets["a_name_c2"]) == "skip"
        assert _label_status(client, run_id, targets["a_email_d1"]) == "skip"
        # b.csv untouched.
        assert _label_status(client, run_id, targets["b_name_d1"]) is None

    def test_columns_exclusion_skips_matching_column_clusters(
        self, cc_client_and_run,
    ) -> None:
        client, run_id, targets = cc_client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={
                "scope": "columns", "source": "a.csv",
                "columns": ["name"],
            },
        )
        assert r.status_code == 200, r.text
        # Saved record's two clusters on a.csv / name -> skip.
        assert _label_status(client, run_id, targets["a_name_c1"]) == "skip"
        assert _label_status(client, run_id, targets["a_name_c2"]) == "skip"
        # a.csv / email draft is on a different column -> untouched.
        assert _label_status(client, run_id, targets["a_email_d1"]) is None
        # b.csv / name draft -> different source -> untouched.
        assert _label_status(client, run_id, targets["b_name_d1"]) is None

    def test_rows_exclusion_does_not_skip_column_clusters(
        self, cc_client_and_run,
    ) -> None:
        """ROWS-scope is per-cell; column-clusters span the whole
        column, so a ROWS exclusion must not touch them."""
        client, run_id, targets = cc_client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "rows", "source": "a.csv", "rows": [1, 2]},
        )
        assert r.status_code == 200, r.text
        # None of the column-cluster targets should be touched.
        for k in (
            "a_name_c1", "a_name_c2", "a_email_d1", "b_name_d1",
        ):
            assert _label_status(client, run_id, targets[k]) is None, k

    def test_cascade_overrides_prior_column_cluster_verdict(
        self, cc_client_and_run,
    ) -> None:
        """Prior TP / FP / partial on a column-cluster must lose to the
        later exclusion -> skip cascade."""
        client, run_id, targets = cc_client_and_run
        # Operator labels c1 (saved) as TP first.
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
        assert _label_status(client, run_id, targets["a_name_c1"]) == "tp"
        # Then excludes the column.
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={
                "scope": "columns", "source": "a.csv",
                "columns": ["name"],
            },
        )
        assert _label_status(client, run_id, targets["a_name_c1"]) == "skip"

    def test_cascade_is_idempotent_on_column_clusters(
        self, cc_client_and_run,
    ) -> None:
        """Re-issuing the same exclusion should not duplicate the skip
        label on a column-cluster already at skip."""
        client, run_id, targets = cc_client_and_run
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={
                "scope": "columns", "source": "a.csv",
                "columns": ["name"],
            },
        )
        before = client.get(f"/api/runs/{run_id}/labels").json()
        before_count = before.get("history_count") or 0
        # Issue the same exclusion again -> no new labels.
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={
                "scope": "columns", "source": "a.csv",
                "columns": ["name"],
            },
        )
        after = client.get(f"/api/runs/{run_id}/labels").json()
        assert (after.get("history_count") or 0) == before_count

    def test_delete_reverts_column_cluster_auto_skip(
        self, cc_client_and_run,
    ) -> None:
        """Removing the exclusion should also revert column-cluster
        auto-skips via UNLABELED tombstones."""
        client, run_id, targets = cc_client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={
                "scope": "columns", "source": "a.csv",
                "columns": ["name"],
            },
        )
        exclusion_id = r.json()["id"]
        assert _label_status(client, run_id, targets["a_name_c1"]) == "skip"
        assert _label_status(client, run_id, targets["a_name_c2"]) == "skip"
        d = client.delete(
            f"/api/runs/{run_id}/exclusions/{exclusion_id}",
        )
        assert d.status_code == 200, d.text
        # Tombstones -> latest_by_target filters UNLABELED out.
        assert _label_status(client, run_id, targets["a_name_c1"]) is None
        assert _label_status(client, run_id, targets["a_name_c2"]) is None

    def test_delete_preserves_manual_column_cluster_relabel(
        self, cc_client_and_run,
    ) -> None:
        """If the operator manually relabels a column-cluster AFTER the
        auto-skip, deleting the exclusion must NOT revert that manual
        verdict."""
        client, run_id, targets = cc_client_and_run
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={
                "scope": "columns", "source": "a.csv",
                "columns": ["name"],
            },
        )
        exclusion_id = r.json()["id"]
        # Operator overrides c1 -> TP after the auto-skip.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "column_cluster",
                "target_id": targets["a_name_c1"],
                "source": "a.csv", "column": "name",
                "columns": ["name"], "status": "tp",
            },
        )
        assert _label_status(client, run_id, targets["a_name_c1"]) == "tp"
        client.delete(f"/api/runs/{run_id}/exclusions/{exclusion_id}")
        # c1 stays TP, c2 (still auto-skip) reverts to unlabeled.
        assert _label_status(client, run_id, targets["a_name_c1"]) == "tp"
        assert _label_status(client, run_id, targets["a_name_c2"]) is None
