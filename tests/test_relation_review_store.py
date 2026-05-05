"""Tests for the relation-review label store.

Mirrors :mod:`tests.test_label_store`: tmp_path JSONL, append +
reload, latest-wins, ``added`` snapshot round-trip, malformed-line
tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.labeling.relation_review_store import (
    RelationReviewStore,
    RelationSnapshot,
    ReviewDecision,
    ReviewIn,
)


def _approve(rel_id: str, author: str = "alice") -> ReviewIn:
    return ReviewIn(
        relation_id=rel_id,
        decision=ReviewDecision.APPROVED,
        author=author,
    )


def _reject(rel_id: str, note: str = "wrong") -> ReviewIn:
    return ReviewIn(
        relation_id=rel_id,
        decision=ReviewDecision.REJECTED,
        note=note,
    )


def _added(rel_id: str = "user_xyz") -> ReviewIn:
    return ReviewIn(
        relation_id=rel_id,
        decision=ReviewDecision.ADDED,
        relation=RelationSnapshot(
            kind="implication",
            columns=["A", "B"],
            lhs=["A"],
            rhs=["B"],
            predicate="if A == 'x' then B is non-null",
            rationale="business rule",
            source="t.csv",
            error_type="cross_column_inconsistency",
        ),
    )


class TestRelationReviewStore:
    def test_append_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "relation_reviews.jsonl"
        s = RelationReviewStore(path)
        s.append(_approve("rel1"))
        s.append(_reject("rel2"))
        # Reload from disk.
        s2 = RelationReviewStore(path)
        latest = s2.latest_by_relation()
        assert set(latest.keys()) == {"rel1", "rel2"}
        assert latest["rel1"].decision == ReviewDecision.APPROVED
        assert latest["rel2"].decision == ReviewDecision.REJECTED

    def test_latest_wins(self, tmp_path: Path) -> None:
        s = RelationReviewStore(tmp_path / "r.jsonl")
        s.append(_approve("rel1"))
        s.append(_reject("rel1", note="changed my mind"))
        latest = s.latest_for("rel1")
        assert latest is not None
        assert latest.decision == ReviewDecision.REJECTED
        # History preserved.
        all_records = s.all()
        assert len(all_records) == 2

    def test_added_relation_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        s = RelationReviewStore(path)
        s.append(_added("user_abc"))
        s2 = RelationReviewStore(path)
        added = s2.added_relations()
        assert len(added) == 1
        rec = added[0]
        assert rec.decision == ReviewDecision.ADDED
        assert rec.relation is not None
        assert rec.relation.columns == ["A", "B"]
        assert rec.relation.predicate.startswith("if A")

    def test_clearing_decision_overrides_added(self, tmp_path: Path) -> None:
        s = RelationReviewStore(tmp_path / "r.jsonl")
        s.append(_added("user_abc"))
        s.append(ReviewIn(
            relation_id="user_abc",
            decision=ReviewDecision.CLEARED,
        ))
        # ``added_relations`` only returns rows whose latest decision is
        # ADDED. After clearing, the user-authored card disappears.
        assert s.added_relations() == []
        latest = s.latest_for("user_abc")
        assert latest is not None
        assert latest.decision == ReviewDecision.CLEARED

    def test_review_id_unique(self, tmp_path: Path) -> None:
        s = RelationReviewStore(tmp_path / "r.jsonl")
        r1 = s.append(_approve("rel1"))
        r2 = s.append(_approve("rel1"))
        assert r1.review_id != r2.review_id

    def test_corrupt_line_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        # Pre-seed with a broken line.
        path.write_text("not-json\n", encoding="utf-8")
        s = RelationReviewStore(path)
        s.append(_approve("rel1"))
        assert "rel1" in s.latest_by_relation()
        assert len(s.all()) == 1


class TestRelationReviewAPI:
    """End-to-end exercise of the FastAPI endpoints. Skipped when the
    full workbench app cannot be assembled in this test environment
    (e.g. missing optional deps)."""

    def _build_app(self, tmp_path: Path):
        try:
            from fastapi.testclient import TestClient  # noqa: WPS433
            from src.labeling.app import create_app  # noqa: WPS433
        except Exception:  # noqa: BLE001
            pytest.skip("workbench app dependencies unavailable")
        project_root = tmp_path
        outputs = project_root / "outputs"
        run_dir = outputs / "run1"
        run_dir.mkdir(parents=True)
        # Stub static dir + minimal run scaffolding.
        static_dir = project_root / "frontend" / "labeling"
        static_dir.mkdir(parents=True)
        (static_dir / "index.html").write_text("<html></html>", encoding="utf-8")
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")
        app = create_app(
            project_root,
            annotator="test",
            static_dir=static_dir,
        )
        return TestClient(app), run_dir

    def test_post_and_get(self, tmp_path: Path) -> None:
        client, run_dir = self._build_app(tmp_path)
        r = client.post(
            "/api/runs/run1/relation-reviews",
            json={"relation_id": "abc", "decision": "approved"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["record"]["decision"] == "approved"
        assert "abc" in body["latest"]
        # GET returns persisted state.
        r2 = client.get("/api/runs/run1/relation-reviews")
        assert r2.status_code == 200
        latest = r2.json()["latest"]
        assert latest["abc"]["decision"] == "approved"
        # File on disk.
        assert (run_dir / "relation_reviews.jsonl").exists()

    def test_added_requires_relation_payload(self, tmp_path: Path) -> None:
        client, _ = self._build_app(tmp_path)
        r = client.post(
            "/api/runs/run1/relation-reviews",
            json={"relation_id": "user_x", "decision": "added"},
        )
        assert r.status_code == 400

    def test_added_round_trip(self, tmp_path: Path) -> None:
        client, _ = self._build_app(tmp_path)
        r = client.post(
            "/api/runs/run1/relation-reviews",
            json={
                "relation_id": "user_x",
                "decision": "added",
                "relation": {
                    "kind": "implication",
                    "columns": ["A"],
                    "predicate": "A is non-null",
                },
            },
        )
        assert r.status_code == 200, r.text
        latest = client.get("/api/runs/run1/relation-reviews").json()["latest"]
        assert latest["user_x"]["decision"] == "added"
        assert latest["user_x"]["relation"]["columns"] == ["A"]
