"""POST /api/runs/{run_id}/duplicate clones a run directory.

Focus: the ``reset_labels`` "clean slate" path must drop EVERY
operator-authored store -- ``labels.jsonl``, the column-cluster verdict
store ``column_clusters.jsonl``, ``exclusions.jsonl`` and
``cell_notes.jsonl`` -- while preserving the detection artifacts
(``summary.json``, ``clusters.json``) and the LLM-side
``column_cluster_drafts.jsonl`` proposals. Regression guard for the bug
where the reset unlinked a non-existent ``column_cluster_labels.jsonl``
and so silently carried the source's column-cluster verdicts into the
"clean" copy.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "dup"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "t.csv"
    csv.write_text("id,name\n1,Acme\n2,Beta\n", encoding="utf-8")
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_dupsrc"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(json.dumps({"config": {"csv_paths": [str(csv)]}}))
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "t.csv__r00000-c000", "source": "t.csv", "row": 0,
         "columns": ["name"], "row_values": {"id": "1", "name": "Acme"},
         "error_types": ["typo"], "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


# Operator-authored stores that a clean-slate clone must NOT inherit.
_OPERATOR_STORES = (
    "labels.jsonl",
    "column_clusters.jsonl",
    "exclusions.jsonl",
    "cell_notes.jsonl",
)


def _seed_operator_state(rd: Path) -> None:
    """Write a line into every operator-authored store plus the LLM
    drafts store (which must be preserved)."""
    for rel in _OPERATOR_STORES:
        (rd / rel).write_text('{"seeded": true}\n', encoding="utf-8")
    (rd / "column_cluster_drafts.jsonl").write_text(
        '{"draft": true}\n', encoding="utf-8"
    )


def test_reset_labels_drops_every_operator_store(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    src_rd = project / "outputs" / run_id
    _seed_operator_state(src_rd)

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/duplicate", json={"reset_labels": True})
        assert r.status_code == 200, r.text
        new_id = r.json()["run_id"]
        assert new_id != run_id

        dest_rd = project / "outputs" / new_id
        assert dest_rd.is_dir()
        # Every operator store is gone in the clone...
        for rel in _OPERATOR_STORES:
            assert not (dest_rd / rel).exists(), f"{rel} should be reset"
        # ...but the detection artifacts + LLM drafts are preserved.
        assert (dest_rd / "summary.json").exists()
        assert (dest_rd / "clusters.json").exists()
        assert (dest_rd / "column_cluster_drafts.jsonl").exists()
        # The source is untouched.
        for rel in _OPERATOR_STORES:
            assert (src_rd / rel).exists()


def test_default_keeps_labels(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    src_rd = project / "outputs" / run_id
    _seed_operator_state(src_rd)

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Default (no reset_labels) is an exact copy -> labels carried.
        r = client.post(f"/api/runs/{run_id}/duplicate", json={})
        assert r.status_code == 200, r.text
        new_id = r.json()["run_id"]
        dest_rd = project / "outputs" / new_id
        for rel in _OPERATOR_STORES:
            assert (dest_rd / rel).exists(), f"{rel} should be copied"


def test_clone_is_unstarred_and_renamed(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/duplicate", json={"reset_labels": True})
        body = r.json()
        info = body["run"]
        assert info.get("starred") in (False, None)
        assert "Copy of" in (info.get("display_name") or "")
