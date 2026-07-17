"""POST /api/runs/{run_id}/labels/tag-trivial-row-duplication auto-tags
every *completely identical* (byte-identical across all columns) row
group ``TRIVIAL``: an LLM-flagged exact duplicate becomes TP, a missed
one becomes a row-level FN.

Mirrors the style of ``tests/test_tag_trivial_whitespace.py``: a small
synthetic run with hand-crafted ``row_duplication`` clusters, a
TestClient flow, and assertions on the dry-run preview + the persisted
append-only result. Also checks the benchmark export materialises the
row-level FN as a ``row_duplication`` "error" cell.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.export import materialize_label_benchmark
from src.labeling.schema import Label, LabelStatus, LabelTarget, cell_target_id
from src.labeling.store import LabelStore


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    """Tiny run with one CSV mixing exact duplicates, a conflicting
    (partial) duplicate, and a unique row:

      * rows 0 & 3  -> "1,Acme,NYC"  EXACT dup, flagged by cluster C0
      * rows 1 & 4  -> "2,Beta,LA"   EXACT dup, NOT flagged (-> FN)
      * rows 2 & 5  -> "3,Gamma,*"   CONFLICTING (city differs),
                                      flagged by cluster C1 (-> left alone)
      * row 6       -> "9,Solo,DC"   unique
    """
    project = tmp_path / "rd"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "dups.csv"
    csv.write_text(
        "id,name,city\n"
        "1,Acme,NYC\n"     # row 0  exact dup of row 3
        "2,Beta,LA\n"      # row 1  exact dup of row 4 (uncovered)
        "3,Gamma,SF\n"     # row 2  conflicting with row 5
        "1,Acme,NYC\n"     # row 3  exact dup of row 0
        "2,Beta,LA\n"      # row 4  exact dup of row 1 (uncovered)
        "3,Gamma,LA\n"     # row 5  conflicting with row 2 (city differs)
        "9,Solo,DC\n",     # row 6  unique
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260604T223616Z_rowdup"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )

    def _dup_cluster(cid: str, row: int, affected: list[int]) -> dict:
        return {
            "cluster_id": cid,
            "source": "dups.csv",
            "row": row,
            "columns": [],
            "row_values": {},
            "error_types": ["row_duplication"],
            "error_type_categories": ["Multi-Row / Inter-Record"],
            "evidence_count": 1,
            "detections": [{
                "issue_id": f"i-{cid}",
                "error_type": "row_duplication",
                "columns": [],
                "affected_rows": affected,
            }],
        }

    (rd / "clusters.json").write_text(json.dumps([
        _dup_cluster("dups.csv__r00000-c000", 0, [0, 3]),  # exact -> TP
        _dup_cluster("dups.csv__r00002-c000", 2, [2, 5]),  # conflict -> skip
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_dry_run_creates_tp_and_fn(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-row-duplication",
            json={"dry_run": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is True
        assert body["scan_dataset"] is True
        # Cluster C0 (exact dup) -> create_tp; rows 1 & 4 (uncovered
        # exact dup) -> create_fn. Cluster C1 (conflicting) untouched.
        assert body["to_create_tp"] == 1
        assert body["to_create_fn"] == 2
        assert body["to_tag"] == 0
        assert body["to_write"] == 3
        assert body["dup_groups"] == 2
        assert body["dup_rows"] == 4
        # Nothing written on dry-run.
        assert client.get(f"/api/runs/{run_id}/labels").json()["history_count"] == 0


def test_apply_writes_tp_and_row_level_fn(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-row-duplication", json={},
        ).json()
        assert r["created_tp"] == 1
        assert r["created_fn"] == 2
        assert r["tagged"] == 0

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        # Exact-dup cluster -> TP + TRIVIAL.
        tp = latest["dups.csv__r00000-c000"]
        assert tp["status"] == "tp"
        assert tp["target"] == "cluster"
        assert "TRIVIAL" in tp["note"]
        # Conflicting-dup cluster -> never touched.
        assert "dups.csv__r00002-c000" not in latest
        # Two ROW-LEVEL FN cell labels (empty column) for rows 1 & 4.
        fn_cells = [
            lab for lab in latest.values()
            if lab["target"] == "cell" and lab["status"] == "fn"
        ]
        assert len(fn_cells) == 2
        assert sorted(c["row"] for c in fn_cells) == [1, 4]
        for c in fn_cells:
            assert (c["column"] or "") == ""
            assert "TRIVIAL" in c["note"]


def test_idempotent(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        first = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-row-duplication", json={},
        ).json()
        assert first["created_tp"] + first["created_fn"] == 3
        # Re-run: every duplicate already carries TRIVIAL -> no writes.
        second = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-row-duplication", json={},
        ).json()
        assert second["created_tp"] == 0
        assert second["created_fn"] == 0
        assert second["tagged"] == 0
        assert second["to_write"] == 0
        assert second["skipped_already_tagged"] == 3


def test_negative_verdict_suppresses_covered_group(tmp_path: Path) -> None:
    """An operator fp on the exact-dup cluster means 'not a duplicate':
    the scan leaves the cluster alone and never FN's its covered rows."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "dups.csv__r00000-c000",
            "source": "dups.csv", "row": 0, "status": "fp",
        })
        assert r.status_code == 200, r.text

        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-row-duplication",
            json={"dry_run": True},
        ).json()
        assert body["to_create_tp"] == 0
        assert body["skipped_negative_verdict"] == 1
        # Rows 1 & 4 (uncovered) are still FN candidates.
        assert body["to_create_fn"] == 2
        # No candidate targets rows 0 or 3 (the fp-dismissed group).
        touched = {c["row"] for c in body["candidates"]}
        assert 0 not in touched and 3 not in touched


def test_conflicting_dup_not_tagged(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-row-duplication",
            json={"dry_run": True},
        ).json()
        # The conflicting cluster (rows 2 & 5) must not appear and its
        # rows must not be FN'd (they are not byte-identical).
        ids = {c["target_id"] for c in body["candidates"]}
        rows = {c["row"] for c in body["candidates"]}
        assert "dups.csv__r00002-c000" not in ids
        assert 2 not in rows and 5 not in rows


def test_source_filter_404(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-row-duplication",
            json={"source": "nope.csv"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------
# Export: a row-level FN (column="") materialises as a row_duplication
# "error" benchmark cell.
# ---------------------------------------------------------------------


def test_row_level_fn_exports_as_row_duplication_error(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(Label(
        target=LabelTarget.CELL,
        target_id=cell_target_id("dups.csv", 1, ""),
        source="dups.csv",
        row=1,
        column="",
        columns=[],
        status=LabelStatus.FN,
        note="(auto: exact-duplicate row) TRIVIAL",
        annotator="t",
    ))
    cells = materialize_label_benchmark([], store)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.source == "dups.csv"
    assert cell.row == 1
    assert cell.column == ""
    assert cell.label == "error"
    assert cell.original_value == ""
    assert cell.error_types == ["row_duplication"]
    assert cell.cluster_ids == []
    assert "TRIVIAL" in cell.note
