"""POST /api/runs/{run_id}/labels/tag-majority-diff-row-duplication
auto-marks every ``row_duplication`` cluster whose flagged rows differ in
a MAJORITY of columns as a FALSE POSITIVE -- a genuine duplicate shares
most of its column values, so differing in more than half the columns
means the rows are distinct records, not a duplicate.

Mirrors ``tests/test_tag_timelike_row_duplication.py``: a small synthetic
run with hand-crafted ``row_duplication`` clusters, a TestClient flow,
and assertions on the dry-run preview + the persisted verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.routers.propagation import _row_diff_columns


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    """Tiny run with one 4-column CSV holding three near-duplicate pairs:

      * rows 0 & 1 -> differ in 3 of 4 columns (name, amount, city)
                      -> MAJORITY diff, flagged by cluster C0  (-> FP)
      * rows 2 & 3 -> differ in 1 of 4 columns (amount only)
                      -> minority diff, flagged by cluster C1  (-> keep)
      * rows 4 & 5 -> differ in 2 of 4 columns (exactly half)
                      -> NOT a strict majority, flagged by C2  (-> keep)
    """
    project = tmp_path / "md"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "dups.csv"
    csv.write_text(
        "id,name,amount,city\n"
        "1,Acme,100,NYC\n"      # row 0
        "1,Beta,200,LA\n"       # row 1  differs in name+amount+city (3/4)
        "2,Gamma,100,SF\n"      # row 2
        "2,Gamma,200,SF\n"      # row 3  differs in amount only (1/4)
        "3,Delta,100,NYC\n"     # row 4
        "3,Delta,200,LA\n",     # row 5  differs in amount+city (2/4 = half)
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260615T000000Z_majdup"
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
        _dup_cluster("dups.csv__r00000-c000", 0, [0, 1]),  # 3/4 -> FP
        _dup_cluster("dups.csv__r00002-c000", 2, [2, 3]),  # 1/4 -> keep
        _dup_cluster("dups.csv__r00004-c000", 4, [4, 5]),  # 2/4 -> keep
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


# ---------------------------------------------------------------------
# Endpoint: dry-run preview.
# ---------------------------------------------------------------------


def test_dry_run_flags_only_strict_majority(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-majority-diff-row-duplication",
            json={"dry_run": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is True
        assert body["ratio"] == 0.5
        assert body["clusters_scanned"] == 3
        # Only C0 (3/4 columns differ) is a candidate. C1 (1/4) and C2
        # (exactly 2/4 = half, not a strict majority) must not appear.
        assert body["to_create_fp"] == 1
        assert body["to_write"] == 1
        ids = {c["target_id"] for c in body["candidates"]}
        assert ids == {"dups.csv__r00000-c000"}
        cand = body["candidates"][0]
        assert cand["action"] == "create_fp"
        assert cand["n_cols"] == 4
        assert cand["n_diff"] == 3
        assert cand["diff_ratio"] == 0.75
        assert set(cand["diff_columns"]) == {"name", "amount", "city"}
        # Nothing written on dry-run.
        assert client.get(
            f"/api/runs/{run_id}/labels"
        ).json()["history_count"] == 0


# ---------------------------------------------------------------------
# Endpoint: apply writes a cluster FP verdict.
# ---------------------------------------------------------------------


def test_apply_writes_cluster_fp(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-majority-diff-row-duplication",
            json={},
        ).json()
        assert r["created_fp"] == 1

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        fp = latest["dups.csv__r00000-c000"]
        assert fp["status"] == "fp"
        assert fp["target"] == "cluster"
        assert "majority of columns" in fp["note"].lower()
        # Minority / exactly-half clusters are never touched.
        assert "dups.csv__r00002-c000" not in latest
        assert "dups.csv__r00004-c000" not in latest


# ---------------------------------------------------------------------
# Endpoint: a custom threshold lowers the bar.
# ---------------------------------------------------------------------


def test_threshold_includes_exactly_half(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # threshold 0.4 -> n_diff > 1.6 -> >=2 columns qualifies, so the
        # exactly-half cluster C2 (2/4) now also flags, plus C0 (3/4).
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-majority-diff-row-duplication",
            json={"dry_run": True, "threshold": 0.4},
        ).json()
        assert body["ratio"] == 0.4
        ids = {c["target_id"] for c in body["candidates"]}
        assert ids == {"dups.csv__r00000-c000", "dups.csv__r00004-c000"}
        assert body["to_create_fp"] == 2


def test_threshold_out_of_range_400(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        for bad in (1.0, 1.5, -0.1):
            r = client.post(
                f"/api/runs/{run_id}/labels/tag-majority-diff-row-duplication",
                json={"threshold": bad},
            )
            assert r.status_code == 400, (bad, r.text)


# ---------------------------------------------------------------------
# Endpoint: idempotent on re-run.
# ---------------------------------------------------------------------


def test_idempotent(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        first = client.post(
            f"/api/runs/{run_id}/labels/tag-majority-diff-row-duplication",
            json={},
        ).json()
        assert first["created_fp"] == 1
        second = client.post(
            f"/api/runs/{run_id}/labels/tag-majority-diff-row-duplication",
            json={},
        ).json()
        assert second["created_fp"] == 0
        assert second["to_write"] == 0
        assert second["skipped_already_tagged"] == 1


# ---------------------------------------------------------------------
# Endpoint: a pre-existing human verdict is never overridden.
# ---------------------------------------------------------------------


def test_existing_verdict_not_overridden(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Operator already ruled the majority-diff cluster a real dup.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "dups.csv__r00000-c000",
            "source": "dups.csv", "row": 0, "status": "tp",
        })
        assert r.status_code == 200, r.text

        body = client.post(
            f"/api/runs/{run_id}/labels/tag-majority-diff-row-duplication",
            json={"dry_run": True},
        ).json()
        assert body["to_create_fp"] == 0
        assert body["skipped_existing_verdict"] == 1
        cand = next(
            c for c in body["candidates"]
            if c["target_id"] == "dups.csv__r00000-c000"
        )
        assert cand["action"] == "skip_existing"
        assert cand["status"] == "tp"

        # Applying must NOT flip the operator's tp to fp.
        client.post(
            f"/api/runs/{run_id}/labels/tag-majority-diff-row-duplication",
            json={},
        )
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["dups.csv__r00000-c000"]["status"] == "tp"


def test_source_filter_404(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-majority-diff-row-duplication",
            json={"source": "nope.csv"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------
# Unit: the differing-columns helper.
# ---------------------------------------------------------------------


def test_row_diff_columns_counts_blank_vs_value() -> None:
    frame = pd.DataFrame({
        "a": ["x", "x"],     # same -> not counted
        "b": ["1", "2"],     # differ -> counted
        "c": ["", "y"],      # blank vs value -> counted (genuine diff)
        "d": ["", ""],       # both blank -> not counted
    })
    assert _row_diff_columns(frame, [0, 1]) == ["b", "c"]


def test_row_diff_columns_multi_row() -> None:
    frame = pd.DataFrame({
        "a": ["x", "x", "x"],   # all same -> not counted
        "b": ["1", "1", "2"],   # one differs -> counted
    })
    assert _row_diff_columns(frame, [0, 1, 2]) == ["b"]
