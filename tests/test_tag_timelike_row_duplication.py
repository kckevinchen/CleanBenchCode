"""POST /api/runs/{run_id}/labels/tag-timelike-row-duplication auto-marks
every ``row_duplication`` cluster whose flagged rows DIFFER in a
time-like column (name + parsed value) as a FALSE POSITIVE -- the inverse
of the exact-duplicate TRIVIAL->TP tagger.

Mirrors ``tests/test_tag_trivial_row_duplication.py``: a small synthetic
run with hand-crafted ``row_duplication`` clusters, a TestClient flow,
and assertions on the dry-run preview + the persisted append-only
verdicts. Also unit-tests the time-like predicates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.routers.propagation import (
    _is_timelike_name,
    _timelike_diff_columns,
    _values_look_datetime,
)


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    """Tiny run with one CSV holding three near-duplicate pairs:

      * rows 0 & 1 -> differ ONLY in ``Date`` (a time-like column)
                      -> FALSE POSITIVE, flagged by cluster C0
      * rows 2 & 3 -> differ ONLY in ``amount`` (NOT time-like)
                      -> left alone, flagged by cluster C1
      * rows 4 & 5 -> differ ONLY in ``MonthNo`` (name matches "month"
                      but bare-int values are NOT datetimes) -> left
                      alone (the value guard), flagged by cluster C2
    """
    project = tmp_path / "td"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "dups.csv"
    csv.write_text(
        "id,name,amount,Date,MonthNo\n"
        "1,Acme,100,2019-06-26,5\n"   # row 0
        "1,Acme,100,2019-06-27,5\n"   # row 1  Date differs vs row 0
        "2,Beta,100,2020-01-01,5\n"   # row 2
        "2,Beta,200,2020-01-01,5\n"   # row 3  amount differs vs row 2
        "3,Gamma,50,2021-03-03,5\n"   # row 4
        "3,Gamma,50,2021-03-03,8\n",  # row 5  MonthNo differs vs row 4
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260615T000000Z_timedup"
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
        _dup_cluster("dups.csv__r00000-c000", 0, [0, 1]),  # Date diff   -> FP
        _dup_cluster("dups.csv__r00002-c000", 2, [2, 3]),  # amount diff -> keep
        _dup_cluster("dups.csv__r00004-c000", 4, [4, 5]),  # MonthNo int -> keep
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


# ---------------------------------------------------------------------
# Endpoint: dry-run preview.
# ---------------------------------------------------------------------


def test_dry_run_flags_only_timelike_diff(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-timelike-row-duplication",
            json={"dry_run": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is True
        assert body["scan_dataset"] is True
        assert body["clusters_scanned"] == 3
        # Only C0 (Date diff) is a candidate. C1 (amount) and C2
        # (bare-int MonthNo) must not appear.
        assert body["to_create_fp"] == 1
        assert body["to_write"] == 1
        ids = {c["target_id"] for c in body["candidates"]}
        assert ids == {"dups.csv__r00000-c000"}
        cand = body["candidates"][0]
        assert cand["action"] == "create_fp"
        assert cand["time_columns"] == ["Date"]
        assert cand["samples"]["Date"] == ["2019-06-26", "2019-06-27"]
        assert cand["group_rows"] == [0, 1]
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
            f"/api/runs/{run_id}/labels/tag-timelike-row-duplication", json={},
        ).json()
        assert r["created_fp"] == 1

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        fp = latest["dups.csv__r00000-c000"]
        assert fp["status"] == "fp"
        assert fp["target"] == "cluster"
        assert "time-like column" in fp["note"].lower()
        # Non-time-diff clusters are never touched.
        assert "dups.csv__r00002-c000" not in latest
        assert "dups.csv__r00004-c000" not in latest


# ---------------------------------------------------------------------
# Endpoint: idempotent on re-run.
# ---------------------------------------------------------------------


def test_idempotent(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        first = client.post(
            f"/api/runs/{run_id}/labels/tag-timelike-row-duplication", json={},
        ).json()
        assert first["created_fp"] == 1
        second = client.post(
            f"/api/runs/{run_id}/labels/tag-timelike-row-duplication", json={},
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
        # Operator already ruled the Date-diff cluster a real duplicate.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "dups.csv__r00000-c000",
            "source": "dups.csv", "row": 0, "status": "tp",
        })
        assert r.status_code == 200, r.text

        body = client.post(
            f"/api/runs/{run_id}/labels/tag-timelike-row-duplication",
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
            f"/api/runs/{run_id}/labels/tag-timelike-row-duplication", json={},
        )
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["dups.csv__r00000-c000"]["status"] == "tp"


def test_source_filter_404(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-timelike-row-duplication",
            json={"source": "nope.csv"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------
# Unit: the time-like predicates.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("Date", True),
    ("UPDATE_DATE", True),
    ("Order Date", True),
    ("created_at", True),
    ("MonthNo", True),       # name matches; value guard handles the rest
    ("amount", False),
    ("city", False),
    ("id", False),
])
def test_is_timelike_name(name: str, expected: bool) -> None:
    assert _is_timelike_name(name) is expected


@pytest.mark.parametrize("values,expected", [
    (["2019-06-26", "2019-06-27"], True),
    (["2019-06-26 00:00:00.000", "2019-12-01 00:00:00.000"], True),
    (["18/11/2018", "1/01/2019"], True),   # day-first
    (["20/6", "21/6"], True),              # single-separator day/month
    (["10:30", "11:45"], True),
    (["5", "8"], False),                   # bare month ints
    (["100", "200"], False),               # quantities
    (["76134-USA", "95012-USA"], False),   # zip-ish, not a date
    (["383.1975", "400.0"], False),        # money
    ([], False),
])
def test_values_look_datetime(values: list[str], expected: bool) -> None:
    assert _values_look_datetime(values) is expected


def test_timelike_diff_columns_ignores_non_time_and_blank() -> None:
    import pandas as pd

    frame = pd.DataFrame({
        "amount": ["100", "100"],
        "Date": ["2019-06-26", "2019-06-27"],
        "Note": ["", "x"],
        "MonthNo": ["5", "8"],
    })
    # Only ``Date`` differs with parseable datetimes; ``MonthNo`` differs
    # but its bare ints are not datetimes; ``amount`` does not differ.
    assert _timelike_diff_columns(frame, [0, 1]) == ["Date"]
    # A date-vs-blank difference is a missing value, not a time diff.
    blank = pd.DataFrame({"Date": ["2019-06-26", ""]})
    assert _timelike_diff_columns(blank, [0, 1]) == []
