"""POST /api/runs/{run_id}/labels/tag-trivial-phone-format auto-detects
phone-format columns, SKIPs every cluster touching them, and writes a
cell-level FN label with the column-majority canonical form for every
row whose surface format differs.

Mirrors the style of test_tag_trivial_whitespace.py: synthetic run,
TestClient walk-through, then asserts the persisted append-only result.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.schema import cell_target_id


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    """Tiny run with a phone column and a non-phone column.

    Phone column ``contact`` (column-majority is ``(DDD) DDD-DDDD``):
      * row 0: ``(575) 433-0200``  -- canonical, no correction
      * row 1: ``(575) 433-0201``  -- canonical
      * row 2: ``(575) 4330202``   -- missing space, needs fix
      * row 3: ``575-433-0203``    -- different layout, needs fix
      * row 4: ``(575) 433-0204``  -- canonical
      * row 5: ``(575) 433-0205``  -- canonical
      * row 6: ``(575) 433-0206``  -- canonical
      * row 7: ``(575) 433-0207``  -- canonical

    Non-phone column ``name``: free-text values; the detector must
    leave it alone entirely (no SKIP cluster, no FN cell).

    Two clusters are pre-seeded: one on the phone column (must end up
    SKIPped) and one on the name column (must NOT be touched).
    """
    project = tmp_path / "phone"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "contacts.csv"
    csv.write_text(
        "id,name,contact\n"
        "1,Alice,(575) 433-0200\n"
        "2,Bob,(575) 433-0201\n"
        "3,Carol,\"(575) 4330202\"\n"
        "4,Dan,575-433-0203\n"
        "5,Eve,(575) 433-0204\n"
        "6,Frank,(575) 433-0205\n"
        "7,Grace,(575) 433-0206\n"
        "8,Henry,(575) 433-0207\n",
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260514T000000Z_phone"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text(json.dumps([
        # Cluster on the phone column -- the endpoint must SKIP it.
        {"cluster_id": "contacts.csv__r00002-c000", "source": "contacts.csv",
         "row": 2, "columns": ["contact"],
         "row_values": {"id": "3", "name": "Carol", "contact": "(575) 4330202"},
         "error_types": ["inconsistency"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
        # Cluster on the name column -- must remain untouched.
        {"cluster_id": "contacts.csv__r00000-c001", "source": "contacts.csv",
         "row": 0, "columns": ["name"],
         "row_values": {"id": "1", "name": "Alice", "contact": "(575) 433-0200"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_dry_run_returns_candidates_without_writing(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        before = client.get(f"/api/runs/{run_id}/labels").json()
        before_history = before["history_count"]

        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-phone-format",
            json={"dry_run": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()

        # One phone column detected.
        cols = [c for c in body["columns"] if c["column"] == "contact"]
        assert len(cols) == 1
        assert cols[0]["template"] == "(DDD) DDD-DDDD"
        assert cols[0]["digit_count"] == 10
        assert cols[0]["cells_to_correct"] == 2  # rows 2 and 3
        assert cols[0]["clusters_to_skip"] == 1  # the contact-column cluster

        # Two cell candidates: rows 2 and 3, with correct canonicals.
        cells = body["cell_candidates"]
        assert len(cells) == 2
        by_row = {c["row"]: c for c in cells}
        assert by_row[2]["canonical"] == "(575) 433-0202"
        assert by_row[2]["original_value"] == "(575) 4330202"
        assert by_row[3]["canonical"] == "(575) 433-0203"
        assert by_row[3]["original_value"] == "575-433-0203"

        # Only the phone-column cluster should be in cluster_candidates.
        clusters = body["cluster_candidates"]
        assert {c["target_id"] for c in clusters} == {
            "contacts.csv__r00002-c000"
        }
        assert all(not c["already_tagged"] for c in clusters)

        # Dry run: no writes, history unchanged.
        after = client.get(f"/api/runs/{run_id}/labels").json()
        assert after["history_count"] == before_history


def test_apply_writes_fn_cells_and_skip_clusters(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-phone-format", json={},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cells_written"] == 2
        assert body["clusters_written"] == 1

        labels = client.get(f"/api/runs/{run_id}/labels").json()
        latest = labels["latest"]

        # Cell labels: rows 2 and 3 should now carry FN with canonical
        # corrected_value and the phone_format tag in the note.
        for row, canonical in [(2, "(575) 433-0202"), (3, "(575) 433-0203")]:
            tid = cell_target_id("contacts.csv", row, "contact")
            assert tid in latest, f"missing cell label for row {row}"
            rec = latest[tid]
            assert rec["status"] == "fn"
            assert rec["corrected_value"] == canonical
            assert "TRIVIAL phone_format" in (rec.get("note") or "")

        # Cluster label: the contact cluster should be SKIPped with
        # the phone_format tag; the name cluster should still be
        # unlabelled.
        skip_id = "contacts.csv__r00002-c000"
        untouched_id = "contacts.csv__r00000-c001"
        assert skip_id in latest
        assert latest[skip_id]["status"] == "skip"
        assert "TRIVIAL phone_format" in (latest[skip_id].get("note") or "")
        assert untouched_id not in latest


def test_apply_is_idempotent(tmp_path: Path) -> None:
    """Running the endpoint twice writes the same labels once -- the
    second pass detects the existing PHONE_TAG tag and skips."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        first = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-phone-format", json={},
        ).json()
        assert first["cells_written"] == 2
        assert first["clusters_written"] == 1

        second = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-phone-format", json={},
        ).json()
        assert second["cells_written"] == 0
        assert second["clusters_written"] == 0
        # The candidates list still surfaces the rows, but they are
        # all flagged already_tagged.
        assert all(c["already_tagged"] for c in second["cell_candidates"])
        assert all(c["already_tagged"] for c in second["cluster_candidates"])


def test_skips_columns_below_threshold(tmp_path: Path) -> None:
    """A column with only 2/8 phone-shaped cells (25%) is well below
    the 80% default threshold; the endpoint must NOT touch it."""
    project = tmp_path / "phone2"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    (project / "data").mkdir()
    csv = project / "data" / "mixed.csv"
    csv.write_text(
        "id,mixed\n"
        "1,Alice\n2,Bob\n3,Carol\n4,Dan\n5,Eve\n6,Frank\n"
        "7,(575) 433-0200\n8,(575) 4330201\n",
        encoding="utf-8",
    )
    (project / "outputs").mkdir()
    run_id = "20260514T000000Z_mixed"
    rd = project / "outputs" / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-phone-format",
            json={"dry_run": True},
        )
        assert r.status_code == 200
        body = r.json()
        # No phone columns detected at the 80% threshold.
        assert body["columns"] == []
        assert body["cell_candidates"] == []
        assert body["cluster_candidates"] == []
        # Lowering the threshold should now detect the column.
        r2 = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-phone-format",
            json={"dry_run": True, "threshold": 0.20, "min_phone_rows": 2},
        )
        body2 = r2.json()
        assert len(body2["columns"]) == 1
        assert body2["columns"][0]["column"] == "mixed"


def test_source_filter_scopes_scan(tmp_path: Path) -> None:
    """``source`` payload key restricts the scan to one CSV."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Use a non-existent source -> 404.
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-phone-format",
            json={"source": "nope.csv", "dry_run": True},
        )
        assert r.status_code == 404
        # Use the real source -> works as before.
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-phone-format",
            json={"source": "contacts.csv", "dry_run": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["cell_candidates"]) == 2
