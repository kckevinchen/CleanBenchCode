"""POST /api/runs/{run_id}/labels/tag-trivial-numeric-format detects columns
of plain numbers written in mixed thousands-separator styles and writes a
cell-level FN label carrying the column-majority canonical value for every
deviant cell.

Mirrors the style of test_tag_trivial_phone_format.py: synthetic run,
TestClient walk-through, then asserts the persisted append-only result.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.schema import cell_target_id


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    """Tiny run modelled on the real bug report: a column of plain integers
    with a lone thousands-comma minority.

      * column ``callers`` (majority plain): rows 0-6 plain, row 5 is
        ``328,670`` (grouped) -> the one deviant, fix ``328670``.
      * column ``small`` (all sub-1000): neutral, nothing to flag.
      * column ``name``: free-text, ignored by the numeric detector.
    """
    project = tmp_path / "numeric"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "calls.csv"
    csv.write_text(
        "id,name,callers,small\n"
        "1,Alice,383481,12\n"
        "2,Bob,400262,34\n"
        "3,Carol,369435,56\n"
        "4,Dan,429452,78\n"
        "5,Eve,299884,90\n"
        "6,Frank,\"328,670\",11\n"
        "7,Grace,291132,22\n"
        "8,Henry,300000,33\n",
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260701T000000Z_numeric"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_dry_run_returns_candidates_without_writing(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        before = client.get(f"/api/runs/{run_id}/labels").json()
        before_history = before["history_count"]

        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-numeric-format",
            json={"dry_run": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()

        # Only the ``callers`` column has a majority/minority split.
        cols = body["columns"]
        assert len(cols) == 1
        assert cols[0]["column"] == "callers"
        assert cols[0]["majority_style"] == "plain"
        assert cols[0]["cells_to_correct"] == 1

        cells = body["cell_candidates"]
        assert len(cells) == 1
        assert cells[0]["row"] == 5
        assert cells[0]["original_value"] == "328,670"
        assert cells[0]["canonical"] == "328670"
        assert cells[0]["style"] == "grouped"
        assert cells[0]["already_tagged"] is False
        assert cells[0]["has_other_label"] is False

        # Dry run: no writes.
        after = client.get(f"/api/runs/{run_id}/labels").json()
        assert after["history_count"] == before_history


def test_apply_writes_fn_cell_with_canonical(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-numeric-format", json={},
        )
        assert r.status_code == 200, r.text
        assert r.json()["cells_written"] == 1

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        tid = cell_target_id("calls.csv", 5, "callers")
        rec = latest[tid]
        assert rec["target"] == "cell"
        assert rec["status"] == "fn"
        assert rec["corrected_value"] == "328670"
        # The note must NOT carry the bare TRIVIAL token (so the
        # convert-to-skip sweep never touches these findings).
        assert "TRIVIAL" not in (rec["note"] or "").upper()
        assert "numeric_format" in (rec["note"] or "").lower()


def test_apply_is_idempotent(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        first = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-numeric-format", json={},
        ).json()
        assert first["cells_written"] == 1

        second = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-numeric-format", json={},
        ).json()
        assert second["cells_written"] == 0
        assert second["skipped_already_tagged"] == 1


def test_existing_human_verdict_is_not_clobbered(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Operator already ruled the deviant cell FP.
        tid = cell_target_id("calls.csv", 5, "callers")
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": tid,
            "source": "calls.csv", "row": 5, "column": "callers",
            "status": "fp", "note": "reviewed",
        })
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-numeric-format", json={},
        ).json()
        assert body["cells_written"] == 0
        assert body["skipped_existing"] == 1

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest[tid]["status"] == "fp"
        assert latest[tid]["note"] == "reviewed"


def test_source_filter_scopes_and_404s(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-numeric-format",
            json={"source": "nope.csv"},
        )
        assert r.status_code == 404
        r2 = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-numeric-format",
            json={"source": "calls.csv"},
        ).json()
        assert r2["cells_written"] == 1


def test_grouped_majority_flags_plain_deviant(tmp_path: Path) -> None:
    """The reverse direction: a grouped-majority column flags the lone plain
    (4+ digit) value and fixes it by ADDING thousands commas."""
    project = tmp_path / "grouped"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "g.csv"
    csv.write_text(
        "id,amount\n"
        "1,\"1,000\"\n"
        "2,\"2,345\"\n"
        "3,\"10,000\"\n"
        "4,3210\n"
        "5,\"9,999\"\n"
        "6,\"4,567\"\n"
        "7,\"7,777\"\n",
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260701T010000Z_grouped"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-numeric-format", json={},
        ).json()
        assert body["cells_written"] == 1
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        tid = cell_target_id("g.csv", 3, "amount")
        assert latest[tid]["corrected_value"] == "3,210"


def test_decimal_and_sign_columns_are_never_flagged(tmp_path: Path) -> None:
    """SAFETY end-to-end: a column of decimals / signed values that share
    digit sequences must NOT be flagged (the rule only touches commas)."""
    project = tmp_path / "safe"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "s.csv"
    csv.write_text(
        "id,dec,signed\n"
        "1,2.5,-0.3\n"
        "2,25,0.3\n"
        "3,3.7,-1.2\n"
        "4,41,4.5\n"
        "5,5.0,6.7\n"
        "6,6.0,-8.1\n"
        "7,7.7,9.0\n",
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260701T020000Z_safe"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-numeric-format", json={},
        ).json()
        assert body["cells_written"] == 0
        assert body["columns"] == []
