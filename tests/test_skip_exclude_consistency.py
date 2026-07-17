"""Skip / exclude consistency for the trivial value-anomaly scan.

Two coupled guarantees:

1. The whitespace / punctuation scan must NOT flag a cell that is inside an
   active exclusion (file / row / column). It writes a ``skip`` cell label for
   it instead (or converts an existing positive cell label to skip), so an
   out-of-scope cell consistently ends up ``skip`` and a re-scan leaves it.
2. Adding an exclusion AFTER tagging converts the in-scope FN cell labels to
   ``skip`` too (the cascade previously only touched clusters), and reverting
   the exclusion restores them.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.export import materialize_label_benchmark
from src.labeling.schema import cell_target_id
from src.labeling.store import LabelStore


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "se"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    (project / "data").mkdir()
    csv = project / "data" / "v.csv"
    # row 0 + row 2 carry a double-space whitespace anomaly; row 1 is clean.
    csv.write_text(
        'id,name\n1,"Acme  Co"\n2,Clean\n3,"Bee  Inc"\n', encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_se"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_scan_skips_excluded_row_and_writes_skip(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/exclusions",
                    json={"source": "v.csv", "scope": "rows", "rows": [0]})

        dry = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"dry_run": True, "scan_dataset": True},
        ).json()
        # Excluded row 0 -> skip_excluded; row 2 -> a normal FN.
        assert dry["to_skip_excluded"] == 1
        assert dry["to_create_fn"] == 1
        by_row = {c["row"]: c for c in dry["candidates"]}
        assert by_row[0]["action"] == "skip_excluded"
        assert by_row[0]["status"] == "skip"
        assert by_row[2]["action"] == "create_fn"

        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        )
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest[cell_target_id("v.csv", 0, "name")]["status"] == "skip"
        assert latest[cell_target_id("v.csv", 2, "name")]["status"] == "fn"

    # Export: excluded row 0 absent; row 2 present as an error.
    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    store = LabelStore(rd / "labels.jsonl")
    by_loc = {(c.source, c.row, c.column): c.label
              for c in materialize_label_benchmark(clusters, store)}
    assert ("v.csv", 0, "name") not in by_loc
    assert by_loc[("v.csv", 2, "name")] == "error"


def test_rescan_does_not_reflag_excluded_skip(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/exclusions",
                    json={"source": "v.csv", "scope": "rows", "rows": [0]})
        client.post(f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
                    json={"scan_dataset": True})
        # Re-scan: the excluded cell is already skip, so it is not re-flagged.
        again = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"dry_run": True, "scan_dataset": True},
        ).json()
        rows = {c["row"] for c in again["candidates"] if c["action"] == "skip_excluded"}
        # Row 0 is already skip -> reported under skipped_excluded, no new write.
        assert again["to_skip_excluded"] == 0
        assert 0 not in rows


def test_excluded_column_scope(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/exclusions",
                    json={"source": "v.csv", "scope": "columns",
                          "columns": ["name"]})
        dry = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"dry_run": True, "scan_dataset": True},
        ).json()
        # Both anomaly cells are in the excluded 'name' column -> skip, no FN.
        assert dry["to_create_fn"] == 0
        assert dry["to_skip_excluded"] == 2


def test_exclude_after_tag_converts_fn_cell_to_skip(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    tid0 = cell_target_id("v.csv", 0, "name")
    with TestClient(app) as client:
        # Tag FIRST -> FN cells on rows 0 and 2.
        client.post(f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
                    json={"scan_dataset": True})
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest[tid0]["status"] == "fn"

        # Exclude row 0 AFTER tagging -> the FN cell must flip to skip.
        exc = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"source": "v.csv", "scope": "rows", "rows": [0]},
        ).json()
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest[tid0]["status"] == "skip"
        # Row 2 (not excluded) stays FN.
        assert latest[cell_target_id("v.csv", 2, "name")]["status"] == "fn"

        # Revert the exclusion -> the cell-skip is tombstoned (back to unlabeled).
        exc_id = exc.get("id") or (exc.get("exclusion") or {}).get("id")
        assert exc_id
        client.delete(f"/api/runs/{run_id}/exclusions/{exc_id}")
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert tid0 not in latest
