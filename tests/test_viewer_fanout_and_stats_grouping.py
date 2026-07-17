"""Consistency between the CSV-viewer cell-state endpoint, the cluster
queue, and the stats bar.

1. The viewer's cluster fan-out must mark the SAME cells the benchmark
   exporter does: the cluster verdict's columns, not each detection's own
   columns (a multi-column relation detection used to mark cells in the
   viewer that the export never emitted).
2. The stats bar must count PER ISSUE the same way the queue collapses
   them: same-cell duplicate clusters that the queue dedups to one item
   must count once in the stats total too.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app


def _run_with(tmp_path: Path, name: str, csv_text: str, clusters: list) -> tuple[Path, str]:
    project = tmp_path / name
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "t.csv"
    csv.write_text(csv_text, encoding="utf-8")
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = f"20260605T000000Z_{name}"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(json.dumps({"config": {"csv_paths": [str(csv)]}}))
    (rd / "clusters.json").write_text(json.dumps(clusters))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_viewer_fanout_uses_cluster_columns_not_detection_columns(tmp_path: Path) -> None:
    """A cluster on column ``a`` whose detection lists columns ``[a, b]``
    and affected row 2 must fan only column ``a`` to row 2 in the viewer
    -- matching the benchmark export, which fans the cluster's columns."""
    project, run_id = _run_with(
        tmp_path, "fanout",
        "id,a,b\n1,X,P\n2,Y,Q\n3,Z,R\n",
        [{"cluster_id": "t.csv__r00000-c000", "source": "t.csv",
          "row": 0, "columns": ["a"],
          "row_values": {"id": "1", "a": "X", "b": "P"},
          "error_types": ["fd_violation"],
          "detections": [{"issue_id": "i1", "error_type": "fd_violation",
                          "columns": ["a", "b"], "affected_rows": [2]}]}],
    )
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "t.csv__r00000-c000",
            "source": "t.csv", "row": 0, "column": "a", "status": "tp",
        })
        assert r.status_code == 200, r.text

        cells = client.get(f"/api/runs/{run_id}/sources/t.csv/cells").json()["cells"]
        by_loc = {(c["row"], c["column"]): c["state"] for c in cells}
        # Anchor cell + the fanned column-a cell are marked tp.
        assert by_loc.get((0, "a")) == "tp"
        assert by_loc.get((2, "a")) == "tp"
        # Column b at the affected row is NOT marked: the cluster's
        # columns are [a], so the fan-out (like the export) skips b.
        assert (2, "b") not in by_loc


def test_stats_total_matches_queue_for_duplicate_cell_clusters(tmp_path: Path) -> None:
    """Two clusters resolving to the SAME (source, row, columns) cell are
    deduped to one queue item; the stats total must count them once too
    (previously they counted as two singletons -> stats drifted high)."""
    project, run_id = _run_with(
        tmp_path, "dupcell",
        "id,a,b\n1,X,P\n2,Y,Q\n",
        [
            {"cluster_id": "t.csv__r00000-c000a", "source": "t.csv",
             "row": 0, "columns": ["a", "b"],
             "row_values": {"id": "1", "a": "X", "b": "P"},
             "error_types": ["fd_violation"], "detections": []},
            {"cluster_id": "t.csv__r00000-c000b", "source": "t.csv",
             "row": 0, "columns": ["a", "b"],
             "row_values": {"id": "1", "a": "X", "b": "P"},
             "error_types": ["fd_violation"], "detections": []},
        ],
    )
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        queue = client.get(f"/api/runs/{run_id}/clusters").json()
        assert queue["total"] == 1  # the two are deduped in the queue

        stats = client.get(f"/api/runs/{run_id}/stats").json()
        # Stats counts the same single issue (not 2).
        assert stats["overall"]["total"] == 1
        assert stats["by_source"]["t.csv"]["total"] == 1
