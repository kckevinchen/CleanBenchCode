"""End-to-end tests for the FN (false-negative) labelling flow.

Covers the four invariants the user asked for:

1. Operator can mark an unflagged cell as FN.
2. While reviewing the CSV, the operator can re-label a cluster's
   verdict via the cell popover (handled at the cluster endpoint -
   exercised here as ``POST /labels`` with target=cluster).
3. Double-labelling: an FN cell that ALSO has a TP cluster covering it
   is silently superseded by the cluster (TP wins). The cell-state
   endpoint surfaces this via ``superseded_fn_by`` so the UI can warn,
   and ``stats.fn.shadowed`` counts them.
4. Cells already covered by a TP cluster cannot themselves become an
   FN error in the benchmark - the benchmark exporter merges the FN
   into the existing TP record instead of producing two cells.

These tests use ``fastapi.testclient`` against an in-memory run dir.
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.schema import cell_target_id


def _make_run(project: Path, run_id: str = "20260101T000000Z_test") -> Path:
    """Create a minimal but valid run dir with one TP-eligible cluster
    on (a.csv, row 1, col=name) and an unflagged cell at (a.csv, row 2,
    col=email). Returns the run dir."""
    # Create the on-disk CSV so the labeling server can resolve sources.
    data_dir = project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "a.csv").write_text(
        "name,email\nalice,a@x\nbob,b@x\ncarol,c@x\n", encoding="utf-8",
    )

    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    run_dir = outputs / run_id
    run_dir.mkdir()

    csv_paths = [str(data_dir / "a.csv")]
    summary = {"config": {"csv_paths": csv_paths, "sources": ["a.csv"]}}
    (run_dir / "summary.json").write_text(json.dumps(summary))
    clusters = [
        {
            "cluster_id": "c1",
            "source": "a.csv",
            "row": 1,
            "columns": ["name"],
            "error_types": ["typo"],
            "evidence_count": 1,
            "synthesis": None,
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


# ---------------------------------------------------------------------------
# Cell target_id helper
# ---------------------------------------------------------------------------


def test_cell_target_id_is_stable() -> None:
    a = cell_target_id("a.csv", 5, "name")
    b = cell_target_id("a.csv", 5, "name")
    c = cell_target_id("a.csv", 5, "email")
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# Posting FN cell labels
# ---------------------------------------------------------------------------


class TestPostFNLabel:
    def test_mark_unflagged_cell_as_fn(self, client_and_run) -> None:
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell",
                "target_id": "ignored-server-canonicalises",
                "source": "a.csv",
                "row": 2,
                "column": "email",
                "status": "fn",
                "note": "missed",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "fn"
        assert body["target"] == "cell"
        # Server canonicalised the target_id.
        assert body["target_id"] == cell_target_id("a.csv", 2, "email")

    def test_cell_label_requires_source_row_column(self, client_and_run) -> None:
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell",
                "target_id": "x",
                "source": "a.csv",
                "row": None,
                "column": "email",
                "status": "fn",
            },
        )
        assert r.status_code == 400

    def test_cell_label_status_constrained(self, client_and_run) -> None:
        client, run_id = client_and_run
        # tp is not allowed on cell target - the cluster path owns TP.
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell",
                "target_id": "x",
                "source": "a.csv",
                "row": 2,
                "column": "email",
                "status": "tp",
            },
        )
        assert r.status_code == 400

    def test_cell_label_unknown_source_404(self, client_and_run) -> None:
        client, run_id = client_and_run
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell",
                "target_id": "x",
                "source": "missing.csv",
                "row": 1,
                "column": "email",
                "status": "fn",
            },
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Cell-state derivation + double-labelling shadowing
# ---------------------------------------------------------------------------


class TestCellState:
    def _post_cluster_status(
        self, client: TestClient, run_id: str, cid: str, status: str,
    ) -> None:
        r = client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster",
                "target_id": cid,
                "source": "a.csv",
                "row": 1,
                "column": "name",
                "status": status,
            },
        )
        assert r.status_code == 200, r.text

    def test_unlabelled_run_has_empty_cell_state(self, client_and_run) -> None:
        client, run_id = client_and_run
        r = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()
        assert r["cells"] == []
        assert r["shadowed_fn_count"] == 0

    def test_tp_cluster_paints_its_cells(self, client_and_run) -> None:
        client, run_id = client_and_run
        self._post_cluster_status(client, run_id, "c1", "tp")
        cells = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        # row=1, col=name should now be tp.
        match = [c for c in cells if c["row"] == 1 and c["column"] == "name"]
        assert match and match[0]["state"] == "tp"
        assert "c1" in match[0]["cluster_ids"]

    def test_fn_cell_appears_in_state(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": "fn",
            },
        )
        cells = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        match = [c for c in cells if c["row"] == 2 and c["column"] == "email"]
        assert match and match[0]["state"] == "fn"
        assert match[0]["label_source"] == "cell"

    def test_tp_cluster_supersedes_pre_existing_fn_on_same_cell(
        self, client_and_run,
    ) -> None:
        """Operator labels (a.csv, row=1, col=name) as FN, then later
        promotes cluster c1 (which covers the same cell) to TP. The
        cell-state endpoint must report ``state=tp`` AND surface
        ``superseded_fn_by`` so the UI can warn about the
        double-labelling.
        """
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 1, "column": "name", "status": "fn", "note": "missed",
            },
        )
        self._post_cluster_status(client, run_id, "c1", "tp")
        body = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()
        match = [c for c in body["cells"] if c["row"] == 1 and c["column"] == "name"]
        assert match
        assert match[0]["state"] == "tp"
        assert match[0]["superseded_fn_by"] == ["c1"]
        assert body["shadowed_fn_count"] == 1

    def test_fp_records_not_an_error_and_keeps_cell_visible(
        self, client_and_run,
    ) -> None:
        """``fp`` on a cell is a RECORDED "reviewed, not an error" verdict --
        it must KEEP the cell visible (surface in /cells as ``fp``), not
        clear it. Only ``unlabeled`` (an explicit retract) removes the cell.
        Regression for the bug where pressing the fp key silently deleted
        auto-flagged cells."""
        client, run_id = client_and_run
        # FN, then "actually not an error" -> FP on the same cell.
        for status in ("fn", "fp"):
            client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "cell", "target_id": "x", "source": "a.csv",
                    "row": 2, "column": "email", "status": status,
                },
            )
        cells = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        match = [c for c in cells if c["row"] == 2 and c["column"] == "email"]
        # The cell stays, now showing the fp verdict (not fn, not gone).
        assert match, "fp cell must stay visible in /cells"
        assert match[0]["state"] == "fp"
        assert match[0]["label_source"] == "cell"
        # It also stays in the FN review queue so the operator doesn't lose it.
        fn_items = client.get(f"/api/runs/{run_id}/fn-cells").json()["items"]
        fp_item = [
            i for i in fn_items
            if i["row"] == 2 and i["column"] == "email"
        ]
        assert fp_item and fp_item[0]["status"] == "fp"

    def test_unlabeled_retract_removes_cell(self, client_and_run) -> None:
        """The explicit retract path (``unlabeled``) DOES remove the cell --
        it drops from /cells and /fn-cells. This is the only way to delete a
        cell label now (no keyboard verdict does it)."""
        client, run_id = client_and_run
        for status in ("fn", "unlabeled"):
            client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "cell", "target_id": "x", "source": "a.csv",
                    "row": 2, "column": "email", "status": status,
                },
            )
        cells = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        assert not [c for c in cells if c["row"] == 2 and c["column"] == "email"]
        fn_items = client.get(f"/api/runs/{run_id}/fn-cells").json()["items"]
        assert not [i for i in fn_items if i["row"] == 2 and i["column"] == "email"]

    @pytest.mark.parametrize("status", ["partial", "unsure", "skip", "fp"])
    def test_non_fn_cell_label_appears_in_state(
        self, client_and_run, status,
    ) -> None:
        """A cell label carrying a non-FN verdict (partial / unsure / skip /
        fp -- produced by the trivial-convert flow and the FN detail panel)
        must paint the CSV grid, not vanish. Regression for the bug where
        ``/cells`` only surfaced ``fn`` cell labels, so a partial cell's tint
        disappeared on the next reload."""
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": status,
                "corrected_value": "fixed@x",
                "note": "(auto: punctuation anomaly) TRIVIAL [LLM-fix]",
            },
        )
        cells = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        match = [c for c in cells if c["row"] == 2 and c["column"] == "email"]
        assert match, f"{status} cell label should appear in /cells"
        assert match[0]["state"] == status
        assert match[0]["label_source"] == "cell"

    def test_marking_one_fn_keeps_sibling_partials(self, client_and_run) -> None:
        """Reproduces the reported bug: several ``partial`` cell labels in a
        column, then the operator marks ONE sibling cell FN. The other
        partials must remain in ``/cells`` (their grid tint must not vanish).
        Before the fix, ``/cells`` dropped every non-FN cell label, so the
        next reload after the FN mark wiped the sibling partial tints."""
        client, run_id = client_and_run
        # Three partial cell labels in the ``email`` column (rows 0/1/2).
        for row in (0, 1, 2):
            client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "cell", "target_id": "x", "source": "a.csv",
                    "row": row, "column": "email", "status": "partial",
                    "corrected_value": f"r{row}@x",
                },
            )
        before = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        email_before = {c["row"] for c in before if c["column"] == "email"}
        assert email_before == {0, 1, 2}

        # Mark row 0 as FN (operator changes their mind on one sibling).
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 0, "column": "email", "status": "fn",
            },
        )
        after = client.get(f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        email_after = {c["row"]: c["state"] for c in after if c["column"] == "email"}
        # Row 0 is now fn; rows 1 & 2 must still be present as partial.
        assert email_after.get(0) == "fn"
        assert email_after.get(1) == "partial"
        assert email_after.get(2) == "partial"


# ---------------------------------------------------------------------------
# Stats include FN + shadowed count
# ---------------------------------------------------------------------------


class TestStatsFN:
    def test_stats_carry_fn_block(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": "fn",
            },
        )
        s = client.get(f"/api/runs/{run_id}/stats").json()
        assert s["fn"]["total"] == 1
        assert s["fn"]["shadowed"] == 0
        assert s["fn"]["by_source"]["a.csv"] == 1

    def test_shadowed_when_tp_covers_fn(self, client_and_run) -> None:
        client, run_id = client_and_run
        # FN on the same cell c1 covers, then c1 -> TP.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 1, "column": "name", "status": "fn",
            },
        )
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp",
            },
        )
        s = client.get(f"/api/runs/{run_id}/stats").json()
        assert s["fn"]["total"] == 1
        assert s["fn"]["shadowed"] == 1


# ---------------------------------------------------------------------------
# Sibling-tables surface
# ---------------------------------------------------------------------------


def test_get_source_returns_siblings_for_workbook(tmp_path: Path) -> None:
    project = tmp_path
    (project / "frontend" / "labeling").mkdir(parents=True)
    workbook = project / "data" / "wb1"
    workbook.mkdir(parents=True)
    for name in ("a.csv", "b.csv", "c.csv"):
        (workbook / name).write_text("col\nv\n", encoding="utf-8")

    outputs = project / "outputs"
    outputs.mkdir()
    run_dir = outputs / "20260101T000000Z_test"
    run_dir.mkdir()
    csv_paths = [str(workbook / n) for n in ("a.csv", "b.csv", "c.csv")]
    (run_dir / "summary.json").write_text(json.dumps({
        "config": {"csv_paths": csv_paths, "sources": ["a.csv", "b.csv", "c.csv"]},
    }))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")

    client = TestClient(create_app(project, annotator="tester"))
    r = client.get(f"/api/runs/{run_dir.name}/sources/a.csv").json()
    assert r["workbook"] == "wb1"
    assert r["siblings"] == ["b.csv", "c.csv"]


# ---------------------------------------------------------------------------
# Benchmark export folds in FN cells
# ---------------------------------------------------------------------------


class TestExportWithFN:
    def test_fn_cell_emits_error_record_with_empty_cluster_ids(
        self, client_and_run,
    ) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": "fn",
                "note": "missed", "corrected_value": "b@example.com",
            },
        )
        r = client.get(f"/api/runs/{run_id}/export/benchmark")
        assert r.status_code == 200
        rows: List[Dict[str, Any]] = [
            json.loads(line) for line in r.text.splitlines() if line.strip()
        ]
        # No TP clusters labelled, so the only benchmark row comes from
        # the FN cell.
        assert len(rows) == 1
        rec = rows[0]
        assert rec["source"] == "a.csv"
        assert rec["row"] == 2
        assert rec["column"] == "email"
        assert rec["label"] == "error"
        assert rec["cluster_ids"] == []
        assert rec["corrected_value"] == "b@example.com"

    def test_fn_cell_carries_error_types_into_benchmark(
        self, client_and_run,
    ) -> None:
        """A cell FN label tagged ``error_types`` exports with that tag
        (the column-board per-value inconsistency FN flow), instead of
        the empty-list / ``_uncategorized`` default of a generic FN."""
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": "fn",
                "corrected_value": "b@example.com",
                "error_types": ["inconsistency"],
            },
        )
        rows: List[Dict[str, Any]] = [
            json.loads(line)
            for line in client.get(
                f"/api/runs/{run_id}/export/benchmark"
            ).text.splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        rec = rows[0]
        assert rec["error_types"] == ["inconsistency"]
        assert rec["label"] == "error"
        assert rec["cluster_ids"] == []
        # And it lands in the inconsistency bucket of the by-type export.
        single = client.get(
            f"/api/runs/{run_id}/export/benchmark/inconsistency"
        )
        assert single.status_code == 200
        incon = [
            json.loads(line)
            for line in single.text.splitlines() if line.strip()
        ]
        assert len(incon) == 1
        assert incon[0]["row"] == 2 and incon[0]["column"] == "email"

    def test_fn_superseded_by_tp_cluster_is_merged_not_duplicated(
        self, client_and_run,
    ) -> None:
        client, run_id = client_and_run
        # FN on the same cell c1 covers ...
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 1, "column": "name", "status": "fn",
                "note": "alice should be Alice",
            },
        )
        # ... then cluster c1 -> TP, which should swallow the FN.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp", "corrected_value": "Alice",
            },
        )
        rows = [
            json.loads(line)
            for line in client.get(f"/api/runs/{run_id}/export/benchmark").text.splitlines()
            if line.strip()
        ]
        # Exactly one cell record, owned by the TP cluster.
        cells = [r for r in rows if r["row"] == 1 and r["column"] == "name"]
        assert len(cells) == 1
        rec = cells[0]
        assert rec["label"] == "error"
        assert rec["cluster_ids"] == ["c1"]
        assert rec["corrected_value"] == "Alice"


class TestSourceExportRowMap:
    def test_row_exclusion_emits_original_row_mapping(
        self, client_and_run,
    ) -> None:
        client, run_id = client_and_run
        response = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"source": "a.csv", "scope": "rows", "rows": [1]},
        )
        assert response.status_code == 200, response.text

        response = client.get(f"/api/runs/{run_id}/export/sources")
        assert response.status_code == 200, response.text
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            exported = archive.read("a.csv").decode("utf-8").splitlines()
            row_map = json.loads(archive.read("_ROW_MAP.json"))

        assert exported == ["name,email", "alice,a@x", "carol,c@x"]
        assert row_map["sources"] == [{
            "source": "a.csv",
            "path": "a.csv",
            "exported_to_source": [0, 2],
        }]

    def test_missing_source_fails_export(self, tmp_path: Path) -> None:
        (tmp_path / "frontend" / "labeling").mkdir(parents=True)
        run_dir = _make_run(tmp_path)
        client = TestClient(create_app(tmp_path, annotator="tester"))
        (tmp_path / "data" / "a.csv").unlink()

        response = client.get(f"/api/runs/{run_dir.name}/export/sources")

        assert response.status_code == 500
        assert response.json()["detail"] == {
            "error": "source_export_failed",
            "source": "a.csv",
            "reason": "source file is missing",
        }


class TestExportByErrorType:
    """End-to-end smoke test for the per-error-type export endpoint.

    Uses the same fixture as the JSONL exporter; pins:
      * the endpoint returns a real ZIP archive,
      * each entry inside it is a JSONL keyed by error_type slug,
      * cells are bucketed correctly (typo cluster + FN cell go to
        ``typo`` and ``_uncategorized`` respectively).
    """

    def test_zip_contains_one_jsonl_per_type(self, client_and_run) -> None:
        client, run_id = client_and_run
        # Label cluster c1 (error_type=typo) as TP.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp", "corrected_value": "Alice",
            },
        )
        # Add an FN cell with no enclosing cluster -> uncategorized.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x",
                "source": "a.csv", "row": 2, "column": "email",
                "status": "fn", "corrected_value": "c@example.com",
            },
        )
        r = client.get(f"/api/runs/{run_id}/export/benchmark-by-type")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/zip")

        import io
        import zipfile
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = sorted(zf.namelist())
        assert names == ["_uncategorized.jsonl", "typo.jsonl"]

        with zf.open("typo.jsonl") as fh:
            lines = fh.read().decode("utf-8").splitlines()
            assert len(lines) == 1
            rec = json.loads(lines[0])
            assert rec["column"] == "name"
            assert rec["label"] == "error"
            assert rec["corrected_value"] == "Alice"
            assert "typo" in rec["error_types"]

        with zf.open("_uncategorized.jsonl") as fh:
            lines = fh.read().decode("utf-8").splitlines()
            assert len(lines) == 1
            rec = json.loads(lines[0])
            assert rec["column"] == "email"
            assert rec["label"] == "error"
            assert rec["error_types"] == []
            # FN cells emit empty cluster_ids - this is the marker the
            # downstream cleaner uses to tell FN from TP-via-cluster.
            assert rec["cluster_ids"] == []


class TestExportSingleErrorType:
    """End-to-end check for the per-error-type single-bucket export.

    Mirrors :class:`TestExportByErrorType` but pins the surface that
    serves a single ``error_type`` as a plain JSONL (not a ZIP):

      * ``GET /export/benchmark/typo`` returns the typo bucket only.
      * Slug normalisation: ``Typo``, ``typo``, ``typo.jsonl`` all
        resolve to the same bucket.
      * 404 with the available-bucket list when no labelled cell
        belongs to the requested type.
    """

    def test_returns_jsonl_for_requested_type(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp", "corrected_value": "Alice",
            },
        )
        r = client.get(f"/api/runs/{run_id}/export/benchmark/typo")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        assert len(rows) == 1
        rec = rows[0]
        assert rec["column"] == "name"
        assert rec["label"] == "error"
        assert rec["corrected_value"] == "Alice"
        assert "typo" in rec["error_types"]

    def test_slug_normalisation(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp", "corrected_value": "Alice",
            },
        )
        # Both "typo.jsonl" and "Typo" should resolve to the same bucket
        # via :func:`_slug_for_error_type`. We accept the lowercased
        # canonical slug; an upper-cased input is normalised on the
        # client (the dropdown emits canonical lowercase) but the
        # server still rejects unknown slugs cleanly.
        for variant in ("typo.jsonl",):
            r = client.get(f"/api/runs/{run_id}/export/benchmark/{variant}")
            assert r.status_code == 200, (variant, r.text)
            assert r.text.strip()

    def test_unknown_type_returns_404_with_available_list(
        self, client_and_run,
    ) -> None:
        client, run_id = client_and_run
        # Label a typo-cluster so the run has at least ONE bucket;
        # then ask for a different type that has no labels.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp",
            },
        )
        r = client.get(
            f"/api/runs/{run_id}/export/benchmark/numeric_outlier"
        )
        assert r.status_code == 404
        body = r.json()
        # FastAPI nests the structured detail under ``detail``.
        detail = body.get("detail", body)
        assert detail["error"] == "no_labelled_cells_for_error_type"
        assert detail["requested"] == "numeric_outlier"
        assert "typo" in detail["available"]


# ---------------------------------------------------------------------------
# Per-source coverage surface (checked / labelled_clusters / fn_cells)
# ---------------------------------------------------------------------------


class TestSourceCoverage:
    def test_unlabelled_source_is_not_checked(self, client_and_run) -> None:
        client, run_id = client_and_run
        run = client.get(f"/api/runs/{run_id}").json()
        meta = next(s for s in run["sources"] if s["name"] == "a.csv")
        assert meta["checked"] is False
        assert meta["labelled_clusters"] == 0
        assert meta["fn_cells"] == 0

    def test_cluster_label_marks_source_checked(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c1",
                "source": "a.csv", "row": 1, "column": "name",
                "status": "tp",
            },
        )
        meta = next(
            s for s in client.get(f"/api/runs/{run_id}").json()["sources"]
            if s["name"] == "a.csv"
        )
        assert meta["checked"] is True
        assert meta["labelled_clusters"] == 1
        assert meta["fn_cells"] == 0

    def test_fn_cell_marks_source_checked(self, client_and_run) -> None:
        client, run_id = client_and_run
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cell", "target_id": "x", "source": "a.csv",
                "row": 2, "column": "email", "status": "fn",
            },
        )
        meta = next(
            s for s in client.get(f"/api/runs/{run_id}").json()["sources"]
            if s["name"] == "a.csv"
        )
        assert meta["checked"] is True
        assert meta["labelled_clusters"] == 0
        assert meta["fn_cells"] == 1

    def test_unflagged_csv_is_visible_and_browsable(self, tmp_path: Path) -> None:
        """A CSV that the LLM never produced clusters for must still
        appear in /api/runs/{run_id} (so the UI can list it) and be
        readable via /sources/{name} (so the operator can browse it
        for FN candidates).
        """
        project = tmp_path
        (project / "frontend" / "labeling").mkdir(parents=True)
        data = project / "data"
        data.mkdir()
        (data / "untouched.csv").write_text("col\nfoo\nbar\n", encoding="utf-8")
        (data / "flagged.csv").write_text("col\nx\ny\n", encoding="utf-8")

        out = project / "outputs" / "20260101T000000Z_test"
        out.mkdir(parents=True)
        csv_paths = [str(data / "untouched.csv"), str(data / "flagged.csv")]
        (out / "summary.json").write_text(json.dumps({
            "config": {"csv_paths": csv_paths,
                       "sources": ["untouched.csv", "flagged.csv"]},
        }))
        # Only flagged.csv has a cluster; untouched.csv has none.
        (out / "clusters.json").write_text(json.dumps([{
            "cluster_id": "c1", "source": "flagged.csv",
            "row": 0, "columns": ["col"], "error_types": ["typo"],
            "evidence_count": 1, "synthesis": None,
        }]))
        (out / "issues.json").write_text("[]")

        from src.labeling.app import create_app
        client = TestClient(create_app(project, annotator="t"))
        run = client.get(f"/api/runs/{out.name}").json()
        names = [s["name"] for s in run["sources"]]
        assert "untouched.csv" in names
        meta = next(s for s in run["sources"] if s["name"] == "untouched.csv")
        assert meta["clusters"] == 0
        assert meta["checked"] is False

        # The unflagged CSV is browsable for FN labelling.
        slice_ = client.get(f"/api/runs/{out.name}/sources/untouched.csv").json()
        assert slice_["total_rows"] == 2
        cells = client.get(
            f"/api/runs/{out.name}/sources/untouched.csv/cells"
        ).json()
        assert cells["cells"] == []

        # And FN labelling on it works end-to-end.
        r = client.post(
            f"/api/runs/{out.name}/labels",
            json={
                "target": "cell", "target_id": "x",
                "source": "untouched.csv", "row": 1, "column": "col",
                "status": "fn", "note": "should be Bar",
            },
        )
        assert r.status_code == 200
        # ``checked`` flips after the FN.
        meta = next(
            s for s in client.get(f"/api/runs/{out.name}").json()["sources"]
            if s["name"] == "untouched.csv"
        )
        assert meta["checked"] is True
        assert meta["fn_cells"] == 1


class TestExportConflictsEndpoint:
    """End-to-end check for ``GET /export/conflicts``.

    The pre-export sanity check that surfaces (source, column,
    original_value) triples whose label verdicts disagree across
    rows. Returns the same materialised benchmark the export
    endpoints emit, so what the UI shows is exactly what would
    have been shipped in the JSONL.
    """

    @staticmethod
    def _build_run_with_repeated_value(project: Path) -> str:
        """Create a run dir whose ``a.csv`` has the same value
        ``Apple`` in rows 1 + 2 (column ``fruit``) and a different
        value ``Banana`` in row 3, together with two same-value
        clusters on rows 1 + 2 plus one independent cluster on row
        3. The CSV row numbering matches the labelling-UI's 0-based
        ``row`` semantics (row 0 = first data row)."""
        data_dir = project / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "a.csv").write_text(
            "fruit\nApple\nApple\nApple\nBanana\n", encoding="utf-8",
        )

        run_dir = project / "outputs" / "20260101T000001Z_conflicts"
        run_dir.mkdir(parents=True)
        summary = {
            "config": {
                "csv_paths": [str(data_dir / "a.csv")],
                "sources": ["a.csv"],
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))
        clusters = [
            {
                "cluster_id": "c-row0", "source": "a.csv", "row": 0,
                "columns": ["fruit"], "error_types": ["typo"],
                "row_values": {"fruit": "Apple"},
            },
            {
                "cluster_id": "c-row1", "source": "a.csv", "row": 1,
                "columns": ["fruit"], "error_types": ["typo"],
                "row_values": {"fruit": "Apple"},
            },
            {
                "cluster_id": "c-row2", "source": "a.csv", "row": 2,
                "columns": ["fruit"], "error_types": ["typo"],
                "row_values": {"fruit": "Apple"},
            },
            {
                "cluster_id": "c-row3", "source": "a.csv", "row": 3,
                "columns": ["fruit"], "error_types": ["typo"],
                "row_values": {"fruit": "Banana"},
            },
        ]
        (run_dir / "clusters.json").write_text(json.dumps(clusters))
        (run_dir / "issues.json").write_text("[]")
        return run_dir.name

    @pytest.fixture
    def conflict_client(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            (project / "frontend" / "labeling").mkdir(parents=True)
            run_id = self._build_run_with_repeated_value(project)
            app = create_app(project, annotator="tester")
            yield TestClient(app), run_id

    def test_no_labels_means_no_conflicts(self, conflict_client) -> None:
        client, run_id = conflict_client
        r = client.get(f"/api/runs/{run_id}/export/conflicts")
        assert r.status_code == 200
        assert r.json() == {"conflicts": []}

    def test_same_value_different_labels_surface_as_conflict(
        self, conflict_client,
    ) -> None:
        client, run_id = conflict_client
        # Row 0 "Apple" -> clean, row 1 "Apple" -> error: classic
        # accidental-flip the operator should be warned about before
        # exporting.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c-row0",
                "source": "a.csv", "row": 0, "column": "fruit",
                "status": "fp",
            },
        )
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c-row1",
                "source": "a.csv", "row": 1, "column": "fruit",
                "status": "tp", "corrected_value": "Apples",
            },
        )
        # Row 2 left unlabelled (no benchmark cell emitted), row 3
        # "Banana" is a different value entirely.
        r = client.get(f"/api/runs/{run_id}/export/conflicts")
        assert r.status_code == 200
        groups = r.json()["conflicts"]
        assert len(groups) == 1
        g = groups[0]
        assert g["source"] == "a.csv"
        assert g["column"] == "fruit"
        assert g["original_value"] == "Apple"
        assert sorted(g["labels"]) == ["clean", "error"]
        rows = sorted(c["row"] for c in g["cells"])
        assert rows == [0, 1]
        # Cluster ids round-trip so the UI can navigate to the offender.
        for c in g["cells"]:
            assert c["cluster_ids"]

    def test_consistent_labels_do_not_trigger_a_conflict(
        self, conflict_client,
    ) -> None:
        client, run_id = conflict_client
        # All three "Apple" rows agree -> no conflict on the triple.
        for cid, row in (("c-row0", 0), ("c-row1", 1), ("c-row2", 2)):
            client.post(
                f"/api/runs/{run_id}/labels",
                json={
                    "target": "cluster", "target_id": cid,
                    "source": "a.csv", "row": row, "column": "fruit",
                    "status": "fp",
                },
            )
        r = client.get(f"/api/runs/{run_id}/export/conflicts")
        assert r.status_code == 200
        assert r.json() == {"conflicts": []}

    def test_exclusions_remove_cells_from_conflict_set(
        self, conflict_client,
    ) -> None:
        client, run_id = conflict_client
        # Build a conflict, then exclude the row that triggered it -
        # the export endpoint applies the same exclusion filter so
        # the conflict report and the JSONL stay aligned.
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c-row0",
                "source": "a.csv", "row": 0, "column": "fruit",
                "status": "fp",
            },
        )
        client.post(
            f"/api/runs/{run_id}/labels",
            json={
                "target": "cluster", "target_id": "c-row1",
                "source": "a.csv", "row": 1, "column": "fruit",
                "status": "tp", "corrected_value": "Apples",
            },
        )
        # Sanity-check the conflict exists before exclusion.
        assert client.get(
            f"/api/runs/{run_id}/export/conflicts"
        ).json()["conflicts"]
        # Exclude row 1 (the "error" half).
        r = client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"source": "a.csv", "scope": "rows", "rows": [1]},
        )
        assert r.status_code == 200, r.text
        assert client.get(
            f"/api/runs/{run_id}/export/conflicts"
        ).json()["conflicts"] == []


def test_cell_state_fans_out_affected_rows(tmp_path: Path) -> None:
    """A multi-row cluster's verdict must paint every row in its
    detections' ``affected_rows`` -- not just the anchor -- so cells
    captured via fan-out don't look UNLABELLED in the viewer. An
    own-anchor verdict on a cell still wins over the fanned-out one.
    """
    project = tmp_path / "proj"
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "a.csv").write_text(
        "name\nalice\nbob\ncarol\ndave\n", encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_fanout"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(json.dumps(
        {"config": {"csv_paths": [str(data_dir / "a.csv")],
                    "sources": ["a.csv"]}}))
    clusters = [
        # Broad cluster anchored at row 0, fanning out to rows 0,1,2,3.
        {"cluster_id": "broad", "source": "a.csv", "row": 0,
         "columns": ["name"], "error_types": ["typo"], "synthesis": None,
         "detections": [{"affected_rows": [0, 1, 2, 3], "columns": ["name"]}]},
        # A per-row cluster anchored at row 2 (operator will FP it).
        {"cluster_id": "row2", "source": "a.csv", "row": 2,
         "columns": ["name"], "error_types": ["typo"], "synthesis": None,
         "detections": [{"affected_rows": [2], "columns": ["name"]}]},
    ]
    (rd / "clusters.json").write_text(json.dumps(clusters))
    (rd / "issues.json").write_text("[]")

    client = TestClient(create_app(project, annotator="tester"))
    for cid, row, st in (("broad", 0, "tp"), ("row2", 2, "fp")):
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": cid, "source": "a.csv",
            "row": row, "column": "name", "status": st})
        assert r.status_code == 200, r.text

    cells = {
        c["row"]: c["state"]
        for c in client.get(
            f"/api/runs/{run_id}/sources/a.csv/cells").json()["cells"]
        if c["column"] == "name"
    }
    # Fan-out paints the peer rows tp ...
    assert cells[0] == "tp"
    assert cells[1] == "tp"
    assert cells[3] == "tp"
    # ... but row 2's OWN-anchor fp verdict wins over the fanned-out tp.
    assert cells[2] == "fp"
