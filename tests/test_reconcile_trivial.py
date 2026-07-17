"""POST /api/runs/{run_id}/labels/reconcile-trivial retracts auto-created
trivial labels whose justification is gone -- e.g. the dataset was normalized
on disk after the label run (collapsing the whitespace that an auto whitespace
label flagged), or the punctuation rule now exempts e-mail / URL cells.

Retraction is a tombstone (``status="unlabeled"``) so the target returns to
"no label" while the append-only history keeps the audit trail. Human / LLM
verdicts (notes not starting with ``(auto:``) are never touched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.export import materialize_label_benchmark
from src.labeling.schema import cell_target_id
from src.labeling.store import LabelStore


def _build_run(tmp_path: Path) -> tuple[Path, str, Path]:
    project = tmp_path / "rec"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "vendors.csv"
    # row 0: double space (whitespace anomaly). row 1: clean. row 2: comma.
    csv.write_text(
        "id,name\n"
        '1,"Fallen Below 18  Days"\n'
        "2,Clean Co\n"
        '3,"Doe,Jane"\n',
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_rec"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")
    return project, run_id, csv


def _normalize_csv(csv: Path, text: str) -> None:
    """Rewrite the CSV on disk and bump mtime so load_frame reloads."""
    csv.write_text(text, encoding="utf-8")
    st = csv.stat()
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def test_stale_whitespace_label_is_retracted(tmp_path: Path) -> None:
    project, run_id, csv = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Tag the whitespace anomaly (row 0) -> creates an FN cell label.
        tag = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        ).json()
        assert tag["created_fn"] == 1
        tid = cell_target_id("vendors.csv", 0, "name")
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert tid in latest

        # Normalize the dataset on disk (collapse the double space).
        _normalize_csv(
            csv,
            'id,name\n1,"Fallen Below 18 Days"\n2,Clean Co\n3,"Doe,Jane"\n',
        )

        # Dry-run preview lists the now-stale label.
        dry = client.post(
            f"/api/runs/{run_id}/labels/reconcile-trivial",
            json={"dry_run": True},
        ).json()
        assert dry["to_retract"] == 1
        assert dry["retracted"] == 0
        assert dry["by_type"] == {"spacing": 1}
        assert dry["candidates"][0]["row"] == 0

        # Apply -> the label is retracted (gone from latest).
        appl = client.post(
            f"/api/runs/{run_id}/labels/reconcile-trivial", json={},
        ).json()
        assert appl["retracted"] == 1
        latest2 = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert tid not in latest2

        # Idempotent: a second run finds nothing.
        again = client.post(
            f"/api/runs/{run_id}/labels/reconcile-trivial", json={},
        ).json()
        assert again["scanned"] == 0
        assert again["retracted"] == 0


def test_persisting_anomaly_is_kept(tmp_path: Path) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        )
        # CSV is NOT changed -> the anomaly persists -> nothing retracted.
        r = client.post(
            f"/api/runs/{run_id}/labels/reconcile-trivial", json={"dry_run": True},
        ).json()
        assert r["to_retract"] == 0
        assert r["scanned"] == 1  # the one auto label was considered


def test_punctuation_label_on_email_is_retracted(tmp_path: Path) -> None:
    """A punctuation auto-label written before the e-mail/URL guard existed is
    retracted, because the live value is no longer a punctuation anomaly."""
    project = tmp_path / "rec2"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    (project / "data").mkdir()
    csv = project / "data" / "c.csv"
    csv.write_text('id,contact\n1,"a@x.com,b@y.com"\n', encoding="utf-8")
    rd = project / "outputs" / "20260101T000000Z_rec2"
    rd.mkdir(parents=True)
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")
    run_id = "20260101T000000Z_rec2"

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Hand-write an auto punctuation FN label on the e-mail cell (as a
        # pre-guard tagger would have).
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "rebuilt",
            "source": "c.csv", "row": 0, "column": "contact",
            "status": "fn", "corrected_value": "a@x.com, b@y.com",
            "note": "(auto: punctuation anomaly) TRIVIAL",
        })
        r = client.post(
            f"/api/runs/{run_id}/labels/reconcile-trivial", json={},
        ).json()
        assert r["retracted"] == 1
        assert r["by_type"] == {"punctuation": 1}
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert cell_target_id("c.csv", 0, "contact") not in latest


def test_human_verdict_is_never_retracted(tmp_path: Path) -> None:
    project, run_id, csv = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # A human FP verdict on the (now-clean) double-space cell, with a
        # note that does NOT start with "(auto:".
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "rebuilt",
            "source": "vendors.csv", "row": 0, "column": "name",
            "status": "fp", "note": "checked by hand",
        })
        _normalize_csv(
            csv,
            'id,name\n1,"Fallen Below 18 Days"\n2,Clean Co\n3,"Doe,Jane"\n',
        )
        r = client.post(
            f"/api/runs/{run_id}/labels/reconcile-trivial", json={},
        ).json()
        assert r["scanned"] == 0
        assert r["retracted"] == 0
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest[cell_target_id("vendors.csv", 0, "name")]["status"] == "fp"


def test_source_filter_scopes_reconcile(tmp_path: Path) -> None:
    project, run_id, csv = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        )
        _normalize_csv(
            csv,
            'id,name\n1,"Fallen Below 18 Days"\n2,Clean Co\n3,"Doe,Jane"\n',
        )
        # Unknown source -> 404.
        bad = client.post(
            f"/api/runs/{run_id}/labels/reconcile-trivial",
            json={"source": "nope.csv"},
        )
        assert bad.status_code == 404
        # Correct source scopes + retracts.
        ok = client.post(
            f"/api/runs/{run_id}/labels/reconcile-trivial",
            json={"source": "vendors.csv"},
        ).json()
        assert ok["retracted"] == 1


# ---------------------------------------------------------------------------
# reclassify-retracted-trivial: turn auto-retracted tombstones into a verdict
# ---------------------------------------------------------------------------


def _tag_normalize_reconcile(client: TestClient, run_id: str, csv: Path) -> str:
    """Tag the row-0 whitespace anomaly, normalize the CSV so the value now
    equals the fix, then reconcile to retract it. Returns the cell target_id."""
    client.post(f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
                json={"scan_dataset": True})
    _normalize_csv(
        csv, 'id,name\n1,"Fallen Below 18 Days"\n2,Clean Co\n3,"Doe,Jane"\n',
    )
    client.post(f"/api/runs/{run_id}/labels/reconcile-trivial", json={})
    return cell_target_id("vendors.csv", 0, "name")


def test_reclassify_retracted_to_fp(tmp_path: Path) -> None:
    project, run_id, csv = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        tid = _tag_normalize_reconcile(client, run_id, csv)
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert tid not in latest  # retracted

        dry = client.post(
            f"/api/runs/{run_id}/labels/reclassify-retracted-trivial",
            json={"dry_run": True},
        ).json()
        assert dry["status"] == "fp"
        assert dry["to_reclassify"] == 1
        assert dry["skipped_still_anomalous"] == 0
        assert dry["by_target"] == {"cell": 1}

        appl = client.post(
            f"/api/runs/{run_id}/labels/reclassify-retracted-trivial", json={},
        ).json()
        assert appl["reclassified"] == 1
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest[tid]["status"] == "fp"

        # Idempotent: latest is no longer unlabeled, so a re-run finds nothing.
        again = client.post(
            f"/api/runs/{run_id}/labels/reclassify-retracted-trivial",
            json={"dry_run": True},
        ).json()
        assert again["scanned"] == 0
        assert again["to_reclassify"] == 0


def test_reclassify_exports_as_clean(tmp_path: Path) -> None:
    project, run_id, csv = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tag_normalize_reconcile(client, run_id, csv)
        client.post(
            f"/api/runs/{run_id}/labels/reclassify-retracted-trivial", json={},
        )
    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    store = LabelStore(rd / "labels.jsonl")
    by_loc = {(c.source, c.row, c.column): c.label
              for c in materialize_label_benchmark(clusters, store)}
    assert by_loc[("vendors.csv", 0, "name")] == "clean"


def test_reclassify_skips_cell_that_is_anomalous_again(tmp_path: Path) -> None:
    project, run_id, csv = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tag_normalize_reconcile(client, run_id, csv)
        # The cell becomes anomalous AGAIN (re-introduce a double space).
        _normalize_csv(
            csv, 'id,name\n1,"Fallen Below  18 Days"\n2,Clean Co\n3,"Doe,Jane"\n',
        )
        r = client.post(
            f"/api/runs/{run_id}/labels/reclassify-retracted-trivial",
            json={"dry_run": True},
        ).json()
        # Safety: a now-anomalous cell is NOT marked clean/fp.
        assert r["to_reclassify"] == 0
        assert r["skipped_still_anomalous"] == 1


def test_reclassify_invalid_status_rejected(tmp_path: Path) -> None:
    project, run_id, csv = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tag_normalize_reconcile(client, run_id, csv)
        r = client.post(
            f"/api/runs/{run_id}/labels/reclassify-retracted-trivial",
            json={"status": "fn"},
        )
        assert r.status_code == 400
