"""POST /api/runs/{run_id}/labels/tag-trivial-whitespace appends the
literal token ``TRIVIAL`` to the note of every TP/PARTIAL/FN label
whose underlying cell value matches the spacing-rule (red-underline)
whitespace anomaly detector.

Mirrors the test style of ``tests/test_propagate_same_value.py``: a
small synthetic run with hand-crafted clusters, a TestClient flow that
walks the operator through pre-labelling, then asserts the dry-run
preview and the persisted append-only result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import _has_whitespace_anomaly, create_app


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    """Tiny run with one CSV containing a mix of cells. We deliberately
    use whitespace anomalies that SURVIVE :func:`src.core.data_loader.load_csv`
    (which strips edge whitespace from every cell): a tab in the
    middle, a double internal space, and a non-breaking space. Those
    are the anomaly classes the labelling-UI's red-underline rule
    actually surfaces in production runs.

      * row 0 contractor="Bata\\tCo"   - tab inside (spacing rule)
      * row 1 contractor="Acme Corp"  - clean
      * row 2 contractor="Acme  Co"   - double internal space (spacing rule)
      * row 3 contractor="AcmeCO"     - capitalisation typo, NO ws
      * row 4 contractor="Delta\\u00a0Inc" - NBSP (spacing rule)

    Three LLM-flagged clusters target rows 0, 3, and 4. The operator
    will TP-label them, then call the trivial-tagger: rows 0 and 4
    qualify, row 3 (capitalisation only) does not.
    """
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "vendors.csv"
    csv.write_text(
        "id,contractor,amount\n"
        "1,Bata\tCo,100\n"          # row 0 - tab in middle
        "2,Acme Corp,150\n"         # row 1 - clean
        "3,\"Acme  Co\",200\n"     # row 2 - double internal space
        "4,AcmeCO,175\n"            # row 3 - capitalisation typo, no ws
        "5,Delta\u00a0Inc,250\n",  # row 4 - NBSP
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260514T000000Z_trivial"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "vendors.csv__r00000-c000", "source": "vendors.csv",
         "row": 0, "columns": ["contractor"],
         "row_values": {"id": "1", "contractor": "Bata\tCo", "amount": "100"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
        {"cluster_id": "vendors.csv__r00003-c000", "source": "vendors.csv",
         "row": 3, "columns": ["contractor"],
         "row_values": {"id": "4", "contractor": "AcmeCO", "amount": "175"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
        {"cluster_id": "vendors.csv__r00004-c000", "source": "vendors.csv",
         "row": 4, "columns": ["contractor"],
         "row_values": {"id": "5", "contractor": "Delta\u00a0Inc", "amount": "250"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_helper_recognises_spacing_rule_anomalies() -> None:
    """The Python helper must agree with the JS ``hasWs`` regex on the
    canonical cases the auto-whitespace pass identifies."""
    assert _has_whitespace_anomaly(" leading")
    assert _has_whitespace_anomaly("trailing ")
    assert _has_whitespace_anomaly("two  spaces")
    assert _has_whitespace_anomaly("a\tb")          # tab
    assert _has_whitespace_anomaly("a\u00A0b")      # NBSP
    assert _has_whitespace_anomaly("a\u200Bb")      # zero-width space
    assert not _has_whitespace_anomaly("clean value")
    assert not _has_whitespace_anomaly("")
    assert not _has_whitespace_anomaly("AcmeCO")


def test_dry_run_returns_candidates_without_writing(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Operator labels all three clusters TP.
        for cid, row in (("vendors.csv__r00000-c000", 0),
                         ("vendors.csv__r00003-c000", 3),
                         ("vendors.csv__r00004-c000", 4)):
            r = client.post(f"/api/runs/{run_id}/labels", json={
                "target": "cluster", "target_id": cid,
                "source": "vendors.csv", "row": row, "column": "contractor",
                "status": "tp", "corrected_value": "Acme Corp",
            })
            assert r.status_code == 200, r.text

        before = client.get(f"/api/runs/{run_id}/labels").json()
        before_history = before["history_count"]

        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"dry_run": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # The tab-in-middle (row 0) and NBSP (row 4) clusters qualify.
        # Row 3 (capitalisation only) does NOT trigger the spacing rule.
        assert body["dry_run"] is True
        assert body["to_tag"] == 2
        assert body["tagged"] == 0
        assert body["skipped_already_tagged"] == 0
        rows = sorted(c["row"] for c in body["candidates"])
        assert rows == [0, 4]
        for cand in body["candidates"]:
            assert cand["status"] == "tp"
            assert cand["target"] == "cluster"
            assert cand["already_tagged"] is False
            assert cand["ws_hits"][0]["column"] == "contractor"

        # No new label records were appended.
        after = client.get(f"/api/runs/{run_id}/labels").json()
        assert after["history_count"] == before_history


def test_apply_appends_trivial_to_note(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Pre-label row 0 cluster TP with an existing note.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "contractor",
            "status": "tp", "corrected_value": "AcmeCo",
            "note": "(auto: whitespace anomaly)",
        })
        assert r.status_code == 200, r.text

        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is False
        assert body["tagged"] == 1
        assert body["skipped_already_tagged"] == 0

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        rec = latest["vendors.csv__r00000-c000"]
        # The append-only store's latest now has TRIVIAL in the note,
        # appended to the original auto-whitespace note.
        assert "TRIVIAL" in rec["note"]
        assert "(auto: whitespace anomaly)" in rec["note"]
        # Other invariants preserved.
        assert rec["status"] == "tp"
        assert rec["corrected_value"] == "AcmeCo"


def test_apply_is_idempotent(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "contractor",
            "status": "tp", "corrected_value": "AcmeCo",
        })
        # First call tags the row.
        first = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={},
        ).json()
        assert first["tagged"] == 1
        # Second call is a no-op: the note now contains TRIVIAL so
        # the candidate is reported under skipped_already_tagged.
        second = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={},
        ).json()
        assert second["tagged"] == 0
        assert second["skipped_already_tagged"] == 1
        assert second["to_tag"] == 0


def test_handles_cell_fn_target(tmp_path: Path) -> None:
    """The auto-whitespace pass writes FN cell labels for cells the
    LLM missed. Those are the bulk of real spacing-rule errors and
    must be picked up too."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # FN cell label on row 2 (double internal space) - LLM never
        # surfaced it as a cluster, operator added it as FN.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell",
            "target_id": "ignored-rebuilt",
            "source": "vendors.csv",
            "row": 2, "column": "contractor",
            "status": "fn",
            "corrected_value": "Acme Co",
        })
        assert r.status_code == 200, r.text

        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"dry_run": True},
        ).json()
        assert r["to_tag"] == 1
        assert r["candidates"][0]["target"] == "cell"
        assert r["candidates"][0]["status"] == "fn"
        assert r["candidates"][0]["row"] == 2


def test_skips_clean_value_label(tmp_path: Path) -> None:
    """A TP label on a cell with no whitespace anomaly must NOT be
    tagged - the spacing rule does not match, so it isn't trivial."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Label the capitalisation-typo cluster (row 3) TP. Value
        # ``AcmeCO`` is a real semantic typo with no whitespace.
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00003-c000",
            "source": "vendors.csv", "row": 3, "column": "contractor",
            "status": "tp", "corrected_value": "Acme Corp",
        })
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={},
        ).json()
        assert body["tagged"] == 0
        assert body["to_tag"] == 0
        # ``scanned`` counts every label considered (status filter pass)
        # even if it doesn't have a ws hit.
        assert body["scanned"] >= 1


def test_source_filter_scopes_to_one_csv(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "contractor",
            "status": "tp", "corrected_value": "AcmeCo",
        })
        # Bogus source filter -> 404 so the UI surfaces a clean error.
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"source": "does-not-exist.csv", "dry_run": True},
        )
        assert r.status_code == 404
        # Real source filter passes.
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"source": "vendors.csv", "dry_run": True},
        )
        assert r.status_code == 200, r.text
        assert r.json()["to_tag"] == 1


def test_invalid_status_rejected(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"statuses": ["bogus"]},
        )
        assert r.status_code == 400
