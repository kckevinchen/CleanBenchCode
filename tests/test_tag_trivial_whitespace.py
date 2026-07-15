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

from src.labeling.app import create_app
from src.labeling.export import materialize_label_benchmark
from src.labeling.routers.propagation import _has_whitespace_anomaly
from src.labeling.store import LabelStore


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


# ---------------------------------------------------------------------
# scan_dataset mode: walk every cell (not just existing labels) and
# auto-create TP (cluster-covered) / FN (uncovered) labels, all TRIVIAL.
# ---------------------------------------------------------------------


def test_scan_dataset_creates_tp_and_fn(tmp_path: Path) -> None:
    """With no pre-labelling, a dataset scan should:
      * row 0 (tab)  -> covered by cluster r00000, unlabelled -> create_tp
      * row 2 (2sp)  -> uncovered by any cluster            -> create_fn
      * row 4 (NBSP) -> covered by cluster r00004, unlabelled -> create_tp
      * row 3 (AcmeCO, no ws) -> nothing.
    """
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Dry-run preview first.
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "dry_run": True},
        ).json()
        assert r["scan_dataset"] is True
        assert r["dry_run"] is True
        assert r["to_create_tp"] == 2
        assert r["to_create_fn"] == 1
        assert r["to_tag"] == 0
        assert r["to_write"] == 3
        # Nothing written on dry-run.
        before = client.get(f"/api/runs/{run_id}/labels").json()
        assert before["history_count"] == 0

        # Apply.
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        ).json()
        assert r["created_tp"] == 2
        assert r["created_fn"] == 1
        assert r["tagged"] == 0
        # The full (uncapped) list of freshly-created target_ids -- consumed
        # by the frontend to scope the follow-up convert in unlabeled-only
        # mode. Here: 2 cluster TPs + 1 FN cell = 3.
        assert len(r["created_target_ids"]) == 3
        assert "vendors.csv__r00000-c000" in r["created_target_ids"]
        assert "vendors.csv__r00004-c000" in r["created_target_ids"]

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        # Two cluster TP labels with TRIVIAL + normalised fix.
        tp0 = latest["vendors.csv__r00000-c000"]
        assert tp0["status"] == "tp"
        assert "TRIVIAL" in tp0["note"]
        assert tp0["corrected_value"] == "Bata Co"   # tab -> space
        tp4 = latest["vendors.csv__r00004-c000"]
        assert tp4["status"] == "tp"
        assert tp4["corrected_value"] == "Delta Inc"  # NBSP -> space
        # One FN cell label for the uncovered double-space cell (row 2).
        fn_cells = [
            lab for lab in latest.values()
            if lab["target"] == "cell" and lab["status"] == "fn"
        ]
        assert len(fn_cells) == 1
        assert fn_cells[0]["row"] == 2
        assert "TRIVIAL" in fn_cells[0]["note"]
        assert fn_cells[0]["corrected_value"] == "Acme Co"  # 2sp -> 1sp


def test_scan_dataset_is_idempotent(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        first = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        ).json()
        assert first["created_tp"] + first["created_fn"] == 3
        # Re-running writes nothing: every anomaly already carries TRIVIAL.
        second = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        ).json()
        assert second["created_tp"] == 0
        assert second["created_fn"] == 0
        assert second["tagged"] == 0
        assert second["skipped_already_tagged"] == 3
        assert second["to_write"] == 0


def test_scan_dataset_respects_fp_verdict(tmp_path: Path) -> None:
    """A cluster the operator ruled FP must NOT be flipped to TP, and the
    covered cell must NOT spawn an FN either -- the human said it is not
    an error."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "contractor",
            "status": "fp",
        })
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "dry_run": True},
        ).json()
        # row 0 cluster (fp) is skipped; row 4 still create_tp; row 2 still
        # create_fn. No TP/FN gets created for the fp-covered row 0.
        assert r["skipped_negative_verdict"] == 1
        assert r["to_create_tp"] == 1
        assert r["to_create_fn"] == 1
        # The fp cluster is not in the actionable set.
        actionable = [
            c for c in r["candidates"] if c["action"] != "already_tagged"
        ]
        assert all(
            c["target_id"] != "vendors.csv__r00000-c000" for c in actionable
        )

        # Apply preserves the FP verdict.
        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        )
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["vendors.csv__r00000-c000"]["status"] == "fp"


def test_whitespace_endpoint_ignores_include_fp(tmp_path: Path) -> None:
    """The ``include_fp`` fp-rescue is exclusive to the punctuation route:
    the whitespace endpoint must ignore the flag and keep preserving fp."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "contractor",
            "status": "fp",
        })
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "include_fp": True},
        ).json()
        # Flag is not honoured here: fp stays skipped, nothing rescued.
        assert r.get("include_fp") is False
        assert r.get("rescued_fp", 0) == 0
        assert r["skipped_negative_verdict"] == 1
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["vendors.csv__r00000-c000"]["status"] == "fp"



def test_scan_dataset_tags_existing_positive_label(tmp_path: Path) -> None:
    """A cluster already ruled TP gets TRIVIAL appended (not recreated),
    preserving its existing corrected_value."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "contractor",
            "status": "tp", "corrected_value": "BataCo",
            "note": "operator note",
        })
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        ).json()
        assert r["tagged"] == 1            # row-0 cluster tagged, not created
        assert r["created_tp"] == 1        # row-4 cluster still created
        assert r["created_fn"] == 1        # row-2 cell still created
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        rec = latest["vendors.csv__r00000-c000"]
        assert rec["status"] == "tp"
        assert rec["corrected_value"] == "BataCo"   # operator fix preserved
        assert "TRIVIAL" in rec["note"]
        assert "operator note" in rec["note"]


def test_scan_dataset_unlabeled_only_skips_positive_cluster(
    tmp_path: Path,
) -> None:
    """``unlabeled_only`` must not re-tag a cluster the operator already
    ruled TP, but still creates labels on the untouched anomaly cells."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "contractor",
            "status": "tp", "corrected_value": "BataCo",
            "note": "operator note",
        })
        # Dry-run: the labelled cluster is reported as skipped, not tagged.
        dry = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "unlabeled_only": True,
                  "dry_run": True},
        ).json()
        assert dry["unlabeled_only"] is True
        assert dry["to_tag"] == 0
        assert dry["skipped_labeled"] == 1
        assert dry["to_create_tp"] == 1   # row-4 cluster still unlabelled
        assert dry["to_create_fn"] == 1   # row-2 cell still unlabelled

        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "unlabeled_only": True},
        ).json()
        assert r["tagged"] == 0
        assert r["skipped_labeled"] == 1
        assert r["created_tp"] == 1
        assert r["created_fn"] == 1
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        rec = latest["vendors.csv__r00000-c000"]
        # Existing verdict untouched -- no TRIVIAL appended.
        assert rec["status"] == "tp"
        assert rec["corrected_value"] == "BataCo"
        assert "TRIVIAL" not in rec["note"]
        assert rec["note"] == "operator note"


def test_scan_dataset_unlabeled_only_skips_existing_cell_label(
    tmp_path: Path,
) -> None:
    """``unlabeled_only`` leaves an existing positive CELL label alone
    while still creating TP on the unlabelled clusters."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Pre-label the uncovered double-space cell (row 2) as FN.
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "ignored-rebuilt",
            "source": "vendors.csv", "row": 2, "column": "contractor",
            "status": "fn", "corrected_value": "Acme Co",
        })
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "unlabeled_only": True},
        ).json()
        assert r["tagged"] == 0
        assert r["skipped_labeled"] == 1     # row-2 cell left untouched
        assert r["created_fn"] == 0          # not re-created
        assert r["created_tp"] == 2          # rows 0 + 4 still created
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        cell = [
            lab for lab in latest.values()
            if lab["target"] == "cell" and lab["row"] == 2
        ][0]
        assert "TRIVIAL" not in cell["note"]


def test_scan_dataset_unlabeled_only_still_creates_when_no_labels(
    tmp_path: Path,
) -> None:
    """With nothing pre-labelled, ``unlabeled_only`` behaves exactly like
    a normal scan -- every anomaly is fresh."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "unlabeled_only": True},
        ).json()
        assert r["created_tp"] == 2
        assert r["created_fn"] == 1
        assert r["tagged"] == 0
        assert r["skipped_labeled"] == 0


def test_scan_dataset_source_filter(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Real source filter still scans the whole dataset (one CSV here).
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "source": "vendors.csv",
                  "dry_run": True},
        ).json()
        assert r["to_write"] == 3
        # Bogus source -> 404 (validated before the branch).
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "source": "nope.csv"},
        )
        assert r.status_code == 404


def _build_multicol_run(tmp_path: Path) -> tuple[Path, str]:
    """One CSV whose row 0 has whitespace anomalies in TWO columns, both
    covered by a single multi-column cluster. Used to prove the created
    TP carries a per-column fix instead of stamping column ``a``'s fix
    onto column ``b``.

      * row 0: a="Foo\\tBar" (tab), b="Baz  Qux" (double space)
    """
    project = tmp_path / "mc"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "m.csv"
    csv.write_text(
        "id,a,b\n"
        "1,Foo\tBar,Baz  Qux\n"
        "2,Clean,Clean\n",
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260514T010000Z_multicol"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(json.dumps({"config": {"csv_paths": [str(csv)]}}))
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "m.csv__r00000-c000", "source": "m.csv",
         "row": 0, "columns": ["a", "b"],
         "row_values": {"id": "1", "a": "Foo\tBar", "b": "Baz  Qux"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_scan_dataset_create_tp_multicolumn_per_column_fix(tmp_path: Path) -> None:
    """A multi-column cluster's auto-created TP must give EACH anomaly
    column its own normalised fix, not the first column's, all the way
    through to the benchmark export."""
    project, run_id = _build_multicol_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        ).json()
        # ONE cluster covering both anomaly columns -> one created TP.
        assert r["created_tp"] == 1
        assert r["created_fn"] == 0

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        tp = latest["m.csv__r00000-c000"]
        assert tp["status"] == "tp"
        # Per-column fixes, each computed from its OWN cell value.
        assert tp["corrected_values"]["a"] == "Foo Bar"
        assert tp["corrected_values"]["b"] == "Baz Qux"

    # End-to-end: the benchmark export gives column b ITS fix, not a's.
    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    store = LabelStore(rd / "labels.jsonl")
    by_key = {
        (c.source, c.row, c.column): c
        for c in materialize_label_benchmark(clusters, store)
    }
    assert by_key[("m.csv", 0, "a")].corrected_value == "Foo Bar"
    assert by_key[("m.csv", 0, "b")].corrected_value == "Baz Qux"


def _build_spanning_affected_rows_run(tmp_path: Path) -> tuple[Path, str]:
    """One CSV where THREE rows of the same column each have a double-
    space anomaly, and each row has its OWN cluster anchored on it - but
    the FIRST cluster's detection lists every sibling row in
    ``affected_rows`` (a multi-row typo whose evidence spans the group).

      * row 0: "Aaa  bbb"  -> cluster r00000 (anchor) + det.affected_rows
                              = [0, 1, 2]
      * row 1: "Ccc  ddd"  -> cluster r00001 (anchor)
      * row 2: "Eee  fff"  -> cluster r00002 (anchor)

    Reproduces the bug where ``cids[0]`` for rows 1 & 2 resolved to the
    spanning cluster r00000, and the ``seen_cluster_ids`` dedup then
    silently skipped them - so only row 0 was tagged.
    """
    project = tmp_path / "span"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "g.csv"
    csv.write_text(
        "id,name\n"
        "1,\"Aaa  bbb\"\n"   # row 0 - double space
        "2,\"Ccc  ddd\"\n"   # row 1 - double space
        "3,\"Eee  fff\"\n",  # row 2 - double space
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260608T020000Z_span"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text(json.dumps([
        # r00000's detection lists ALL three rows as affected (the
        # spanning evidence that triggered the bug).
        {"cluster_id": "g.csv__r00000-c000", "source": "g.csv",
         "row": 0, "columns": ["name"],
         "row_values": {"id": "1", "name": "Aaa  bbb"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1,
         "detections": [{"issue_id": "i0", "error_type": "typo",
                         "columns": ["name"], "affected_rows": [0, 1, 2]}]},
        {"cluster_id": "g.csv__r00001-c000", "source": "g.csv",
         "row": 1, "columns": ["name"],
         "row_values": {"id": "2", "name": "Ccc  ddd"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
        {"cluster_id": "g.csv__r00002-c000", "source": "g.csv",
         "row": 2, "columns": ["name"],
         "row_values": {"id": "3", "name": "Eee  fff"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_scan_attributes_each_sibling_to_its_own_anchor_cluster(
    tmp_path: Path,
) -> None:
    """Regression: a multi-row detection whose ``affected_rows`` span the
    sibling rows must NOT collapse those siblings onto one cluster. Each
    double-space row has its own anchored cluster, so all THREE must be
    created as TP - not just the spanning cluster's anchor row.
    """
    project, run_id = _build_spanning_affected_rows_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "dry_run": True},
        ).json()
        # All three sibling rows are attributed to their OWN cluster.
        assert r["to_create_tp"] == 3, r
        assert r["to_create_fn"] == 0
        assert r["to_write"] == 3

        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        ).json()
        assert r["created_tp"] == 3

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        # Each row's OWN cluster carries a TP + TRIVIAL + its own fix.
        for cid, fix in (
            ("g.csv__r00000-c000", "Aaa bbb"),
            ("g.csv__r00001-c000", "Ccc ddd"),
            ("g.csv__r00002-c000", "Eee fff"),
        ):
            rec = latest[cid]
            assert rec["status"] == "tp", cid
            assert "TRIVIAL" in rec["note"], cid
            assert rec["corrected_value"] == fix, cid


def test_spanning_negative_verdict_does_not_suppress_siblings(
    tmp_path: Path,
) -> None:
    """If the spanning cluster (r00000) is ruled fp/skip, its sibling
    rows must STILL be tagged via their own clusters - the negative
    verdict only suppresses r00000's OWN anchor cell, not the group.
    """
    project, run_id = _build_spanning_affected_rows_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Operator rules the spanning cluster FP.
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "g.csv__r00000-c000",
            "source": "g.csv", "row": 0, "column": "name", "status": "fp",
        })
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "dry_run": True},
        ).json()
        # r00000's own row is suppressed (negative verdict); rows 1 & 2
        # are still created on their own clusters.
        assert r["to_create_tp"] == 2, r
        assert r["skipped_negative_verdict"] == 1
        rows = sorted(
            c["row"] for c in r["candidates"] if c["action"] == "create_tp"
        )
        assert rows == [1, 2]

