"""POST /api/runs/{run_id}/labels/convert-trivial-to-skip rewrites every
latest label whose ``note`` carries the ``TRIVIAL`` marker to
``status="skip"`` so the benchmark exporter drops those cells from the
eval set.

Mirrors the test style of ``tests/test_tag_trivial_whitespace.py``: a
small synthetic run with hand-crafted clusters, a TestClient flow that
walks the operator through pre-labelling + TRIVIAL-tagging, then asserts
the dry-run preview, the persisted append-only result, and the
end-to-end benchmark export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.export import materialize_label_benchmark
from src.labeling.schema import cell_target_id
from src.labeling.store import LabelStore


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    """Tiny run with one CSV and three LLM-flagged clusters on rows
    0, 2, 4 (all column ``contractor``). The operator TP-labels them;
    the test then marks a subset TRIVIAL and converts those to skip.
    """
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "vendors.csv"
    csv.write_text(
        "id,contractor,amount\n"
        "1,Bata Co,100\n"           # row 0
        "2,Acme Corp,150\n"         # row 1 - clean (no cluster)
        "3,Cee Inc,200\n"           # row 2
        "4,Dee LLC,175\n"           # row 3 - clean (no cluster)
        "5,Eee Ltd,250\n",          # row 4
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260608T000000Z_toskip"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "vendors.csv__r00000-c000", "source": "vendors.csv",
         "row": 0, "columns": ["contractor"],
         "row_values": {"id": "1", "contractor": "Bata Co", "amount": "100"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
        {"cluster_id": "vendors.csv__r00002-c000", "source": "vendors.csv",
         "row": 2, "columns": ["contractor"],
         "row_values": {"id": "3", "contractor": "Cee Inc", "amount": "200"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
        {"cluster_id": "vendors.csv__r00004-c000", "source": "vendors.csv",
         "row": 4, "columns": ["contractor"],
         "row_values": {"id": "5", "contractor": "Eee Ltd", "amount": "250"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def _tp_label(client: TestClient, run_id: str, cid: str, row: int,
              note: str = "") -> None:
    r = client.post(f"/api/runs/{run_id}/labels", json={
        "target": "cluster", "target_id": cid,
        "source": "vendors.csv", "row": row, "column": "contractor",
        "status": "tp", "corrected_value": "Fixed", "note": note,
    })
    assert r.status_code == 200, r.text


def test_dry_run_returns_candidates_without_writing(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Two clusters TRIVIAL, one plain TP.
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto: whitespace anomaly) TRIVIAL")
        _tp_label(client, run_id, "vendors.csv__r00002-c000", 2,
                  note="manual TRIVIAL")
        _tp_label(client, run_id, "vendors.csv__r00004-c000", 4,
                  note="real semantic typo")

        before = client.get(f"/api/runs/{run_id}/labels").json()
        before_history = before["history_count"]

        r = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"dry_run": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is True
        # Only the two TRIVIAL-noted labels are in scope.
        assert body["scanned"] == 2
        assert body["to_convert"] == 2
        assert body["converted"] == 0
        assert body["already_skip"] == 0
        assert body["by_status"] == {"tp": 2}
        rows = sorted(c["row"] for c in body["candidates"])
        assert rows == [0, 2]
        for cand in body["candidates"]:
            assert cand["status"] == "tp"
            assert cand["target"] == "cluster"
            assert cand["already_skip"] is False
            assert "TRIVIAL" in cand["note"].upper()

        # No new label records were appended.
        after = client.get(f"/api/runs/{run_id}/labels").json()
        assert after["history_count"] == before_history


def test_apply_converts_trivial_to_skip(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto: whitespace anomaly) TRIVIAL")
        _tp_label(client, run_id, "vendors.csv__r00004-c000", 4,
                  note="real semantic typo")

        r = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is False
        assert body["converted"] == 1
        assert body["already_skip"] == 0

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        # The TRIVIAL cluster is now skip; its note (incl. the marker)
        # is preserved for the audit trail.
        triv = latest["vendors.csv__r00000-c000"]
        assert triv["status"] == "skip"
        assert "TRIVIAL" in triv["note"].upper()
        # The plain semantic typo is untouched.
        real = latest["vendors.csv__r00004-c000"]
        assert real["status"] == "tp"


def test_apply_is_idempotent(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto: whitespace anomaly) TRIVIAL")
        first = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip", json={},
        ).json()
        assert first["converted"] == 1
        # Second call: the label is already skip (and still TRIVIAL-noted),
        # so it is reported under already_skip and not re-written.
        second = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip", json={},
        ).json()
        assert second["converted"] == 0
        assert second["already_skip"] == 1
        assert second["to_convert"] == 0
        assert second["by_status"] == {}


def test_non_trivial_labels_untouched(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # No TRIVIAL note anywhere.
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="a real typo")
        _tp_label(client, run_id, "vendors.csv__r00002-c000", 2)
        r = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip", json={},
        ).json()
        assert r["scanned"] == 0
        assert r["converted"] == 0
        assert r["candidates"] == []
        # Both verdicts remain TP.
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["vendors.csv__r00000-c000"]["status"] == "tp"
        assert latest["vendors.csv__r00002-c000"]["status"] == "tp"


def test_converts_fn_cell_target(tmp_path: Path) -> None:
    """A TRIVIAL-noted FN cell label (the bulk of whitespace typos) must
    also flip to skip, so it stops exporting as an ``error`` record.

    Uses an operator-tagged TRIVIAL note (not an ``(auto: ...)`` provenance
    string) so the conversion path is exercised directly; the stale-cell
    re-validation guard for auto-tagged labels has its own tests below.
    """
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "ignored-rebuilt",
            "source": "vendors.csv", "row": 1, "column": "contractor",
            "status": "fn", "corrected_value": "Acme Corp",
            "note": "operator: trivial whitespace TRIVIAL",
        })
        assert r.status_code == 200, r.text

        body = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip", json={},
        ).json()
        assert body["converted"] == 1
        assert body["stale_retracted"] == 0
        assert body["by_status"] == {"fn": 1}

    # End-to-end: the FN cell no longer exports (skip drops it).
    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    store = LabelStore(rd / "labels.jsonl")
    cells = materialize_label_benchmark(clusters, store)
    assert not any(
        c.source == "vendors.csv" and c.row == 1 for c in cells
    )


def test_revalidate_cleans_stale_auto_cell_to_fp(tmp_path: Path) -> None:
    """An ``(auto: spacing anomaly)`` cell label whose live CSV value is no
    longer anomalous (dataset normalized after the run) must NOT be converted
    to the chosen verdict. By default it becomes ``fp`` (a clean
    hard-negative -- the cell IS clean now), so a stale whitespace flag never
    becomes a partial / error ground-truth record and is not left unlabeled."""
    project, run_id = _build_run(tmp_path)  # vendors.csv is already clean
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Hand-place an auto-spacing FN on a CLEAN cell (simulates a label
        # created before the dataset was normalized).
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "ignored-rebuilt",
            "source": "vendors.csv", "row": 1, "column": "contractor",
            "status": "fn", "corrected_value": "Acme Corp",
            "note": "(auto: spacing anomaly) TRIVIAL",
        })
        assert r.status_code == 200, r.text

        body = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "partial"},
        ).json()
        # Stale -> cleaned to fp, NOT converted to partial, NOT unlabeled.
        assert body["converted"] == 0
        assert body["stale_cleaned"] == 1
        assert body["stale_retracted"] == 1  # legacy alias
        assert body["stale_status"] == "fp"
        assert body["by_status"] == {}

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        tid = cell_target_id("vendors.csv", 1, "contractor")
        # The cell is now a labelled fp (clean), not an unlabeled tombstone.
        assert latest[tid]["status"] == "fp"


def test_stale_status_unlabeled_restores_retract(tmp_path: Path) -> None:
    """``stale_status: unlabeled`` keeps the old behaviour: the stale auto
    label is retracted to an ``unlabeled`` tombstone with the reconcile
    marker so ``reclassify-retracted-trivial`` can pick it up."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "ignored-rebuilt",
            "source": "vendors.csv", "row": 1, "column": "contractor",
            "status": "fn", "corrected_value": "Acme Corp",
            "note": "(auto: spacing anomaly) TRIVIAL",
        })
        body = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "partial", "stale_status": "unlabeled"},
        ).json()
        assert body["stale_cleaned"] == 1
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        tid = cell_target_id("vendors.csv", 1, "contractor")
        assert tid not in latest  # unlabeled tombstone filtered out
        recl = client.post(
            f"/api/runs/{run_id}/labels/reclassify-retracted-trivial",
            json={"status": "fp", "dry_run": True},
        ).json()
        assert recl["to_reclassify"] == 1


def test_excluded_cell_forced_to_skip_not_chosen_verdict(tmp_path: Path) -> None:
    """A TRIVIAL label inside an active exclusion scope must be forced to
    ``skip`` -- the convert path must never re-paint the chosen verdict over
    an out-of-scope cell. Uses a genuinely anomalous cell so the stale guard
    doesn't fire (isolating the exclusion behaviour). The label is created
    AFTER the exclusion so the add-time cascade doesn't pre-skip it, leaving
    the convert endpoint to do the forcing."""
    project, run_id = _build_ws_run(tmp_path)  # row 2 contractor = "Cee  Inc"
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Exclude the whole column first (no labels yet -> nothing to cascade).
        ex = client.post(f"/api/runs/{run_id}/exclusions", json={
            "source": "vendors.csv", "scope": "columns",
            "columns": ["contractor"],
        })
        assert ex.status_code == 200, ex.text

        # Now hand-place a still-anomalous TRIVIAL fn on the excluded column.
        r = client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "ignored-rebuilt",
            "source": "vendors.csv", "row": 2, "column": "contractor",
            "status": "fn", "corrected_value": "Cee Inc",
            "note": "(auto: spacing anomaly) TRIVIAL",
        })
        assert r.status_code == 200, r.text

        # Convert TRIVIAL -> partial. The excluded cell must go to skip, not
        # partial, even though its value is still anomalous (not stale).
        body = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "partial"},
        ).json()
        assert body["converted"] == 0
        assert body["excluded_to_skip"] == 1
        assert body["stale_cleaned"] == 0

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        tid = cell_target_id("vendors.csv", 2, "contractor")
        assert latest[tid]["status"] == "skip"


def test_revalidate_false_forces_conversion_of_stale_auto_cell(
    tmp_path: Path,
) -> None:
    """``revalidate: false`` restores the unconditional move: even a stale
    auto-tagged label is converted (escape hatch for the rare case the
    operator wants the raw behaviour)."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "ignored-rebuilt",
            "source": "vendors.csv", "row": 1, "column": "contractor",
            "status": "fn", "corrected_value": "Acme Corp",
            "note": "(auto: spacing anomaly) TRIVIAL",
        })
        body = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "partial", "revalidate": False},
        ).json()
        assert body["converted"] == 1
        assert body["stale_cleaned"] == 0
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        tid = cell_target_id("vendors.csv", 1, "contractor")
        assert latest[tid]["status"] == "partial"


def test_source_filter_scopes_conversion(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto) TRIVIAL")
        # Unknown source -> 404.
        r = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"source": "nope.csv"},
        )
        assert r.status_code == 404
        # Correct source scopes the scan and converts.
        r = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"source": "vendors.csv"},
        ).json()
        assert r["converted"] == 1


def test_target_ids_filter_scopes_conversion(tmp_path: Path) -> None:
    """``target_ids`` restricts the convert to the given labels only -- the
    mechanism that keeps the trivial taggers' ``unlabeled_only`` apply from
    re-painting pre-existing reviewed labels. A TRIVIAL label NOT in the
    allow-list is left completely untouched."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Three TRIVIAL clusters; only one is in the allow-list. Use plain
        # human-appended TRIVIAL notes (not the ``(auto: ...)`` prefix) so the
        # convert's stale-revalidation stays out of the way -- this test is
        # about target_ids scoping only.
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="manual TRIVIAL")
        _tp_label(client, run_id, "vendors.csv__r00002-c000", 2,
                  note="manual TRIVIAL")
        _tp_label(client, run_id, "vendors.csv__r00004-c000", 4,
                  note="reviewed FN cell TRIVIAL")

        r = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"target_ids": ["vendors.csv__r00000-c000"]},
        ).json()
        # Only the one allow-listed label is scanned + converted.
        assert r["scanned"] == 1
        assert r["converted"] == 1

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["vendors.csv__r00000-c000"]["status"] == "skip"
        # The two NOT in the allow-list keep their reviewed TP verdict.
        assert latest["vendors.csv__r00002-c000"]["status"] == "tp"
        assert latest["vendors.csv__r00004-c000"]["status"] == "tp"


def test_empty_target_ids_converts_nothing(tmp_path: Path) -> None:
    """An empty allow-list converts nothing (the frontend short-circuits,
    but the backend must also be safe if called)."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto) TRIVIAL")
        r = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"target_ids": []},
        ).json()
        assert r["scanned"] == 0
        assert r["converted"] == 0
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["vendors.csv__r00000-c000"]["status"] == "tp"


def test_target_ids_must_be_a_list(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"target_ids": "vendors.csv__r00000-c000"},
        )
        assert r.status_code == 400


def test_end_to_end_skip_drops_cell_from_benchmark(tmp_path: Path) -> None:
    """The whole point: after conversion the TRIVIAL cluster's cell is
    gone from the exported benchmark, while the untouched TP cell stays.
    """
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto: whitespace anomaly) TRIVIAL")
        _tp_label(client, run_id, "vendors.csv__r00004-c000", 4,
                  note="real semantic typo")
        client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip", json={},
        )

    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    store = LabelStore(rd / "labels.jsonl")
    by_key = {
        (c.source, c.row, c.column): c
        for c in materialize_label_benchmark(clusters, store)
    }
    # Row 0 (now skip) is dropped; row 4 (TP) survives as an error.
    assert ("vendors.csv", 0, "contractor") not in by_key
    assert by_key[("vendors.csv", 4, "contractor")].label == "error"


def _build_ws_run(tmp_path: Path) -> tuple[Path, str]:
    """Run whose CSV carries real whitespace anomalies, with one cluster
    covering a double-space cell and one anomaly the LLM missed. Drives
    the actual whitespace scan tagger (not hand-placed labels)."""
    project = tmp_path / "ws2"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "vendors.csv"
    csv.write_text(
        "id,contractor,amount\n"
        "1,\"Acme  Co\",100\n"      # row 0 - double space, covered by cluster
        "2,Bee Corp,150\n"          # row 1 - clean
        "3,\"Cee  Inc\",200\n",    # row 2 - double space, NOT covered (FN)
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260608T010000Z_ws"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "vendors.csv__r00000-c000", "source": "vendors.csv",
         "row": 0, "columns": ["contractor"],
         "row_values": {"id": "1", "contractor": "Acme  Co", "amount": "100"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_default_workflow_tagger_then_convert_drops_from_export(
    tmp_path: Path,
) -> None:
    """The UI default: run the whitespace scan tagger, then immediately
    convert TRIVIAL -> skip (what the modal's "Mark as skip" toggle does).
    The covered cluster cell AND the missed FN cell must BOTH drop out of
    the exported benchmark.
    """
    project, run_id = _build_ws_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # 1. Tagger (scan mode) writes a TP cluster (row 0) + an FN cell
        #    (row 2), both TRIVIAL.
        tag = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        ).json()
        assert tag["created_tp"] == 1
        assert tag["created_fn"] == 1

        # 2. Convert chains right after (the default toggle).
        conv = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip", json={},
        ).json()
        # Both the TP cluster and the FN cell were TRIVIAL -> both move.
        assert conv["converted"] == 2
        assert conv["by_status"] == {"tp": 1, "fn": 1}

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["vendors.csv__r00000-c000"]["status"] == "skip"

    # End-to-end: neither the covered cell nor the missed FN cell exports.
    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    store = LabelStore(rd / "labels.jsonl")
    cells = materialize_label_benchmark(clusters, store)
    assert not any(c.source == "vendors.csv" for c in cells)


def test_unlabeled_only_apply_does_not_repaint_reviewed_label(
    tmp_path: Path,
) -> None:
    """REGRESSION: the unlabeled-only tagger apply (scan -> scoped convert)
    must NOT flip a pre-existing reviewed label to skip.

    Reproduces the production bug where running the trivial tagger with
    ``unlabeled_only`` still painted every TRIVIAL-noted label run-wide to
    skip, because the follow-up convert ignored ``unlabeled_only``. The fix:
    the scan returns ``created_target_ids`` and the convert is scoped to
    them, so only the freshly-created cell moves.
    """
    project, run_id = _build_ws_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # A pre-existing, operator-reviewed TP on the row-0 cluster that
        # also carries a TRIVIAL marker (e.g. an earlier auto-tag / LLM-fix).
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "contractor",
            "status": "tp", "corrected_value": "Acme Co",
            "note": "(auto: whitespace anomaly) TRIVIAL [LLM-fix]",
        })

        # 1. Tagger in scan + unlabeled_only mode: only the uncovered row-2
        #    anomaly is fresh, so exactly one FN is created. The reviewed
        #    row-0 cluster is skipped.
        tag = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True, "unlabeled_only": True},
        ).json()
        assert tag["created_fn"] == 1
        assert tag["created_tp"] == 0
        assert tag["tagged"] == 0
        assert tag["skipped_labeled"] == 1
        created = tag["created_target_ids"]
        assert len(created) == 1   # the row-2 FN cell only

        # 2. Scoped convert (what the frontend now sends in unlabeled-only
        #    mode): status=skip but restricted to the created ids.
        conv = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "skip", "target_ids": created},
        ).json()
        assert conv["scanned"] == 1
        assert conv["converted"] == 1

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        # The reviewed cluster KEEPS its TP verdict + corrected value.
        rec = latest["vendors.csv__r00000-c000"]
        assert rec["status"] == "tp", "reviewed label must not be repainted"
        assert rec["corrected_value"] == "Acme Co"
        # The freshly-created FN cell is the only thing that moved to skip.
        fresh = latest[created[0]]
        assert fresh["status"] == "skip"


@pytest.mark.parametrize("verdict, expected_label", [
    ("fp", "clean"),
    ("partial", "error"),
    ("unsure", "ambiguous"),
])
def test_uncovered_fn_cell_exports_by_verdict(
    tmp_path: Path, verdict: str, expected_label: str,
) -> None:
    """An UNCOVERED (FN) cell converted to fp / partial / unsure must export
    with the mapped verdict, exactly like the covered (cluster) cell -- not be
    silently dropped (regression: cell-level export was FN-only)."""
    project, run_id = _build_ws_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        tag = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-whitespace",
            json={"scan_dataset": True},
        ).json()
        assert tag["created_tp"] == 1 and tag["created_fn"] == 1
        conv = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": verdict},
        ).json()
        assert conv["converted"] == 2

    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    store = LabelStore(rd / "labels.jsonl")
    by_loc = {(c.source, c.row, c.column): c.label
              for c in materialize_label_benchmark(clusters, store)}
    # Covered cluster cell (row 0) AND uncovered FN cell (row 2) both export
    # with the same mapped verdict.
    assert by_loc[("vendors.csv", 0, "contractor")] == expected_label
    assert by_loc[("vendors.csv", 2, "contractor")] == expected_label


def test_convert_to_fp_marks_cluster_clean(tmp_path: Path) -> None:
    """``status=fp`` moves a TRIVIAL cluster verdict to ``fp`` so the
    exporter emits the cell as ``clean`` (a hard negative), not an error."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto: whitespace anomaly) TRIVIAL")
        body = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "fp"},
        ).json()
        assert body["status"] == "fp"
        assert body["converted"] == 1
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["vendors.csv__r00000-c000"]["status"] == "fp"

    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    store = LabelStore(rd / "labels.jsonl")
    by_key = {
        (c.source, c.row, c.column): c
        for c in materialize_label_benchmark(clusters, store)
    }
    assert by_key[("vendors.csv", 0, "contractor")].label == "clean"


def test_convert_to_partial_and_unsure(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto) TRIVIAL")
        _tp_label(client, run_id, "vendors.csv__r00002-c000", 2,
                  note="(auto) TRIVIAL")

        r1 = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "partial"},
        ).json()
        assert r1["status"] == "partial"
        assert r1["converted"] == 2
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["vendors.csv__r00000-c000"]["status"] == "partial"

        # Re-running with unsure moves them again (partial -> unsure).
        r2 = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "unsure"},
        ).json()
        assert r2["converted"] == 2
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest["vendors.csv__r00002-c000"]["status"] == "unsure"


def test_convert_idempotent_at_chosen_status(tmp_path: Path) -> None:
    """Idempotency is relative to the TARGET status: a second fp-convert is
    a no-op (reported under already_skip), not another write."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto) TRIVIAL")
        first = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "fp"},
        ).json()
        assert first["converted"] == 1
        second = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "fp"},
        ).json()
        assert second["converted"] == 0
        assert second["already_skip"] == 1
        assert second["to_convert"] == 0


def test_invalid_status_rejected(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        _tp_label(client, run_id, "vendors.csv__r00000-c000", 0,
                  note="(auto) TRIVIAL")
        # Not a LabelStatus value at all.
        r = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "bogus"},
        )
        assert r.status_code == 400
        # A real status that is not a valid convert target (fn / unlabeled).
        r2 = client.post(
            f"/api/runs/{run_id}/labels/convert-trivial-to-skip",
            json={"status": "fn"},
        )
        assert r2.status_code == 400

