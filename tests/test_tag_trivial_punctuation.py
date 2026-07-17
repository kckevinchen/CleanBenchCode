"""POST /api/runs/{run_id}/labels/tag-trivial-punctuation: the punctuation
analogue of the whitespace tagger. In ``scan_dataset`` mode it walks every
cell, auto-creates a TP (covered by a cluster) or FN (uncovered) label for each
Tier-1 punctuation anomaly, tags it ``TRIVIAL``, and writes the deterministic
fix as the correction. Guards (grouped numbers, function calls, numeric
parentheticals) must be left untouched.

Mirrors the style of ``tests/test_tag_trivial_whitespace.py``: a synthetic run,
a TestClient walk-through, then assertions on the persisted append-only result
and the materialised benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.export import materialize_label_benchmark
from src.labeling.routers.propagation import _has_punctuation_anomaly
from src.labeling.schema import Label, LabelStatus, LabelTarget, cell_target_id
from src.labeling.store import LabelStore


def _build_run(tmp_path: Path) -> tuple[Path, str]:
    """Tiny run whose ``vendor`` column mixes punctuation anomalies with
    cells every Tier-1 guard must leave alone.

      * row 0 vendor="Acme(Holdings)" - missing space before "(" (anomaly)
      * row 1 vendor="Acme Corp"      - clean
      * row 2 vendor="Doe,Jane"       - missing space after comma (anomaly)
      * row 3 vendor="varchar(255)"   - GUARD: numeric parenthetical
      * row 4 vendor="1,000"          - GUARD: grouped number
      * row 5 vendor="f(x)"           - GUARD: single-letter function call

    One LLM cluster covers row 0; row 2 is uncovered. So a scan_dataset pass
    creates a TP on row 0's cluster and an FN cell on row 2, and touches
    nothing else.
    """
    project = tmp_path / "punct"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "vendors.csv"
    csv.write_text(
        "id,vendor,note\n"
        "1,Acme(Holdings),a\n"      # row 0 - missing space before (
        "2,Acme Corp,b\n"           # row 1 - clean
        "3,\"Doe,Jane\",c\n"       # row 2 - missing space after comma
        "4,varchar(255),d\n"        # row 3 - guard: numeric paren
        "5,\"1,000\",e\n"          # row 4 - guard: grouped number
        "6,f(x),f\n",               # row 5 - guard: single-letter func
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260514T000000Z_punct"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "vendors.csv__r00000-c000", "source": "vendors.csv",
         "row": 0, "columns": ["vendor"],
         "row_values": {"id": "1", "vendor": "Acme(Holdings)", "note": "a"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_helper_recognises_tier1_punctuation_anomalies() -> None:
    assert _has_punctuation_anomaly("Acme(Holdings)")
    assert _has_punctuation_anomaly("Doe,Jane")
    assert not _has_punctuation_anomaly("varchar(255)")
    assert not _has_punctuation_anomaly("1,000")
    assert not _has_punctuation_anomaly("f(x)")
    assert not _has_punctuation_anomaly("Acme Corp")
    assert not _has_punctuation_anomaly("")


def test_dry_run_scans_dataset_without_writing(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        before = client.get(f"/api/runs/{run_id}/labels").json()
        before_history = before["history_count"]

        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"dry_run": True, "scan_dataset": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is True
        # Only the two anomaly cells qualify; the three guard cells and the
        # clean cell do not.
        assert body["anomaly_cells"] == 2
        assert body["to_create_tp"] == 1   # row 0, covered by the cluster
        assert body["to_create_fn"] == 1   # row 2, uncovered
        assert body["tagged"] == 0

        by_row = {c["row"]: c for c in body["candidates"]}
        assert set(by_row) == {0, 2}
        # The preview carries the deterministic fix (before -> after).
        assert by_row[0]["normalized"] == "Acme (Holdings)"
        assert by_row[2]["normalized"] == "Doe, Jane"
        assert by_row[0]["action"] == "create_tp"
        assert by_row[2]["action"] == "create_fn"

        # Nothing persisted on a dry run.
        after = client.get(f"/api/runs/{run_id}/labels").json()
        assert after["history_count"] == before_history


def test_apply_writes_fix_and_is_idempotent(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["created_tp"] == 1
        assert body["created_fn"] == 1

        # The uncovered row-2 comma anomaly is now an FN cell with the
        # deterministic punctuation fix.
        fn_tid = cell_target_id("vendors.csv", 2, "vendor")
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        fn = latest[fn_tid]
        assert fn["status"] == "fn"
        assert "TRIVIAL" in (fn["note"] or "").upper()
        assert fn["corrected_value"] == "Doe, Jane"

        # The covered row-0 cluster is now a TP carrying the TRIVIAL note.
        cl = latest["vendors.csv__r00000-c000"]
        assert cl["status"] == "tp"
        assert "TRIVIAL" in (cl["note"] or "").upper()

        # Re-running is a no-op (idempotent on the shared TRIVIAL token).
        r2 = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True},
        )
        body2 = r2.json()
        assert body2["created_tp"] == 0
        assert body2["created_fn"] == 0
        assert body2["skipped_already_tagged"] >= 1


def test_unlabeled_only_skips_existing_positive_label(
    tmp_path: Path,
) -> None:
    """``unlabeled_only`` leaves a reviewed positive label untouched while
    still creating labels on the still-unlabelled anomaly cells."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Operator already ruled the row-0 cluster TP.
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "vendor",
            "status": "tp", "corrected_value": "Acme (Holdings)",
            "note": "reviewed",
        })
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True, "unlabeled_only": True},
        ).json()
        assert body["unlabeled_only"] is True
        assert body["tagged"] == 0
        assert body["skipped_labeled"] == 1
        assert body["created_tp"] == 0      # only the labelled cluster covers it
        assert body["created_fn"] == 1      # row-2 comma cell still fresh

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        cl = latest["vendors.csv__r00000-c000"]
        assert cl["status"] == "tp"
        assert cl["note"] == "reviewed"     # no TRIVIAL appended
        assert "TRIVIAL" not in (cl["note"] or "").upper()


def test_include_fp_rescues_fp_cell(tmp_path: Path) -> None:
    """``include_fp`` re-checks an ``fp`` cell: when the deterministic rule
    fires, the false positive is flipped to an FN carrying the fix + TRIVIAL,
    instead of being preserved."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    fp_tid = cell_target_id("vendors.csv", 2, "vendor")
    with TestClient(app) as client:
        # Operator earlier dismissed the row-2 comma anomaly as a false
        # positive -- but it IS a self-evident punctuation defect.
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": fp_tid,
            "source": "vendors.csv", "row": 2, "column": "vendor",
            "status": "fp", "note": "not an error",
        })
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"dry_run": True, "scan_dataset": True, "include_fp": True},
        ).json()
        assert body["include_fp"] is True
        assert body["rescued_fp"] == 1
        assert body["skipped_negative_verdict"] == 0   # fp was rescued, not skipped
        row2 = next(c for c in body["candidates"] if c["row"] == 2)
        assert row2["action"] == "create_fn"
        assert row2["target_id"] == fp_tid
        assert row2["normalized"] == "Doe, Jane"
        # The preview flags the prior verdict so a rescued fp reads as
        # "fp -> fn", not a bare "fn" that looks pre-existing.
        assert row2["prior_status"] == "fp"

        # Apply flips the persisted verdict fp -> fn with the fix.
        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True, "include_fp": True},
        )
        fn = client.get(f"/api/runs/{run_id}/labels").json()["latest"][fp_tid]
        assert fn["status"] == "fn"
        assert fn["corrected_value"] == "Doe, Jane"
        assert "TRIVIAL" in (fn["note"] or "").upper()


def test_include_fp_rescues_fp_cluster(tmp_path: Path) -> None:
    """A covered cluster ruled ``fp`` is flipped to TP under ``include_fp``,
    the cluster analogue of the fp-cell rescue."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "vendor",
            "status": "fp", "note": "dismissed",
        })
        # Dry-run first: the rescued fp cluster's create_tp candidate carries
        # prior_status="fp" so the preview shows "fp -> tp" (not a bare TP that
        # looks like a pre-existing confirmed positive), while the genuinely
        # unlabelled row-2 cell's create_fn carries no prior_status.
        preview = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"dry_run": True, "scan_dataset": True, "include_fp": True},
        ).json()
        cl_cand = next(
            c for c in preview["candidates"]
            if c["target_id"] == "vendors.csv__r00000-c000"
        )
        assert cl_cand["action"] == "create_tp"
        assert cl_cand["status"] == "tp"
        assert cl_cand["prior_status"] == "fp"
        fresh = next(c for c in preview["candidates"] if c["row"] == 2)
        assert fresh["action"] == "create_fn"
        assert fresh["prior_status"] is None

        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True, "include_fp": True},
        ).json()
        assert body["rescued_fp"] == 1
        assert body["created_tp"] == 1      # the rescued fp cluster
        assert body["created_fn"] == 1      # row-2 uncovered cell (unlabelled)

        cl = client.get(
            f"/api/runs/{run_id}/labels"
        ).json()["latest"]["vendors.csv__r00000-c000"]
        assert cl["status"] == "tp"
        assert "TRIVIAL" in (cl["note"] or "").upper()


def test_include_fp_preserves_unsure_and_skip(tmp_path: Path) -> None:
    """``include_fp`` rescues ONLY ``fp``; ``unsure`` / ``skip`` verdicts are
    still preserved (they are deliberate deferrals, not dismissals)."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    unsure_tid = cell_target_id("vendors.csv", 2, "vendor")
    with TestClient(app) as client:
        # row-2 cell -> unsure; row-0 cluster -> skip.
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": unsure_tid,
            "source": "vendors.csv", "row": 2, "column": "vendor",
            "status": "unsure",
        })
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cluster", "target_id": "vendors.csv__r00000-c000",
            "source": "vendors.csv", "row": 0, "column": "vendor",
            "status": "skip",
        })
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True, "include_fp": True},
        ).json()
        assert body["rescued_fp"] == 0
        assert body["skipped_negative_verdict"] == 2
        assert body["created_tp"] == 0
        assert body["created_fn"] == 0

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest[unsure_tid]["status"] == "unsure"
        assert latest["vendors.csv__r00000-c000"]["status"] == "skip"


def test_fp_preserved_without_include_fp(tmp_path: Path) -> None:
    """Regression: with ``include_fp`` absent (default), an ``fp`` cell is
    preserved exactly as before -- no rescue."""
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    fp_tid = cell_target_id("vendors.csv", 2, "vendor")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": fp_tid,
            "source": "vendors.csv", "row": 2, "column": "vendor",
            "status": "fp",
        })
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True},
        ).json()
        assert body["include_fp"] is False
        assert body["rescued_fp"] == 0
        assert body["skipped_negative_verdict"] == 1
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest[fp_tid]["status"] == "fp"


def test_guards_are_never_written(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True},
        )
        labels = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        written_rows = {
            lb["row"] for lb in labels.values()
            if lb.get("source") == "vendors.csv"
        }
        # rows 3 (varchar(255)), 4 (1,000) and 5 (f(x)) are guarded.
        assert written_rows == {0, 2}


def test_exported_benchmark_carries_punctuation_fix(tmp_path: Path) -> None:
    project, run_id = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True},
        )

    rd = project / "outputs" / run_id
    clusters = json.loads((rd / "clusters.json").read_text())
    store = LabelStore(rd / "labels.jsonl")
    by_loc = {
        (c.source, c.row, c.column): c
        for c in materialize_label_benchmark(clusters, store)
        if c.label == "error"
    }
    # The covered row-0 cell exports with its deterministic fix end-to-end.
    cell = by_loc[("vendors.csv", 0, "vendor")]
    assert cell.original_value == "Acme(Holdings)"
    assert cell.corrected_value == "Acme (Holdings)"


def test_export_backfills_punctuation_fix_for_uncorrected_trivial_cell(
    tmp_path: Path,
) -> None:
    """A TRIVIAL-noted error cell with no own correction (e.g. fanned out
    across a multi-row cluster) gets its punctuation fix recomputed from its
    OWN value by the exporter -- never inheriting a sibling's correction."""
    store = LabelStore(tmp_path / "labels.jsonl")
    cluster = {
        "cluster_id": "c1", "source": "a.csv", "row": 0,
        "columns": ["vendor"], "row_values": {"vendor": "Acme(Holdings)"},
        "error_types": ["typo"], "detections": [],
    }
    store.append(Label(
        target=LabelTarget.CLUSTER,
        target_id="c1",
        source="a.csv",
        row=0,
        column="vendor",
        columns=["vendor"],
        status=LabelStatus.TP,
        corrected_value=None,                  # no own-anchor fix
        note="(auto: punctuation anomaly) TRIVIAL",
        annotator="t",
    ))
    cells = materialize_label_benchmark([cluster], store)
    cell = next(c for c in cells if c.column == "vendor")
    assert cell.label == "error"
    assert cell.original_value == "Acme(Holdings)"
    assert cell.corrected_value == "Acme (Holdings)"


def _build_convention_run(tmp_path: Path) -> tuple[Path, str]:
    """A CSV with a column whose punctuation pattern is a CONVENTION (every
    cell is ``word,word``) and a separate column with one isolated anomaly.

    The convention column must be left entirely alone; the isolated anomaly
    must still be flagged.
    """
    project = tmp_path / "conv"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "tweets.csv"
    # ``pairs`` column: every cell is comma-joined (the convention).
    # ``vendor`` column: clean except one row with a missing-space paren.
    csv.write_text(
        "id,pairs,vendor\n"
        '1,"people,analytics",Acme Corp\n'
        '2,"analytics,world",Beta LLC\n'
        '3,"looking,forward",Gamma Inc\n'
        '4,"use,code",Delta Co\n'
        '5,"list,speakers",Acme(Holdings)\n',
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260514T000000Z_conv"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")
    return project, run_id


def test_column_convention_is_not_flagged(tmp_path: Path) -> None:
    project, run_id = _build_convention_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"dry_run": True, "scan_dataset": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Only the isolated paren anomaly in `vendor` qualifies; the entire
        # comma-joined `pairs` column is suppressed as a convention.
        by_loc = {(c["column"], c["row"]) for c in body["candidates"]}
        assert by_loc == {("vendor", 4)}
        assert body["anomaly_cells"] == 1
        cand = next(
            c for c in body["candidates"] if c["column"] == "vendor"
        )
        assert cand["normalized"] == "Acme (Holdings)"

    # And nothing from the convention column is written on apply.
    with TestClient(app) as client:
        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-punctuation",
            json={"scan_dataset": True},
        )
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        written = [
            lb for lb in latest.values() if lb.get("source") == "tweets.csv"
        ]
        assert len(written) == 1
        assert written[0]["column"] == "vendor"
        assert written[0]["row"] == 4

