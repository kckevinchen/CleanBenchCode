"""POST /api/runs/{run_id}/labels/tag-trivial-inconsistency auto-tags
*trivial* (normalisation-collision) inconsistencies as COLUMN_CLUSTER TP
verdicts.

Two distinct values in one column are trivially inconsistent when they
collapse to the same string after case-fold + drop-whitespace +
drop-punctuation (``_letter_core``). Purely numeric collisions (``2.5``
vs ``25``) are EXCLUDED because stripping ``.`` / ``-`` merges different
numbers.

Mirrors ``tests/test_tag_trivial_row_duplication.py``: a small synthetic
run with hand-crafted ``column_cluster_drafts.jsonl`` + human verdicts,
a TestClient flow, and assertions on the dry-run preview, the persisted
append-only result, idempotency, and that the synthesised clusters
resolve in the benchmark export index.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app
from src.labeling.correction_suggester import (
    build_cluster_index,
    load_drafts,
    load_labels,
)
from src.labeling._helpers import (
    _inconsistency_collision_groups,
    _is_numeric_norm_key,
    _trivial_inconsistency_groups,
)
from src.labeling.schema import (
    ExclusionIn,
    ExclusionScope,
    Label,
    LabelStatus,
    LabelTarget,
)
from src.labeling.store import LabelStore


def test_is_numeric_norm_key() -> None:
    assert _is_numeric_norm_key("25")          # 2.5 / 25 collapse here
    assert _is_numeric_norm_key("03")          # -0.3 / 0.3
    assert not _is_numeric_norm_key("covid")
    assert not _is_numeric_norm_key("a2024900115")  # has a letter
    assert not _is_numeric_norm_key("")


def test_trivial_groups_rule() -> None:
    # Casing + punctuation collisions are kept; numeric is dropped;
    # singletons are ignored.
    groups = _trivial_inconsistency_groups([
        "Clarington Orioles", "Clarington orioles",   # casing
        "#covid", "covid",                            # punctuation
        "2.5", "25",                                  # numeric -> dropped
        "lonely",                                     # singleton -> dropped
    ])
    keys = {k: (raws, kind) for k, raws, kind in groups}
    assert set(keys) == {"claringtonorioles", "covid"}
    assert keys["claringtonorioles"][1] == "casing"
    assert keys["covid"][1] == "punctuation"


def test_collision_groups_flags_numeric() -> None:
    groups = _inconsistency_collision_groups(["2.5", "25", "#covid", "covid"])
    flags = {k: is_num for k, _raws, _kind, is_num in groups}
    assert flags["25"] is True
    assert flags["covid"] is False


def _build_run(tmp_path: Path) -> tuple[Path, str, Path]:
    """Tiny run whose single CSV column ``city`` holds five collision
    groups exercising every branch:

      * {USA, U.S.A.}    punctuation, NO draft cluster   -> create_tp_new
      * {Calif, calif}   casing, draft c1 labelled tp    -> captured_human
      * {Foo, foo}       casing, draft c2 labelled fp    -> skipped_negative
      * {Bar, bar}       casing, draft c3 UNLABELLED     -> create_tp_reuse
      * {2.5, 25}        numeric                         -> numeric_skipped
    """
    project = tmp_path / "ti"
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "geo.csv"
    csv.write_text(
        "city\n"
        "USA\n"        # row 0
        "U.S.A.\n"     # row 1  -> trivial w/ row 0 (no draft) -> create_new
        "Calif\n"      # row 2
        "calif\n"      # row 3  -> trivial w/ row 2 (draft c1 tp) -> captured
        "Foo\n"        # row 4
        "foo\n"        # row 5  -> trivial w/ row 4 (draft c2 fp) -> skip_neg
        "Bar\n"        # row 6
        "bar\n"        # row 7  -> trivial w/ row 6 (draft c3 unlabelled) reuse
        "2.5\n"        # row 8
        "25\n"         # row 9  -> numeric collision -> skipped
        "Unique\n",    # row 10 singleton
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260616T000000Z_incon"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")

    # Three LLM draft clusters for (geo.csv, city) in positional order
    # c1, c2, c3 -- build_cluster_index assigns c{idx} by position.
    (rd / "column_cluster_drafts.jsonl").write_text(
        json.dumps({
            "source": "geo.csv",
            "column": "city",
            "draft_clusters": [
                {"kind": "casing", "canonical": None,
                 "values": ["Calif", "calif"]},          # c1
                {"kind": "casing", "canonical": None,
                 "values": ["Foo", "foo"]},              # c2
                {"kind": "casing", "canonical": None,
                 "values": ["Bar", "bar"]},              # c3
            ],
        }) + "\n",
        encoding="utf-8",
    )

    # Human verdicts: c1 = tp (captured), c2 = fp (rejected). c3 unlabelled.
    store = LabelStore(rd / "labels.jsonl")
    store.append_many([
        Label(
            target=LabelTarget.COLUMN_CLUSTER,
            target_id="colclust\x00geo.csv\x00city\x00c1",
            source="geo.csv", row=None, column="city", columns=["city"],
            status=LabelStatus.TP, note="", annotator="human",
        ),
        Label(
            target=LabelTarget.COLUMN_CLUSTER,
            target_id="colclust\x00geo.csv\x00city\x00c2",
            source="geo.csv", row=None, column="city", columns=["city"],
            status=LabelStatus.FP, note="", annotator="human",
        ),
    ])
    return project, run_id, rd


def test_dry_run_counts_every_branch(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is True
        assert body["groups_total"] == 4          # numeric excluded
        assert body["numeric_skipped"] == 1
        assert body["captured_human"] == 1        # {Calif, calif}
        assert body["skipped_negative"] == 1      # {Foo, foo}
        assert body["to_create_new"] == 1         # {USA, U.S.A.}
        assert body["to_create_reuse"] == 1       # {Bar, bar}
        assert "not_captured" not in body         # removed (misleading: rule FPs)
        assert body["to_write"] == 2
        # Every candidate carries a stable selection key.
        assert all("key" in c for c in body["candidates"])
        # Nothing written on dry-run.
        assert (
            client.get(f"/api/runs/{run_id}/labels").json()["history_count"] == 2
        )


def test_apply_creates_tp_and_synthesises_draft(tmp_path: Path) -> None:
    project, run_id, rd = _build_run(tmp_path)
    drafts_before = sum(1 for _ in open(rd / "column_cluster_drafts.jsonl"))
    app = create_app(project, annotator="auto")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency", json={},
        ).json()
        assert r["created_new"] == 1
        assert r["created_reuse"] == 1

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        # Reuse path: c3 now carries an auto TP verdict.
        reuse = latest["colclust\x00geo.csv\x00city\x00c3"]
        assert reuse["status"] == "tp"
        assert reuse["target"] == "column_cluster"
        assert "trivial inconsistency" in reuse["note"].lower()
        # Synth path: a NEW positional cluster (c4) was created for {USA,..}.
        synth = latest["colclust\x00geo.csv\x00city\x00c4"]
        assert synth["status"] == "tp"
        assert "trivial inconsistency" in synth["note"].lower()
        # Human verdicts are untouched (still their original note).
        assert latest["colclust\x00geo.csv\x00city\x00c1"]["status"] == "tp"
        assert latest["colclust\x00geo.csv\x00city\x00c1"]["note"] == ""
        assert latest["colclust\x00geo.csv\x00city\x00c2"]["status"] == "fp"

    # One synthesised draft doc line was appended for (geo.csv, city).
    drafts_after = sum(1 for _ in open(rd / "column_cluster_drafts.jsonl"))
    assert drafts_after == drafts_before + 1

    # The synth cluster resolves to EXACTLY the {USA, U.S.A.} value-set at
    # its positional id, so it feeds the inconsistency benchmark.
    drafts = load_drafts(rd / "column_cluster_drafts.jsonl")
    labels = load_labels(rd / "labels.jsonl")
    cidx = build_cluster_index(drafts, labels)
    by_id = {c.local_id: c for c in cidx[("geo.csv", "city")]}
    assert by_id["c4"].value_set == frozenset({"USA", "U.S.A."})
    assert by_id["c4"].status == "tp"
    assert by_id["c3"].value_set == frozenset({"Bar", "bar"})
    assert by_id["c3"].status == "tp"


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    project, run_id, rd = _build_run(tmp_path)
    app = create_app(project, annotator="auto")
    with TestClient(app) as client:
        first = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency", json={},
        ).json()
        assert first["to_write"] == 2
        drafts_after_first = sum(
            1 for _ in open(rd / "column_cluster_drafts.jsonl")
        )
        labels_after_first = (
            client.get(f"/api/runs/{run_id}/labels").json()["history_count"]
        )

        second = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        ).json()
        assert second["to_write"] == 0
        assert second["already_auto_tagged"] == 2
        assert second["captured_human"] == 1     # auto not counted as human

        applied = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency", json={},
        ).json()
        assert applied["created_new"] == 0
        assert applied["created_reuse"] == 0

    # No new drafts and no new labels on the idempotent re-apply.
    assert sum(1 for _ in open(rd / "column_cluster_drafts.jsonl")) == (
        drafts_after_first
    )
    app2 = create_app(project, annotator="auto")
    with TestClient(app2) as client:
        assert (
            client.get(f"/api/runs/{run_id}/labels").json()["history_count"]
            == labels_after_first
        )


def test_numeric_groups_never_tagged(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency", json={},
        ).json()
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        # No auto verdict mentions the numeric "25"/"2.5" collision: the
        # only auto clusters are {USA,..} and {Bar, bar}.
        drafts = load_drafts(_rd / "column_cluster_drafts.jsonl")
        labels = load_labels(_rd / "labels.jsonl")
        cidx = build_cluster_index(drafts, labels)
        auto = [
            c for c in cidx[("geo.csv", "city")]
            if c.status == "tp" and "trivial inconsistency" in (c.note or "").lower()
        ]
        tagged_values = {v for c in auto for v in c.value_set}
        assert "2.5" not in tagged_values and "25" not in tagged_values
        assert r["numeric_skipped"] == 1


def test_source_filter_404(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"source": "nope.csv"},
        )
        assert r.status_code == 404


def test_source_filter_scopes_scan(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"source": "geo.csv", "dry_run": True},
        ).json()
        assert r["groups_total"] == 4
        assert all(c["source"] == "geo.csv" for c in r["candidates"])


def test_csv_exclusion_drops_all_groups(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Exclude the whole CSV: nothing in it is a finding anymore.
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "csv", "source": "geo.csv"},
        )
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        ).json()
        assert r["groups_total"] == 0
        assert r["to_write"] == 0
        assert r["excluded_files"] == 1


def test_column_exclusion_skips_column(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Exclude the only column -> every group is dropped, counted as a
        # skipped column slice (not a CSV-level drop).
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "columns", "source": "geo.csv",
                  "columns": ["city"]},
        )
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        ).json()
        assert r["groups_total"] == 0
        assert r["excluded_columns_skipped"] == 1
        assert r["excluded_files"] == 0


def test_row_exclusion_drops_only_excluded_group(tmp_path: Path) -> None:
    project, run_id, rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        baseline = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        ).json()
        # rows 4 & 5 are the {Foo, foo} group (a negative human cluster);
        # excluding them removes that group from the scan entirely so the
        # skipped_negative count drops by one.
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "rows", "source": "geo.csv", "rows": [4, 5]},
        )
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        ).json()
        assert r["groups_total"] == baseline["groups_total"] - 1
        assert r["skipped_negative"] == baseline["skipped_negative"] - 1
        # The {USA, U.S.A.} create-new group is untouched.
        assert r["to_create_new"] == baseline["to_create_new"]


def test_excluded_rows_not_in_synthesised_cluster(tmp_path: Path) -> None:
    """A row excluded from the scan must not leak into a synthesised
    cluster's value-set."""
    project = tmp_path / "ex"
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "g.csv"
    # rows 0,1 -> USA/U.S.A. (kept); row 2 -> "u s a" also normalises to the
    # same key but will be EXCLUDED, so it must not join the synthesised
    # cluster.
    csv.write_text("city\nUSA\nU.S.A.\nu s a\nSolo\n", encoding="utf-8")
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260616T010101Z_exrow"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")
    (rd / "column_cluster_drafts.jsonl").write_text("")
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(
            f"/api/runs/{run_id}/exclusions",
            json={"scope": "rows", "source": "g.csv", "rows": [2]},
        )
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency", json={},
        ).json()
        assert r["created_new"] == 1
    drafts = load_drafts(rd / "column_cluster_drafts.jsonl")
    labels = load_labels(rd / "labels.jsonl")
    cidx = build_cluster_index(drafts, labels)
    synth = cidx[("g.csv", "city")][0]
    # Only the two kept rows' values; the excluded "u s a" is absent.
    assert synth.value_set == frozenset({"USA", "U.S.A."})


def test_selected_keys_tags_only_chosen_pairs(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="auto")
    with TestClient(app) as client:
        preview = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        ).json()
        # Pick ONLY the create-new group ({USA, U.S.A.}); leave the reuse
        # group ({Bar, bar}) unselected.
        new_key = next(
            c["key"] for c in preview["candidates"]
            if c["action"] == "create_tp_new"
        )
        reuse_key = next(
            c["key"] for c in preview["candidates"]
            if c["action"] == "create_tp_reuse"
        )
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"selected_keys": [new_key]},
        ).json()
        assert r["selection_active"] is True
        assert r["selected_count"] == 1
        assert r["created_new"] == 1
        assert r["created_reuse"] == 0

        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        # The chosen group was synthesised (c4) + tagged TP.
        synth = latest["colclust\x00geo.csv\x00city\x00c4"]
        assert synth["status"] == "tp"
        # The UNSELECTED reuse target (c3) got NO auto verdict.
        assert "colclust\x00geo.csv\x00city\x00c3" not in latest
        # Re-running with the other key now tags the reuse group.
        r2 = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"selected_keys": [reuse_key]},
        ).json()
        assert r2["created_reuse"] == 1
        latest2 = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        assert latest2["colclust\x00geo.csv\x00city\x00c3"]["status"] == "tp"


def test_empty_selection_writes_nothing(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="auto")
    with TestClient(app) as client:
        before = client.get(f"/api/runs/{run_id}/labels").json()["history_count"]
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"selected_keys": []},
        ).json()
        assert r["selection_active"] is True
        assert r["selected_count"] == 0
        assert r["created_new"] == 0
        assert r["created_reuse"] == 0
        after = client.get(f"/api/runs/{run_id}/labels").json()["history_count"]
        assert after == before


def test_tagged_pair_appears_in_flagged_list_as_tp(tmp_path: Path) -> None:
    """A tagged pair surfaces in the main label-page queue
    (``/column-clusters``) as a TP inconsistency col-cluster item -- i.e.
    it behaves exactly like an LLM-flagged + human-TP cluster."""
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="auto")
    with TestClient(app) as client:
        preview = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        ).json()
        new_key = next(
            c["key"] for c in preview["candidates"]
            if c["action"] == "create_tp_new"
        )
        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"selected_keys": [new_key]},
        )
        queue = client.get(f"/api/runs/{run_id}/column-clusters").json()
        tagged = [
            it for it in queue["items"]
            if it["label_status"] == "tp"
            and set(v["value"] for v in it["values"]) == {"USA", "U.S.A."}
        ]
        assert len(tagged) == 1
        item = tagged[0]
        assert item["is_col_cluster"] is True
        assert item["error_types"] == ["inconsistency"]
        assert item["from_draft"] is True


def test_selected_keys_invalid_type_400(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"selected_keys": "not-a-list"},
        )
        assert r.status_code == 400


def _rejected_key(client, run_id: str) -> str:
    """The selection key of the run's single ``skipped_negative`` group
    ({Foo, foo}, rejected fp via draft c2)."""
    preview = client.post(
        f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
        json={"dry_run": True},
    ).json()
    return next(
        c["key"] for c in preview["candidates"]
        if c["action"] == "skipped_negative"
    )


def test_rejected_group_is_overridable_in_preview(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        ).json()
        assert body["skipped_negative"] == 1
        assert body["to_override"] == 1
        neg = next(
            c for c in body["candidates"] if c["action"] == "skipped_negative"
        )
        # Carries the existing verdict + the full cluster value-set to flip.
        assert neg["status"] == "fp"
        assert neg["cluster_values"] == ["Foo", "foo"]
        assert "key" in neg


def test_tag_all_never_overrides_rejected(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="auto")
    with TestClient(app) as client:
        # selected_keys omitted -> "tag all"; must NOT touch the fp cluster.
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency", json={},
        ).json()
        assert r["created_override"] == 0
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        # c2 ({Foo, foo}) stays fp.
        assert latest["colclust\x00geo.csv\x00city\x00c2"]["status"] == "fp"


def test_override_flips_rejected_to_tp(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="auto")
    with TestClient(app) as client:
        key = _rejected_key(client, run_id)
        r = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"selected_keys": [key]},
        ).json()
        assert r["created_override"] == 1
        assert r["created_new"] == 0
        assert r["created_reuse"] == 0
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        flipped = latest["colclust\x00geo.csv\x00city\x00c2"]
        assert flipped["status"] == "tp"
        assert "trivial inconsistency" in flipped["note"].lower()
        # It now surfaces in the flagged list as a TP inconsistency.
        queue = client.get(f"/api/runs/{run_id}/column-clusters").json()
        tp_foo = [
            it for it in queue["items"]
            if it["label_status"] == "tp"
            and set(v["value"] for v in it["values"]) == {"Foo", "foo"}
        ]
        assert len(tp_foo) == 1


def test_override_is_idempotent(tmp_path: Path) -> None:
    project, run_id, _rd = _build_run(tmp_path)
    app = create_app(project, annotator="auto")
    with TestClient(app) as client:
        key = _rejected_key(client, run_id)
        client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"selected_keys": [key]},
        )
        # After the flip, the group is no longer rejected; a re-scan shows
        # it as already-auto-tagged with nothing left to override.
        body = client.post(
            f"/api/runs/{run_id}/labels/tag-trivial-inconsistency",
            json={"dry_run": True},
        ).json()
        assert body["skipped_negative"] == 0
        assert body["to_override"] == 0
        assert body["already_auto_tagged"] == 1



