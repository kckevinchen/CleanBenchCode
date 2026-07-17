"""Tests for LLM cell-correction of trivially-tagged PARTIAL labels
(``src.labeling.trivial_corrector`` + the ``trivial-partial-corrections``
job endpoints). The LLM is stubbed throughout -- no live service.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.labeling import trivial_corrector as tc
from src.labeling.app import RunContext, create_app
from src.labeling.routers import trivial_corrections as tc_router
from src.labeling.schema import Label, LabelStatus, LabelTarget, cell_target_id

# ---------------------------------------------------------------------------
# Stub LLM
# ---------------------------------------------------------------------------


class StubLLM:
    """Returns a canned ``corrected_value`` chosen by substring match on the
    prompt's original value; records the prompts it saw."""

    def __init__(self, mapping: dict[str, str], *, raise_on: str | None = None):
        self.mapping = mapping
        self.raise_on = raise_on
        self.prompts: list[str] = []

    def query(self, model, prompt, **kw):
        self.prompts.append(prompt)
        if self.raise_on and self.raise_on in prompt:
            raise RuntimeError("boom")
        for needle, fix in self.mapping.items():
            if needle in prompt:
                return SimpleNamespace(
                    text=json.dumps({"corrected_value": fix}), error=None,
                )
        return SimpleNamespace(text=json.dumps({"corrected_value": ""}), error=None)


# ---------------------------------------------------------------------------
# parse_corrected_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ('{"corrected_value": "John Smith"}', "John Smith"),
    ('```json\n{"corrected_value": "CA"}\n```', "CA"),
    ('Here you go: {"corrected_value": "Acme Inc"} done', "Acme Inc"),
    ('{"value": "fallback"}', "fallback"),
    ('"bare string"', "bare string"),
    ('{"corrected_value": ""}', ""),
])
def test_parse_corrected_value(raw: str, expected: str) -> None:
    assert tc.parse_corrected_value(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "not json at all", "{no quotes}"])
def test_parse_corrected_value_none(raw: str) -> None:
    assert tc.parse_corrected_value(raw) is None


def test_build_cell_prompt_includes_context() -> None:
    p = tc.build_cell_prompt(
        column="name", original_value="Jhon  Smith",
        row_values={"id": "1", "name": "Jhon  Smith"},
        column_samples=["Alice", "Bob"],
    )
    assert "name" in p
    assert "Jhon  Smith" in p
    assert "Alice" in p
    assert "corrected_value" in p


# ---------------------------------------------------------------------------
# Run fixtures
# ---------------------------------------------------------------------------


def _build_run(tmp_path: Path) -> tuple[Path, str, Path]:
    project = tmp_path / "fix"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    (project / "data").mkdir()
    csv = project / "data" / "v.csv"
    csv.write_text(
        'id,name\n1,"Jhon  Smith"\n2,Alice\n3,"Acme(Holdings)"\n',
        encoding="utf-8",
    )
    rd = project / "outputs" / "20260101T000000Z_fix"
    rd.mkdir(parents=True)
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")
    return project, "20260101T000000Z_fix", csv


def _partial_trivial_cell(
    ctx: RunContext, *, row: int, column: str, corrected: str, note: str,
) -> str:
    tid = cell_target_id("v.csv", row, column)
    ctx.store.append(Label(
        target=LabelTarget.CELL, target_id=tid, source="v.csv",
        row=row, column=column, columns=[column], status=LabelStatus.PARTIAL,
        corrected_value=corrected, note=note, annotator="t",
    ))
    return tid


# ---------------------------------------------------------------------------
# collect_partial_trivial_targets
# ---------------------------------------------------------------------------


def test_collect_only_partial_trivial_unfixed(tmp_path: Path) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    ctx = RunContext(project / "outputs" / run_id)
    # eligible: partial + TRIVIAL + not yet [LLM-fix]
    _partial_trivial_cell(ctx, row=0, column="name", corrected="Jhon Smith",
                          note="(auto: spacing anomaly) TRIVIAL")
    # excluded: already LLM-fixed
    _partial_trivial_cell(ctx, row=2, column="name", corrected="x",
                          note="(auto: punctuation anomaly) TRIVIAL [LLM-fix]")
    # excluded: not TRIVIAL
    ctx.store.append(Label(
        target=LabelTarget.CELL, target_id=cell_target_id("v.csv", 1, "name"),
        source="v.csv", row=1, column="name", columns=["name"],
        status=LabelStatus.PARTIAL, note="manual partial", annotator="t",
    ))

    targets = tc.collect_partial_trivial_targets(ctx)
    assert [(t.row, t.column, t.original_value) for t in targets] == [
        (0, "name", "Jhon  Smith"),   # live value (double space preserved)
    ]


def test_collect_reads_live_value_after_normalization(tmp_path: Path) -> None:
    import os
    project, run_id, csv = _build_run(tmp_path)
    ctx = RunContext(project / "outputs" / run_id)
    _partial_trivial_cell(ctx, row=0, column="name", corrected="Jhon Smith",
                          note="(auto: spacing anomaly) TRIVIAL")
    # Normalize the CSV on disk (collapse the double space) -> collect must see it.
    csv.write_text('id,name\n1,"Jhon Smith"\n2,Alice\n3,"Acme(Holdings)"\n',
                   encoding="utf-8")
    st = csv.stat()
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    targets = tc.collect_partial_trivial_targets(ctx)
    assert targets[0].original_value == "Jhon Smith"


# ---------------------------------------------------------------------------
# generate + write
# ---------------------------------------------------------------------------


def test_generate_and_write_corrects_label(tmp_path: Path) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    ctx = RunContext(project / "outputs" / run_id)
    tid = _partial_trivial_cell(ctx, row=0, column="name", corrected="Jhon Smith",
                                note="(auto: spacing anomaly) TRIVIAL")

    targets = tc.collect_partial_trivial_targets(ctx)
    stub = StubLLM({"Jhon  Smith": "John Smith"})
    results = tc.generate_corrections(ctx, targets, llm_client=stub)
    summary = tc.write_corrections(ctx, results, annotator="t")

    assert summary == {
        "labels_written": 1, "cells_corrected": 1, "unchanged": 0, "failed": 0,
    }
    lab = ctx.store.latest_by_target()[tid]
    assert lab.status == LabelStatus.PARTIAL
    assert lab.corrected_value == "John Smith"
    assert tc.LLM_FIX_MARKER in lab.note
    # idempotent: re-collect skips the [LLM-fix] label.
    assert tc.collect_partial_trivial_targets(ctx) == []


def test_failed_llm_call_leaves_label_for_retry(tmp_path: Path) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    ctx = RunContext(project / "outputs" / run_id)
    tid = _partial_trivial_cell(ctx, row=0, column="name", corrected="Jhon Smith",
                                note="(auto: spacing anomaly) TRIVIAL")
    targets = tc.collect_partial_trivial_targets(ctx)
    stub = StubLLM({}, raise_on="Jhon  Smith")
    results = tc.generate_corrections(ctx, targets, llm_client=stub)
    summary = tc.write_corrections(ctx, results, annotator="t")
    assert summary["failed"] == 1
    assert summary["labels_written"] == 0
    # The label is untouched (no [LLM-fix]) so a re-run retries it.
    lab = ctx.store.latest_by_target()[tid]
    assert tc.LLM_FIX_MARKER not in (lab.note or "")
    assert len(tc.collect_partial_trivial_targets(ctx)) == 1


def test_unchanged_value_is_marked_but_counted_unchanged(tmp_path: Path) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    ctx = RunContext(project / "outputs" / run_id)
    tid = _partial_trivial_cell(ctx, row=1, column="name", corrected="Alice",
                                note="(auto: spacing anomaly) TRIVIAL")
    targets = tc.collect_partial_trivial_targets(ctx)
    # LLM says the value is already correct (returns it unchanged).
    stub = StubLLM({"Alice": "Alice"})
    results = tc.generate_corrections(ctx, targets, llm_client=stub)
    summary = tc.write_corrections(ctx, results, annotator="t")
    assert summary["unchanged"] == 1
    assert summary["cells_corrected"] == 0
    # Still marked so it is not reprocessed.
    lab = ctx.store.latest_by_target()[tid]
    assert tc.LLM_FIX_MARKER in lab.note


# ---------------------------------------------------------------------------
# collect_fn_targets + FN status-preserving write
# ---------------------------------------------------------------------------


def _fn_cell(
    ctx: RunContext, *, row: int, column: str,
    corrected: str | None = None, note: str = "",
) -> str:
    tid = cell_target_id("v.csv", row, column)
    ctx.store.append(Label(
        target=LabelTarget.CELL, target_id=tid, source="v.csv",
        row=row, column=column, columns=[column], status=LabelStatus.FN,
        corrected_value=corrected, note=note, annotator="t",
    ))
    return tid


def test_collect_fn_targets_includes_all_fn_regardless_of_note(tmp_path: Path) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    ctx = RunContext(project / "outputs" / run_id)
    # FN with a deterministic punctuation fix already present (no TRIVIAL gate).
    _fn_cell(ctx, row=2, column="name", corrected="Acme (Holdings)",
             note="(auto: punctuation anomaly) TRIVIAL")
    # FN marked by hand, no note, no fix.
    _fn_cell(ctx, row=0, column="name")
    # Excluded: already [LLM-fix]-marked (idempotent re-run).
    _fn_cell(ctx, row=1, column="name", corrected="Alice",
             note="(auto: punctuation anomaly) TRIVIAL [LLM-fix]")

    targets = tc.collect_fn_targets(ctx)
    # Both unfixed FN cells collected (live values), the [LLM-fix] one skipped.
    assert sorted((t.row, t.original_value) for t in targets) == [
        (0, "Jhon  Smith"),
        (2, "Acme(Holdings)"),
    ]
    # Forcing regeneration re-includes the [LLM-fix] one.
    assert len(tc.collect_fn_targets(ctx, skip_already_fixed=False)) == 3


def test_collect_fn_targets_ignores_non_fn_cells(tmp_path: Path) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    ctx = RunContext(project / "outputs" / run_id)
    _fn_cell(ctx, row=0, column="name")
    # A partial cell must NOT be picked up by the FN collector.
    _partial_trivial_cell(ctx, row=2, column="name", corrected="x",
                          note="(auto: punctuation anomaly) TRIVIAL")
    targets = tc.collect_fn_targets(ctx)
    assert [(t.row, t.column) for t in targets] == [(0, "name")]


def test_fn_fix_preserves_fn_status(tmp_path: Path) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    ctx = RunContext(project / "outputs" / run_id)
    tid = _fn_cell(ctx, row=0, column="name",
                   note="(auto: punctuation anomaly) TRIVIAL")
    targets = tc.collect_fn_targets(ctx)
    stub = StubLLM({"Jhon  Smith": "John Smith"})
    results = tc.generate_corrections(ctx, targets, llm_client=stub)
    summary = tc.write_corrections(ctx, results, annotator="bot")
    assert summary == {
        "labels_written": 1, "cells_corrected": 1, "unchanged": 0, "failed": 0,
    }
    lab = ctx.store.latest_by_target()[tid]
    # The cell stays an FN (NOT downgraded to partial) and gets the LLM fix.
    assert lab.status == LabelStatus.FN
    assert lab.corrected_value == "John Smith"
    assert tc.LLM_FIX_MARKER in lab.note
    # Idempotent: the [LLM-fix] marker excludes it from a re-collect.
    assert tc.collect_fn_targets(ctx) == []


def test_partial_fix_still_preserves_partial_status(tmp_path: Path) -> None:
    # Regression: the status-preserving write must not change the partial flow.
    project, run_id, _csv = _build_run(tmp_path)
    ctx = RunContext(project / "outputs" / run_id)
    tid = _partial_trivial_cell(ctx, row=0, column="name", corrected="Jhon Smith",
                                note="(auto: spacing anomaly) TRIVIAL")
    targets = tc.collect_partial_trivial_targets(ctx)
    results = tc.generate_corrections(ctx, targets, llm_client=StubLLM({"Jhon  Smith": "John Smith"}))
    tc.write_corrections(ctx, results, annotator="t")
    assert ctx.store.latest_by_target()[tid].status == LabelStatus.PARTIAL


# ---------------------------------------------------------------------------
# Endpoint (background job) -- with stubbed LLM factory
# ---------------------------------------------------------------------------


def _poll_until_done(client: TestClient, run_id: str, job_id: str,
                     timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(
            f"/api/runs/{run_id}/trivial-partial-corrections/{job_id}",
        ).json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_endpoint_dry_run_previews_without_llm(tmp_path: Path, monkeypatch) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    # If the LLM were called in dry-run, this stub would record it.
    stub = StubLLM({"Jhon  Smith": "John Smith"})
    monkeypatch.setattr(tc_router, "_llm_factory", lambda: stub)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # Seed a partial+TRIVIAL label through the live store.
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "x", "source": "v.csv",
            "row": 0, "column": "name", "status": "partial",
            "corrected_value": "Jhon Smith",
            "note": "(auto: spacing anomaly) TRIVIAL",
        })
        r = client.post(
            f"/api/runs/{run_id}/labels/trivial-partial-corrections",
            json={"dry_run": True},
        ).json()
        assert r["dry_run"] is True
        assert r["total_targets"] == 1
        assert r["targets"][0]["row"] == 0
    assert stub.prompts == []  # no LLM calls on dry-run


def test_endpoint_runs_job_and_writes_fix(tmp_path: Path, monkeypatch) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    stub = StubLLM({"Jhon  Smith": "John Smith"})
    monkeypatch.setattr(tc_router, "_llm_factory", lambda: stub)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        client.post(f"/api/runs/{run_id}/labels", json={
            "target": "cell", "target_id": "x", "source": "v.csv",
            "row": 0, "column": "name", "status": "partial",
            "corrected_value": "Jhon Smith",
            "note": "(auto: spacing anomaly) TRIVIAL",
        })
        start = client.post(
            f"/api/runs/{run_id}/labels/trivial-partial-corrections", json={},
        ).json()
        assert start["total_targets"] == 1
        assert start["job_id"]
        body = _poll_until_done(client, run_id, start["job_id"])
        assert body["status"] == "done"
        assert body["summary"]["cells_corrected"] == 1
        # The persisted label now carries the LLM fix.
        latest = client.get(f"/api/runs/{run_id}/labels").json()["latest"]
        tid = cell_target_id("v.csv", 0, "name")
        assert latest[tid]["corrected_value"] == "John Smith"
        assert tc.LLM_FIX_MARKER in latest[tid]["note"]


def test_endpoint_no_targets_returns_done(tmp_path: Path, monkeypatch) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    monkeypatch.setattr(tc_router, "_llm_factory", lambda: StubLLM({}))
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/trivial-partial-corrections", json={},
        ).json()
        assert r["job_id"] is None
        assert r["status"] == "done"
        assert r["total_targets"] == 0


def test_endpoint_unknown_source_404(tmp_path: Path) -> None:
    project, run_id, _csv = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/labels/trivial-partial-corrections",
            json={"source": "nope.csv"},
        )
        assert r.status_code == 404
