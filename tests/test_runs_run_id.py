"""Regression tests for launch-time run_id assignment (follow-up item 1).

The old launcher inferred which ``outputs/<id>/`` dir a detection job
created by diffing the run registry after the job exited
(``_resolve_run_id``). Under concurrent launches that cross-wired jobs to
each other's dirs, and because the auto-generated name was only
second-resolution two same-model launches in the same second minted the
*same* dir name and clobbered each other.

The fix assigns a unique ``run_id`` at launch (``_mint_run_id``) and hands
it to the child via ``--run-name``; the child writes ``outputs/<id>/``
verbatim. These tests pin that behaviour: the id is unique + filesystem
safe, the argv carries it, and the Job records it immediately (not after
reaping), distinct across concurrent launches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.labeling import runs as runs_mod
from src.labeling.cli_builders import build_detection_argv
from src.labeling.runs import JobManager, _mint_run_id

_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z_.+_[0-9a-f]{8}$")


# ---------------------------------------------------------------------------
# _mint_run_id
# ---------------------------------------------------------------------------


class TestMintRunId:
    def test_format(self) -> None:
        rid = _mint_run_id("gpt-5.2")
        assert _RUN_ID_RE.match(rid), rid
        assert "gpt-5.2" in rid

    def test_unique_across_calls_same_model(self) -> None:
        # Same model, same wall-clock second: uniqueness must come from
        # the random suffix, not the timestamp (the old collision bug).
        ids = {_mint_run_id("gpt-5.2") for _ in range(200)}
        assert len(ids) == 200

    def test_filesystem_safe(self) -> None:
        rid = _mint_run_id("weird/model name,with:junk")
        assert "/" not in rid
        assert " " not in rid
        assert ":" not in rid
        assert _RUN_ID_RE.match(rid), rid

    def test_comma_list_folds_to_family(self) -> None:
        rid = _mint_run_id("gpt-5.2,gpt-5.4,gpt-5.5")
        assert "gpt5-family" in rid

    @pytest.mark.parametrize("spec", [None, "", "   ", ",", "///"])
    def test_empty_or_junk_spec_falls_back(self, spec) -> None:
        rid = _mint_run_id(spec)
        assert "_model_" in rid
        assert _RUN_ID_RE.match(rid), rid


# ---------------------------------------------------------------------------
# build_detection_argv carries --run-name
# ---------------------------------------------------------------------------


def test_detection_argv_emits_run_name() -> None:
    argv = build_detection_argv({"model": "gpt-5.2", "run_name": "RID_123"})
    assert "--run-name" in argv
    assert argv[argv.index("--run-name") + 1] == "RID_123"


def test_detection_argv_omits_run_name_when_absent() -> None:
    # Direct-CLI parity: no run_name in payload -> no flag, child auto-names.
    argv = build_detection_argv({"model": "gpt-5.2"})
    assert "--run-name" not in argv


# ---------------------------------------------------------------------------
# JobManager: run_id assigned at launch, unique under concurrency
# ---------------------------------------------------------------------------


class _FakePopen:
    """Stand-in for subprocess.Popen: never actually spawns a child."""

    def __init__(self, *args, **kwargs) -> None:
        self.pid = 4242

    def wait(self) -> int:
        return 0

    def poll(self) -> int:
        return 0


@pytest.fixture()
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobManager:
    monkeypatch.setattr(runs_mod.subprocess, "Popen", _FakePopen)
    return JobManager(tmp_path, log_dir=tmp_path / "logs" / "jobs")


def test_detection_run_id_set_at_launch(manager: JobManager) -> None:
    job = manager.launch_detection({"csv": "data/x.csv", "model": "gpt-5.2"})
    # Set immediately -- not deferred to the reaper as the old code did.
    assert job.run_id is not None
    assert _RUN_ID_RE.match(job.run_id), job.run_id
    # ...and the child is told to use exactly that dir name.
    assert job.argv[job.argv.index("--run-name") + 1] == job.run_id


def test_concurrent_detection_launches_get_distinct_run_ids(
    manager: JobManager,
) -> None:
    payload = {"csv": "data/x.csv", "model": "gpt-5.2"}
    jobs = [manager.launch_detection(dict(payload)) for _ in range(25)]
    run_ids = {j.run_id for j in jobs}
    assert len(run_ids) == 25, "concurrent same-model launches must not collide"
    # Each job's argv carries its own id (no cross-wiring).
    for j in jobs:
        assert j.argv[j.argv.index("--run-name") + 1] == j.run_id


def test_relation_detection_run_id_set_at_launch(
    manager: JobManager, tmp_path: Path,
) -> None:
    discovery = tmp_path / "outputs" / "disco"
    discovery.mkdir(parents=True)
    (discovery / "relations.jsonl").write_text("")
    job = manager.launch_relation_detection(
        {"discovery_run": str(discovery), "model": "gpt-5.2"}
    )
    assert job.run_id is not None
    assert job.argv[job.argv.index("--run-name") + 1] == job.run_id


def test_non_dir_creating_kind_has_no_run_id(manager: JobManager) -> None:
    # The generic _launch path must default run_id to None for kinds that
    # operate on an existing run (report/web_verify/...) rather than minting.
    job = manager._launch("report", ["-m", "experiments.build_report", "x"])
    assert job.run_id is None
