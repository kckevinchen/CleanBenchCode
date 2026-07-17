"""Regression tests: relation-only detection launches reroute to the
relation-discovery pipeline.

Selecting only multi-column relation types (e.g. ``fd_violation``) in the
detection launcher used to spawn ``run_cleaning_detection``, which routes
every such type to the relation pipeline and then has nothing left to
scan -- producing an empty run. The launcher now detects this and hands
the job to ``run_relation_discovery`` (which does discover -> validate ->
detect in one invocation), so the run actually produces findings.

These tests pin:

1. ``_is_relation_only_detection`` -- the routing predicate (hermetic
   taxonomy + config so it is deterministic regardless of the live YAML).
2. ``JobManager.launch_detection`` -- end-to-end reroute to ``kind=
   "relations"`` for relation-only selections, and *no* reroute for
   single-column / mixed / dedicated-handler selections.
3. Drift guards against the real taxonomy + the cleaning runner's
   ``DEDICATED_HANDLER_TYPES``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling import runs as runs_mod  # noqa: E402
from src.labeling.runs import (  # noqa: E402
    JobManager,
    _DEDICATED_HANDLER_TYPES,
    _is_relation_only_detection,
)


# ---------------------------------------------------------------------------
# Hermetic mini-project (deterministic taxonomy + config)
# ---------------------------------------------------------------------------


_TAXONOMY_YAML = """\
types:
  - key: typo
    name: Typo
    category: "Single-Column / Typo"
  - key: numeric_outlier
    name: Numeric Outlier
    category: "Single-Column / Distribution"
  - key: fd_violation
    name: FD Violation
    category: "Multi-Column / Intra-Record"
  - key: row_calculation_inconsistency
    name: Row Calculation
    category: "Multi-Column / Intra-Record"
  - key: string_calculation_inconsistency
    name: String Calculation
    category: "Multi-Column / Intra-Record"
  - key: row_duplication
    name: Row Duplication
    category: "Multi-Row / Inter-Record"
  - key: inter_table_inconsistency
    name: Inter-Table
    category: "Multi-Table / Relational"
"""


def _make_project(tmp_path: Path, *, inter_column_enabled: bool = True) -> Path:
    """Lay down a minimal project tree the routing helpers can read."""
    (tmp_path / "prompts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "prompts" / "error_types.yaml").write_text(
        _TAXONOMY_YAML, encoding="utf-8"
    )
    (tmp_path / "config" / "experiment.yaml").write_text(
        f"inter_column_pipeline:\n  enabled: {str(inter_column_enabled).lower()}\n",
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------


class TestRelationOnlyPredicate:
    def test_single_relation_type_routes(self, tmp_path: Path) -> None:
        root = _make_project(tmp_path)
        assert _is_relation_only_detection({"error_types": "fd_violation"}, root)

    def test_all_relation_types_route(self, tmp_path: Path) -> None:
        root = _make_project(tmp_path)
        payload = {
            "error_types": (
                "fd_violation,row_calculation_inconsistency,"
                "string_calculation_inconsistency"
            )
        }
        assert _is_relation_only_detection(payload, root)

    def test_single_column_type_does_not_route(self, tmp_path: Path) -> None:
        root = _make_project(tmp_path)
        assert not _is_relation_only_detection({"error_types": "typo"}, root)

    def test_mixed_selection_does_not_route(self, tmp_path: Path) -> None:
        # A single-column type in the mix gives the per-chunk detector
        # real work, so detection must stay detection.
        root = _make_project(tmp_path)
        assert not _is_relation_only_detection(
            {"error_types": "typo,fd_violation"}, root
        )

    def test_dedicated_handler_types_do_not_route(self, tmp_path: Path) -> None:
        # row_duplication / inter_table_inconsistency are detected
        # in-process by the cleaning runner, so they stay on the
        # detection path even though they are non-single-column.
        root = _make_project(tmp_path)
        assert not _is_relation_only_detection(
            {"error_types": "row_duplication"}, root
        )
        assert not _is_relation_only_detection(
            {"error_types": "inter_table_inconsistency"}, root
        )
        assert not _is_relation_only_detection(
            {"error_types": "fd_violation,row_duplication"}, root
        )

    def test_empty_or_default_selection_does_not_route(self, tmp_path: Path) -> None:
        # No error_types -> full taxonomy (includes single-column work).
        root = _make_project(tmp_path)
        assert not _is_relation_only_detection({}, root)
        assert not _is_relation_only_detection({"error_types": ""}, root)
        assert not _is_relation_only_detection({"error_types": None}, root)

    def test_disabled_inter_column_pipeline_does_not_route(
        self, tmp_path: Path
    ) -> None:
        # With the pipeline off the per-chunk detector owns these types,
        # so detection is the correct path even for relation-only sets.
        root = _make_project(tmp_path, inter_column_enabled=False)
        assert not _is_relation_only_detection({"error_types": "fd_violation"}, root)


# ---------------------------------------------------------------------------
# JobManager end-to-end reroute
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
    _make_project(tmp_path)
    return JobManager(tmp_path, log_dir=tmp_path / "logs" / "jobs")


def _argv_str(job) -> str:
    return " ".join(job.argv)


class TestRelationOnlyRouting:
    def test_fd_violation_only_reroutes_to_relations(
        self, manager: JobManager
    ) -> None:
        job = manager.launch_detection(
            {"csv": "data/x.csv", "model": "gpt-5.2", "error_types": "fd_violation"}
        )
        assert job.kind == "relations"
        argv = _argv_str(job)
        assert "experiments.run_relation_discovery" in argv
        assert "run_cleaning_detection" not in argv
        # The selected type is forwarded so discovery scans only it.
        assert job.argv[job.argv.index("--error-types") + 1] == "fd_violation"

    def test_single_column_stays_detection(self, manager: JobManager) -> None:
        job = manager.launch_detection(
            {"csv": "data/x.csv", "model": "gpt-5.2", "error_types": "typo"}
        )
        assert job.kind == "detection"
        assert "experiments.run_cleaning_detection" in _argv_str(job)

    def test_mixed_stays_detection(self, manager: JobManager) -> None:
        job = manager.launch_detection(
            {
                "csv": "data/x.csv",
                "model": "gpt-5.2",
                "error_types": "typo,fd_violation",
            }
        )
        assert job.kind == "detection"
        assert "experiments.run_cleaning_detection" in _argv_str(job)

    def test_default_selection_stays_detection(self, manager: JobManager) -> None:
        job = manager.launch_detection({"csv": "data/x.csv", "model": "gpt-5.2"})
        assert job.kind == "detection"


# ---------------------------------------------------------------------------
# Drift guards (real taxonomy + cleaning runner constants)
# ---------------------------------------------------------------------------


class TestNoDrift:
    def test_dedicated_handler_set_matches_cleaning_runner(self) -> None:
        from experiments.run_cleaning_detection import DEDICATED_HANDLER_TYPES

        assert _DEDICATED_HANDLER_TYPES == DEDICATED_HANDLER_TYPES

    def test_real_taxonomy_scopes_unchanged(self) -> None:
        # Guard the live taxonomy against scope/category drift, decoupled
        # from the config flag: the three Multi-Column intra-record types
        # must be non-single-column and lack a dedicated handler (so they
        # route); typo must stay single-column; and the dedicated types
        # must be non-single-column (otherwise excluding them is moot).
        from src.labeling.runs import _scope_prefix, load_taxonomy_keys

        category_by_key = {
            str(t.get("key")): str(t.get("category") or "")
            for t in load_taxonomy_keys(PROJECT_ROOT)
        }
        for relation_type in (
            "fd_violation",
            "row_calculation_inconsistency",
            "string_calculation_inconsistency",
        ):
            assert relation_type in category_by_key, relation_type
            assert _scope_prefix(category_by_key[relation_type]) != "single-column"
            assert relation_type not in _DEDICATED_HANDLER_TYPES
        assert _scope_prefix(category_by_key["typo"]) == "single-column"
        for dedicated in ("row_duplication", "inter_table_inconsistency"):
            assert _scope_prefix(category_by_key[dedicated]) != "single-column"
            assert dedicated in _DEDICATED_HANDLER_TYPES
