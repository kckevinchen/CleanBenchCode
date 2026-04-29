"""Regression tests for labeling-server path-traversal defences.

Pinned by code-review Pass 5a HIGH#1 + #2:

* ``_safe_run_dir`` (and ``_validate_run_id``) refuse run_ids that
  escape ``outputs_dir`` via ``..``, separator injection, NUL bytes,
  or symlinks redirecting the resolved run dir outside the workspace.
* ``_validate_source_name`` refuses path-shaped source names (``..``,
  ``/``, ``\\``, NUL).

Without these checks the unauthenticated DELETE /api/runs/<id>
endpoint could ``shutil.rmtree`` arbitrary directories, and CSV
fetch endpoints could leak files outside the run dir.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from src.labeling.app import create_app


def _make_minimal_run(project: Path, run_id: str = "20260101T000000Z_test") -> Path:
    """Create a minimal valid run dir so non-traversal paths return 200/204
    (so we can be sure 4xx responses to evil paths are NOT a generic 404
    that would happen anyway)."""
    data_dir = project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "a.csv").write_text("name,email\nalice,a@x\nbob,b@x\n", encoding="utf-8")
    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    run_dir = outputs / run_id
    run_dir.mkdir()
    summary = {"config": {"csv_paths": [str(data_dir / "a.csv")], "sources": ["a.csv"]}}
    (run_dir / "summary.json").write_text(json.dumps(summary))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")
    return run_dir


@pytest.fixture
def client_project(tmp_path: Path):
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    run_dir = _make_minimal_run(project)
    app = create_app(project, annotator="tester")
    with TestClient(app) as client:
        yield client, project, run_dir.name


# ---------------------------------------------------------------------------
# DELETE /api/runs/<run_id>  — must never escape outputs/
# ---------------------------------------------------------------------------


class TestDeleteRunTraversal:
    # Note: httpx + Starlette URL-normalise ``..`` / ``.`` segments
    # client-side, so by the time the request hits the validator those
    # have already collapsed. We focus on payloads that SURVIVE
    # normalisation (backslashes, NUL, embedded ``..`` in a segment),
    # plus the critical symlink-redirect path. Acceptable rejection
    # codes are 400 (validator), 404 (route mismatch / not found), or
    # 405 (URL normalised to a route without that method).
    @pytest.mark.parametrize("evil", [
        "..",
        ".",
        "../etc",
        "../../etc/passwd",
        "foo/bar",         # literal slash (TestClient won't encode)
        "foo\\bar",        # backslash
        "a\x00b",          # NUL byte
    ])
    def test_invalid_run_ids_blocked(self, client_project, evil: str) -> None:
        client, project, _good = client_project
        r = client.delete(f"/api/runs/{quote(evil, safe='')}")
        assert r.status_code in (400, 404, 405), (evil, r.status_code, r.text)
        # Critical invariant: the workspace and any file outside outputs/
        # are intact regardless of what the response code happened to be.
        assert (project / "outputs").is_dir()

    def test_legitimate_delete_still_works(self, client_project) -> None:
        client, project, run_id = client_project
        r = client.delete(f"/api/runs/{run_id}")
        assert r.status_code == 200, r.text
        assert not (project / "outputs" / run_id).exists()

    def test_symlink_redirected_run_rejected(self, client_project, tmp_path: Path) -> None:
        client, project, _good = client_project
        # Place an attacker target OUTSIDE the workspace.
        outside = tmp_path / "_secret"
        outside.mkdir()
        (outside / "important.txt").write_text("do not delete")
        # Drop a symlink inside outputs/ pointing at it.
        link_name = "linked_run"
        link = project / "outputs" / link_name
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported on this platform")

        r = client.delete(f"/api/runs/{link_name}")
        # Either 400 (validator caught it) or 404 (parent != outputs_root
        # after resolution). Critical: the symlink target must survive.
        assert r.status_code in (400, 404), r.text
        assert (outside / "important.txt").exists()
        assert (outside / "important.txt").read_text() == "do not delete"


# ---------------------------------------------------------------------------
# GET /api/runs/<run_id>/sources/<name>  — name validator
# ---------------------------------------------------------------------------


class TestSourceNameValidator:
    # Same caveat as above: httpx normalises bare ``..`` / ``.``. The
    # values below preserve the path-shape past normalisation so the
    # validator is the layer doing the rejection.
    @pytest.mark.parametrize("evil", [
        "foo\\bar",     # backslash separator
        "a\x00b",       # NUL byte
        "foo/bar",      # literal slash splits the URL but routes elsewhere
    ])
    def test_path_shaped_source_names_rejected(self, client_project, evil: str) -> None:
        client, _project, run_id = client_project
        r = client.get(f"/api/runs/{run_id}/sources/{quote(evil, safe='')}")
        # 400 from validator OR 404 from route mismatch — both prevent
        # filesystem access outside the run dir.
        assert r.status_code in (400, 404), (evil, r.status_code, r.text)


# ---------------------------------------------------------------------------
# GET /api/runs/<run_id>  — readonly endpoint must also be safe
# ---------------------------------------------------------------------------


class TestGetRunTraversal:
    @pytest.mark.parametrize("evil", [
        "foo\\bar",
        "a\x00b",
        "foo/bar",
    ])
    def test_invalid_run_ids_blocked(self, client_project, evil: str) -> None:
        client, _project, _good = client_project
        r = client.get(f"/api/runs/{quote(evil, safe='')}")
        assert r.status_code in (400, 404, 405), (evil, r.status_code, r.text)
