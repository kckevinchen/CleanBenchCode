"""Virtual run-folders: create / assign / rename / collapse / delete.

Folders group runs in the dashboard WITHOUT moving any run directory on
disk -- membership is a ``folder_id`` on each run's ``run_metadata.json``
sidecar, and folder definitions live in ``outputs/folders.json``. These
tests cover the REST surface in ``src/labeling/routers/runs.py`` plus the
guarantees that make folders safe: deleting a folder orphans its members
back to "Unfiled" (never deletes runs), and a path-traversal run-id is
rejected rather than written.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app


def _build_project(tmp_path: Path, run_ids: list[str]) -> Path:
    project = tmp_path / "fld"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "t.csv"
    csv.write_text("id,name\n1,Acme\n", encoding="utf-8")
    outputs = project / "outputs"
    outputs.mkdir()
    for rid in run_ids:
        rd = outputs / rid
        rd.mkdir()
        (rd / "summary.json").write_text(
            json.dumps({"config": {"csv_paths": [str(csv)]}})
        )
        (rd / "clusters.json").write_text("[]")
    return project


def _folder_id_of(client: TestClient, run_id: str) -> str:
    runs = client.get("/api/runs").json()["runs"]
    return next(r["folder_id"] for r in runs if r["run_id"] == run_id)


def test_runs_endpoint_exposes_empty_folders(tmp_path: Path) -> None:
    project = _build_project(tmp_path, ["20260101T000000Z_a"])
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        body = client.get("/api/runs").json()
        assert body["folders"] == []
        assert body["runs"][0]["folder_id"] == ""


def test_create_empty_folder(tmp_path: Path) -> None:
    project = _build_project(tmp_path, ["20260101T000000Z_a"])
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post("/api/folders", json={"name": "Experiments"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["moved"] == 0
        fid = body["folder"]["id"]
        assert fid and body["folder"]["name"] == "Experiments"
        assert body["folder"]["collapsed"] is False
        # Surfaced via both endpoints.
        assert [f["id"] for f in client.get("/api/folders").json()["folders"]] == [fid]
        assert [f["id"] for f in client.get("/api/runs").json()["folders"]] == [fid]
        # On disk.
        stored = json.loads((project / "outputs" / "folders.json").read_text())
        assert stored["folders"][0]["id"] == fid


def test_create_folder_with_members(tmp_path: Path) -> None:
    ids = ["20260101T000000Z_a", "20260102T000000Z_b", "20260103T000000Z_c"]
    project = _build_project(tmp_path, ids)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            "/api/folders", json={"name": "Batch", "run_ids": ids[:2]}
        )
        assert r.status_code == 200, r.text
        fid = r.json()["folder"]["id"]
        assert r.json()["moved"] == 2
        assert _folder_id_of(client, ids[0]) == fid
        assert _folder_id_of(client, ids[1]) == fid
        assert _folder_id_of(client, ids[2]) == ""   # untouched


def test_assign_and_unfile(tmp_path: Path) -> None:
    ids = ["20260101T000000Z_a", "20260102T000000Z_b"]
    project = _build_project(tmp_path, ids)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        fid = client.post("/api/folders", json={"name": "F"}).json()["folder"]["id"]
        # Move both in.
        r = client.post(
            "/api/folders/assign", json={"run_ids": ids, "folder_id": fid}
        )
        assert r.status_code == 200 and r.json()["moved"] == 2
        assert _folder_id_of(client, ids[0]) == fid
        # Unfile one via null.
        r = client.post(
            "/api/folders/assign", json={"run_ids": [ids[0]], "folder_id": None}
        )
        assert r.status_code == 200 and r.json()["moved"] == 1
        assert _folder_id_of(client, ids[0]) == ""
        assert _folder_id_of(client, ids[1]) == fid


def test_assign_to_unknown_folder_404(tmp_path: Path) -> None:
    ids = ["20260101T000000Z_a"]
    project = _build_project(tmp_path, ids)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            "/api/folders/assign", json={"run_ids": ids, "folder_id": "deadbeef"}
        )
        assert r.status_code == 404
        # The run stays unfiled - nothing was written.
        assert _folder_id_of(client, ids[0]) == ""


def test_rename_and_collapse(tmp_path: Path) -> None:
    project = _build_project(tmp_path, ["20260101T000000Z_a"])
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        fid = client.post("/api/folders", json={"name": "Old"}).json()["folder"]["id"]
        r = client.patch(f"/api/folders/{fid}", json={"name": "New", "collapsed": True})
        assert r.status_code == 200, r.text
        assert r.json()["folder"]["name"] == "New"
        assert r.json()["folder"]["collapsed"] is True
        # Persisted.
        f = client.get("/api/folders").json()["folders"][0]
        assert f["name"] == "New" and f["collapsed"] is True


def test_patch_missing_folder_404(tmp_path: Path) -> None:
    project = _build_project(tmp_path, ["20260101T000000Z_a"])
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        assert client.patch("/api/folders/nope", json={"name": "x"}).status_code == 404


def test_delete_folder_orphans_members_to_unfiled(tmp_path: Path) -> None:
    ids = ["20260101T000000Z_a", "20260102T000000Z_b"]
    project = _build_project(tmp_path, ids)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        fid = client.post(
            "/api/folders", json={"name": "F", "run_ids": ids}
        ).json()["folder"]["id"]
        r = client.delete(f"/api/folders/{fid}")
        assert r.status_code == 200, r.text
        assert r.json()["unfiled"] == 2
        # Folder gone; runs survive and are unfiled.
        assert client.get("/api/folders").json()["folders"] == []
        for rid in ids:
            assert (project / "outputs" / rid).is_dir()
            assert _folder_id_of(client, rid) == ""


def test_delete_missing_folder_404(tmp_path: Path) -> None:
    project = _build_project(tmp_path, ["20260101T000000Z_a"])
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        assert client.delete("/api/folders/nope").status_code == 404


def test_assign_rejects_path_traversal(tmp_path: Path) -> None:
    project = _build_project(tmp_path, ["20260101T000000Z_a"])
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        fid = client.post("/api/folders", json={"name": "F"}).json()["folder"]["id"]
        r = client.post(
            "/api/folders/assign",
            json={"run_ids": ["../../etc"], "folder_id": fid},
        )
        # safe_run_dir rejects the traversal id with a 400.
        assert r.status_code == 400


def test_assign_skips_nonexistent_run(tmp_path: Path) -> None:
    project = _build_project(tmp_path, ["20260101T000000Z_a"])
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        fid = client.post("/api/folders", json={"name": "F"}).json()["folder"]["id"]
        r = client.post(
            "/api/folders/assign",
            json={"run_ids": ["20990101T000000Z_ghost"], "folder_id": fid},
        )
        # Valid syntactically, but no such run dir -> skipped, not an error.
        assert r.status_code == 200 and r.json()["moved"] == 0


def test_assign_empty_run_ids_is_noop(tmp_path: Path) -> None:
    project = _build_project(tmp_path, ["20260101T000000Z_a"])
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        fid = client.post("/api/folders", json={"name": "F"}).json()["folder"]["id"]
        r = client.post(
            "/api/folders/assign", json={"run_ids": [], "folder_id": fid}
        )
        assert r.status_code == 200 and r.json()["moved"] == 0


def test_assign_batch_is_atomic_on_bad_id(tmp_path: Path) -> None:
    """A traversal id anywhere in the batch must abort with 400 WITHOUT
    half-applying the move to the valid ids that precede it."""
    ids = ["20260101T000000Z_a", "20260102T000000Z_b"]
    project = _build_project(tmp_path, ids)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        fid = client.post("/api/folders", json={"name": "F"}).json()["folder"]["id"]
        # Valid id FIRST, traversal id second -> the old code wrote the
        # first run before raising. Now nothing should be written.
        r = client.post(
            "/api/folders/assign",
            json={"run_ids": [ids[0], "../../etc"], "folder_id": fid},
        )
        assert r.status_code == 400
        assert _folder_id_of(client, ids[0]) == ""   # NOT half-applied
        assert _folder_id_of(client, ids[1]) == ""


def test_create_folder_bad_run_id_leaves_no_orphan(tmp_path: Path) -> None:
    """A bad run_id in the create payload must 400 BEFORE the folder is
    persisted, so a rejected request leaves no phantom empty folder."""
    ids = ["20260101T000000Z_a"]
    project = _build_project(tmp_path, ids)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            "/api/folders",
            json={"name": "Oops", "run_ids": [ids[0], "../../etc"]},
        )
        assert r.status_code == 400
        # No folder was created, and the valid run stays unfiled.
        assert client.get("/api/folders").json()["folders"] == []
        assert not (project / "outputs" / "folders.json").exists()
        assert _folder_id_of(client, ids[0]) == ""


def test_duplicate_clone_starts_unfiled(tmp_path: Path) -> None:
    """Duplicating a filed run yields a clone in "Unfiled" (membership is
    NOT inherited), matching the starred=False reset."""
    ids = ["20260101T000000Z_a"]
    project = _build_project(tmp_path, ids)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        fid = client.post(
            "/api/folders", json={"name": "F", "run_ids": ids}
        ).json()["folder"]["id"]
        assert _folder_id_of(client, ids[0]) == fid
        dup = client.post(f"/api/runs/{ids[0]}/duplicate", json={})
        assert dup.status_code == 200, dup.text
        new_id = dup.json()["run_id"]
        assert _folder_id_of(client, new_id) == ""        # clone is unfiled
        assert _folder_id_of(client, ids[0]) == fid       # source untouched


def test_corrupt_folders_json_is_tolerated(tmp_path: Path) -> None:
    """A malformed/garbage folders.json must degrade to "no folders"
    rather than 500 the whole runs dashboard (GET /api/runs reads it)."""
    project = _build_project(tmp_path, ["20260101T000000Z_a"])
    folders_path = project / "outputs" / "folders.json"
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # 1) Invalid JSON.
        folders_path.write_text("{not valid json", encoding="utf-8")
        assert client.get("/api/runs").json()["folders"] == []
        assert client.get("/api/folders").json()["folders"] == []
        # 2) Right shape, but garbage + a record missing 'id' + a
        #    duplicate id -> bad records dropped, dupes deduped.
        folders_path.write_text(
            json.dumps({"folders": [
                "not a dict",
                {"name": "no id"},
                {"id": "dup", "name": "first"},
                {"id": "dup", "name": "second"},
            ]}),
            encoding="utf-8",
        )
        folders = client.get("/api/runs").json()["folders"]
        assert [f["id"] for f in folders] == ["dup"]
        assert folders[0]["name"] == "first"
        # Dashboard still serves 200.
        assert client.get("/api/runs").status_code == 200
