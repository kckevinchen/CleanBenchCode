"""Web-verification endpoints + freshness reload.

The labeling app reads ``web_verifications.json`` from the run dir on
demand. We assert:
  * GET /clusters/{cid} now includes the ``web_verification`` field
  * GET /clusters/{cid}/web-verification 404s when missing, 200 when present
  * GET /web-verifications returns counts + items
  * Mtime-aware reload picks up a freshly-written file mid-session
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.labeling.app import create_app


def _build_run(tmp_path: Path) -> tuple[Path, str, Path]:
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    data_dir = project / "data"
    data_dir.mkdir()
    csv = data_dir / "contracts.csv"
    csv.write_text(
        "id,contractor\n1,AcmeCo\n2,Acme Corp\n",
        encoding="utf-8",
    )
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260601T000000Z_webverify"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text(json.dumps([
        {"cluster_id": "contracts.csv__r00000-c000", "source": "contracts.csv",
         "row": 0, "columns": ["contractor"],
         "row_values": {"id": "1", "contractor": "AcmeCo"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
        {"cluster_id": "contracts.csv__r00001-c000", "source": "contracts.csv",
         "row": 1, "columns": ["contractor"],
         "row_values": {"id": "2", "contractor": "Acme Corp"},
         "error_types": ["typo"], "error_type_categories": [],
         "evidence_count": 1, "detections": []},
    ]))
    (rd / "issues.json").write_text("[]")
    return project, run_id, rd


def _verification_record(cid: str, **overrides):
    base = {
        "cluster_id": cid,
        "source": "contracts.csv",
        "row": 0,
        "column": "contractor",
        "suspect_value": "AcmeCo",
        "entity_kind": "org",
        "verdict": "confirmed_error",
        "canonical_value": "Acme Corp",
        "confidence": 0.9,
        "evidence": [{
            "url": "https://acme.example.com",
            "title": "Acme Corp",
            "quote": "Acme Corp is the official name",
        }],
        "reasoning": "Web confirms.",
        "queries": ['"AcmeCo" official site'],
        "n_results": 3,
        "n_unique_urls": 2,
        "error": None,
        "parse_error": None,
        "llm_attempts": 1,
        "llm_endpoint": "https://stub/",
        "ts": "2026-06-01T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_get_cluster_without_verifications_file(tmp_path: Path):
    project, run_id, _ = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run_id}/clusters/contracts.csv__r00000-c000")
        assert r.status_code == 200
        assert r.json()["web_verification"] is None

        r2 = client.get(
            f"/api/runs/{run_id}/clusters/contracts.csv__r00000-c000/web-verification"
        )
        assert r2.status_code == 404


def test_get_cluster_includes_verification_when_present(tmp_path: Path):
    project, run_id, rd = _build_run(tmp_path)
    (rd / "web_verifications.json").write_text(json.dumps([
        _verification_record("contracts.csv__r00000-c000"),
    ]))
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run_id}/clusters/contracts.csv__r00000-c000")
        body = r.json()
        wv = body["web_verification"]
        assert wv is not None
        assert wv["verdict"] == "confirmed_error"
        assert wv["canonical_value"] == "Acme Corp"
        assert wv["evidence"][0]["url"] == "https://acme.example.com"

        # Dedicated endpoint matches.
        r2 = client.get(
            f"/api/runs/{run_id}/clusters/contracts.csv__r00000-c000/web-verification"
        )
        assert r2.status_code == 200
        assert r2.json()["verdict"] == "confirmed_error"


def test_list_endpoint_counts_and_items(tmp_path: Path):
    project, run_id, rd = _build_run(tmp_path)
    (rd / "web_verifications.json").write_text(json.dumps([
        _verification_record("contracts.csv__r00000-c000"),
        _verification_record("contracts.csv__r00001-c000",
                             verdict="false_positive",
                             canonical_value=None),
    ]))
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run_id}/web-verifications")
        body = r.json()
        assert body["available"] is True
        assert body["counts"]["confirmed_error"] == 1
        assert body["counts"]["false_positive"] == 1
        assert body["counts"]["ambiguous"] == 0
        assert body["counts"]["errored"] == 0
        assert len(body["items"]) == 2


def test_list_endpoint_empty_when_file_missing(tmp_path: Path):
    project, run_id, _ = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run_id}/web-verifications")
        body = r.json()
        assert body["available"] is False
        assert body["items"] == []


def test_mtime_reload_picks_up_new_writes(tmp_path: Path):
    """RunContext caches verifications keyed by file mtime. Writing the
    file mid-session must invalidate that cache so the next request
    sees fresh data."""
    project, run_id, rd = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        # 1) No file yet.
        r = client.get(f"/api/runs/{run_id}/clusters/contracts.csv__r00000-c000")
        assert r.json()["web_verification"] is None

        # 2) Write the file.
        path = rd / "web_verifications.json"
        path.write_text(json.dumps([
            _verification_record("contracts.csv__r00000-c000"),
        ]))
        # Bump mtime ahead so the reload sees a strictly newer file
        # even on filesystems with 1-second resolution.
        future = time.time() + 5
        os.utime(path, (future, future))

        # 3) Next request must reload.
        r = client.get(f"/api/runs/{run_id}/clusters/contracts.csv__r00000-c000")
        assert r.json()["web_verification"]["verdict"] == "confirmed_error"


# ---------------------------------------------------------------------------
# POST /clusters/{cid}/web-verification - on-demand single-cluster verify.
# We monkeypatch ``verify_single_cluster`` so tests don't make real HTTP
# calls to Tavily / Azure OpenAI.
# ---------------------------------------------------------------------------


def _install_fake_verify(monkeypatch, factory):
    """Replace ``experiments.run_web_verify.verify_single_cluster`` with
    ``factory(run_dir, cluster_id, **kw)``. ``factory`` may return a
    Verification dataclass or raise VerifyError."""
    import experiments.run_web_verify as rw

    def _fake(run_dir, cluster_id, **kw):
        return factory(run_dir, cluster_id, **kw)

    monkeypatch.setattr(rw, "verify_single_cluster", _fake)


def test_post_web_verification_success(tmp_path: Path, monkeypatch):
    from src.verifier.verify import Verification, write_verifications

    project, run_id, rd = _build_run(tmp_path)
    cid = "contracts.csv__r00000-c000"

    def fake(run_dir, cluster_id, **kw):
        # Mirror the real helper: write the record to disk so the
        # endpoint's ctx.verification(cid) lookup succeeds.
        rec = Verification(**_verification_record(cluster_id))
        write_verifications(run_dir / "web_verifications.json", [rec])
        return rec

    _install_fake_verify(monkeypatch, fake)

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/clusters/{cid}/web-verification")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cluster_id"] == cid
        assert body["verdict"] == "confirmed_error"
        # And subsequent GET sees it (the in-memory cache was reloaded).
        r = client.get(f"/api/runs/{run_id}/clusters/{cid}/web-verification")
        assert r.status_code == 200
        assert r.json()["verdict"] == "confirmed_error"


def test_post_web_verification_already_verified_returns_409(
    tmp_path: Path, monkeypatch,
):
    from experiments.run_web_verify import VerifyError

    project, run_id, _ = _build_run(tmp_path)
    cid = "contracts.csv__r00000-c000"

    def fake(run_dir, cluster_id, **kw):
        raise VerifyError("already_verified", "already has a verification")

    _install_fake_verify(monkeypatch, fake)

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/clusters/{cid}/web-verification")
        assert r.status_code == 409
        assert "already" in r.json()["detail"].lower()


def test_post_web_verification_gate_rejected_returns_422(
    tmp_path: Path, monkeypatch,
):
    from experiments.run_web_verify import VerifyError

    project, run_id, _ = _build_run(tmp_path)
    cid = "contracts.csv__r00000-c000"

    def fake(run_dir, cluster_id, **kw):
        raise VerifyError("gate_rejected", "non-entity column")

    _install_fake_verify(monkeypatch, fake)

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/clusters/{cid}/web-verification")
        assert r.status_code == 422
        assert "non-entity" in r.json()["detail"]


def test_post_web_verification_unknown_cluster_returns_404(tmp_path: Path):
    project, run_id, _ = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/clusters/does-not-exist/web-verification",
        )
        assert r.status_code == 404


def test_post_web_verification_unknown_run_returns_404(tmp_path: Path):
    project, _, _ = _build_run(tmp_path)
    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            "/api/runs/no-such-run/clusters/x/web-verification",
        )
        # _get_ctx (cache.get) raises 404 for unknown runs.
        assert r.status_code == 404

