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
from typing import Any, Dict

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
# We monkeypatch ``verify_single_cluster`` so tests don't make real
# Foundry Agents calls.
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


def test_post_web_verification_handles_person_clusters_by_default(
    tmp_path: Path, monkeypatch,
):
    """Person-name verification is ON by default in the gate, so the
    UI's POST handler does NOT need to pass ``include_person=True``
    explicitly. A First-Name cluster should reach the verifier
    helper and produce a record. Regression test for the original
    UI bug where person-name clusters were 422-ed because the
    handler didn't override the (now-removed) PII default."""
    from src.verifier.verify import Verification, write_verifications

    project, run_id, rd = _build_run(tmp_path)
    cid = "contracts.csv__r00000-c000"
    captured: Dict[str, Any] = {}

    def fake(run_dir, cluster_id, **kw):
        captured.update(kw)
        rec = Verification(**_verification_record(cluster_id))
        write_verifications(run_dir / "web_verifications.json", [rec])
        return rec

    _install_fake_verify(monkeypatch, fake)

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/clusters/{cid}/web-verification")
        assert r.status_code == 200, r.text
    # The handler doesn't need to pass include_person any more --
    # the verifier helper defaults to True. We just check the call
    # reached it (the helper would 422 on a person cluster if the
    # default were wrong, but in this stub it always succeeds; the
    # important contract is the helper IS called from the handler).
    assert "retry_errored" in captured, (
        "UI POST handler must reach verify_single_cluster"
    )


# ---------------------------------------------------------------------------
# Retry-errored: the UI must be able to re-run a verification that
# previously failed (e.g. Azure agent timed out / threw a DecodeError),
# without manual intervention. The endpoint accepts ``?retry=true`` and
# the verifier helper exposes ``retry_errored=`` for the same purpose.
# ---------------------------------------------------------------------------


def test_post_web_verification_retry_replaces_errored_record(
    tmp_path: Path, monkeypatch,
):
    """When the prior record's ``error`` is set, ``?retry=true`` must
    re-run and overwrite the errored record on disk; the response is
    the fresh record."""
    from src.verifier.verify import Verification, write_verifications

    project, run_id, rd = _build_run(tmp_path)
    cid = "contracts.csv__r00000-c000"

    # Seed an errored prior verification.
    errored_rec = Verification(**_verification_record(
        cid, verdict="errored", confidence=0.0,
        evidence=[], canonical_value=None,
        error="agent timed out",
    ))
    write_verifications(rd / "web_verifications.json", [errored_rec])

    captured: Dict[str, Any] = {}

    def fake(run_dir, cluster_id, **kw):
        captured.update(kw)
        # Real helper writes the fresh record itself; mirror that.
        from src.verifier.verify import load_verifications as _load
        prior = _load(run_dir / "web_verifications.json")
        # Drop the errored one (matches helper's overwrite semantics).
        prior = [v for v in prior if v.cluster_id != cluster_id]
        new = Verification(**_verification_record(
            cluster_id, verdict="confirmed_error", confidence=0.9,
            error=None,
        ))
        prior.append(new)
        write_verifications(run_dir / "web_verifications.json", prior)
        return new

    _install_fake_verify(monkeypatch, fake)

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/clusters/{cid}/web-verification?retry=true",
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["verdict"] == "confirmed_error"
        assert body["error"] is None

    # The retry flag MUST flow through to the verifier helper.
    assert captured.get("retry_errored") is True

    # Disk now has exactly the fresh record - the errored one is gone.
    persisted = json.loads(
        (rd / "web_verifications.json").read_text(encoding="utf-8"),
    )
    assert len(persisted) == 1
    assert persisted[0]["cluster_id"] == cid
    assert persisted[0]["verdict"] == "confirmed_error"
    assert persisted[0]["error"] is None


def test_post_web_verification_default_does_not_retry(
    tmp_path: Path, monkeypatch,
):
    """Sanity: omitting ``?retry`` keeps the default behavior - the
    verifier helper receives ``retry_errored=False``."""
    from src.verifier.verify import Verification, write_verifications

    project, run_id, _ = _build_run(tmp_path)
    cid = "contracts.csv__r00000-c000"
    captured: Dict[str, Any] = {}

    def fake(run_dir, cluster_id, **kw):
        captured.update(kw)
        rec = Verification(**_verification_record(cluster_id))
        write_verifications(run_dir / "web_verifications.json", [rec])
        return rec

    _install_fake_verify(monkeypatch, fake)

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run_id}/clusters/{cid}/web-verification")
        assert r.status_code == 200

    assert captured.get("retry_errored") is False


def test_post_web_verification_retry_propagates_409_for_non_errored(
    tmp_path: Path, monkeypatch,
):
    """Even with ``?retry=true``, a non-errored prior record must NOT
    be overwritten - the helper raises ``already_verified`` and the
    handler surfaces a 409. The flag is meant for *recovery only*."""
    from experiments.run_web_verify import VerifyError

    project, run_id, _ = _build_run(tmp_path)
    cid = "contracts.csv__r00000-c000"

    def fake(run_dir, cluster_id, **kw):
        # The real helper enforces this guard - we mirror it here so
        # the test pins the contract from the endpoint's perspective.
        if not kw.get("retry_errored"):
            raise VerifyError("already_verified", "non-errored record exists")
        # When the prior record is non-errored, even retry is refused.
        # We simulate this by raising regardless when called.
        raise VerifyError("already_verified", "non-errored record exists")

    _install_fake_verify(monkeypatch, fake)

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run_id}/clusters/{cid}/web-verification?retry=true",
        )
        assert r.status_code == 409
        assert "already" in r.json()["detail"].lower() or \
               "non-errored" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Helper-level guards (verify_single_cluster) -- regression-tested at
# the unit level so that bugs in the helper's prior-record logic don't
# need a TestClient round-trip to surface.
# ---------------------------------------------------------------------------


def test_verify_single_cluster_retry_errored_replaces_existing(
    tmp_path: Path, monkeypatch,
):
    """End-to-end (no agent): ``retry_errored=True`` makes the helper
    skip the ``already_verified`` raise when the prior record's
    ``error`` is set, run the verifier, and write the fresh record
    in place of the errored one."""
    import experiments.run_web_verify as rw
    from src.verifier.verify import Verification, write_verifications

    project, run_id, rd = _build_run(tmp_path)
    cid = "contracts.csv__r00000-c000"

    # Seed an errored prior.
    errored = Verification(**_verification_record(
        cid, verdict="errored", confidence=0.0,
        evidence=[], canonical_value=None, error="agent timed out",
    ))
    write_verifications(rd / "web_verifications.json", [errored])

    # Stub Verifier.run() so we don't hit Azure. We patch the actual
    # symbol the helper imports inside its function body.
    fresh = Verification(**_verification_record(
        cid, verdict="confirmed_error", confidence=0.85,
        error=None,
    ))

    class _FakeVerifier:
        def __init__(self, *a, **kw):
            pass
        def run(self, clusters):
            return [fresh]

    monkeypatch.setattr(rw, "Verifier", _FakeVerifier)
    # Avoid making real Azure calls in __post_init__.
    monkeypatch.setattr(rw, "AzureAgentBackend", lambda **kw: object())

    out = rw.verify_single_cluster(rd, cid, retry_errored=True)

    assert out.verdict == "confirmed_error"
    persisted = json.loads(
        (rd / "web_verifications.json").read_text(encoding="utf-8"),
    )
    assert len(persisted) == 1
    assert persisted[0]["verdict"] == "confirmed_error"
    assert persisted[0]["error"] is None


def test_verify_single_cluster_refuses_retry_on_non_errored(
    tmp_path: Path, monkeypatch,
):
    """Sanity: a non-errored prior record is NOT replaced by a retry.
    The helper raises ``already_verified``."""
    import experiments.run_web_verify as rw
    from src.verifier.verify import Verification, write_verifications

    project, run_id, rd = _build_run(tmp_path)
    cid = "contracts.csv__r00000-c000"

    good = Verification(**_verification_record(cid, error=None))
    write_verifications(rd / "web_verifications.json", [good])

    with __import__("pytest").raises(rw.VerifyError) as ei:
        rw.verify_single_cluster(rd, cid, retry_errored=True)
    assert ei.value.code == "already_verified"

