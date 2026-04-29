"""Original-xlsx access endpoints: download + HTML preview.

Verifies:
* auto-detection of ``<corpus>_xlsx`` sibling directory,
* per-source ``has_xlsx`` flag in the summary + slice endpoints,
* download endpoint serves the raw xlsx with ``attachment`` headers,
* preview endpoint renders an HTML page with a tab per sheet,
* path-traversal / unknown-workbook attempts are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from src.labeling.app import create_app


def _write_xlsx(path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Inventory"
    ws.append(["sku", "qty", "price"])
    ws.append(["a", 1, 9.99])
    ws.append(["b", 2, 19.50])
    ws2 = wb.create_sheet("Notes")
    ws2.append(["author", "comment"])
    ws2.append(["alice", "<script>alert(1)</script>"])  # XSS test
    wb.save(path)


@pytest.fixture
def project_with_xlsx(tmp_path: Path):
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)

    # Curated corpus: one workbook directory with a CSV.
    corpus = project / "data" / "curated_demo"
    wb_dir = corpus / "demo.xlsx"
    wb_dir.mkdir(parents=True)
    csv_path = wb_dir / "table1.csv"
    csv_path.write_text("sku,qty,price\na,1,9.99\nb,2,19.5\n", encoding="utf-8")

    # Sibling originals dir: ``<corpus>_xlsx/<workbook>``.
    xlsx_dir = project / "data" / "curated_demo_xlsx"
    xlsx_dir.mkdir()
    _write_xlsx(xlsx_dir / "demo.xlsx")

    # Minimal run pointing at the curated CSV.
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_demo"
    run_dir = outputs / run_id
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "config": {"csv_paths": [str(csv_path)]},
    }))
    (run_dir / "clusters.json").write_text("[]")
    (run_dir / "issues.json").write_text("[]")

    app = create_app(project, annotator="tester")
    with TestClient(app) as client:
        yield client, run_id


def test_run_summary_advertises_xlsx_availability(project_with_xlsx):
    client, run_id = project_with_xlsx
    r = client.get(f"/api/runs/{run_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["xlsx_available"] is True
    assert body["xlsx_source_dir"].endswith("curated_demo_xlsx")
    assert body["sources"][0]["table_group"] == "demo.xlsx"
    assert body["sources"][0]["has_xlsx"] is True


def test_source_slice_advertises_has_xlsx(project_with_xlsx):
    client, run_id = project_with_xlsx
    r = client.get(f"/api/runs/{run_id}/sources/table1.csv")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["workbook"] == "demo.xlsx"
    assert body["has_xlsx"] is True


def test_download_xlsx_returns_attachment(project_with_xlsx):
    client, run_id = project_with_xlsx
    r = client.get(f"/api/runs/{run_id}/workbooks/demo.xlsx/xlsx")
    assert r.status_code == 200
    # File magic for xlsx (zip): PK\x03\x04
    assert r.content[:4] == b"PK\x03\x04"
    cd = r.headers.get("content-disposition", "")
    assert "attachment" in cd.lower()
    assert "demo.xlsx" in cd


def test_preview_renders_sheets_and_escapes(project_with_xlsx):
    client, run_id = project_with_xlsx
    r = client.get(f"/api/runs/{run_id}/workbooks/demo.xlsx/preview")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    html = r.text
    # Both sheet names rendered as tabs.
    assert ">Inventory<" in html
    assert ">Notes<" in html
    # Cell values present.
    assert ">sku<" in html and ">qty<" in html
    # The XSS payload is HTML-escaped, never raw.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    # Download link is offered in the header.
    assert f"/api/runs/{run_id}/workbooks/demo.xlsx/xlsx" in html


def test_unknown_workbook_returns_404(project_with_xlsx):
    client, run_id = project_with_xlsx
    r = client.get(f"/api/runs/{run_id}/workbooks/other.xlsx/xlsx")
    assert r.status_code == 404
    r2 = client.get(f"/api/runs/{run_id}/workbooks/other.xlsx/preview")
    assert r2.status_code == 404


def test_path_traversal_workbook_rejected(project_with_xlsx):
    client, run_id = project_with_xlsx
    # Embedded backslash. URL-normalised paths with ``..`` collapse on
    # the client, so we focus on payloads that survive normalisation
    # but should still be rejected by the workbook validator.
    r = client.get(f"/api/runs/{run_id}/workbooks/foo%5Cbar/xlsx")
    assert r.status_code == 404


def test_preview_without_xlsx_dir_returns_404(tmp_path: Path):
    """If no sibling _xlsx dir exists, has_xlsx=False and download/preview 404."""
    project = tmp_path / "ws"
    project.mkdir()
    (project / "frontend" / "labeling").mkdir(parents=True)
    corpus = project / "data" / "curated_lonely" / "wb.xlsx"
    corpus.mkdir(parents=True)
    csv = corpus / "t.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    outputs = project / "outputs"
    outputs.mkdir()
    run_id = "20260101T000000Z_lonely"
    rd = outputs / run_id
    rd.mkdir()
    (rd / "summary.json").write_text(json.dumps({
        "config": {"csv_paths": [str(csv)]},
    }))
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")

    app = create_app(project, annotator="t")
    with TestClient(app) as client:
        body = client.get(f"/api/runs/{run_id}").json()
        assert body["xlsx_available"] is False
        assert body["xlsx_source_dir"] is None
        assert body["sources"][0]["has_xlsx"] is False
        assert client.get(
            f"/api/runs/{run_id}/workbooks/wb.xlsx/xlsx"
        ).status_code == 404
