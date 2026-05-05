"""Render screenshots referenced by docs/progress_update.tex.

Outputs PNGs into ``docs/`` next to the ``.tex`` file:

  - pipeline.png                  (polished diagram from
                                   docs/pipeline_diagram.html)
  - pipeline_dense.png            (the #flow mermaid section of
                                   docs/pipeline.html, cropped)
  - screenshot_runs_dashboard.png (workbench /ui/ runs tab)
  - screenshot_launch_form.png    (workbench /ui/ launch tab)
  - screenshot_labeling_view.png  (workbench /ui/label.html?run=...)
  - screenshot_html_report.png    (a run's report.html)

Requires the workbench server to be running on ``--server``
(default http://localhost:8765) for the four UI shots.
The pipeline and report shots are loaded via ``file://``
and do not need the server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
OUTPUTS_DIR = REPO_ROOT / "outputs"

VIEWPORT = {"width": 1440, "height": 900}


def pick_run(server: str, *, want: str = "report") -> str:
    """Pick a run from /api/runs.

    ``want="report"``   -> prefer runs that have ``report.html``.
    ``want="clusters"`` -> prefer runs with ``clusters > 0``.
    Falls back to the most recent run if no preferred match exists.
    """
    with urllib.request.urlopen(f"{server}/api/runs", timeout=5) as resp:
        data = json.load(resp)
    runs = data.get("runs", [])
    if want == "clusters":
        for r in runs:
            if (r.get("clusters") or 0) > 0:
                return r["run_id"]
    elif want == "report":
        for r in runs:
            if r.get("has_report"):
                return r["run_id"]
    if runs:
        return runs[0]["run_id"]
    raise SystemExit("No runs found via /api/runs")


def find_report_html(run_id: str) -> Path:
    p = OUTPUTS_DIR / run_id / "report.html"
    if not p.exists():
        raise SystemExit(f"report.html missing: {p}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8765")
    ap.add_argument(
        "--run-id",
        default=None,
        help="Run id for the report screenshot (default: auto-pick "
        "a run with report.html).",
    )
    ap.add_argument(
        "--label-run-id",
        default=None,
        help="Run id for the labelling-view screenshot (default: "
        "auto-pick a run with clusters>0, falling back to --run-id).",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DOCS_DIR),
        help="Where to write PNGs (default: docs/).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_run_id = args.run_id or pick_run(args.server, want="report")
    label_run_id = (
        args.label_run_id
        or pick_run(args.server, want="clusters")
        or report_run_id
    )
    print(f"[screenshots] report run_id = {report_run_id}")
    print(f"[screenshots] label  run_id = {label_run_id}")

    flowchart_html = DOCS_DIR / "pipeline_diagram.html"
    if not flowchart_html.exists():
        raise SystemExit(f"missing {flowchart_html}")
    dense_html = DOCS_DIR / "pipeline.html"
    if not dense_html.exists():
        raise SystemExit(f"missing {dense_html}")
    report_html = find_report_html(report_run_id)

    from playwright.sync_api import sync_playwright

    targets = [
        # (name, kind, url-or-path, post-load action, full_page)
        ("pipeline.png", "file", flowchart_html, "crop_body", True),
        (
            "pipeline_dense.png",
            "file",
            dense_html,
            "crop_flow",
            False,
        ),
        (
            "screenshot_runs_dashboard.png",
            "url",
            f"{args.server}/ui/",
            "wait_runs",
            False,
        ),
        (
            "screenshot_launch_form.png",
            "url",
            f"{args.server}/ui/",
            "click_launch",
            False,
        ),
        (
            "screenshot_labeling_view.png",
            "url",
            f"{args.server}/ui/label.html?run={label_run_id}",
            "wait_label",
            False,
        ),
        ("screenshot_html_report.png", "file", report_html, None, False),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=2,  # crisper images for LaTeX
        )
        page = ctx.new_page()

        for name, kind, target, action, full_page in targets:
            out_path = out_dir / name
            url = (
                f"file://{Path(target).resolve()}" if kind == "file" else target
            )
            print(f"[screenshots] {name}  <-  {url}")
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception as e:
                print(f"  goto failed: {e}; retrying with domcontentloaded")
                page.goto(url, wait_until="domcontentloaded", timeout=30000)

            if action == "crop_body":
                # Polished diagram: snapshot just the layout (no chrome).
                try:
                    locator = page.locator("body")
                    locator.screenshot(path=str(out_path))
                    print(f"  -> {out_path}  ({out_path.stat().st_size} bytes)")
                    continue
                except Exception as e:
                    print(f"  body crop failed: {e}; falling back to full page")
            elif action == "crop_flow":
                # Dense pipeline.html: wait for mermaid, then crop
                # the #flow section's .panel (the diagram area only).
                try:
                    page.wait_for_selector(
                        "section#flow .panel .mermaid svg", timeout=15000
                    )
                except Exception:
                    pass
                time.sleep(0.6)
                try:
                    locator = page.locator("section#flow .panel")
                    locator.screenshot(path=str(out_path))
                    print(f"  -> {out_path}  ({out_path.stat().st_size} bytes)")
                    continue
                except Exception as e:
                    print(f"  flow crop failed: {e}; falling back to full page")
            elif action == "click_launch":
                # Activate the "Launch new run" tab on the workbench.
                try:
                    page.click('button.tab[data-tab="launch"]', timeout=5000)
                    page.wait_for_selector(
                        'section[data-panel="launch"]:not(.hidden)',
                        timeout=5000,
                    )
                except Exception as e:
                    print(f"  could not switch to launch tab: {e}")
                time.sleep(0.5)
            elif action == "wait_mermaid":
                # Mermaid renders client-side; wait for any SVG to appear.
                try:
                    page.wait_for_selector(".mermaid svg", timeout=8000)
                except Exception:
                    pass
                time.sleep(0.5)
            elif action == "wait_runs":
                try:
                    page.wait_for_function(
                        "document.getElementById('runs-list')"
                        " && !document.getElementById('runs-list').textContent"
                        ".includes('loading')",
                        timeout=5000,
                    )
                except Exception:
                    pass
                time.sleep(0.3)
            elif action == "wait_label":
                # Wait for clusters to render (or for the empty-state).
                try:
                    page.wait_for_function(
                        "(() => {"
                        "  const el = document.querySelector("
                        "    '#cluster-list, .cluster-list, "
                        "    [data-role=\"cluster-list\"]');"
                        "  if (!el) return true;"
                        "  const t = el.textContent || '';"
                        "  return !t.includes('loading');"
                        "})()",
                        timeout=8000,
                    )
                except Exception:
                    pass
                time.sleep(1.5)

            page.screenshot(path=str(out_path), full_page=full_page)
            print(f"  -> {out_path}  ({out_path.stat().st_size} bytes)")

        browser.close()

    print("[screenshots] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
