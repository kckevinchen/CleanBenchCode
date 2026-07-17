"""A/B test: re-run a few tables on the LATEST Chrome (150) using the existing
login (injected via storage_state), and compare the Cleanup-suggestions output
to what Chrome 148 produced. Decisive test of whether the feature is
browser-version dependent (client-side) or server-side (version-independent).

Saves new captures to outputs/html_v150/ and prints a per-table diff.
"""

import json
import os
import sys
import time

from parse_cleanup_html import expand_ranges, parse_html
from playwright.sync_api import sync_playwright
from run_all import import_csv, run_cleanup

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
CHROME150 = os.path.join(HERE, "..", "browsers", "chrome-linux64", "chrome")
STATE = os.path.join(OUT, "auth_state.json")
HTML_148 = os.path.join(OUT, "html")
HTML_150 = os.path.join(OUT, "html_v150")
MANIFEST = os.path.join(OUT, "manifest.json")
os.makedirs(HTML_150, exist_ok=True)

TABLES = [
    "03d4cd518bb384be83d224597c7aa5c5698145f9.xlsx_845641bb.csv",  # DUPLICATE ROWS
    "047a17db9cd660bcc59982658992245f666403c7.xlsx_e6aa6cbe.csv",  # COLUMN FORMAT
    "0803f2d418afc201eb7d82f8f1ca149c5ed9d4c3.xlsx_8658ff28.csv",  # INVALID DATA
    "0cb0363a3fb3c431bf71f3067feb9ba7cec0a4c7.xlsx_9a6e117e.csv",  # WHITESPACE
]


def fingerprint(parsed):
    """Order-independent signature of suggestions: {(category, type, ranges, value)}."""
    sig = set()
    for s in parsed["suggestions"]:
        sig.add((s["category"], s["extra"].get("type", ""), tuple(s["ranges"]),
                 s["value"].strip()))
    return sig


def save_html(page, source, outdir):
    side = page.locator(".waffle-inline-assistant-sidebar")
    if side.count():
        try:
            box = side.first.bounding_box()
            if box:
                cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                last = -1
                for _ in range(25):
                    h = side.first.evaluate("el => el.scrollHeight")
                    page.mouse.move(cx, cy)
                    page.mouse.wheel(0, 3000)
                    page.wait_for_timeout(350)
                    if h == last:
                        break
                    last = h
                page.mouse.wheel(0, -60000)
                page.wait_for_timeout(250)
        except Exception:  # noqa: BLE001
            pass
        html = side.first.evaluate("el => el.outerHTML")
    else:
        html = page.content()
    path = os.path.join(outdir, source + ".html")
    with open(path, "w") as f:
        f.write(html)
    return path


def main():
    manifest = {m["source"]: m for m in json.load(open(MANIFEST))}
    print(f"Chrome 150 binary: {CHROME150}")
    os.environ.setdefault("DISPLAY", ":99")

    p = sync_playwright().start()
    browser = p.chromium.launch(
        executable_path=CHROME150,
        headless=False,
        args=["--no-sandbox", "--disable-dev-shm-usage",
              "--disable-blink-features=AutomationControlled", "--lang=en-US"],
    )
    ctx = browser.new_context(storage_state=STATE)
    ver = ctx.new_page().evaluate("() => navigator.userAgent")
    print("UA:", ver)

    results = []
    try:
        for source in TABLES:
            rec = manifest.get(source)
            if not rec:
                print("  missing manifest:", source)
                continue
            page = ctx.new_page()
            page.set_default_timeout(20000)
            try:
                page.goto("https://sheets.new", wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(8000)
                if "accounts.google.com" in page.url:
                    print("  LOGIN REQUIRED on Chrome150 -- cookie injection rejected.")
                    print("  (Will need a one-time VNC login in the 150 browser.)")
                    results.append((source, None, "login_required"))
                    page.close()
                    break
                import_csv(page, rec["csv_path"])
                run_cleanup(page)
                path = save_html(page, source, HTML_150)
                parsed150 = parse_html(open(path).read())
                parsed148 = parse_html(open(os.path.join(HTML_148, source + ".html")).read())
                sig150, sig148 = fingerprint(parsed150), fingerprint(parsed148)
                match = sig150 == sig148
                results.append((source, match, (len(sig148), len(sig150))))
                print(f"  {source[:36]:36} 148={len(sig148)} 150={len(sig150)} "
                      f"MATCH={match}")
                if not match:
                    print("    only-in-148:", sig148 - sig150)
                    print("    only-in-150:", sig150 - sig148)
            except Exception as e:  # noqa: BLE001
                print("  ERROR:", type(e).__name__, str(e)[:160])
                results.append((source, None, "error"))
            finally:
                try:
                    page.close()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(2)
    finally:
        browser.close()
        p.stop()

    ok = [r for r in results if r[1] is True]
    bad = [r for r in results if r[1] is False]
    print(f"\n=== A/B SUMMARY: {len(ok)} identical, {len(bad)} differ, "
          f"{len(results)-len(ok)-len(bad)} skipped/error ===")
    if results and all(r[1] is True for r in results):
        print("VERDICT: Chrome 148 and 150 produce IDENTICAL suggestions -> "
              "feature is server-side; the run is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
