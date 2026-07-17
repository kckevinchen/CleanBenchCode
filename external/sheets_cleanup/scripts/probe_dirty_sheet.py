"""Probe: create a deliberately-dirty sheet (duplicate row + whitespace),
trigger Cleanup suggestions, and dump the panel HTML so we can design a parser.
"""

import os
import sys

from harness import OUT, connect, new_page, shot


def type_grid(page):
    # click name box, go to A1
    page.keyboard.press("Control+Home")
    page.wait_for_timeout(500)
    # Enter a small grid with a duplicate row (Alice/NYC twice) and trailing
    # whitespace ("Bob "). Tab = next col, Enter = next row.
    rows = [
        ["name", "city"],
        ["Alice", "NYC"],
        ["Alice", "NYC"],
        ["Bob ", "LA"],
    ]
    for r in rows:
        for c, val in enumerate(r):
            page.keyboard.type(val, delay=20)
            if c < len(r) - 1:
                page.keyboard.press("Tab")
        page.keyboard.press("Enter")
        page.wait_for_timeout(150)
    page.wait_for_timeout(1500)


def open_cleanup(page):
    page.locator('[role="menubar"]').get_by_text("Data", exact=True).first.click()
    page.wait_for_timeout(800)
    page.locator('[role="menuitem"]:visible', has_text="Data cleanup").first.hover()
    page.wait_for_timeout(1000)
    page.locator('[role="menuitem"]:visible', has_text="Cleanup suggestions").first.click()
    page.wait_for_timeout(4000)


def dump_panel(page):
    # The sidebar lives on the right; capture a generous container outerHTML.
    candidates = [
        '[aria-label*="leanup"]',
        '[class*="cleanup"]',
        '.docs-sidebar-container',
        '#docs-sidebar',
        '[role="complementary"]',
    ]
    html = None
    used = None
    for sel in candidates:
        loc = page.locator(sel)
        if loc.count():
            try:
                html = loc.first.evaluate("el => el.outerHTML")
                used = sel
                break
            except Exception:  # noqa: BLE001
                continue
    return used, html


def main():
    p, browser, ctx = connect()
    page = new_page(ctx)
    page.set_default_timeout(15000)
    try:
        page.goto("https://sheets.new", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(9000)
        print("new sheet:", page.url.split("?")[0])
        type_grid(page)
        shot(page, "probe_01_data")
        open_cleanup(page)
        shot(page, "probe_02_panel")

        used, html = dump_panel(page)
        print("panel selector used:", used)
        if html:
            path = os.path.join(OUT, "sample_panel.html")
            with open(path, "w") as f:
                f.write(html)
            print("panel html bytes:", len(html), "->", path)

        # Also dump the full page HTML for offline analysis
        full = page.content()
        fp = os.path.join(OUT, "sample_full_page.html")
        with open(fp, "w") as f:
            f.write(full)
        print("full page bytes:", len(full), "->", fp)

        # Quick visible text of panel
        for sel in ['[aria-label*="leanup"]', '[class*="cleanup"]']:
            loc = page.locator(sel)
            if loc.count():
                print(f"\npanel text via {sel}:\n", loc.first.inner_text()[:600])
                break
    except Exception as e:  # noqa: BLE001
        print("ERROR:", type(e).__name__, str(e)[:200])
        shot(page, "probe_error")
    finally:
        browser.close()
        p.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
