"""Self-contained: open the sheet by URL and walk
Data -> Data cleanup -> Cleanup suggestions, capturing shots + panel DOM.

Pass a sheet URL as argv[1], else uses the default test sheet.
"""

import sys

from harness import connect, new_page, shot

DEFAULT = "https://docs.google.com/spreadsheets/d/1a_ERZTMvxsk-LJnr_imdxOLB3aAgyd6PdKIME8TxfH8/edit"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    p, browser, ctx = connect()
    page = new_page(ctx)
    page.set_default_timeout(12000)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(9000)  # editor boot
        print("title:", page.title())
        shot(page, "s2_00_loaded")

        # 1) Data menu (scope to menubar to avoid stray matches)
        page.locator('[role="menubar"]').get_by_text("Data", exact=True).first.click()
        page.wait_for_timeout(1000)
        shot(page, "s2_01_data_open")

        # 2) Data cleanup submenu opener -- a real menuitem, hover to expand
        cleanup = page.locator('[role="menuitem"]:visible', has_text="Data cleanup").first
        print("data-cleanup menuitem visible:", cleanup.count() > 0)
        cleanup.hover()
        page.wait_for_timeout(1200)
        shot(page, "s2_02_cleanup_submenu")

        # dump visible submenu items
        vis = page.locator('[role="menuitem"]:visible')
        labels = []
        for i in range(vis.count()):
            t = (vis.nth(i).inner_text() or "").replace("\n", " ").strip()
            if t:
                labels.append(t)
        print("visible menuitems now:")
        for t in labels:
            print("   -", t[:60])

        # 3) Click Cleanup suggestions
        cs = page.locator('[role="menuitem"]:visible', has_text="Cleanup suggestions").first
        if cs.count() == 0:
            print("WARN: Cleanup suggestions menuitem not visible; trying any text")
            cs = page.get_by_text("Cleanup suggestions").first
        cs.click()
        page.wait_for_timeout(5000)
        shot(page, "s2_03_panel")

        # 4) Inspect the side panel
        for sel in [
            '[aria-label*="leanup"]',
            '[class*="cleanup"]',
            '.docs-sidebar',
            '[role="complementary"]',
        ]:
            loc = page.locator(sel)
            if loc.count():
                try:
                    txt = loc.first.inner_text()
                except Exception:  # noqa: BLE001
                    txt = "(no text)"
                print(f"\n=== panel via {sel} (count={loc.count()}) ===")
                print(txt[:1200])
                break
    except Exception as e:  # noqa: BLE001
        print("FLOW ERROR:", type(e).__name__, str(e)[:200])
        shot(page, "s2_error")
    finally:
        browser.close()
        p.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
