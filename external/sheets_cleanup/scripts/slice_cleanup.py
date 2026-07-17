"""Vertical slice part 2: from the open Sheet, navigate
Data -> Data cleanup -> Cleanup suggestions and capture the panel DOM + shots.
"""

import sys

from harness import connect, shot


def get_sheet_page(ctx):
    for pg in ctx.pages:
        if "spreadsheets/d/" in pg.url:
            return pg
    return None


def open_data_cleanup(page):
    # Data menu
    page.locator('[role="menubar"]').get_by_text("Data", exact=True).first.click()
    page.wait_for_timeout(1200)
    # Hover the "Data cleanup" submenu opener
    page.get_by_text("Data cleanup", exact=True).first.hover()
    page.wait_for_timeout(1500)
    shot(page, "slice_12_cleanup_submenu")
    # Dump submenu item texts
    items = page.locator('[role="menuitem"]:visible')
    n = items.count()
    print("visible menu items after hover:")
    labels = []
    for i in range(n):
        t = (items.nth(i).inner_text() or "").replace("\n", " ").strip()
        if t:
            labels.append(t)
            print("   -", t[:60])
    return labels


def main():
    p, browser, ctx = connect()
    try:
        page = get_sheet_page(ctx)
        if not page:
            print("ERROR: no spreadsheet tab open; run slice_open_sheet.py first")
            return 1
        page.bring_to_front()
        page.wait_for_timeout(1500)
        print("sheet url:", page.url)
        open_data_cleanup(page)

        # Click "Cleanup suggestions"
        cs = page.get_by_text("Cleanup suggestions", exact=False).first
        if cs.count() == 0:
            print("WARN: 'Cleanup suggestions' not found by text")
        else:
            cs.click()
            page.wait_for_timeout(4000)
            shot(page, "slice_13_cleanup_panel")
            print("clicked Cleanup suggestions")
            # Dump any side-panel text
            panel = page.locator('[role="complementary"], .docs-sidebar, [aria-label*="leanup"]')
            print("panel matches:", panel.count())
            try:
                print("panel text:\n", panel.first.inner_text()[:800])
            except Exception as e:  # noqa: BLE001
                print("panel read err:", e)
    finally:
        browser.close()
        p.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
