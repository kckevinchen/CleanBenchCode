"""Vertical slice: open an existing converted Sheet, trigger
Data -> Data cleanup -> Cleanup suggestions, and screenshot every step so we can
see the panel structure before building a scraper.

Sheet used: '25039521' inside the _ported subfolder (a benchmark table).
"""

import sys

from harness import connect, new_page, shot

PORTED_FOLDER = "https://drive.google.com/drive/folders/1wQVOd6O2j82h95i-Po4Ma6fK_yCjkseS"


def find_sheet_id(page):
    page.goto(PORTED_FOLDER, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(3500)
    # The Sheet is the row whose name has no .csv extension: "25039521"
    rows = page.locator('[role="row"]')
    n = rows.count()
    print("rows in ported folder:", n)
    sheet_row = None
    for i in range(n):
        el = rows.nth(i)
        txt = (el.inner_text() or "")
        first = txt.split("\n")[0].strip()
        print(f"  row{i}: {first[:40]}")
        if first == "25039521":  # converted sheet, no extension
            sheet_row = el
            break
    if not sheet_row:
        return None
    # open it (double click) -> a new tab usually opens for Sheets
    before = set(p.url for p in page.context.pages)
    sheet_row.dblclick()
    page.wait_for_timeout(6000)
    after = [p for p in page.context.pages if p.url not in before]
    target = None
    for pg in page.context.pages:
        if "spreadsheets/d/" in pg.url:
            target = pg
            break
    return target


def main():
    p, browser, ctx = connect()
    page = new_page(ctx)
    try:
        sheet_page = find_sheet_id(page)
        if not sheet_page:
            print("ERROR: could not open the Sheet")
            return 1
        sheet_page.bring_to_front()
        sheet_page.wait_for_timeout(8000)  # let Sheets editor fully load
        print("sheet url:", sheet_page.url)
        print("sheet title:", sheet_page.title())
        shot(sheet_page, "slice_10_sheet_loaded")

        # Open the Data menu (top menubar item with text 'Data')
        data_menu = sheet_page.locator('#docs-menu, [role="menubar"]').get_by_text(
            "Data", exact=True
        ).first
        data_menu.click()
        sheet_page.wait_for_timeout(1500)
        shot(sheet_page, "slice_11_data_menu")

        # Print visible menu item texts to locate the cleanup entry
        items = sheet_page.locator('[role="menuitem"]')
        n = items.count()
        print("Data menu items:", n)
        for i in range(n):
            t = (items.nth(i).inner_text() or "").replace("\n", " ")[:50]
            if t.strip():
                print("   -", t)
    finally:
        # leave browser open; just detach
        browser.close()
        p.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
