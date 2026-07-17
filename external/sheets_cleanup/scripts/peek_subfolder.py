"""Descend into the first subfolder of the target folder and report inner file
type (Google Sheet vs CSV). Writes a screenshot."""

import sys

from harness import connect, new_page, shot

PARENT = "https://drive.google.com/drive/folders/1hn4ZYd8QYNu1nULipr23bUU53zSo9-Wk"


def main():
    p, browser, ctx = connect()
    page = new_page(ctx)
    try:
        page.goto(PARENT, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)
        # open a NORMAL subfolder (not the _ported ones)
        target = page.locator('[role="row"]:has-text("005c05b7029cb909b653bd2942b5272c5030e130")').first
        name = (target.inner_text() or "").split("\n")[0]
        print("opening subfolder:", name[:60])
        target.dblclick()
        page.wait_for_timeout(4000)
        shot(page, "subfolder_open")
        print("url:", page.url)
        # list inner rows with their text + any 'Google Sheets' hint
        rows = page.locator('[role="row"]')
        n = rows.count()
        print("inner rows:", n)
        for i in range(min(n, 6)):
            el = rows.nth(i)
            txt = (el.inner_text() or "").replace("\n", " | ")[:90]
            al = el.get_attribute("aria-label") or ""
            print(f"  row{i}: text=[{txt}]  aria=[{al[:60]}]")
    finally:
        page.close()
        browser.close()
        p.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
