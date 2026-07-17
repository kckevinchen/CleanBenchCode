"""Open the target Drive folder and enumerate its items.

Detects each item's name, Drive id, and whether it is a Google Sheet vs a raw
file (e.g. CSV), by scrolling the virtualized list to the end.

Writes outputs/drive_folder_items.json and a screenshot.
"""

import json
import os
import sys

from harness import OUT, connect, new_page, shot

FOLDER_NAME = os.environ.get("FOLDER_NAME", "curated_bing21_500_dedup")


def open_folder(page):
    page.goto(
        f"https://drive.google.com/drive/search?q={FOLDER_NAME}",
        wait_until="domcontentloaded",
        timeout=60000,
    )
    page.wait_for_timeout(4000)
    # Click the folder row (a folder, not the .zip/json). Folders open on dblclick.
    row = page.locator(f'c-wiz [role="row"]:has-text("{FOLDER_NAME}")').first
    # Fallback: any element with matching aria-label that is a folder
    candidates = page.locator(f'[aria-label*="{FOLDER_NAME}"]')
    target = None
    n = candidates.count()
    for i in range(n):
        el = candidates.nth(i)
        al = (el.get_attribute("aria-label") or "")
        if "Folder" in al or "folder" in al:
            target = el
            break
    (target or row).dblclick()
    page.wait_for_timeout(5000)
    return page.url


def enumerate_items(page):
    """Scroll the file list and collect unique items by data-id."""
    seen = {}
    stagnant = 0
    last_count = 0
    for _ in range(80):
        rows = page.locator('[role="row"][data-id], c-wiz [data-id][aria-label]')
        cnt = rows.count()
        for i in range(cnt):
            el = rows.nth(i)
            did = el.get_attribute("data-id")
            al = el.get_attribute("aria-label") or ""
            if did and did not in seen:
                seen[did] = al
        # scroll the list container
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(700)
        if len(seen) == last_count:
            stagnant += 1
            if stagnant >= 4:
                break
        else:
            stagnant = 0
            last_count = len(seen)
    return seen


def classify(aria_label):
    al = aria_label.lower()
    if "google sheets" in al or "google-sheets" in al:
        return "sheet"
    if al.endswith(".csv") or ".csv " in al or "text" in al:
        return "csv"
    return "other"


def main():
    p, browser, ctx = connect()
    page = new_page(ctx)
    try:
        url = open_folder(page)
        print("folder url:", url)
        shot(page, "folder_open")
        items = enumerate_items(page)
        shot(page, "folder_enumerated")

        records = []
        for did, al in items.items():
            records.append({"id": did, "aria_label": al, "kind": classify(al)})
        kinds = {}
        for r in records:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1

        outpath = os.path.join(OUT, "drive_folder_items.json")
        with open(outpath, "w") as f:
            json.dump({"folder_url": url, "items": records}, f, indent=2)

        print("total items found:", len(records))
        print("by kind:", kinds)
        print("sample:")
        for r in records[:8]:
            print("  ", r["kind"], "|", r["aria_label"][:70], "|", r["id"][:20])
        print("wrote", outpath)
    finally:
        page.close()
        browser.close()
        p.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
