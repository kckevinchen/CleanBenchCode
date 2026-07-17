"""Prototype: create a blank Sheet, import ONE local CSV via File > Import,
then run Cleanup suggestions and save the panel HTML. Screenshot every step.

This proves the import + cleanup + save loop before scaling to all tables.
"""

import json
import os
import sys

from harness import OUT, connect, new_page, shot

MANIFEST = os.path.join(OUT, "manifest.json")


def menu_import(page, csv_path):
    # File menu
    page.locator('[role="menubar"]').get_by_text("File", exact=True).first.click()
    page.wait_for_timeout(800)
    page.locator('[role="menuitem"]:visible', has_text="Import").first.click()
    page.wait_for_timeout(2500)
    shot(page, "imp_01_dialog")

    # The import/upload UI is the Google Picker (an iframe). Click the Upload tab,
    # then set files on the <input type=file> via the file chooser.
    # Try tab by text across frames.
    clicked_upload = False
    for fr in page.frames:
        try:
            up = fr.get_by_text("Upload", exact=True)
            if up.count():
                up.first.click()
                clicked_upload = True
                break
        except Exception:  # noqa: BLE001
            continue
    page.wait_for_timeout(1500)
    shot(page, "imp_02_upload_tab")
    print("clicked Upload tab:", clicked_upload)

    # Now trigger the file chooser. The picker shows a "Browse" button.
    fc = None
    try:
        with page.expect_file_chooser(timeout=8000) as fci:
            done = False
            for fr in page.frames:
                try:
                    b = fr.get_by_text("Browse", exact=False)
                    if b.count():
                        b.first.click()
                        done = True
                        break
                except Exception:  # noqa: BLE001
                    continue
            if not done:
                # maybe a hidden input exists; click the drop area text
                for fr in page.frames:
                    try:
                        d = fr.get_by_text("Drag", exact=False)
                        if d.count():
                            d.first.click()
                            break
                    except Exception:  # noqa: BLE001
                        continue
        fc = fci.value
    except Exception as e:  # noqa: BLE001
        print("file_chooser path failed:", str(e)[:120])

    if fc is not None:
        fc.set_files(csv_path)
        print("set file via chooser:", os.path.basename(csv_path))
    else:
        # fallback: set_input_files on any file input across frames
        set_ok = False
        for fr in page.frames:
            try:
                inp = fr.locator('input[type="file"]')
                if inp.count():
                    inp.first.set_input_files(csv_path)
                    set_ok = True
                    break
            except Exception:  # noqa: BLE001
                continue
        print("set file via input:", set_ok)

    page.wait_for_timeout(4000)
    shot(page, "imp_03_after_select")


def click_import_data(page):
    # After upload, an import-options dialog appears with an "Import data" button.
    for label in ["Import data", "Import"]:
        for fr in page.frames:
            try:
                b = fr.get_by_text(label, exact=True)
                if b.count():
                    b.first.click()
                    print("clicked:", label)
                    page.wait_for_timeout(6000)
                    return True
            except Exception:  # noqa: BLE001
                continue
    return False


def run_cleanup(page):
    page.locator('[role="menubar"]').get_by_text("Data", exact=True).first.click()
    page.wait_for_timeout(800)
    page.locator('[role="menuitem"]:visible', has_text="Data cleanup").first.hover()
    page.wait_for_timeout(1000)
    page.locator('[role="menuitem"]:visible', has_text="Cleanup suggestions").first.click()
    page.wait_for_timeout(5000)


def save_panel(page, source):
    side = page.locator(".waffle-inline-assistant-sidebar")
    html = side.first.evaluate("el => el.outerHTML") if side.count() else page.content()
    path = os.path.join(OUT, "html", source + ".html")
    with open(path, "w") as f:
        f.write(html)
    return path, side.count() > 0


def main():
    man = json.load(open(MANIFEST))
    rec = man[0]
    csv_path, source = rec["csv_path"], rec["source"]
    print("table:", source, "rows:", rec["n_rows"], "cols:", rec["n_cols"])

    p, browser, ctx = connect()
    page = new_page(ctx)
    page.set_default_timeout(15000)
    try:
        page.goto("https://sheets.new", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(9000)
        sheet_url = page.url.split("?")[0]
        print("blank sheet:", sheet_url)

        menu_import(page, csv_path)
        click_import_data(page)
        shot(page, "imp_04_imported")
        # verify some data is present
        print("A1 area text:", page.locator("#waffle-grid-container").inner_text()[:80].replace("\n", " ") if page.locator("#waffle-grid-container").count() else "(grid not found)")

        run_cleanup(page)
        shot(page, "imp_05_panel")
        path, ok = save_panel(page, source)
        print("panel sidebar present:", ok, "saved:", path)
    except Exception as e:  # noqa: BLE001
        print("ERROR:", type(e).__name__, str(e)[:200])
        shot(page, "imp_error")
    finally:
        browser.close()
        p.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
