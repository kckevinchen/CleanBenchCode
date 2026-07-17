"""Batch runner: for every benchmark table, create a blank Google Sheet, import
the local CSV, run Data > Data cleanup > Cleanup suggestions, and save the panel
HTML for offline parsing.

Resumable: tables whose HTML already exists are skipped. Per-table failures are
logged and do not abort the run. Appends one line per table to
outputs/run_results.jsonl.

Usage:
  python run_all.py [--limit N] [--start I] [--no-trash]
"""

import argparse
import json
import os
import sys
import time
import traceback

from harness import OUT, connect, new_page, shot
from parse_cleanup_html import parse_html

MANIFEST = os.path.join(OUT, "manifest.json")
HTML_DIR = os.path.join(OUT, "html")
RESULTS = os.path.join(OUT, "run_results.jsonl")
os.makedirs(HTML_DIR, exist_ok=True)


def import_csv(page, csv_path):
    page.locator('[role="menubar"]').get_by_text("File", exact=True).first.click()
    page.wait_for_timeout(700)
    page.locator('[role="menuitem"]:visible', has_text="Import").first.click()
    page.wait_for_timeout(2500)

    # Click the "Upload" tab inside the picker iframe.
    for fr in page.frames:
        try:
            up = fr.get_by_text("Upload", exact=True)
            if up.count():
                up.first.click()
                break
        except Exception:  # noqa: BLE001
            continue
    page.wait_for_timeout(1200)

    # Set the file directly on the hidden <input type=file> (any frame).
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
    if not set_ok:
        raise RuntimeError("no file input found in import picker")
    page.wait_for_timeout(3500)

    # Click "Import data".
    for label in ["Import data", "Import"]:
        done = False
        for fr in page.frames:
            try:
                b = fr.get_by_text(label, exact=True)
                if b.count():
                    b.first.click()
                    done = True
                    break
            except Exception:  # noqa: BLE001
                continue
        if done:
            break
    page.wait_for_timeout(5500)


def run_cleanup(page):
    page.locator('[role="menubar"]').get_by_text("Data", exact=True).first.click()
    page.wait_for_timeout(700)
    page.locator('[role="menuitem"]:visible', has_text="Data cleanup").first.hover()
    page.wait_for_timeout(900)
    page.locator('[role="menuitem"]:visible', has_text="Cleanup suggestions").first.click()
    # Wait for the panel to settle (spinner gone / text populated), up to ~20s.
    deadline = time.time() + 20
    while time.time() < deadline:
        page.wait_for_timeout(1200)
        side = page.locator(".waffle-inline-assistant-sidebar")
        if side.count():
            txt = side.first.inner_text()
            if txt and "Loading" not in txt:
                break


def save_html(page, source):
    side = page.locator(".waffle-inline-assistant-sidebar")
    if side.count():
        # Scroll the panel to force any virtualized/lazy suggestions to render
        # before we snapshot, so the saved HTML is as complete as possible.
        try:
            box = side.first.bounding_box()
            if box:
                cx = box["x"] + box["width"] / 2
                cy = box["y"] + box["height"] / 2
                last_h = -1
                for _ in range(25):
                    h = side.first.evaluate("el => el.scrollHeight")
                    page.mouse.move(cx, cy)
                    page.mouse.wheel(0, 3000)
                    page.wait_for_timeout(350)
                    if h == last_h:
                        break
                    last_h = h
                # scroll back to top so the captured DOM starts clean
                page.mouse.wheel(0, -60000)
                page.wait_for_timeout(250)
        except Exception:  # noqa: BLE001
            pass
        html = side.first.evaluate("el => el.outerHTML")
    else:
        html = page.content()
    path = os.path.join(HTML_DIR, source + ".html")
    with open(path, "w") as f:
        f.write(html)
    return path


def trash_sheet(page):
    """Best-effort move-to-trash so Drive doesn't accumulate 200+ files."""
    try:
        page.locator('[role="menubar"]').get_by_text("File", exact=True).first.click()
        page.wait_for_timeout(600)
        page.locator('[role="menuitem"]:visible', has_text="Move to trash").first.click()
        page.wait_for_timeout(1500)
        # confirm dialog "Move to trash"
        btn = page.get_by_role("button", name="Move to trash")
        if btn.count():
            btn.first.click()
            page.wait_for_timeout(1200)
    except Exception:  # noqa: BLE001
        pass


def process_one(ctx, rec, trash):
    source = rec["source"]
    page = new_page(ctx)
    page.set_default_timeout(20000)
    result = {"source": source, "status": "ok", "sheet_id": None, "n_suggestions": 0,
              "categories": [], "error": None}
    try:
        page.goto("https://sheets.new", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(8000)
        if "accounts.google.com" in page.url:
            raise RuntimeError("redirected to login -- session expired")
        result["sheet_id"] = page.url.split("/d/")[1].split("/")[0]

        import_csv(page, rec["csv_path"])
        run_cleanup(page)
        path = save_html(page, source)

        parsed = parse_html(open(path).read())
        result["n_suggestions"] = len(parsed["suggestions"])
        result["categories"] = sorted({s["category"] for s in parsed["suggestions"]})
        if trash:
            trash_sheet(page)
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        try:
            shot(page, f"err_{source[:40]}")
        except Exception:  # noqa: BLE001
            pass
        traceback.print_exc()
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass
    return result


def already_done():
    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done.add(r["source"])
                except Exception:  # noqa: BLE001
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST,
                    help="path to manifest JSON (default: the benchmark manifest)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-trash", action="store_true")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    man = man[args.start:]
    if args.limit:
        man = man[: args.limit]

    done = already_done()
    print(f"tables to consider: {len(man)} | already done: {len(done)}")

    p, browser, ctx = connect()
    t0 = time.time()
    n_ok = n_err = n_skip = 0
    try:
        with open(RESULTS, "a") as out:
            for i, rec in enumerate(man):
                src = rec["source"]
                if src in done:
                    n_skip += 1
                    continue
                r = process_one(ctx, rec, trash=not args.no_trash)
                out.write(json.dumps(r) + "\n")
                out.flush()
                if r["status"] == "ok":
                    n_ok += 1
                else:
                    n_err += 1
                el = time.time() - t0
                rate = el / max(n_ok + n_err, 1)
                print(
                    f"[{i+1}/{len(man)}] {src[:42]:42} "
                    f"{r['status']:5} sugg={r['n_suggestions']} "
                    f"cats={','.join(r['categories']) or '-'} "
                    f"({rate:.0f}s/tbl)",
                    flush=True,
                )
                if r["status"] == "error" and r["error"] and "session expired" in r["error"]:
                    print("ABORTING: Google session expired. Re-login via VNC and rerun.",
                          flush=True)
                    break
                time.sleep(2.5)
    finally:
        browser.close()
        p.stop()
    print(f"\nDONE ok={n_ok} err={n_err} skip={n_skip} elapsed={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
