"""Step 3: for each table, upload its .xlsx, open it in Excel for the web, open
the Copilot pane, ask for data-quality issues, and save Copilot's response.

    python scripts/run_excel_copilot.py --limit 1 --watch     # try ONE table, visible + slow
    python scripts/run_excel_copilot.py                       # full run (resumable)

Resumable: tables already in outputs/results.jsonl (status ok) are skipped.
We save, per table: outputs/html/<source>.html (Copilot pane DOM),
outputs/text/<source>.txt (answer text), and step screenshots in outputs/shots/.

IMPORTANT: the Copilot UI selectors below are best-effort. Run with --watch
first; if a step fails, read the saved screenshots, update the SELECTORS dict,
and re-run. See AGENTS.md for how to harden selectors with codegen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from _common import (
    DATA,
    HTML_DIR,
    OUT,
    TEXT_DIR,
    XLSX_DIR,
    launch_context,
    shot,
)

RESULTS = OUT / "results.jsonl"
CELLS_DIR = OUT / "cells"
CELLS_DIR.mkdir(parents=True, exist_ok=True)

# Opening an uploaded workbook on this managed device bounces through an AAD
# "Pick an account" page (multiple work accounts). Choose the Copilot-licensed
# one. Override with the M365_EMAIL env var if needed.
ACCOUNT_EMAIL = os.environ.get("M365_EMAIL", "t-kaiwenchen@microsoft.com")

# Excel for the web renders inside an iframe whose URL contains "officeapps".
# The data-quality feature is the Data ribbon tab -> "Clean Data" button, which
# opens the "Clean Data with Copilot" task pane. We only read its suggestions.
SELECTORS = {
    "upload_button": [
        'text="Upload a file"',
        '[aria-label*="Upload" i]',
    ],
}


def try_click(scope, sel_list, timeout=8000):
    for sel in sel_list:
        try:
            loc = scope.locator(sel).first
            if loc.count():
                loc.click(timeout=timeout)
                return sel
        except Exception:  # noqa: BLE001
            continue
    return None


def editor_tab(ctx, before_urls):
    """The newly opened Excel-for-web editor tab (OneDrive doc). The editor URL
    contains '/_layouts/15/doc' (Doc.aspx or doc2.aspx). Only NEW tabs (not in
    before_urls) are considered, so a stale editor is never mis-selected."""
    for pp in ctx.pages:
        if pp.url in before_urls:
            continue
        u = (pp.url or "").lower()
        if "sharepoint.com" in u and "_layouts/15/doc" in u:
            return pp
    return None


def handle_account_picker(ctx, email=ACCOUNT_EMAIL, timeout=8000):
    """If an AAD 'Pick an account' page is showing, choose `email`."""
    for pp in ctx.pages:
        u = (pp.url or "").lower()
        if "login.microsoftonline" not in u and "login.live" not in u:
            continue
        try:
            tile = pp.get_by_text(email, exact=False).first
            if tile.count():
                tile.click(timeout=timeout)
                pp.wait_for_timeout(3000)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def officeapps_frame(page, timeout=45):
    """The Excel app iframe (URL contains 'officeapps'). It only attaches when
    the editor tab is opened within this same connected session."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for fr in page.frames:
            if "officeapps" in (fr.url or "").lower():
                return fr
        page.wait_for_timeout(1500)
    return None


def wait_workbook_ready(page, max_s=75):
    """Wait until the visible 'still loading the workbook' banner clears."""
    deadline = time.time() + max_s
    while time.time() < deadline:
        loading = False
        for fr in page.frames:
            try:
                for txt in ("still loading the workbook", "Please wait"):
                    el = fr.get_by_text(txt, exact=False).first
                    if el.count() and el.is_visible():
                        loading = True
                        break
            except Exception:  # noqa: BLE001
                pass
            if loading:
                break
        if not loading:
            return True
        page.wait_for_timeout(1500)
    return False


def _click_first(scope, makers, timeout=3500):
    """Click the first locator (from a list of zero-arg lambdas) that exists."""
    for mk in makers:
        try:
            loc = mk().first
            if loc.count():
                loc.click(timeout=timeout)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def open_clean_data(fr, page):
    """Activate the Data ribbon tab and open the Clean Data pane.

    Handles both ribbon layouts: a wide window shows the 'Clean Data' button
    directly, while a narrow window (e.g. a laptop screen) collapses it under a
    'Data Tools' dropdown -> Data > Data Tools > Clean Data. Success is verified
    by the 'Clean Data with Copilot' pane appearing.
    """
    data_tab = [
        lambda: fr.get_by_role("tab", name="Data", exact=True),
        lambda: fr.get_by_role("tab", name="Data", exact=False),
        lambda: fr.get_by_text("Data", exact=True),
    ]
    clean_btn = [
        lambda: fr.get_by_role("menuitem", name="Clean Data", exact=False),
        lambda: fr.get_by_role("button", name="Clean Data", exact=False),
        lambda: fr.get_by_text("Clean Data", exact=True),
    ]
    data_tools = [
        lambda: fr.get_by_role("button", name="Data Tools", exact=False),
        lambda: fr.get_by_text("Data Tools", exact=False),
    ]

    def pane_open():
        return fr.get_by_text("Clean Data with Copilot", exact=False).count() > 0

    for _ in range(6):
        _click_first(fr, data_tab)
        page.wait_for_timeout(1200)
        clicked = _click_first(fr, clean_btn)
        if not clicked:
            # Narrow ribbon: expand the collapsed 'Data Tools' group first.
            if _click_first(fr, data_tools):
                page.wait_for_timeout(1200)
                clicked = _click_first(fr, clean_btn)
        if clicked:
            page.wait_for_timeout(2500)
            if pane_open():
                return True
        page.wait_for_timeout(1000)
    return pane_open()


def _pane_info(fr):
    """Return (text, html) of the 'Clean Data with Copilot' pane, or ('', '')."""
    h = fr.get_by_text("Clean Data with Copilot", exact=False).first
    try:
        if not h.count():
            return "", ""
        out = h.evaluate(
            """el => { let n = el;
                   for (let i = 0; i < 7 && n.parentElement; i++) n = n.parentElement;
                   return [n.innerText, n.outerHTML]; }"""
        )
        return (out[0] or ""), (out[1] or "")
    except Exception:  # noqa: BLE001
        return "", ""


HIGHLIGHT_JS = (
    "() => [...document.querySelectorAll('.ewr-data-cleansing-bad-highlight')]"
    ".map(e => ({top: parseInt(e.style.top)||0, height: parseInt(e.style.height)||0,"
    " left: parseInt(e.style.left)||0, width: parseInt(e.style.width)||0}))"
)


def _select_card(card, page):
    """Select a suggestion card so Excel highlights its affected cells, without
    touching Apply/Ignore: click the card's type label (top), else its top-left."""
    try:
        card.locator("label").first.click(timeout=4000)
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        card.click(position={"x": 24, "y": 18}, timeout=4000)
    except Exception:  # noqa: BLE001
        pass


def capture_clean_data(fr, page):
    """Wait for analysis, capture the whole pane, then SELECT each suggestion
    card and record its affected-cell highlight rects
    (.ewr-data-cleansing-bad-highlight). Capture-only: selecting never applies a
    change. Returns (text, html, card_data) with card_data a list of
    {html, rects}; rects are content-space px overlays for offline row mapping.
    """
    deadline = time.time() + 90
    while time.time() < deadline:
        if fr.get_by_role("button", name="Ignore", exact=False).count():
            break
        # Completion or error messages also end the wait.
        if (fr.get_by_text("didn't find suggestions", exact=False).count()
                or fr.get_by_text("No clean data suggestions", exact=False).count()
                or fr.get_by_text("Something went wrong", exact=False).count()):
            break
        page.wait_for_timeout(2000)
    page.wait_for_timeout(1500)  # let the list settle

    text, html = _pane_info(fr)

    cards = fr.locator(".suggestion-card")
    card_data = []
    for i in range(cards.count()):
        card = cards.nth(i)
        try:
            chtml = card.evaluate("e => e.outerHTML")
        except Exception:  # noqa: BLE001
            chtml = ""
        _select_card(card, page)
        page.wait_for_timeout(700)
        try:
            rects = fr.evaluate(HIGHLIGHT_JS)
        except Exception:  # noqa: BLE001
            rects = []
        card_data.append({"html": chtml, "rects": rects})
    return text, html, card_data


def process_one(ctx, rec, watch):
    source = rec["source"]
    xlsx = XLSX_DIR / rec["xlsx"]
    res = {"source": source, "status": "ok", "error": None, "answer_chars": 0}
    page = ctx.new_page()
    ed = None
    try:
        before = {pp.url for pp in ctx.pages}
        page.goto("https://excel.cloud.microsoft/", wait_until="domcontentloaded",
                  timeout=90000)
        page.wait_for_timeout(5000)
        shot(page, f"{source}__01_landing")

        # Upload the xlsx via the file chooser behind "Upload a file".
        with page.expect_file_chooser(timeout=20000) as fc_info:
            if not try_click(page, SELECTORS["upload_button"]):
                raise RuntimeError("could not find 'Upload a file' button")
        fc_info.value.set_files(str(xlsx))

        # The workbook opens in a new tab, often via an AAD account picker.
        for _ in range(30):
            page.wait_for_timeout(2000)
            handle_account_picker(ctx)
            ed = editor_tab(ctx, before)
            if ed:
                break
        if not ed:
            raise RuntimeError("workbook editor tab did not open")
        ed.bring_to_front()
        shot(ed, f"{source}__02_opening")

        # The Excel app frame only attaches because we opened the tab in-session.
        fr = officeapps_frame(ed)
        if not fr:
            raise RuntimeError("Excel app frame (officeapps) not found")
        wait_workbook_ready(ed)
        ed.wait_for_timeout(2500)

        # Open Data ribbon tab -> Clean Data (handles the collapsed 'Data Tools'
        # group on narrow windows: Data > Data Tools > Clean Data).
        shot(ed, f"{source}__03_ribbon")
        if not open_clean_data(fr, ed):
            raise RuntimeError("could not open Clean Data (Data > Data Tools > Clean Data)")
        shot(ed, f"{source}__04_clean_pane")

        # Read the pane, then select each card to capture its affected cells.
        text, html, card_data = capture_clean_data(fr, ed)
        shot(ed, f"{source}__05_captured")

        (HTML_DIR / f"{source}.html").write_text(html or ed.content(), encoding="utf-8")
        (TEXT_DIR / f"{source}.txt").write_text(text, encoding="utf-8")
        (CELLS_DIR / f"{source}.json").write_text(
            json.dumps({"source": source, "cards": card_data}, ensure_ascii=False),
            encoding="utf-8")
        res["answer_chars"] = len(text)
        res["n_cards"] = len(card_data)
        res["n_highlights"] = sum(len(c["rects"]) for c in card_data)
        # A transient Clean Data failure shouldn't count as done, so reruns retry it.
        if (not text.strip()) or ("Something went wrong" in text):
            res["status"] = "error"
            res["error"] = "clean data pane error or empty"
    except Exception as e:  # noqa: BLE001
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
        shot(ed or page, f"{source}__ERROR")
    finally:
        try:
            if ed:
                ed.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass
    return res


def already_done():
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done.add(r["source"])
                except Exception:  # noqa: BLE001
                    pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process at most N tables")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--watch", action="store_true",
                    help="headed + slow, for debugging the flow")
    args = ap.parse_args()

    man_path = DATA / "manifest_xlsx.json"
    if not man_path.exists():
        print("ERROR: run scripts/convert_to_xlsx.py first.")
        return 1
    manifest = json.loads(man_path.read_text())[args.start:]
    if args.limit:
        manifest = manifest[: args.limit]

    done = already_done()
    print(f"tables: {len(manifest)} | already done: {len(done)}")

    slow = 400 if args.watch else 0
    with sync_playwright() as p:
        ctx = launch_context(p, headed=True, slow_mo=slow)
        n_ok = n_err = 0
        with open(RESULTS, "a", encoding="utf-8") as out:
            for i, rec in enumerate(manifest):
                if rec["source"] in done:
                    continue
                r = process_one(ctx, rec, args.watch)
                out.write(json.dumps(r) + "\n")
                out.flush()
                n_ok += r["status"] == "ok"
                n_err += r["status"] == "error"
                print(f"[{i+1}/{len(manifest)}] {rec['source'][:40]:40} "
                      f"{r['status']:5} chars={r['answer_chars']} "
                      f"cards={r.get('n_cards', 0)} hl={r.get('n_highlights', 0)} "
                      f"{r['error'] or ''}", flush=True)
                time.sleep(2)
        ctx.close()
    print(f"\nDONE ok={n_ok} err={n_err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
