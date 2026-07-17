"""Apply Copilot's Clean Data fixes and capture the modified workbook, so we can
recover the SUGGESTED FIX for every flagged cell by diffing before vs after.

Many Clean Data suggestions (Inconsistent pattern / text / order, Extra spaces)
never state the replacement value in the pane -- they only show an example. The
actual fix is only revealed when you click Apply. So, per table that has
actionable suggestions, this script re-opens the workbook, snapshots the grid
(clipboard TSV of the used range), clicks Apply on every actionable card,
snapshots the grid again, and saves both to
outputs/applied/<source>.{before,after}.tsv. Offline, scripts/diff_applied.py
diffs the two snapshots to emit the per-cell fixes.

    python scripts/apply_and_capture.py --only dfd8d8e1 --limit 1 --watch  # validate one
    python scripts/apply_and_capture.py                                    # all w/ suggestions

NOTE: this DELIBERATELY clicks Apply (unlike run_excel_copilot.py, which is
capture-only). It only mutates the throwaway cloud copy of each fresh upload;
the local source data is never touched. Resumable: tables whose
outputs/applied/<source>.xlsx already exists are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time

from playwright.sync_api import sync_playwright

from _common import OUT, ROOT, launch_context, shot
from run_excel_copilot import (
    ACCOUNT_EMAIL,  # noqa: F401  (kept for parity / env override visibility)
    SELECTORS,
    editor_tab,
    handle_account_picker,
    officeapps_frame,
    open_clean_data,
    try_click,
    wait_workbook_ready,
)

APPLIED_DIR = OUT / "applied"
APPLIED_DIR.mkdir(parents=True, exist_ok=True)
APPLIED_RESULTS = OUT / "applied_results.jsonl"
CELLS_DIR = OUT / "cells"

APPLY_BTN = '[data-testid="SuggestionCardApplyButton"]'


def source_to_xlsx():
    """source (csv name) -> Path of its uploaded .xlsx, from every batch manifest."""
    m = {}
    for man in sorted(ROOT.glob("*/manifest_xlsx.json")):
        xdir = man.parent / "xlsx"
        try:
            for rec in json.loads(man.read_text()):
                m.setdefault(rec["source"], xdir / rec["xlsx"])
        except Exception:  # noqa: BLE001
            continue
    return m


def source_to_csv():
    """source (csv name) -> Path of its source CSV, from every batch manifest."""
    m = {}
    for man in sorted(ROOT.glob("*/manifest_xlsx.json")):
        cdir = man.parent / "csv"
        try:
            for rec in json.loads(man.read_text()):
                m.setdefault(rec["source"], cdir / rec["source"])
        except Exception:  # noqa: BLE001
            continue
    return m


def index_to_col(idx):
    """0-based column index -> letter (0 -> A, 26 -> AA)."""
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def table_range(csv_path):
    """Exact A1 range covering the table, derived from its source CSV dims."""
    try:
        with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.reader(fh))
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    ncols = max((len(r) for r in rows), default=1)
    return f"A1:{index_to_col(max(0, ncols - 1))}{len(rows)}"


def actionable_sources():
    """Sources whose captured cards include at least one Apply button (i.e. a fix
    Copilot can actually perform). Read from the existing outputs/cells/*.json."""
    out = []
    for jf in sorted(CELLS_DIR.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        cards = data.get("cards", [])
        if any("SuggestionCardApplyButton" in (c.get("html") or "") for c in cards):
            out.append(data.get("source") or jf.stem)
    return out


def apply_all_cards(fr, page, max_apply=80):
    """Click Apply on each actionable card until none remain. The list re-renders
    after each apply (the applied card loses its Apply button), so we always click
    the first remaining Apply button rather than iterating by index."""
    applied = 0
    for _ in range(max_apply):
        btn = fr.locator(APPLY_BTN).first
        try:
            if not btn.count():
                break
        except Exception:  # noqa: BLE001
            break
        try:
            btn.scroll_into_view_if_needed(timeout=3000)
        except Exception:  # noqa: BLE001
            pass
        try:
            btn.click(timeout=6000)
        except Exception:  # noqa: BLE001
            try:
                btn.evaluate("e => e.click()")
            except Exception:  # noqa: BLE001
                break
        applied += 1
        page.wait_for_timeout(1600)
    return applied


def capture_grid_tsv(ed, fr, range_str, tries=3):
    """Select an explicit A1 range via the Name Box and copy it, then read the
    clipboard as TSV.

    We first overwrite the OS clipboard with a sentinel, press Ctrl+C, then poll
    until the clipboard no longer holds the sentinel. Excel's async clipboard
    write lags the keystroke, so without this the AFTER snapshot can read the
    stale BEFORE copy (making the diff empty). Selecting by a typed range (not
    Ctrl+Shift+End) doesn't need the grid to already hold focus, and a clipboard
    copy is not virtualized, so the whole table is captured. Returns TSV text."""
    sentinel = "__HARNESS_CLIP__"
    for _ in range(tries):
        try:
            nb = fr.locator('input[aria-label*="Name Box"]').first
            if not nb.count():
                ed.wait_for_timeout(800)
                continue
            nb.click(timeout=4000)
            nb.fill(range_str)
            ed.keyboard.press("Enter")
            ed.wait_for_timeout(600)
            ed.evaluate(
                "async (s) => { try { await navigator.clipboard.writeText(s); }"
                " catch (e) {} }", sentinel)
            ed.wait_for_timeout(150)
            ed.keyboard.press("Control+c")
            txt = ""
            for _ in range(20):
                ed.wait_for_timeout(300)
                cur = ed.evaluate(
                    "async () => { try { return await navigator.clipboard.readText(); }"
                    " catch (e) { return ''; } }")
                if cur and cur.strip() and cur != sentinel:
                    txt = cur.replace("\r\n", "\n").replace("\r", "\n")
                    break
            if txt:
                return txt
        except Exception:  # noqa: BLE001
            pass
        ed.wait_for_timeout(800)
    return ""


def process_one(ctx, source, xlsx, range_str, watch=False):
    res = {"source": source, "status": "ok", "error": None, "applied": 0}
    page = ctx.new_page()
    ed = None
    try:
        before = {pp.url for pp in ctx.pages}
        page.goto("https://excel.cloud.microsoft/", wait_until="domcontentloaded",
                  timeout=90000)
        page.wait_for_timeout(5000)

        with page.expect_file_chooser(timeout=20000) as fc:
            if not try_click(page, SELECTORS["upload_button"]):
                raise RuntimeError("could not find 'Upload a file' button")
        fc.value.set_files(str(xlsx))

        for _ in range(30):
            page.wait_for_timeout(2000)
            handle_account_picker(ctx)
            ed = editor_tab(ctx, before)
            if ed:
                break
        if not ed:
            raise RuntimeError("workbook editor tab did not open")
        ed.bring_to_front()

        fr = officeapps_frame(ed)
        if not fr:
            raise RuntimeError("Excel app frame (officeapps) not found")
        wait_workbook_ready(ed)
        ed.wait_for_timeout(2500)

        if not open_clean_data(fr, ed):
            raise RuntimeError("could not open Clean Data")

        # Wait for analysis to finish (an Apply or Ignore button appears).
        deadline = time.time() + 90
        while time.time() < deadline:
            if (fr.locator(APPLY_BTN).count()
                    or fr.get_by_role("button", name="Ignore", exact=False).count()):
                break
            ed.wait_for_timeout(2000)
        ed.wait_for_timeout(1500)

        # Snapshot the grid (clipboard TSV) before and after applying, so the
        # diff is purely the cells Copilot changed -- no CSV-vs-Excel formatting
        # noise, and not limited to on-screen cells.
        before_tsv = capture_grid_tsv(ed, fr, range_str)
        res["applied"] = apply_all_cards(fr, ed)
        shot(ed, f"{source}__APPLIED")
        ed.wait_for_timeout(1500)
        after_tsv = capture_grid_tsv(ed, fr, range_str)
        (APPLIED_DIR / f"{source}.before.tsv").write_text(before_tsv, encoding="utf-8", newline="")
        (APPLIED_DIR / f"{source}.after.tsv").write_text(after_tsv, encoding="utf-8", newline="")
        res["before_chars"] = len(before_tsv)
        res["after_chars"] = len(after_tsv)
        if not after_tsv.strip():
            res["status"] = "error"
            res["error"] = "empty after snapshot"
    except Exception as e:  # noqa: BLE001
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
        shot(ed or page, f"{source}__APPLY_ERROR")
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


def already_applied():
    suffix = ".after.tsv"
    return {f.name[: -len(suffix)] for f in APPLIED_DIR.glob("*.after.tsv")
            if f.stat().st_size > 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process at most N tables")
    ap.add_argument("--only", default="",
                    help="only sources whose name contains this substring")
    ap.add_argument("--watch", action="store_true", help="headed + verbose menu dumps")
    args = ap.parse_args()

    s2x = source_to_xlsx()
    csvs = source_to_csv()
    sources = actionable_sources()
    if args.only:
        sources = [s for s in sources if args.only in s]
    done = already_applied()
    todo = [s for s in sources if s not in done and s in s2x]
    if args.limit:
        todo = todo[: args.limit]
    print(f"actionable tables: {len(sources)} | already applied: {len(done)} | "
          f"to do now: {len(todo)}", flush=True)

    with sync_playwright() as p:
        ctx = launch_context(p, headed=True)
        try:
            ctx.grant_permissions(["clipboard-read", "clipboard-write"])
        except Exception:  # noqa: BLE001
            pass
        n_ok = n_err = 0
        with open(APPLIED_RESULTS, "a", encoding="utf-8") as out:
            for i, src in enumerate(todo):
                rng = table_range(csvs.get(src)) or "A1:Z5000"
                r = process_one(ctx, src, s2x[src], rng, watch=args.watch)
                out.write(json.dumps(r) + "\n")
                out.flush()
                n_ok += r["status"] == "ok"
                n_err += r["status"] == "error"
                print(f"[{i+1}/{len(todo)}] {src[:42]:42} {r['status']:5} "
                      f"applied={r['applied']} {r['error'] or ''}", flush=True)
                time.sleep(2)
        ctx.close()
    print(f"\nDONE ok={n_ok} err={n_err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
