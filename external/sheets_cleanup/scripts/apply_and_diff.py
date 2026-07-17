"""Apply Google Sheets 'Cleanup suggestions' and capture the sheet before/after,
so we can recover the SUGGESTED FIX for every changed cell by diffing.

The panel scrape (run_all.py) only records what the suggestion cards DISPLAY,
which for most categories is not the actual replacement value (WHITESPACE shows a
space marker, DUPLICATE ROWS/INVALID DATA show nothing, COLUMN FORMAT shows a
display preview). The real fix is only realised when you click the accept button
on a card. So, per table, this script:

  1. imports the CSV into a fresh Google Sheet,
  2. exports the sheet as CSV (BEFORE snapshot),
  3. opens Data > Data cleanup > Cleanup suggestions,
  4. clicks the accept button on every actionable card EXCEPT 'Remove'
     (duplicate-row removal is relational and would shift every row below it,
     breaking the cell-aligned diff; those rows are already captured separately),
  5. exports the sheet again as CSV (AFTER snapshot),
  6. saves both to outputs/applied/<source>.{before,after}.csv.

Offline, scripts/diff_cleanup.py diffs the two snapshots to emit per-cell fixes.

Both snapshots come from Google's own CSV export of the same sheet, so a cell
differs only when Apply actually changed it. Resumable: tables whose
outputs/applied/<source>.after.csv already exists (non-empty) are skipped.

    python scripts/apply_and_diff.py --only 0605e88a --limit 1   # validate one
    python scripts/apply_and_diff.py                             # all w/ suggestions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback

from harness import OUT, connect, new_page, shot
from parse_cleanup_html import parse_html
from run_all import import_csv, run_cleanup, trash_sheet

MANIFEST = os.path.join(OUT, "manifest.json")
APPLIED_DIR = os.path.join(OUT, "applied")
RESULTS = os.path.join(OUT, "apply_results.jsonl")
HTML_DIR = os.path.join(OUT, "html")
os.makedirs(APPLIED_DIR, exist_ok=True)

ACCEPT_BTN = ".waffle-inline-assistant-accept-button"
NF_CHECK = ".waffle-inline-assistant-nf-check"
# Accept-button labels we do NOT click: row removal shifts every row below it,
# which would misalign the before/after cell diff. Duplicate rows are relational
# and already captured by the panel scrape, so we exclude them here.
SKIP_ACTIONS = {"remove", "delete"}


def sheet_ids_from_url(url):
    """(sheet_id, gid) from a Sheets edit URL; gid defaults to 0."""
    sheet_id = url.split("/d/")[1].split("/")[0]
    gid = "0"
    if "gid=" in url:
        gid = url.split("gid=")[1].split("&")[0].split("#")[0]
    return sheet_id, gid


def index_to_col(idx):
    """0-based column index -> letter (0 -> A, 26 -> AA)."""
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def table_range(csv_path):
    """Exact A1 range covering the table (header + data), from its source CSV."""
    try:
        with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.reader(fh))
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    ncols = max((len(r) for r in rows), default=1)
    return f"A1:{index_to_col(max(0, ncols - 1))}{len(rows)}"


def capture_grid_tsv(page, range_str, tries=3):
    """Select an explicit A1 range via the Name Box and Ctrl+C it, then read the
    clipboard as TSV.

    We overwrite the OS clipboard with a sentinel first, press Ctrl+C, then poll
    until the clipboard no longer holds the sentinel -- Sheets' async clipboard
    write lags the keystroke, so without this the AFTER snapshot could read the
    stale BEFORE copy. Ctrl+C copies the DISPLAYED cell text, so number-format
    changes (COLUMN FORMAT) show up in the diff. Returns TSV text."""
    sentinel = "__HARNESS_CLIP__"
    for _ in range(tries):
        try:
            nb = page.locator("#t-name-box")
            if not nb.count():
                nb = page.locator(".waffle-name-box")
            if not nb.count():
                page.wait_for_timeout(800)
                continue
            nb.first.click(timeout=4000)
            nb.first.fill(range_str)
            page.keyboard.press("Enter")
            page.wait_for_timeout(600)
            page.evaluate(
                "async (s) => { try { await navigator.clipboard.writeText(s); }"
                " catch (e) {} }", sentinel)
            page.wait_for_timeout(150)
            page.keyboard.press("Control+c")
            for _ in range(20):
                page.wait_for_timeout(300)
                cur = page.evaluate(
                    "async () => { try { return await navigator.clipboard.readText(); }"
                    " catch (e) { return ''; } }")
                if cur and cur.strip() and cur != sentinel:
                    return cur.replace("\r\n", "\n").replace("\r", "\n")
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(800)
    return ""


def _actionable_buttons(page):
    """Visible accept controls to click, as (locator, label). Two kinds:
    - `.waffle-inline-assistant-accept-button` (WHITESPACE 'Trim all', DUPLICATE
      ROWS 'Remove', ...): clicked unless the label is a skip action.
    - `.waffle-inline-assistant-nf-check` (COLUMN FORMAT accept, an icon with no
      text): always actionable, labelled 'Apply formatting'."""
    out = []
    btns = page.locator(ACCEPT_BTN)
    try:
        n = btns.count()
    except Exception:  # noqa: BLE001
        n = 0
    for i in range(n):
        b = btns.nth(i)
        try:
            if not b.is_visible():
                continue
            label = (b.inner_text() or "").strip()
        except Exception:  # noqa: BLE001
            continue
        if label.lower() in SKIP_ACTIONS:
            continue
        out.append((b, label or "Apply"))
    nf = page.locator(NF_CHECK)
    try:
        m = nf.count()
    except Exception:  # noqa: BLE001
        m = 0
    for i in range(m):
        b = nf.nth(i)
        try:
            if not b.is_visible():
                continue
        except Exception:  # noqa: BLE001
            continue
        out.append((b, "Apply formatting"))
    return out


def apply_cards(page, max_apply=200):
    """Click the accept button on each actionable (non-skip) card until none
    remain. Each apply re-renders the panel (the applied card disappears), so we
    always click the first remaining actionable button. Stops if no progress is
    made, so skipped 'Remove' cards never cause an infinite loop."""
    applied = 0
    actions = []
    stale = 0
    for _ in range(max_apply):
        cands = _actionable_buttons(page)
        if not cands:
            break
        before_n = len(cands)
        btn, label = cands[0]
        try:
            btn.scroll_into_view_if_needed(timeout=3000)
        except Exception:  # noqa: BLE001
            pass
        ok = False
        try:
            btn.click(timeout=6000)
            ok = True
        except Exception:  # noqa: BLE001
            try:
                btn.evaluate("e => e.click()")
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
        if ok:
            applied += 1
            actions.append(label)
        page.wait_for_timeout(1600)
        # progress check: actionable count should drop after a successful apply
        after_n = len(_actionable_buttons(page))
        if after_n >= before_n:
            stale += 1
            if stale >= 2:
                break
        else:
            stale = 0
    return applied, actions


def process_one(ctx, rec, trash):
    source = rec["source"]
    page = new_page(ctx)
    page.set_default_timeout(20000)
    res = {"source": source, "status": "ok", "sheet_id": None, "applied": 0,
           "actions": [], "categories": [], "before_chars": 0, "after_chars": 0,
           "changed": None, "error": None}
    rng = table_range(rec["csv_path"]) or "A1:Z5000"
    try:
        page.goto("https://sheets.new", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(8000)
        if "accounts.google.com" in page.url:
            raise RuntimeError("redirected to login -- session expired")
        sheet_id, _gid = sheet_ids_from_url(page.url)
        res["sheet_id"] = sheet_id

        import_csv(page, rec["csv_path"])
        page.wait_for_timeout(1500)

        before_tsv = capture_grid_tsv(page, rng)
        if not before_tsv.strip():
            raise RuntimeError("empty BEFORE snapshot")
        res["before_chars"] = len(before_tsv)

        run_cleanup(page)
        # record which categories were present, for reporting
        side = page.locator(".waffle-inline-assistant-sidebar")
        if side.count():
            try:
                parsed = parse_html(side.first.evaluate("el => el.outerHTML"))
                res["categories"] = sorted({s["category"] for s in parsed["suggestions"]})
            except Exception:  # noqa: BLE001
                pass

        res["applied"], res["actions"] = apply_cards(page)
        page.wait_for_timeout(1500)
        shot(page, f"applied_{source[:40]}")

        after_tsv = capture_grid_tsv(page, rng)
        if not after_tsv.strip():
            raise RuntimeError("empty AFTER snapshot")
        res["after_chars"] = len(after_tsv)
        res["changed"] = before_tsv != after_tsv

        with open(os.path.join(APPLIED_DIR, source + ".before.tsv"), "w",
                  encoding="utf-8", newline="") as f:
            f.write(before_tsv)
        with open(os.path.join(APPLIED_DIR, source + ".after.tsv"), "w",
                  encoding="utf-8", newline="") as f:
            f.write(after_tsv)

        if trash:
            trash_sheet(page)
    except Exception as e:  # noqa: BLE001
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
        try:
            shot(page, f"applyerr_{source[:40]}")
        except Exception:  # noqa: BLE001
            pass
        traceback.print_exc()
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass
    return res


def already_done():
    done = set()
    for f in os.listdir(APPLIED_DIR):
        if f.endswith(".after.tsv") and os.path.getsize(os.path.join(APPLIED_DIR, f)) > 0:
            done.add(f[: -len(".after.tsv")])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--only", default="",
                    help="only sources whose name contains this substring")
    ap.add_argument("--no-trash", action="store_true")
    ap.add_argument("--only-with-suggestions", action="store_true",
                    help="restrict to tables that had >=1 cleanup suggestion in the "
                         "prior run (from run_results.jsonl)")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    if args.only:
        man = [m for m in man if args.only in m["source"]]
    if args.only_with_suggestions:
        keep = set()
        rr = os.path.join(OUT, "run_results.jsonl")
        if os.path.exists(rr):
            for line in open(rr):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("n_suggestions", 0) > 0:
                        keep.add(r["source"])
                except Exception:  # noqa: BLE001
                    pass
        man = [m for m in man if m["source"] in keep]
    man = man[args.start:]
    if args.limit:
        man = man[: args.limit]

    done = already_done()
    print(f"tables to consider: {len(man)} | already applied: {len(done)}", flush=True)

    p, browser, ctx = connect()
    try:
        ctx.grant_permissions(["clipboard-read", "clipboard-write"])
    except Exception:  # noqa: BLE001
        pass
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
                n_ok += r["status"] == "ok"
                n_err += r["status"] == "error"
                el = time.time() - t0
                rate = el / max(n_ok + n_err, 1)
                print(f"[{i+1}/{len(man)}] {src[:40]:40} {r['status']:5} "
                      f"applied={r['applied']} acts={','.join(r['actions']) or '-'} "
                      f"({rate:.0f}s/tbl)", flush=True)
                if r["status"] == "error" and r["error"] and "session expired" in r["error"]:
                    print("ABORTING: Google session expired. Re-login and rerun.", flush=True)
                    break
                time.sleep(2)
    finally:
        browser.close()
        p.stop()
    print(f"\nDONE ok={n_ok} err={n_err} skip={n_skip} elapsed={time.time()-t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
