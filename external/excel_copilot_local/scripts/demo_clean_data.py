"""DEMO: end-to-end 'Clean Data with Copilot' automation for a SINGLE table.

This is a presenter-friendly walkthrough of the whole harness for one CSV. It
narrates each step to the console as it happens, so you can watch the browser do
the work and follow along:

    CSV -> .xlsx (offline)  ->  upload to Excel for the web  ->  open the
    Clean Data pane  ->  capture Copilot's suggestions + cell highlights  ->
    parse them into cell-level findings  ->  click Apply and show the
    before -> after value of every changed cell.

Run it (use the project venv, which has Playwright + openpyxl + bs4):

    .\\venv\\Scripts\\python.exe scripts\\demo_clean_data.py                 # default demo table
    .\\venv\\Scripts\\python.exe scripts\\demo_clean_data.py --source 01c464d # pick by substring
    .\\venv\\Scripts\\python.exe scripts\\demo_clean_data.py --csv path\\to\\any.csv
    .\\venv\\Scripts\\python.exe scripts\\demo_clean_data.py --no-apply       # capture only, never Apply

Prereqs (same as the batch harness): you have already run scripts\\login.py once
so ./edge_dbg holds your M365 login, and scripts\\check_copilot.py printed PASS.

By default it then clicks Apply on every suggestion so the audience sees Copilot
actually fix the data, and prints the before -> after value for each changed
cell. This only mutates the throwaway copy that was just uploaded -- your local
CSV is never touched. Pass --no-apply to stop after capturing (never clicks
Apply). Artifacts land in outputs/demo/ (kept separate from the real batch
outputs), and the Edge editor tab is left open at the end so you can point at it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from _common import OUT, ROOT, launch_context, shot
from convert_to_xlsx import convert_one, safe_table_name
from parse_response import (
    col_to_index,
    match_values,
    measure_row_height,
    parse_card,
    read_csv_rows,
    rects_to_excel_rows,
    resolve_cells,
)
from run_excel_copilot import (
    SELECTORS,
    capture_clean_data,
    editor_tab,
    handle_account_picker,
    officeapps_frame,
    open_clean_data,
    try_click,
    wait_workbook_ready,
)
from apply_and_capture import apply_all_cards, capture_grid_tsv, table_range
from diff_applied import index_to_col as tsv_col, norm as tsv_norm, parse_tsv

try:
    from bs4 import BeautifulSoup
except Exception:  # noqa: BLE001
    BeautifulSoup = None

DEMO_DIR = OUT / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# A known-good default: this table has an easy-to-explain phone-number format
# inconsistency in column E ("Engineer Phone"), so the demo reliably shows a
# result. Override with --source / --csv.
DEFAULT_SOURCE = "01c464d"


# --- pretty console narration ------------------------------------------------

def banner(step: str, title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n  {step}  {title}\n{line}", flush=True)


def info(msg: str) -> None:
    print(f"   {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"   [ok] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"   [!]  {msg}", flush=True)


def preview_csv(rows, n_rows: int = 5, n_cols: int = 6, width: int = 16) -> None:
    """Print a small, truncated grid so the audience sees the raw data."""
    def cell(v):
        v = (v or "").replace("\n", " ")
        return (v[: width - 1] + "\u2026") if len(v) > width else v.ljust(width)

    for r in rows[: n_rows + 1]:  # header + n_rows
        print("   | " + " | ".join(cell(c) for c in r[:n_cols])
              + (" | \u2026" if len(r) > n_cols else " |"), flush=True)
    if len(rows) > n_rows + 1:
        info(f"... ({len(rows) - 1} data rows total)")


# --- target selection --------------------------------------------------------

def load_manifest_sources():
    """source (csv name) -> {'header': [...], 'csv': Path} from every batch."""
    info_map = {}
    for man_path in sorted(ROOT.glob("*/manifest_xlsx.json")):
        csv_dir = man_path.parent / "csv"
        try:
            for rec in json.loads(man_path.read_text()):
                info_map.setdefault(rec["source"], {
                    "header": rec.get("header", []),
                    "csv": csv_dir / rec["source"],
                })
        except Exception:  # noqa: BLE001
            continue
    return info_map


def resolve_target(args):
    """Return (source_name, csv_path). Supports an arbitrary --csv or a --source
    substring match against the merged manifests."""
    if args.csv:
        p = Path(args.csv)
        if not p.exists():
            print(f"ERROR: --csv file not found: {p}")
            sys.exit(2)
        return p.name, p

    sources = load_manifest_sources()
    if not sources:
        print("ERROR: no */manifest_xlsx.json found. Run scripts/convert_to_xlsx.py first.")
        sys.exit(2)

    needle = args.source or DEFAULT_SOURCE
    matches = [s for s in sources if needle in s]
    if not matches:
        # Fall back to the first table so the demo still runs.
        first = sorted(sources)[0]
        warn(f"no source matched '{needle}'; using first table instead: {first}")
        return first, sources[first]["csv"]
    src = sorted(matches)[0]
    if len(matches) > 1:
        info(f"'{needle}' matched {len(matches)} tables; using {src}")
    return src, sources[src]["csv"]


# --- finding formatting ------------------------------------------------------

def cells_for_card(rec, rects, row_h, crows):
    """Map one parsed card to affected cell refs, preferring Excel's own
    highlight overlays (ground truth) with a value sanity check, and falling
    back to locating the value(s) in the source column. Mirrors parse_response."""
    ci = col_to_index(rec.get("column"))
    hrows = rects_to_excel_rows(rects, row_h)
    hl_cells, hl_vals = [], set()
    for er in hrows:
        hl_cells.append(f"{rec.get('column') or ''}{er}")
        if ci is not None and crows and 0 <= er - 1 < len(crows) and ci < len(crows[er - 1]):
            hl_vals.add(crows[er - 1][ci])
    shown = set(match_values(rec))
    if hrows and shown:
        if rec.get("type") == "Typo suggestion":
            sanity = any(re.search(r"\b" + re.escape(w) + r"\b", v)
                         for w in shown for v in hl_vals)
        else:
            sanity = bool(shown & hl_vals)
    else:
        sanity = None
    if hl_cells and sanity is not False:
        return hl_cells, "highlight"
    return resolve_cells(rec, crows), "value"


def detail_line(rec):
    """A short 'here's the fix' line tailored to the suggestion type."""
    t = rec.get("type", "")
    if t == "Typo suggestion" and rec.get("current"):
        return f"Replace '{rec['current']}' \u2192 '{rec.get('suggested', '')}'"
    if t == "Inconsistent pattern":
        ex = rec.get("examples", [])
        if len(ex) >= 2:
            return f"Align pattern, e.g. '{ex[0]}' \u2192 '{ex[1]}'"
    if t == "Inconsistent text" and rec.get("variations"):
        return "Unify variations: " + ", ".join(f"'{v}'" for v in rec["variations"])
    if t == "Inconsistent order" and rec.get("values"):
        return "Reorder values: " + ", ".join(f"'{v}'" for v in rec["values"])
    return ""


def show_findings(cards_data, header, crows):
    """Parse every captured card and print a readable, cell-level summary."""
    if BeautifulSoup is None:
        warn("beautifulsoup4 not installed; skipping structured parse.")
        return 0

    row_h = measure_row_height([rc for c in cards_data for rc in c.get("rects", [])])
    total_cells = 0
    n = 0
    for c in cards_data:
        el = BeautifulSoup(c.get("html", ""), "html.parser").select_one(".suggestion-card")
        if el is None:
            continue
        rec = parse_card(el)
        ci = col_to_index(rec.get("column"))
        hdr = header[ci] if (ci is not None and 0 <= ci < len(header)) else None
        cells, source = cells_for_card(rec, c.get("rects", []), row_h, crows)
        total_cells += len(cells)
        n += 1

        col = rec.get("column")
        where = f"Column {col} ({hdr})" if col and hdr else (
            f"Column {col}" if col else "Whole table")
        print(f"\n   [{n}] {rec.get('type', 'Suggestion')}  \u2014  {where}", flush=True)
        if rec.get("text"):
            info(f"      Copilot: {rec['text'].strip()}")
        d = detail_line(rec)
        if d:
            info(f"      Fix:     {d}")
        if cells:
            shown = ", ".join(cells[:12]) + (" ..." if len(cells) > 12 else "")
            info(f"      Cells:   {len(cells)} affected [{source}]  {shown}")
        else:
            info(f"      Scope:   {rec.get('scope', 'column')} (no specific cells)")
    return total_cells


# --- apply + before/after diff -----------------------------------------------

def diff_grids(before_tsv, after_tsv):
    """Per-cell (A1, before, after) tuples where a cell changed. Both snapshots
    come from Excel's own clipboard, so a cell differs only when Apply changed
    it (no CSV-vs-Excel formatting noise). Mirrors diff_applied.py."""
    before, after = parse_tsv(before_tsv), parse_tsv(after_tsv)
    changes = []
    for r in range(max(len(before), len(after))):
        brow = before[r] if r < len(before) else []
        arow = after[r] if r < len(after) else []
        for c in range(max(len(brow), len(arow))):
            bv = tsv_norm(brow[c]) if c < len(brow) else ""
            av = tsv_norm(arow[c]) if c < len(arow) else ""
            if bv != av:
                changes.append((f"{tsv_col(c)}{r + 1}", bv, av))  # TSV row 0 == Excel row 1
    return changes


def apply_and_show(ed, fr, source, csv_path):
    """Snapshot the grid, click Apply on every card, snapshot again, and print
    the before -> after value for each changed cell. Mutates only the throwaway
    uploaded copy (the local CSV is never touched)."""
    banner("STEP 6/6", "Apply Copilot's fixes and show before -> after")
    rng = table_range(csv_path) or "A1:Z5000"
    info(f"Snapshotting the grid (range {rng}) before applying ...")
    before_tsv = capture_grid_tsv(ed, fr, rng)
    if not before_tsv.strip():
        warn("could not read the grid before applying; skipping the apply step.")
        return
    info("Clicking Apply on every suggestion (mutates only this uploaded copy) ...")
    n = apply_all_cards(fr, ed)
    ok(f"Applied {n} suggestion(s); the pane now shows no remaining suggestions.")
    shot(ed, f"demo_{source}__06_applied")
    ed.wait_for_timeout(1500)
    info("Snapshotting the grid after applying ...")
    after_tsv = capture_grid_tsv(ed, fr, rng)
    (DEMO_DIR / f"{source}.before.tsv").write_text(before_tsv, encoding="utf-8", newline="")
    (DEMO_DIR / f"{source}.after.tsv").write_text(after_tsv, encoding="utf-8", newline="")

    changes = diff_grids(before_tsv, after_tsv)
    if not changes:
        warn("no cell changes detected (Copilot's Apply made no net change).")
        return
    print("", flush=True)
    ok(f"Copilot changed {len(changes)} cell(s):")
    for cell, bv, av in changes[:25]:
        info(f"   {cell:>6}:  '{bv}'  \u2192  '{av}'")
    if len(changes) > 25:
        info(f"   ... and {len(changes) - 25} more (see {source}.after.tsv)")


# --- the demo flow ------------------------------------------------------------

def run_demo(ctx, source, xlsx, header, crows, csv_path, do_apply, keep_open,
             close_editor):
    page = ctx.new_page()
    ed = None
    try:
        banner("STEP 2/6", "Upload the workbook to Excel for the web")
        before = {pp.url for pp in ctx.pages}
        info("Opening https://excel.cloud.microsoft/ ...")
        page.goto("https://excel.cloud.microsoft/", wait_until="domcontentloaded",
                  timeout=90000)
        page.wait_for_timeout(5000)
        shot(page, f"demo_{source}__01_landing")

        info("Clicking 'Upload a file' and selecting the .xlsx ...")
        with page.expect_file_chooser(timeout=20000) as fc_info:
            if not try_click(page, SELECTORS["upload_button"]):
                raise RuntimeError("could not find the 'Upload a file' button")
        fc_info.value.set_files(str(xlsx))
        ok("Upload started; waiting for the workbook to open ...")

        # The workbook opens in a new tab, often via an AAD 'Pick an account' page.
        for _ in range(30):
            page.wait_for_timeout(2000)
            handle_account_picker(ctx)
            ed = editor_tab(ctx, before)
            if ed:
                break
        if not ed:
            raise RuntimeError("the workbook editor tab did not open")
        ed.bring_to_front()
        shot(ed, f"demo_{source}__02_opening")

        fr = officeapps_frame(ed)
        if not fr:
            raise RuntimeError("Excel app frame (officeapps) not found")
        info("Workbook is opening; waiting for it to finish loading ...")
        wait_workbook_ready(ed)
        ed.wait_for_timeout(2500)
        ok("Workbook is open in Excel for the web.")

        banner("STEP 3/6", "Open Data > Clean Data (the Copilot pane)")
        shot(ed, f"demo_{source}__03_ribbon")
        if not open_clean_data(fr, ed):
            raise RuntimeError("could not open Clean Data (Data > Data Tools > Clean Data)")
        ok("'Clean Data with Copilot' pane is open.")
        shot(ed, f"demo_{source}__04_clean_pane")

        banner("STEP 4/6", "Let Copilot analyze and capture its suggestions")
        info("Waiting for analysis to finish and selecting each card to")
        info("highlight the exact cells it flags (nothing is applied yet) ...")
        text, html, cards_data = capture_clean_data(fr, ed)
        shot(ed, f"demo_{source}__05_captured")

        (DEMO_DIR / f"{source}.txt").write_text(text, encoding="utf-8")
        (DEMO_DIR / f"{source}.html").write_text(html or ed.content(), encoding="utf-8")
        (DEMO_DIR / f"{source}.cells.json").write_text(
            json.dumps({"source": source, "cards": cards_data}, ensure_ascii=False),
            encoding="utf-8")
        n_cards = len(cards_data)
        n_hl = sum(len(c.get("rects", [])) for c in cards_data)
        ok(f"Captured pane text ({len(text)} chars), {n_cards} suggestion card(s), "
           f"{n_hl} highlight rect(s).")

        banner("STEP 5/6", "Parse Copilot's response into cell-level findings")
        applied = False
        if n_cards == 0:
            ok("Copilot found no data-quality issues in this table (it looks clean).")
        else:
            total = show_findings(cards_data, header, crows)
            print("", flush=True)
            ok(f"{n_cards} suggestion(s) mapped to {total} affected cell(s).")

        if do_apply and n_cards:
            apply_and_show(ed, fr, source, csv_path)
            applied = True
        elif not do_apply:
            banner("STEP 6/6", "Apply (skipped)")
            info("--no-apply was set, so Copilot's fixes were NOT applied.")

        print(f"\n   Artifacts saved under: {DEMO_DIR}", flush=True)
        info(f"pane text  : {DEMO_DIR / (source + '.txt')}")
        info(f"pane DOM   : {DEMO_DIR / (source + '.html')}")
        info(f"cards+cells: {DEMO_DIR / (source + '.cells.json')}")
        if applied:
            info(f"before TSV : {DEMO_DIR / (source + '.before.tsv')}")
            info(f"after TSV  : {DEMO_DIR / (source + '.after.tsv')}")
        info(f"screenshots: {OUT / 'shots'}  (demo_{source}__*.png)")

        if close_editor:
            try:
                ed.close()
            except Exception:  # noqa: BLE001
                pass
        elif keep_open:
            print("\n   The Edge editor tab is left open so you can inspect the "
                  "result.\n   Close it manually when you're done.", flush=True)
    except Exception as e:  # noqa: BLE001
        shot(ed or page, f"demo_{source}__ERROR")
        warn(f"{type(e).__name__}: {e}")
        warn("See the latest outputs/shots/demo_*__*.png to find where it stopped,")
        warn("then harden the SELECTORS in scripts/run_excel_copilot.py (see AGENTS.md).")
        raise
    finally:
        # Leave the editor tab as decided above; only close the throwaway
        # 'Upload a file' landing tab.
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="One-table Clean Data demo.")
    ap.add_argument("--source", default="",
                    help=f"substring of a benchmark CSV name (default: {DEFAULT_SOURCE})")
    ap.add_argument("--csv", default="",
                    help="path to any CSV to demo instead of a benchmark table")
    ap.add_argument("--close-editor", action="store_true",
                    help="close the Excel editor tab when done (default: keep open)")
    ap.add_argument("--no-apply", dest="apply", action="store_false",
                    help="capture only; do NOT click Apply (default: apply the fixes)")
    ap.set_defaults(apply=True)
    args = ap.parse_args()

    banner("STEP 1/6", "Pick a table and convert CSV -> .xlsx (offline)")
    source, csv_path = resolve_target(args)
    info(f"Table:  {source}")
    info(f"CSV:    {csv_path}")

    rows = read_csv_rows(csv_path)
    if not rows:
        print(f"ERROR: could not read CSV rows from {csv_path}")
        return 2
    header = rows[0]
    info(f"Shape:  {len(rows) - 1} data rows x {len(header)} columns")
    print("   Preview:", flush=True)
    preview_csv(rows)

    xlsx = DEMO_DIR / (Path(source).stem + ".xlsx")
    n_data, n_cols = convert_one(csv_path, xlsx, safe_table_name(0))
    ok(f"Wrote Excel Table workbook: {xlsx.name}  ({n_data} rows x {n_cols} cols)")

    t0 = time.time()
    with sync_playwright() as p:
        info("Connecting to your logged-in Edge (over CDP) ...")
        ctx = launch_context(p, headed=True)
        if args.apply:
            # The apply-step grid snapshots read the OS clipboard; grant once.
            try:
                ctx.grant_permissions(["clipboard-read", "clipboard-write"])
            except Exception:  # noqa: BLE001
                pass
        try:
            run_demo(ctx, source, xlsx, header, rows, csv_path,
                     do_apply=args.apply,
                     keep_open=not args.close_editor,
                     close_editor=args.close_editor)
        finally:
            # Disconnect Playwright only; Edge (and any kept-open tab) stays up.
            ctx.close()
    banner("DONE", f"Demo finished in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
