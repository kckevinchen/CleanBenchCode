# AGENTS.md — Excel Copilot local harness

Instructions for a coding agent (GitHub Copilot / Claude in VS Code) helping the
user run and harden this harness on their **Windows** machine. Read this fully
before editing.

## What this project does
Drives **Copilot in Excel** (web) over the 217-table data-quality benchmark and
saves Copilot's responses for offline scoring. It is the Microsoft counterpart to
a finished Google Sheets "Cleanup suggestions" run (same benchmark). The end goal:
compare what Copilot flags against the gold benchmark, per cell.

Pipeline: **convert CSV→xlsx → log in once → verify Copilot → per-table
(upload → open → Copilot prompt → capture response) → parse → score.**

## Architecture & conventions
- **Stack:** Python + Playwright (sync API), Chromium, a *persistent* browser
  profile in `./profile` (so M365 login is done once, interactively).
- **`scripts/_common.py`** — shared paths + `launch_context()` (persistent,
  headed), `shot()`, `wait_stable_text()` (detects when a streamed Copilot answer
  stops growing). Reuse these; don't re-implement.
- **Headed, not headless.** Copilot must render and conditional-access is calmer
  with a visible browser. Keep `headed=True`.
- **Selectors are centralized** in `scripts/run_excel_copilot.py` → `SELECTORS`
  dict. This is the ONLY place to edit when the UI doesn't match. Each entry is a
  list tried in order; add new candidates to the front.
- **Excel web runs inside iframes.** Always search across `page.frames`, not just
  the top frame. Use the provided `find_in_frames()` helper.
- **Resumability:** `outputs/results.jsonl` records each table; reruns skip
  `status==ok`. Never clobber it; always append.
- **Capture-only:** never click Copilot's "apply"/"insert"/"keep" actions. We
  only read the response. Saving `outerHTML` + `inner_text` of the pane is enough.
- **Offline steps** (`convert_to_xlsx.py`, `parse_response.py`) must not need a
  browser and must be deterministic.

## Most likely things to fix (in priority order)
1. **Copilot pane input selector** (`SELECTORS["copilot_input"]`). Copilot's
   composer is often a `contenteditable` div, not a textarea. After `record.py`,
   replace with the exact selector. Submitting may need `Enter` or a Send button.
2. **Copilot ribbon button** (`SELECTORS["copilot_ribbon"]`). It may live in the
   Home ribbon or as a floating button; confirm the aria-label.
3. **Response-complete detection.** If answers are cut off, raise `idle_secs`/
   `max_secs` in the `wait_stable_text(...)` call, or detect the disappearance of
   the streaming/Stop button instead of text stability.
4. **Pane scrape root** (`SELECTORS["copilot_pane"]`). Make sure it's the answer
   container, not the whole sidebar (which includes the input box and history).
5. **Upload/open timing.** Big workbooks open slowly; bump the `wait_for_timeout`
   values in `process_one()` if the editor isn't ready.

## Hardening selectors with codegen
1. `python scripts\login.py` (once) so `./profile` is authenticated.
2. `python scripts\record.py` — opens Excel web with the Playwright Inspector.
3. Manually: upload one xlsx from `data/xlsx/`, open it, click Copilot, type a
   prompt, send. The Inspector shows robust selectors for each click.
4. Paste those selectors into the `SELECTORS` dict. Prefer role/aria-based
   selectors (`get_by_role`, `aria-label`) over brittle CSS classes.
5. Validate: `python scripts\run_excel_copilot.py --limit 1 --watch`.

## Definition of done for one table
`outputs/text/<source>.txt` contains a real Copilot answer (non-empty, mentions
cells/values), `outputs/html/<source>.html` has the pane DOM, and
`results.jsonl` shows `status=ok`. Then `parse_response.py` yields cell-level
detections in `findings.jsonl`.

## Scoring (later, optional)
The benchmark gold is cell-level. To score, map Copilot's cell refs to benchmark
cells. **Excel cell row N corresponds to benchmark 0-based data row N-2** (row 1
is the header; first data row is Excel row 2). Column letter → header name via
`data/manifest_xlsx.json`. Mirror the Google run's metric (cell-level recall /
micro-F1). Ask the user for the benchmark file if scoring locally.

## Guardrails
- Don't switch to headless. Don't store credentials in code. Don't auto-apply
  Copilot edits. Don't delete `results.jsonl`, `profile/`, or saved outputs.
- Keep changes surgical and within `scripts/`. If a fix balloons, ask the user.
