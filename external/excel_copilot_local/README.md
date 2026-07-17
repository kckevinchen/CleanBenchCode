# Excel Copilot local harness (Windows)

Test **Copilot in Excel** ("clean up / find data-quality issues") against the
same 217-table benchmark we ran through Google Sheets Cleanup — but on your own
Windows machine, where your company M365 (Copilot-licensed) account can sign in.

It uploads each table's workbook to your OneDrive, opens it in Excel for the web,
asks Copilot to find data-quality issues, and **saves Copilot's response** so we
can parse and score it later. It does not apply any change to your data.

> Why local? The server/VM is blocked by Microsoft conditional-access. Your
> laptop is a trusted device, so login works there.

---

## Setup (one time)

1. **Install Python 3.10+** from https://www.python.org/downloads/ (check "Add
   python.exe to PATH" during install).
2. Open **PowerShell** in this folder and run:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m playwright install chromium
   ```
   If `Activate.ps1` is blocked, run once:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

3. **Convert the data** (CSV → .xlsx with Excel Tables, done locally):
   ```powershell
   python scripts\convert_to_xlsx.py
   ```

---

## Run

```powershell
# 1) Log in to M365 once (a browser opens; sign in, then press Enter here)
python scripts\login.py

# 2) Confirm THIS account has Copilot in Excel (prints PASS or FAIL)
python scripts\check_copilot.py

# 3) Try ONE table, visible and slow, to validate the flow
python scripts\run_excel_copilot.py --limit 1 --watch

# 4) If step 3 looks right, run the whole set (resumable)
python scripts\run_excel_copilot.py

# 5) Extract detections from the saved responses
python scripts\parse_response.py
```

Outputs land in `outputs/`:
- `html/<source>.html` — Copilot pane DOM per table
- `text/<source>.txt` — Copilot's answer text per table
- `shots/` — step-by-step screenshots (great for debugging)
- `results.jsonl` — one line per table (status, answer size)
- `findings.jsonl` / `findings.md` — parsed detections

The run is **resumable**: re-running skips tables already marked `ok`. If your
login expires, run `login.py` again and re-run.

---

## If a step fails

The Copilot UI selectors are **best-effort** (Microsoft's web UI changes and I
couldn't test against the real Copilot from the server). When `--watch` shows a
step failing:

1. Look at the latest `outputs/shots/*.png` to see where it stopped.
2. Open `scripts/run_excel_copilot.py`, find the `SELECTORS` dict, and fix the
   relevant entry.
3. Use `python scripts\record.py` to record the real clicks and read exact
   selectors (see `AGENTS.md`).

`AGENTS.md` in this folder is written for a coding agent (GitHub Copilot / Claude
in VS Code) — open this folder in VS Code and ask the agent to "harden the
Copilot selectors using AGENTS.md".

---

## Notes / caveats
- **Data handling:** you're copying benchmark CSVs to a company device and
  uploading them to your OneDrive. Confirm that's allowed by your org policy.
- **Copilot output is conversational**, not a tidy per-cell panel like Google's.
  Expect to tune `parse_response.py` once you see real answers.
- Keep volume reasonable; this drives a licensed product through its UI.
