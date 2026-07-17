# Google Sheets Cleanup Suggestions baseline

Black-box browser harness for capturing, parsing, and scoring Google Sheets
**Cleanup suggestions** against a local benchmark.

This integration requires an authenticated Google account and browser profile.
Review Google service terms and your organization's data policy before uploading
tables. Profiles, browser downloads, HTML captures, screenshots, outputs, and
source workbooks are ignored and must never be committed.

## Setup

```bash
python -m venv external/sheets_cleanup/venv
external/sheets_cleanup/venv/bin/pip install -r external/sheets_cleanup/requirements.txt
external/sheets_cleanup/venv/bin/playwright install chromium
```

Set `CHROME` if Playwright's Chromium binary is not at the launcher default:

```bash
CHROME=/path/to/chrome bash external/sheets_cleanup/scripts/launch_browser.sh
```

Authenticate through the browser/VNC session, build a local manifest, then run
the harness:

```bash
external/sheets_cleanup/venv/bin/python external/sheets_cleanup/scripts/build_manifest_v8.py
external/sheets_cleanup/venv/bin/python external/sheets_cleanup/scripts/run_all.py
external/sheets_cleanup/venv/bin/python external/sheets_cleanup/scripts/score.py
```

This package is an optional black-box product baseline, not part of the core
detector or workbench runtime.