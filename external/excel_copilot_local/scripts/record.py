"""Helper: open Playwright's codegen recorder against Excel for the web, using
your saved login profile. Click through the Copilot flow once and codegen prints
the exact selectors -- paste them into run_excel_copilot.py's SELECTORS dict.

    python scripts/record.py

(See AGENTS.md "Hardening selectors" for the workflow.)
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

from _common import launch_context


def main() -> int:
    with sync_playwright() as p:
        ctx = launch_context(p, headed=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://excel.cloud.microsoft/", wait_until="domcontentloaded",
                  timeout=90000)
        try:
            # Available in recent Playwright: opens the Inspector/recorder.
            page.pause()
        except Exception as e:  # noqa: BLE001
            print("page.pause() unavailable:", e)
            print("Alternatively run:  python -m playwright codegen "
                  "--load-storage=. https://excel.cloud.microsoft/")
            input("Press Enter to close...")
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
