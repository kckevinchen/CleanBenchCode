"""Step 1: log in to Microsoft 365 once, into the persistent browser profile.

    python scripts/login.py

A Chromium window opens at office.com. Sign in with your company M365 account
(the one that HAS Copilot). Solve any MFA. When you see the Office home page,
come back to this terminal and press Enter. Your login is saved in ./profile and
reused by every later step -- you should not need to log in again.
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

from _common import launch_context, shot


def main() -> int:
    with sync_playwright() as p:
        ctx = launch_context(p, headed=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://www.office.com/", wait_until="domcontentloaded", timeout=90000)
        print("\n" + "=" * 70)
        print("A browser window opened. Sign in to your COMPANY Microsoft 365")
        print("account (the one with Copilot). Complete any MFA prompts.")
        print("When you reach the Office home page, return here and press Enter.")
        print("=" * 70 + "\n")
        try:
            input("Press Enter once you are signed in... ")
        except EOFError:
            page.wait_for_timeout(120000)
        shot(page, "login_done")
        # Quick signal of success
        url = page.url
        print("current url:", url)
        print("login saved to ./profile" if "login.microsoftonline" not in url
              else "WARNING: still on a login URL -- you may not be fully signed in")
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
