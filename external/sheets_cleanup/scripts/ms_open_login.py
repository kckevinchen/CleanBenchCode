"""Open the Microsoft 365 sign-in page in the CDP browser so the user can log in
via VNC. Also reports whether we already appear authenticated.

Run again after logging in to confirm: it navigates to excel.office.com and
checks we are not bounced to the login page.
"""

import sys

from harness import connect, new_page, shot


def main():
    p, browser, ctx = connect()
    page = new_page(ctx)
    try:
        page.goto("https://www.office.com/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(5000)
        url = page.url
        title = page.title()
        print("url:", url)
        print("title:", title)
        shot(page, "ms_login_state")
        signed_in = "login.microsoftonline.com" not in url and "/login" not in url.lower()
        # Stronger signal: presence of the app launcher / account button
        try:
            has_me = page.locator('#O365_MainLink_MePhoto, [aria-label*="Account manager"]').count() > 0
        except Exception:  # noqa: BLE001
            has_me = False
        print("looks_signed_in:", signed_in, "| account_widget:", has_me)
        print("\nIf not signed in: switch to the VNC window and log in to your "
              "Microsoft 365 account on this page.")
    finally:
        browser.close()
        p.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
