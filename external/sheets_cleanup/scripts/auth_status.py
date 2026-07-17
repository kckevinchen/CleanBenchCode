"""Read-only auth status: list tabs + check Google auth cookies. Never navigates."""

import sys

from playwright.sync_api import sync_playwright

AUTH = {"SID", "SSID", "HSID", "SAPISID", "__Secure-1PSID", "__Secure-3PSID"}


def main():
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        ctx = b.contexts[0]
        print("open tabs:")
        for pg in ctx.pages:
            try:
                print(f"  - {pg.url}")
            except Exception as e:  # noqa: BLE001
                print("  - (err)", e)
        cookies = ctx.cookies()
        present = sorted(AUTH & {c["name"] for c in cookies})
        print("\nauth cookies:", present if present else "NONE")
        print("logged_in:", bool(present))
    return 0


if __name__ == "__main__":
    sys.exit(main())
