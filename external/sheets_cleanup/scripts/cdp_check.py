"""Connect to the launched Chromium over CDP, report state, screenshot.

Usage: python scripts/cdp_check.py [--goto URL] [--shot PATH]
"""

import argparse
import os
import sys

from playwright.sync_api import sync_playwright

CDP = os.environ.get("CDP_URL", "http://127.0.0.1:9222")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goto", default=None)
    ap.add_argument("--shot", default=os.path.join(OUT, "cdp_state.png"))
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP)
        ctx = browser.contexts[0]
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if args.goto:
            page.goto(args.goto, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2500)
        print("contexts:", len(browser.contexts))
        print("pages:", [pg.url for pg in ctx.pages])
        print("active url:", page.url)
        print("active title:", page.title())
        os.makedirs(os.path.dirname(args.shot), exist_ok=True)
        page.screenshot(path=args.shot, full_page=False)
        print("screenshot:", os.path.abspath(args.shot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
