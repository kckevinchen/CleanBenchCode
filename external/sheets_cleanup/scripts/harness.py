"""Helpers shared by the Sheets-cleanup harness scripts.

Connects to the already-running, user-authenticated Chromium over CDP so we
never script the Google login itself.
"""

import os

from playwright.sync_api import sync_playwright

CDP_URL = os.environ.get("CDP_URL", "http://127.0.0.1:9222")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "outputs")
SHOTS = os.path.join(OUT, "shots")
os.makedirs(SHOTS, exist_ok=True)


def connect():
    """Return (playwright, browser, context) connected over CDP."""
    p = sync_playwright().start()
    browser = p.chromium.connect_over_cdp(CDP_URL)
    ctx = browser.contexts[0]
    return p, browser, ctx


def new_page(ctx):
    page = ctx.new_page()
    page.set_default_timeout(45000)
    return page


def shot(page, name):
    path = os.path.join(SHOTS, name if name.endswith(".png") else name + ".png")
    page.screenshot(path=path)
    return path
