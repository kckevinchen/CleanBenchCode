"""Shared helpers for the local Excel + Copilot harness.

Unlike the server-side Google Sheets harness (which attached to a browser over
CDP), this runs Playwright directly on your Windows machine and uses a
*persistent browser profile* so you log in to Microsoft 365 only once.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE_PROFILE = ROOT / "edge_dbg"    # persistent Edge profile dir (holds your M365 login)
# DATA and OUT roots are overridable so extra CSV batches (e.g. data_remaining/)
# can be run while combining results into one outputs/ folder. For example:
#   $env:HARNESS_DATA = "C:\excel_copilot_local\data_remaining"
# leaves HARNESS_OUT unset, so all results still land in the shared outputs/.
DATA = Path(os.environ.get("HARNESS_DATA") or (ROOT / "data"))
CSV_DIR = DATA / "csv"
XLSX_DIR = DATA / "xlsx"
OUT = Path(os.environ.get("HARNESS_OUT") or (ROOT / "outputs"))
SHOTS = OUT / "shots"
HTML_DIR = OUT / "html"
TEXT_DIR = OUT / "text"

for _d in (EDGE_PROFILE, XLSX_DIR, OUT, SHOTS, HTML_DIR, TEXT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Drive Microsoft Edge over CDP (managed-device friendly) ----------------
# This machine is org-managed: Microsoft conditional-access only admits a
# normally-launched Edge (Windows SSO), and Playwright's own
# launch_persistent_context(channel="msedge") is dropped by Edge's startup
# relaunch. So we launch Edge ourselves with a remote-debugging port on a
# dedicated, NON-default user-data-dir (the default dir forbids remote
# debugging), then connect Playwright to it over CDP. Login persists in
# EDGE_PROFILE so you only sign in once.
EDGE_PORT = int(os.environ.get("PW_EDGE_PORT", "9222"))

_EDGE_EXE_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def _edge_exe() -> str:
    for c in _EDGE_EXE_CANDIDATES:
        if Path(c).exists():
            return c
    raise RuntimeError("msedge.exe not found; install Microsoft Edge.")


def _debug_port_up(port: int = EDGE_PORT) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/json/version", timeout=2
        ) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_edge(port: int = EDGE_PORT, profile: Path = EDGE_PROFILE,
               wait_secs: float = 30.0) -> None:
    """Ensure an Edge with remote debugging is up on `port`.

    Reuses an already-running one; otherwise launches a detached Edge on a
    dedicated profile so the M365 login persists between steps.
    """
    if _debug_port_up(port):
        return
    profile.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    subprocess.Popen(
        [
            _edge_exe(),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--start-maximized",
            "about:blank",
        ],
        creationflags=flags,
        close_fds=True,
    )
    deadline = time.time() + wait_secs
    while time.time() < deadline:
        if _debug_port_up(port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Edge did not expose debug port {port} within {wait_secs}s.")


class _CtxProxy:
    """Wraps the CDP default BrowserContext so existing callers keep working.

    Every attribute proxies to the real context; only .close() is overridden to
    merely disconnect Playwright (Edge keeps running, so the login persists for
    the next step). The user closes Edge manually when the whole run is done.
    """

    def __init__(self, browser, ctx):
        object.__setattr__(self, "_browser", browser)
        object.__setattr__(self, "_ctx", ctx)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_ctx"), name)

    def close(self):
        try:
            object.__getattribute__(self, "_browser").close()
        except Exception:  # noqa: BLE001
            pass


def launch_context(p, headed: bool = True, slow_mo: int = 0):
    """Connect Playwright to a normally-launched Edge over CDP (see start_edge).

    `headed`/`slow_mo` are accepted for backward-compatibility but ignored
    (Edge is already a real, visible browser; CDP cannot apply slow_mo).
    """
    start_edge()
    browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{EDGE_PORT}")
    if not browser.contexts:
        raise RuntimeError("connected to Edge but found no browser context.")
    return _CtxProxy(browser, browser.contexts[0])


def shot(page, name: str) -> str:
    path = SHOTS / (name if name.endswith(".png") else name + ".png")
    try:
        page.screenshot(path=str(path))
    except Exception:  # noqa: BLE001
        pass
    return str(path)


def first_page(ctx):
    return ctx.pages[0] if ctx.pages else ctx.new_page()


def wait_stable_text(get_text, idle_secs: float = 4.0, max_secs: float = 90.0,
                     poll: float = 1.0) -> str:
    """Poll a text getter until it stops changing for `idle_secs` (or times out).

    Used to detect when a streaming Copilot answer has finished.
    """
    start = time.time()
    last = None
    last_change = time.time()
    while time.time() - start < max_secs:
        cur = get_text() or ""
        if cur != last:
            last = cur
            last_change = time.time()
        elif time.time() - last_change >= idle_secs and cur.strip():
            return cur
        time.sleep(poll)
    return last or ""
