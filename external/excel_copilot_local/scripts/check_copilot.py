"""Step 2: verify this account actually has *Copilot in Excel*.

    python scripts/check_copilot.py

Creates a blank workbook on Excel for the web and searches every frame for a
Copilot ribbon button. Prints PASS (Copilot present) or FAIL (not licensed),
and saves a screenshot to outputs/shots/check_copilot.png.

This is the go/no-go gate: if it prints FAIL, the batch run won't work and you
need a different account/license.
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

from _common import launch_context, shot


def find_copilot(frame):
    selectors = [
        '[aria-label*="Copilot" i]',
        '[title*="Copilot" i]',
        'button:has-text("Copilot")',
        '[data-unique-id*="Copilot" i]',
        '[id*="Copilot" i]',
    ]
    hits = 0
    for sel in selectors:
        try:
            hits += frame.locator(sel).count()
        except Exception:  # noqa: BLE001
            pass
    return hits


def main() -> int:
    with sync_playwright() as p:
        ctx = launch_context(p, headed=True)
        page = ctx.new_page()
        page.goto("https://excel.cloud.microsoft/", wait_until="domcontentloaded",
                  timeout=90000)
        page.wait_for_timeout(6000)
        # Create a blank workbook
        for txt in ["Create blank workbook", "Blank workbook", "New blank workbook"]:
            try:
                el = page.get_by_text(txt, exact=False).first
                if el.count():
                    el.click()
                    break
            except Exception:  # noqa: BLE001
                pass
        page.wait_for_timeout(18000)  # workbook editor boot

        # The editor may be in a new tab.
        wb = page
        for pp in ctx.pages:
            if "sharepoint.com" in pp.url or "wopi" in pp.url.lower():
                wb = pp
        wb.bring_to_front()
        wb.wait_for_timeout(8000)
        shot(wb, "check_copilot")

        total = 0
        per_frame = []
        for fr in wb.frames:
            h = find_copilot(fr)
            if h:
                per_frame.append((fr.url[:50], h))
            total += h

        print("\n" + "=" * 60)
        if total > 0:
            print("PASS: Copilot appears available in Excel for the web.")
            for u, h in per_frame:
                print(f"   match in frame {u} x{h}")
        else:
            print("FAIL: No Copilot button found in Excel for the web.")
            print("This account likely lacks a Microsoft 365 Copilot license.")
        print("screenshot: outputs/shots/check_copilot.png")
        print("=" * 60)
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
