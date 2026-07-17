"""Parse a saved Google Sheets 'Cleanup suggestions' panel HTML into structured
detections.

The panel lives in `.waffle-inline-assistant-sidebar` (title "Cleanup
suggestions"). Suggestions are grouped under category labels
(`.waffle-inline-assistant-sidebar-category-label`, e.g. "WHITESPACE"). Each
suggestion is a `.waffle-inline-assistant-suggestion-details` holding one or more
`.waffle-inline-assistant-gridrange-chip` cell/range references, a value preview,
and an accept button (e.g. "Trim all").

We extract, per suggestion: category, accept-action label, the A1 ranges, and the
displayed value. Whitespace is rendered as text plus `tw-space` marker divs; we
reconstruct it as text + a single trailing/leading space sentinel so the value is
informative without being exact-fragile.

Returns a list of dicts; `expand_ranges` turns A1 ranges into (col_letter, row)
cell tuples (1-based row as shown in the Sheets grid, i.e. including header row).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

SIDEBAR = ".waffle-inline-assistant-sidebar"
CATEGORY = ".waffle-inline-assistant-sidebar-category-label"
DETAILS = ".waffle-inline-assistant-suggestion-details"
NF_ROW = ".waffle-inline-assistant-nf-row"
CHIP = ".waffle-inline-assistant-gridrange-chip"
TW_ROW = ".waffle-inline-assistant-tw-row"
ACCEPT = ".waffle-inline-assistant-accept-button"

A1_CELL = re.compile(r"^([A-Z]+)(\d+)$")
A1_RANGE = re.compile(r"^([A-Z]+)(\d+):([A-Z]+)(\d+)$")
A1_COL = re.compile(r"^([A-Z]+):([A-Z]+)$")
NF_LABEL = re.compile(r"Column\s+([A-Z]+)\s+Change format from\s+(.*?)\s+to\s+(.*)$")


@dataclass
class Suggestion:
    category: str
    action: str
    ranges: list[str]
    value: str
    raw_text: str = ""
    extra: dict = field(default_factory=dict)


def _col_to_idx(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def _idx_to_col(idx: int) -> str:
    s = ""
    while idx > 0:
        idx, r = divmod(idx - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def _value_from_tw_row(tw) -> str:
    """Reconstruct a whitespace-preview value: text nodes are literal chars; each
    `tw-space` div is one otherwise-invisible space."""
    parts: list[str] = []
    for child in tw.children:
        name = getattr(child, "name", None)
        if name is None:  # NavigableString
            parts.append(str(child))
        elif "waffle-inline-assistant-tw-space" in (child.get("class") or []):
            parts.append(" ")
        else:
            parts.append(child.get_text())
    return "".join(parts)


def _has_hidden_ancestor(el, stop) -> bool:
    """True if el or an ancestor (up to `stop`) is display:none / visibility:hidden."""
    cur = el
    while cur is not None and cur is not stop:
        style = (cur.get("style") or "")
        if "display: none" in style or "visibility: hidden" in style:
            return True
        cur = cur.parent
    return False


def parse_html(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    side = soup.select_one(SIDEBAR)
    result = {"present": bool(side), "no_suggestions": False, "suggestions": []}
    if not side:
        return result

    text = side.get_text(" ", strip=True)

    # Walk in document order so each suggestion inherits the most recent category
    # label. Each real suggestion is exactly one `details` element; inside it we
    # detect the type (number-format nf-row vs whitespace tw-row). We deliberately
    # do NOT match nf-row/tw-row at the top level, since they are nested inside
    # `details` and would double-count.
    current_cat = ""
    suggestions: list[Suggestion] = []
    seen = set()
    for el in side.select(f"{CATEGORY}, {DETAILS}"):
        classes = el.get("class") or []
        if "waffle-inline-assistant-sidebar-category-label" in classes:
            current_cat = el.get_text(strip=True)
            continue

        # Skip cards inside a hidden container (virtualized measurement copies).
        if _has_hidden_ancestor(el, side):
            continue

        chips = [c.get_text(strip=True) for c in el.select(CHIP)]
        nf = el.select_one(NF_ROW)
        tw = el.select_one(TW_ROW)
        accept = el.select_one(ACCEPT)
        action = accept.get_text(strip=True) if accept else ""

        if nf is not None:
            label = nf.get("aria-label", "") or ""
            orig = nf.select_one(".waffle-inline-assistant-nf-original-preview")
            sugg = nf.select_one(".waffle-inline-assistant-nf-suggested-preview")
            m = NF_LABEL.search(label)
            col = (m.group(1) if m else (chips[0] if chips else "")).strip()
            from_v = orig.get_text(strip=True) if orig else (m.group(2) if m else "")
            to_v = sugg.get_text(strip=True) if sugg else (m.group(3) if m else "")
            s = Suggestion(
                category=current_cat,
                action=action or "Apply formatting",
                ranges=chips or ([col] if col else []),
                value=from_v,
                raw_text=(label or el.get_text(" ", strip=True)),
                extra={"type": "number_format", "from": from_v, "to": to_v, "column": col},
            )
        elif tw is not None:
            s = Suggestion(
                category=current_cat,
                action=action,
                ranges=chips,
                value=_value_from_tw_row(tw),
                raw_text=el.get_text(" ", strip=True),
                extra={"type": "whitespace"},
            )
        else:
            rc = el.select_one(".waffle-inline-assistant-recommendation-container")
            s = Suggestion(
                category=current_cat,
                action=action,
                ranges=chips,
                value=rc.get_text(" ", strip=True) if rc else "",
                raw_text=el.get_text(" ", strip=True),
                extra={"type": "other"},
            )

        key = (s.category, s.extra.get("type"), tuple(s.ranges), s.value,
               s.extra.get("to", ""))
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(s)
    result["suggestions"] = [s.__dict__ for s in suggestions]
    result["no_suggestions"] = len(suggestions) == 0 and "No suggestions available" in text
    return result


def expand_ranges(ranges: list[str]) -> list[tuple[str, int]]:
    """Expand A1 refs (single cell, rect range, or full-column) to (col, row).

    Full-column refs (A:A) are returned as (col, 0) sentinels since the row span
    is the whole sheet; callers should treat row==0 as 'whole column'.
    """
    cells: list[tuple[str, int]] = []
    for r in ranges:
        r = r.strip()
        m = A1_CELL.match(r)
        if m:
            cells.append((m.group(1), int(m.group(2))))
            continue
        m = A1_RANGE.match(r)
        if m:
            c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            for ci in range(_col_to_idx(c1), _col_to_idx(c2) + 1):
                for ri in range(r1, r2 + 1):
                    cells.append((_idx_to_col(ci), ri))
            continue
        m = A1_COL.match(r)
        if m:
            for ci in range(_col_to_idx(m.group(1)), _col_to_idx(m.group(2)) + 1):
                cells.append((_idx_to_col(ci), 0))
            continue
        if re.match(r"^[A-Z]+$", r):  # bare column letter, e.g. "C" -> whole column
            cells.append((r, 0))
            continue
    return cells


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/sample_full_page.html"
    out = parse_html(open(path).read())
    print(json.dumps(out, indent=2, ensure_ascii=False))
    for s in out["suggestions"]:
        print("cells:", expand_ranges(s["ranges"]))
