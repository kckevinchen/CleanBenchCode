"""Step 4 (offline): parse the saved 'Clean Data with Copilot' panes (HTML) into
structured, cell-mappable findings.

    python scripts/parse_response.py

Each suggestion card in outputs/html/<source>.html is extracted into:
  - type:       Typo suggestion | Inconsistent pattern | Inconsistent text | ...
  - column:     the Excel column letter Copilot referenced (A, B, ...)
  - header:     that column's header name (from manifest_xlsx.json)
  - current / suggested:  for 'Typo suggestion'
  - examples:   for 'Inconsistent pattern' (current -> aligned forms)
  - variations: for 'Inconsistent text'
  - text:       the description sentence
  - cells/n_cells: the affected cell refs (A1), resolved by locating the value(s)
                in the source column. A single detection can map to MANY cells
                (Copilot's pager steps through cells that share the same value).

Writes outputs/findings.jsonl, outputs/findings.csv and outputs/findings.md.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

from _common import HTML_DIR, OUT, ROOT

CELLS_DIR = OUT / "cells"
TABLE_RE = re.compile(r"^Tbl_\d+$")


def col_to_index(letter):
    """Excel column letter -> 0-based index ('A'->0, 'E'->4, 'AA'->26)."""
    letter = (letter or "").strip().upper()
    if not letter or not letter.isalpha():
        return None
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def index_to_col(idx):
    """0-based index -> Excel column letter (0->'A', 4->'E', 26->'AA')."""
    s = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def load_sources():
    """source (csv name) -> {'header': [...], 'csv': Path}, merged from every
    */manifest_xlsx.json so both data/ and data_remaining/ tables resolve."""
    info = {}
    for man_path in sorted(ROOT.glob("*/manifest_xlsx.json")):
        csv_dir = man_path.parent / "csv"
        try:
            for rec in json.loads(man_path.read_text()):
                info.setdefault(rec["source"], {
                    "header": rec.get("header", []),
                    "csv": csv_dir / rec["source"],
                })
        except Exception:  # noqa: BLE001
            continue
    return info


def read_csv_rows(path):
    if not path or not path.exists():
        return None
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.reader(f))


def match_values(rec):
    """The concrete cell value(s) a finding points at (best per type)."""
    stype = rec.get("type")
    if stype == "Typo suggestion":
        return [rec["current"]] if rec.get("current") else []
    if stype == "Inconsistent pattern":
        # examples are [current, aligned, ...]; cells hold the 'current' form.
        return [v for v in rec.get("examples", [])[0::2] if v]
    return [v for v in (rec.get("values") or []) if v]


def resolve_cells(rec, rows):
    """Map a finding to affected cell refs (A1) by locating its value(s) in the
    source column. A single value can match many cells (the pager case)."""
    col_idx = col_to_index(rec.get("column"))
    if col_idx is None or not rows:
        return []
    vals = match_values(rec)
    if not vals:
        return []
    letter = index_to_col(col_idx)
    typo = rec.get("type") == "Typo suggestion"
    # Typos flag a WORD inside a cell; other types are whole-cell values.
    pats = [re.compile(r"\b" + re.escape(v) + r"\b") for v in vals] if typo else None
    wanted = {v.strip() for v in vals}
    cells = []
    for r in range(1, len(rows)):  # row 0 is the header
        row = rows[r]
        if col_idx >= len(row):
            continue
        cell = row[col_idx]
        if (any(p.search(cell) for p in pats) if typo else cell.strip() in wanted):
            cells.append(f"{letter}{r + 1}")  # csv row r -> Excel row r+1
    return cells


def measure_row_height(rects):
    """Infer the content-space row height (px) from highlight rects so the row
    mapping survives any zoom/row size: GCD of heights and top-gaps."""
    vals = [int(r.get("height", 0)) for r in rects if r.get("height", 0) > 0]
    tops = sorted({int(r.get("top", 0)) for r in rects})
    vals += [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
    g = 0
    for v in vals:
        g = math.gcd(g, v)
    return g if g >= 10 else 20


def rects_to_excel_rows(rects, row_height):
    """content top 0 = Excel row 1 (header); each row_height band is one row.
    Row 1 is excluded: a card selection sometimes also highlights the column
    header, but the header is never a flagged data cell."""
    out = set()
    for r in rects:
        n = max(1, round(r.get("height", 0) / row_height))
        base = round(r.get("top", 0) / row_height) + 1
        for k in range(n):
            row = base + k
            if row >= 2:
                out.add(row)
    return sorted(out)


def sentence(card):
    """The description sentence, stopping before any nested examples/checkbox div."""
    b = card.find("b")
    span = b.find_parent("span") if b else None
    if not span:
        return ""
    parts = []
    for child in span.children:
        if getattr(child, "name", None) == "div":
            break
        parts.append(child.get_text() if hasattr(child, "get_text") else str(child))
    return "".join(parts).strip()


BLOCKED_LABELS = {
    "give feedback", "apply", "ignore",
    "ai-generated suggestions may be incorrect",
    "is this type of alert useful for you?",
    "replace selected with:",
}


def card_type(card, desc):
    """The suggestion's category. Most cards label it directly (e.g. 'Typo
    suggestion'); alert cards (e.g. 'Invalid email addresses') put a feedback
    label first, so skip blocked labels and the description itself."""
    for el in card.find_all(["label", "span"]):
        t = el.get_text(strip=True)
        if not t or t.lower() in BLOCKED_LABELS:
            continue
        if desc and (t == desc or t in desc):
            continue
        return t
    return ""


def parse_card(card):
    desc = sentence(card)
    stype = card_type(card, desc)
    # Actionable suggestions have an Apply button; 'alerts' only have a thumbs
    # up/down 'useful?' widget and offer no correction.
    actionable = bool(card.select_one('[data-testid="SuggestionCardApplyButton"]'))
    bolds = [b.get_text(strip=True) for b in card.find_all("b")]
    col = bolds[0] if bolds else None
    # 'Extra spaces' flags the whole TABLE, so its first <b> is the table name.
    table_level = bool(col and TABLE_RE.match(col))
    rec = {"type": stype, "actionable": actionable,
           "column": None if table_level else col}
    values = []
    if stype == "Typo suggestion" and len(bolds) >= 3:
        rec["current"] = bolds[1]
        rec["suggested"] = bolds[2]
        values = [bolds[1]]
    elif stype == "Inconsistent pattern":
        rec["examples"] = [e.get_text(strip=True)
                           for e in card.select('[data-testid="ExampleTextContent"]')]
        values = rec["examples"]
    elif stype == "Inconsistent text":
        rec["variations"] = [inp.get("name")
                             for inp in card.find_all("input", attrs={"type": "checkbox"})
                             if inp.get("name")]
        values = rec["variations"]
    elif stype == "Inconsistent order":
        # The flagged cell value(s) appear as quoted bullets in the description.
        values = re.findall(r"[\u2018']([^\u2019']+)[\u2019']", desc)
    # scope guides the scoring step:
    #   cell   - names specific value(s) to locate within the column
    #   column - flags a whole column with no specific value (e.g. alerts)
    #   table  - flags the whole table (e.g. Extra spaces)
    if table_level:
        rec["scope"] = "table"
    elif values:
        rec["scope"] = "cell"
    else:
        rec["scope"] = "column"
    rec["values"] = values
    rec["text"] = desc
    return rec


def parse_html(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    return [parse_card(c) for c in soup.select(".suggestion-card")]


def main() -> int:
    sources = load_sources()
    csv_cache = {}
    rows = []
    per_table = {}
    n_files = 0
    for cj in sorted(CELLS_DIR.glob("*.json")):
        n_files += 1
        source = cj.name[:-5]  # strip ".json"
        try:
            cards = json.loads(cj.read_text(encoding="utf-8")).get("cards", [])
        except Exception:  # noqa: BLE001
            cards = []
        info = sources.get(source, {})
        hdr = info.get("header", [])
        if source not in csv_cache:
            csv_cache[source] = read_csv_rows(info.get("csv"))
        crows = csv_cache[source]
        # One row height per table, inferred from all highlight rects.
        row_h = measure_row_height([rc for c in cards for rc in c.get("rects", [])])
        sugg = []
        for c in cards:
            el = BeautifulSoup(c.get("html", ""), "html.parser").select_one(".suggestion-card")
            if el is None:
                continue
            s = parse_card(el)
            ci = col_to_index(s.get("column"))
            s["header"] = hdr[ci] if (ci is not None and 0 <= ci < len(hdr)) else None
            # Highlight-derived cells (ground truth), with a value sanity check.
            hrows = rects_to_excel_rows(c.get("rects", []), row_h)
            hl_cells, hl_vals = [], set()
            for er in hrows:
                hl_cells.append(f"{s.get('column') or ''}{er}")
                if ci is not None and crows and 0 <= er - 1 < len(crows) and ci < len(crows[er - 1]):
                    hl_vals.add(crows[er - 1][ci])
            shown = set(match_values(s))
            if hrows and shown:
                if s.get("type") == "Typo suggestion":
                    # the misspelled word sits INSIDE a cell, not as its whole value
                    sanity = any(re.search(r"\b" + re.escape(w) + r"\b", v)
                                 for w in shown for v in hl_vals)
                else:
                    sanity = bool(shown & hl_vals)
            else:
                sanity = None
            if hl_cells and sanity is not False:
                s["cells"] = hl_cells
                s["cell_source"] = "highlight"
            else:
                s["cells"] = resolve_cells(s, crows)
                s["cell_source"] = "value"
            s["sanity"] = sanity
            s["n_cells"] = len(s["cells"])
            sugg.append(s)
            rows.append({"source": source, **s})
        per_table[source] = sugg

    (OUT / "findings.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )

    # Flat CSV (utf-8-sig so Excel opens unicode correctly).
    csv_cols = ["source", "type", "scope", "actionable", "column", "header",
                "current", "suggested", "values", "n_cells", "cells",
                "cell_source", "sanity", "text"]
    with open(OUT / "findings.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for r in rows:
            w.writerow([
                r.get("source", ""),
                r.get("type", ""),
                r.get("scope", ""),
                r.get("actionable", ""),
                r.get("column") or "",
                r.get("header") or "",
                r.get("current", ""),
                r.get("suggested", ""),
                " | ".join(r.get("values") or []),
                r.get("n_cells", 0),
                " ".join(r.get("cells") or []),
                r.get("cell_source", ""),
                r.get("sanity", ""),
                (r.get("text") or "").replace("\n", " ").strip(),
            ])

    by_type = {}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    flagged = sum(1 for v in per_table.values() if v)
    lines = [
        "# Excel Clean Data \u2014 parsed suggestions",
        "",
        f"- tables parsed: **{n_files}**",
        f"- Tables with >=1 suggestion: **{flagged}**",
        f"- Total suggestions: **{len(rows)}**",
        "",
        "## By type",
        "",
    ]
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {t or '(blank)'}: **{n}**")
    by_scope = {}
    for r in rows:
        by_scope[r.get("scope")] = by_scope.get(r.get("scope"), 0) + 1
    n_alert = sum(1 for r in rows if not r.get("actionable", True))
    lines += ["", "## By scope (cell-mappability)", ""]
    for sc in ("cell", "column", "table"):
        if by_scope.get(sc):
            lines.append(f"- {sc}: **{by_scope[sc]}**")
    lines.append(f"- non-actionable alerts (no correction): **{n_alert}**")
    total_cells = sum(r.get("n_cells", 0) for r in rows)
    multi = sum(1 for r in rows if r.get("n_cells", 0) > 1)
    by_hl = sum(1 for r in rows if r.get("cell_source") == "highlight")
    by_val = sum(1 for r in rows if r.get("cell_source") == "value")
    sanity_fail = sum(1 for r in rows if r.get("sanity") is False)
    lines += ["", "## Cell resolution", ""]
    lines.append(f"- total affected cells: **{total_cells}**")
    lines.append(f"- detections mapping to >1 cell: **{multi}**")
    lines.append(f"- cells from highlight: **{by_hl}** | from value fallback: **{by_val}**")
    lines.append(f"- sanity-check failures (review): **{sanity_fail}**")
    lines += ["", "## Per-table suggestions", ""]
    for source in sorted(per_table):
        sugg = per_table[source]
        if not sugg:
            continue
        lines.append(f"### {source}")
        for s in sugg:
            if s.get("scope") == "table":
                loc = "whole table"
            else:
                col = s.get("column") or "?"
                loc = f"col {col}" + (f" ({s['header']})" if s.get("header") else "")
            alert = "" if s.get("actionable", True) else " _(alert, no correction)_"
            if s["type"] == "Typo suggestion":
                detail = f"'{s.get('current')}' -> '{s.get('suggested')}'"
            elif s["type"] == "Inconsistent pattern":
                detail = " | ".join(s.get("examples", []))
            elif s["type"] == "Inconsistent text":
                detail = " / ".join(s.get("variations", []))
            elif s.get("values"):
                detail = " / ".join(s["values"])
            else:
                detail = s.get("text", "")
            ncell = s.get("n_cells", 0)
            shown = " ".join(s.get("cells", [])[:10])
            cellinfo = f" -> {ncell} cell(s): {shown}" if ncell else ""
            lines.append(f"- **{s['type']}**{alert} [{loc}]: {detail}{cellinfo}")
        lines.append("")
    (OUT / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:8]))
    print(f"\nwrote {OUT / 'findings.jsonl'}, {OUT / 'findings.csv'} "
          f"({len(rows)} suggestions) and {OUT / 'findings.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
