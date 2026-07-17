"""Report what Google Sheets 'Cleanup suggestions' flagged across all captured
tables. No benchmark scoring -- just the raw detections, organized.

Outputs:
  outputs/findings.jsonl  -- one row per suggestion (machine-readable)
  outputs/findings.md     -- human-readable summary + per-table detail
"""

import collections
import json
import os
import re

from harness import OUT
from parse_cleanup_html import parse_html

MANIFEST = os.path.join(OUT, "manifest.json")
HTML_DIR = os.path.join(OUT, "html")
A1_CELL = re.compile(r"^([A-Z]+)(\d+)$")

# category -> scope of the chip references
ROW_CATS = {"DUPLICATE ROWS"}
COL_CATS = {"COLUMN FORMAT"}


def col_letter_to_name(letter, header):
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    idx -= 1
    return header[idx] if 0 <= idx < len(header) else f"col[{letter}]"


def describe(s, header):
    """Return (scope, location_str, detail_str) for one suggestion."""
    cat = s["category"]
    chips = s["ranges"]
    if cat in ROW_CATS:
        rows = [c for c in chips if c.isdigit()]
        return ("row", "rows " + ",".join(rows), s["value"][:60])
    if cat in COL_CATS:
        col = chips[0].split(":")[0] if chips else "?"
        name = col_letter_to_name(col, header)
        frm = s["extra"].get("from", "")
        to = s["extra"].get("to", "")
        return ("column", f"col '{name}'", f"{frm} -> {to}")
    # cell-scoped (WHITESPACE, INVALID DATA, other)
    locs = []
    for c in chips:
        m = A1_CELL.match(c.strip())
        if m:
            name = col_letter_to_name(m.group(1), header)
            locs.append(f"{c}('{name}' row{int(m.group(2))-2})")
        else:
            locs.append(c)
    return ("cell", "; ".join(locs), repr(s["value"])[:60])


def main():
    manifest = {m["source"]: m for m in json.load(open(MANIFEST))}

    rows = []
    per_table = {}
    cat_counts = collections.Counter()
    cat_tables = collections.Counter()
    cell_type_counts = collections.Counter()

    for fname in sorted(os.listdir(HTML_DIR)):
        if not fname.endswith(".html"):
            continue
        src = fname[:-5]
        header = manifest.get(src, {}).get("header", [])
        parsed = parse_html(open(os.path.join(HTML_DIR, fname)).read())
        sugg = parsed["suggestions"]
        per_table[src] = sugg
        cats_here = set()
        for s in sugg:
            cat = s["category"]
            cat_counts[cat] += 1
            cats_here.add(cat)
            scope, loc, detail = describe(s, header)
            cell_type_counts[scope] += 1
            rows.append({
                "source": src,
                "dataset": manifest.get(src, {}).get("dataset", ""),
                "category": cat,
                "scope": scope,
                "location": loc,
                "detail": detail,
                "ranges": s["ranges"],
                "value": s["value"],
                "action": s["action"],
            })
        for c in cats_here:
            cat_tables[c] += 1

    with open(os.path.join(OUT, "findings.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tables = len([f for f in os.listdir(HTML_DIR) if f.endswith(".html")])
    n_flagged = sum(1 for s in per_table.values() if s)

    lines = [
        "# Google Sheets Cleanup — detections found",
        "",
        f"- Tables processed: **{n_tables}**",
        f"- Tables with >=1 suggestion: **{n_flagged}**  "
        f"(clean: {n_tables - n_flagged})",
        f"- Total suggestions: **{len(rows)}**",
        "",
        "## By category",
        "| category | suggestions | tables | scope |",
        "|---|---:|---:|---|",
    ]
    scope_of = {"DUPLICATE ROWS": "row", "COLUMN FORMAT": "column",
                "WHITESPACE": "cell", "INVALID DATA": "cell"}
    for cat, cnt in cat_counts.most_common():
        lines.append(f"| {cat} | {cnt} | {cat_tables[cat]} | "
                     f"{scope_of.get(cat, '?')} |")
    lines += [
        "",
        f"_Scope totals: cell={cell_type_counts['cell']}, "
        f"column={cell_type_counts['column']}, row={cell_type_counts['row']}_",
        "",
        "## Per-table detections",
        "",
    ]
    for src in sorted(per_table):
        sugg = per_table[src]
        if not sugg:
            continue
        header = manifest.get(src, {}).get("header", [])
        ds = manifest.get(src, {}).get("dataset", "")
        lines.append(f"### {src}  ({ds})")
        for s in sugg:
            scope, loc, detail = describe(s, header)
            lines.append(f"- **{s['category']}** [{scope}] {loc} — {detail}")
        lines.append("")

    with open(os.path.join(OUT, "findings.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # console summary
    print("\n".join(lines[:18]))
    print(f"\nwrote {os.path.join(OUT, 'findings.jsonl')} ({len(rows)} rows)")
    print(f"wrote {os.path.join(OUT, 'findings.md')}")


if __name__ == "__main__":
    main()
