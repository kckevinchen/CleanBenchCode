"""Expand findings into a cell-level CSV: one row per affected cell, with the
cell's REAL value looked up from the source CSV.

    python scripts/map_cells.py

Reads outputs/findings.jsonl, writes outputs/excel_finding.csv with columns:
  source, cell, column, excel_row, data_row, header, value, type,
  suggested_fix, cell_source, text
where data_row is the benchmark 0-based data row (= excel_row - 2; row 1 is the
header, first data row is Excel row 2). `suggested_fix` is the SINGLE corrected
cell value: Copilot's actual "Apply" result (outputs/fixes.csv, from
apply_and_capture.py + diff_applied.py) when available, otherwise the full cell
reconstructed from the pane's typo suggestion (misspelled word -> suggestion).
Blank only when the cell was already correct or the issue is non-actionable
(e.g. invalid email).
"""

from __future__ import annotations

import csv
import json
import re
import sys

from _common import OUT, ROOT
from parse_response import col_to_index

CELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def csv_paths():
    """source (csv name) -> Path to its source CSV, from every */manifest_xlsx.json."""
    paths = {}
    for man in sorted(ROOT.glob("*/manifest_xlsx.json")):
        csv_dir = man.parent / "csv"
        try:
            for rec in json.loads(man.read_text()):
                paths.setdefault(rec["source"], csv_dir / rec["source"])
        except Exception:  # noqa: BLE001
            continue
    return paths


def load_fixes():
    """(source, cell) -> applied 'after' value, from outputs/fixes.csv if present."""
    fixes = {}
    p = OUT / "fixes.csv"
    if not p.exists():
        return fixes
    with open(p, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            fixes[(row["source"], row["cell"])] = row["after"]
    return fixes


def reconstruct_cell_fix(value, current, suggested):
    """Full-cell typo fix: the pane states only the corrected WORD, so splice it
    back into the cell by replacing the misspelled word `current` in `value`."""
    if not suggested:
        return ""
    if not current or not value:
        return suggested
    pat = re.compile(r"\b" + re.escape(current) + r"\b")
    new, n = pat.subn(suggested, value, count=1)
    if n:
        return new
    if current in value:
        return value.replace(current, suggested, 1)
    return suggested


def main() -> int:
    paths = csv_paths()
    fixes = load_fixes()
    cache = {}
    out = []
    for line in (OUT / "findings.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        f = json.loads(line)
        src = f["source"]
        if src not in cache:
            p = paths.get(src)
            try:
                cache[src] = (list(csv.reader(open(p, encoding="utf-8", errors="replace")))
                              if p and p.exists() else None)
            except Exception:  # noqa: BLE001
                cache[src] = None
        rows = cache[src]
        for cell in f.get("cells", []):
            m = CELL_RE.match(cell)
            if not m:
                continue
            letter, xrow = m.group(1).upper(), int(m.group(2))
            cidx = col_to_index(letter)
            value = ""
            if (rows and cidx is not None and 0 <= xrow - 1 < len(rows)
                    and cidx < len(rows[xrow - 1])):
                value = rows[xrow - 1][cidx]
            applied = fixes.get((src, cell), "")
            fix = applied or reconstruct_cell_fix(
                value, f.get("current", ""), f.get("suggested", ""))
            out.append([
                src, cell, letter, xrow, xrow - 2, f.get("header") or "", value,
                f.get("type", ""), fix,
                f.get("cell_source", ""),
                (f.get("text") or "").replace("\n", " ").strip(),
            ])

    cols = ["source", "cell", "column", "excel_row", "data_row", "header", "value",
            "type", "suggested_fix", "cell_source", "text"]
    with open(OUT / "excel_finding.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(out)
    print(f"wrote {OUT / 'excel_finding.csv'} with {len(out)} cell rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
