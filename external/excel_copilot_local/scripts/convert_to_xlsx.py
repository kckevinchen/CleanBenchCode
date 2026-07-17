"""Convert each benchmark CSV into an .xlsx workbook with the data as a named
Excel Table.

Copilot in Excel works most reliably when the data is a real Table, so we do
that offline and deterministically here, before any browser automation.

Run once after extracting the package:
    python scripts/convert_to_xlsx.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from _common import CSV_DIR, DATA, XLSX_DIR


def safe_table_name(i: int) -> str:
    # Excel table names must start with a letter/underscore and have no spaces.
    return f"Tbl_{i:04d}"


def convert_one(src_csv: Path, dst_xlsx: Path, table_name: str) -> tuple[int, int]:
    with open(src_csv, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        rows = [["(empty)"]]
    # Normalize ragged rows to the header width.
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # De-dup / fill blank header cells (Excel tables require unique non-empty headers).
    header = rows[0]
    seen = {}
    for j, h in enumerate(header):
        h = (h or "").strip() or f"Column{j+1}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 0
        header[j] = h
    rows[0] = header

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    for r in rows:
        ws.append(r)

    n_rows = len(rows)
    n_cols = width
    ref = f"A1:{get_column_letter(n_cols)}{n_rows}"
    table = Table(displayName=table_name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False
    )
    ws.add_table(table)
    dst_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(dst_xlsx)
    return n_rows - 1, n_cols


def main() -> int:
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    out = []
    failures = []
    for i, rec in enumerate(manifest):
        source = rec["source"]
        src_csv = CSV_DIR / source
        xlsx_name = source.rsplit(".", 1)[0] + ".xlsx"
        dst = XLSX_DIR / xlsx_name
        if not src_csv.exists():
            failures.append(f"missing CSV: {source}")
            continue
        try:
            n_rows, n_cols = convert_one(src_csv, dst, safe_table_name(i))
            rec = dict(rec)
            rec["xlsx"] = xlsx_name
            rec["n_rows"] = n_rows
            rec["n_cols"] = n_cols
            out.append(rec)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{source}: {type(e).__name__}: {e}")

    (DATA / "manifest_xlsx.json").write_text(json.dumps(out, indent=2))
    print(f"converted: {len(out)} / {len(manifest)}")
    if failures:
        print(f"failures: {len(failures)}")
        for f in failures[:10]:
            print("  ", f)
    print(f"xlsx dir: {XLSX_DIR}")
    print(f"wrote {DATA / 'manifest_xlsx.json'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
