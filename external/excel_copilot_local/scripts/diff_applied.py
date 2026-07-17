"""Diff each table's before/after grid snapshots to recover the exact per-cell
fix that Copilot's Clean Data "Apply" produced.

    python scripts/diff_applied.py

scripts/apply_and_capture.py saved, per table, a clipboard TSV of the used range
before and after clicking Apply:
  outputs/applied/<source>.before.tsv
  outputs/applied/<source>.after.tsv
Both come from Excel's own clipboard, so a cell differs only when Apply actually
changed it (no CSV-vs-Excel formatting noise). This writes:
  outputs/fixes.jsonl - one record per changed cell
  outputs/fixes.csv   - cols: source, cell, column, excel_row, data_row, before, after
where `cell` is the A1 ref (e.g. B47) and data_row = excel_row - 2 (benchmark
0-based; row 1 is the header, first data row is Excel row 2).
"""

from __future__ import annotations

import csv
import io
import json
import sys

from _common import OUT

APPLIED_DIR = OUT / "applied"
AFTER_SUFFIX = ".after.tsv"


def index_to_col(idx):
    """0-based column index -> letter (0 -> A, 26 -> AA)."""
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def parse_tsv(text):
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.reader(io.StringIO(text), delimiter="\t"))


def norm(s):
    return (s or "").strip()


def main() -> int:
    rows_out = []
    n_tables = 0
    for af in sorted(APPLIED_DIR.glob("*" + AFTER_SUFFIX)):
        source = af.name[: -len(AFTER_SUFFIX)]
        bf = APPLIED_DIR / f"{source}.before.tsv"
        if not bf.exists():
            print("  no before snapshot for", source, flush=True)
            continue
        before = parse_tsv(bf.read_text(encoding="utf-8"))
        after = parse_tsv(af.read_text(encoding="utf-8"))
        n_tables += 1
        for r in range(max(len(before), len(after))):
            brow = before[r] if r < len(before) else []
            arow = after[r] if r < len(after) else []
            for c in range(max(len(brow), len(arow))):
                bv = norm(brow[c]) if c < len(brow) else ""
                av = norm(arow[c]) if c < len(arow) else ""
                if bv == av:
                    continue
                col = index_to_col(c)
                xrow = r + 1  # TSV row 0 == Excel row 1 (the header)
                rows_out.append({
                    "source": source, "cell": f"{col}{xrow}", "column": col,
                    "excel_row": xrow, "data_row": xrow - 2,
                    "before": bv, "after": av,
                })

    with open(OUT / "fixes.jsonl", "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["source", "cell", "column", "excel_row", "data_row", "before", "after"]
    with open(OUT / "fixes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows_out:
            w.writerow([r[c] for c in cols])

    print(f"{n_tables} tables compared -> {len(rows_out)} changed cells", flush=True)
    print(f"wrote {OUT / 'fixes.csv'} and {OUT / 'fixes.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
