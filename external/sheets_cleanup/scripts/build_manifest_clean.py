"""Build a manifest of the 785 non-benchmark ("clean") tables in
data/_merged_v2_all_sources for the false-positive run.

Output: outputs/manifest_clean.json  [{source, csv_path, n_rows, n_cols, header}]
Excludes the 217 benchmark sources (those are already captured).
"""

import csv
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH = os.path.join(REPO, "benchmark", "merged_v2", "benchmark_cell.json")
POOL = os.path.join(REPO, "data", "_merged_v2_all_sources")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "manifest_clean.json")


def benchmark_sources():
    seen = set()
    for line in open(BENCH):
        line = line.strip()
        if line:
            seen.add(json.loads(line)["source"])
    return seen


def probe(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    header = rows[0] if rows else []
    return len(rows) - 1, len(header), header


def main():
    bench = benchmark_sources()
    entries = sorted(os.listdir(POOL))
    manifest = []
    skipped = []
    for name in entries:
        if not name.lower().endswith(".csv"):
            continue
        if name in bench:
            continue
        path = os.path.realpath(os.path.join(POOL, name))
        if not os.path.isfile(path):
            skipped.append((name, "missing"))
            continue
        try:
            n_rows, n_cols, header = probe(path)
        except Exception as e:  # noqa: BLE001
            skipped.append((name, f"probe:{e}"))
            continue
        if n_rows <= 0 or n_cols <= 0:
            skipped.append((name, f"empty rows={n_rows} cols={n_cols}"))
            continue
        manifest.append(
            {"source": name, "csv_path": path, "n_rows": n_rows,
             "n_cols": n_cols, "header": header}
        )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"pool entries: {len(entries)} | benchmark excluded: {len(bench)}")
    print(f"clean tables in manifest: {len(manifest)}")
    print(f"skipped: {len(skipped)}")
    for s in skipped[:10]:
        print("  skip:", s)
    rows = [m["n_rows"] for m in manifest]
    cols = [m["n_cols"] for m in manifest]
    if rows:
        rows.sort()
        cols.sort()
        print(f"rows: min={rows[0]} median={rows[len(rows)//2]} "
              f"p90={rows[int(len(rows)*0.9)]} max={rows[-1]} total={sum(rows)}")
        print(f"cols: min={cols[0]} median={cols[len(cols)//2]} "
              f"p90={cols[int(len(cols)*0.9)]} max={cols[-1]}")
        big = sum(1 for r in rows if r > 1000)
        print(f"tables with >1000 rows: {big}")
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    sys.exit(main())
