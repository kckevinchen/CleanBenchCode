"""Build a manifest mapping each benchmark source table -> local CSV path.

Output: outputs/manifest.json  [{source, csv_path, n_rows, n_cols, header:[...]}]
Sources are resolved from the canonical ``data/CleanBench`` extraction, with a
one-time walk of ``data/`` as a fallback.
"""

import csv
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH = os.path.join(REPO, "benchmark", "CleanBench", "benchmark_cell.json")
PRIMARY = os.path.join(REPO, "data", "CleanBench")
DATA = os.path.join(REPO, "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "manifest.json")


def benchmark_sources():
    srcs = []
    seen = set()
    with open(BENCH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)["source"]
            if s not in seen:
                seen.add(s)
                srcs.append(s)
    return srcs


def index_dir(d):
    idx = {}
    if not os.path.isdir(d):
        return idx
    for dirpath, _dirs, files in os.walk(d):
        for name in files:
            idx.setdefault(name, os.path.join(dirpath, name))
    return idx


def walk_index(root, wanted):
    idx = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name in wanted and name not in idx:
                idx[name] = os.path.join(dirpath, name)
        if len(idx) == len(wanted):
            break
    return idx


def probe(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0] if rows else []
    return len(rows) - 1, len(header), header


def main():
    srcs = benchmark_sources()
    primary = index_dir(PRIMARY)

    resolved = {}
    missing = []
    for s in srcs:
        if s in primary:
            resolved[s] = primary[s]
        else:
            missing.append(s)

    if missing:
        found = walk_index(DATA, set(missing))
        for s in missing:
            if s in found:
                resolved[s] = found[s]

    manifest = []
    still_missing = []
    for s in srcs:
        p = resolved.get(s)
        if not p:
            still_missing.append(s)
            continue
        try:
            n_rows, n_cols, header = probe(p)
        except Exception as e:  # noqa: BLE001
            still_missing.append(f"{s} (probe error: {e})")
            continue
        manifest.append(
            {"source": s, "csv_path": p, "n_rows": n_rows, "n_cols": n_cols, "header": header}
        )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"benchmark tables: {len(srcs)}")
    print(f"resolved: {len(manifest)}")
    print(f"still missing: {len(still_missing)}")
    for m in still_missing[:10]:
        print("  MISSING:", m)
    tot_rows = sum(m["n_rows"] for m in manifest)
    print(f"total data rows across tables: {tot_rows}")
    print(f"wrote {os.path.abspath(OUT)}")
    return 0 if not still_missing else 1


if __name__ == "__main__":
    sys.exit(main())
