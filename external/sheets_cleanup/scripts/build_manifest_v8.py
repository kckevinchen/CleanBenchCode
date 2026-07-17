"""Build a Google-Sheets harness manifest covering the FULL v8 benchmark corpus.

Output: outputs/manifest_v8.json  [{source, csv_path, n_rows, n_cols, header}]

csv_path points at each dataset's `dirty.csv` inside the REIN v8 datasets tree
(external/rein/rein-benchmark/datasets/<ds>/dirty.csv). That file is the exact
artifact the v8 scorer grades against, so importing it into Sheets keeps cell
coordinates aligned with the benchmark (sheet row = dirty.csv row; data_row =
sheet_row - 2). n_rows is the DATA row count (excludes the header).
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATASETS = os.path.join(REPO, "external", "rein", "rein-benchmark", "datasets")
MANIFEST = os.path.join(DATASETS, "_manifest.json")
OUT = os.path.join(HERE, "..", "outputs", "manifest_v8.json")


def main():
    ds_manifest = json.load(open(MANIFEST))
    recs = []
    missing = 0
    for ds, entry in sorted(ds_manifest["datasets"].items()):
        dirty = os.path.join(DATASETS, ds, "dirty.csv")
        if not os.path.exists(dirty):
            missing += 1
            continue
        with open(dirty, encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            continue
        header = rows[0]
        recs.append({
            "source": entry["source"],
            "csv_path": dirty,
            "n_rows": max(0, len(rows) - 1),
            "n_cols": max((len(r) for r in rows), default=len(header)),
            "header": header,
        })
    with open(OUT, "w") as f:
        json.dump(recs, f)
    print(f"wrote {OUT} with {len(recs)} tables ({missing} missing dirty.csv)")


if __name__ == "__main__":
    main()
