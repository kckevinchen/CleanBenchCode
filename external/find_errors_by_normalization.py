"""
Heuristic error finder: detects values in the same column that normalize to the same key
(after removing digit-containing values, lowercasing, stripping punctuation/symbols/whitespace).

Runs in parallel with multiprocessing; findings are flushed to disk as they arrive.
"""

import os
import re
import csv
import glob
import json
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count

DATA_DIR = "/datadrive/yeye/_DATA/PBI.allDataCSV.data_extracted_from_flow_run1_2022_06_20"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_JSON  = os.path.join(OUT_DIR, "normalization_errors.jsonl")  # one JSON obj per line -> streamable
OUT_CSV   = os.path.join(OUT_DIR, "normalization_errors.csv")
OUT_STATS = os.path.join(OUT_DIR, "normalization_errors.stats")


def sanitize_for_csv(val: str) -> str:
    """Replace embedded newlines/tabs/carriage-returns so they don't break CSV rows."""
    return re.sub(r"[\r\n\t]+", " ", val)


def normalize_key(val: str) -> str:
    """Lowercase, then strip all punctuation, symbols, and whitespace."""
    val = val.lower()
    val = re.sub(r"[^\w]", "", val, flags=re.UNICODE)   # keep only word chars (letters/digits/_)
    val = re.sub(r"[_\d]", "", val)                      # then drop digits and underscores
    return val


def find_errors_by_normalization(df: dict) -> list:
    """
    df: dict mapping column_name -> list of raw string values.

    Returns list of dicts:
      { "column", "key", "variants", "cell_count" }
    where cell_count = number of (non-empty, non-digit) rows whose value is one of the variants.
    """
    findings = []
    for col, values in df.items():
        # Drop values that contain any digit
        text_vals = [v for v in values if v and not re.search(r"\d", v)]
        # Dedup while preserving first-seen order
        distinct = list(dict.fromkeys(text_vals))

        key_to_vals = defaultdict(list)
        for v in distinct:
            key = normalize_key(v)
            if key:
                key_to_vals[key].append(v)

        for key, variants in key_to_vals.items():
            if len(variants) > 1:
                variant_set = set(variants)
                cell_count = sum(1 for v in text_vals if v in variant_set)
                findings.append({
                    "column": col,
                    "key": key,
                    "variants": variants,
                    "cell_count": cell_count,
                })

    return findings


def process_file(filepath: str):
    """Worker: read one CSV, return (filename, findings, had_error)."""
    filename = os.path.basename(filepath)
    try:
        with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return filename, [], True
            columns = {col: [] for col in reader.fieldnames if col}
            for row in reader:
                for col in columns:
                    val = (row.get(col) or "").strip()
                    columns[col].append(val)

        findings = find_errors_by_normalization(columns)
        for entry in findings:
            entry["file"] = filename
        return filename, findings, False
    except Exception:
        return filename, [], True


def main(limit=None):
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if limit:
        csv_files = csv_files[:limit]

    total = len(csv_files)
    workers = min(cpu_count(), 32)
    print(f"Processing {total} files with {workers} workers...")
    print(f"Streaming output to:\n  {OUT_JSON}\n  {OUT_CSV}\n")

    done = 0
    skipped = 0
    total_findings = 0
    # use sets to count distinct (file, col) pairs and total error cells
    error_cols: set[tuple] = set()   # (file, column)
    total_error_cells = 0

    with open(OUT_JSON, "w", encoding="utf-8") as fj, \
         open(OUT_CSV,  "w", encoding="utf-8", newline="") as fc:

        cw = csv.writer(fc, quoting=csv.QUOTE_ALL)
        cw.writerow(["file", "column", "num_variants", "cell_count", "variants"])
        fc.flush()

        with Pool(processes=workers) as pool:
            for filename, findings, had_error in pool.imap_unordered(process_file, csv_files, chunksize=10):
                done += 1
                if had_error:
                    skipped += 1

                for entry in findings:
                    fj.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    clean_variants = [sanitize_for_csv(v) for v in entry["variants"]]
                    cw.writerow([
                        sanitize_for_csv(entry["file"]),
                        sanitize_for_csv(entry["column"]),
                        len(entry["variants"]),
                        entry["cell_count"],
                        "___".join(clean_variants),
                    ])
                    total_findings += 1
                    error_cols.add((entry["file"], entry["column"]))
                    total_error_cells += entry["cell_count"]

                fj.flush()
                fc.flush()

                if done % 100 == 0:
                    print(f"  {done}/{total} files done, {total_findings} findings, "
                          f"{len(error_cols)} error cols, {total_error_cells} error cells so far...")

    files_ok = done - skipped
    stats_lines = [
        f"files_total:        {total}",
        f"files_processed:    {files_ok}",
        f"files_skipped:      {skipped}",
        f"error_col_instances:{len(error_cols)}",
        f"total_findings:     {total_findings}",
        f"total_error_cell:   {total_error_cells}",
    ]

    print("\n--- Stats ---")
    for line in stats_lines:
        print(line)

    with open(OUT_STATS, "w", encoding="utf-8") as fs:
        fs.write("\n".join(stats_lines) + "\n")

    print(f"\nJSON  output : {OUT_JSON}")
    print(f"CSV   output : {OUT_CSV}")
    print(f"Stats output : {OUT_STATS}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
