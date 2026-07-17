"""Detect same-column values that collapse to one normalization key."""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count


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


def main(data_dir: str, out_dir: str, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "normalization_errors.jsonl")
    out_csv = os.path.join(out_dir, "normalization_errors.csv")
    out_stats = os.path.join(out_dir, "normalization_errors.stats")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if limit:
        csv_files = csv_files[:limit]

    total = len(csv_files)
    workers = min(cpu_count(), 32)
    print(f"Processing {total} files with {workers} workers...")
    print(f"Streaming output to:\n  {out_json}\n  {out_csv}\n")

    done = 0
    skipped = 0
    total_findings = 0
    # use sets to count distinct (file, col) pairs and total error cells
    error_cols: set[tuple] = set()   # (file, column)
    total_error_cells = 0

    with open(out_json, "w", encoding="utf-8") as fj, \
         open(out_csv, "w", encoding="utf-8", newline="") as fc:

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

    with open(out_stats, "w", encoding="utf-8") as fs:
        fs.write("\n".join(stats_lines) + "\n")

    print(f"\nJSON  output : {out_json}")
    print(f"CSV   output : {out_csv}")
    print(f"Stats output : {out_stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/normalization_baseline")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, limit=args.limit)
