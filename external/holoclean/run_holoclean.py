#!/usr/bin/env python3
"""Run upstream HoloClean as a repair-to-detection baseline over REIN tables."""

from __future__ import print_function

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HOLOCLEAN_DIR = Path(__file__).resolve().parent
REPO_ROOT = HOLOCLEAN_DIR.parents[1]
REIN_DIR = REPO_ROOT / "external" / "rein"
DATASETS_DIR = REIN_DIR / "rein-benchmark" / "datasets"
RESULTS_DIR = REIN_DIR / "results"

sys.path.insert(0, str(HOLOCLEAN_DIR))
from adapter import DETECTOR, pinned_commit, run_method  # noqa: E402


def dataset_names(ds_root):
    manifest_path = ds_root / "_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return sorted(manifest["datasets"])
    return sorted(p.name for p in ds_root.iterdir() if p.is_dir() and (p / "dirty.csv").exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "predictions_holoclean.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test only; manager runs full corpus")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--force", action="store_true", help="ignore per-table cache")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    names = dataset_names(args.datasets_dir)
    if args.limit:
        names = names[: args.limit]

    print("=== HoloClean run ===", flush=True)
    print("detector        : %s" % DETECTOR, flush=True)
    print("upstream commit : %s" % pinned_commit(), flush=True)
    print("datasets        : %d" % len(names), flush=True)
    print("predictions     : %s" % args.out, flush=True)

    attempted = []
    errors = {}
    per_table = {}
    total_pred = 0
    started = time.time()
    with args.out.open("w", encoding="utf-8") as fout:
        for idx, slug in enumerate(names, 1):
            attempted.append(slug)
            ds_dir = args.datasets_dir / slug
            try:
                result = run_method(ds_dir, slug, timeout=args.timeout, force=args.force)
                per_table[slug] = {
                    "detections": len(result.cells),
                    "rules": result.rules,
                    "elapsed": result.elapsed,
                    "error": result.error,
                }
                if result.error:
                    errors[slug] = result.error
                for i, j in sorted(result.cells):
                    fout.write(
                        json.dumps(
                            {"dataset": slug, "detector": DETECTOR, "row": int(i), "col": int(j)}
                        )
                        + "\n"
                    )
                    total_pred += 1
                if args.verbose:
                    print(
                        "[%d/%d] %s: %d cells, rules=%d, error=%s"
                        % (idx, len(names), slug, len(result.cells), result.rules, bool(result.error)),
                        flush=True,
                    )
            except Exception as exc:
                errors[slug] = "%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc())
                per_table[slug] = {"detections": 0, "rules": 0, "elapsed": 0, "error": errors[slug]}
            if idx % 25 == 0 or idx == len(names):
                print(
                    "  ...%d/%d datasets, %d predictions, %d tables with notes/errors, %.1fs"
                    % (idx, len(names), total_pred, len(errors), time.time() - started),
                    flush=True,
                )

    Path(str(args.out) + ".ran.json").write_text(
        json.dumps({"per_detector": {DETECTOR: attempted}, "all_loaded": attempted}),
        encoding="utf-8",
    )
    Path(str(args.out) + ".errors.json").write_text(
        json.dumps(errors, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(str(args.out) + ".tables.json").write_text(
        json.dumps(per_table, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=== HoloClean summary ===", flush=True)
    print("attempted datasets : %d" % len(attempted), flush=True)
    print("predicted cells    : %d" % total_pred, flush=True)
    print("tables noted/error : %d" % len(errors), flush=True)
    print("elapsed seconds    : %.1f" % (time.time() - started), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
