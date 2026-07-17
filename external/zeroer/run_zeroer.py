"""Run upstream ZeroER on the REIN benchmark tables.

Outputs ``external/rein/results/predictions_zeroer.jsonl`` plus the required
``.ran.json`` scope sidecar. ZeroER is used unsupervised; no benchmark labels or
clean tables are read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REIN_DIR = REPO_ROOT / "external" / "rein"
sys.path.insert(0, str(REIN_DIR))

import common as C  # type: ignore  # noqa: E402
from adapter import detect_duplicate_rows  # noqa: E402

DETECTOR = "zeroer"


def _dataset_names(ds_root: Path, limit: int = 0) -> list[str]:
    manifest_path = ds_root / "_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = sorted(manifest["datasets"])
    return names[:limit] if limit else names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    parser.add_argument("--out", default=str(C.RESULTS_DIR / "predictions_zeroer.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=0, help="smoke-test only; do not use for full run"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None, help="explicit smoke-test dataset slugs"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--run-transitivity", action="store_true")
    args = parser.parse_args()

    ds_root = Path(args.datasets_dir)
    ds_names = args.datasets if args.datasets else _dataset_names(ds_root, args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pred = 0
    errors: dict[str, str] = {}
    per_dataset_stats: dict[str, dict] = {}
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        for k, ds in enumerate(ds_names, 1):
            t0 = time.time()
            try:
                df = C.load_rein_df(ds_root / ds / "dirty.csv")
                result = detect_duplicate_rows(
                    df,
                    threshold=args.threshold,
                    run_transitivity=args.run_transitivity,
                )
                for row, col in sorted(result.cells):
                    fout.write(
                        json.dumps(
                            {"dataset": ds, "detector": DETECTOR, "row": row, "col": col},
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    n_pred += 1
                stat = dict(result.stats)
                stat["runtime_s"] = round(time.time() - t0, 4)
                per_dataset_stats[ds] = stat
                if str(stat.get("status")) != "ok":
                    print(f"[warn] {DETECTOR} on {ds}: {stat.get('status')}", flush=True)
            except Exception as exc:
                errors[ds] = f"{type(exc).__name__}: {exc}"
                per_dataset_stats[ds] = {"status": "runner_error", "error": errors[ds]}
                print(f"[err ] {DETECTOR} on {ds}: {errors[ds]}", flush=True)
            if k % 25 == 0 or k == len(ds_names):
                elapsed = time.time() - started
                print(
                    f"  ...{k}/{len(ds_names)} datasets, {n_pred} cells, "
                    f"{len(errors)} runner errors, {elapsed:.1f}s",
                    flush=True,
                )

    ran_path = Path(str(out_path) + ".ran.json")
    ran_path.write_text(
        json.dumps({"per_detector": {DETECTOR: ds_names}, "all_loaded": ds_names}, indent=2),
        encoding="utf-8",
    )
    meta_path = Path(str(out_path) + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "detector": DETECTOR,
                "datasets_attempted": len(ds_names),
                "cells": n_pred,
                "runner_errors": errors,
                "n_runner_errors": len(errors),
                "threshold": args.threshold,
                "run_transitivity": args.run_transitivity,
                "runtime_s": round(time.time() - started, 4),
                "per_dataset": per_dataset_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== run_zeroer summary ===", flush=True)
    print(
        f"{DETECTOR:>16}: {n_pred:6d} cells over {len(ds_names):4d} datasets, "
        f"{time.time() - started:6.1f}s, {len(errors)} runner errors",
        flush=True,
    )
    print(f"predictions -> {out_path} ({n_pred} rows)", flush=True)
    print(f"scope -> {ran_path}", flush=True)
    print(f"metadata -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
