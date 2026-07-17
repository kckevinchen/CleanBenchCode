#!/usr/bin/env python3
"""Run a validated ADR'24 method over the REIN materialized benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ADR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADR_DIR.parents[1]
REIN_DIR = REPO_ROOT / "external" / "rein"
DATASETS_DIR = REIN_DIR / "rein-benchmark" / "datasets"
RESULTS_DIR = REIN_DIR / "results"

sys.path.insert(0, str(ADR_DIR))
from adapter import detector_name, pinned_commit, run_method  # noqa: E402


def dataset_names(ds_root: Path) -> list[str]:
    manifest_path = ds_root / "_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return sorted(manifest["datasets"])
    return sorted(p.name for p in ds_root.iterdir() if p.is_dir() and (p / "dirty.csv").exists())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--method",
        choices=["horizon", "relative_trust", "unified", "holistic", "bigdansing", "daisy"],
        default="horizon",
    )
    ap.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test only; do not use for full run")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--force", action="store_true", help="ignore per-table cache")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    detector = detector_name(args.method)
    out_path = args.out or (RESULTS_DIR / f"predictions_{detector}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = dataset_names(args.datasets_dir)
    if args.limit:
        names = names[: args.limit]

    print("=== ADR'24 run ===", flush=True)
    print(f"method          : {args.method}", flush=True)
    print(f"detector        : {detector}", flush=True)
    print(f"upstream commit : {pinned_commit()}", flush=True)
    print(f"datasets        : {len(names)}", flush=True)
    print(f"predictions     : {out_path}", flush=True)

    attempted: list[str] = []
    errors: dict[str, str] = {}
    total_pred = 0
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        for idx, slug in enumerate(names, 1):
            attempted.append(slug)
            ds_dir = args.datasets_dir / slug
            try:
                result = run_method(
                    args.method,
                    ds_dir,
                    slug,
                    timeout=args.timeout,
                    force=args.force,
                )
                if result.error:
                    errors[slug] = result.error
                for i, j in sorted(result.cells):
                    fout.write(
                        json.dumps(
                            {"dataset": slug, "detector": detector, "row": int(i), "col": int(j)}
                        )
                        + "\n"
                    )
                    total_pred += 1
                if args.verbose:
                    print(
                        f"[{idx}/{len(names)}] {slug}: {len(result.cells)} cells, "
                        f"rules={result.rules}, error={bool(result.error)}",
                        flush=True,
                    )
            except Exception as exc:
                errors[slug] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            if idx % 25 == 0 or idx == len(names):
                print(
                    f"  ...{idx}/{len(names)} datasets, {total_pred} predictions, "
                    f"{len(errors)} tables with notes/errors, {time.time() - started:.1f}s",
                    flush=True,
                )

    Path(str(out_path) + ".ran.json").write_text(
        json.dumps({"per_detector": {detector: attempted}, "all_loaded": attempted}),
        encoding="utf-8",
    )
    Path(str(out_path) + ".errors.json").write_text(
        json.dumps(errors, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=== ADR'24 summary ===", flush=True)
    print(f"attempted datasets : {len(attempted)}", flush=True)
    print(f"predicted cells    : {total_pred}", flush=True)
    print(f"tables noted/error : {len(errors)}", flush=True)
    print(f"elapsed seconds    : {time.time() - started:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
