"""Run ED2 over the REIN benchmark materialization.

Outputs follow the v8 baseline integration contract:
``external/rein/results/predictions_ed2.jsonl`` plus ``.ran.json`` and the
few-shot sidecars consumed by ``external/rein/score.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

ED2_DIR = Path(__file__).resolve().parent
REPO_ROOT = ED2_DIR.parents[1]
REIN_DIR = REPO_ROOT / "external" / "rein"
DATASETS_DIR = REIN_DIR / "rein-benchmark" / "datasets"
RESULTS_DIR = REIN_DIR / "results"

sys.path.insert(0, str(ED2_DIR))
from adapter import (  # noqa: E402
    DEFAULT_LABELING_BUDGET,
    DEFAULT_XGB_ROUNDS,
    DETECTOR,
    SEED,
    detect_dataset_quiet,
    ensure_py3_vendor,
    pinned_commit,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-dir", default=str(DATASETS_DIR))
    ap.add_argument("--out", default=str(RESULTS_DIR / "predictions_ed2.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="smoke-test only; never use for full run")
    ap.add_argument("--labeling-budget", type=int, default=DEFAULT_LABELING_BUDGET)
    ap.add_argument("--synthetic-density", choices=["dense", "realistic"], default="dense")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--xgb-rounds", type=int, default=DEFAULT_XGB_ROUNDS)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ds_root = Path(args.datasets_dir)
    manifest_path = ds_root / "_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ds_names = sorted(manifest["datasets"])
    if args.limit:
        ds_names = ds_names[: args.limit]

    ensure_py3_vendor()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    attempted: list[str] = []
    labeled_rows: dict[str, list[int]] = {}
    synthetic_cells: dict[str, list[list[int]]] = {}
    errors: dict[str, str] = {}
    total_pred = 0
    total_labels = 0
    total_injections = 0

    print("=== ED2 run ===", flush=True)
    print(f"upstream commit : {pinned_commit()}", flush=True)
    print(f"datasets        : {len(ds_names)}", flush=True)
    print(f"labeling budget : {args.labeling_budget} cell labels/table", flush=True)
    print(f"synthetic mode  : {args.synthetic_density}, seed={args.seed}", flush=True)
    print(f"xgboost rounds  : {args.xgb_rounds}", flush=True)
    print(f"predictions     : {out_path}", flush=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, ds in enumerate(ds_names, 1):
            attempted.append(ds)
            ds_dir = ds_root / ds
            t0 = time.time()
            try:
                result = detect_dataset_quiet(
                    ds_dir,
                    labeling_budget=args.labeling_budget,
                    synthetic_density=args.synthetic_density,
                    seed=args.seed,
                    xgb_rounds=args.xgb_rounds,
                    quiet=not args.verbose,
                )
            except Exception as exc:
                result = None
                errors[ds] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

            if result is not None:
                if result.error:
                    errors[ds] = result.error
                if result.labeled_rows:
                    labeled_rows[ds] = result.labeled_rows
                synthetic_cells[ds] = result.synthetic_cells
                total_labels += result.n_labels
                total_injections += result.n_injections
                for i, j in sorted(result.cells):
                    fout.write(
                        json.dumps(
                            {
                                "dataset": ds,
                                "detector": DETECTOR,
                                "row": int(i),
                                "col": int(j),
                            }
                        )
                        + "\n"
                    )
                    total_pred += 1
            if idx % 25 == 0 or idx == len(ds_names):
                elapsed = time.time() - started
                print(
                    f"  ...{idx}/{len(ds_names)} datasets, {total_pred} predictions, "
                    f"{len(errors)} errors, {elapsed:.1f}s",
                    flush=True,
                )
            elif args.verbose:
                print(
                    f"[{idx}/{len(ds_names)}] {ds}: "
                    f"{0 if result is None else len(result.cells)} cells in {time.time() - t0:.2f}s",
                    flush=True,
                )

    Path(str(out_path) + ".ran.json").write_text(
        json.dumps({"per_detector": {DETECTOR: attempted}, "all_loaded": attempted}),
        encoding="utf-8",
    )
    Path(str(out_path) + ".labeled_rows.json").write_text(
        json.dumps(labeled_rows),
        encoding="utf-8",
    )
    Path(str(out_path) + ".synthetic_cells.json").write_text(
        json.dumps(synthetic_cells),
        encoding="utf-8",
    )
    Path(str(out_path) + ".errors.json").write_text(
        json.dumps(errors, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=== ED2 summary ===", flush=True)
    print(f"attempted datasets : {len(attempted)}", flush=True)
    print(f"predicted cells    : {total_pred}", flush=True)
    print(f"labeled rows       : {sum(len(v) for v in labeled_rows.values())}", flush=True)
    print(f"cell labels used   : {total_labels}", flush=True)
    print(f"synthetic cells    : {total_injections}", flush=True)
    print(f"tables errored     : {len(errors)}", flush=True)
    print(f"elapsed seconds    : {time.time() - started:.1f}", flush=True)
    print(f"predictions -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
