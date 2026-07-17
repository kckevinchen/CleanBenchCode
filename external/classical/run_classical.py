"""Runner: classical (non-LLM) detectors on the materialized benchmark datasets.

Zero-shot, label-free, deterministic. Reuses the REIN harness' dataset materialization
(``external/rein/rein-benchmark/datasets/``) and writes predictions in the shared
schema (``{dataset, detector, row, col}``) into ``external/rein/results/`` so
``external/rein/score.py`` and ``analyze_by_type.py`` pick them up unchanged.

Examples (repo root, rein venv with pyod installed)::

    python external/classical/run_classical.py --detector pattern
    python external/classical/run_classical.py --detector openrefine --detector pyod
    python external/classical/run_classical.py --detector pattern --limit 20   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Pin BLAS threads (defensive; pyod/sklearn).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_REIN_DIR = Path(__file__).resolve().parents[1] / "rein"
sys.path.insert(0, str(_REIN_DIR))
import common as C  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as D  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", action="append", required=True, choices=list(D.DETECTORS),
                    help="detector(s) to run (repeatable)")
    ap.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="override output filename tag (single detector)")
    args = ap.parse_args()

    ds_root = Path(args.datasets_dir)
    manifest = json.loads((ds_root / "_manifest.json").read_text())
    ds_names = sorted(manifest["datasets"])
    if args.limit:
        ds_names = ds_names[: args.limit]

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for det in args.detector:
        tag = args.tag if (args.tag and len(args.detector) == 1) else det
        out_path = C.RESULTS_DIR / f"predictions_{tag}.jsonl"
        n_pred = 0
        ran: list[str] = []
        t0 = time.time()
        errors = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for k, ds in enumerate(ds_names, 1):
                ds_dir = ds_root / ds
                try:
                    df = C.load_rein_df(ds_dir / "dirty.csv")
                except Exception as e:
                    print(f"[skip] {ds}: {e}")
                    continue
                ran.append(ds)
                try:
                    cells = D.DETECTORS[det](df)
                except Exception as e:
                    errors += 1
                    print(f"[err ] {det} on {ds}: {e}")
                    continue
                for (i, j) in cells:
                    f.write(json.dumps({"dataset": ds, "detector": tag,
                                        "row": int(i), "col": int(j)}) + "\n")
                    n_pred += 1
                if k % 200 == 0:
                    print(f"  {det}: ...{k}/{len(ds_names)} ({n_pred} cells)", flush=True)
        Path(str(out_path) + ".ran.json").write_text(
            json.dumps({"per_detector": {tag: ran}, "all_loaded": ran}))
        print(f"{det:>11}: {n_pred} cells over {len(ran)} datasets, "
              f"{time.time() - t0:.1f}s, {errors} errors -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
