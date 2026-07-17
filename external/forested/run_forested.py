"""Runner: ForestED (LLM-induced decision-tree ensemble) on benchmark/merged_v2.

ForestED (mengqiwang1/ForestED, "Ensembling LLM-Induced Decision Trees for
Explainable and Robust Error Detection") is a **few-shot** error detector: from a
clean/dirty pair it GP-samples ~5% of rows as labeled examples, prompts an LLM to
synthesize a *decision-tree spec* of rule/GNN nodes per partition, trains any GNN
nodes, runs the forest over the full table, and ensembles partitions by majority
vote. We run it **honestly** (no real benchmark labels): the "dirty" table is the
real table with a fixed **synthetic** injection (``external/synth_fewshot/``) and
the "clean" oracle is the original, so its ground truth = the synthetic errors
only. Detection = cells the consensus marks erroneous; the GP-sampled rows are
held out and the synthetic cells neutralized at scoring time.

Its OpenAI seam is routed to our Azure ``LLMClient`` by ``adapter.py``; the
consensus voter is forced onto CPU (it hardcodes ``device="cuda"``).

Examples (forested venv)::

    external/forested/venv/bin/python external/forested/run_forested.py --limit 1
    external/forested/venv/bin/python external/forested/run_forested.py \
        --scope sample --clean-sample 100 --workers 6 --model gpt-5.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATASETS_DIR = REPO_ROOT / "external" / "rein" / "rein-benchmark" / "datasets"
RESULTS_DIR = REPO_ROOT / "external" / "rein" / "results"
CACHE_DIR = HERE / "cache"
WORK_DIR = HERE / "work"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "external" / "rein"))
sys.path.insert(0, str(REPO_ROOT / "external" / "synth_fewshot"))

import common as C  # noqa: E402
import inject as SI  # noqa: E402
import adapter  # noqa: E402

_DF = None  # ForestED module, loaded once with the LLMClient seam patched
_CATALOG = None
_GPSAMPLE = None


def _load(model: str, effort: str):
    global _DF, _CATALOG, _GPSAMPLE
    if _DF is None:
        _DF = adapter.load_decision_forest(model=model, reasoning_effort=effort)
        sys.path.insert(0, str(HERE / "ForestED"))
        from data_catalog import build_data_catalog
        from gp_sampling import sample_by_gp_uncertainty
        _CATALOG, _GPSAMPLE = build_data_catalog, sample_by_gp_uncertainty
    return _DF


def _read_csv(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def _json_default(o):
    """Coerce numpy scalar types (int64/float64) so the catalog is JSON-serializable."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_one(ds: str, args) -> dict:
    cache_p = CACHE_DIR / f"{ds}__{args.model.replace('/', '_')}.json"
    if cache_p.exists() and not args.no_cache:
        try:
            return json.loads(cache_p.read_text())
        except Exception:
            pass

    ds_dir = DATASETS_DIR / ds
    rec = {"dataset": ds, "cells": [], "dev_rows": [], "synth_cells": [],
           "status": "ok", "n_cells": 0, "time_s": 0.0}
    t0 = time.time()
    try:
        DF = _load(args.model, args.reasoning_effort)
        df = _read_csv(ds_dir / "dirty.csv")
        headers = list(df.columns)
        col_idx = {c: j for j, c in enumerate(headers)}
        protected = SI.load_protected_cells(ds_dir / "gt_cells.jsonl")
        df_syn, injs = SI.build_synthetic_table(df, protected)
        rec["synth_cells"] = [[x["i"], x["j"]] for x in injs]

        work = WORK_DIR / ds
        work.mkdir(parents=True, exist_ok=True)
        clean_p, dirty_p = work / "clean.csv", work / "dirty.csv"
        df.to_csv(clean_p, index=False)
        df_syn.to_csv(dirty_p, index=False)

        cat = _CATALOG(str(dirty_p))
        cat_p = work / "catalog.json"
        cat_p.write_text(json.dumps(cat, default=_json_default))

        samp_p = work / "sample.csv"
        frac = max(args.sample_frac, args.min_sample / max(1, len(df)))
        frac = min(0.9, frac)
        # PCA components must be <= min(n_rows, n_features); cap for tiny tables.
        pca = max(2, min(20, len(df) - 2, len(headers)))
        _GPSAMPLE(str(dirty_p), output_csv=str(samp_p), sample_frac=frac,
                  pca_components=pca)
        try:
            srows = _read_csv(samp_p)
            rec["dev_rows"] = sorted(int(x) for x in srows["sample_row_id"].tolist())
        except Exception:
            rec["dev_rows"] = []

        outdir = work / "out"
        DF.run_pipeline(str(clean_p), str(dirty_p), str(samp_p), str(outdir),
                        args.model, str(cat_p), voting="majority", pred_model="gnn")

        cells = _extract_cells(outdir, headers, col_idx, len(df))
        rec["cells"] = cells
    except Exception as e:
        rec["status"] = f"error: {type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-1200:]

    rec["n_cells"] = len(rec["cells"])
    rec["time_s"] = round(time.time() - t0, 1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(rec))
    return rec


def _extract_cells(outdir: Path, headers, col_idx, n_rows) -> list:
    """Prefer the ensemble consensus matrix; fall back to unioning per-partition
    reports. Returns [[row, col], ...] for cells predicted erroneous."""
    npz = outdir / "consensus_predictions_matrix.npz"
    if npz.exists():
        try:
            m = np.load(npz)["predictions"]
            return [[int(i), int(j)] for i, j in zip(*np.where(m == 1))]
        except Exception:
            pass
    cells = set()
    for rep in outdir.glob("partition_*/full_table_report.csv"):
        try:
            r = pd.read_csv(rep)
            for _, row in r[r["prediction"] == 1].iterrows():
                c = row["column"]
                if c in col_idx:
                    cells.add((int(row["row_id"]), col_idx[c]))
        except Exception:
            continue
    return [[i, j] for (i, j) in sorted(cells)]


def _labeled(ds):
    ae = DATASETS_DIR / ds / "actual_errors.csv"
    return ae.exists() and ae.stat().st_size > 0


def select_datasets(args):
    manifest = json.loads((DATASETS_DIR / "_manifest.json").read_text())
    all_ds = sorted(manifest["datasets"])
    labeled = [d for d in all_ds if _labeled(d)]
    clean = [d for d in all_ds if not _labeled(d)]
    if args.scope == "all":
        sel = all_ds
    elif args.scope == "sample":
        import random
        random.Random(42).shuffle(clean)
        sel = sorted(labeled + clean[: args.clean_sample])
    else:
        sel = labeled
    return sel[: args.limit] if args.limit else sel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["labeled", "all", "sample"], default="sample")
    ap.add_argument("--clean-sample", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--sample-frac", type=float, default=0.05)
    ap.add_argument("--min-sample", type=int, default=8, help="floor on sampled rows")
    ap.add_argument("--tag", default="forested")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    datasets = select_datasets(args)
    print(f"ForestED: {len(datasets)} tables (scope={args.scope}, workers={args.workers}, "
          f"model={args.model})")
    _load(args.model, args.reasoning_effort)  # warm the module before threads

    records, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, ds, args): ds for ds in datasets}
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            done += 1
            print(f"  [{done}/{len(datasets)}] {rec['dataset'][:18]:18} "
                  f"cells={rec['n_cells']:4} {rec['status'][:16]:16} {rec['time_s']:6.1f}s",
                  flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"predictions_{args.tag}.jsonl"
    n_pred = 0
    ran, labeled_rows, synth = [], {}, {}
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            ds = rec["dataset"]
            ran.append(ds)
            labeled_rows[ds] = rec.get("dev_rows", [])
            synth[ds] = rec.get("synth_cells", [])
            for (i, j) in rec["cells"]:
                f.write(json.dumps({"dataset": ds, "detector": args.tag,
                                    "row": int(i), "col": int(j)}) + "\n")
                n_pred += 1
    Path(str(out_path) + ".ran.json").write_text(
        json.dumps({"per_detector": {args.tag: ran}, "all_loaded": ran}))
    Path(str(out_path) + ".labeled_rows.json").write_text(json.dumps(labeled_rows))
    Path(str(out_path) + ".synthetic_cells.json").write_text(json.dumps(synth))

    errs = [r for r in records if r["status"] != "ok"]
    print("=== run_forested summary ===")
    print(f"tables   : {len(records)} ({len(errs)} non-ok)")
    print(f"cells    : {sum(r['n_cells'] for r in records)} -> {out_path} ({n_pred} rows)")
    print(f"LLM usage: {adapter.usage()}")
    if errs:
        print("non-ok  :", {r['dataset'][:14]: r['status'][:30] for r in errs[:6]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
