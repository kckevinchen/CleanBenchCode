"""Runner: evaluate TabClean on CleanBench as a cell-level detector.

TabClean (https://github.com/Wangyibo321/TabClean) is a FEW-SHOT, multi-agent
LLM pipeline that synthesizes a reusable Python *cleaning program*: it builds an
adaptive development sample from dirty/ground-truth pairs, iterates
(diagnose -> strategize -> code -> test -> review) keeping the best-scoring code,
then applies that program to the full table. We:

  * reuse the materialized REIN datasets (``dirty.csv`` + ``clean.csv``),
  * capture the dev rows it is allowed to learn from (deterministic
    ``random_state=42`` sample, same as the pipeline) so the scorer can hold
    them out (few-shot: never test on training rows),
  * run the pipeline, then take **detection = cells whose repaired value differs
    from the dirty value** in ``final_cleaned.csv`` (same "changed cells"
    semantics used for CleanAgent),
  * write predictions in the shared schema for ``external/rein/score.py``.

Its LLM seam is routed to our Azure ``LLMClient`` by ``adapter.patch()``.

Examples (from repo root)::

    python external/tabclean/run_tabclean.py --limit 1            # pilot 1 table
    python external/tabclean/run_tabclean.py --scope labeled --workers 6
    python external/tabclean/run_tabclean.py --scope sample --clean-sample 100
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

# Pin BLAS threads before pandas/numpy load (defensive; pipeline forks subprocs).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TABCLEAN_SRC = HERE / "TabClean"
DATASETS_DIR = REPO_ROOT / "external" / "rein" / "rein-benchmark" / "datasets"
RESULTS_DIR = REPO_ROOT / "external" / "rein" / "results"
CACHE_DIR = HERE / "cache"
WORK_DIR = HERE / "work"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TABCLEAN_SRC))
sys.path.insert(0, str(HERE.parent / "synth_fewshot"))

import adapter  # noqa: E402

adapter.patch()

from pipeline import build_pipeline, compute_dev_set  # noqa: E402

SYNTH_TMP = HERE.parent / "synth_fewshot" / "tmp_tabclean"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _gt_cells_path(ds: str) -> Path:
    return DATASETS_DIR / ds / "gt_cells.jsonl"


def _build_synthetic(ds: str, dirty: pd.DataFrame, density: str = "dense"):
    """Synthetic few-shot inputs for one dataset. Returns
    ``(df_syn, dirty_syn_path, dev_indices, dev_gt_text, synth_cells)``.

    Trains TabClean ONLY on synthetic injected errors (fixed src/synth recipe),
    never the real benchmark labels: ``df_syn`` is the table with synthetic errors,
    the dev rows are the ones the pipeline samples (``random_state=42``), and the
    dev GT lists only the synthetic injections in those rows. Real errors are
    never revealed and become the held-out test target. ``density`` is ``dense``
    (~half the rows) or ``realistic`` (match the table's real error count)."""
    import inject as SI  # external/synth_fewshot/inject.py

    gt = _gt_cells_path(ds)
    protected = SI.load_protected_cells(gt)
    n_errors = SI.count_real_errors(gt) if density == "realistic" else None
    df_syn, injs = SI.build_synthetic_table(dirty, protected, n_errors=n_errors)
    dev_indices = SI.select_train_rows(dirty)  # the rows profile_node will sample
    dev_gt_text = SI.dev_gt_errors_text(injs, dev_indices)
    out = SYNTH_TMP / density / ds
    out.mkdir(parents=True, exist_ok=True)
    dirty_syn_path = out / "dirty.csv"
    df_syn.to_csv(dirty_syn_path, index=False, encoding="utf-8")
    synth_cells = [[x["i"], x["j"]] for x in injs]
    return df_syn, dirty_syn_path, dev_indices, dev_gt_text, synth_cells


def _changed_cells(dirty: pd.DataFrame, cleaned: pd.DataFrame) -> list[tuple[int, int]]:
    """Cells whose repaired value differs from the dirty value (= detections)."""
    cols = [c for c in dirty.columns if c in cleaned.columns]
    n = min(len(dirty), len(cleaned))
    col_idx = {c: j for j, c in enumerate(dirty.columns)}
    out: list[tuple[int, int]] = []
    for c in cols:
        d = dirty[c].to_numpy()
        x = cleaned[c].to_numpy()
        j = col_idx[c]
        for i in range(n):
            if str(d[i]) != str(x[i]):
                out.append((i, j))
    return out


def run_one(ds: str, args) -> dict:
    """Run TabClean on one dataset; return a result record (also cached)."""
    # Namespace the cache by mode: synthetic-few-shot runs on a DIFFERENT table
    # (df_syn) with synthetic dev errors, so they must not share the real-label
    # cache (otherwise a synthetic run silently returns the real-label result).
    suffix = "__synth" if getattr(args, "synthetic_fewshot", False) else ""
    if suffix and getattr(args, "synthetic_density", "dense") == "realistic":
        suffix = "__synth_realistic"
    cache_path = CACHE_DIR / f"{ds}{suffix}.json"
    if cache_path.exists() and not args.no_cache:
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    ds_dir = DATASETS_DIR / ds
    dirty_path = ds_dir / "dirty.csv"
    gt_path = ds_dir / "clean.csv"
    rec: dict = {"dataset": ds, "changed": [], "dev_rows": [], "synth_cells": [],
                 "status": "ok", "n_changed": 0, "time_s": 0.0, "cost": 0.0, "calls": 0}
    t0 = time.time()
    usage0 = adapter.usage_snapshot()
    try:
        dirty = _read_csv(dirty_path)
        if getattr(args, "synthetic_fewshot", False):
            # Train on SYNTHETIC errors only; the pipeline runs on df_syn.
            df_syn, dirty_syn_path, dev_indices, dev_gt_text, synth_cells = _build_synthetic(
                ds, dirty, density=getattr(args, "synthetic_density", "dense"))
            pipe_dirty_path = dirty_syn_path
            dev_rows = len(dev_indices)
            compare_df = df_syn  # detection = changed vs what the pipeline saw
            rec["synth_cells"] = synth_cells
        else:
            dev_rows, dev_gt_text = compute_dev_set(
                str(dirty_path), str(gt_path), min_rows=args.dev_min_rows
            )
            dev_sample = dirty.sample(n=min(dev_rows, len(dirty)), random_state=42)
            dev_indices = sorted(int(i) for i in dev_sample.index.tolist())
            pipe_dirty_path = dirty_path
            compare_df = dirty
        rec["dev_rows"] = sorted(int(i) for i in dev_indices)

        # Namespace the per-table work dir by run mode/density so that concurrent
        # runs (e.g. synthetic dense vs realistic) never collide on WORK_DIR/<ds>.
        if getattr(args, "synthetic_fewshot", False):
            work_sub = "synth_" + getattr(args, "synthetic_density", "dense")
        else:
            work_sub = "real"
        work = WORK_DIR / work_sub / ds
        work.mkdir(parents=True, exist_ok=True)
        graph = build_pipeline()
        result = graph.invoke(
            {
                "dirty_path": str(Path(pipe_dirty_path).resolve()),
                "work_dir": str(work.resolve()),
                "dev_rows": dev_rows,
                "dev_dirty_path": "",
                "dev_gt_errors": dev_gt_text,
                "profile": "", "data_peek": "", "structural_analysis": "",
                "skills_text": "", "error_types_catalog": "",
                "diagnosis": "", "strategy": "", "prev_strategy": "",
                "cleaning_code": "", "dev_metrics": "", "dev_feedback": "",
                "eval_history": "{}", "diff_summary": "", "review": "",
                "iteration_history": "",
                "best_f1": -1.0, "best_cleaning_code": "", "best_strategy": "",
                "best_iteration": 0, "iteration": 0, "cleaned_path": "",
                "no_improve_count": 0,
            },
            {"recursion_limit": args.recursion_limit},
        )
        cleaned_path = result.get("cleaned_path", "")
        if cleaned_path and os.path.exists(cleaned_path):
            cleaned = _read_csv(Path(cleaned_path))
            rec["changed"] = _changed_cells(compare_df, cleaned)
        else:
            rec["status"] = "no_cleaned_output"
    except Exception as e:  # keep the batch alive
        rec["status"] = f"error: {type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-1500:]

    u = adapter.usage_snapshot()
    rec["n_changed"] = len(rec["changed"])
    rec["time_s"] = round(time.time() - t0, 1)
    rec["cost"] = round(u["cost"] - usage0["cost"], 4)
    rec["calls"] = u["calls"] - usage0["calls"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rec))
    return rec


def _labeled(ds: str) -> bool:
    ae = DATASETS_DIR / ds / "actual_errors.csv"
    return ae.exists() and ae.stat().st_size > 0


def select_datasets(args) -> list[str]:
    manifest = json.loads((DATASETS_DIR / "_manifest.json").read_text())
    all_ds = sorted(manifest["datasets"])
    labeled = [d for d in all_ds if _labeled(d)]
    clean = [d for d in all_ds if not _labeled(d)]
    if args.scope == "all":
        sel = all_ds
    elif args.scope == "sample":
        import random
        random.Random(42).shuffle(clean)
        sel = labeled + clean[: args.clean_sample]
        sel = sorted(sel)
    else:  # labeled
        sel = labeled
    if args.limit:
        sel = sel[: args.limit]
    return sel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["labeled", "all", "sample"], default="labeled",
                    help="labeled (217, default) | all (1002) | sample (labeled + N clean)")
    ap.add_argument("--clean-sample", type=int, default=100,
                    help="with --scope sample: how many clean tables to add")
    ap.add_argument("--limit", type=int, default=0, help="only the first N selected datasets")
    ap.add_argument("--workers", type=int, default=1, help="concurrent datasets")
    ap.add_argument("--recursion-limit", type=int, default=120)
    ap.add_argument("--dev-min-rows", type=int, default=20,
                    help="floor for TabClean's adaptive dev set (default 20, matching "
                         "raha/baran's labeling budget). Our benchmark tables are ~100 "
                         "rows, so the native floor of 100 would label the whole table "
                         "and leave no held-out test rows.")
    ap.add_argument("--max-iterations", type=int, default=0,
                    help="override pipeline MAX_ITERATIONS (0 = keep default 10)")
    ap.add_argument("--tag", default="tabclean")
    ap.add_argument("--synthetic-fewshot", action="store_true",
                    help="train ONLY on synthetic injected errors (fixed src/synth recipe), "
                         "never the real benchmark labels. Real errors become the held-out "
                         "test set (synthetic cells neutralized, dev rows held out). Scores "
                         "an honest few-shot number without value-leak; pair with --scope all.")
    ap.add_argument("--synthetic-density", choices=["dense", "realistic"], default="dense",
                    help="synthetic-fewshot density: 'dense' (~half the rows) or 'realistic' "
                         "(match each table's real error count).")
    ap.add_argument("--no-cache", action="store_true", help="ignore per-table cache")
    args = ap.parse_args()

    if args.max_iterations:
        import pipeline
        pipeline.MAX_ITERATIONS = args.max_iterations

    datasets = select_datasets(args)
    print(f"TabClean: {len(datasets)} datasets (scope={args.scope}, workers={args.workers}, "
          f"model={adapter._DEFAULT_MODEL}, effort={adapter._REASONING_EFFORT})")

    records: list[dict] = []
    done = 0

    def _go(ds):
        return run_one(ds, args)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_go, ds): ds for ds in datasets}
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                print(f"  [{done}/{len(datasets)}] {rec['dataset'][:18]:18} "
                      f"changed={rec['n_changed']:4} {rec['status'][:18]:18} "
                      f"{rec['time_s']:6.1f}s ${rec['cost']:.3f} ({rec['calls']} calls)",
                      flush=True)
    else:
        for ds in datasets:
            rec = _go(ds)
            records.append(rec)
            done += 1
            print(f"  [{done}/{len(datasets)}] {rec['dataset'][:18]:18} "
                  f"changed={rec['n_changed']:4} {rec['status'][:18]:18} "
                  f"{rec['time_s']:6.1f}s ${rec['cost']:.3f} ({rec['calls']} calls)",
                  flush=True)

    # Emit predictions + sidecars in the shared score.py schema.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"predictions_{args.tag}.jsonl"
    n_pred = 0
    ran: list[str] = []
    labeled_rows: dict[str, list[int]] = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            ds = rec["dataset"]
            ran.append(ds)
            labeled_rows[ds] = rec.get("dev_rows", [])
            for (i, j) in rec["changed"]:
                f.write(json.dumps({"dataset": ds, "detector": args.tag,
                                    "row": int(i), "col": int(j)}) + "\n")
                n_pred += 1
    Path(str(out_path) + ".ran.json").write_text(
        json.dumps({"per_detector": {args.tag: ran}, "all_loaded": ran}))
    Path(str(out_path) + ".labeled_rows.json").write_text(json.dumps(labeled_rows))
    if getattr(args, "synthetic_fewshot", False):
        synth = {r["dataset"]: r.get("synth_cells", []) for r in records}
        Path(str(out_path) + ".synthetic_cells.json").write_text(json.dumps(synth))
        print(f"synthetic cells : sidecar written for {len(synth)} datasets")

    errs = [r for r in records if r["status"] != "ok"]
    total_changed = sum(r["n_changed"] for r in records)
    # NOTE: per-record cost is unreliable under --workers>1 (each table's usage
    # window captures concurrent workers' calls too, so summing double-counts).
    # The global accumulator is the only correct total.
    global_usage = adapter.usage_snapshot()
    print("=== run_tabclean summary ===")
    print(f"datasets        : {len(records)} ({len(errs)} non-ok)")
    print(f"changed cells   : {total_changed}  -> predictions {out_path} ({n_pred} rows)")
    print(f"dev (held-out)  : sidecar written for {len(labeled_rows)} datasets")
    print(f"est. LLM cost   : ${global_usage['cost']:.2f}  ({adapter.usage_summary()})")
    if args.workers == 1:
        print(f"  (per-table sum: ${sum(r['cost'] for r in records):.2f})")
    if errs:
        print("non-ok statuses:", {r["dataset"][:14]: r["status"][:30] for r in errs[:8]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
