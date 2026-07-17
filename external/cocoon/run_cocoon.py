"""Runner: authors' upstream Cocoon on the v8 REIN benchmark.

This intentionally drives the vendored upstream code in ``cocoon_repo/`` rather
than the earlier hand-written prompt. Programmatic entry point used: Cocoon's
``CleanUnusual.run`` node, which proposes per-column value-cleaning mappings from
ordered distinct values. Detection is every cell whose value is rewritten by the
mapping. The upstream LLM seam (``call_llm_chat``) is patched by ``adapter.py`` to
route through ``src.core.llm_client.LLMClient``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATASETS_DIR = REPO_ROOT / "external" / "rein" / "rein-benchmark" / "datasets"
RESULTS_DIR = REPO_ROOT / "external" / "rein" / "results"
CACHE_DIR = HERE / "cache"
BENCHMARK_TAG = "v8"
UPSTREAM_COMMIT = "410a4f7"
DEFAULT_MODEL = "gpt-4o"  # upstream defaults to gpt-4-turbo; gpt-4o is served closest match
MAX_DISTINCT = 100
MIN_NONEMPTY = 2

sys.path.insert(0, str(REPO_ROOT / "external" / "rein"))
import common as C  # noqa: E402

sys.path.insert(0, str(HERE))
import adapter  # noqa: E402

_COCOON = None


def _load(model: str, effort: str):
    global _COCOON
    if _COCOON is None:
        _COCOON = adapter.load_cocoon(model=model, reasoning_effort=effort)
    return _COCOON


def _cache_path(ds: str, model: str) -> Path:
    safe_model = model.replace("/", "_").replace(",", "_")
    return CACHE_DIR / BENCHMARK_TAG / f"{ds}__{safe_model}__upstream_{UPSTREAM_COMMIT}.json"


def purge_stale_cache() -> None:
    """Remove old cache files not nested under the v8 benchmark tag."""
    if not CACHE_DIR.exists():
        return
    for child in CACHE_DIR.iterdir():
        if child.name == BENCHMARK_TAG:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _read_csv(path: Path) -> pd.DataFrame:
    return C.load_rein_df(path)


def _ordered_distinct_df(df: pd.DataFrame, col: str, max_distinct: int = MAX_DISTINCT) -> pd.DataFrame:
    vals = df[col].fillna("").astype(str)
    vals = vals[vals.str.len() < 200]
    vc = vals.value_counts(dropna=False)
    shown = list(vc.head(max_distinct).index)
    return pd.DataFrame({col: shown})


def _eligible_columns(df: pd.DataFrame, max_distinct: int = MAX_DISTINCT) -> list[str]:
    cols: list[str] = []
    n_rows = max(1, len(df))
    for col in df.columns:
        vals = df[col].fillna("").astype(str)
        nonempty = vals[vals.str.strip() != ""]
        if len(nonempty) < MIN_NONEMPTY:
            continue
        distinct = nonempty.nunique(dropna=False)
        if distinct < 2 or distinct > max_distinct:
            # Mirrors upstream CleanUnusual.extract: high-cardinality columns are
            # treated as too complex to clean automatically.
            continue
        if distinct == n_rows and n_rows > 20:
            continue
        cols.append(col)
    return cols


def _normalize_mapping(mapping) -> dict[str, str]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in mapping.items():
        if k is None or v is None:
            continue
        ks, vs = str(k), str(v)
        if ks != vs and ks.strip() != vs.strip():
            out[ks] = vs
    return out


def _cells_from_mapping(df: pd.DataFrame, col: str, mapping: dict[str, str]) -> list[list[int]]:
    j = list(df.columns).index(col)
    vals = df[col].fillna("").astype(str)
    bad = set(mapping)
    return [[int(i), int(j)] for i, v in vals.items() if str(v) in bad]


def _run_clean_node(cocoon, col: str, sample_values: pd.DataFrame, reason: str):
    node = cocoon.CleanUnusual()
    extract_output = (col, sample_values, reason, MAX_DISTINCT, len(sample_values))
    return node.run(extract_output, use_cache=False)


def run_one(ds: str, args) -> dict:
    cache_p = _cache_path(ds, args.model)
    if cache_p.exists() and not args.no_cache:
        try:
            return json.loads(cache_p.read_text())
        except Exception:
            pass

    rec = {
        "dataset": ds,
        "status": "ok",
        "cells": [],
        "mappings": {},
        "columns_attempted": [],
        "time_s": 0.0,
    }
    t0 = time.time()
    try:
        cocoon = _load(args.model, args.reasoning_effort)
        df = _read_csv(DATASETS_DIR / ds / "dirty.csv")
        reason = (
            "Potential weird characters, typos, or redundant/inconsistent "
            "representations of the same concept."
        )
        cells: set[tuple[int, int]] = set()
        for col in _eligible_columns(df, args.max_distinct):
            rec["columns_attempted"].append(col)
            sample_values = _ordered_distinct_df(df, col, args.max_distinct)
            if len(sample_values) < 2:
                continue
            out = _run_clean_node(cocoon, col, sample_values, reason)
            mapping = _normalize_mapping(out.get("mapping", {}) if isinstance(out, dict) else {})
            if mapping:
                rec["mappings"][col] = mapping
                for i, j in _cells_from_mapping(df, col, mapping):
                    cells.add((i, j))
        rec["cells"] = [[i, j] for (i, j) in sorted(cells)]
    except Exception as e:
        rec["status"] = f"error: {type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-2000:]

    rec["time_s"] = round(time.time() - t0, 1)
    rec["n_cells"] = len(rec["cells"])
    rec["usage"] = adapter.usage_snapshot()
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def select_datasets(limit: int) -> list[str]:
    manifest = json.loads((DATASETS_DIR / "_manifest.json").read_text())
    datasets = sorted(manifest["datasets"])
    return datasets[:limit] if limit else datasets


def write_outputs(records: list[dict], tag: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"predictions_{tag}.jsonl"
    n_pred = 0
    ran: list[str] = []
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in sorted(records, key=lambda r: r["dataset"]):
            ds = rec["dataset"]
            ran.append(ds)
            for i, j in rec.get("cells", []):
                f.write(json.dumps({"dataset": ds, "detector": tag, "row": int(i), "col": int(j)}) + "\n")
                n_pred += 1
    Path(str(out_path) + ".ran.json").write_text(
        json.dumps({"per_detector": {tag: ran}, "all_loaded": ran})
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--concurrency", "--workers", dest="concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="cocoon")
    ap.add_argument("--max-distinct", type=int, default=MAX_DISTINCT)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--purge-stale-cache", action="store_true")
    args = ap.parse_args()

    if args.purge_stale_cache:
        purge_stale_cache()

    datasets = select_datasets(args.limit)
    print(f"Cocoon upstream: {len(datasets)} datasets (commit={UPSTREAM_COMMIT}, model={args.model}, "
          f"concurrency={args.concurrency}, benchmark={BENCHMARK_TAG})", flush=True)
    print("Entry point: cocoon_data.CleanUnusual.run via adapter.load_cocoon(); "
          "detection = mapping-induced cell rewrites", flush=True)
    _load(args.model, args.reasoning_effort)

    records: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, ds, args): ds for ds in datasets}
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            done += 1
            print(f"  [{done}/{len(datasets)}] {rec['dataset'][:18]:18} "
                  f"cols={len(rec.get('columns_attempted', [])):2} cells={rec.get('n_cells', 0):4} "
                  f"{rec['status'][:35]:35} {rec['time_s']:6.1f}s", flush=True)

    out_path = write_outputs(records, args.tag)
    errs = [r for r in records if r.get("status") != "ok"]
    total_cells = sum(int(r.get("n_cells", 0)) for r in records)
    usage = adapter.usage_snapshot()
    summary = {
        "upstream_commit": UPSTREAM_COMMIT,
        "benchmark_tag": BENCHMARK_TAG,
        "model": args.model,
        "entry_point": "cocoon_data.CleanUnusual.run",
        "datasets": len(records),
        "errors": len(errs),
        "predicted_cells": total_cells,
        "usage": usage,
        "non_ok": {r["dataset"]: r.get("status") for r in errs[:25]},
    }
    Path(str(out_path) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print("=== run_cocoon summary ===")
    print(f"tables      : {len(records)} ({len(errs)} non-ok)")
    print(f"cells       : {total_cells} -> {out_path}")
    print(f"ran sidecar : {out_path}.ran.json")
    print(f"LLM usage   : {adapter.usage_summary()}")
    if records:
        print(f"per table   : calls={usage['calls']/len(records):.2f}, est ${usage['cost']/len(records):.4f}")
    if errs:
        print("non-ok      :", {r["dataset"][:14]: r.get("status", "")[:60] for r in errs[:8]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
