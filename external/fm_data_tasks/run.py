"""Run authors' fm_data_tasks error-detection prompt on the REIN benchmark.

This runner vendors HazyResearch/fm_data_tasks unchanged under
``fm_data_tasks_repo/`` and drives its upstream error-detection serialization:

    <entire row as "col: val. ...">

    Is there an error in <target col>: <target value>? Yes or No?

Upstream called GPT-3 via Manifest.  ``adapter.py`` preserves that Manifest-style
``run(prompt)`` seam while delegating to our Azure ``LLMClient``.  Default model is
``gpt-4o-mini`` because text-davinci-002/003 are deprecated and not served here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
UPSTREAM = HERE / "fm_data_tasks_repo"
CACHE_DIR = HERE / "cache"
CACHE_VERSION = "v8"
DETECTOR = "fm_data_tasks"
SEED = 20240601

sys.path.insert(0, str(REPO_ROOT / "external" / "rein"))
sys.path.insert(0, str(REPO_ROOT / "external" / "synth_fewshot"))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import adapter  # noqa: E402
from fm_data_tasks.utils import data_utils  # noqa: E402


def _upstream_commit() -> str:
    head = UPSTREAM / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            return (UPSTREAM / ".git" / ref[5:]).read_text(encoding="utf-8").strip()
        return ref
    except Exception:
        return "unknown"


def _safe_model(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", model)


def _cache_path(ds: str, args: argparse.Namespace) -> Path:
    return CACHE_DIR / (
        f"{CACHE_VERSION}__{ds}__{_safe_model(args.model)}__s{args.shots}"
        f"__inst{int(args.add_task_instruction)}.json"
    )


def _query_text(row_values: list[str], headers: list[str], target_col: str) -> str:
    row = pd.Series({h: str(v) for h, v in zip(headers, row_values)})
    row["col_name"] = target_col
    row["is_clean"] = 1
    return data_utils.serialize_error_detection(
        row,
        add_prefix=False,
        instruction="",
        suffix="?",
        sep_tok=".",
        nan_tok="nan",
    ).strip()


def _is_yes(text: str) -> bool:
    m = re.search(r"[A-Za-z]+", text.strip())
    return bool(m and m.group(0).lower().startswith("yes"))


def _build_synthetic_examples(ds_dir: Path, df: pd.DataFrame, headers: list[str], shots: int):
    if shots <= 0:
        return "", [], []
    import inject as SI

    protected = SI.load_protected_cells(ds_dir / "gt_cells.jsonl")
    df_syn, injections = SI.build_synthetic_table(df, protected, seed=SEED)
    synth_cells = [[int(x["i"]), int(x["j"])] for x in injections]
    synth_set = {(int(x["i"]), int(x["j"])) for x in injections}
    labeled_rows = SI.select_train_rows(df, budget=max(20, shots))

    records = []
    for i in labeled_rows:
        values = [str(df_syn.iat[i, j]) for j in range(len(headers))]
        for j, col in enumerate(headers):
            records.append({
                "text": _query_text(values, headers, col),
                "label_str": "Yes\n" if (i, j) in synth_set else "No\n",
            })
    train = pd.DataFrame(records)
    if train.empty:
        return "", labeled_rows, synth_cells
    np.random.seed(SEED)
    sample = data_utils.sample_train_data(train, min(shots, len(train)))
    examples = "\n\n".join(
        (txt + label).strip() for txt, label in zip(sample["text"], sample["label_str"])
    )
    return examples + "\n", labeled_rows, synth_cells


def _prompt(base_query: str, examples: str, add_task_instruction: bool) -> str:
    query = (examples + "\n" + base_query).strip() if examples else base_query
    if add_task_instruction:
        return f"Classify errors. nan values are not errors. {query}"
    return query


def run_table(ds: str, args: argparse.Namespace) -> dict:
    cache_p = _cache_path(ds, args)
    if cache_p.exists() and not args.no_cache:
        try:
            rec = json.loads(cache_p.read_text(encoding="utf-8"))
            rec["cached"] = True
            return rec
        except Exception:
            pass

    ds_dir = Path(args.datasets_dir) / ds
    rec = {
        "dataset": ds,
        "status": "ok",
        "cells": [],
        "labeled_rows": [],
        "synth_cells": [],
        "calls": 0,
        "prompt_chars": 0,
        "completion_chars": 0,
        "errors": 0,
        "time_s": 0.0,
        "cached": False,
    }
    t0 = time.time()
    try:
        df = C.load_rein_df(ds_dir / "dirty.csv")
        headers = list(df.columns)
        examples, labeled_rows, synth_cells = _build_synthetic_examples(
            ds_dir, df, headers, args.shots
        )
        rec["labeled_rows"] = labeled_rows
        rec["synth_cells"] = synth_cells

        manifest = adapter.ManifestLLMClient(args.model, args.reasoning_effort)
        cells: set[tuple[int, int]] = set()
        for i in range(len(df)):
            values = [str(df.iat[i, j]) for j in range(len(headers))]
            if not any(v.strip() for v in values):
                continue
            for j, col in enumerate(headers):
                base = _query_text(values, headers, col)
                prompt = _prompt(base, examples, args.add_task_instruction)
                pred = manifest.run(prompt)
                rec["calls"] += 1
                rec["prompt_chars"] += len(prompt)
                rec["completion_chars"] += len(pred)
                if not pred:
                    rec["errors"] += 1
                elif _is_yes(pred):
                    cells.add((i, j))
        rec["cells"] = [[int(i), int(j)] for i, j in sorted(cells)]
    except Exception as e:
        rec["status"] = f"error: {type(e).__name__}: {e}"
    rec["time_s"] = round(time.time() - t0, 1)
    rec["upstream_commit"] = _upstream_commit()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(rec), encoding="utf-8")
    return rec


def select_datasets(args: argparse.Namespace) -> list[str]:
    manifest = json.loads((Path(args.datasets_dir) / "_manifest.json").read_text())
    datasets = sorted(manifest["datasets"])
    return datasets[: args.limit] if args.limit else datasets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shots", type=int, default=0, help="synthetic few-shot examples")
    ap.add_argument("--tag", default=DETECTOR)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--add-task-instruction", dest="add_task_instruction", action="store_true", default=True)
    ap.add_argument("--no-task-instruction", dest="add_task_instruction", action="store_false")
    args = ap.parse_args()

    datasets = select_datasets(args)
    print(
        f"fm_data_tasks upstream ED: {len(datasets)} tables, model={args.model}, "
        f"shots={args.shots}, concurrency={args.concurrency}, commit={_upstream_commit()}"
    )
    print("Model substitution: upstream GPT-3 text-davinci-002/003 -> Azure gpt-4o-mini")

    records: list[dict] = []
    done = {"n": 0, "cached": 0, "cells": 0, "calls": 0, "errors": 0}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_table, ds, args): ds for ds in datasets}
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            with lock:
                done["n"] += 1
                done["cached"] += int(rec.get("cached", False))
                done["cells"] += len(rec.get("cells", []))
                done["calls"] += int(rec.get("calls", 0))
                done["errors"] += int(rec.get("errors", 0))
                print(
                    f"  [{done['n']}/{len(datasets)}] {rec['dataset'][:20]:20} "
                    f"flags={len(rec.get('cells', [])):4} calls={rec.get('calls', 0):5} "
                    f"{rec.get('status', 'ok')[:22]:22} {rec.get('time_s', 0):6.1f}s",
                    flush=True,
                )

    records.sort(key=lambda r: r["dataset"])
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.RESULTS_DIR / f"predictions_{args.tag}.jsonl"
    n_pred = 0
    ran = [r["dataset"] for r in records]
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            for i, j in rec.get("cells", []):
                f.write(json.dumps({
                    "dataset": rec["dataset"],
                    "detector": args.tag,
                    "row": int(i),
                    "col": int(j),
                }) + "\n")
                n_pred += 1
    Path(str(out_path) + ".ran.json").write_text(
        json.dumps({"per_detector": {args.tag: ran}, "all_loaded": ran}),
        encoding="utf-8",
    )
    if args.shots:
        Path(str(out_path) + ".labeled_rows.json").write_text(
            json.dumps({r["dataset"]: r.get("labeled_rows", []) for r in records}),
            encoding="utf-8",
        )
        Path(str(out_path) + ".synthetic_cells.json").write_text(
            json.dumps({r["dataset"]: r.get("synth_cells", []) for r in records}),
            encoding="utf-8",
        )
    meta_path = Path(str(out_path) + ".meta.json")
    meta_path.write_text(json.dumps({
        "detector": args.tag,
        "upstream_repo": "https://github.com/HazyResearch/fm_data_tasks",
        "upstream_commit": _upstream_commit(),
        "cache_version": CACHE_VERSION,
        "model": args.model,
        "model_deviation": "text-davinci-002/003 deprecated/not served; substituted gpt-4o-mini",
        "shots": args.shots,
        "datasets": len(records),
        "predictions": n_pred,
        "calls": sum(int(r.get("calls", 0)) for r in records),
        "prompt_chars": sum(int(r.get("prompt_chars", 0)) for r in records),
        "completion_chars": sum(int(r.get("completion_chars", 0)) for r in records),
        "errors": sum(int(r.get("errors", 0)) for r in records),
        "records": records,
    }, indent=2), encoding="utf-8")

    print("=== fm_data_tasks summary ===")
    print(f"tables      : {len(records)} ({done['cached']} cached)")
    print(f"calls       : {done['calls']} ({done['errors']} empty/error responses)")
    print(f"flags       : {done['cells']} cells -> {out_path} ({n_pred} rows)")
    print(f"run sidecar : {out_path}.ran.json")
    print(f"meta        : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
