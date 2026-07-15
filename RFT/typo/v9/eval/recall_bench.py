"""Recall-only benchmark for the v9 trainee prompt.

Goal: measure how many *gold typo cells* the v9 trainee prompt
(``PROMPT_V9``) surfaces when run with ``o4-mini`` at
``reasoning_effort=low``. This deliberately skips the precision
grader -- recall is the only thing we iterate on for the trainee.

Pipeline
--------
1. Load the cell-level typo benchmark JSONL. Gold positives are
   cells with ``label == 'error'`` AND ``'typo' in error_types``.
   Each gold cell carries its ``original_value`` (the string the
   trainee must surface) and ``corrected_value`` (for eyeballing).
2. Group gold cells by ``(source, column)`` (67 columns).
3. For each such column, read the FULL column from the source CSV
   under ``data/benchmark_sources_v6`` (the benchmark only labels a
   sample of cells, so we reconstruct the true distinct-value
   listing the trainee would see in production).
4. Build the v5 column-audit user message and call o4-mini with
   ``PROMPT_V9`` as the developer message.
5. A gold cell is a HIT if its ``original_value`` appears (verbatim
   or whitespace-normalised) among the predicted ``value`` strings.
6. Report pool recall, per-column recall, and dump every miss (FN)
   so we can tell proper-noun misses (by design) from ordinary-word
   misses (a prompt bug).

Predictions are cached on disk keyed by a hash of
``(PROMPT_V9, model, reasoning_effort, column input)`` so re-runs
after a *non-prompt* change are cheap; any edit to ``PROMPT_V9``
invalidates the cache automatically.

Usage::

    python -m RFT.typo.v9.eval.recall_bench
    python -m RFT.typo.v9.eval.recall_bench --concurrency 12 --limit 10
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common._rft_common import build_client  # noqa: E402
from RFT._common import typo_task as t  # shared primitives (no peer dep)  # noqa: E402
from RFT._common.typo_task import is_v5_trivial_correction  # noqa: E402
from RFT.typo.v9 import PROMPT_V9  # noqa: E402

API_VERSION = "2025-04-01-preview"
ENDPOINT = "https://dmxaz-test-aoai-eu2.openai.azure.com/"
DEPLOYMENT = "o4-mini"

BENCHMARK = _PROJECT_ROOT / "benchmark" / "benchmark_typo_20260512T213238Z_gpt5-family.jsonl"
SOURCES_DIR = _PROJECT_ROOT / "data" / "benchmark_sources_v6"
OUT_DIR = _PROJECT_ROOT / "outputs" / "v9_recall"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s) -> str:
    """Whitespace-normalised string key for matching."""
    return " ".join(str(s).split())


def _prompt_hash() -> str:
    h = hashlib.sha256()
    h.update(PROMPT_V9.encode("utf-8"))
    h.update(f"|{DEPLOYMENT}|".encode("utf-8"))
    return h.hexdigest()[:12]


def _load_gold():
    """Return ``{(source, column): [gold_cell, ...]}`` and the source map."""
    by_col: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for line in BENCHMARK.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "typo" not in (r.get("error_types") or []):
            continue
        if r.get("label") != "error":
            continue
        by_col[(r["source"], r["column"])].append(r)
    return by_col


def _csv_lookup() -> dict[str, Path]:
    return {p.name: p for p in SOURCES_DIR.rglob("*.csv")}


def _column_values(csv_path: Path, column: str):
    """Return ``[(value, count), ...]`` for the full column, or None."""
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return None
    if column not in df.columns:
        return None
    counts = df[column].value_counts()
    return [(str(v), int(c)) for v, c in counts.items()]


# ---------------------------------------------------------------------------
# Per-column run
# ---------------------------------------------------------------------------

def _predicted_values(raw: str | None) -> set[str]:
    if not raw:
        return set()
    s = raw.strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return set()
    out = set()
    for item in obj.get("typos", []) or []:
        v = item.get("value")
        if isinstance(v, str):
            out.add(_norm(v))
    return out


def run_column(client, response_format, source, column, gold_cells, values, cache_dir):
    cache_key = hashlib.sha256(
        f"{_prompt_hash()}|{source}|{column}".encode("utf-8")
    ).hexdigest()[:16]
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        raw = cached.get("raw")
        err = cached.get("error")
    else:
        user_msg = t.build_user_message(
            column=column, source=source, values_with_counts=values,
        )
        messages = [
            {"role": "developer", "content": PROMPT_V9},
            {"role": "user", "content": user_msg},
        ]
        raw, err = None, None
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=messages,
                response_format=response_format,
                reasoning_effort="low",
            )
            raw = resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        cache_path.write_text(json.dumps({"raw": raw, "error": err}))

    pred = _predicted_values(raw)
    results = []
    for g in gold_cells:
        ov = _norm(g["original_value"])
        hit = ov in pred
        results.append({
            "source": source,
            "column": column,
            "row": g["row"],
            "original_value": g["original_value"],
            "corrected_value": g.get("corrected_value"),
            "trivial": is_v5_trivial_correction(
                g.get("original_value"), g.get("corrected_value")
            ),
            "hit": hit,
        })
    return {
        "source": source,
        "column": column,
        "error": err,
        "n_pred": len(pred),
        "cells": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Only run the first N columns (debug).")
    ap.add_argument("--tag", default="", help="Label for this run in output names.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)

    by_col = _load_gold()
    csv_map = _csv_lookup()
    response_format = t.build_response_format()
    client = build_client(ENDPOINT, API_VERSION)

    jobs = []
    skipped = []
    for (source, column), gold_cells in sorted(by_col.items()):
        csv_path = csv_map.get(source)
        if csv_path is None:
            skipped.append((source, column, "no_csv"))
            continue
        values = _column_values(csv_path, column)
        if not values:
            skipped.append((source, column, "no_column"))
            continue
        jobs.append((source, column, gold_cells, values))

    if args.limit:
        jobs = jobs[: args.limit]

    print(f"prompt_hash={_prompt_hash()}  columns={len(jobs)}  "
          f"skipped={len(skipped)}  gold_cells="
          f"{sum(len(g) for _, _, g, _ in jobs)}")

    col_results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(run_column, client, response_format, s, c, g, v, cache_dir): (s, c)
            for (s, c, g, v) in jobs
        }
        done = 0
        for fut in as_completed(futs):
            col_results.append(fut.result())
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} columns done "
                      f"({time.time() - t0:.0f}s)")

    # Aggregate
    all_cells = [c for r in col_results for c in r["cells"]]
    errs = [r for r in col_results if r["error"]]

    def _recall(cells):
        if not cells:
            return 0.0, 0, 0
        hits = sum(1 for c in cells if c["hit"])
        return hits / len(cells), hits, len(cells)

    pool_r, pool_hit, pool_n = _recall(all_cells)
    nontriv = [c for c in all_cells if not c["trivial"]]
    nt_r, nt_hit, nt_n = _recall(nontriv)
    triv = [c for c in all_cells if c["trivial"]]
    tr_r, tr_hit, tr_n = _recall(triv)

    # per-column recall
    per_col = []
    for r in col_results:
        rc, h, n = _recall(r["cells"])
        per_col.append((r["source"], r["column"], h, n, rc, r["n_pred"], r["error"]))

    misses = [c for c in all_cells if not c["hit"]]

    print("\n" + "=" * 60)
    print(f"POOL recall (all gold):      {pool_hit}/{pool_n} = {pool_r:.4f}")
    print(f"NON-TRIVIAL recall:          {nt_hit}/{nt_n} = {nt_r:.4f}")
    print(f"TRIVIAL recall:              {tr_hit}/{tr_n} = {tr_r:.4f}")
    print(f"API errors (columns):        {len(errs)}")
    print("=" * 60)

    # Save artefacts
    tag = (args.tag + "_") if args.tag else ""
    ph = _prompt_hash()
    cells_path = OUT_DIR / f"{tag}cells_{ph}.jsonl"
    misses_path = OUT_DIR / f"{tag}misses_{ph}.jsonl"
    summary_path = OUT_DIR / f"{tag}summary_{ph}.json"

    with cells_path.open("w") as f:
        for c in all_cells:
            f.write(json.dumps(c) + "\n")
    with misses_path.open("w") as f:
        for c in sorted(misses, key=lambda x: (not x["trivial"], x["column"])):
            f.write(json.dumps(c) + "\n")
    summary = {
        "prompt_hash": ph,
        "deployment": DEPLOYMENT,
        "reasoning_effort": "low",
        "pool_recall": pool_r, "pool_hit": pool_hit, "pool_n": pool_n,
        "nontrivial_recall": nt_r, "nontrivial_hit": nt_hit, "nontrivial_n": nt_n,
        "trivial_recall": tr_r, "trivial_hit": tr_hit, "trivial_n": tr_n,
        "api_errors": len(errs),
        "n_columns": len(col_results),
        "skipped": skipped,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print worst columns (with at least one miss)
    print("\nColumns with misses (recall < 1.0):")
    for s, c, h, n, rc, npred, err in sorted(per_col, key=lambda x: x[4]):
        if rc < 1.0:
            flag = f"  ERROR:{err}" if err else ""
            print(f"  {rc:.2f}  {h}/{n}  pred={npred:<3}  {s} :: {c}{flag}")

    print(f"\nMisses dumped -> {misses_path}")
    print(f"Summary       -> {summary_path}")


if __name__ == "__main__":
    main()
