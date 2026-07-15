"""Build train/valid/eval splits for the inconsistency RFT (v1).

Whole-column rows: one RFT example per auditable column (>=2 distinct
values) of a benchmark source. The reward is gold-free (the rulebook
judge in ``_grader.py`` derives truth from the column itself), so the
splits do NOT carry gold -- but we still split BY SOURCE so no CSV leaks
across train / valid / eval, and we include CLEAN columns (no labeled
error) so the judge can teach the model to stay quiet.

Deterministic: ``random.Random(seed).shuffle`` over the sorted benchmark
sources, so re-running reproduces the SAME eval-source hold-out (used by
``eval/run_baseline.py --sources``).

Each row::

    {
      "messages": [ {developer: rulebook}, {user: column audit prompt} ],
      "column":             "<header>",                 # raw, grader item.*
      "source":             "<csv>",
      "column_values_json": "[[value,count], ...]"      # array for the judge
    }

Usage::

    python -m RFT.inconsistency.v1.scripts.build_splits
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.inconsistency.v1 import PROMPT_INCONSISTENCY, build_user_message  # noqa: E402

BENCHMARK = (
    _PROJECT_ROOT / "benchmark"
    / "benchmark_inconsistency_20260515T010112Z_gpt5-family.jsonl"
)
SOURCES_DIR = _PROJECT_ROOT / "data" / "source_inconsistency_bing21"
OUT_DIR = _PROJECT_ROOT / "RFT" / "inconsistency" / "v1" / "data"
MIN_DISTINCT_VALUES = 2
MAX_VALUES = 200


def _benchmark_sources() -> list[str]:
    srcs = set()
    for line in BENCHMARK.read_text().splitlines():
        if line.strip():
            srcs.add(json.loads(line)["source"])
    return sorted(srcs)


def _labeled_cols() -> set[tuple[str, str]]:
    out = set()
    for line in BENCHMARK.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["source"], r["column"]))
    return out


def _csv_lookup() -> dict[str, Path]:
    return {p.name: p for p in SOURCES_DIR.rglob("*.csv")}


def _auditable_columns(csv_path: Path):
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for col in df.columns:
        counts = df[col].value_counts()
        vals = [(str(v), int(c)) for v, c in counts.items() if str(v).strip() != ""]
        if len(vals) >= MIN_DISTINCT_VALUES:
            out[str(col)] = vals
    return out


def _row(source: str, column: str, values) -> dict:
    # Field layout mirrors the proven typo v5 build_record: `column` and
    # `source` are RAW strings (the grader's plain-text template inserts
    # them directly); `column_values_json` is a JSON array [[value,count],
    # ...] capped at MAX_VALUES, which the PR judge scans for recall.
    capped = [[str(v), int(c)] for v, c in values[:MAX_VALUES]]
    return {
        "messages": [
            {"role": "developer", "content": PROMPT_INCONSISTENCY},
            {"role": "user", "content": build_user_message(column, source, values)},
        ],
        "column": column,
        "source": source,
        "column_values_json": json.dumps(capped, ensure_ascii=False),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--eval-frac", type=float, default=0.30)
    ap.add_argument("--valid-frac", type=float, default=0.10)
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = _benchmark_sources()
    labeled = _labeled_cols()
    csv_map = _csv_lookup()

    rng = random.Random(args.seed)
    rng.shuffle(sources)
    n = len(sources)
    n_eval = round(n * args.eval_frac)
    n_valid = round(n * args.valid_frac)
    eval_src = set(sources[:n_eval])
    valid_src = set(sources[n_eval:n_eval + n_valid])
    train_src = set(sources[n_eval + n_valid:])

    split_of = {}
    for s in train_src:
        split_of[s] = "train"
    for s in valid_src:
        split_of[s] = "valid"
    for s in eval_src:
        split_of[s] = "eval"

    rows = {"train": [], "valid": [], "eval": []}
    n_labeled_cols = {"train": 0, "valid": 0, "eval": 0}
    for source in sorted(split_of):
        csv_path = csv_map.get(source)
        if csv_path is None:
            continue
        split = split_of[source]
        for column, values in sorted(_auditable_columns(csv_path).items()):
            rows[split].append(_row(source, column, values))
            if (source, column) in labeled:
                n_labeled_cols[split] += 1

    for split in ("train", "valid", "eval"):
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows[split]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split:5}: {len(rows[split]):4} cols "
              f"({n_labeled_cols[split]} labeled)  -> {path}")

    manifest = {
        "benchmark": BENCHMARK.name,
        "seed": args.seed,
        "eval_frac": args.eval_frac, "valid_frac": args.valid_frac,
        "n_sources": n,
        "sources": {"train": sorted(train_src), "valid": sorted(valid_src),
                    "eval": sorted(eval_src)},
        "rows": {k: len(v) for k, v in rows.items()},
        "labeled_cols": n_labeled_cols,
        "prompt_chars": len(PROMPT_INCONSISTENCY),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {args.output_dir / 'manifest.json'}")
    print(f"eval sources held out: {len(eval_src)} (never trained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
