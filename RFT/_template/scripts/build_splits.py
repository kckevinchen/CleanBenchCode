"""DATA -- EDIT THIS for a new run (build the train/valid/eval splits).

Writes ``data/{train,valid,eval}.jsonl`` next to the spec. Each row is one
column-audit example with the trainee messages BAKED IN, plus the raw fields
the grader reads:

    {
      "messages": [
        {"role": "developer", "content": <TRAINEE_PROMPT>},
        {"role": "user",      "content": <build_user_message(...)>}
      ],
      "column": "<header>",
      "source": "<table id>",
      "column_values_json": [["value", count], ...]   # grader evidence
      # optional gold for `rft run` micro-F1: "gold_values": [...], "ambiguous_values": [...]
    }

Split BY SOURCE (all of a table's columns land in one split) so the eval split
shares no table with train -- zero leakage (see [[reference-rft-eval]]).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from RFT._template._prompt import TRAINEE_PROMPT, build_user_message

_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = _DIR / "data"


def build_row(source: str, column: str, values_with_counts: list[tuple[str, int]]) -> dict:
    return {
        "messages": [
            {"role": "developer", "content": TRAINEE_PROMPT},
            {"role": "user", "content": build_user_message(column, source, values_with_counts)},
        ],
        "column": column,
        "source": source,
        "column_values_json": [[v, c] for v, c in values_with_counts],
        # TODO: if you have gold labels, add them here and set gold_field= in spec.py.
    }


def iter_rows(sources_dir: Path):
    """Yield one row per (csv, column). EDIT: filter to the columns you want."""
    for csv_path in sorted(sources_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception:  # noqa: BLE001
            continue
        for column in df.columns:
            counts = df[column].value_counts()
            vals = [(str(v), int(c)) for v, c in counts.items()]
            if not vals:
                continue
            yield build_row(csv_path.name, column, vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, required=True, help="Dir of source CSVs.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    args = ap.parse_args()

    rows = list(iter_rows(args.sources))
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    sources = sorted(by_source)
    random.Random(args.seed).shuffle(sources)
    n = len(sources)
    n_valid = int(n * args.valid_frac)
    n_eval = int(n * args.eval_frac)
    splits = {
        "eval": sources[:n_eval],
        "valid": sources[n_eval : n_eval + n_valid],
        "train": sources[n_eval + n_valid :],
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, srcs in splits.items():
        out = DATA_DIR / f"{split}.jsonl"
        with out.open("w") as fh:
            for s in srcs:
                for r in by_source[s]:
                    fh.write(json.dumps(r) + "\n")
        print(f"{split}: {len(srcs)} sources -> {out}")


if __name__ == "__main__":
    main()
