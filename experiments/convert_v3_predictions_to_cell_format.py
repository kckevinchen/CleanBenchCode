"""Convert v3 per-column predictions to v1/v2 per-cell format.

For each column in the v3 predictions.jsonl, emit one row per distinct value
that appears in `gold_typos` UNION `parsed_typos`. Cells where both gold and
the v3 model labelled the value as clean are omitted (they would correspond
to every other distinct value in the column).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _index_by_value(items):
    out = {}
    for it in items or []:
        v = it.get("value")
        if v is None:
            continue
        out[v] = it
    return out


def convert(in_path: Path, out_path: Path) -> dict:
    n_in = 0
    n_out = 0
    counts = {"tp": 0, "fp": 0, "fn": 0}
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            col = json.loads(line)
            n_in += 1
            gold_map = _index_by_value(col.get("gold_typos"))
            pred_map = _index_by_value(col.get("parsed_typos"))
            values = sorted(set(gold_map) | set(pred_map))
            for v in values:
                g = gold_map.get(v)
                p = pred_map.get(v)
                gold_label = "typo" if g is not None else "clean"
                parsed_label = "typo" if p is not None else "clean"
                if gold_label == "typo" and parsed_label == "typo":
                    counts["tp"] += 1
                elif gold_label == "clean" and parsed_label == "typo":
                    counts["fp"] += 1
                elif gold_label == "typo" and parsed_label == "clean":
                    counts["fn"] += 1
                rec = {
                    "row_index": col.get("row_index"),
                    "source": col.get("source"),
                    "column": col.get("column"),
                    "original_value": v,
                    "gold_label": gold_label,
                    "gold_correction": (g or {}).get("correction"),
                    "gold_flavour": (g or {}).get("flavour"),
                    "parsed_label": parsed_label,
                    "parsed_correction": (p or {}).get("correction"),
                    "parsed_flavour": (p or {}).get("flavour"),
                    "parsed_rationale": (p or {}).get("rationale"),
                    "error": col.get("error"),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    return {"n_columns_in": n_in, "n_cells_out": n_out, **counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-path",
        default="outputs/typo_rft/eval_v3_ft_deployed_clean/predictions.jsonl",
    )
    ap.add_argument(
        "--out-path",
        default="outputs/typo_rft/eval_v3_ft_deployed_clean/predictions_cells.jsonl",
    )
    args = ap.parse_args()
    stats = convert(Path(args.in_path), Path(args.out_path))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
