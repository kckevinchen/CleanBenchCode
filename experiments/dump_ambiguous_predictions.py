"""Dump predictions that landed on ambiguous-labeled gold cells.

Usage:
    python -m experiments.dump_ambiguous_predictions <run_dir> <output.jsonl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK = Path("benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl")


def main(argv):
    if len(argv) != 2:
        print("usage: dump_ambiguous_predictions.py <run_dir> <output.jsonl>")
        sys.exit(2)
    run_dir = Path(argv[0])
    out_path = Path(argv[1])

    # gold: (source, row, column) -> record, restricted to typo + ambiguous
    gold = {}
    for line in BENCHMARK.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "typo" not in (r.get("error_types") or []):
            continue
        if r.get("label") != "ambiguous":
            continue
        gold[(r["source"], int(r["row"]), r["column"])] = r

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fout:
        for line in (run_dir / "issues.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            pred = json.loads(line)
            if pred.get("error_type") != "typo":
                continue
            src = pred.get("source")
            cols = pred.get("columns") or []
            rows = pred.get("affected_rows") or ([pred.get("row")] if pred.get("row") is not None else [])
            for col in cols:
                for row in rows:
                    if row is None:
                        continue
                    key = (src, int(row), col)
                    g = gold.get(key)
                    if g is None:
                        continue
                    rec = {
                        "source": src,
                        "row": int(row),
                        "column": col,
                        "observed": pred.get("observed"),
                        "predicted_fix": pred.get("suggested_fix"),
                        "predicted_explanation": pred.get("explanation"),
                        "gold_label": g.get("label"),
                        "gold_original_value": g.get("original_value"),
                        "gold_corrected_value": g.get("corrected_value"),
                        "gold_llm_suggested_value": g.get("llm_suggested_value"),
                        "issue_id": pred.get("id"),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
    print(f"Wrote {n} ambiguous-cell prediction(s) -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
