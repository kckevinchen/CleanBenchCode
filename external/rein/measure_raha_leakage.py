"""Measure raha's ground-truth label leakage on the benchmark.

raha is FEW-SHOT: per table it samples ``LABELING_BUDGET`` tuples (rows)
and labels every cell in them with the benchmark ground truth (via ``clean.csv``),
then trains a per-table classifier and predicts the rest. Because it labels whole
rows, the fraction of *cells* whose label is leaked equals the fraction of *rows*
labeled. This script reports that leakage for a fixed budget and for proportional
budgets, over the labeled tables (where errors actually exist).

    python external/rein/measure_raha_leakage.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> None:
    man = json.loads((C.DATASETS_DIR / "_manifest.json").read_text())["datasets"]
    labeled = [e["n_rows"] for e in man.values() if not e.get("clean_universe")]
    labeled.sort()
    tot = sum(labeled)

    def fixed(b):
        rows = sum(min(b, n) for n in labeled)
        return 100 * rows / tot, 100 * st.mean(min(b, n) / n for n in labeled)

    def frac(f, mn=1):
        rows = sum(min(max(mn, math.ceil(f * n)), n) for n in labeled)
        return 100 * rows / tot, 100 * st.mean(min(max(mn, math.ceil(f * n)), n) / n for n in labeled)

    print(f"labeled tables: {len(labeled)}  (rows: min={min(labeled)} "
          f"median={int(st.median(labeled))} max={max(labeled)})")
    print(f"\n{'budget':<22}{'agg cell-leak':>14}{'mean per-table':>16}")
    agg, mean = fixed(20)
    print(f"{'fixed=20 (standard)':<22}{agg:>13.1f}%{mean:>15.1f}%")
    for f in (0.10, 0.05, 0.02):
        agg, mean = frac(f)
        print(f"{f'frac={f} (min 1 row)':<22}{agg:>13.1f}%{mean:>15.1f}%")


if __name__ == "__main__":
    main()
