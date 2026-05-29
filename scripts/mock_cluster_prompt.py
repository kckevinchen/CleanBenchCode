"""One-shot mock test for the column-clustering proposer prompt.

Loads a CSV, enumerates candidate columns the same way
``experiments.run_column_clustering`` would, and for each candidate
runs the seed proposer prompt ONCE per column (no adaptive loop, no
prompt-ensemble). Prints the value list the LLM sees and the clusters
it proposed. Intended as a fast inspection harness when iterating on
the prompt itself; not part of the production pipeline.

Usage::

    venv/bin/python3 -m scripts.mock_cluster_prompt \\
        --csv data/bing21_grouped/<workbook>/<file>.csv \\
        [--column <header>]  [--model gpt-5.2-mini]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import load_csv
from src.core.llm_client import LLMClient
from src.core.value_cluster.proposer import ValueClusterProposer
from experiments.run_column_clustering import _is_candidate, _value_counts


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--column", default=None,
                        help="Restrict to a single column.")
    parser.add_argument("--model", default="gpt-5.2-mini")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--reasoning-summary", default="detailed")
    parser.add_argument("--min-distinct", type=int, default=2)
    parser.add_argument("--max-distinct-ratio", type=float, default=0.95)
    args = parser.parse_args(argv)

    df = load_csv(args.csv)
    print(f"CSV: {args.csv}")
    print(f"shape: rows={len(df)}, cols={len(df.columns)}")

    llm = LLMClient()
    proposer = ValueClusterProposer(
        llm=llm,
        model=args.model,
        prompt_path=PROJECT_ROOT / "prompts" / "cluster_column_values.yaml",
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary,
    )

    n_rows = int(len(df))
    for col in df.columns:
        if args.column is not None and col != args.column:
            continue
        counts = _value_counts(df[col])
        if not _is_candidate(
            n_rows, len(counts),
            min_distinct=args.min_distinct,
            max_distinct_ratio=args.max_distinct_ratio,
        ):
            continue
        print()
        print("=" * 80)
        print(f"Column: {col!r}  ({len(counts)} distinct, {n_rows} rows)")
        print("-- value list the LLM sees: --")
        for v, c in counts:
            print(f'  - "{v}" (count={c})')
        clusters = proposer.propose_once(
            source=args.csv.name,
            column=col,
            value_counts=counts,
        )
        print("-- proposed clusters: --")
        if not clusters:
            print("  (none)")
        for i, cl in enumerate(clusters):
            print(
                f"  [{i}] kind={cl.kind.value}  "
                f"canonical={cl.canonical!r}  "
                f"values={list(cl.values)}"
            )
            if cl.target_style:
                print(f"       target_style={cl.target_style!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
