"""Re-run the violation-compilation step against an existing relation
discovery run, after gate-logic changes. Loads source frames, calls
_run_violation_compilation_pass, and overwrites
relation_violations.jsonl / clusters.* / issues.* in place.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import load_csv  # noqa: E402
from experiments.run_relation_discovery import (  # noqa: E402
    _run_violation_compilation_pass,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--csv-dir", required=True)
    p.add_argument("--min-pass-rate", type=float, default=0.5)
    p.add_argument("--max-pass-rate", type=float, default=1.0 - 1e-9)
    p.add_argument("--timeout", type=float, default=8.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = Path(args.run_dir).resolve()
    csv_dir = Path(args.csv_dir).resolve()

    sources_path = run_dir / "sources.json"
    with sources_path.open() as fh:
        sources = json.load(fh)
    source_names = {s["source"] for s in sources}

    source_frames: Dict[str, pd.DataFrame] = {}
    for csv_path in csv_dir.rglob("*.csv"):
        if csv_path.name not in source_names:
            continue
        try:
            source_frames[csv_path.name] = load_csv(csv_path)
        except Exception as exc:
            logging.warning("skip %s: %s", csv_path, exc)
    logging.info("loaded %d/%d source frames", len(source_frames), len(source_names))

    stats = _run_violation_compilation_pass(
        run_dir=run_dir,
        source_frames=source_frames,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
        timeout_seconds=args.timeout,
    )

    # Merge the fresh violation stats into ``summary.json`` so the
    # dashboard's counters (and any downstream consumer that reads
    # ``summary["violations"]``) pick up the recompile result. The
    # original discovery run writes summary.json at the end of its
    # pipeline with ``"violations": violations_stats``; recompile
    # only rebuilds that sub-block and leaves everything else alone.
    summary_out = run_dir / "summary.json"
    if summary_out.is_file():
        try:
            summary_doc = json.loads(summary_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning(
                "summary.json is not valid JSON; skipping merge", )
            summary_doc = None
        if isinstance(summary_doc, dict) and stats is not None:
            summary_doc["violations"] = stats
            summary_out.write_text(
                json.dumps(summary_doc, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logging.info("updated %s with fresh violation stats", summary_out)

    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
