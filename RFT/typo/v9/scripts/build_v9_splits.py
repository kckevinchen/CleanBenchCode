"""Build v9 train/valid/eval/demo splits.

v9 does two things to the inherited v8 rows:

1. **Retargets the trainee prompt** -- every row's ``messages[0].content``
   is replaced with ``RFT.typo.v9.PROMPT_V9`` (same as the v8 builder).
2. **Reconciles the gold set** -- gold typo cells that are proper-noun-token
   character-level slips are moved from ``gold_typos`` into
   ``ambiguous_values`` (FP-immune, no recall credit), so the recall reward
   leg stops fighting the v9 precision grader's "names out of scope" rule.
   See ``RFT/typo/v9/_gold_reconcile.py`` for the (deterministic, offline)
   classifier and the rationale.

   **As of 2026-05-28 reconciliation (step 2) is OFF by default.** Decision:
   the hand-maintained ``_COMMON_TOKENS`` word list mis-classifies real common
   words (``Forfiet->Forfeit``, ``Cttage->Cottage``) in name columns as
   proper-noun slips. The v9 LLM precision grader now judges common-word vs
   proper-name scope at grading time (verifiability rule, ``_grader_variants``),
   so gold is passed through from the human benchmark unchanged. Pass
   ``--reconcile`` (or ``--apply-report``) to re-enable the deterministic pass.

The classifier is heuristic and meant to be reviewed: this script writes
``gold_reconciliation_report.json`` listing every gold cell, its decision,
and a one-line reason. Eyeball it before training. If you disagree with any
call, edit the report's ``reconcile``/``keep`` lists and re-run with
``--apply-report <file>`` to apply your curated decisions verbatim.

Usage::

    # 1. dry run -- classify only, write the report, touch no splits
    python -m RFT.typo.v9.scripts.build_v9_splits --report-only

    # 2. generate splits using the heuristic decisions
    python -m RFT.typo.v9.scripts.build_v9_splits

    # 3. generate splits from a human-curated report
    python -m RFT.typo.v9.scripts.build_v9_splits --apply-report \\
        RFT/typo/v9/data/gold_reconciliation_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v9 import PROMPT_V9  # noqa: E402
from RFT.typo.v9._gold_reconcile import classify_cell  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_v9_splits")

SPLITS = ("train.jsonl", "valid.jsonl", "eval.jsonl", "demo.jsonl")


def _cell_key(source: str, column: str, value: str) -> str:
    return f"{source}␟{column}␟{value}"


def _retarget_prompt(rec: dict, new_prompt: str) -> dict:
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("row missing non-empty 'messages'")
    if msgs[0].get("role") != "developer":
        raise ValueError(
            f"messages[0] expected role=developer, got {msgs[0].get('role')!r}"
        )
    msgs = list(msgs)
    msgs[0] = {**msgs[0], "content": new_prompt}
    return {**rec, "messages": msgs}


def _loads(field: str | list | None, default):
    if isinstance(field, list):
        return field
    if not isinstance(field, str) or not field:
        return default
    try:
        return json.loads(field)
    except json.JSONDecodeError:
        return default


def _decide(rec: dict, include_review: bool, override: dict[str, str] | None):
    """Return (reconcile_values:set, decisions:list) for one record."""
    column = rec.get("column", "")
    source = rec.get("source", "")
    gold = _loads(rec.get("gold_typos"), [])
    reconcile_values: set[str] = set()
    decisions = []
    for g in gold:
        if not isinstance(g, dict):
            continue
        value = g.get("value", "")
        correction = g.get("correction", "")
        key = _cell_key(source, column, value)
        if override is not None:
            decision = override.get(key, "keep")
            reason = "from --apply-report"
        else:
            decision, reason = classify_cell(column, value, correction)
        decisions.append({
            "source": source, "column": column,
            "value": value, "correction": correction,
            "decision": decision, "reason": reason,
        })
        if decision == "reconcile" or (include_review and decision == "review"):
            reconcile_values.add(value)
    return reconcile_values, decisions


def _apply(rec: dict, reconcile_values: set[str]) -> dict:
    """Move reconciled gold cells into ambiguous_values."""
    if not reconcile_values:
        return rec
    gold = _loads(rec.get("gold_typos"), [])
    amb = _loads(rec.get("ambiguous_values"), [])
    new_gold = [
        g for g in gold
        if not (isinstance(g, dict) and g.get("value") in reconcile_values)
    ]
    amb_set = list(dict.fromkeys([*amb, *sorted(reconcile_values)]))
    return {
        **rec,
        "gold_typos": json.dumps(new_gold, ensure_ascii=False),
        "ambiguous_values": json.dumps(amb_set, ensure_ascii=False),
    }


def _load_override(report_path: Path) -> dict[str, str]:
    """Build a {cell_key: decision} map from a (possibly edited) report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    override: dict[str, str] = {}
    for d in report.get("decisions", []):
        override[_cell_key(d["source"], d["column"], d["value"])] = d["decision"]
    return override


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-data-dir", type=Path,
                   default=_PROJECT_ROOT / "RFT" / "typo" / "v8" / "data")
    p.add_argument("--output-dir", type=Path,
                   default=_PROJECT_ROOT / "RFT" / "typo" / "v9" / "data")
    p.add_argument("--report-only", action="store_true",
                   help="Classify and write the report; do not write splits.")
    p.add_argument("--include-review", action="store_true",
                   help="Also reconcile cells classified 'review' (uncertain).")
    p.add_argument("--apply-report", type=Path, default=None,
                   help="Apply decisions from a curated report verbatim "
                        "(implies --reconcile).")
    p.add_argument("--reconcile", action="store_true",
                   help="Run the deterministic gold reconciliation "
                        "(classify_cell). OFF by default as of 2026-05-28: gold "
                        "passes through from source unchanged and the LLM "
                        "precision grader decides common-word vs proper-name "
                        "scope at grading time.")
    args = p.parse_args(argv)

    do_reconcile = args.reconcile or (args.apply_report is not None)
    override = _load_override(args.apply_report) if args.apply_report else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_decisions = []
    counts = {"keep": 0, "reconcile": 0, "review": 0}
    per_split_written = {}

    for fname in SPLITS:
        src = args.source_data_dir / fname
        if not src.exists():
            logger.warning("source %s missing -- skipped", fname)
            continue
        out_rows = []
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if do_reconcile:
                reconcile_values, decisions = _decide(
                    rec, args.include_review, override,
                )
            else:
                reconcile_values, decisions = set(), []
            for d in decisions:
                d["split"] = fname
                counts[d["decision"]] += 1
            all_decisions.extend(decisions)
            rec = _retarget_prompt(rec, PROMPT_V9)
            rec = _apply(rec, reconcile_values)
            out_rows.append(rec)

        if not args.report_only:
            dst = args.output_dir / fname
            with dst.open("w", encoding="utf-8") as fh:
                for r in out_rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            per_split_written[fname] = len(out_rows)
            logger.info("wrote %d rows -> %s", len(out_rows), dst)

    report = {
        "source_data_dir": str(args.source_data_dir),
        "prompt_module": "RFT.typo.v9._prompt.PROMPT_V9",
        "classifier": (
            "RFT.typo.v9._gold_reconcile.classify_cell"
            if override is None else f"--apply-report {args.apply_report}"
        ),
        "include_review": args.include_review,
        "counts": counts,
        "decisions": all_decisions,
    }
    report_path = args.output_dir / "gold_reconciliation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("report -> %s", report_path)
    logger.info(
        "decisions: keep=%d reconcile=%d review=%d",
        counts["keep"], counts["reconcile"], counts["review"],
    )

    if not args.report_only:
        manifest = {
            "source_data_dir": str(args.source_data_dir),
            "prompt_module": "RFT.typo.v9._prompt.PROMPT_V9",
            "prompt_chars": len(PROMPT_V9),
            "reconcile_counts": counts,
            "rows_written": per_split_written,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
