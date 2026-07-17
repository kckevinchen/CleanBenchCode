"""Offline validation of the v11 GOLD-FREE recall judge (and precision judge).

The whole v11 bet is that an LLM recall judge can replace the deterministic
gold-recall leg without (a) letting the model collapse to empty predictions or
(b) hallucinating typos on clean columns (which would punish correct
abstention). This harness checks both BEFORE any training compute is spent.

It runs the v11 judges the same way Azure RFT will at train time -- substitute
the grader's ``{{item.*}}`` / ``{{sample.output_text}}`` placeholders, call the
grader model, parse the ``Score:`` line -- against a labelled fixture (default:
v10's ``eval.jsonl``, which carries ``gold_typos`` so we know which columns are
clean vs dirty). The gold is used ONLY to define the expected answer for the
check; the judge itself never sees it.

For every fixture column we score the RECALL judge on two synthetic outputs:

  * ``empty`` -- ``{"typos": []}`` (proposes nothing).
  * ``gold``  -- proposes exactly the column's gold typos (dirty columns only).

Expected (the whole point of the recall leg):

  * CLEAN column (no gold typos): G should be empty -> ``empty`` scores HIGH
    (recall vacuous = 1.0). A LOW score here is a judge HALLUCINATION (it
    invented a typo to "miss") -- the precision-erosion risk.
  * DIRTY column (>=1 gold typo): ``empty`` is WRONG -> should score LOW
    (missed every typo -> recall ~ 0); ``gold`` should score HIGH.

The real Azure grader runs on gpt-4o-2024-08-06 / o3-mini, which are not in the
local LLMClient pool; per the repo convention (see v10's judge_pr_check) we call
the GPT-5 family as a faithful offline stand-in -- the developer prompt is
identical, so this validates the judge's BEHAVIOUR even though the exact scoring
model differs. Re-run with ``--grader-model o3-mini-2025-01-31`` baked in if you
intend to train with o3-mini.

Usage::

    python -m RFT.typo.v11.eval.judge_check
    python -m RFT.typo.v11.eval.judge_check --judge precision --limit 12
    python -m RFT.typo.v11.eval.judge_check --eval RFT/typo/v10/data/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common import typo_task as _v5  # noqa: E402
from RFT.typo.v11._grader import (  # noqa: E402
    build_precision_judge_v11,
    build_recall_judge_v11,
)
from src.core.llm_client import LLMClient  # noqa: E402

# Default fixture: v10's eval split -- labelled, so we know clean vs dirty.
DEFAULT_EVAL = _PROJECT_ROOT / "RFT" / "typo" / "v10" / "data" / "eval.jsonl"
DEFAULT_CALL_MODEL = "gpt-5.2,gpt-5.4,gpt-5.5"

_PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")
_SCORE_LINE_RE = re.compile(
    r"^\s*score\s*[:=]\s*([-+]?\d*\.?\d+)", re.IGNORECASE | re.MULTILINE
)

_JUDGE_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "recall": build_recall_judge_v11,
    "precision": build_precision_judge_v11,
}


def _render(template: str, mapping: dict[str, str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in mapping:
            raise KeyError(f"unresolved placeholder {{{{ {key} }}}}")
        return mapping[key]

    return _PLACEHOLDER_RE.sub(_sub, template)


def _parse_score(text: str) -> float | None:
    m = _SCORE_LINE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _dump(typos: list[dict[str, Any]]) -> str:
    return json.dumps({"typos": typos}, ensure_ascii=False)


def _gold_output(gold_typos: list[dict[str, Any]]) -> str:
    typos = [
        {
            "value": g["value"],
            "correction": g.get("correction", g["value"]),
            "flavour": g.get("flavour", "spelling"),
            "rationale": "benchmark gold",
        }
        for g in gold_typos
    ]
    return _dump(typos)


def _score_one(client, model, dev, usr_template, item, output_text, range_):
    mapping = {
        "item.column": item["column"],
        "item.source": item["source"],
        "item.column_values_json": item["column_values_json"],
        "sample.output_text": output_text,
    }
    user = _render(usr_template, mapping)
    resp = client.query(
        model=model, prompt=user, system_prompt=dev, reasoning_effort="low"
    )
    if not resp.ok:
        return None, f"<query failed: {resp.error}>"
    score = _parse_score(resp.text)
    if score is None:
        return None, resp.text
    lo, hi = range_
    return float(max(lo, min(hi, score))), resp.text


def run_column(client, model, dev, usr_template, range_, item):
    gold = json.loads(item.get("gold_typos") or "[]")
    is_clean = len(gold) == 0
    out: dict[str, Any] = {
        "column": item["column"],
        "source": item["source"],
        "is_clean": is_clean,
        "n_gold": len(gold),
    }
    empty_score, empty_raw = _score_one(
        client, model, dev, usr_template, item, _dump([]), range_
    )
    out["empty_score"] = empty_score
    out["empty_reward"] = None if empty_score is None else 5.0 * empty_score
    # Keep the judge's reasoning when empty is judged WRONG: on a CLEAN column
    # that is a hallucination (it invented a missed typo); on a DIRTY column it
    # is the expected recall collapse.
    if empty_score is None or empty_score < 0.5:
        out["empty_raw"] = empty_raw[:600]
    if not is_clean:
        gold_score, gold_raw = _score_one(
            client, model, dev, usr_template, item, _gold_output(gold), range_
        )
        out["gold_score"] = gold_score
        out["gold_reward"] = None if gold_score is None else 5.0 * gold_score
        if gold_score is None:
            out["gold_raw"] = gold_raw[:300]
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default="recall", choices=sorted(_JUDGE_BUILDERS),
                    help="Which v11 judge to validate (default: recall, the new "
                         "gold-free leg).")
    ap.add_argument("--grader-model", default=_v5.TYPO_V5_DEFAULT_GRADER_MODEL,
                    choices=list(_v5.TYPO_V5_GRADER_MODEL_WHITELIST),
                    help="Grader model baked into the judge spec (the real Azure "
                         "grader). Does NOT change which model is called offline "
                         "-- see --model.")
    ap.add_argument("--model", default=DEFAULT_CALL_MODEL,
                    help="Model (or comma-separated rotation) actually called as "
                         "the offline stand-in grader.")
    ap.add_argument("--eval", default=str(DEFAULT_EVAL), type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--degeneracy-threshold", type=float, default=2.5,
                    help="Flag a DIRTY column if its empty reward (0-5) is >= "
                         "this. Low is good -- empty should collapse on dirty.")
    ap.add_argument("--clean-pass-threshold", type=float, default=2.5,
                    help="A CLEAN column PASSES if its empty reward (0-5) is >= "
                         "this. High is good -- empty is correct on clean.")
    ap.add_argument("--out", default="outputs/v11_judge_check.json")
    args = ap.parse_args()

    judge = _JUDGE_BUILDERS[args.judge](grader_model=args.grader_model)
    msgs = judge["input"]
    dev = next(m["content"] for m in msgs if m["role"] == "developer")
    usr_template = next(m["content"] for m in msgs if m["role"] == "user")
    range_ = judge.get("range", [0.0, 1.0])

    rows = [
        json.loads(line)
        for line in args.eval.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        rows = rows[: args.limit]

    client = LLMClient()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(run_column, client, args.model, dev, usr_template,
                      range_, item)
            for item in rows
        ]
        for fut in as_completed(futs):
            results.append(fut.result())

    clean = [r for r in results if r["is_clean"]]
    dirty = [r for r in results if not r["is_clean"]]

    clean_pass = sum(
        1 for r in clean
        if r["empty_reward"] is not None
        and r["empty_reward"] >= args.clean_pass_threshold
    )
    degenerate = [
        r for r in dirty
        if r["empty_reward"] is not None
        and r["empty_reward"] >= args.degeneracy_threshold
    ]
    hallucinated = [
        r for r in clean
        if r["empty_reward"] is not None
        and r["empty_reward"] < args.clean_pass_threshold
    ]

    print(f"\n========== v11 {args.judge.upper()}-judge check ==========")
    print(f"call model: {args.model}   fixture: {args.eval.name}   columns: "
          f"{len(results)} ({len(clean)} clean / {len(dirty)} dirty)\n")

    print("--- CLEAN columns (empty is CORRECT -> want HIGH reward) ---")
    for r in sorted(clean, key=lambda x: (x["empty_reward"] is None,
                                          x["empty_reward"] or 0)):
        er = r["empty_reward"]
        tag = "  " if (er is not None and er >= args.clean_pass_threshold) else "!!"
        print(f"  {tag} empty={_fmt(er)}  {r['source']} :: {r['column']}")
    print(f"  clean pass (empty reward >= {args.clean_pass_threshold}): "
          f"{clean_pass}/{len(clean)}   hallucinated (judge invented a miss): "
          f"{len(hallucinated)}   mean empty reward="
          f"{_mean([r['empty_reward'] for r in clean]):.2f}\n")

    print("--- DIRTY columns (empty is WRONG -> want LOW empty, HIGH gold) ---")
    for r in sorted(dirty, key=lambda x: -(x["empty_reward"] or 0)):
        er, gr = r["empty_reward"], r.get("gold_reward")
        tag = "!!" if (er is not None and er >= args.degeneracy_threshold) else "  "
        print(f"  {tag} empty={_fmt(er)} gold={_fmt(gr)} (n_gold={r['n_gold']})"
              f"  {r['source']} :: {r['column']}")
    print(f"  degenerate (empty reward >= {args.degeneracy_threshold}): "
          f"{len(degenerate)}/{len(dirty)}   mean empty reward="
          f"{_mean([r['empty_reward'] for r in dirty]):.2f}  mean gold reward="
          f"{_mean([r.get('gold_reward') for r in dirty]):.2f}\n")

    verdict = "PASS" if not degenerate and clean_pass == len(clean) else "REVIEW"
    print(f"VERDICT: {verdict}  (want: empty collapses on dirty, stays high on "
          f"clean, no hallucinations)")

    out_path = _PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "judge": args.judge,
        "grader_model": args.grader_model,
        "call_model": args.model,
        "fixture": str(args.eval),
        "summary": {
            "n_clean": len(clean), "n_dirty": len(dirty),
            "clean_pass": clean_pass,
            "n_hallucinated": len(hallucinated),
            "n_degenerate": len(degenerate),
            "mean_clean_empty_reward": _mean([r["empty_reward"] for r in clean]),
            "mean_dirty_empty_reward": _mean([r["empty_reward"] for r in dirty]),
            "mean_dirty_gold_reward": _mean([r.get("gold_reward") for r in dirty]),
            "verdict": verdict,
        },
        "results": results,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
