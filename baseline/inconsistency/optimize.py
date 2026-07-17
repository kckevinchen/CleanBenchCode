"""CLI entry point for the inconsistency prompt-optimization loop.

    az login   # required before any live call

    # full loop, pooled pbi+bing, 6 iterations
    python -m baseline.inconsistency.optimize --iters 6

    # out-of-distribution test: optimize on bing, hold out all pbi
    python -m baseline.inconsistency.optimize --dev-dataset bing21_500

    # smoke-test the whole loop with NO API calls (stub model + stub agents)
    python -m baseline.inconsistency.optimize --dry-run --iters 2

Outputs land under ``outputs/inconsistency_promptopt/<ts>/`` (per-iter prompts,
critiques, leakage reports, dev metrics, decisions) plus ``best_prompt.md``,
``test_metrics.json`` and ``loop_summary.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.inconsistency import data, loop  # noqa: E402
from src.core.llm_client import configured_endpoint_pool  # noqa: E402

PROMPTS_DIR = ROOT / "baseline" / "prompts" / "inconsistency"
API_VERSION = "2025-04-01-preview"
OUT_ROOT = ROOT / "outputs" / "inconsistency_promptopt"


# ---------------------------------------------------------------------------
# Stub backends for --dry-run (exercise the full loop without any API call)
# ---------------------------------------------------------------------------


def _stub_detector(prompt, source, column, values):
    """Flag the most-frequent value of every column (mostly FPs) — deterministic."""
    if not values:
        return json.dumps({"findings": []})
    top = values[0][0]
    return json.dumps(
        {
            "findings": [
                {
                    "value": top,
                    "corrected_value": "",
                    "kind": "casing",
                    "rationale": "stub: flags the majority value",
                }
            ]
        }
    )


class _StubAgentBackend:
    """Reviewer stub emits one safe + one leaky suggestion; editor stub appends a line."""

    def __call__(self, instructions: str, user_input: str) -> str:
        if "reviewer" in instructions.splitlines()[0].lower():
            leaked = "a specific value"
            m = re.search(r"value=('([^']*)'|\"([^\"]*)\")", user_input)
            if m:
                leaked = m.group(2) or m.group(3) or leaked
            return json.dumps(
                {
                    "dominant_problem": "precision",
                    "summary": "Stub critique for dry-run.",
                    "suggestions": [
                        {
                            "failure_mode": "over-flagging the majority value",
                            "hypothesized_cause": "no guard against flagging the dominant form",
                            "rule_change": "Never flag the column's dominant/most-frequent form.",
                            "kind": "scope",
                            "priority": "high",
                        },
                        {
                            "failure_mode": "leaky example",
                            "hypothesized_cause": "n/a",
                            "rule_change": f"Always rewrite {leaked} specifically.",
                            "kind": "general",
                            "priority": "low",
                        },
                    ],
                }
            )
        # editor
        m = re.search(r"# Current rulebook.*?\n\n(.*)\n\n# Reviewer", user_input, re.DOTALL)
        head = m.group(1) if m else user_input
        return head + "\n\n*Stub editor note: prefer the dominant convention.*\n"


# ---------------------------------------------------------------------------
# Live backends
# ---------------------------------------------------------------------------


def _build_live(args):
    from baseline._common import build_client
    from baseline.inconsistency.agents import ResponsesAgentBackend
    from baseline.inconsistency.evaluator import Gpt55Detector

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    if not endpoints:
        endpoints = configured_endpoint_pool()
    if not endpoints:
        raise SystemExit("live optimization requires --endpoints or AZURE_OPENAI_ENDPOINTS")
    clients = [build_client(ep, API_VERSION) for ep in endpoints]
    detector = Gpt55Detector(
        clients,
        model=args.model,
        effort=args.effort,
        temperature=args.temperature,
        cache_dir=Path(args.out_dir) / "detector_cache" if args.out_dir else None,
    )
    agent_backend = ResponsesAgentBackend(
        clients[0], model=args.agent_model, effort=args.agent_effort
    )
    return detector, agent_backend


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--benchmark", default=str(ROOT / data.DEFAULT_BENCHMARK))
    ap.add_argument(
        "--data-dirs",
        default=",".join(data.DEFAULT_DATA_DIRS),
        help="Comma-separated dataset dirs to resolve source CSVs from.",
    )
    ap.add_argument(
        "--exclusions",
        default=str(ROOT / data.DEFAULT_EXCLUSIONS),
        help="ignored.jsonl of curator-ignored CSVs/columns/rows/cells (FP-immune, dropped).",
    )
    ap.add_argument(
        "--no-exclusions",
        action="store_true",
        help="Audit everything, including curator-ignored columns/files.",
    )
    ap.add_argument("--dev-frac", type=float, default=0.34)
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.16,
        help="Fraction of sources used ONLY for candidate selection "
        "(held out from the reviewer). 0 = select on dev (less robust).",
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--dev-dataset",
        default=None,
        help="Restrict dev to this dataset (e.g. bing21_500); all other "
        "datasets become an out-of-distribution test.",
    )
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument(
        "--margin",
        type=float,
        default=0.005,
        help="Accept an edit only if the selection objective improves by more than this "
        "(positive default guards against accepting noise-level gains).",
    )
    ap.add_argument(
        "--objective",
        default="f1",
        choices=list(loop.OBJECTIVES),
        help="Gold-derived quantity to hill-climb (default f1). 'fbeta' uses --fbeta.",
    )
    ap.add_argument(
        "--fbeta", type=float, default=2.0, help="Beta for --objective fbeta (>1 favours recall)."
    )
    ap.add_argument(
        "--max-prompt-chars",
        type=int,
        default=6000,
        help="Soft budget for the editable rulebook head (seed ~4200). The editor "
        "is told to stay under this; the selection objective is penalised above it.",
    )
    ap.add_argument(
        "--length-penalty",
        type=float,
        default=0.02,
        help="Objective points subtracted per 1000 chars over --max-prompt-chars "
        "(0 disables length control).",
    )
    ap.add_argument("--model", default="gpt-5.5", help="Detector model.")
    ap.add_argument("--effort", default="none", help="Detector reasoning effort.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Detector temperature.")
    ap.add_argument("--agent-model", default="gpt-5.5", help="Reviewer/editor model.")
    ap.add_argument("--agent-effort", default="medium", help="Reviewer/editor reasoning effort.")
    ap.add_argument("--endpoints", default="", help="Comma-separated Azure endpoint pool.")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--max-sources",
        type=int,
        default=0,
        help="Cap dev AND test to the first N sources (cheap live probe; 0 = no cap).",
    )
    ap.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip clean columns (precision signal). Not recommended.",
    )
    ap.add_argument("--leakage-min-len", type=int, default=4)
    ap.add_argument("--out-dir", default="")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full loop with stub model + stub agents (no API calls).",
    )
    args = ap.parse_args()

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        Path(args.out_dir) if args.out_dir else OUT_ROOT / (ts + ("_dry" if args.dry_run else ""))
    )
    args.out_dir = str(out_dir)

    # Load gold + resolve CSVs + split by source.
    cols, source_dataset = data.load_gold(args.benchmark)
    csv_map = data.csv_lookup([d.strip() for d in args.data_dirs.split(",") if d.strip()])
    exclusions = None
    if not args.no_exclusions:
        exclusions = data.load_exclusions(args.exclusions, csv_map=csv_map)
        print(f"exclusions: {exclusions.stats()}")
    split = data.split_sources(
        source_dataset,
        cols,
        dev_frac=args.dev_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        dev_dataset=args.dev_dataset,
    )
    dev_sources = set(split["dev"])
    val_sources = set(split["val"])
    test_sources = set(split["test"])
    if args.max_sources:
        dev_sources = set(sorted(dev_sources)[: args.max_sources])
        val_sources = set(sorted(val_sources)[: args.max_sources])
        test_sources = set(sorted(test_sources)[: args.max_sources])

    seed_rulebook = (PROMPTS_DIR / "rulebook.md").read_text(encoding="utf-8")
    reviewer_prompt = (PROMPTS_DIR / "reviewer.md").read_text(encoding="utf-8")
    editor_prompt = (PROMPTS_DIR / "prompt_editor.md").read_text(encoding="utf-8")
    # Tell the editor the concrete length budget (literal replace avoids touching
    # any other braces in the template).
    editor_prompt = editor_prompt.replace("{max_chars}", f"{args.max_prompt_chars:,}")

    if args.dry_run:
        detector, agent_backend = _stub_detector, _StubAgentBackend()
    else:
        detector, agent_backend = _build_live(args)

    print(f"dev sources={len(dev_sources)} {split['dev_stats']}")
    print(f"val sources={len(val_sources)} {split['val_stats']}")
    print(f"test sources={len(test_sources)} {split['test_stats']}")
    print(f"-> {out_dir}", flush=True)

    summary = loop.run_loop(
        cols=cols,
        source_dataset=source_dataset,
        csv_map=csv_map,
        dev_sources=dev_sources,
        val_sources=val_sources,
        test_sources=test_sources,
        seed_rulebook=seed_rulebook,
        reviewer_prompt=reviewer_prompt,
        editor_prompt=editor_prompt,
        detector=detector,
        agent_backend=agent_backend,
        out_dir=out_dir,
        split_meta=split,
        iters=args.iters,
        margin=args.margin,
        objective=args.objective,
        fbeta=args.fbeta,
        max_prompt_chars=args.max_prompt_chars,
        length_penalty=args.length_penalty,
        include_clean=not args.no_clean,
        concurrency=args.concurrency,
        leakage_min_len=args.leakage_min_len,
        exclusions=exclusions,
    )

    print("\n" + "=" * 60)
    print(
        f"best prompt head = {summary['best_head_chars']} chars "
        f"(budget {args.max_prompt_chars})"
    )
    g = summary["generalization"]
    print(
        f"objective={args.objective}  selection on {summary['selection_split']}  "
        f"(iters run: {summary['iters_run']})"
    )
    print(
        f"  dev  f1={g['dev_f1']:.3f}   val f1={g['val_f1']:.3f}   " f"TEST f1={g['test_f1']:.3f}"
    )
    print(
        f"  generalization gap val->test={g['val_to_test_gap']:+.3f}  "
        f"dev->test={g['dev_to_test_gap']:+.3f}"
    )
    tm = summary["test_metrics"]
    print(
        f"TEST  precision={tm['precision_incl_clean']:.3f} recall={tm['recall']:.3f} "
        f"f1={tm['f1_incl_clean']:.3f}  (TP={tm['tp']} FP={tm['fp']} FN={tm['fn']})"
    )
    for ds, dm in summary["test_by_dataset"].items():
        print(f"  {ds:14} f1={dm['f1']:.3f} (P={dm['precision']:.3f} R={dm['recall']:.3f})")
    print(f"artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
