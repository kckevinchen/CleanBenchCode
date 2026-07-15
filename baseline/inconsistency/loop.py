"""The orchestrator — the deterministic "main agent".

Greedy hill-climb on dev F1:

    eval(seed) -> best
    repeat:
        critique = reviewer(best, dev evidence)        # sees data
        critique = leakage_filter(critique)            # mechanical barrier
        head'    = editor(best.head, critique)         # never sees data
        cand     = head' + frozen Output section       # schema preserved
        if eval(cand).f1 > best.f1 + margin: best = cand
    eval(best) on the held-out TEST split once, broken down by dataset.

All model access is injected (``detector`` + ``agent_backend``) so the same code
path runs live, under ``--dry-run``, and in tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baseline.inconsistency import agents
from baseline.inconsistency._task import (
    has_output_contract,
    join_rulebook,
    split_rulebook,
)
from baseline.inconsistency.evaluator import Detector, evaluate
from baseline.inconsistency.leakage import build_blocklist, scrub_critique, scrub_text

OBJECTIVES = (
    "f1",
    "f1_labeled",
    "precision",
    "precision_labeled",
    "recall",
    "fbeta",
)


def _objective(metrics: dict[str, Any], name: str = "f1", fbeta: float = 1.0) -> float:
    """The quantity we hill-climb, picked from the gold-derived metrics.

    ``f1`` / ``precision`` use the honest *incl-clean* figures (clean-column FPs
    counted); ``*_labeled`` use labeled columns only; ``fbeta`` combines
    incl-clean precision with recall (``beta``>1 favours recall, the rulebook's
    natural lean).
    """
    if name in ("f1", "f1_incl_clean"):
        return float(metrics["f1_incl_clean"])
    if name == "f1_labeled":
        return float(metrics["f1_labeled"])
    if name in ("precision", "precision_incl_clean"):
        return float(metrics["precision_incl_clean"])
    if name == "precision_labeled":
        return float(metrics["precision_labeled"])
    if name == "recall":
        return float(metrics["recall"])
    if name == "fbeta":
        p, r, b2 = metrics["precision_incl_clean"], metrics["recall"], fbeta * fbeta
        return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0
    raise ValueError(f"unknown objective: {name!r} (choose from {OBJECTIVES})")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def run_loop(
    *,
    cols: dict[tuple[str, str], dict[str, Any]],
    source_dataset: dict[str, str],
    csv_map: dict[str, Path],
    dev_sources: set[str],
    test_sources: set[str],
    val_sources: set[str] | None = None,
    seed_rulebook: str,
    reviewer_prompt: str,
    editor_prompt: str,
    detector: Detector,
    agent_backend: agents.AgentBackend,
    out_dir: Path,
    split_meta: dict[str, Any] | None = None,
    iters: int = 6,
    margin: float = 0.0,
    objective: str = "f1",
    fbeta: float = 1.0,
    max_prompt_chars: int = 6000,
    length_penalty: float = 0.02,
    include_clean: bool = True,
    concurrency: int = 8,
    leakage_min_len: int = 4,
    exclusions: Any = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_meta:
        _write_json(out_dir / "split.json", split_meta)

    head, tail = split_rulebook(seed_rulebook)
    if not has_output_contract(seed_rulebook):
        raise ValueError("seed rulebook is missing its Output-section contract")

    # Gold + column headers restricted to dev, for the leakage blocklist.
    # Honour exclusions so we don't audit (or block headers from) ignored columns.
    if exclusions:
        cols = {(s, c): g for (s, c), g in cols.items() if not exclusions.drops_column(s, c)}
    dev_cols = {k: v for k, v in cols.items() if k[0] in dev_sources}
    dev_columns = {c for (s, c) in cols if s in dev_sources}

    # Selection is on val (the reviewer never sees val's errors); if no val split
    # is provided, fall back to selecting on dev.
    val_sources = set(val_sources or set())
    sel_sources = val_sources if val_sources else dev_sources
    sel_name = "val" if val_sources else "dev"

    def _len_penalty(head_text: str) -> float:
        """Penalty (in objective units) for an editable head over the char budget."""
        over = max(0, len(head_text) - max_prompt_chars)
        return length_penalty * over / 1000.0

    def score(metrics: dict[str, Any], head_text: str) -> tuple[float, float, float]:
        """Return (effective, raw_objective, penalty). Selection uses effective."""
        raw = _objective(metrics, objective, fbeta)
        pen = _len_penalty(head_text)
        return raw - pen, raw, pen

    def _eval(prompt: str, sources: set[str]) -> dict[str, Any]:
        return evaluate(
            prompt,
            sources,
            cols,
            source_dataset,
            csv_map,
            detector,
            include_clean=include_clean,
            concurrency=concurrency,
            exclusions=exclusions,
        )

    # ---- iteration 0: the seed ----
    best_full = seed_rulebook
    best_head = head
    best_dev = _eval(best_full, dev_sources)  # reviewer evidence
    best_sel = _eval(best_full, sel_sources) if val_sources else best_dev  # selection
    best_sel_obj, best_raw, best_pen = score(best_sel["metrics"], head)

    iter0 = out_dir / "iter_00"
    iter0.mkdir(exist_ok=True)
    (iter0 / "prompt.md").write_text(best_full)
    _write_json(iter0 / "dev_metrics.json", best_dev["metrics"])
    _write_json(iter0 / "val_metrics.json", best_sel["metrics"])

    history = [
        {
            "iter": 0,
            "accepted": True,
            f"{sel_name}_objective": best_sel_obj,
            "raw_objective": best_raw,
            "length_penalty": best_pen,
            "head_chars": len(head),
            "source": "seed",
        }
    ]

    for i in range(1, iters + 1):
        idir = out_dir / f"iter_{i:02d}"
        idir.mkdir(exist_ok=True)

        critique = agents.review(
            agent_backend,
            reviewer_prompt,
            best_head,
            best_dev["metrics"],
            best_dev["evidence"],
        )
        _write_json(idir / "critique_raw.json", critique)

        blocklist = build_blocklist(
            dev_cols=dev_cols,
            evidence=best_dev["evidence"]["fp"] + best_dev["evidence"]["fn"],
            column_headers=dev_columns,
            min_len=leakage_min_len,
        )
        filtered, leak_report = scrub_critique(critique, blocklist)
        _write_json(idir / "critique_filtered.json", filtered)
        _write_json(idir / "leakage_report.json", leak_report)

        if not filtered.get("suggestions"):
            _write_json(
                idir / "decision.json", {"accepted": False, "reason": "no_suggestions_after_filter"}
            )
            history.append({"iter": i, "accepted": False, "reason": "no_suggestions"})
            break

        new_head = agents.edit_prompt(agent_backend, editor_prompt, best_head, filtered)
        # Output-side barrier: scrub any dev value the editor may have emitted as
        # an example (even one independently hallucinated that coincides with a
        # dev value/header). Guarantees no dev value ever sits in the prompt.
        new_head, out_redactions = scrub_text(new_head, blocklist)
        leak_report["editor_output_redactions"] = out_redactions
        _write_json(idir / "leakage_report.json", leak_report)
        candidate = join_rulebook(new_head, tail)
        (idir / "prompt.md").write_text(candidate)

        if len(new_head.strip()) < 200 or not has_output_contract(candidate):
            _write_json(
                idir / "decision.json", {"accepted": False, "reason": "invalid_candidate_prompt"}
            )
            history.append({"iter": i, "accepted": False, "reason": "invalid_prompt"})
            continue

        cand_sel = _eval(candidate, sel_sources)
        cand_sel_obj, cand_raw, cand_pen = score(cand_sel["metrics"], new_head)
        _write_json(idir / "val_metrics.json", cand_sel["metrics"])

        accepted = cand_sel_obj > best_sel_obj + margin
        _write_json(
            idir / "decision.json",
            {
                "accepted": accepted,
                "objective": objective,
                "selection_split": sel_name,
                f"{sel_name}_objective_candidate": cand_sel_obj,
                f"{sel_name}_objective_best": best_sel_obj,
                "raw_objective_candidate": cand_raw,
                "length_penalty_candidate": cand_pen,
                "head_chars": len(new_head),
                "margin": margin,
                "delta": cand_sel_obj - best_sel_obj,
                "candidate_val_metrics": cand_sel["metrics"],
            },
        )
        history.append(
            {
                "iter": i,
                "accepted": accepted,
                f"{sel_name}_objective": cand_sel_obj,
                f"{sel_name}_objective_best_before": best_sel_obj,
                "raw_objective": cand_raw,
                "length_penalty": cand_pen,
                "head_chars": len(new_head),
            }
        )

        if accepted:
            best_full, best_head = candidate, new_head
            best_sel, best_sel_obj = cand_sel, cand_sel_obj
            # Recompute dev evidence for the next reviewer round (cand_sel is the
            # dev eval already when there is no separate val split).
            best_dev = _eval(candidate, dev_sources) if val_sources else cand_sel
            _write_json(idir / "dev_metrics.json", best_dev["metrics"])

    # ---- final: best on held-out TEST, once ----
    (out_dir / "best_prompt.md").write_text(best_full)
    test_eval = _eval(best_full, test_sources)
    _write_json(out_dir / "test_metrics.json", test_eval["metrics"])

    dev_f1 = best_dev["metrics"]["f1_incl_clean"]
    val_f1 = best_sel["metrics"]["f1_incl_clean"]
    test_f1 = test_eval["metrics"]["f1_incl_clean"]
    summary = {
        "objective": objective,
        "fbeta": fbeta,
        "selection_split": sel_name,
        "max_prompt_chars": max_prompt_chars,
        "length_penalty": length_penalty,
        "iters_run": len(history) - 1,
        f"best_{sel_name}_objective": best_sel_obj,
        "best_head_chars": len(best_head),
        "best_dev_metrics": best_dev["metrics"],
        "best_val_metrics": best_sel["metrics"],
        "test_metrics": test_eval["metrics"],
        "test_by_dataset": test_eval["metrics"]["by_dataset"],
        "generalization": {
            "dev_f1": dev_f1,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "val_to_test_gap": val_f1 - test_f1,
            "dev_to_test_gap": dev_f1 - test_f1,
        },
        "history": history,
        "n_dev_sources": len(dev_sources),
        "n_val_sources": len(val_sources),
        "n_test_sources": len(test_sources),
    }
    _write_json(out_dir / "loop_summary.json", summary)
    return summary
