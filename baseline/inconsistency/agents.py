"""The two LLM agents: reviewer (sees data) and prompt editor (never sees data).

Both run over an injectable :class:`AgentBackend` — a callable
``(instructions, user_input) -> text`` — so the loop uses the live Responses-API
backend while tests and ``--dry-run`` inject deterministic stubs.

The barrier lives in the *call sites*, not here: :func:`review` is handed dev
evidence; :func:`edit_prompt` is handed only the rulebook head + already-filtered
suggestions. Neither function can reach data the orchestrator did not pass it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from baseline.inconsistency._task import _strip_fences


class AgentBackend(Protocol):
    def __call__(self, instructions: str, user_input: str) -> str: ...


# ---------------------------------------------------------------------------
# Reviewer (sees dev metrics + FP/FN evidence)
# ---------------------------------------------------------------------------


def _fmt_fp(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "  (none)"
    rows = []
    for e in examples:
        rows.append(
            f'  - column={e.get("column")!r} value={e.get("value")!r} '
            f'kind={e.get("kind")!r} fix={e.get("pred_correction")!r} :: '
            f'{e.get("rationale")!r}'
        )
    return "\n".join(rows)


def _fmt_fn(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "  (none)"
    return "\n".join(
        f'  - column={e.get("column")!r} missed_value={e.get("value")!r} '
        f'gold_fix={e.get("gold_correction")!r}'
        for e in examples
    )


def build_review_input(
    rulebook_head: str,
    metrics: dict[str, Any],
    evidence: dict[str, Any],
    *,
    fp_cap: int = 40,
    fn_cap: int = 40,
) -> str:
    m = metrics
    return (
        "# Current rulebook (editable part)\n\n"
        f"{rulebook_head}\n\n"
        "# Dev-set score\n"
        f"  precision(incl clean)={m.get('precision_incl_clean'):.3f}  "
        f"recall={m.get('recall'):.3f}  f1={m.get('f1_incl_clean'):.3f}\n"
        f"  TP={m.get('tp')}  FP={m.get('fp')} (labeled={m.get('fp_labeled')}, "
        f"clean={m.get('fp_clean')})  FN={m.get('fn')}\n\n"
        "# False positives (flagged but NOT an error) — fix by tightening/scoping\n"
        f"{_fmt_fp(evidence.get('fp', [])[:fp_cap])}\n\n"
        "# False negatives (real errors missed) — fix by broadening/clarifying\n"
        f"{_fmt_fn(evidence.get('fn', [])[:fn_cap])}\n\n"
        "Produce the JSON critique now. Describe every pattern generically — do "
        "not quote any value or column header above verbatim."
    )


def _parse_critique(raw: str) -> dict[str, Any]:
    """Tolerantly parse the reviewer reply into a critique dict."""
    s = _strip_fences(raw or "")
    # Grab the outermost JSON object if there is surrounding prose.
    if not s.startswith("{"):
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            s = m.group(0)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return {
            "dominant_problem": "balanced",
            "summary": "",
            "suggestions": [],
            "_parse_error": True,
            "_raw": raw,
        }
    if not isinstance(obj, dict):
        return {"dominant_problem": "balanced", "summary": "", "suggestions": []}
    obj.setdefault("suggestions", [])
    if not isinstance(obj["suggestions"], list):
        obj["suggestions"] = []
    return obj


def review(
    backend: AgentBackend,
    reviewer_prompt: str,
    rulebook_head: str,
    metrics: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Run the reviewer; return a parsed critique dict (sees dev data)."""
    user_input = build_review_input(rulebook_head, metrics, evidence)
    raw = backend(reviewer_prompt, user_input)
    return _parse_critique(raw)


# ---------------------------------------------------------------------------
# Prompt editor (sees only rulebook head + filtered suggestions)
# ---------------------------------------------------------------------------


_EDITOR_FIELDS = ("failure_mode", "hypothesized_cause", "rule_change", "kind", "priority")


def build_edit_input(rulebook_head: str, filtered_critique: dict[str, Any]) -> str:
    # Project each suggestion onto a fixed whitelist: even if the reviewer added
    # an unexpected field, it can never reach the editor (defense-in-depth on top
    # of the leakage filter).
    suggestions = [
        {k: s[k] for k in _EDITOR_FIELDS if k in s}
        for s in filtered_critique.get("suggestions", [])
        if isinstance(s, dict)
    ]
    return (
        "# Current rulebook (editable part — rewrite this)\n\n"
        f"{rulebook_head}\n\n"
        "# Reviewer suggestions (already stripped of any data values)\n"
        f"{json.dumps(suggestions, indent=2, ensure_ascii=False)}\n\n"
        "Rewrite the rulebook to apply these suggestions. Return only the revised "
        "rulebook markdown (the editable part), no Output/JSON section."
    )


def edit_prompt(
    backend: AgentBackend,
    editor_prompt: str,
    rulebook_head: str,
    filtered_critique: dict[str, Any],
) -> str:
    """Run the editor; return the revised editable head (never sees data)."""
    user_input = build_edit_input(rulebook_head, filtered_critique)
    raw = backend(editor_prompt, user_input)
    text = (raw or "").strip()
    # Editors sometimes wrap the whole doc in a fence despite instructions.
    if text.startswith("```"):
        text = _strip_fences(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Live backend (Responses API, gpt-5.5)
# ---------------------------------------------------------------------------


class ResponsesAgentBackend:
    """Live agent backend: one Responses-API call per invocation."""

    def __init__(self, client: Any, *, model: str = "gpt-5.5", effort: str = "medium") -> None:
        self.client = client
        self.model = model
        self.effort = effort

    def __call__(self, instructions: str, user_input: str) -> str:
        from baseline._common import extract_responses_output_text

        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
            reasoning={"effort": self.effort},
        )
        return extract_responses_output_text(resp)
