"""TRAINEE PROMPT + output schema -- EDIT THIS for a new run (the "prompt").

The trainee sees ``TRAINEE_PROMPT`` as its developer message and replies with
the strict JSON schema from ``build_response_format()``. Keep the trainee and
the grader sharing ONE rulebook spine: ``_grader.py`` re-embeds ``TRAINEE_PROMPT``
so the judge grades against the same rules the trainee was given.

Per [[no-benchmark-strings-in-prompts]]: any worked examples here must be
invented, never copied from the benchmark (any split).
"""

from __future__ import annotations

from typing import Any

# The JSON array key the trainee emits. Used by the grader and by the `run`
# verb's prediction parser. Keep it in sync with build_response_format() below.
FINDINGS_KEY = "findings"

# Optional kind enum for the findings (set to () to drop the field).
KINDS: tuple[str, ...] = ("kind_a", "kind_b")

# ---- The trainee developer message (your rulebook) ------------------------
# Replace the body with your task's rules. Tune for RECALL: tell the model to
# over-propose; the grader prunes precision (see RFT.md decisions D11).
TRAINEE_PROMPT = (
    "You audit ONE CSV column for <DESCRIBE THE DEFECT TYPE>. You are given the "
    "column header and its distinct values with row counts. Return EVERY value "
    "that is a genuine in-scope defect, each with a convention-matching "
    "corrected_value. Over-propose for recall; a separate grader prunes "
    "precision.\n\n"
    "Out of scope: distinct real-world entities, sibling error types, and any "
    "value that is already correct.\n\n"
    'Reply with a JSON object: {"findings": [{"value", "corrected_value", '
    '"kind", "rationale"}]}.'
)


def build_user_message(
    column: str,
    source: str,
    values_with_counts: list[tuple[str, int]],
    *,
    max_values: int = 200,
) -> str:
    """Format one column-audit prompt (the trainee's user message).

    ``values_with_counts`` is ``[(value, count), ...]`` ordered by descending
    frequency; capped at ``max_values`` so wide columns stay in budget.
    """
    shown = values_with_counts[:max_values]
    trunc = "" if len(values_with_counts) <= max_values else f" (top {max_values})"
    lines = "\n".join(f'  - "{v}" (count={c})' for v, c in shown) or "  (none)"
    return (
        f"Column header: {column}\n"
        f"Source CSV: {source}\n\n"
        f"Distinct values ({len(values_with_counts)} total{trunc}):\n"
        f"{lines}\n\n"
        "Audit this column per the rulebook and return the findings JSON object."
    )


def build_response_format() -> dict[str, Any]:
    """Strict JSON schema for the trainee's reply.

    Azure RFT requires all fields ``"string"`` (no nullable unions) and
    ``additionalProperties: false`` -- keep that shape if you edit this.
    """
    finding_props: dict[str, Any] = {
        "value": {"type": "string", "description": "Defective value, verbatim."},
        "corrected_value": {
            "type": "string",
            "description": 'Canonical fix; empty string "" if none is defensible.',
        },
        "rationale": {"type": "string", "description": "One short sentence."},
    }
    required = ["value", "corrected_value", "rationale"]
    if KINDS:
        finding_props["kind"] = {"type": "string", "enum": list(KINDS)}
        required.insert(2, "kind")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{FINDINGS_KEY}_schema",
            "schema": {
                "type": "object",
                "properties": {
                    FINDINGS_KEY: {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": finding_props,
                            "required": required,
                            "additionalProperties": False,
                        },
                    }
                },
                "required": [FINDINGS_KEY],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
