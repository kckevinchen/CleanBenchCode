"""Trainee prompt + I/O schema for the inconsistency RFT (v1).

Whole-column framing: the trainee sees one column's distinct values (with
row counts) and emits every minority value that departs from the column's
dominant representational convention, paired with a convention-matching
correction. The developer message IS the rulebook
(``prompts/rulebook.md``) so the trainee and the LLM judge share one spine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent
RULEBOOK_PATH = _DIR / "prompts" / "rulebook.md"

# The developer/system message. The rulebook already carries the output
# schema (sec.4) and the self-check (sec.5).
PROMPT_INCONSISTENCY = RULEBOOK_PATH.read_text(encoding="utf-8")

KINDS = ("casing", "punctuation", "pattern", "text", "number_format")


def build_user_message(
    column: str,
    source: str,
    values_with_counts: list[tuple[str, int]],
    *,
    max_values: int = 200,
) -> str:
    """Format one column-audit prompt.

    ``values_with_counts`` is ``[(value, count), ...]`` ordered by
    descending frequency. Capped at ``max_values`` so very wide columns
    stay within the context budget (the dominant convention is visible
    from the high-frequency head regardless).
    """
    shown = values_with_counts[:max_values]
    trunc = (
        ""
        if len(values_with_counts) <= max_values
        else f" (showing top {max_values} by frequency)"
    )
    lines = "\n".join(f'  - "{v}" (count={c})' for v, c in shown) or "  (none)"
    return (
        f"Column header: {column}\n"
        f"Source CSV: {source}\n\n"
        f"Distinct values in this column "
        f"({len(values_with_counts)} total{trunc}):\n"
        f"{lines}\n\n"
        "Audit this column for representational inconsistency per the "
        "rulebook and return the findings JSON object."
    )


def build_response_format() -> dict[str, Any]:
    """Strict JSON schema for the trainee's reply (sec.4 of the rulebook)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "inconsistency_findings",
            "schema": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": {
                                    "type": "string",
                                    "description": (
                                        "Minority value, verbatim from the "
                                        "input list."
                                    ),
                                },
                                "corrected_value": {
                                    "type": "string",
                                    "description": (
                                        "Convention-matching canonical form; "
                                        "empty string \"\" if none is "
                                        "defensible."
                                    ),
                                },
                                "kind": {
                                    "type": "string",
                                    "enum": list(KINDS),
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "One short sentence.",
                                },
                            },
                            "required": [
                                "value",
                                "corrected_value",
                                "kind",
                                "rationale",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["findings"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
