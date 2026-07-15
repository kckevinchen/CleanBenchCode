"""Per-column typo-audit task assets for the GPT-5.5 baseline.

Vendored from ``RFT.typo.v5._task`` so ``baseline/`` is self-contained and does
not reach into the RFT tree. These define the *user message* (the per-column
distinct-value listing) and the strict JSON *response format* shared by both
baseline prompts. The system prompt itself lives in ``baseline/prompts/*.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Three typo flavours and the cap on distinct values shown to the model.
FLAVOURS: tuple[str, ...] = ("spelling", "punctuation", "spacing")
MAX_DISTINCT = 500


def format_distinct_values(
    values_with_counts: Sequence[tuple[str, int]],
    *,
    max_distinct: int = MAX_DISTINCT,
) -> tuple[str, int, bool]:
    """Render distinct values as a numbered listing, count-descending.

    Returns ``(listing_text, n_shown, truncated)``.
    """
    items = sorted(
        values_with_counts,
        key=lambda kv: (-int(kv[1]), str(kv[0])),
    )
    truncated = len(items) > max_distinct
    shown = items[:max_distinct]
    width = len(str(len(shown))) if shown else 1
    cw = max((len(str(c)) for _, c in shown), default=1)
    lines = [
        f"  {i:>{width}}.  {int(c):>{cw}}x  {str(v)!r}" for i, (v, c) in enumerate(shown, start=1)
    ]
    return "\n".join(lines), len(shown), truncated


def build_user_message(
    *,
    column: str,
    source: str,
    values_with_counts: Sequence[tuple[str, int]],
    max_distinct: int = MAX_DISTINCT,
) -> str:
    listing, n_shown, truncated = format_distinct_values(
        values_with_counts,
        max_distinct=max_distinct,
    )
    trunc_note = (
        f"\n  (showing the {n_shown} most-frequent values; the column " f"has more)"
        if truncated
        else ""
    )
    return (
        f"Column header: {column}\n"
        f"Source CSV: {source}\n"
        f"Distinct values in this column ({n_shown} shown"
        f"{trunc_note}):\n"
        f"{listing}\n\n"
        "Return the JSON object listing every typo you find."
    )


def build_response_format() -> dict[str, Any]:
    """Strict chat-style ``response_format`` for the typo column audit."""
    typo_entry = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "description": (
                    "The typo-bearing distinct value, copied " "verbatim from the input listing."
                ),
            },
            "correction": {
                "type": "string",
                "description": "Intended canonical form.",
            },
            "flavour": {
                "type": "string",
                "enum": list(FLAVOURS),
            },
            "rationale": {
                "type": "string",
                "description": "One short sentence naming the defect.",
            },
        },
        "required": ["value", "correction", "flavour", "rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "typo_column_audit",
            "schema": {
                "type": "object",
                "properties": {
                    "typos": {
                        "type": "array",
                        "items": typo_entry,
                        "description": (
                            "Every typo found in the column. Empty " "if the column has no typos."
                        ),
                    },
                },
                "required": ["typos"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
