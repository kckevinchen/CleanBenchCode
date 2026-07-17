"""Per-column inconsistency-audit task assets + rulebook surgery.

Vendored / adapted from ``RFT.inconsistency.v1`` so ``external/baseline/`` stays
self-contained (decoupled from the RFT tree):

* ``build_user_message`` / ``build_response_format`` / ``KINDS``
  <- ``RFT.inconsistency.v1._prompt``
* ``parse_findings``  <- ``RFT.inconsistency.v1.eval.run_baseline._parse_findings``
  (extended to keep ``kind`` + ``rationale`` so the reviewer can see *why* the
  detector flagged each value).

It also owns the **rulebook surgery** that makes the information barrier
mechanical: :func:`split_rulebook` peels the frozen Output section (the JSON
schema the parser depends on) off the editable rules, so the prompt editor can
only ever rewrite the rules and never the response contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

# The five inconsistency sub-patterns (the rulebook's ``kind`` enum).
KINDS: tuple[str, ...] = ("casing", "punctuation", "pattern", "text", "number_format")

# Distinct values shown to the detector per column (the rulebook says ~200).
MAX_DISTINCT = 200

# Everything from this header to the end of the rulebook is FROZEN: it carries
# the JSON output schema the parser relies on and the closing recall stance.
OUTPUT_SECTION_ANCHOR = "## 4. Output"


def norm(s: Any) -> str:
    """Whitespace-collapsed string key (matches the grader's normaliser)."""
    return " ".join(str(s).split())


def build_user_message(
    *,
    column: str,
    source: str,
    values_with_counts: Sequence[tuple[str, int]],
    max_values: int = MAX_DISTINCT,
) -> str:
    """Format one column-audit user message (count-descending value listing)."""
    items = sorted(values_with_counts, key=lambda kv: (-int(kv[1]), str(kv[0])))
    truncated = len(items) > max_values
    shown = items[:max_values]
    trunc = f" (showing top {max_values} by frequency)" if truncated else ""
    lines = "\n".join(f'  - "{v}" (count={c})' for v, c in shown) or "  (none)"
    return (
        f"Column header: {column}\n"
        f"Source CSV: {source}\n\n"
        f"Distinct values in this column ({len(shown)} total{trunc}):\n"
        f"{lines}\n\n"
        "Audit this column for representational inconsistency per the "
        "rulebook and return the findings JSON object."
    )


def build_response_format() -> dict[str, Any]:
    """Strict chat-style ``response_format`` (the rulebook's sec.4 schema)."""
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
                                    "description": "Minority value, verbatim from the input list.",
                                },
                                "corrected_value": {
                                    "type": "string",
                                    "description": (
                                        'Convention-matching canonical form; empty string "" '
                                        "if none is defensible."
                                    ),
                                },
                                "kind": {"type": "string", "enum": list(KINDS)},
                                "rationale": {
                                    "type": "string",
                                    "description": "One short sentence.",
                                },
                            },
                            "required": ["value", "corrected_value", "kind", "rationale"],
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


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    return s


def parse_findings(raw: str | None) -> dict[str, dict[str, Any]]:
    """Parse the detector reply into ``{norm(value): {corrected_value, kind, rationale}}``.

    Tolerates ```` ```json ```` fences. Unparseable / empty -> ``{}``.
    """
    if not raw:
        return {}
    try:
        obj = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for it in obj.get("findings", []) or []:
        if not isinstance(it, dict):
            continue
        v = it.get("value")
        if isinstance(v, str):
            out[norm(v)] = {
                "corrected_value": it.get("corrected_value"),
                "kind": it.get("kind"),
                "rationale": it.get("rationale"),
            }
    return out


def split_rulebook(text: str) -> tuple[str, str]:
    """Split a rulebook into ``(editable_head, frozen_tail)`` at the Output section.

    ``frozen_tail`` starts at :data:`OUTPUT_SECTION_ANCHOR` (the JSON schema +
    closing stance) and is re-attached verbatim after every edit, so the editor
    can never alter the response contract. If the anchor is absent the whole
    text is editable and the tail is empty (callers should treat that as a
    validation failure).
    """
    idx = text.find(OUTPUT_SECTION_ANCHOR)
    if idx == -1:
        return text, ""
    return text[:idx].rstrip() + "\n", text[idx:]


def join_rulebook(editable_head: str, frozen_tail: str) -> str:
    """Re-assemble a full rulebook from an edited head + the frozen tail."""
    return editable_head.rstrip() + "\n\n" + frozen_tail.lstrip()


def has_output_contract(text: str) -> bool:
    """True if ``text`` still carries the frozen Output schema anchors.

    Used to validate an editor result before it is allowed into a run.
    """
    return OUTPUT_SECTION_ANCHOR in text and all(
        k in text for k in ('"findings"', '"value"', '"corrected_value"', '"kind"')
    )
