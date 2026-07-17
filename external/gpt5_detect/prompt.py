"""Unified single-prompt cell-error detector prompt (GPT-5.5 baseline).

The flagship pipeline issues ONE LLM call per (chunk x error_type). This baseline
instead asks the model to detect ALL FOUR cell-level error types in a SINGLE
prompt per chunk:

    typo, inconsistency, semantic_incompatible, numeric_outlier

To stay faithful to the benchmark's taxonomy, the per-type DEFINITION / EXAMPLE /
NOTES blocks are pulled verbatim from ``prompts/error_types.yaml`` (the same
source of truth the per-type pipeline uses) and concatenated into one system
prompt. The output contract is one JSON ``issues`` array with one entry per
affected CELL, so parsing to (row, column) is unambiguous for cell-level scoring.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ERROR_TYPES_YAML = REPO_ROOT / "prompts" / "error_types.yaml"

# The four single-column (cell-level) error types, in taxonomy order.
CELL_TYPES = ["typo", "inconsistency", "semantic_incompatible", "numeric_outlier"]


def _load_type_blocks() -> dict[str, dict]:
    data = yaml.safe_load(ERROR_TYPES_YAML.read_text(encoding="utf-8"))
    return data["types"]


def build_system_prompt() -> str:
    """Compose the single multi-type system prompt from the taxonomy YAML."""
    types = _load_type_blocks()
    blocks: list[str] = []
    for key in CELL_TYPES:
        t = types[key]
        block = [
            f"========================= ERROR TYPE: {key} =========================",
            f"NAME: {t.get('name', key)}    TAXONOMY TAG: {t.get('category', '')}",
            "",
            "DEFINITION",
            str(t.get("description", "")).strip(),
        ]
        if t.get("example"):
            block += ["", "REFERENCE EXAMPLES (guidance; may or may not appear in the data)",
                      str(t["example"]).strip()]
        if t.get("notes"):
            block += ["", "NOTES", str(t["notes"]).strip()]
        blocks.append("\n".join(block))

    type_defs = "\n\n".join(blocks)

    return f"""\
You are a meticulous data-quality auditor. You inspect a slice of a CSV table and
flag CELL-LEVEL data-quality errors in a SINGLE pass. You consider FOUR error
types at once and label each finding with the type it matches:

  - typo                  (single-cell text typo: spelling / internal-whitespace / punctuation)
  - inconsistency         (a value formatted/represented differently from its column's peers)
  - semantic_incompatible (a value whose meaning does not fit its column's concept)
  - numeric_outlier       (a numeric value implausibly far from its column's distribution)

FAVOR RECALL. A downstream human reviewer can reject false positives, but a missed
defect is gone forever. When a cell plausibly matches one of the four definitions,
FLAG it; use ``confidence: "low"`` instead of staying silent on partial evidence.

Confidence: ``high`` = unambiguous; ``medium`` = clear evidence with one
uncertainty; ``low`` = borderline / sparse evidence. When undecided between two
levels, pick the lower one.

Empty cells (NULL, NaN, ``""``, whitespace-only) are NEVER defects; ignore them.

Pick the SINGLE best-fitting error type per cell. If a cell could match two types,
choose the most specific one and flag it once (do not emit duplicate issues for
the same cell). Stay within these four types; ignore multi-row / multi-column /
cross-table problems (handled elsewhere).

The four error-type definitions (authoritative):

{type_defs}

OUTPUT FORMAT
-------------
Respond with a SINGLE JSON object and nothing else (no Markdown fences, no prose):

{{
  "issues": [
    {{
      "row": <int>,                 // the __row__ value of the affected cell
      "column": "<column name>",    // exactly one column (the affected cell)
      "error_type": "typo" | "inconsistency" | "semantic_incompatible" | "numeric_outlier",
      "observed": "<verbatim problematic value>",
      "explanation": "<1-2 sentence justification>",
      "corrected_value": "<literal canonical replacement>" | null,
      "confidence": "high" | "medium" | "low"
    }}
  ]
}}

Emit ONE issue per affected CELL (one row + one column). For a group-level
inconsistency that affects many cells of a column, emit one issue per affected
cell. If nothing applies, return exactly: {{"issues": []}}
"""


USER_TEMPLATE = """\
Audit the CSV slice below for the FOUR cell-level error types defined in your
instructions. Flag every matching cell.

COLUMN HEADERS (the complete set of columns you may reference):
{headers}

ROWS (Markdown table; the first column `__row__` is the 0-indexed source CSV row
number, covering rows [{start_row}, {end_row})):

{rows}

WHITESPACE MARKERS in the table (sentinels for characters a Markdown table would
otherwise hide); treat each marker as the original character:
  * "\u2423" = SPACE (regular or non-breaking).
  * "\u2192" = TAB.
  * "\u21b5" = NEWLINE inside a cell.
  * "\u00b7" = zero-width char (ZWSP, ZWNJ, ZWJ, soft hyphen, BOM).

Return the JSON object now.
"""


def build_user_prompt(headers: list[str], rows_markdown: str, start_row: int, end_row: int) -> str:
    return USER_TEMPLATE.format(
        headers=", ".join(headers),
        rows=rows_markdown,
        start_row=start_row,
        end_row=end_row,
    )


if __name__ == "__main__":  # quick inspection: print the composed system prompt
    print(build_system_prompt())
