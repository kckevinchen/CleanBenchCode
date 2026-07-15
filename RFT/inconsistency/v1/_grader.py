"""Gold-free reward grader for the inconsistency RFT (v1).

v1 keeps it simple: a SINGLE ``score_model`` judge that, sharing the
trainee's rulebook, scores the model's ``findings`` on BOTH precision and
recall and returns ``5 * F1`` directly in [0, 5] (pass 4). No multi
grader, no python gate, no x5 wrapper.

The Azure-facing details that the (failed) first config got wrong are
kept FIXED here -- they match the proven typo v5/v10 jobs:
  * grader user template is PLAIN TEXT (``Column: {{item.column}}``...),
    not a JSON-object string;
  * ``item.*`` fields: ``column``/``source`` raw, ``column_values_json``
    = ``[[value,count],...]`` array (see ``scripts/build_splits``);
  * the trainee ``response_format`` uses all-``string`` fields (no
    nullable ``["string","null"]`` union).

No gold set is used: the benchmark is incomplete, so the judge derives
the truth set from the column evidence under the rulebook instead of
punishing real-but-unlabeled flags.
"""
from __future__ import annotations

from typing import Any

from RFT.inconsistency.v1._prompt import PROMPT_INCONSISTENCY

# Training-time judge. Azure RFT score_model whitelist (see typo_v5).
DEFAULT_GRADER_MODEL = "gpt-4o-2024-08-06"
_GRADER_MODEL_WHITELIST = ("gpt-4o-2024-08-06", "o3-mini-2025-01-31")


_JUDGE_DEVELOPER = (
    "You grade a data-quality model that audits ONE CSV column for "
    "representational INCONSISTENCY. Judge it on BOTH precision and "
    "recall under the rulebook below. The model is tuned for recall and "
    "deliberately over-proposes; be a precise, fair referee so it learns "
    "where it over-reached and where it missed.\n\n"
    "===== RULEBOOK (the SAME spine the model was given) =====\n"
    f"{PROMPT_INCONSISTENCY}\n"
    "===== END RULEBOOK =====\n\n"
    "Inputs (plain-text labelled fields):\n"
    "  - column:            CSV column header.\n"
    "  - source:            source table id.\n"
    "  - column_values:     JSON array of [value, count] pairs -- your\n"
    "                       only evidence for the dominant convention.\n"
    "  - model_output:      the audit model's raw JSON reply (an object\n"
    "                       with a 'findings' array of {value,\n"
    "                       corrected_value, kind, rationale}).\n\n"
    "PROCEDURE\n"
    "  Step 1 -- Parse model_output as JSON. If invalid or it has no\n"
    "    'findings' array, return Score: 0.0.\n"
    "  Step 2 -- From column_values, characterise the column: (i) its\n"
    "    DOMINANT convention (casing / shape / punctuation / wording /\n"
    "    number-format), if any, and (ii) any sets of values that are the\n"
    "    SAME real-world entity written in different surface forms. A\n"
    "    legitimately heterogeneous column with NO dominant convention AND\n"
    "    no repeated-entity aliases has nothing to be inconsistent with.\n"
    "  Step 3 -- PRECISION. Score EACH finding:\n"
    "      1.0  value is genuinely in scope under EITHER lens -- (A) it is\n"
    "           one surface form of an entity ALSO written differently\n"
    "           elsewhere in the column, OR (B) it is a minority departure\n"
    "           from the dominant convention -- AND corrected_value is a\n"
    "           faithful canonical / convention-matching fix.\n"
    "      0.5  genuinely in scope but corrected_value is wrong, empty,\n"
    "           or a no-op (equal to the value).\n"
    "      0.0  NOT a real inconsistency: a DIFFERENT real-world entity, a\n"
    "           purely stylistic 50/50 tie with no minority and no shared\n"
    "           entity, a sibling error type (typo / semantic / outlier /\n"
    "           FD), or a no-convention column.\n"
    "    precision = mean over findings. If 'findings' is empty, set\n"
    "    precision = 1.0 (vacuous; recall carries the empty case).\n"
    "  Step 4 -- RECALL. Independently scan EVERY distinct value in\n"
    "    column_values and, under the SAME rulebook, build the set G of\n"
    "    genuine in-scope inconsistencies: minority departures from the\n"
    "    dominant convention (Lens B) PLUS the off-forms of any entity that\n"
    "    appears in two or more surface forms (Lens A). EXCLUDE distinct\n"
    "    entities, purely stylistic ties, and sibling-type defects. A value\n"
    "    in G is CAUGHT iff it appears as the 'value' of a finding scored\n"
    "    1.0 or 0.5. recall = caught / |G|. If G is empty, recall = 1.0.\n"
    "  Step 5 -- Return Score = 5 * F1, where F1 = 2*precision*recall /\n"
    "    (precision + recall + 1e-6). Clamp to [0.0, 5.0]. If 'findings'\n"
    "    is empty AND G is empty, return Score: 5.0.\n\n"
    "Respond with EXACTLY two lines:\n"
    "Score: <float 0.0-5.0>\n"
    "Reasoning: <one short sentence>"
)

# Plain-text user template (mirrors v5). No whitespace inside {{ }}.
_JUDGE_USER = (
    "Column: {{item.column}}\n"
    "Source: {{item.source}}\n"
    "Column values (JSON): {{item.column_values_json}}\n"
    "\n"
    "Model output:\n"
    "{{sample.output_text}}"
)


def build_grader(grader_model: str = DEFAULT_GRADER_MODEL) -> dict[str, Any]:
    """Single score_model grader = 5 * F1(LLM precision, LLM recall)."""
    if grader_model not in _GRADER_MODEL_WHITELIST:
        raise ValueError(
            f"grader_model {grader_model!r} not in Azure RFT score_model "
            f"whitelist {_GRADER_MODEL_WHITELIST!r}"
        )
    return {
        "name": "inconsistency_v1_pr",
        "type": "score_model",
        "model": grader_model,
        "input": [
            {"role": "developer", "content": _JUDGE_DEVELOPER},
            {"role": "user", "content": _JUDGE_USER},
        ],
        "pass_threshold": 4.0,
        "range": [0.0, 5.0],
    }
