"""Reusable RFT grader builders.

The single hardest part of standing up a new RFT run is the grader. This
module factors out the **gold-free precision/recall judge** that
``inconsistency_v1`` proved out -- a single ``score_model`` that shares the
trainee's rulebook, scores the model's findings on both precision and recall,
and returns ``5 * F1`` in ``[0, 5]`` -- so a new task can get a working grader
by passing its rulebook text and a one-line task description.

Why gold-free: the benchmarks are incomplete, so a judge that derives the truth
set from the column evidence under the rulebook beats one that punishes real-
but-unlabeled flags. It is also the simplest config that reliably clears Azure
preprocessing (the Azure-facing layout below matches the proven typo v5/v10 and
inconsistency v1 jobs -- plain-text user template, ``item.*`` raw fields).
"""

from __future__ import annotations

from typing import Any

# Azure RFT ``score_model`` whitelist (models the service allows as a grader).
RFT_SCORE_MODEL_WHITELIST: tuple[str, ...] = (
    "gpt-4o-2024-08-06",
    "o3-mini-2025-01-31",
)
DEFAULT_GRADER_MODEL = "gpt-4o-2024-08-06"

# Standard per-column-audit item fields (what build_splits must emit). Override
# ``user_template`` if a task exposes a different evidence shape. No whitespace
# inside ``{{ }}`` (Azure template requirement).
DEFAULT_COLUMN_AUDIT_USER_TEMPLATE = (
    "Column: {{item.column}}\n"
    "Source: {{item.source}}\n"
    "Column values (JSON): {{item.column_values_json}}\n"
    "\n"
    "Model output:\n"
    "{{sample.output_text}}"
)


def validate_grader_model(grader_model: str) -> str:
    if grader_model not in RFT_SCORE_MODEL_WHITELIST:
        raise ValueError(
            f"grader_model {grader_model!r} not in Azure RFT score_model "
            f"whitelist {RFT_SCORE_MODEL_WHITELIST!r}"
        )
    return grader_model


def _pr_judge_developer(
    *, task_description: str, rulebook: str, findings_key: str, finding_shape: str
) -> str:
    """Generic gold-free precision/recall judge prompt.

    Task-specific scope lives entirely in ``rulebook`` (the same spine the
    trainee saw); this scaffolding only defines HOW to turn that into a score.
    """
    return (
        f"You grade a data-quality model that audits ONE CSV column for "
        f"{task_description}. Judge it on BOTH precision and recall under the "
        f"rulebook below. The model is tuned for recall and deliberately "
        f"over-proposes; be a precise, fair referee so it learns where it "
        f"over-reached and where it missed.\n\n"
        f"===== RULEBOOK (the SAME spine the model was given) =====\n"
        f"{rulebook}\n"
        f"===== END RULEBOOK =====\n\n"
        f"Inputs (plain-text labelled fields):\n"
        f"  - column:        CSV column header.\n"
        f"  - source:        source table id.\n"
        f"  - column_values: JSON array of [value, count] pairs -- your only\n"
        f"                   evidence for what the column should look like.\n"
        f"  - model_output:  the audit model's raw JSON reply (an object with\n"
        f"                   a '{findings_key}' array of {finding_shape}).\n\n"
        f"PROCEDURE\n"
        f"  Step 1 -- Parse model_output as JSON. If invalid or it has no\n"
        f"    '{findings_key}' array, return Score: 0.0.\n"
        f"  Step 2 -- PRECISION. Score EACH finding under the rulebook:\n"
        f"      1.0  a genuine in-scope defect AND a faithful corrected_value;\n"
        f"      0.5  a genuine defect but the fix is wrong, empty, or a no-op;\n"
        f"      0.0  not a real in-scope defect (a distinct entity, a sibling\n"
        f"           error type, or out of scope per the rulebook).\n"
        f"    precision = mean over findings. If '{findings_key}' is empty,\n"
        f"    set precision = 1.0 (vacuous; recall carries the empty case).\n"
        f"  Step 3 -- RECALL. Independently scan EVERY distinct value in\n"
        f"    column_values and, under the SAME rulebook, build the set G of\n"
        f"    values that ARE genuine in-scope defects. A value in G is CAUGHT\n"
        f"    iff it is the 'value' of a finding scored 1.0 or 0.5.\n"
        f"    recall = |caught| / |G|. If G is empty, recall = 1.0.\n"
        f"  Step 4 -- Return Score = 5 * F1, F1 = 2*precision*recall /\n"
        f"    (precision + recall + 1e-6). Clamp to [0.0, 5.0]. If\n"
        f"    '{findings_key}' is empty AND G is empty, return Score: 5.0.\n\n"
        f"Respond with EXACTLY two lines:\n"
        f"Score: <float 0.0-5.0>\n"
        f"Reasoning: <one short sentence>"
    )


def build_pr_judge_grader(
    *,
    name: str,
    rulebook: str,
    task_description: str,
    findings_key: str = "findings",
    finding_shape: str = "{value, corrected_value, kind, rationale}",
    grader_model: str = DEFAULT_GRADER_MODEL,
    user_template: str = DEFAULT_COLUMN_AUDIT_USER_TEMPLATE,
    pass_threshold: float = 4.0,
) -> dict[str, Any]:
    """A gold-free ``score_model`` grader = ``5 * F1(precision, recall)``.

    This is the recommended starting grader for a new per-column-audit task:
    pass the trainee's ``rulebook`` and a short ``task_description`` (e.g.
    ``"typos"`` or ``"representational INCONSISTENCY"``) and you have a working
    reward. Returns the Azure RFT grader dict.
    """
    validate_grader_model(grader_model)
    developer = _pr_judge_developer(
        task_description=task_description,
        rulebook=rulebook,
        findings_key=findings_key,
        finding_shape=finding_shape,
    )
    return {
        "name": name,
        "type": "score_model",
        "model": grader_model,
        "input": [
            {"role": "developer", "content": developer},
            {"role": "user", "content": user_template},
        ],
        "pass_threshold": pass_threshold,
        "range": [0.0, 5.0],
    }
