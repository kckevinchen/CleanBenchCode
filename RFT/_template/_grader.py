"""GRADER -- EDIT THIS for a new run (or leave the default).

The default is the shared gold-free ``5 * F1(precision, recall)`` judge
(:func:`RFT._common.graders.build_pr_judge_grader`): pass your rulebook
(``TRAINEE_PROMPT``) and a one-line task description and you have a working
reward -- no grader written from scratch.

Need more? Add reward-shape variants or python sub-graders in this module and
return the appropriate grader dict per ``variant``.
"""

from __future__ import annotations

from typing import Any

from RFT._common.graders import DEFAULT_GRADER_MODEL, build_pr_judge_grader
from RFT._template._prompt import FINDINGS_KEY, TRAINEE_PROMPT

# One short noun phrase naming the defect, e.g. "typos" or
# "representational INCONSISTENCY". Folded into the judge prompt.
TASK_DESCRIPTION = "<the defect this task detects>"


def build_grader(variant: str, grader_model: str = DEFAULT_GRADER_MODEL) -> dict[str, Any]:
    """Return the Azure RFT grader. ``variant`` is ignored for the single
    gold-free judge; branch on it if you add reward-shape variants."""
    return build_pr_judge_grader(
        name="example_pr",
        rulebook=TRAINEE_PROMPT,
        task_description=TASK_DESCRIPTION,
        findings_key=FINDINGS_KEY,
        grader_model=grader_model,
    )
