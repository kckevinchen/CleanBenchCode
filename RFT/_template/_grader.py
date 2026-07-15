"""GRADER -- EDIT THIS for a new run (or leave the default).

The default is the shared gold-free ``5 * F1(precision, recall)`` judge
(:func:`RFT._common.graders.build_pr_judge_grader`): pass your rulebook
(``TRAINEE_PROMPT``) and a one-line task description and you have a working
reward -- no grader written from scratch.

Need more? For reward-shape variants (r1/r2/r3) or python sub-graders
(trivial-FP gate, column cap, gold-recall leg), see ``RFT/typo/v5/_task.py`` and
``RFT/typo/v9/_grader_variants.py`` for multigrader examples, then return the
appropriate dict per ``variant`` here.
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
        name="example_v1_pr",
        rulebook=TRAINEE_PROMPT,
        task_description=TASK_DESCRIPTION,
        findings_key=FINDINGS_KEY,
        grader_model=grader_model,
    )
