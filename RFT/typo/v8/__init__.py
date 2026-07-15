"""Typo RFT v8.

v8 swaps the trainee developer message from the v4 rulebook (used by
v6/v7) to PROMPT_V8 -- a recall-leaning prompt (r4 in the v7 sweep
harness) that lifts o4-mini recall from 0.758 to 0.939 on the v7 eval
split. Splits, grader, response schema, base model, and training
recipe are otherwise inherited from v7.

The v7 grader (RFT.typo.v7._grader_variants) is unchanged: the
precision sub-grader prunes the larger candidate pool the new prompt
emits, while the recall sub-grader rewards the lifted recall.
"""

from RFT.typo.v8._prompt import PROMPT_V8  # noqa: F401
