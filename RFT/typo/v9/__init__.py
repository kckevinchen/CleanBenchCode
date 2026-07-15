"""Typo RFT v9.

v9 forks BOTH sides of v7/v8. The trainee developer message
(``PROMPT_V9``) is v8's prompt with the proper-noun-hunting clauses
removed and names put out of scope for character-level fixes, so the
trainee no longer proposes the corrections the grader would reject. The
rulebook used by the precision sub-grader is forked at
``RFT/typo/v9/prompts/rulebook.md`` to put proper nouns out of scope for
character-level typo corrections:

  * Predictions whose value is a proper noun (personal / geographic /
    address / brand / org / product / scientific-binomial) AND whose
    defect is a character-level slip score as ``fp`` (precision 0.0).
  * Predictions on proper-noun values whose defect is STRUCTURAL
    (adjacent duplicated word, doubled punctuation) remain ``valid``
    -- these defects are recognisable from the value alone without
    world knowledge about the canonical name.

The reward shapes (r1 additive, r2 harmonic, r3 weighted) and the
trivial-FP / column-cap / python-recall sub-graders are inherited
unchanged from v7.
"""

from RFT.typo.v9._prompt import PROMPT_V9  # noqa: F401
from RFT.typo.v9._grader_variants import (  # noqa: F401
    V9_VARIANTS,
    build_grader_v9,
)
