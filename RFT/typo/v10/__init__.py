"""Typo RFT v10.

v10 forks v9 with a SINGLE change: the trainee developer message
(``PROMPT_V10``) drops v9's two explicit "recall-push" passages
(the "Your job is RECALL ... propose generously" sentence and the
"Borderline cases ARE in scope ... include it" clause). v10 does not
chase the *trivial* typo cells those passages bought back -- the removal
costs ~6 pool-recall cells, all in the trivial bucket, while non-trivial
recall is unchanged (see ``RFT/typo/v10/_prompt.py`` for the numbers).

The grader is the same recipe v9 introduced (proper nouns out of scope for
character-level slips; r1/r2/r3 reward shapes over the trivial-FP / column-cap /
python-recall sub-graders). To keep this version self-contained -- importing no
peer version -- v10 carries its OWN copy of that grader
(``RFT.typo.v10._grader_variants``) and rulebook
(``RFT/typo/v10/prompts/rulebook.md``, identical to v9's). Shared rulebook-free
primitives come from ``RFT._common.typo_task``.

Only the trainee prompt and the data splits (retargeted to ``PROMPT_V10``)
differ from v9.
"""

from RFT.typo.v10._prompt import PROMPT_V10  # noqa: F401

# v10's own grader (a copy of the v9 recipe, loading v10's rulebook) -- no
# import of any peer version.
from RFT.typo.v10._grader_variants import (  # noqa: F401
    V10_VARIANTS,
    build_grader_v10,
)
