"""Typo RFT v11 -- fully gold-free reward (two LLM judges).

v11 trains the typo trainee on UNLABELLED columns so the entire human
benchmark can be reserved for eval. With no gold set, the deterministic
gold-recall sub-grader is gone; its job -- supplying a recall signal and
stopping the model collapsing to the empty prediction -- is taken over by a
second LLM ``score_model`` judge:

  * PRECISION judge -- v10's precision judge verbatim (grades proposals).
  * RECALL judge    -- NEW: scans the column for in-scope typos the model
                       MISSED, so an empty prediction on a dirty column scores
                       ~0 while a genuinely clean column still scores 1.0.

Plus the two gold-free deterministic gates (trivial-FP, column-cap). The
trainee prompt and rulebook are v10's verbatim (this version imports no peer
version). See ``RFT/typo/v11/_grader.py`` for the full design and the Azure
``score_model`` whitelist constraint on the grader model.
"""

from RFT.typo.v11._prompt import PROMPT_V11  # noqa: F401
from RFT.typo.v11._grader import (  # noqa: F401
    V11_VARIANTS,
    build_grader_v11,
    build_precision_judge_v11,
    build_recall_judge_v11,
)
