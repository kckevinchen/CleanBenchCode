"""Task spec for the typo v11 RFT -- fully GOLD-FREE training.

v11 trains on UNLABELLED columns (``data/curated_bing21_dedup_training``) and
reserves the whole human benchmark for eval, so the reward has no gold set: the
grader is ``build_grader_v11`` -- two LLM ``score_model`` judges (precision +
recall) plus the two deterministic python gates (trivial-FP, column-cap). There
are no gold-anchored variants; the reward grid is the gold-free r1/r2/r3.

The trainee prompt (``PROMPT_V11``) and rulebook are v10's verbatim, so the
high-recall starting point (RFT.md D11) carries over. ``gold_field`` below wires
the *post-hoc* ``rft run`` micro-F1 scorer to the benchmark eval split's gold;
it is NOT used by the training grader (which never sees a label).

Build the splits before submitting::

    python -m RFT.typo.v11.scripts.build_v11_splits
"""

from __future__ import annotations

from pathlib import Path

from RFT._common import typo_task as _v5  # shared primitives (no peer-version dep)
from RFT._common.spec import Hparams, make_spec
from RFT.typo.v11 import V11_VARIANTS, build_grader_v11

_DIR = Path(__file__).resolve().parent


SPEC = make_spec(
    name="typo_v11",
    error_type="typo",
    version="v11",
    data_dir=_DIR / "data",
    response_format=_v5.build_response_format,
    build_grader=build_grader_v11,
    findings_key="typos",
    gold_field="gold_typos",
    ambiguous_field="ambiguous_values",
    default_suffix="typo-rft-v11",
    default_grader_model=_v5.TYPO_V5_DEFAULT_GRADER_MODEL,
    grader_model_whitelist=tuple(_v5.TYPO_V5_GRADER_MODEL_WHITELIST),
    reward_variants=tuple(V11_VARIANTS),
    default_reward_variant="r1",
    hparams=Hparams(n_epochs=2, reasoning_effort="medium"),
)
