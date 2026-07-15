"""Task spec for the typo v10 RFT (replaces ``scripts/typo_rft_v10.py``).

v10 = v9 grader verbatim, only the trainee prompt (``PROMPT_V10``, baked into
the splits) differs. Beyond the r1/r2/r3 reward shapes, v10 exposes two
gold-free grader variants (``nogold`` and ``nogold-recall``) that the old
launcher selected via ``--no-gold-reward`` / ``--llm-recall`` flags; here they
are simply additional reward variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from RFT._common import typo_task as _v5  # shared primitives (no peer-version dep)
from RFT._common.spec import Hparams, make_spec
from RFT.typo.v10 import V10_VARIANTS, build_grader_v10
from RFT.typo.v10._grader_nogold import (
    build_grader_v10_nogold,
    build_grader_v10_nogold_recall,
)

_DIR = Path(__file__).resolve().parent

# r1/r2/r3 are gold-anchored; nogold / nogold-recall are the gold-free judges.
REWARD_VARIANTS = tuple(V10_VARIANTS) + ("nogold", "nogold-recall")


def _grader(variant: str, grader_model: str) -> dict[str, Any]:
    if variant == "nogold":
        return build_grader_v10_nogold(grader_model=grader_model)
    if variant == "nogold-recall":
        return build_grader_v10_nogold_recall(grader_model=grader_model)
    return build_grader_v10(variant, grader_model=grader_model)


SPEC = make_spec(
    name="typo_v10",
    error_type="typo",
    version="v10",
    data_dir=_DIR / "data",
    response_format=_v5.build_response_format,
    build_grader=_grader,
    findings_key="typos",
    gold_field="gold_typos",
    ambiguous_field="ambiguous_values",
    default_suffix="typo-rft-v10",
    default_grader_model=_v5.TYPO_V5_DEFAULT_GRADER_MODEL,
    grader_model_whitelist=tuple(_v5.TYPO_V5_GRADER_MODEL_WHITELIST),
    reward_variants=REWARD_VARIANTS,
    default_reward_variant="r1",
    hparams=Hparams(n_epochs=2, reasoning_effort="medium"),
)
