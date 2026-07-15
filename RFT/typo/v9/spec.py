"""Task spec for the typo v9 RFT (replaces ``scripts/typo_rft_v9.py``).

v9 = v8 trainee prompt with proper-nouns out of scope + the forked precision
rulebook. The lifecycle is entirely generic (``RFT.cli`` / ``RFT._common.engine``);
only the grader (``build_grader_v9``), the trainee schema, and the v9 data splits
are version-specific. Shared primitives come from ``RFT._common.typo_task`` --
this version imports no peer version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from RFT._common import typo_task as _v5  # shared primitives (no peer-version dep)
from RFT._common.spec import Hparams, make_spec
from RFT.typo.v9 import V9_VARIANTS, build_grader_v9

_DIR = Path(__file__).resolve().parent


def _grader(variant: str, grader_model: str) -> dict[str, Any]:
    return build_grader_v9(variant, grader_model=grader_model)


SPEC = make_spec(
    name="typo_v9",
    error_type="typo",
    version="v9",
    data_dir=_DIR / "data",
    response_format=_v5.build_response_format,
    build_grader=_grader,
    findings_key="typos",
    gold_field="gold_typos",
    ambiguous_field="ambiguous_values",
    default_suffix="typo-rft-v9",
    default_grader_model=_v5.TYPO_V5_DEFAULT_GRADER_MODEL,
    grader_model_whitelist=tuple(_v5.TYPO_V5_GRADER_MODEL_WHITELIST),
    reward_variants=tuple(V9_VARIANTS),
    default_reward_variant="r1",
    hparams=Hparams(n_epochs=2, reasoning_effort="medium"),
)
