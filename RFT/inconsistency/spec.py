"""Task spec for the inconsistency v1 RFT (replaces ``scripts/inconsistency_rft.py``).

Single gold-free reward variant (one rulebook judge scoring 5*F1(precision,
recall)); whole-column trainee. The supervisor + endpoint-rotation that this
task pioneered now lives in the generic engine and applies to every task.

The eval split here carries no baked gold (the benchmark is incomplete /
gold-free), so ``scorer`` is ``None``: ``rft run`` will dump deployed-model
predictions, and scoring against the benchmark stays in ``eval/run_baseline.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from RFT._common.spec import Hparams, make_spec
from RFT.inconsistency.v1 import build_response_format
from RFT.inconsistency.v1._grader import (
    DEFAULT_GRADER_MODEL,
    _GRADER_MODEL_WHITELIST,
    build_grader,
)

_DIR = Path(__file__).resolve().parent


def _grader(variant: str, grader_model: str) -> dict[str, Any]:
    # Inconsistency has a single reward shape; the variant is ignored.
    return build_grader(grader_model=grader_model)


# gold-free eval split (no baked gold) -> findings_key only, scorer stays None;
# score against the benchmark with eval/run_baseline.py.
SPEC = make_spec(
    name="inconsistency_v1",
    error_type="inconsistency",
    version="v1",
    data_dir=_DIR / "data",
    response_format=build_response_format,
    build_grader=_grader,
    findings_key="findings",
    default_suffix="inc-v1",
    default_grader_model=DEFAULT_GRADER_MODEL,
    grader_model_whitelist=tuple(_GRADER_MODEL_WHITELIST),
    reward_variants=("default",),
    default_reward_variant="default",
    hparams=Hparams(n_epochs=2, reasoning_effort="medium"),
)
