"""v7 grader variants -- three reward shapes for the typo column-audit task.

All three variants share the same sub-graders as v5 plus the v5 gold
(python recall) sub-grader, which v5/v6 defined but DID NOT wire into
``calculate_output``. v6's abstain-attractor is a direct consequence
of that omission: the model gets reward=5.0 (max) for outputting
``{"typos": []}`` on every column.

v7 keeps the two mechanical precision gates (trivial-FP, column-cap)
as multiplicative guards (so over-cap or trivial-FP predictions still
zero the reward), and combines the LLM precision score with the
python recall score in three different shapes.

Notation in the formulas:
  P = typo_v5_score_model_extras   (LLM-judged precision, [0,1])
  R = typo_v5_python_gold          (gold-set recall,       [0,1])
  T = typo_v5_python_trivial       (mechanical gate,       {0,1})
  C = typo_v5_python_column_cap    (mechanical gate,       {0,1})

Variants (all clamped to [0, 5] via ``range``):
  r1 (additive)        : 2.5 * T * C * (P + R)
  r2 (harmonic / F1)   : 5   * T * C * 2 * P * R / (P + R + 1e-6)
  r3 (weighted, P>R)   : 5   * T * C * (0.6 * P + 0.4 * R)

For all three, an empty prediction on a positive column scores at most
2.5 (r1), ~0 (r2), 3.0 (r3) instead of 5.0 -- the abstain attractor
is broken. An empty prediction on a clean column still scores 5.0
(R=1.0 when G=0 by construction, P=1.0 on empty per the score_model
prompt), so v7 does NOT punish the model for correctly abstaining on
clean columns.

``pass_threshold`` is set so a perfect-precision-but-zero-recall
prediction does NOT pass on r1/r2 but does on r3 (r3 is the
"precision-first" variant, kept as a fallback if r1/r2 induce too
many false positives).
"""

from __future__ import annotations

from typing import Any

from RFT.typo.v5 import _task as _v5

V7_VARIANTS: tuple[str, ...] = ("r1", "r2", "r3")


def _variant_formula(variant: str) -> tuple[str, float]:
    """Return (calculate_output expression, pass_threshold) for a variant."""
    if variant == "r1":
        return (
            "2.5 * typo_v5_python_trivial * typo_v5_python_column_cap * "
            "(typo_v5_score_model_extras + typo_v5_python_gold)",
            4.0,
        )
    if variant == "r2":
        return (
            "5 * typo_v5_python_trivial * typo_v5_python_column_cap * "
            "2 * typo_v5_score_model_extras * typo_v5_python_gold / "
            "(typo_v5_score_model_extras + typo_v5_python_gold + 1e-6)",
            3.5,
        )
    if variant == "r3":
        return (
            "5 * typo_v5_python_trivial * typo_v5_python_column_cap * "
            "(0.6 * typo_v5_score_model_extras + 0.4 * typo_v5_python_gold)",
            4.0,
        )
    raise ValueError(
        f"unknown v7 reward variant {variant!r}; expected one of "
        f"{V7_VARIANTS!r}"
    )


def build_grader_v7(
    variant: str,
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """Multigrader = (P precision + R recall + trivial + column_cap)."""
    extras = _v5.build_score_model_extras_grader(grader_model=grader_model)
    trivial = _v5.build_python_trivial_grader()
    column_cap = _v5.build_python_column_cap_grader()
    gold = _v5.build_python_grader()
    formula, pass_threshold = _variant_formula(variant)
    return {
        "type": "multi",
        "name": f"typo_v7_multigrader_{variant}",
        "graders": {
            "typo_v5_score_model_extras": extras,
            "typo_v5_python_trivial": trivial,
            "typo_v5_python_column_cap": column_cap,
            "typo_v5_python_gold": gold,
        },
        "calculate_output": formula,
        "pass_threshold": pass_threshold,
        "range": [0.0, 5.0],
    }
