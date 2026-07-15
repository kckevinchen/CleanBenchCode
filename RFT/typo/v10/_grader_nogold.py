"""v10 "no-gold" grader -- precision + trivial gate ONLY.

This is a deliberate ABLATION of the v9/v10 reward: it drops both the
gold-recall leg (``typo_v5_python_gold``, R) and the column-cap gate
(``typo_v5_python_column_cap``, C). The reward keeps only:

    P = typo_v9_score_model_extras   (LLM precision under the v9 rulebook)
    T = typo_v5_python_trivial       (mechanical trivial-FP gate)

    reward = 5 * T * P            range [0, 5]   pass_threshold 4.0

KNOWN DEGENERACY (accepted by design): the precision sub-grader returns
1.0 for an empty ``typos`` array (nothing to penalise), so a model that
proposes NOTHING scores the maximum reward. With no recall leg there is
no counter-pressure, so the optimal policy under this grader is to
predict nothing. This variant exists only to measure what precision-only
shaping does; it is NOT expected to be a good production reward.

The precision sub-grader and rulebook are v9's verbatim (v10's own copy in
``RFT.typo.v10._grader_variants``).
"""

from __future__ import annotations

from typing import Any

from RFT._common import typo_task as _v5  # shared primitives (no peer-version dep)
from RFT.typo.v10._grader_variants import build_score_model_extras_grader_v10


def build_grader_v10_nogold(
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """Multigrader = (v9 P precision + v5 trivial gate). No recall, no column-cap."""
    extras = build_score_model_extras_grader_v10(grader_model=grader_model)
    trivial = _v5.build_python_trivial_grader()
    return {
        "type": "multi",
        "name": "typo_v10_multigrader_nogold",
        "graders": {
            "typo_v9_score_model_extras": extras,
            "typo_v5_python_trivial": trivial,
        },
        "calculate_output": (
            "5 * typo_v5_python_trivial * typo_v9_score_model_extras"
        ),
        "pass_threshold": 4.0,
        "range": [0.0, 5.0],
    }


# ---------------------------------------------------------------------------
# Precision + recall in a SINGLE LLM judge (no gold set, no python recall leg)
# ---------------------------------------------------------------------------
#
# The plain nogold grader above is degenerate: an empty ``typos`` array scores
# the maximum reward because the precision judge only grades values the model
# DID propose. This builder removes that degeneracy WITHOUT a gold set by
# making the same LLM judge also look at every distinct value in
# ``column_values`` and penalise genuine typos the model left out of ``typos``
# (i.e. values it implicitly labelled clean = false negatives). The judge now
# returns an F1 of precision over the proposed typos and recall over the
# rulebook-true typos it finds in the column, so "propose nothing" no longer
# scores 1.0 when real typos exist.

# Anchors patched out of the v9 precision judge's developer message.
_V9_INTRO_PRECISION_ONLY = (
    "The audit model is tuned for RECALL and "
    "deliberately over-proposes; your job is to be PRECISE -- judge "
    "each predicted typo on its own merits and mark the wrong ones, "
    "so the model learns where it over-reached."
)
_V10_INTRO_PRECISION_RECALL = (
    "Judge the model on BOTH precision and recall. For PRECISION, judge "
    "each predicted typo on its own merits and mark the wrong ones. For "
    "RECALL, you must ALSO scan every distinct value in column_values and "
    "find genuine typos (per the rulebook) that the model did NOT include "
    "in its 'typos' array -- a value the model omitted is implicitly "
    "claimed CLEAN, and omitting a real typo is a recall error that must "
    "be penalised. Proposing nothing is therefore NOT safe: it is scored "
    "as missing every true typo in the column."
)

_V9_STEP2_EMPTY_SHORTCUT = (
    "  Step 2 -- If the 'typos' array is empty, return "
    "Score: 1.0 (the model made no positive claims, so there is "
    "nothing to penalise on precision).\n"
)
_V10_STEP2_NO_SHORTCUT = (
    "  Step 2 -- Do NOT short-circuit on an empty 'typos' array. An empty "
    "array means the model claims the whole column is clean; you must "
    "still run the recall check in Step 5, which will penalise it for "
    "every genuine typo present.\n"
)

_V9_STEP4_PRECISION_ONLY = (
    "  Step 4 -- Return Score: mean(rewards) over all 'typos' "
    "entries. Clamp to [0.0, 1.0].\n\n"
)
_V10_STEP4_PRECISION_RECALL = (
    "  Step 4 -- PRECISION. Let precision = mean(rewards) over the "
    "'typos' entries from Step 3. If 'typos' is empty, set precision = "
    "1.0 (vacuous -- recall in Step 5 carries the empty case).\n"
    "  Step 5 -- RECALL. Independently scan EVERY distinct value in "
    "column_values and, using the SAME v9 rulebook scope as above, "
    "identify the set G of values that ARE genuine in-scope typos (a "
    "value is in G only if a correction to it would score 'valid' under "
    "Step 3 -- so name-specific character slips, casing-only, padding-"
    "whitespace-only, pure numbers / dates / id codes are NOT in G). For "
    "each value in G, the model CAUGHT it iff that value appears as the "
    "'value' of a 'typos' entry classified valid or ambiguous in Step 3. "
    "Let caught = number of G values caught and missed = number of G "
    "values not caught. Set recall = caught / (caught + missed). If G is "
    "empty (the column has no genuine typos), set recall = 1.0.\n"
    "  Step 6 -- Return Score: the F1 of precision and recall, "
    "2 * precision * recall / (precision + recall + 1e-6), clamped to "
    "[0.0, 1.0]. If precision and recall are both vacuous (no 'typos' "
    "entries AND G empty), return Score: 1.0.\n\n"
)


def build_score_model_pr_grader_v10(
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """v9 precision judge extended to also penalise MISSED typos (recall).

    Reuses the v9 rulebook and per-prediction precision classification
    verbatim, then adds a recall pass over all ``column_values`` so that
    omitting a genuine typo (implicitly labelling a defective cell clean)
    is penalised. Returns a single float in [0, 1] = F1(precision, recall).
    No gold set is used; the judge determines the truth set from the
    column evidence under the rulebook.
    """
    base = build_score_model_extras_grader_v10(grader_model=grader_model)
    developer = base["input"][0]["content"]

    for old, new in (
        (_V9_INTRO_PRECISION_ONLY, _V10_INTRO_PRECISION_RECALL),
        (_V9_STEP2_EMPTY_SHORTCUT, _V10_STEP2_NO_SHORTCUT),
        (_V9_STEP4_PRECISION_ONLY, _V10_STEP4_PRECISION_RECALL),
    ):
        if old not in developer:
            raise RuntimeError(
                "v9 precision judge text drifted; could not patch anchor:\n"
                f"{old!r}"
            )
        developer = developer.replace(old, new)

    base["name"] = "typo_v10_score_model_pr"
    base["input"][0]["content"] = developer
    return base


def build_grader_v10_nogold_recall(
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """Gold-free grader that keeps recall pressure INSIDE the LLM judge.

    reward = 5 * trivial_gate * PR_judge, where PR_judge is an F1 of LLM
    precision over the proposed typos and LLM recall over the genuine
    typos it finds in the column. Unlike ``build_grader_v10_nogold`` this
    is NOT degenerate: proposing nothing is scored as missing every true
    typo, so the empty-output policy collapses to a low reward.
    """
    pr = build_score_model_pr_grader_v10(grader_model=grader_model)
    trivial = _v5.build_python_trivial_grader()
    return {
        "type": "multi",
        "name": "typo_v10_multigrader_nogold_recall",
        "graders": {
            "typo_v10_score_model_pr": pr,
            "typo_v5_python_trivial": trivial,
        },
        "calculate_output": (
            "5 * typo_v5_python_trivial * typo_v10_score_model_pr"
        ),
        "pass_threshold": 4.0,
        "range": [0.0, 5.0],
    }
