"""Asset dumper for the v3 typo-detection RFT bundle.

Extracts the prompt / grader / schema assets from
``experiments/_typo_v3_task.py`` and writes them as plain text /
JSON under ``RFT/typo/v3/prompts/`` and ``RFT/typo/v3/grader/``.

Run from the repo root:

    python -m RFT.typo.v3._dump_assets

Re-run after edits to ``experiments/_typo_v3_task.py`` to refresh
the snapshot. The frozen copies under ``RFT/typo/v3/scripts/`` are
the canonical record of what was used for the v3 fine-tune.

v3 differences from v2:

* The grader is a **Python grader** (not a score-model grader), so
  there is no separate ``grader_developer.md`` or
  ``grader_user_template.md``. Instead we dump the grader source as
  ``grader/grader_source.py`` for code review.
* Labels are not enumerated -- v3 emits a set of typo objects, not
  a label. ``prompts/labels.json`` records the *flavour* enum and
  the ambiguous-credit rule instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RFT._common import _typo_v3_task as t

HERE = Path(__file__).parent


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({len(content)} bytes)")


def dump_prompts() -> None:
    _write(
        HERE / "prompts" / "rulebook.md",
        "# Typo rulebook (v3, per-column audit)\n\n"
        + t.TYPO_V3_RULEBOOK + "\n",
    )
    _write(
        HERE / "prompts" / "system_instruction.md",
        "# Trainee developer/system message (v3)\n\n"
        "Sent as the ``developer``-role turn in every training and "
        "inference call. Embeds the v3 rulebook verbatim. Unlike v2, "
        "the trainee returns a SET of typo objects per column (an "
        "empty array means the column is clean).\n\n"
        "---\n\n" + t.TYPO_V3_SYSTEM_INSTRUCTION + "\n",
    )

    example_values = [
        ("Governor", 47),
        ("Senator", 31),
        ("Mayor", 12),
        ("Govenor", 1),
    ]
    example_user = t.build_user_message(
        column="<COLUMN_HEADER>",
        source="<SOURCE_CSV>",
        values_with_counts=example_values,
        max_distinct=t.TYPO_V3_MAX_DISTINCT,
    )
    _write(
        HERE / "prompts" / "user_message_template.md",
        "# Trainee user message template (v3)\n\n"
        "Built by ``experiments._typo_v3_task.build_user_message``. "
        "The listing covers EVERY distinct non-null value in the "
        f"column, ordered count-descending, capped at "
        f"{t.TYPO_V3_MAX_DISTINCT}.\n\n"
        "---\n\n"
        "```\n" + example_user + "\n```\n",
    )

    _write(
        HERE / "prompts" / "response_format.json",
        json.dumps(t.build_response_format(), indent=2) + "\n",
    )

    _write(
        HERE / "prompts" / "labels.json",
        json.dumps(
            {
                "flavours": list(t.TYPO_V3_FLAVOURS),
                "output_shape": (
                    "JSON object {typos: [{value, correction, flavour, "
                    "rationale}, ...]}; empty list means 'no typos'."
                ),
                "ambiguous_scoring_rule": (
                    "v3 carries an ``ambiguous_values`` list per "
                    "column (the union of v2 cell-level ambiguous "
                    "gold rows). Predictions on ambiguous values "
                    "are removed from BOTH the predicted and gold "
                    "sets before scoring -- credited neither true "
                    "positive nor false positive."
                ),
                "correction_mismatch_rule": (
                    "A predicted (value, correction) pair where "
                    "``value`` matches gold but ``correction`` does "
                    "not is counted as BOTH a false positive AND a "
                    "false negative."
                ),
                "normalisation": (
                    "Match key uses case-folding + collapsed "
                    "whitespace; otherwise verbatim."
                ),
            },
            indent=2,
        )
        + "\n",
    )


def dump_grader() -> None:
    grader = t.build_grader()
    _write(
        HERE / "grader" / "grader.json",
        json.dumps(grader, indent=2) + "\n",
    )
    _write(
        HERE / "grader" / "grader_source.py",
        "# Python grader source for v3 typo column-audit task.\n"
        "# This file is a snapshot of "
        "``experiments._typo_v3_task.GRADER_SOURCE`` for code review.\n"
        "# Azure RFT executes this code in a sandboxed Python\n"
        "# environment (256 KB code limit, 2 GB memory, no network).\n"
        "# The grader must never raise; it always returns a number\n"
        "# in [0.0, 5.0]. See scoring_rubric.md for the rubric.\n\n"
        + t.GRADER_SOURCE.lstrip(),
    )
    _write(
        HERE / "grader" / "scoring_rubric.md",
        "# Grader scoring rubric v3 (set-F1; Python grader)\n\n"
        "Per-column score, computed deterministically by the Python "
        "grader in ``grader_source.py``:\n\n"
        "```\n"
        "1. Decode item.gold_typos and item.ambiguous_values from\n"
        "   their JSON-encoded strings.\n"
        "2. Decode sample.output_text as JSON; take ``typos`` list\n"
        "   (any parse failure -> 0.0).\n"
        "3. Remove entries whose normalised value is in\n"
        "   ambiguous_values from BOTH pred and gold (ambiguous-\n"
        "   either-correct rule).\n"
        "4. Build maps {norm(value) -> norm(correction)} for pred\n"
        "   and gold. Normalisation: case-fold + collapse whitespace.\n"
        "5. TP: keys in both maps with matching corrections.\n"
        "   FP: pred-only keys + correction-mismatch keys.\n"
        "   FN: gold-only keys + correction-mismatch keys.\n"
        "   (Correction-mismatch counts as both FP and FN.)\n"
        "6. Both empty -> 5.0 (column is genuinely clean, full credit).\n"
        "   TP == 0 -> 0.0.\n"
        "   Otherwise -> F1 * 5.0, clamped to [0.0, 5.0].\n"
        "```\n\n"
        "There is no ``pass_threshold`` / ``range`` field for a "
        "Python grader -- only ``type``, ``name``, ``source`` are "
        "accepted by the Azure RFT GA API.\n\n"
        "The local helper ``experiments._typo_v3_task.compute_column_f1`` "
        "mirrors this logic for offline eval scripts (used by "
        "``experiments.eval_typo_finetune_v3``).\n",
    )


def main() -> None:
    dump_prompts()
    dump_grader()


if __name__ == "__main__":
    main()
