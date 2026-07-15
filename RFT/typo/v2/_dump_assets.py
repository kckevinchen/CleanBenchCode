"""One-shot helper that extracts the prompt / grader / schema assets
from ``experiments/_typo_task.py`` and writes them as plain
text / JSON next to this script (v2 binary task).

Run from the repo root:

    python -m RFT.typo.v2._dump_assets

Re-run after edits to ``experiments/_typo_task.py`` if you want to
refresh the snapshot. The frozen copies under
``RFT/typo/v2/scripts/`` are the canonical record of what was used
for the v2 fine-tune.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RFT._common import _typo_task as t

HERE = Path(__file__).parent


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({len(content)} bytes)")


def dump_prompts() -> None:
    _write(HERE / "prompts" / "rulebook.md",
           "# Typo rulebook (v2, binary)\n\n" + t.TYPO_RULEBOOK + "\n")
    _write(HERE / "prompts" / "system_instruction.md",
           "# Trainee developer/system message (v2)\n\n"
           "Sent as the `developer`-role turn in every training and "
           "inference call. Embeds the rulebook verbatim so the trainee "
           "has the policy in-context. v2 instructs the trainee to "
           "commit to a binary label (`typo` | `clean`); `ambiguous` "
           "is not a valid output.\n\n"
           "---\n\n" + t.TYPO_SYSTEM_INSTRUCTION + "\n")

    example_user = t.build_user_message(
        column="<COLUMN_HEADER>",
        source="<SOURCE_CSV>",
        candidate_value="<CANDIDATE_VALUE>",
        sibling_values=[f"<SIBLING_{i+1}>" for i in range(3)],
        max_siblings=20,
    )
    _write(HERE / "prompts" / "user_message_template.md",
           "# Trainee user message template\n\n"
           "Built by ``experiments._typo_task.build_user_message``. "
           "The sibling list is a sample of up to "
           f"{20} distinct values from the same column.\n\n"
           "---\n\n"
           "```\n" + example_user + "\n```\n")

    _write(HERE / "prompts" / "response_format.json",
           json.dumps(t.build_response_format(), indent=2) + "\n")

    _write(HERE / "prompts" / "labels.json",
           json.dumps({
               "gold_labels": list(t.TYPO_LABELS),
               "output_labels": list(t.TYPO_OUTPUT_LABELS),
               "flavours": list(t.TYPO_FLAVOURS),
               "ambiguous_scoring_rule": (
                   "Eval rows with gold_label='ambiguous' are credited "
                   "as correct regardless of whether the model emits "
                   "'typo' or 'clean'."
               ),
           }, indent=2) + "\n")


def dump_grader() -> None:
    grader = t.build_grader("<GRADER_MODEL_NAME>")
    _write(HERE / "grader" / "grader.json",
           json.dumps(grader, indent=2) + "\n")
    _write(HERE / "grader" / "grader_developer.md",
           "# Grader developer/system message (v2)\n\n"
           "The grader is a ``score_model`` (per Azure OpenAI RFT). "
           "It sees the gold row via ``item.*`` template substitution "
           "and the trainee's reply via ``sample.output_text``, then "
           "emits a 0-5 score. v2 grades a binary model -- the trainee "
           "must commit to `typo` or `clean`. On `ambiguous` gold "
           "rows, EITHER answer scores 5.\n\n"
           "---\n\n" + grader["input"][0]["content"] + "\n")
    _write(HERE / "grader" / "grader_user_template.md",
           "# Grader user template\n\n"
           "Template substitution happens server-side: ``{{item.x}}`` "
           "and ``{{sample.output_text}}`` are textually replaced. "
           "All ``item.*`` values are JSON-encoded strings so the "
           "resulting message is valid JSON.\n\n"
           "---\n\n"
           "```\n" + grader["input"][1]["content"] + "\n```\n")
    _write(HERE / "grader" / "scoring_rubric.md",
           "# Grader scoring rubric v2 (binary; ambiguous either-way)\n\n"
           "```\n"
           "5: gold_label=='ambiguous' (either binary label accepted);\n"
           "   OR gold_label=='clean'  and model.label=='clean';\n"
           "   OR gold_label=='typo'   and model.label=='typo' AND\n"
           "      model.correction equals gold_correction up to case and\n"
           "      collapsed whitespace.\n"
           "4: gold_label=='typo' and model.label=='typo' but the\n"
           "   correction is missing, partially wrong, or off by one\n"
           "   character.\n"
           "2: gold_label and model.label are opposites\n"
           "   ('typo' vs 'clean').\n"
           "1: JSON parses but required fields are missing, malformed,\n"
           "   OR model.label is 'ambiguous' (schema violation).\n"
           "0: output is empty or not valid JSON.\n"
           "```\n\n"
           f"`pass_threshold` = **{grader['pass_threshold']}**, "
           f"`range` = **{grader['range']}**.\n")


def main() -> None:
    dump_prompts()
    dump_grader()


if __name__ == "__main__":
    main()
