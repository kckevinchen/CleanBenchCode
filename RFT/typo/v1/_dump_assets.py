"""One-shot helper that extracts the prompt / grader / schema assets
from ``experiments/_typo_task.py`` and writes them as plain
text / JSON next to this script.

Run from the repo root:

    python -m RFT.typo.v1._dump_assets

This is intentionally not wired into any CLI -- it is a frozen snapshot
of the typo RFT v1 experiment. Re-run after edits to
``experiments/_typo_task.py`` if you want to refresh the snapshot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running both as a module and as a plain script.
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
           "# Typo rulebook (v1)\n\n" + t.TYPO_RULEBOOK + "\n")
    _write(HERE / "prompts" / "system_instruction.md",
           "# Trainee developer/system message\n\n"
           "Sent as the `developer`-role turn in every training and "
           "inference call. Embeds the rulebook verbatim so the trainee "
           "has the policy in-context.\n\n"
           "---\n\n" + t.TYPO_SYSTEM_INSTRUCTION + "\n")

    # User-message template -- show the format with placeholders so a
    # reader can see the structure without running Python.
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
           f"{20} distinct values from the same column (capped to keep "
           "the prompt bounded for wide columns).\n\n"
           "---\n\n"
           "```\n" + example_user + "\n```\n")

    _write(HERE / "prompts" / "response_format.json",
           json.dumps(t.build_response_format(), indent=2) + "\n")

    _write(HERE / "prompts" / "labels.json",
           json.dumps({
               "labels": list(t.TYPO_LABELS),
               "flavours": list(t.TYPO_FLAVOURS),
           }, indent=2) + "\n")


def dump_grader() -> None:
    grader = t.build_grader("<GRADER_MODEL_NAME>")
    _write(HERE / "grader" / "grader.json",
           json.dumps(grader, indent=2) + "\n")
    _write(HERE / "grader" / "grader_developer.md",
           "# Grader developer/system message\n\n"
           "The grader is a ``score_model`` (per Azure OpenAI RFT). "
           "It sees the gold row via ``item.*`` template substitution "
           "and the trainee's reply via ``sample.output_text``, then "
           "emits a 0-5 score.\n\n"
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
           "# Grader scoring rubric (verbatim from the developer message)\n\n"
           "```\n"
           "5: model.label == gold_label AND, when gold_label=='typo',\n"
           "   model.correction equals gold_correction up to case and\n"
           "   collapsed whitespace.\n"
           "4: model.label == gold_label but, when gold_label=='typo',\n"
           "   the correction is missing, partially wrong, or off by one\n"
           "   character.\n"
           "3: model.label is 'ambiguous' when gold is 'typo' or 'clean'\n"
           "   (or vice versa) AND the rationale is consistent with the\n"
           "   rulebook.\n"
           "2: model.label is the opposite of gold_label\n"
           "   ('typo' vs 'clean').\n"
           "1: JSON parses but required fields are missing or malformed.\n"
           "0: output is empty or not valid JSON.\n"
           "```\n\n"
           f"`pass_threshold` = **{grader['pass_threshold']}**, "
           f"`range` = **{grader['range']}**.\n")


def main() -> None:
    dump_prompts()
    dump_grader()


if __name__ == "__main__":
    main()
