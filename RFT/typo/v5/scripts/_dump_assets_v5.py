"""Dump v5 prompts + grader spec to disk for inspection / RFT upload.

Reads everything from ``RFT.typo.v5._task`` and writes:

* ``RFT/typo/v5/prompts/system_instruction.md`` -- assembled trainee
  developer prompt (rulebook + output-shape block).
* ``RFT/typo/v5/prompts/user_message_template.md`` -- shape of the
  per-column user message (with placeholders).
* ``RFT/typo/v5/prompts/response_format.json`` -- the strict JSON
  schema the trainee must conform to.
* ``RFT/typo/v5/prompts/labels.json`` -- the flavour enum (also in
  the schema, but a flat file is convenient for downstream tooling).
* ``RFT/typo/v5/grader/grader.json`` -- the uploadable grader spec.

Idempotent; safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v5 import _task as _typo_v5_task  # noqa: E402


PROMPTS_DIR = Path("RFT/typo/v5/prompts")
GRADER_DIR = Path("RFT/typo/v5/grader")


USER_MESSAGE_TEMPLATE = """\
Column header: {column}
Source CSV: {source}
Distinct values in this column ({n_shown} shown[truncation note]):
  1.  <count>x  '<value>'
  2.  <count>x  '<value>'
  ...

Return the JSON object listing every typo you find.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grader-model",
        default=_typo_v5_task.TYPO_V5_DEFAULT_GRADER_MODEL,
        choices=list(_typo_v5_task.TYPO_V5_GRADER_MODEL_WHITELIST),
    )
    args = parser.parse_args(argv)

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    GRADER_DIR.mkdir(parents=True, exist_ok=True)

    (PROMPTS_DIR / "system_instruction.md").write_text(
        _typo_v5_task.TYPO_V5_SYSTEM_INSTRUCTION + "\n",
        encoding="utf-8",
    )
    (PROMPTS_DIR / "user_message_template.md").write_text(
        USER_MESSAGE_TEMPLATE, encoding="utf-8",
    )
    (PROMPTS_DIR / "response_format.json").write_text(
        json.dumps(
            _typo_v5_task.build_response_format(),
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (PROMPTS_DIR / "labels.json").write_text(
        json.dumps(
            {"flavours": list(_typo_v5_task.TYPO_V5_FLAVOURS)},
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    grader = _typo_v5_task.build_grader(grader_model=args.grader_model)
    (GRADER_DIR / "grader.json").write_text(
        json.dumps(grader, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Mirror the embedded python source as a separate file for code
    # review. The shipped grader is the in-grader.json copy; this
    # mirror is purely human-readable.
    (GRADER_DIR / "grader_source.py").write_text(
        "# Python sub-grader source for v5 typo column-audit task.\n"
        "# This is a verbatim copy of "
        "``_typo_v5_task.TYPO_V5_PYTHON_GRADER_SOURCE``\n"
        "# for code review. Azure RFT executes the version embedded "
        "in grader.json\n"
        "# inside its sandbox (256 KB code, 2 GB RAM, 1 CPU, 2 min, "
        "no network).\n"
        + _typo_v5_task.TYPO_V5_PYTHON_GRADER_SOURCE,
        encoding="utf-8",
    )
    # Same idea for the trivial-FP gate sub-grader.
    (GRADER_DIR / "trivial_grader_source.py").write_text(
        "# Python trivial-FP gate sub-grader source for v5 typo task.\n"
        "# Verbatim copy of "
        "``_typo_v5_task.TYPO_V5_PYTHON_TRIVIAL_GRADER_SOURCE``\n"
        "# for code review. Azure RFT runs the in-grader.json copy.\n"
        + _typo_v5_task.TYPO_V5_PYTHON_TRIVIAL_GRADER_SOURCE,
        encoding="utf-8",
    )
    # And the column-cap (spam) gate sub-grader.
    (GRADER_DIR / "column_cap_grader_source.py").write_text(
        "# Python column-cap gate sub-grader source for v5 typo task.\n"
        "# Verbatim copy of "
        "``_typo_v5_task.TYPO_V5_PYTHON_COLUMN_CAP_GRADER_SOURCE``\n"
        "# for code review. Azure RFT runs the in-grader.json copy.\n"
        + _typo_v5_task.TYPO_V5_PYTHON_COLUMN_CAP_GRADER_SOURCE,
        encoding="utf-8",
    )

    print("Wrote:")
    for p in [
        PROMPTS_DIR / "system_instruction.md",
        PROMPTS_DIR / "user_message_template.md",
        PROMPTS_DIR / "response_format.json",
        PROMPTS_DIR / "labels.json",
        GRADER_DIR / "grader.json",
        GRADER_DIR / "grader_source.py",
        GRADER_DIR / "trivial_grader_source.py",
        GRADER_DIR / "column_cap_grader_source.py",
    ]:
        print(f"  {p}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
