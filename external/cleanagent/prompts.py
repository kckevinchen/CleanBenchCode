"""Load CleanAgent's prompt constants from the pinned upstream clone.

The wrapper intentionally does not redistribute upstream prompt text. Run
``setup.sh`` first; this module extracts literal assignments from
``CleanAgent/main.py`` without importing the interactive application.
"""

from __future__ import annotations

import ast
from pathlib import Path

_UPSTREAM_MAIN = Path(__file__).resolve().parent / "CleanAgent" / "main.py"


def _load_upstream_constants() -> tuple[str, str]:
    if not _UPSTREAM_MAIN.is_file():
        raise RuntimeError(
            "CleanAgent upstream source is missing; run "
            "`bash external/cleanagent/setup.sh` first"
        )
    tree = ast.parse(_UPSTREAM_MAIN.read_text(encoding="utf-8"))
    wanted = {"CANDIDATE_COLUMN_TYPES", "COL_ANNOTATOR_SYSTEM_MESSAGE"}
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, str):
            values[target.id] = value
    missing = wanted - values.keys()
    if missing:
        raise RuntimeError(
            f"CleanAgent upstream constants missing from {_UPSTREAM_MAIN}: "
            f"{sorted(missing)}"
        )
    return values["CANDIDATE_COLUMN_TYPES"], values["COL_ANNOTATOR_SYSTEM_MESSAGE"]


CANDIDATE_COLUMN_TYPES, COL_ANNOTATOR_SYSTEM_MESSAGE = _load_upstream_constants()

# CleanAgent's coder maps each annotated type to one dataprep.clean function
# (see the documented clean_<type> list in CleanAgent/main.py). Normalized type
# label (lowercased) -> dataprep.clean function name.
TYPE_TO_CLEANER = {
    "email": "clean_email",
    "phone": "clean_phone",
    "date": "clean_date",
    "country": "clean_country",
    "ip address": "clean_ip",
    "ip": "clean_ip",
}


def normalize_type(label: str) -> str | None:
    """Map a free-form annotation label to a known cleaner key, or None."""
    if not label:
        return None
    key = label.strip().strip("*").strip().lower()
    if key in TYPE_TO_CLEANER:
        return key
    # tolerate phrases like "date (mm/dd/yyyy)" / "country name"
    for k in TYPE_TO_CLEANER:
        if key.startswith(k):
            return k
    return None
