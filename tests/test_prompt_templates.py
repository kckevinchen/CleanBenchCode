"""Smoke tests for every YAML prompt template under ``prompts/``.

These guard against drift between the placeholders a template expects
(e.g. ``{table_overview}``) and the kwargs the corresponding renderer
passes in. A missing escape on a literal ``{...}`` in a JSON example
is the most common failure mode -- ``str.format`` raises ``KeyError``
on the first such placeholder, which would crash the very first
production call. Catching it in CI is cheap.

The test does not validate correctness of the prompt body -- only that
``str.format(**dummy)`` returns without raising.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Dict, Iterable

import pytest
import yaml

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


# Per-prompt-file kwargs covering every placeholder the matching
# renderer fills. Add new placeholders here whenever a template grows
# one (the test will fail with a KeyError pointing at the missing
# name, which is the desired alarm).
PROMPT_FIXTURES: Dict[str, Dict[str, str]] = {
    "error_types.yaml": {
        "type_key": "typo",
        "type_name": "Typo",
        "type_category": "Single-Column / Typo",
        "type_description": "desc",
        "type_example": "ex",
        "type_notes": "notes",
        "table_overview": "(none)",
        "headers": "a, b",
        "rows": "(table)",
        "start_row": "0",
        "end_row": "1",
    },
    "synthesise_cluster.yaml": {
        "cluster_id": "x_y_z",
        "row": "0",
        "detection_count": "2",
        "available_keys": "typo, inconsistency",
        "row_values": "  - a: 1",
        "detections": "  [1] ...",
        "taxonomy_refs": "  * typo: ...",
    },
    "discover_relations.yaml": {
        "type_key": "fd_violation",
        "type_name": "FD Violation",
        "type_category": "Multi-Column",
        "type_description": "desc",
        "type_example": "ex",
        "type_notes": "notes",
        "table_overview": "(none)",
        "row_sample": "(none)",
        "priors_section": "(none)",
        "error_type_kinds": "  - fd",
    },
    "verify_entity.yaml": {
        "source": "a.csv",
        "pattern": "single_cell",
        "primary_row": "0",
        "row": "0",
        "rows": "0",
        "primary_col": "name",
        "column": "name",
        "columns": "name",
        "n_records": "1",
        "entity_kind": "org",
        "error_types": "typo",
        "suspect_value": "Acme Corp",
        "suggested_fix": "(none)",
        "sibling_values": "  - Acme Inc",
        "records_block": "  Record 1: ...",
        "task_block": "Use the Web Search tool ...",
    },
    "canonicalize_values.yaml": {
        "source": "a.csv",
        "column": "Country",
        "value_list": "  - 'US'\n  - 'USA'",
    },
    "validate_relation.yaml": {
        "kind": "fd",
        "columns": '["a", "b"]',
        "lhs": '["a"]',
        "rhs": '["b"]',
        "tables": "[]",
        "predicate": "a -> b",
        "executable": "(none)",
        "rationale": "",
        "source": "a.csv",
        "error_type": "fd_violation",
        "headers": '["a", "b"]',
        "schema": "  - a: object",
        "column_samples": "  - a: ['1']",
        "sample_csv": "a,b\n1,2",
        "n_rows": "1",
        "table_metadata": "total rows: 1",
    },
    "detect_relation_violation.yaml": {
        "source": "a.csv",
        "error_type": "fd_violation",
        "kind": "fd",
        "columns": "a, b",
        "predicate": "a -> b",
        "rationale": "",
        "confidence": "medium",
        "refusal_reason": "external knowledge required",
        "table_md": "| __row__ | a | b |\n|---|---|---|\n| 0 | 1 | 2 |",
        "n_rows": "1",
    },
    "aggregate_relations.yaml": {
        "source": "a.csv",
        "error_type": "fd_violation",
        "cluster_size": "2",
        "columns": "  - a",
        "cluster_rules": "  1. ...",
        "sample_rows": "(none)",
    },
    "generalize_relations.yaml": {
        "source": "a.csv",
        "error_type": "fd_violation",
        "cluster_size": "2",
        "columns": "  - a",
        "cluster_rules": "  1. ...",
        "sample_rows": "(none)",
    },
    "generate_prompt_variant.yaml": {
        "type_key": "typo",
        "type_name": "Typo",
        "type_category": "Single-Column / Typo",
        "description": "desc",
        "example": "ex",
        "notes": "notes",
        "pass_index": "1",
    },
    "confirm_union_join.yaml": {
        "workbook": "a.xlsx",
        "sheet_headers_block": "  - s1: a, b",
        "joinable_pairs_block": "  (none)",
        "row_samples_block": "  (none)",
        "unionable_groups_block": "  (none)",
    },
    "verify_multi_table_violations.yaml": {
        "workbook": "a.xlsx",
        "group_id": "u1",
        "group_kind": "union",
        "sheets": "['a.csv', 'b.csv']",
        "primary_key": "['K']",
        "column_mapping_block": "  - 'K': a.csv:'K', b.csv:'K'",
        "violations_block": "  [v1] attribute_conflict columns=['V']",
    },
    "detect_multi_table_inconsistency.yaml": {
        "workbook": "a.xlsx",
        "group_id": "u1",
        "group_kind": "union",
        "sheets": "['a.csv', 'b.csv']",
        "primary_key": "['K']",
        "column_mapping_block": "  - 'K': a.csv:'K', b.csv:'K'",
        "row_samples_block": "  === 'a.csv' (1 row) ===\n  row_idx | K\n  0 | 1",
        "page_index": 1,
        "n_pages": 1,
        "n_sample_rows": 50,
    },
}


def _iter_prompt_files() -> Iterable[Path]:
    return sorted(PROMPT_DIR.glob("*.yaml"))


@pytest.mark.parametrize(
    "path",
    list(_iter_prompt_files()),
    ids=lambda p: p.name,
)
def test_prompt_template_formats_with_fixture(path: Path) -> None:
    """Every YAML prompt's ``user_template`` (or ``base.user_template``)
    must format successfully against the test fixture for that file.

    The kwargs in :data:`PROMPT_FIXTURES` are intentionally a SUPER-set
    of what each renderer actually passes -- ``str.format`` ignores
    extra kwargs, only missing ones raise. So adding a new prompt only
    requires adding its placeholders to the fixture; trimming an
    existing placeholder is automatically forgiven.
    """
    fixture = PROMPT_FIXTURES.get(path.name)
    if fixture is None:
        pytest.skip(f"no fixture registered for {path.name}; add one to PROMPT_FIXTURES")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(raw, dict), f"{path.name}: top level must be a mapping"

    # A few prompts use a ``base:`` block for the system / user pair;
    # the rest expose them at the top level. Both shapes need to format.
    templates_to_check: list[str] = []
    if "base" in raw and isinstance(raw["base"], dict):
        ut = raw["base"].get("user_template")
        if isinstance(ut, str) and ut:
            templates_to_check.append(ut)
    ut = raw.get("user_template")
    if isinstance(ut, str) and ut:
        templates_to_check.append(ut)

    assert templates_to_check, f"{path.name}: no user_template found"

    # Some prompts use ``${name}`` substitutions for shared numeric
    # thresholds (string.Template); apply the ``defaults:`` block first
    # if present, mirroring what the production loader does.
    defaults = raw.get("defaults") or {}
    if isinstance(defaults, dict) and defaults:
        defaults_str = {str(k): str(v) for k, v in defaults.items()}
        templates_to_check = [
            Template(t).safe_substitute(defaults_str) for t in templates_to_check
        ]

    for template in templates_to_check:
        # ``.format`` raises KeyError on missing placeholders and
        # ValueError on syntax errors (e.g. an unescaped literal
        # ``{`` in a JSON example). Both are exactly the regressions
        # we want this test to catch.
        try:
            template.format(**fixture)
        except (KeyError, ValueError, IndexError) as exc:
            pytest.fail(
                f"{path.name}: template.format(...) raised {type(exc).__name__}: "
                f"{exc}. If you added a new placeholder, register it in "
                f"tests/test_prompt_templates.py::PROMPT_FIXTURES."
            )


def test_every_prompt_has_a_fixture() -> None:
    """Catch new prompt files added without a fixture entry."""
    on_disk = {p.name for p in _iter_prompt_files()}
    registered = set(PROMPT_FIXTURES.keys())
    missing = on_disk - registered
    assert not missing, (
        f"prompt(s) without a fixture in PROMPT_FIXTURES: {sorted(missing)}. "
        "Register one so the format() smoke test covers it."
    )
