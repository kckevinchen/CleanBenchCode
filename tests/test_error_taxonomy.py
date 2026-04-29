"""Tests for ``src.core.error_taxonomy.load_taxonomy``: YAML schema +
``ErrorTaxonomy`` lookup / select / render behaviour.

Pinned by Pass 6c MED gap. A typo'd taxonomy file would break every
run silently before this coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import yaml

from src.core.data_loader import COLUMN_MODE_ALL, CsvChunk
from src.core.error_taxonomy import ErrorTaxonomy, ErrorType, load_taxonomy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _good_taxonomy_dict() -> Dict[str, Any]:
    return {
        "base": {
            "system": "you are a careful auditor",
            "user_template": (
                "TYPE={type_key}\nNAME={type_name}\nCAT={type_category}\n"
                "DESC={type_description}\nEX={type_example}\nNOTES={type_notes}\n"
                "HEADERS={headers}\nROWS:\n{rows}\nRANGE={start_row}..{end_row}"
            ),
        },
        "types": {
            "typo": {
                "name": "Typo",
                "category": "Lexical",
                "description": "Misspellings",
                "example": "Smtih -> Smith",
                "notes": "be conservative",
            },
            "casing": {
                "name": "Casing",
                "category": "Lexical",
                "description": "Inconsistent capitalisation",
                # example + notes optional
            },
        },
    }


def _write_yaml(tmp_path: Path, content: Any, name: str = "tax.yaml") -> Path:
    p = tmp_path / name
    # ``sort_keys=False`` so the on-disk order matches the dict literal —
    # the taxonomy preserves insertion order, and several tests assert on it.
    p.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_taxonomy — happy path
# ---------------------------------------------------------------------------


class TestLoadHappyPath:
    def test_loads_full_taxonomy(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, _good_taxonomy_dict())
        tax = load_taxonomy(path)
        assert tax.source_path == path
        assert tax.system_prompt == "you are a careful auditor"
        assert "{type_key}" in tax.user_template
        assert tax.keys() == ["typo", "casing"]
        # Optional fields default to "" when omitted.
        casing = tax.get("casing")
        assert casing.example == ""
        assert casing.notes == ""

    def test_optional_fields_coerce_none_to_empty(self, tmp_path: Path) -> None:
        # YAML can serialise an explicit ``null`` for example/notes.
        d = _good_taxonomy_dict()
        d["types"]["typo"]["example"] = None
        d["types"]["typo"]["notes"] = None
        tax = load_taxonomy(_write_yaml(tmp_path, d))
        typo = tax.get("typo")
        assert typo.example == ""
        assert typo.notes == ""

    def test_system_prompt_is_stripped(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        d["base"]["system"] = "   spaced out\n  "
        tax = load_taxonomy(_write_yaml(tmp_path, d))
        assert tax.system_prompt == "spaced out"


# ---------------------------------------------------------------------------
# load_taxonomy — error paths
# ---------------------------------------------------------------------------


class TestLoadErrorPaths:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_taxonomy(tmp_path / "nope.yaml")

    def test_top_level_not_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- a list\n- not\n- a mapping\n")
        with pytest.raises(ValueError, match="top level must be a mapping"):
            load_taxonomy(path)

    def test_missing_base_raises(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        del d["base"]
        with pytest.raises(ValueError, match="missing required field 'base'"):
            load_taxonomy(_write_yaml(tmp_path, d))

    def test_base_not_mapping_raises(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        d["base"] = "just a string"
        with pytest.raises(ValueError, match="'base' must be a mapping"):
            load_taxonomy(_write_yaml(tmp_path, d))

    def test_missing_base_system_raises(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        del d["base"]["system"]
        with pytest.raises(ValueError, match="'system' in base"):
            load_taxonomy(_write_yaml(tmp_path, d))

    def test_missing_base_user_template_raises(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        del d["base"]["user_template"]
        with pytest.raises(ValueError, match="'user_template' in base"):
            load_taxonomy(_write_yaml(tmp_path, d))

    def test_missing_types_raises(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        del d["types"]
        with pytest.raises(ValueError, match="'types' in <root>"):
            load_taxonomy(_write_yaml(tmp_path, d))

    def test_types_not_mapping_raises(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        d["types"] = ["a list", "not a dict"]
        with pytest.raises(ValueError, match="'types' must be a mapping"):
            load_taxonomy(_write_yaml(tmp_path, d))

    def test_empty_types_raises(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        d["types"] = {}
        with pytest.raises(ValueError, match="must contain at least one entry"):
            load_taxonomy(_write_yaml(tmp_path, d))

    def test_type_body_not_mapping_raises(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        d["types"]["typo"] = "should be a dict"
        with pytest.raises(ValueError, match="types.typo must be a mapping"):
            load_taxonomy(_write_yaml(tmp_path, d))

    def test_missing_required_type_field_raises(self, tmp_path: Path) -> None:
        d = _good_taxonomy_dict()
        del d["types"]["typo"]["description"]
        with pytest.raises(ValueError, match="'description' in types.typo"):
            load_taxonomy(_write_yaml(tmp_path, d))


# ---------------------------------------------------------------------------
# ErrorTaxonomy lookup / select / render
# ---------------------------------------------------------------------------


class TestTaxonomyLookups:
    def test_get_unknown_key_raises(self, tmp_path: Path) -> None:
        tax = load_taxonomy(_write_yaml(tmp_path, _good_taxonomy_dict()))
        with pytest.raises(KeyError, match="Unknown error_type"):
            tax.get("not_real")

    def test_select_none_returns_all_in_order(self, tmp_path: Path) -> None:
        tax = load_taxonomy(_write_yaml(tmp_path, _good_taxonomy_dict()))
        sel = tax.select(None)
        assert [t.key for t in sel] == ["typo", "casing"]

    def test_select_subset_preserves_taxonomy_order(
        self, tmp_path: Path
    ) -> None:
        tax = load_taxonomy(_write_yaml(tmp_path, _good_taxonomy_dict()))
        # Caller passes [casing, typo] but taxonomy order is [typo, casing].
        sel = tax.select(["casing", "typo"])
        assert [t.key for t in sel] == ["typo", "casing"]

    def test_select_unknown_key_raises_with_known_listed(
        self, tmp_path: Path
    ) -> None:
        tax = load_taxonomy(_write_yaml(tmp_path, _good_taxonomy_dict()))
        with pytest.raises(KeyError) as ei:
            tax.select(["typo", "not_real"])
        assert "not_real" in str(ei.value)
        assert "casing" in str(ei.value)


class TestRenderUserPrompt:
    def test_substitution(self, tmp_path: Path) -> None:
        tax = load_taxonomy(_write_yaml(tmp_path, _good_taxonomy_dict()))
        et = tax.get("typo")
        df = pd.DataFrame({"name": ["alice", "bob"]})
        chunk = CsvChunk(
            source=Path("a.csv"), start_row=0, end_row=2,
            headers=["name"], frame=df, column_mode=COLUMN_MODE_ALL,
        )
        out = tax.render_user_prompt(et, chunk)
        assert "TYPE=typo" in out
        assert "NAME=Typo" in out
        assert "CAT=Lexical" in out
        assert "DESC=Misspellings" in out
        assert "EX=Smtih -> Smith" in out
        assert "NOTES=be conservative" in out
        assert "HEADERS=name" in out
        assert "RANGE=0..2" in out

    def test_optional_fields_use_placeholders(self, tmp_path: Path) -> None:
        tax = load_taxonomy(_write_yaml(tmp_path, _good_taxonomy_dict()))
        et = tax.get("casing")  # no example / notes
        df = pd.DataFrame({"name": ["alice"]})
        chunk = CsvChunk(
            source=Path("a.csv"), start_row=0, end_row=1,
            headers=["name"], frame=df, column_mode=COLUMN_MODE_ALL,
        )
        out = tax.render_user_prompt(et, chunk)
        assert "EX=(no reference example provided)" in out
        assert "NOTES=(no additional guidance)" in out

    def test_undefined_placeholder_raises_keyerror(
        self, tmp_path: Path
    ) -> None:
        d = _good_taxonomy_dict()
        d["base"]["user_template"] = "TYPE={type_key} BOGUS={not_a_real_field}"
        tax = load_taxonomy(_write_yaml(tmp_path, d))
        et = tax.get("typo")
        df = pd.DataFrame({"name": ["alice"]})
        chunk = CsvChunk(
            source=Path("a.csv"), start_row=0, end_row=1,
            headers=["name"], frame=df, column_mode=COLUMN_MODE_ALL,
        )
        with pytest.raises(KeyError, match="undefined placeholder"):
            tax.render_user_prompt(et, chunk)
