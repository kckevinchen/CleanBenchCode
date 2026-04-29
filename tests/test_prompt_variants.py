"""Tests for ``src.adaptive.prompt_variants.PromptVariantGenerator``.

Pinned by Pass 6c HIGH coverage gap. The Good-Turing experiment relies
on the variant generator producing per-pass rewrites that are:

* parsed safely from JSON (with or without ``json`` fences)
* fall back to seed definitions on LLM error or parse failure (rather
  than crashing the pass)
* produce a new ErrorTaxonomy with rewrites substituted in but NON-
  selected types passed through unchanged
* emit one record per seed type, in seed order, regardless of the
  thread-pool fan-out
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest
import yaml

from src.adaptive.prompt_variants import (
    PromptVariantGenerator,
    _strip_json_fences,
    save_variant_records,
)
from src.core.error_taxonomy import ErrorTaxonomy, ErrorType
from src.core.llm_client import LLMResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed(key: str = "typo", **overrides) -> ErrorType:
    base = {
        "key": key, "name": key.title(), "category": "Cat",
        "description": f"seed-desc-{key}", "example": f"seed-ex-{key}",
        "notes": f"seed-notes-{key}",
    }
    base.update(overrides)
    return ErrorType(**base)


def _taxonomy(types: List[ErrorType]) -> ErrorTaxonomy:
    return ErrorTaxonomy(
        system_prompt="SYS",
        user_template="TYPE={type_key} {type_description}",
        types=tuple(types),
    )


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    p = tmp_path / "variants.yaml"
    p.write_text(yaml.safe_dump({
        "base": {
            "system": "rewrite the error type",
            "user_template": (
                "key={type_key} name={type_name} cat={type_category}\n"
                "desc={description}\nnotes={notes}\nex={example}\n"
                "pass={pass_index}"
            ),
        }
    }))
    return p


class _StubLLM:
    """Records calls and returns scripted ``LLMResponse`` values."""

    def __init__(self, responses: List[LLMResponse]) -> None:
        self.endpoints = ["http://stub"]
        self._responses = list(responses)
        self.calls: List[dict] = []

    def query(self, model, prompt, *, reasoning_effort, reasoning_summary,
              system_prompt):
        self.calls.append({
            "model": model, "prompt": prompt,
            "system_prompt": system_prompt,
        })
        if not self._responses:
            return LLMResponse(text="", endpoint="http://stub", model=model,
                                attempts=1, error="no scripted response")
        return self._responses.pop(0)


def _ok_response(desc: str, notes: str = "", example: str = "") -> LLMResponse:
    body = json.dumps({"description": desc, "notes": notes, "example": example})
    return LLMResponse(text=body, endpoint="http://stub", model="m", attempts=1)


# ---------------------------------------------------------------------------
# _strip_json_fences
# ---------------------------------------------------------------------------


class TestStripFences:
    def test_no_fence_passthrough(self) -> None:
        assert _strip_json_fences('{"a":1}') == '{"a":1}'

    def test_json_fence_stripped(self) -> None:
        assert _strip_json_fences('```json\n{"a":1}\n```') == '{"a":1}'

    def test_bare_fence_stripped(self) -> None:
        assert _strip_json_fences('```\n{"a":1}\n```') == '{"a":1}'

    def test_uppercase_json_tag_stripped(self) -> None:
        assert _strip_json_fences('```JSON\n{"a":1}\n```') == '{"a":1}'

    def test_fence_with_extra_whitespace(self) -> None:
        assert _strip_json_fences('  ```json\n{"a":1}\n```  ') == '{"a":1}'


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


class TestTemplateLoading:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PromptVariantGenerator(
                llm=_StubLLM([]), model="m",
                template_path=tmp_path / "nope.yaml",
            )

    def test_missing_base_block_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump({"foo": "bar"}))
        with pytest.raises(ValueError, match="missing `base:`"):
            PromptVariantGenerator(
                llm=_StubLLM([]), model="m", template_path=p,
            )

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump({"base": {"system": "s"}}))
        with pytest.raises(ValueError, match="missing required field"):
            PromptVariantGenerator(
                llm=_StubLLM([]), model="m", template_path=p,
            )


# ---------------------------------------------------------------------------
# generate_for_type
# ---------------------------------------------------------------------------


class TestGenerateForType:
    def test_success_path_returns_variant(self, template_path: Path) -> None:
        llm = _StubLLM([_ok_response("new desc", notes="new notes",
                                       example="new ex")])
        g = PromptVariantGenerator(
            llm=llm, model="m", template_path=template_path,
        )
        rec = g.generate_for_type(_seed("typo"), pass_index=2)
        assert rec["ok"] is True
        assert rec["error"] is None
        assert rec["pass_index"] == 2
        assert rec["type_key"] == "typo"
        assert rec["seed_description"] == "seed-desc-typo"
        assert rec["variant_description"] == "new desc"
        assert rec["variant_notes"] == "new notes"
        assert rec["variant_example"] == "new ex"
        assert rec["variant"].description == "new desc"
        # Prompt rendering received the seed text.
        assert "seed-desc-typo" in llm.calls[0]["prompt"]
        assert llm.calls[0]["system_prompt"] == "rewrite the error type"

    def test_fenced_response_parses(self, template_path: Path) -> None:
        body = '```json\n' + json.dumps({"description": "fenced"}) + '\n```'
        llm = _StubLLM([LLMResponse(text=body, endpoint="http://stub",
                                      model="m", attempts=1)])
        g = PromptVariantGenerator(
            llm=llm, model="m", template_path=template_path,
        )
        rec = g.generate_for_type(_seed("typo"), pass_index=1)
        assert rec["ok"] is True
        assert rec["variant_description"] == "fenced"

    def test_empty_description_falls_back_to_seed(
        self, template_path: Path
    ) -> None:
        body = json.dumps({"description": "", "notes": "x"})
        llm = _StubLLM([LLMResponse(text=body, endpoint="http://stub",
                                      model="m", attempts=1)])
        g = PromptVariantGenerator(
            llm=llm, model="m", template_path=template_path,
        )
        rec = g.generate_for_type(_seed("typo"), pass_index=1)
        assert rec["ok"] is False
        assert rec["error"] and "parse" in rec["error"]
        # Variant === seed on fallback.
        assert rec["variant"].description == "seed-desc-typo"

    def test_invalid_json_falls_back_to_seed(self, template_path: Path) -> None:
        llm = _StubLLM([LLMResponse(text="not json", endpoint="http://stub",
                                      model="m", attempts=1)])
        g = PromptVariantGenerator(
            llm=llm, model="m", template_path=template_path,
        )
        rec = g.generate_for_type(_seed("typo"), pass_index=1)
        assert rec["ok"] is False
        assert rec["variant"].description == "seed-desc-typo"

    def test_llm_error_falls_back_to_seed(self, template_path: Path) -> None:
        llm = _StubLLM([LLMResponse(text="", endpoint="http://stub",
                                      model="m", attempts=3, error="HTTP 500")])
        g = PromptVariantGenerator(
            llm=llm, model="m", template_path=template_path,
        )
        rec = g.generate_for_type(_seed("typo"), pass_index=1)
        assert rec["ok"] is False
        assert rec["error"] == "HTTP 500"
        assert rec["variant"].description == "seed-desc-typo"

    def test_top_level_array_falls_back(self, template_path: Path) -> None:
        # Top-level non-object → parse error path.
        llm = _StubLLM([LLMResponse(text='["nope"]', endpoint="http://stub",
                                      model="m", attempts=1)])
        g = PromptVariantGenerator(
            llm=llm, model="m", template_path=template_path,
        )
        rec = g.generate_for_type(_seed("typo"), pass_index=1)
        assert rec["ok"] is False
        assert rec["variant"].description == "seed-desc-typo"


# ---------------------------------------------------------------------------
# generate_for_taxonomy
# ---------------------------------------------------------------------------


class TestGenerateForTaxonomy:
    def test_substitutes_only_selected_types(self, template_path: Path) -> None:
        # Three types in the taxonomy; only two of them get rewritten.
        seed_a = _seed("a")
        seed_b = _seed("b")
        seed_c = _seed("c")
        tax = _taxonomy([seed_a, seed_b, seed_c])
        # 2 scripted responses (one per selected type, order doesn't
        # matter — generator dispatches in parallel but we have only
        # one stub queue).
        llm = _StubLLM([
            _ok_response("rewritten-a"),
            _ok_response("rewritten-b"),
        ])
        g = PromptVariantGenerator(
            llm=llm, model="m", template_path=template_path,
            concurrency=1,  # serial → response order matches seed order
        )
        new_tax, variant_types, records = g.generate_for_taxonomy(
            tax, [seed_a, seed_b], pass_index=1,
        )
        # Variant types preserve seed order.
        assert [t.key for t in variant_types] == ["a", "b"]
        assert variant_types[0].description == "rewritten-a"
        assert variant_types[1].description == "rewritten-b"
        # New taxonomy: a and b rewritten, c unchanged, all in the
        # original taxonomy order.
        assert [t.key for t in new_tax.types] == ["a", "b", "c"]
        assert new_tax.get("a").description == "rewritten-a"
        assert new_tax.get("c").description == "seed-desc-c"
        # Same shared system + user_template.
        assert new_tax.system_prompt == tax.system_prompt
        assert new_tax.user_template == tax.user_template
        # One record per selected type.
        assert len(records) == 2
        assert {r["type_key"] for r in records} == {"a", "b"}

    def test_empty_seed_types_returns_unchanged_taxonomy(
        self, template_path: Path
    ) -> None:
        tax = _taxonomy([_seed("a")])
        g = PromptVariantGenerator(
            llm=_StubLLM([]), model="m", template_path=template_path,
        )
        new_tax, variant_types, records = g.generate_for_taxonomy(
            tax, [], pass_index=1,
        )
        assert variant_types == []
        assert records == []
        assert [t.key for t in new_tax.types] == ["a"]
        assert new_tax.get("a").description == "seed-desc-a"

    def test_failure_still_yields_record_for_every_seed(
        self, template_path: Path
    ) -> None:
        tax = _taxonomy([_seed("a"), _seed("b")])
        # First succeeds, second errors out — but BOTH still appear in records.
        llm = _StubLLM([
            _ok_response("rewritten-a"),
            LLMResponse(text="", endpoint="http://stub", model="m",
                        attempts=3, error="HTTP 500"),
        ])
        g = PromptVariantGenerator(
            llm=llm, model="m", template_path=template_path, concurrency=1,
        )
        new_tax, variant_types, records = g.generate_for_taxonomy(
            tax, [_seed("a"), _seed("b")], pass_index=1,
        )
        assert len(records) == 2
        # b fell back to seed.
        assert new_tax.get("b").description == "seed-desc-b"
        assert new_tax.get("a").description == "rewritten-a"


# ---------------------------------------------------------------------------
# save_variant_records
# ---------------------------------------------------------------------------


class TestSaveVariantRecords:
    def test_writes_one_file_per_record(
        self, tmp_path: Path, template_path: Path
    ) -> None:
        seed_a = _seed("a")
        tax = _taxonomy([seed_a])
        llm = _StubLLM([_ok_response("rewritten-a", notes="n", example="ex")])
        g = PromptVariantGenerator(
            llm=llm, model="m", template_path=template_path, concurrency=1,
        )
        _, _, records = g.generate_for_taxonomy(tax, [seed_a], pass_index=1)
        out = tmp_path / "pass1"
        save_variant_records(records, run_subdir=out)
        path = out / "a.json"
        assert path.is_file()
        data = json.loads(path.read_text())
        # ``variant`` is serialised as a dict (not the dataclass).
        assert data["variant"]["key"] == "a"
        assert data["variant"]["description"] == "rewritten-a"
        assert data["type_key"] == "a"
