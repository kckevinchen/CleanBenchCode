"""Tests for the column-clustering prompt-ensemble generator.

The detection-side ``PromptVariantGenerator`` is tightly bound to
``ErrorType``; this generator's job is the column-clustering variant
of the same idea: paraphrase the flat
``prompts/cluster_column_values.yaml`` proposer prompt (system +
user_template) while preserving the ``{source}`` / ``{column}`` /
``{value_list}`` placeholders. These tests pin the contract end-to-end
without touching the network -- a fake LLM client supplies hand-built
responses for the rewriter prompt.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive.prompt_variants import (  # noqa: E402
    ClusterPromptVariant,
    ClusterPromptVariantGenerator,
    _placeholders_preserved,
    _template_format_safe,
    save_cluster_variant_records,
)


GENERATOR_TEMPLATE = (
    PROJECT_ROOT / "prompts" / "generate_cluster_prompt_variant.yaml"
)


# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    text: str
    reasoning_summary: str = ""
    error: Optional[str] = None


class _ScriptedLLM:
    """Returns scripted responses keyed by call order. The generator
    spins up a thread pool, so we don't bother to validate per-prompt
    matching here -- the order of responses just needs to be stable
    enough to make assertions about counts and content."""

    def __init__(self, responses: List[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def query(self, *args, **kwargs) -> _FakeResponse:
        self.calls.append(dict(kwargs))
        if not self._responses:
            raise AssertionError("scripted LLM out of responses")
        return self._responses.pop(0)


def _ok_payload(*, pass_index: int) -> str:
    payload = {
        "system": (
            f"Paraphrased system #{pass_index}: identify standardization "
            "clusters in JSON."
        ),
        "user_template": (
            f"VARIANT #{pass_index}\n"
            "source: {source}\ncolumn: {column}\nvalues:\n{value_list}"
        ),
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# _placeholders_preserved
# ---------------------------------------------------------------------------


class TestPlaceholdersPreserved:
    def test_all_three_present_once(self) -> None:
        text = "src={source} col={column} vals:\n{value_list}"
        assert _placeholders_preserved(text) is True

    def test_missing_one(self) -> None:
        text = "src={source} col={column}"
        assert _placeholders_preserved(text) is False

    def test_duplicate_one(self) -> None:
        text = "{source} {source} {column} {value_list}"
        assert _placeholders_preserved(text) is False

    def test_empty(self) -> None:
        assert _placeholders_preserved("") is False


# ---------------------------------------------------------------------------
# _template_format_safe
# ---------------------------------------------------------------------------


class TestTemplateFormatSafe:
    def test_seed_style_template_is_safe(self) -> None:
        # The three required placeholders plus literal text -- safe.
        text = "src={source} col={column} vals:\n{value_list}"
        assert _template_format_safe(text) is True

    def test_doubled_braces_for_json_example_are_safe(self) -> None:
        # Properly escaped JSON example: braces are doubled so
        # str.format leaves them as literals.
        text = (
            "src={source} col={column} vals:\n{value_list}\n"
            'Reply with: {{"clusters": [{{"values": [...]}}]}}'
        )
        assert _template_format_safe(text) is True

    def test_unescaped_json_example_is_unsafe(self) -> None:
        # The exact failure mode that triggered the runtime warning
        # ``prompt missing placeholder '\n  "clusters"'``: a JSON
        # example with raw ``{`` instead of ``{{``.
        text = (
            "src={source} col={column} vals:\n{value_list}\n"
            'Reply with: {\n  "clusters": [...]\n}'
        )
        assert _template_format_safe(text) is False

    def test_unknown_named_field_is_unsafe(self) -> None:
        # A stray ``{foo}`` placeholder beyond the three allowed
        # ones would also raise KeyError at proposer call time.
        text = "src={source} col={column} {foo} vals:\n{value_list}"
        assert _template_format_safe(text) is False


# ---------------------------------------------------------------------------
# ClusterPromptVariantGenerator
# ---------------------------------------------------------------------------


class TestClusterPromptVariantGenerator:
    def test_generates_n_variants_in_order(self) -> None:
        responses = [_FakeResponse(_ok_payload(pass_index=i + 1)) for i in range(3)]
        llm = _ScriptedLLM(responses)
        gen = ClusterPromptVariantGenerator(
            llm=llm,  # type: ignore[arg-type]
            model="gpt-test",
            template_path=GENERATOR_TEMPLATE,
            reasoning_effort="low",
            reasoning_summary="detailed",
            concurrency=1,
        )
        variants = gen.generate(
            base_system="seed system",
            base_user_template=(
                "src={source} col={column} vals:\n{value_list}"
            ),
            n_variants=3,
        )
        assert len(variants) == 3
        # Pass indices are 1..N (0 is reserved for the identity seed).
        assert [v.pass_index for v in variants] == [1, 2, 3]
        assert all(v.ok for v in variants)
        for v in variants:
            assert "{source}" in v.user_template
            assert "{column}" in v.user_template
            assert "{value_list}" in v.user_template

    def test_zero_variants_returns_empty_list(self) -> None:
        llm = _ScriptedLLM([])
        gen = ClusterPromptVariantGenerator(
            llm=llm,  # type: ignore[arg-type]
            model="gpt-test",
            template_path=GENERATOR_TEMPLATE,
        )
        assert gen.generate(
            base_system="x", base_user_template="x", n_variants=0,
        ) == []

    def test_llm_error_falls_back_to_seed(self) -> None:
        responses = [_FakeResponse(text="", error="rate limit")]
        llm = _ScriptedLLM(responses)
        gen = ClusterPromptVariantGenerator(
            llm=llm,  # type: ignore[arg-type]
            model="gpt-test",
            template_path=GENERATOR_TEMPLATE,
            concurrency=1,
        )
        seed_system = "SEED SYSTEM"
        seed_template = "src={source} col={column} vals:\n{value_list}"
        variants = gen.generate(
            base_system=seed_system,
            base_user_template=seed_template,
            n_variants=1,
        )
        assert len(variants) == 1
        v = variants[0]
        assert v.ok is False
        # Fallback: variant body is the seed (so callers can blindly
        # use it without checking ``ok``).
        assert v.system == seed_system
        assert v.user_template == seed_template
        assert v.error == "rate limit"

    def test_invalid_json_falls_back_to_seed(self) -> None:
        responses = [_FakeResponse(text="not json")]
        llm = _ScriptedLLM(responses)
        gen = ClusterPromptVariantGenerator(
            llm=llm,  # type: ignore[arg-type]
            model="gpt-test",
            template_path=GENERATOR_TEMPLATE,
            concurrency=1,
        )
        seed_system = "SEED SYSTEM"
        seed_template = "src={source} col={column} vals:\n{value_list}"
        variants = gen.generate(
            base_system=seed_system,
            base_user_template=seed_template,
            n_variants=1,
        )
        assert len(variants) == 1
        assert variants[0].ok is False
        assert variants[0].system == seed_system
        assert variants[0].user_template == seed_template
        assert variants[0].error and variants[0].error.startswith("parse:")

    def test_missing_placeholder_falls_back_to_seed(self) -> None:
        # Variant drops {value_list} -- must be rejected so a downstream
        # ``str.format(value_list=...)`` doesn't blow up.
        bad_payload = json.dumps({
            "system": "ok",
            "user_template": "src={source} col={column}",  # no {value_list}
        })
        llm = _ScriptedLLM([_FakeResponse(bad_payload)])
        gen = ClusterPromptVariantGenerator(
            llm=llm,  # type: ignore[arg-type]
            model="gpt-test",
            template_path=GENERATOR_TEMPLATE,
            concurrency=1,
        )
        variants = gen.generate(
            base_system="seed",
            base_user_template="seed {source} {column} {value_list}",
            n_variants=1,
        )
        assert len(variants) == 1
        assert variants[0].ok is False
        assert variants[0].user_template == "seed {source} {column} {value_list}"

    def test_unescaped_json_braces_fall_back_to_seed(self) -> None:
        # The exact failure observed in production: the variant
        # paraphrased the prompt but emitted a raw ``{ "clusters": ... }``
        # JSON example instead of ``{{ "clusters": ... }}``. Python
        # ``str.format`` would then read ``\n  "clusters"`` as a
        # placeholder name and raise KeyError at proposer call time.
        # The variant validator must catch this and fall back.
        bad_template = (
            "src={source} col={column} vals:\n{value_list}\n"
            'Reply with: {\n  "clusters": [...]\n}'
        )
        bad_payload = json.dumps({
            "system": "ok",
            "user_template": bad_template,
        })
        llm = _ScriptedLLM([_FakeResponse(bad_payload)])
        gen = ClusterPromptVariantGenerator(
            llm=llm,  # type: ignore[arg-type]
            model="gpt-test",
            template_path=GENERATOR_TEMPLATE,
            concurrency=1,
        )
        seed_template = "seed {source} {column} {value_list}"
        variants = gen.generate(
            base_system="seed",
            base_user_template=seed_template,
            n_variants=1,
        )
        assert len(variants) == 1
        assert variants[0].ok is False
        assert variants[0].user_template == seed_template
        # Diagnostic should mention the brace-escaping rule so an
        # operator reading cluster_prompt_variants.jsonl can fix the
        # generator template.
        assert variants[0].error and "unescaped brace" in variants[0].error

    def test_strips_json_fences(self) -> None:
        fenced = (
            "```json\n"
            + _ok_payload(pass_index=1)
            + "\n```"
        )
        llm = _ScriptedLLM([_FakeResponse(fenced)])
        gen = ClusterPromptVariantGenerator(
            llm=llm,  # type: ignore[arg-type]
            model="gpt-test",
            template_path=GENERATOR_TEMPLATE,
            concurrency=1,
        )
        variants = gen.generate(
            base_system="seed",
            base_user_template="src={source} col={column} vals:\n{value_list}",
            n_variants=1,
        )
        assert variants[0].ok is True

    def test_template_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such.yaml"
        with pytest.raises(FileNotFoundError):
            ClusterPromptVariantGenerator(
                llm=_ScriptedLLM([]),  # type: ignore[arg-type]
                model="x",
                template_path=missing,
            )


# ---------------------------------------------------------------------------
# save_cluster_variant_records
# ---------------------------------------------------------------------------


class TestSaveClusterVariantRecords:
    def test_writes_jsonl_with_one_record_per_variant(
        self, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        variants = [
            ClusterPromptVariant(
                pass_index=1, system="a", user_template="aa",
                ok=True, raw_text="{}", reasoning_summary="",
                error=None,
            ),
            ClusterPromptVariant(
                pass_index=2, system="b", user_template="bb",
                ok=False, raw_text="bad", reasoning_summary="",
                error="parse: bad",
            ),
        ]
        path = save_cluster_variant_records(variants, run_dir=run_dir)
        assert path == run_dir / "cluster_prompt_variants.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        rec0 = json.loads(lines[0])
        assert rec0["pass_index"] == 1
        assert rec0["ok"] is True
        rec1 = json.loads(lines[1])
        assert rec1["ok"] is False
        assert rec1["error"] == "parse: bad"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stale = run_dir / "cluster_prompt_variants.jsonl"
        stale.write_text("STALE\n", encoding="utf-8")
        save_cluster_variant_records(
            [ClusterPromptVariant(
                pass_index=1, system="x", user_template="x",
            )],
            run_dir=run_dir,
        )
        body = stale.read_text(encoding="utf-8")
        assert "STALE" not in body
        assert body.startswith("{")
