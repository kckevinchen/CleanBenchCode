"""Smoke tests for the inline validation pass invoked by
``experiments.run_relation_discovery._run_validation_pass``.

The pass is a thin wrapper around
:func:`src.core.relations.validator.validate_relations`; the real
discovery runner end-to-end exercise lives in the integration suite,
so here we just verify wiring: layer auto-resolution, missing-prompt
short-circuit, and JSONL emission.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from experiments.run_relation_discovery import _run_validation_pass


class _StubLLM:
    """Stand-in LLM that yields a verifiable response that always
    passes (``check`` returns ``True`` for every row). The real
    LLMClient is not exercised here."""

    def __init__(self) -> None:
        self.calls = 0

    def query(self, *, model: str, prompt: str, reasoning_effort: str,
              system_prompt: str) -> Any:
        self.calls += 1

        class _R:
            text = json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return df['A'].notna()\n",
            })
            attempts = 1

        return _R()


def _seed_layer(run_dir: Path, layer: str, relations: List[Dict[str, Any]]) -> None:
    layer_files = {
        "aggregated":  "relations_aggregated.jsonl",
        "originals":   "relations.jsonl",
    }
    p = run_dir / layer_files[layer]
    with p.open("w", encoding="utf-8") as fh:
        for r in relations:
            fh.write(json.dumps(r))
            fh.write("\n")


def _relation(species_key: str = "abc123") -> Dict[str, Any]:
    return {
        "species_key": species_key,
        "source": "t.csv",
        "error_type": "x",
        "kind": "implication",
        "columns": ["A"],
        "lhs": ["A"],
        "rhs": ["A"],
        "predicate": "A is non-null",
        "executable": "A is not null",
        "rationale": "",
        "confidence": "high",
    }


class TestRunValidationPass:
    def test_skips_when_prompt_missing(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stats = _run_validation_pass(
            run_dir=run_dir,
            source_frames={"t.csv": pd.DataFrame({"A": [1, 2, 3]})},
            llm=_StubLLM(),
            model="m",
            prompt_path=tmp_path / "missing.yaml",
            layer="auto",
            concurrency=1,
            sample_rows=5,
            timeout_seconds=2.0,
            max_violation_samples=2,
            limit=None,
        )
        assert stats is None
        assert not (run_dir / "relation_verifications.jsonl").exists()

    def test_no_relations_emits_zero_stats(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(
            "system_prompt: ''\nuser_template: ''\n", encoding="utf-8"
        )
        stats = _run_validation_pass(
            run_dir=run_dir,
            source_frames={},
            llm=_StubLLM(),
            model="m",
            prompt_path=prompt_path,
            layer="auto",
            concurrency=1,
            sample_rows=5,
            timeout_seconds=2.0,
            max_violation_samples=2,
            limit=None,
        )
        assert stats is not None
        assert stats["n_relations"] == 0
        assert stats["layer"] == "auto"

    def test_auto_layer_picks_highest(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Seed both layers with distinct keys so we can verify
        # which layer was picked from the verifications JSONL.
        _seed_layer(run_dir, "originals", [_relation("orig1")])
        _seed_layer(run_dir, "aggregated", [_relation("agg1")])

        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(
            "system_prompt: ''\nuser_template: ''\n", encoding="utf-8"
        )

        stats = _run_validation_pass(
            run_dir=run_dir,
            source_frames={"t.csv": pd.DataFrame({"A": [1, 2, 3]})},
            llm=_StubLLM(),
            model="m",
            prompt_path=prompt_path,
            layer="auto",
            concurrency=1,
            sample_rows=5,
            timeout_seconds=2.0,
            max_violation_samples=2,
            limit=None,
        )
        assert stats is not None
        assert stats["layer"] == "aggregated"
        assert stats["n_relations"] == 1
        # JSONL written; species_key tracks the aggregated layer.
        path = run_dir / "relation_verifications.jsonl"
        assert path.is_file()
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 1
        assert rows[0]["species_key"] == "agg1"
        assert rows[0]["layer"] == "aggregated"

    def test_explicit_layer_overrides_auto(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _seed_layer(run_dir, "originals", [_relation("orig1")])
        _seed_layer(run_dir, "aggregated", [_relation("agg1")])

        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(
            "system_prompt: ''\nuser_template: ''\n", encoding="utf-8"
        )
        stats = _run_validation_pass(
            run_dir=run_dir,
            source_frames={"t.csv": pd.DataFrame({"A": [1, 2, 3]})},
            llm=_StubLLM(),
            model="m",
            prompt_path=prompt_path,
            layer="originals",
            concurrency=1,
            sample_rows=5,
            timeout_seconds=2.0,
            max_violation_samples=2,
            limit=None,
        )
        assert stats is not None
        assert stats["layer"] == "originals"
        rows = [
            json.loads(l)
            for l in (run_dir / "relation_verifications.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert rows[0]["species_key"] == "orig1"

    def test_limit_caps_relations(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _seed_layer(run_dir, "originals", [
            _relation(f"r{i}") for i in range(5)
        ])
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(
            "system_prompt: ''\nuser_template: ''\n", encoding="utf-8"
        )
        stats = _run_validation_pass(
            run_dir=run_dir,
            source_frames={"t.csv": pd.DataFrame({"A": [1, 2, 3]})},
            llm=_StubLLM(),
            model="m",
            prompt_path=prompt_path,
            layer="auto",
            concurrency=1,
            sample_rows=5,
            timeout_seconds=2.0,
            max_violation_samples=2,
            limit=2,
        )
        assert stats is not None
        assert stats["n_relations"] == 2


class TestBuildRelationsArgvValidate:
    def test_validate_default_emits_no_negative_flag(self) -> None:
        from src.labeling.runs import build_relations_argv
        argv = build_relations_argv({"csv": "data/t.csv"})
        assert "--no-validate" not in argv

    def test_validate_false_emits_negative_flag(self) -> None:
        from src.labeling.runs import build_relations_argv
        argv = build_relations_argv({"csv": "data/t.csv", "validate": False})
        assert "--no-validate" in argv

    def test_validate_layer_flag_passed_through(self) -> None:
        from src.labeling.runs import build_relations_argv
        argv = build_relations_argv({
            "csv": "data/t.csv",
            "validate": True,
            "validate_layer": "originals",
            "validate_limit": 50,
        })
        assert "--validate-layer" in argv
        i = argv.index("--validate-layer")
        assert argv[i + 1] == "originals"
        assert "--validate-limit" in argv
        j = argv.index("--validate-limit")
        assert argv[j + 1] == "50"
