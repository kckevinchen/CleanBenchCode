"""Tests for src.core.relation_generalize: collapse value-keyed rule families."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.core.llm_client import LLMResponse
from src.core.relation_generalize import (
    GeneralizationOutcome,
    GeneralizationPromptConfig,
    GeneralizationResult,
    _coerce_replacement,
    cluster_value_conditional,
    generalize_relations,
    is_value_conditional,
    load_generalization_prompt,
    write_generalization_report_json,
    write_generalized_jsonl,
)
from src.core.relations import Relation
from src.core.table_digest import build_table_digest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Description": ["Exemption 10 MW", "Exemption 5 MW", "Exemption 1 MW", "Other"],
        "Capacity": [10000, 5000, 1000, 200],
    })


def _digest(df: pd.DataFrame):
    return build_table_digest(df, source="t.csv")


def _rel(predicate: str, *, cols=("Description", "Capacity")) -> Relation:
    return Relation(
        source="t.csv",
        error_type="threshold_violation",
        kind="implication",
        columns=cols,
        predicate=predicate,
    )


class _StubLLM:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)

    def query(self, *, model, prompt, reasoning_effort, reasoning_summary, system_prompt):
        if not self.responses:
            return LLMResponse(text="", endpoint="stub://0", model=model, attempts=1, error="exhausted")
        return LLMResponse(text=self.responses.pop(0), endpoint="stub://0", model=model, attempts=1)


# ---------------------------------------------------------------------------
# is_value_conditional
# ---------------------------------------------------------------------------


class TestIsValueConditional:
    def test_quoted_literal_present_in_digest(self) -> None:
        df = _frame()
        digest = _digest(df)
        rel = _rel("If Description == 'Exemption 10 MW' then Capacity <= 10000")
        assert is_value_conditional(rel, digest) is True

    def test_no_quoted_literal_returns_false(self) -> None:
        df = _frame()
        digest = _digest(df)
        rel = _rel("Capacity <= 10000")
        assert is_value_conditional(rel, digest) is False

    def test_quoted_but_not_in_digest_returns_false(self) -> None:
        df = _frame()
        digest = _digest(df)
        rel = _rel("If Description == 'Nonexistent Value Xyz' then Capacity <= 10000")
        assert is_value_conditional(rel, digest) is False

    def test_no_digest_falls_back_to_heuristic(self) -> None:
        rel = _rel("If Description == 'Exemption 10 MW' then Capacity <= 10000")
        # Without a digest we can't verify; quoted text not a column name -> True.
        assert is_value_conditional(rel, None) is True

    def test_empty_predicate(self) -> None:
        rel = _rel("")
        assert is_value_conditional(rel, None) is False


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestClusterValueConditional:
    def test_groups_value_conditional_family(self) -> None:
        df = _frame()
        digest = _digest(df)
        rels = [
            _rel("If Description == 'Exemption 10 MW' then Capacity <= 10000"),
            _rel("If Description == 'Exemption 5 MW' then Capacity <= 5000"),
            _rel("If Description == 'Exemption 1 MW' then Capacity <= 1000"),
        ]
        clusters = cluster_value_conditional(rels, {"t.csv": digest}, min_cluster_size=2)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_skips_non_value_conditional(self) -> None:
        df = _frame()
        digest = _digest(df)
        rels = [
            _rel("Capacity <= 10000"),
            _rel("Capacity > 0"),
        ]
        clusters = cluster_value_conditional(rels, {"t.csv": digest}, min_cluster_size=2)
        assert clusters == []

    def test_below_min_size_dropped(self) -> None:
        df = _frame()
        digest = _digest(df)
        rels = [_rel("If Description == 'Exemption 10 MW' then Capacity <= 10000")]
        clusters = cluster_value_conditional(rels, {"t.csv": digest}, min_cluster_size=2)
        assert clusters == []


# ---------------------------------------------------------------------------
# _coerce_replacement
# ---------------------------------------------------------------------------


class TestCoerceReplacement:
    def _template(self) -> Relation:
        return _rel("template")

    def test_happy_path(self) -> None:
        payload = {
            "generalized": {
                "kind": "implication",
                "columns": ["Description", "Capacity"],
                "predicate": "If Description matches 'Exemption N MW' then Capacity <= N*1000",
                "lhs": ["Description"],
                "rhs": ["Capacity"],
            },
            "reason": "parametric form",
        }
        rel, reason = _coerce_replacement(payload, self._template())
        assert rel is not None
        assert reason == "parametric form"
        assert "Exemption N MW" in rel.predicate

    def test_decline_returns_none(self) -> None:
        rel, reason = _coerce_replacement({"generalized": None, "reason": "no pattern"}, self._template())
        assert rel is None
        assert reason == "no pattern"

    def test_unknown_column_rejected(self) -> None:
        payload = {"generalized": {"columns": ["NOPE"], "predicate": "p"}}
        rel, reason = _coerce_replacement(payload, self._template())
        assert rel is None
        assert "unknown column" in reason

    def test_empty_predicate_rejected(self) -> None:
        payload = {"generalized": {"columns": ["Description"], "predicate": ""}}
        rel, reason = _coerce_replacement(payload, self._template())
        assert rel is None
        assert "empty predicate" in reason

    def test_non_object_response(self) -> None:
        rel, reason = _coerce_replacement([], self._template())
        assert rel is None


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestGeneralizeRelations:
    def _setup(self):
        df = _frame()
        digest = _digest(df)
        prompt_cfg = GeneralizationPromptConfig(
            system_prompt="sys",
            user_template="cluster has {cluster_size}: {cluster_rules}\n{sample_rows}\n{columns}\n{source}\n{error_type}",
        )
        return df, digest, prompt_cfg

    def test_generalizes_family(self) -> None:
        df, digest, prompt_cfg = self._setup()
        rels = [
            _rel("If Description == 'Exemption 10 MW' then Capacity <= 10000"),
            _rel("If Description == 'Exemption 5 MW' then Capacity <= 5000"),
            _rel("If Description == 'Exemption 1 MW' then Capacity <= 1000"),
        ]
        canned = json.dumps({
            "generalized": {
                "kind": "implication",
                "columns": ["Description", "Capacity"],
                "predicate": "If Description == 'Exemption N MW' then Capacity <= N*1000",
                "lhs": ["Description"], "rhs": ["Capacity"],
            },
            "reason": "ok",
        })
        result = generalize_relations(
            rels,
            digests={"t.csv": digest},
            source_frames={"t.csv": df},
            prompt_cfg=prompt_cfg,
            llm=_StubLLM([canned]),  # type: ignore[arg-type]
            model="m",
            concurrency=1,
        )
        assert result.n_clusters == 1
        assert result.n_generalized == 1
        assert result.n_originals_collapsed == 3

    def test_decline_keeps_originals(self) -> None:
        df, digest, prompt_cfg = self._setup()
        rels = [
            _rel("If Description == 'Exemption 10 MW' then Capacity <= 10000"),
            _rel("If Description == 'Exemption 5 MW' then Capacity <= 5000"),
        ]
        result = generalize_relations(
            rels,
            digests={"t.csv": digest},
            source_frames={"t.csv": df},
            prompt_cfg=prompt_cfg,
            llm=_StubLLM([json.dumps({"generalized": None, "reason": "distinct"})]),  # type: ignore[arg-type]
            model="m",
            concurrency=1,
        )
        assert result.n_generalized == 0
        assert result.n_declined == 1

    def test_no_clusters_returns_empty(self) -> None:
        df, digest, prompt_cfg = self._setup()
        result = generalize_relations(
            [_rel("Capacity > 0")],
            digests={"t.csv": digest},
            source_frames={"t.csv": df},
            prompt_cfg=prompt_cfg,
            llm=_StubLLM([]),  # type: ignore[arg-type]
            model="m",
        )
        assert result.outcomes == []

    def test_malformed_json_yields_error(self) -> None:
        df, digest, prompt_cfg = self._setup()
        rels = [
            _rel("If Description == 'Exemption 10 MW' then Capacity <= 10000"),
            _rel("If Description == 'Exemption 5 MW' then Capacity <= 5000"),
        ]
        result = generalize_relations(
            rels,
            digests={"t.csv": digest},
            source_frames={"t.csv": df},
            prompt_cfg=prompt_cfg,
            llm=_StubLLM(["not json"]),  # type: ignore[arg-type]
            model="m",
            concurrency=1,
        )
        assert result.n_errored == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_writes_jsonl_with_generalized_from(self, tmp_path: Path) -> None:
        rels = [
            _rel("If Description == 'Exemption 10 MW' then Capacity <= 10000"),
            _rel("If Description == 'Exemption 5 MW' then Capacity <= 5000"),
        ]
        replacement = _rel("If Description == 'Exemption N MW' then Capacity <= N*1000")
        result = GeneralizationResult(outcomes=[
            GeneralizationOutcome(cluster=rels, replacement=replacement, reason="ok"),
        ])
        out = tmp_path / "gen.jsonl"
        n = write_generalized_jsonl(result, out)
        assert n == 1
        rec = json.loads(out.read_text().strip())
        assert "generalized_from" in rec
        assert rec["n_originals"] == 2

    def test_report_includes_declined(self, tmp_path: Path) -> None:
        rels = [_rel("If Description == 'X' then Capacity <= 1")]
        result = GeneralizationResult(outcomes=[
            GeneralizationOutcome(cluster=rels, replacement=None, reason="distinct"),
        ])
        out = tmp_path / "report.json"
        write_generalization_report_json(result, out)
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["replaced"] is False


class TestLoadPrompt:
    def test_loads_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text("system: sys\nuser_template: 't {cluster_size}'\n", encoding="utf-8")
        cfg = load_generalization_prompt(path)
        assert cfg.system_prompt == "sys"
        assert "{cluster_size}" in cfg.user_template
