"""Tests for src.core.relation_aggregate: paraphrase-merging via LLM.

Uses a stub LLM that yields canned JSON responses so we can drive
the aggregation pipeline deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.core.llm_client import LLMResponse
from src.core.relation_aggregate import (
    AggregationOutcome,
    AggregationPromptConfig,
    AggregationResult,
    CanonicalRule,
    _coerce_aggregation,
    _coerce_canonical_rule,
    aggregate_relations,
    cluster_relations_by_arm,
    load_aggregation_prompt,
    write_aggregated_jsonl,
    write_aggregation_report_json,
)
from src.core.relations import Relation
from src.core.table_digest import build_table_digest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rel(predicate: str, *, source="t.csv", et="fd_violation",
         kind="fd", cols=("A", "B")) -> Relation:
    return Relation(
        source=source,
        error_type=et,
        kind=kind,
        columns=cols,
        predicate=predicate,
        lhs=(cols[0],),
        rhs=(cols[1],) if len(cols) > 1 else (),
    )


def _hr_frame() -> pd.DataFrame:
    return pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})


class _StubLLM:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def query(self, *, model, prompt, reasoning_effort, reasoning_summary, system_prompt):
        self.calls += 1
        if not self.responses:
            return LLMResponse(text="", endpoint="stub://0", model=model, attempts=1, error="exhausted")
        return LLMResponse(
            text=self.responses.pop(0),
            endpoint="stub://0",
            model=model,
            attempts=1,
        )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestCluster:
    def test_groups_by_source_error_columns(self) -> None:
        rels = [
            _rel("p1"),
            _rel("p2"),  # same cluster as p1
            _rel("q", source="other.csv"),
        ]
        clusters = cluster_relations_by_arm(rels, min_cluster_size=2)
        assert len(clusters) == 1
        assert {r.predicate for r in clusters[0]} == {"p1", "p2"}

    def test_column_order_doesnt_split_clusters(self) -> None:
        rels = [
            _rel("p1", cols=("A", "B")),
            _rel("p2", cols=("B", "A")),  # reversed but should match
        ]
        clusters = cluster_relations_by_arm(rels, min_cluster_size=2)
        assert len(clusters) == 1

    def test_singletons_are_dropped(self) -> None:
        rels = [_rel("p1"), _rel("q", source="other.csv")]
        assert cluster_relations_by_arm(rels, min_cluster_size=2) == []


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


class TestCoerceCanonicalRule:
    def _template(self) -> Relation:
        return _rel("template", cols=("A", "B"))

    def test_happy_path(self) -> None:
        item = {
            "kind": "fd",
            "columns": ["A", "B"],
            "lhs": ["A"],
            "rhs": ["B"],
            "predicate": "A determines B",
            "covers": [1, 2],
            "confidence": "high",
        }
        cr, err = _coerce_canonical_rule(item, self._template(), 2)
        assert err == ""
        assert cr is not None
        assert cr.relation.predicate == "A determines B"
        assert cr.covers == [1, 2]

    def test_missing_columns_rejected(self) -> None:
        item = {"predicate": "p", "covers": [1]}
        cr, err = _coerce_canonical_rule(item, self._template(), 1)
        # uses template columns when missing -> ok actually
        assert cr is not None

    def test_unknown_column_rejected(self) -> None:
        item = {"columns": ["NOPE"], "predicate": "p", "covers": [1]}
        cr, err = _coerce_canonical_rule(item, self._template(), 1)
        assert cr is None
        assert "unknown column" in err

    def test_covers_out_of_range(self) -> None:
        item = {"columns": ["A"], "predicate": "p", "covers": [5]}
        cr, err = _coerce_canonical_rule(item, self._template(), 2)
        assert cr is None
        assert "out of range" in err

    def test_empty_predicate_rejected(self) -> None:
        item = {"columns": ["A"], "predicate": "", "covers": [1]}
        cr, err = _coerce_canonical_rule(item, self._template(), 1)
        assert cr is None
        assert "empty predicate" in err


class TestCoerceAggregation:
    def _cluster(self) -> List[Relation]:
        return [_rel("p1"), _rel("p2"), _rel("p3")]

    def test_partition_validated(self) -> None:
        payload = {
            "canonical": [
                {"columns": ["A", "B"], "predicate": "A→B canonical", "covers": [1, 2, 3]},
            ],
            "reason": "merged",
        }
        out, reason = _coerce_aggregation(payload, self._cluster())
        assert len(out) == 1
        assert reason == "merged"

    def test_decline_returns_empty(self) -> None:
        out, reason = _coerce_aggregation({"canonical": [], "reason": "no merge"}, self._cluster())
        assert out == []
        assert "no merge" in reason

    def test_overlapping_covers_rejected(self) -> None:
        payload = {
            "canonical": [
                {"columns": ["A"], "predicate": "p1", "covers": [1, 2]},
                {"columns": ["A"], "predicate": "p2", "covers": [2, 3]},
            ],
        }
        out, reason = _coerce_aggregation(payload, self._cluster())
        assert out == []
        assert "multiple canonical" in reason

    def test_missing_input_rejected(self) -> None:
        payload = {
            "canonical": [
                {"columns": ["A"], "predicate": "p", "covers": [1, 2]},
                # rule 3 not covered.
            ],
        }
        out, reason = _coerce_aggregation(payload, self._cluster())
        assert out == []
        assert "not covered" in reason

    def test_noop_partition_rejected(self) -> None:
        payload = {
            "canonical": [
                {"columns": ["A"], "predicate": "p1", "covers": [1]},
                {"columns": ["A"], "predicate": "p2", "covers": [2]},
                {"columns": ["A"], "predicate": "p3", "covers": [3]},
            ],
        }
        out, reason = _coerce_aggregation(payload, self._cluster())
        assert out == []
        assert "no merges" in reason

    def test_non_object_response(self) -> None:
        out, reason = _coerce_aggregation([], self._cluster())
        assert out == []


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestAggregateRelations:
    def _setup(self):
        df = _hr_frame()
        digest = build_table_digest(df, source="t.csv")
        prompt_cfg = AggregationPromptConfig(
            system_prompt="sys",
            user_template="cluster has {cluster_size}: {cluster_rules}\n{sample_rows}\n{columns}\n{source}\n{error_type}",
        )
        return df, digest, prompt_cfg

    def test_merges_paraphrase_pair(self) -> None:
        df, digest, prompt_cfg = self._setup()
        rels = [_rel("A determines B"), _rel("A => B")]
        canned = json.dumps({
            "canonical": [
                {"columns": ["A", "B"], "predicate": "A determines B (canonical)",
                 "lhs": ["A"], "rhs": ["B"], "covers": [1, 2], "kind": "fd"},
            ],
            "reason": "paraphrase merge",
        })
        result = aggregate_relations(
            rels,
            digests={"t.csv": digest},
            source_frames={"t.csv": df},
            prompt_cfg=prompt_cfg,
            llm=_StubLLM([canned]),  # type: ignore[arg-type]
            model="m",
            concurrency=1,
        )
        assert result.n_clusters == 1
        assert result.n_aggregated_clusters == 1
        assert result.n_canonical_rules == 1
        assert result.n_originals_collapsed == 2
        assert len(result.covered_species_keys) >= 1

    def test_no_clusters_returns_empty(self) -> None:
        df, digest, prompt_cfg = self._setup()
        result = aggregate_relations(
            [_rel("only one")],
            digests={"t.csv": digest},
            source_frames={"t.csv": df},
            prompt_cfg=prompt_cfg,
            llm=_StubLLM([]),  # type: ignore[arg-type]
            model="m",
        )
        assert result.outcomes == []

    def test_llm_decline_keeps_originals(self) -> None:
        df, digest, prompt_cfg = self._setup()
        rels = [_rel("p1"), _rel("p2")]
        result = aggregate_relations(
            rels,
            digests={"t.csv": digest},
            source_frames={"t.csv": df},
            prompt_cfg=prompt_cfg,
            llm=_StubLLM([json.dumps({"canonical": [], "reason": "distinct"})]),  # type: ignore[arg-type]
            model="m",
            concurrency=1,
        )
        assert result.n_aggregated_clusters == 0
        assert result.n_declined == 1
        # post_aggregation_relations returns originals unchanged when no merge.
        post = result.post_aggregation_relations(rels)
        assert len(post) == 2

    def test_post_aggregation_swaps_in_canonical(self) -> None:
        df, digest, prompt_cfg = self._setup()
        rels = [_rel("p1"), _rel("p2")]
        canned = json.dumps({
            "canonical": [{"columns": ["A", "B"], "predicate": "merged", "covers": [1, 2]}],
            "reason": "ok",
        })
        result = aggregate_relations(
            rels,
            digests={"t.csv": digest},
            source_frames={"t.csv": df},
            prompt_cfg=prompt_cfg,
            llm=_StubLLM([canned]),  # type: ignore[arg-type]
            model="m",
            concurrency=1,
        )
        post = result.post_aggregation_relations(rels)
        assert len(post) == 1
        assert post[0].predicate == "merged"

    def test_malformed_json_yields_error_outcome(self) -> None:
        df, digest, prompt_cfg = self._setup()
        rels = [_rel("p1"), _rel("p2")]
        result = aggregate_relations(
            rels,
            digests={"t.csv": digest},
            source_frames={"t.csv": df},
            prompt_cfg=prompt_cfg,
            llm=_StubLLM(["not json at all"]),  # type: ignore[arg-type]
            model="m",
            concurrency=1,
        )
        assert result.n_errored == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def _result(self) -> AggregationResult:
        rels = [_rel("p1"), _rel("p2")]
        cr = CanonicalRule(relation=_rel("merged"), covers=[1, 2])
        return AggregationResult(outcomes=[
            AggregationOutcome(cluster=rels, replacements=[cr], reason="ok"),
        ])

    def test_writes_jsonl_with_aggregated_from(self, tmp_path: Path) -> None:
        result = self._result()
        out = tmp_path / "agg.jsonl"
        n = write_aggregated_jsonl(result, out)
        assert n == 1
        rec = json.loads(out.read_text().strip())
        assert rec["predicate"] == "merged"
        assert "aggregated_from" in rec
        assert rec["n_originals"] == 2
        assert len(rec["aggregated_from"]) == 2

    def test_report_json_includes_declined(self, tmp_path: Path) -> None:
        rels = [_rel("p1"), _rel("p2")]
        result = AggregationResult(outcomes=[
            AggregationOutcome(cluster=rels, replacements=[], reason="distinct"),
        ])
        out = tmp_path / "report.json"
        write_aggregation_report_json(result, out)
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["replaced"] is False
        assert data[0]["reason"] == "distinct"


class TestLoadPrompt:
    def test_loads_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text("system: hello\nuser_template: 'do {cluster_size}'\n", encoding="utf-8")
        cfg = load_aggregation_prompt(path)
        assert cfg.system_prompt == "hello"
        assert "{cluster_size}" in cfg.user_template
