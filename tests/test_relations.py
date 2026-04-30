"""Tests for src.core.relations: prompt loader, row sampler, JSON
parsing, Relation dedup, SchemaDiscoverer (with a fake LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
import yaml

from src.core.error_taxonomy import ErrorType, ErrorTaxonomy, load_taxonomy
from src.core.llm_client import LLMResponse
from src.core.relations import (
    DiscoveryPromptConfig,
    Relation,
    SchemaDiscoverer,
    _fold_predicate,
    _parse_relations_json,
    load_discovery_prompt,
    sample_rows_for_prompt,
    write_relations_jsonl,
)
from src.core.table_digest import build_table_digest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hr_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EmpID": [f"E{i:03d}" for i in range(8)],
            "Zip": ["94103"] * 3 + ["10001"] * 3 + ["60601"] * 2,
            "State": ["CA"] * 3 + ["NY"] * 3 + ["IL"] * 2,
            "UnitPrice": ["10", "20", "30", "40", "50", "60", "70", "80"],
            "Quantity": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "TotalPrice": ["10", "40", "90", "160", "250", "360", "490", "640"],
        }
    )


def _make_taxonomy() -> ErrorTaxonomy:
    types = (
        ErrorType(
            key="fd_violation",
            name="FD Violation",
            category="Multi-Column / Intra-Record",
            description="A determines B",
            example="Zip -> State",
            notes="...",
        ),
        ErrorType(
            key="row_calculation_inconsistency",
            name="Row Calculation",
            category="Multi-Column / Intra-Record",
            description="c == a*b",
            example="total = price * qty",
            notes="...",
        ),
    )
    return ErrorTaxonomy(
        system_prompt="sys",
        user_template="{type_key}",
        types=types,
    )


def _make_discovery_cfg(tmp_path: Path) -> DiscoveryPromptConfig:
    p = Path(__file__).resolve().parent.parent / "prompts" / "discover_relations.yaml"
    return load_discovery_prompt(p)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoadDiscoveryPrompt:
    def test_loads_real_yaml(self, tmp_path: Path) -> None:
        cfg = _make_discovery_cfg(tmp_path)
        assert "fd_violation" in cfg.types
        assert "row_calculation_inconsistency" in cfg.types
        assert cfg.system_prompt
        assert "{table_overview}" in cfg.user_template

    def test_render_substitutes_kinds_and_overview(self, tmp_path: Path) -> None:
        cfg = _make_discovery_cfg(tmp_path)
        tax = _make_taxonomy()
        df = _hr_frame()
        d = build_table_digest(df, source="hr.csv")
        out = cfg.render(
            error_type=tax.get("fd_violation"),
            digest=d,
            row_sample_md="| __row__ |\n|---|\n| 0 |",
        )
        assert "fd_violation" in out
        assert "TABLE OVERVIEW" in out
        assert "ALLOWED `kind`" in out
        assert "- fd" in out

    def test_unknown_error_type_raises(self, tmp_path: Path) -> None:
        cfg = _make_discovery_cfg(tmp_path)
        tax = _make_taxonomy()
        bogus = ErrorType(
            key="not_in_yaml", name="X", category="Y",
            description="d", example="", notes="",
        )
        d = build_table_digest(_hr_frame(), source="x.csv")
        with pytest.raises(KeyError):
            cfg.render(error_type=bogus, digest=d, row_sample_md="")


# ---------------------------------------------------------------------------
# Row sampling
# ---------------------------------------------------------------------------


class TestSampleRows:
    def test_default_picks_endpoints(self) -> None:
        df = pd.DataFrame({"x": list(range(10))})
        out = sample_rows_for_prompt(df, n_samples=3)
        assert "0" in out and "9" in out

    def test_excludes_total_rows(self) -> None:
        df = pd.DataFrame(
            {"label": ["a"] * 5 + ["Total"], "v": list(range(6))}
        )
        out = sample_rows_for_prompt(
            df, n_samples=3, total_row_indices=[5]
        )
        assert "Total" not in out

    def test_short_table_returns_all(self) -> None:
        df = pd.DataFrame({"x": [1, 2]})
        out = sample_rows_for_prompt(df, n_samples=5)
        # Both rows present.
        assert "1" in out and "2" in out

    def test_empty_table_returns_placeholder(self) -> None:
        df = pd.DataFrame({"x": []})
        assert "empty table" in sample_rows_for_prompt(df, n_samples=3)


# ---------------------------------------------------------------------------
# JSON parsing + Relation canonicalization
# ---------------------------------------------------------------------------


class TestParseRelationsJson:
    def test_plain_json(self) -> None:
        text = json.dumps({"relations": [{"kind": "fd", "columns": ["A", "B"]}]})
        items, err = _parse_relations_json(text)
        assert err is None
        assert items == [{"kind": "fd", "columns": ["A", "B"]}]

    def test_fenced_json(self) -> None:
        text = "```json\n" + json.dumps({"relations": [{"kind": "fd"}]}) + "\n```"
        items, err = _parse_relations_json(text)
        assert err is None
        assert items == [{"kind": "fd"}]

    def test_empty_response(self) -> None:
        items, err = _parse_relations_json("")
        assert items == []
        assert err == "empty response"

    def test_garbage(self) -> None:
        items, err = _parse_relations_json("not json at all")
        assert items == []
        assert err is not None


class TestRelationSpeciesKey:
    def _r(self, **kw: Any) -> Relation:
        defaults: Dict[str, Any] = dict(
            source="x.csv",
            error_type="fd_violation",
            kind="fd",
            columns=("Zip", "State"),
            predicate="Zip determines State",
            lhs=("Zip",),
            rhs=("State",),
        )
        defaults.update(kw)
        return Relation(**defaults)

    def test_paraphrase_collapses(self) -> None:
        a = self._r(predicate="Zip determines State")
        b = self._r(predicate="Zip   determines  State.")
        assert a.species_key() == b.species_key()

    def test_directional_keys_differ(self) -> None:
        # Composite FDs (multi-column lhs or rhs) keep direction.
        a = self._r(
            kind="fd",
            columns=("A", "B", "C"),
            lhs=("A", "B"),
            rhs=("C",),
            predicate="(A,B) determine C",
        )
        b = self._r(
            kind="fd",
            columns=("A", "B", "C"),
            lhs=("C",),
            rhs=("A", "B"),
            predicate="C determines (A,B)",
        )
        assert a.species_key() != b.species_key()

    def test_bijection_collapses_for_simple_fd(self) -> None:
        # Single-column A -> B and B -> A fold to the same species.
        a = self._r(lhs=("Zip",), rhs=("State",),
                    predicate="Zip determines State")
        b = self._r(lhs=("State",), rhs=("Zip",),
                    predicate="State determines Zip")
        assert a.species_key() == b.species_key()

    def test_bijection_does_not_collapse_other_kinds(self) -> None:
        # Non-fd kinds must stay direction-sensitive.
        a = self._r(kind="ordered", lhs=("StartDate",), rhs=("EndDate",),
                    predicate="Start <= End")
        b = self._r(kind="ordered", lhs=("EndDate",), rhs=("StartDate",),
                    predicate="End <= Start")
        assert a.species_key() != b.species_key()

    def test_kind_change_diverges(self) -> None:
        a = self._r(kind="fd")
        b = self._r(kind="composite_fd")
        assert a.species_key() != b.species_key()

    def test_fold_predicate_strips_punct(self) -> None:
        # Trailing punctuation, repeated whitespace, and case
        # variations all collapse to the same fold.
        assert _fold_predicate("Zip determines State.") == _fold_predicate(
            "zip   determines  state"
        )


# ---------------------------------------------------------------------------
# SchemaDiscoverer with a fake LLM
# ---------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self, responses: Dict[str, str]):
        # Map error_type key -> reply text.
        self.responses = responses
        self.endpoints = ["fake://0"]
        self.calls: List[Dict[str, Any]] = []

    def query(self, model, prompt, *, reasoning_effort, reasoning_summary, system_prompt):
        # Identify which type the prompt is asking about by header line.
        et = "fd_violation"
        for key in self.responses:
            if f"TYPE KEY:      {key}" in prompt:
                et = key
                break
        self.calls.append({"model": model, "type": et})
        return LLMResponse(
            text=self.responses.get(et, "{\"relations\": []}"),
            endpoint="fake://0",
            model=model,
            attempts=1,
        )


class TestSchemaDiscoverer:
    def test_end_to_end_single_pass(self, tmp_path: Path) -> None:
        df = _hr_frame()
        digest = build_table_digest(df, source="hr.csv")
        tax = _make_taxonomy()
        cfg = _make_discovery_cfg(tmp_path)

        fake = _FakeLLM(
            responses={
                "fd_violation": json.dumps(
                    {
                        "relations": [
                            {
                                "kind": "fd",
                                "columns": ["Zip", "State"],
                                "lhs": ["Zip"],
                                "rhs": ["State"],
                                "predicate": "Zip determines State",
                                "executable": "Zip -> State",
                                "confidence": "high",
                                "rationale": "Zip codes are state-bound.",
                            },
                            # Duplicate phrasing should fold.
                            {
                                "kind": "fd",
                                "columns": ["Zip", "State"],
                                "lhs": ["Zip"],
                                "rhs": ["State"],
                                "predicate": "Zip   determines State.",
                                "confidence": "high",
                                "rationale": "dup",
                            },
                            # Hallucinated column should be dropped.
                            {
                                "kind": "fd",
                                "columns": ["Zip", "NotAColumn"],
                                "predicate": "bogus",
                                "confidence": "low",
                            },
                        ]
                    }
                ),
                "row_calculation_inconsistency": json.dumps(
                    {
                        "relations": [
                            {
                                "kind": "product",
                                "columns": ["UnitPrice", "Quantity", "TotalPrice"],
                                "predicate": "TotalPrice == UnitPrice * Quantity",
                                "executable": "TotalPrice == UnitPrice * Quantity",
                                "confidence": "high",
                                "rationale": "Header naming + numeric dtypes.",
                            }
                        ]
                    }
                ),
            }
        )

        disc = SchemaDiscoverer(
            llm=fake,  # type: ignore[arg-type]
            prompt_config=cfg,
            taxonomy=tax,
            model="fake-model",
            digests={"hr.csv": digest},
            source_frames={"hr.csv": df},
            error_types=list(tax.types),
        )
        reports = disc.run()
        assert len(reports) == 2
        rels_by_type = {rep.error_type: rep.relations for rep in reports}

        fd_rels = rels_by_type["fd_violation"]
        assert len(fd_rels) == 1  # dup folded, hallucinated dropped
        assert fd_rels[0].kind == "fd"
        assert fd_rels[0].columns == ("Zip", "State")
        assert fd_rels[0].lhs == ("Zip",)
        assert fd_rels[0].rhs == ("State",)

        calc_rels = rels_by_type["row_calculation_inconsistency"]
        assert len(calc_rels) == 1
        assert calc_rels[0].kind == "product"

        # JSONL persistence carries the species key.
        out = tmp_path / "relations.jsonl"
        n = write_relations_jsonl(reports, out)
        assert n == 2
        lines = [json.loads(L) for L in out.read_text().splitlines()]
        assert all("species_key" in L for L in lines)
        assert {L["error_type"] for L in lines} == {
            "fd_violation", "row_calculation_inconsistency",
        }

    def test_empty_response_yields_zero_relations(self, tmp_path: Path) -> None:
        df = _hr_frame()
        digest = build_table_digest(df, source="hr.csv")
        tax = _make_taxonomy()
        cfg = _make_discovery_cfg(tmp_path)
        fake = _FakeLLM(responses={})  # always returns {"relations": []}
        disc = SchemaDiscoverer(
            llm=fake,  # type: ignore[arg-type]
            prompt_config=cfg,
            taxonomy=tax,
            model="fake-model",
            digests={"hr.csv": digest},
            source_frames={"hr.csv": df},
            error_types=list(tax.types),
        )
        reports = disc.run()
        assert all(len(r.relations) == 0 for r in reports)
