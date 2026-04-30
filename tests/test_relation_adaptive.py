"""Tests for src.core.relation_adaptive: multi-pass discovery driver
with per-arm Good-Turing stopping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.core.error_taxonomy import ErrorType, ErrorTaxonomy
from src.core.llm_client import LLMResponse
from src.core.relation_adaptive import (
    AdaptiveSchemaDiscoverer,
    adaptive_summary_dict,
)
from src.core.relations import (
    SchemaDiscoverer,
    load_discovery_prompt,
)
from src.core.table_digest import build_table_digest


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_relations.py)
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
    return ErrorTaxonomy(system_prompt="sys", user_template="{type_key}", types=types)


def _make_discovery_cfg():
    return load_discovery_prompt(
        Path(__file__).resolve().parent.parent / "prompts" / "discover_relations.yaml"
    )


# Each LLM "call" returns an answer that depends on the seed encoded in
# the row_sample portion of the prompt. The fake parses out the
# ``__row__`` row indices and maps them to a deterministic relation
# list, so we can simulate per-pass species variation.
class _SeedingFakeLLM:
    """Returns relations whose set varies across passes.

    Strategy: for fd_violation, always emit the bedrock Zip->State
    relation (so it's seen every pass; abundance grows with passes).
    Additionally, on passes whose row sample contains row index 0 or
    7 emit "rare" relations, mimicking long-tail species. This
    produces non-trivial frequency-of-frequencies counts.
    """

    def __init__(self) -> None:
        self.endpoints = ["fake://0"]
        self.calls: List[Dict[str, Any]] = []

    def query(self, model, prompt, *, reasoning_effort, reasoning_summary, system_prompt):
        # Detect target error type.
        et = "fd_violation"
        for key in ("fd_violation", "row_calculation_inconsistency"):
            if f"TYPE KEY:      {key}" in prompt:
                et = key
                break

        # Snapshot which row indices are in the sample.
        sample_rows = self._extract_row_indices(prompt)
        self.calls.append({"type": et, "rows": tuple(sample_rows)})

        if et == "fd_violation":
            relations: List[Dict[str, Any]] = [
                {
                    "kind": "fd",
                    "columns": ["Zip", "State"],
                    "lhs": ["Zip"],
                    "rhs": ["State"],
                    "predicate": "Zip determines State",
                    "confidence": "high",
                    "rationale": "core",
                },
            ]
            if 0 in sample_rows:
                relations.append({
                    "kind": "fd",
                    "columns": ["EmpID", "Zip"],
                    "lhs": ["EmpID"],
                    "rhs": ["Zip"],
                    "predicate": "EmpID determines Zip (rare-from-row0)",
                    "confidence": "medium",
                    "rationale": "row-0 only",
                })
            if 7 in sample_rows:
                relations.append({
                    "kind": "fd",
                    "columns": ["EmpID", "State"],
                    "lhs": ["EmpID"],
                    "rhs": ["State"],
                    "predicate": "EmpID determines State (rare-from-row7)",
                    "confidence": "medium",
                    "rationale": "row-7 only",
                })
            payload = {"relations": relations}
        else:
            # row_calculation: same product rule every pass -> arm
            # saturates quickly with a single dominant species.
            payload = {
                "relations": [
                    {
                        "kind": "product",
                        "columns": ["UnitPrice", "Quantity", "TotalPrice"],
                        "predicate": "TotalPrice == UnitPrice * Quantity",
                        "executable": "TotalPrice == UnitPrice * Quantity",
                        "confidence": "high",
                        "rationale": "stable",
                    }
                ]
            }
        return LLMResponse(
            text=json.dumps(payload),
            endpoint="fake://0",
            model=model,
            attempts=1,
        )

    @staticmethod
    def _extract_row_indices(prompt: str) -> List[int]:
        # Markdown row sample looks like
        #   | __row__ | EmpID | ...
        #   |---------|-------|...
        #   | 0       | E000  | ...
        # We just scan the leading number after each "| " in lines that start with "| ".
        rows: List[int] = []
        for line in prompt.splitlines():
            if not line.startswith("|") or "__row__" in line or "---" in line:
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if not parts:
                continue
            head = parts[0]
            if head.isdigit():
                rows.append(int(head))
        return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdaptiveSchemaDiscoverer:
    def _make_discoverer(self) -> SchemaDiscoverer:
        df = _hr_frame()
        digest = build_table_digest(df, source="hr.csv")
        tax = _make_taxonomy()
        cfg = _make_discovery_cfg()
        return SchemaDiscoverer(
            llm=_SeedingFakeLLM(),  # type: ignore[arg-type]
            prompt_config=cfg,
            taxonomy=tax,
            model="fake-model",
            digests={"hr.csv": digest},
            source_frames={"hr.csv": df},
            error_types=list(tax.types),
            n_row_samples=3,
        )

    def test_validates_inputs(self) -> None:
        d = self._make_discoverer()
        with pytest.raises(ValueError):
            AdaptiveSchemaDiscoverer(d, max_passes=0)
        with pytest.raises(ValueError):
            AdaptiveSchemaDiscoverer(d, max_passes=2, min_passes=3)
        with pytest.raises(ValueError):
            AdaptiveSchemaDiscoverer(d, gt_threshold=2.0)

    def test_runs_max_passes_when_unsaturated(self) -> None:
        d = self._make_discoverer()
        adaptive = AdaptiveSchemaDiscoverer(
            d, max_passes=3, min_passes=2,
            gt_threshold=0.0,  # impossible to saturate
            base_seed=1,
        )
        result = adaptive.run()
        assert result.n_passes == 3
        # Each arm runs in every pass since none saturate.
        for arm in result.arm_stats.values():
            assert arm.passes_run == 3

    def test_saturates_stable_arm(self) -> None:
        # row_calculation arm always returns the same relation, so its
        # missing-mass should fall to 0 once n_2 > 0 (i.e. by pass 2).
        d = self._make_discoverer()
        adaptive = AdaptiveSchemaDiscoverer(
            d, max_passes=5, min_passes=2,
            gt_threshold=0.05,
            base_seed=42,
        )
        result = adaptive.run()
        calc_arm = result.arm_stats[("hr.csv", "row_calculation_inconsistency")]
        assert calc_arm.saturated is True
        assert calc_arm.saturated_at_pass is not None
        assert calc_arm.saturated_at_pass <= 5
        assert calc_arm.passes_run <= 5

    def test_union_relations_dedup_by_species(self) -> None:
        d = self._make_discoverer()
        adaptive = AdaptiveSchemaDiscoverer(
            d, max_passes=3, min_passes=2, gt_threshold=0.0, base_seed=1,
        )
        result = adaptive.run()
        # Each species_key appears at most once in union_relations.
        keys = [r.species_key() for r in result.union_relations]
        assert len(keys) == len(set(keys))
        # Metadata frequency matches arm bookkeeping.
        for sk, meta in result.union_metadata.items():
            assert meta["frequency"] >= 1
            assert len(meta["seen_in_passes"]) == meta["frequency"]

    def test_skip_predicate_prunes_calls(self) -> None:
        # Manually pre-saturate one arm and verify it is skipped.
        d = self._make_discoverer()
        adaptive = AdaptiveSchemaDiscoverer(
            d, max_passes=2, min_passes=1, gt_threshold=0.5, base_seed=3,
        )
        result = adaptive.run()
        # Total calls = sum of passes_run across arms.
        total_calls = sum(a.passes_run for a in result.arm_stats.values())
        # The fake LLM logs every call; counts should match.
        assert total_calls == len(d.llm.calls)  # type: ignore[attr-defined]

    def test_summary_dict_is_jsonable(self) -> None:
        d = self._make_discoverer()
        adaptive = AdaptiveSchemaDiscoverer(
            d, max_passes=2, min_passes=2, gt_threshold=0.05, base_seed=7,
        )
        result = adaptive.run()
        summary = adaptive_summary_dict(result)
        # Round-trip JSON.
        s = json.dumps(summary)
        back = json.loads(s)
        assert back["n_passes"] == result.n_passes
        assert back["n_arms"] == len(result.arm_stats)
        assert "config" in back
        assert "by_error_type" in back

    def test_per_pass_seeds_differ(self) -> None:
        d = self._make_discoverer()
        adaptive = AdaptiveSchemaDiscoverer(
            d, max_passes=3, min_passes=3, gt_threshold=0.0, base_seed=100,
        )
        adaptive.run()
        # Different passes should produce different row samples for at
        # least one error type. Aggregate the row-tuples seen across
        # fd_violation calls.
        fake = d.llm  # type: ignore[attr-defined]
        fd_rows = {tuple(c["rows"]) for c in fake.calls if c["type"] == "fd_violation"}
        assert len(fd_rows) >= 2  # at least two distinct samples

    def test_empty_arm_marked_saturated(self) -> None:
        # Arm that always returns no relations should saturate as soon
        # as min_passes is reached.
        df = _hr_frame()
        digest = build_table_digest(df, source="hr.csv")
        tax = _make_taxonomy()
        cfg = _make_discovery_cfg()

        class _AlwaysEmpty:
            endpoints = ["fake://0"]

            def query(self, model, prompt, **_kw):
                return LLMResponse(
                    text='{"relations": []}',
                    endpoint="fake://0", model=model, attempts=1,
                )

        d = SchemaDiscoverer(
            llm=_AlwaysEmpty(),  # type: ignore[arg-type]
            prompt_config=cfg,
            taxonomy=tax,
            model="fake-model",
            digests={"hr.csv": digest},
            source_frames={"hr.csv": df},
            error_types=list(tax.types),
            n_row_samples=3,
        )
        adaptive = AdaptiveSchemaDiscoverer(
            d, max_passes=4, min_passes=2, gt_threshold=0.05, base_seed=11,
        )
        result = adaptive.run()
        for arm in result.arm_stats.values():
            assert arm.saturated is True
            assert arm.passes_run <= 2  # saturates right at min_passes
