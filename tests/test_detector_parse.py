"""Tests for ``src.core.detector``: JSON parsing + per-type detection orchestration.

Pinned by Pass 6c HIGH coverage gap. Uses a stub :class:`LLMClient` so
no real Azure traffic is generated. Coverage:

* ``parse_llm_json`` extracts JSON from raw / fenced / wrapped responses
* ``_coerce_issues`` normalises list / dict / wrapped shapes
* ``PerTypeCleaningDetector`` stamps ``error_type*`` on every issue, runs
  one call per (chunk, type), respects ``concurrency=1`` deterministic
  ordering, propagates LLM errors and parse errors into ``ChunkReport``
* taxonomy ``error_types`` argument is honoured (subset)
* empty error_types list raises
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.core.data_loader import COLUMN_MODE_ALL, CsvChunk
from src.core.detector import (
    ChunkReport,
    PerTypeCleaningDetector,
    _coerce_issues,
    _filter_pure_numeric_typo_issues,
    parse_llm_json,
)
from src.core.error_taxonomy import ErrorTaxonomy, ErrorType
from src.core.llm_client import LLMResponse


# ---------------------------------------------------------------------------
# parse_llm_json
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_empty_response(self) -> None:
        issues, err = parse_llm_json("")
        assert issues == []
        assert err == "empty response"

    def test_bare_array(self) -> None:
        issues, err = parse_llm_json('[{"row":1,"col":"name"}]')
        assert err is None
        assert issues == [{"row": 1, "col": "name"}]

    def test_bare_object_wrapped_in_issues_key(self) -> None:
        # Object with "issues" key — coerced via ``_coerce_issues``.
        issues, err = parse_llm_json('{"issues":[{"row":1},{"row":2}]}')
        assert err is None
        assert issues == [{"row": 1}, {"row": 2}]

    def test_object_with_findings_key(self) -> None:
        issues, err = parse_llm_json('{"findings":[{"x":1}]}')
        assert err is None
        assert issues == [{"x": 1}]

    def test_object_without_known_key_returns_self(self) -> None:
        # A plain dict with no known wrapper key → treated as a single
        # issue dict.
        issues, err = parse_llm_json('{"row":1,"col":"name"}')
        assert err is None
        assert issues == [{"row": 1, "col": "name"}]

    def test_fenced_json_block_extracted(self) -> None:
        text = 'Here are the issues:\n```json\n[{"row":1}]\n```\nDone.'
        issues, err = parse_llm_json(text)
        assert err is None
        assert issues == [{"row": 1}]

    def test_unfenced_json_with_prose_around_recovers(self) -> None:
        # No fence, but the bracket-balanced fallback finds the JSON
        # inside the prose. This is the "LLM forgot the fence"
        # recovery path; without it a single missing fence drops every
        # detection from that call.
        text = 'Sure! [{"row":1}] is the answer.'
        issues, err = parse_llm_json(text)
        assert err is None
        assert issues == [{"row": 1}]

    def test_unfenced_pure_garbage_still_fails(self) -> None:
        # No JSON-shaped substring at all → parse still fails.
        text = "I cannot answer this question."
        issues, err = parse_llm_json(text)
        assert issues == []
        assert err == "failed to parse JSON from response"

    def test_invalid_json_returns_error(self) -> None:
        issues, err = parse_llm_json("{not really json}")
        assert issues == []
        assert err == "failed to parse JSON from response"

    def test_fence_without_language_tag(self) -> None:
        # ``` ... ``` (no "json" hint) still parses.
        text = "```\n[{\"a\":1}]\n```"
        issues, err = parse_llm_json(text)
        assert err is None
        assert issues == [{"a": 1}]

    def test_list_with_non_dict_items_filtered(self) -> None:
        issues, err = parse_llm_json('[{"row":1},"oops",null,{"row":2}]')
        assert err is None
        assert issues == [{"row": 1}, {"row": 2}]


class TestCoerceIssues:
    def test_list_of_dicts(self) -> None:
        assert _coerce_issues([{"a": 1}, {"b": 2}]) == [{"a": 1}, {"b": 2}]

    def test_list_filters_non_dicts(self) -> None:
        assert _coerce_issues([{"a": 1}, "str", 42, None]) == [{"a": 1}]

    def test_dict_with_issues_key(self) -> None:
        assert _coerce_issues({"issues": [{"a": 1}]}) == [{"a": 1}]

    def test_dict_with_errors_key(self) -> None:
        assert _coerce_issues({"errors": [{"a": 1}]}) == [{"a": 1}]

    def test_dict_with_results_key(self) -> None:
        assert _coerce_issues({"results": [{"a": 1}]}) == [{"a": 1}]

    def test_dict_without_known_key(self) -> None:
        assert _coerce_issues({"a": 1}) == [{"a": 1}]

    def test_unsupported_type_returns_empty(self) -> None:
        assert _coerce_issues("hello") == []
        assert _coerce_issues(42) == []
        assert _coerce_issues(None) == []


# ---------------------------------------------------------------------------
# PerTypeCleaningDetector — uses a stub LLMClient
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal stand-in for :class:`LLMClient` honoured by the detector."""

    def __init__(self, *, responses: Optional[List[LLMResponse]] = None) -> None:
        self.endpoints = ["http://stub-1", "http://stub-2"]
        self._responses = list(responses) if responses else []
        self._default = LLMResponse(
            text='[{"row":0,"observed":"x"}]', endpoint=self.endpoints[0],
            model="stub", attempts=1,
        )
        self.calls: List[dict] = []

    def query(self, model, prompt, *, reasoning_effort, reasoning_summary,
              system_prompt):
        self.calls.append({
            "model": model, "prompt": prompt,
            "reasoning_effort": reasoning_effort,
            "reasoning_summary": reasoning_summary,
            "system_prompt": system_prompt,
        })
        if self._responses:
            return self._responses.pop(0)
        return self._default


def _taxonomy(keys: List[str]) -> ErrorTaxonomy:
    types = tuple(
        ErrorType(
            key=k, name=k.replace("_", " ").title(),
            category="Cat",
            description=f"description-of-{k}",
            example="ex", notes="notes",
        )
        for k in keys
    )
    return ErrorTaxonomy(
        system_prompt="SYS",
        user_template=(
            "TYPE={type_key}\nNAME={type_name}\nCAT={type_category}\n"
            "DESC={type_description}\nEX={type_example}\nNOTES={type_notes}\n"
            "HEADERS={headers}\nROWS:\n{rows}\nRANGE={start_row}..{end_row}"
        ),
        types=types,
    )


def _chunk(rows: int = 3) -> CsvChunk:
    df = pd.DataFrame({"name": [f"v{i}" for i in range(rows)]})
    return CsvChunk(
        source=Path("a.csv"),
        start_row=0, end_row=rows,
        headers=["name"],
        frame=df, column_mode=COLUMN_MODE_ALL,
    )


class TestPerTypeCleaningDetectorInit:
    def test_empty_error_types_raises(self) -> None:
        tax = _taxonomy(["typo"])
        # Passing an empty subset → constructor refuses (the taxonomy has
        # types, but the caller restricted the run to none).
        with pytest.raises(ValueError, match="at least one error type"):
            PerTypeCleaningDetector(_StubLLM(), tax, model="m", error_types=[])

    def test_default_concurrency_caps_at_8(self) -> None:
        llm = _StubLLM()
        # 2 endpoints → default concurrency 2 (min(8, max(1, 2))).
        d = PerTypeCleaningDetector(llm, _taxonomy(["typo"]), model="m")
        assert d.concurrency == 2

    def test_explicit_concurrency_honoured(self) -> None:
        llm = _StubLLM()
        d = PerTypeCleaningDetector(
            llm, _taxonomy(["typo"]), model="m", concurrency=1,
        )
        assert d.concurrency == 1

    def test_explicit_zero_or_negative_concurrency_falls_back_to_default(self) -> None:
        llm = _StubLLM()
        d = PerTypeCleaningDetector(
            llm, _taxonomy(["typo"]), model="m", concurrency=0,
        )
        assert d.concurrency == 2  # default for 2 endpoints


class TestPerTypeCleaningDetectorRun:
    def test_one_call_per_chunk_x_type_with_stamping(self) -> None:
        llm = _StubLLM()
        tax = _taxonomy(["typo", "casing"])
        d = PerTypeCleaningDetector(llm, tax, model="gpt-test", concurrency=1)
        reports = d.run([_chunk(), _chunk()])
        # 2 chunks × 2 types = 4 calls.
        assert len(reports) == 4
        assert len(llm.calls) == 4
        # Each report has the right type metadata stamped.
        keys = [r.error_type for r in reports]
        # concurrency=1 preserves chunk-major order: c0(typo), c0(casing), c1(typo), c1(casing).
        assert keys == ["typo", "casing", "typo", "casing"]
        # Issues inherit error_type tags from their report.
        for r in reports:
            for issue in r.issues:
                assert issue["error_type"] == r.error_type
                assert issue["error_type_name"] == r.error_type_name
                assert issue["error_type_category"] == r.error_type_category
                assert issue["category"] == r.error_type_category.lower()

    def test_subset_of_error_types(self) -> None:
        llm = _StubLLM()
        tax = _taxonomy(["typo", "casing", "spacing"])
        d = PerTypeCleaningDetector(
            llm, tax, model="m", concurrency=1,
            error_types=tax.select(["casing"]),
        )
        reports = d.run([_chunk()])
        assert len(reports) == 1
        assert reports[0].error_type == "casing"

    def test_propagates_llm_error(self) -> None:
        llm = _StubLLM(responses=[
            LLMResponse(text="", endpoint="http://stub", model="m",
                        attempts=3, error="HTTP 500"),
        ])
        tax = _taxonomy(["typo"])
        d = PerTypeCleaningDetector(llm, tax, model="m", concurrency=1)
        reports = d.run([_chunk()])
        assert len(reports) == 1
        r = reports[0]
        assert r.error == "HTTP 500"
        assert r.attempts == 3
        # Errored response → no parse attempt → no issues stamped.
        assert r.issues == []
        assert r.parse_error is None

    def test_propagates_parse_error(self) -> None:
        llm = _StubLLM(responses=[
            LLMResponse(text="not json at all", endpoint="http://stub",
                        model="m", attempts=1),
        ])
        tax = _taxonomy(["typo"])
        d = PerTypeCleaningDetector(llm, tax, model="m", concurrency=1)
        reports = d.run([_chunk()])
        r = reports[0]
        assert r.error is None
        assert r.parse_error == "failed to parse JSON from response"
        assert r.issues == []

    def test_prompt_uses_taxonomy_template(self) -> None:
        llm = _StubLLM()
        tax = _taxonomy(["typo"])
        d = PerTypeCleaningDetector(llm, tax, model="m", concurrency=1)
        d.run([_chunk(rows=2)])
        # First call's prompt must reflect the taxonomy substitution.
        call = llm.calls[0]
        assert "TYPE=typo" in call["prompt"]
        assert "NAME=Typo" in call["prompt"]
        assert "DESC=description-of-typo" in call["prompt"]
        assert call["system_prompt"] == "SYS"

    def test_filters_pure_numeric_typo_findings(self) -> None:
        llm = _StubLLM(
            responses=[
                LLMResponse(
                    text=json.dumps(
                        {
                            "issues": [
                                {
                                    "row": 0,
                                    "columns": ["value"],
                                    "observed": "6",
                                    "corrected_value": {"value": "6.00"},
                                },
                                {
                                    "row": 1,
                                    "columns": ["value"],
                                    "observed": "555-123-4567",
                                    "corrected_value": {"value": "5551234567"},
                                },
                                {
                                    "row": 2,
                                    "columns": ["value"],
                                    "observed": "199,5",
                                    "corrected_value": {"value": "199,50"},
                                },
                                {
                                    "row": 3,
                                    "columns": ["value"],
                                    "observed": "1133,21",
                                    "corrected_value": {"value": "1133, 21"},
                                },
                            ]
                        }
                    ),
                    endpoint="http://stub",
                    model="m",
                    attempts=1,
                )
            ]
        )
        chunk = CsvChunk(
            source=Path("numbers.csv"),
            start_row=0,
            end_row=4,
            headers=["value"],
            frame=pd.DataFrame({"value": ["6", "555-123-4567", "199,5", "1133,21"]}),
            column_mode="single",
        )
        detector = PerTypeCleaningDetector(
            llm,
            _taxonomy(["typo"]),
            model="m",
            concurrency=1,
        )

        report = detector.run([chunk])[0]

        assert [issue["row"] for issue in report.issues] == [1, 3]
        assert report.metadata["filtered_pure_numeric_typo_issues"] == 2

    def test_numeric_findings_remain_for_non_typo_type(self) -> None:
        issues = [{"row": 0, "columns": ["value"], "observed": "6"}]
        chunk = CsvChunk(
            source=Path("numbers.csv"),
            start_row=0,
            end_row=1,
            headers=["value"],
            frame=pd.DataFrame({"value": ["6"]}),
            column_mode="single",
        )
        kept, filtered = _filter_pure_numeric_typo_issues(
            issues,
            chunk,
            _taxonomy(["numeric_outlier"]).types[0],
        )

        assert kept == issues
        assert filtered == 0

    def test_concurrency_path_preserves_position_order(self) -> None:
        # Concurrency > 1 shouldn't scramble the final order — the
        # detector reassembles by ``position``.
        llm = _StubLLM()
        tax = _taxonomy(["typo", "casing"])
        d = PerTypeCleaningDetector(llm, tax, model="m", concurrency=4)
        reports = d.run([_chunk(), _chunk(), _chunk()])
        assert len(reports) == 6
        # Stable ordering: chunk-major then type-major.
        seq = [(r.start_row, r.error_type) for r in reports]
        # All three chunks share start_row=0 in our helper, so we use
        # the call order recorded on the stub instead. Since position
        # ordering is by (chunk_idx, type_idx), the sequence of error_type
        # values across the 6 reports must be ['typo','casing']*3.
        assert [r.error_type for r in reports] == ["typo", "casing"] * 3


def test_coerce_issues_clean_sentinel_object_yields_no_issues():
    """Regression: an LLM 'nothing wrong' object must NOT become a
    phantom single-issue cluster."""
    assert _coerce_issues({"status": "clean"}) == []
    assert _coerce_issues({"no_issues_found": True}) == []
    assert _coerce_issues({"ok": True, "summary": "looks fine"}) == []
    issues, err = parse_llm_json('{"status": "clean"}')
    assert issues == []


def test_coerce_issues_bare_single_issue_object_is_kept():
    """A bare object that carries issue-like fields IS a real issue."""
    payload = {"row": 3, "column": "Name", "explanation": "typo"}
    assert _coerce_issues(payload) == [payload]
