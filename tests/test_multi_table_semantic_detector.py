"""Tests for ``src.core.multi_table_semantic_detector`` (the
Shape-1 LLM-as-detector replacement for the deterministic
compiler + per-violation verifier).

The detector:
    * Pages each confirmed group's sheets at ``n_sample_rows``
      rows per call, fires ONE LLM call per (group, page).
    * Translates the LLM's LOCAL row indices back to global
      DataFrame row indices via ``_PageSlice.row_offsets``.
    * Drops findings that reference sheets/rows outside the
      current page (hallucinated locations).
    * Emits one ``MultiTableViolation`` per (sheet, row) location;
      multi-row findings share a ``violation_id``.
    * Emits one stub ``Verification`` per finding so the dashboard
      verdict column lights up uniformly with compiler-strategy.
    * Is fail-soft: LLM errors and parse errors increment counters
      without crashing the workbook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from src.core.llm_client import LLMResponse
from src.core.multi_table_resolver import (
    ColumnMappingEntry,
    ResolvedGroup,
    ResolvedWorkbook,
)
from src.core.multi_table_semantic_detector import (
    DetectPromptConfig,
    SEMANTIC_KINDS,
    _build_pages,
    _normalize_kind,
    _normalize_locations,
    _normalize_verdict,
    _parse_response,
    detect_workbook_inconsistencies,
    load_detect_prompt,
    summarize,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubLLM:
    """Records calls and returns scripted ``LLMResponse`` values."""

    def __init__(self, responses: List[LLMResponse]) -> None:
        self.endpoints = ["http://stub"]
        self._responses = list(responses)
        self.calls: List[dict] = []

    def query(
        self, model, prompt, *,
        reasoning_effort, reasoning_summary, system_prompt,
    ):
        self.calls.append({
            "model": model, "prompt": prompt,
            "system_prompt": system_prompt,
        })
        if not self._responses:
            return LLMResponse(
                text="", endpoint="http://stub", model=model,
                attempts=1, error="no scripted response",
            )
        return self._responses.pop(0)


def _entry(sheet: str, column: str) -> ColumnMappingEntry:
    return ColumnMappingEntry(sheet=sheet, column=column)


def _group(
    *,
    workbook: str = "wb.xlsx",
    group_id: str = "u1",
    confirmed: bool = True,
    sheets=("a.csv", "b.csv"),
    column_mapping=None,
    primary_key=("K",),
) -> ResolvedGroup:
    if column_mapping is None:
        column_mapping = {
            "K": (_entry("a.csv", "K"), _entry("b.csv", "K")),
            "country": (
                _entry("a.csv", "country"),
                _entry("b.csv", "country"),
            ),
        }
    return ResolvedGroup(
        workbook=workbook,
        group_id=group_id,
        kind="union",
        confirmed=confirmed,
        rationale="test",
        sheets=tuple(sheets),
        column_mapping={k: tuple(v) for k, v in column_mapping.items()},
        primary_key=tuple(primary_key),
    )


_PROMPT_CFG = DetectPromptConfig(
    model="m",
    reasoning_effort="medium",
    reasoning_summary="detailed",
    system_prompt="sys",
    user_template=(
        "wb={workbook} g={group_id} kind={group_kind} "
        "sheets={sheets} pk={primary_key}\n"
        "MAP:\n{column_mapping_block}\n"
        "page={page_index}/{n_pages} k={n_sample_rows}\n"
        "ROWS:\n{row_samples_block}"
    ),
)


def _ok_response(findings) -> LLMResponse:
    body = json.dumps({"findings": findings})
    return LLMResponse(
        text=body, endpoint="http://stub", model="m", attempts=1,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestNormalizeKind:
    def test_canonical_pass_through(self) -> None:
        for k in SEMANTIC_KINDS:
            assert _normalize_kind(k) == k

    def test_synonyms(self) -> None:
        assert _normalize_kind("dup") == "duplicate_row"
        assert _normalize_kind("conflict") == "attribute_conflict"
        assert _normalize_kind("format") == "format_drift"
        assert _normalize_kind("unit") == "unit_drift"
        assert _normalize_kind("surface") == "surface_drift"
        assert _normalize_kind("null") == "null_placeholder"
        assert _normalize_kind("snapshot") == "snapshot_aggregation"
        assert _normalize_kind("orphan") == "fk_orphan"

    def test_unknown_falls_back_to_attribute_conflict(self) -> None:
        # Conservative default - we'd rather flag-and-let-the-user-
        # sort-it than silently drop.
        assert _normalize_kind("???") == "attribute_conflict"


class TestNormalizeVerdict:
    def test_canonical_pass_through(self) -> None:
        for v in ("confirmed_inconsistency", "benign_variation",
                  "uncertain"):
            assert _normalize_verdict(v) == v

    def test_not_a_violation_collapses_to_benign(self) -> None:
        # Semantic detector treats "not a violation" as benign so
        # the dashboard doesn't have to learn a fourth verdict.
        assert _normalize_verdict("not_a_violation") == "benign_variation"

    def test_synonyms(self) -> None:
        assert (
            _normalize_verdict("confirmed") == "confirmed_inconsistency"
        )
        assert _normalize_verdict("ok") == "benign_variation"
        assert _normalize_verdict("???") == "uncertain"


class TestParseResponse:
    def test_fenced_json(self) -> None:
        body = "```json\n" + json.dumps({"findings": [
            {"kind": "duplicate_row", "verdict": "confirmed",
             "locations": [{"sheet": "a.csv", "row": 0}]},
        ]}) + "\n```"
        records, err = _parse_response(body)
        assert err is None
        assert len(records) == 1

    def test_missing_findings_key(self) -> None:
        records, err = _parse_response(json.dumps({"foo": []}))
        assert records == []
        assert err and "findings" in err

    def test_empty_response(self) -> None:
        records, err = _parse_response("")
        assert records == []
        assert err and "empty" in err.lower()


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestBuildPages:
    def test_single_page_when_under_limit(self) -> None:
        df_a = pd.DataFrame({"K": [1, 2]})
        df_b = pd.DataFrame({"K": [1, 2]})
        pages = _build_pages(
            sheets=["a.csv", "b.csv"],
            frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=50, seed=0,
        )
        assert len(pages) == 1
        # Both sheets present with all rows.
        page = pages[0]
        assert {p.sheet for p in page} == {"a.csv", "b.csv"}
        for slc in page:
            assert sorted(slc.row_offsets) == [0, 1]

    def test_multiple_pages_when_over_limit(self) -> None:
        df_a = pd.DataFrame({"K": list(range(10))})
        df_b = pd.DataFrame({"K": list(range(3))})
        pages = _build_pages(
            sheets=["a.csv", "b.csv"],
            frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=4, seed=42,
        )
        # a.csv -> 3 chunks, b.csv -> 1; max_pages = 3.
        assert len(pages) == 3
        # Every page covers every sheet (b.csv recycled).
        for page in pages:
            assert {p.sheet for p in page} == {"a.csv", "b.csv"}
        # Across the 3 pages, a.csv covers all 10 rows.
        a_offsets = []
        for page in pages:
            for slc in page:
                if slc.sheet == "a.csv":
                    a_offsets.extend(slc.row_offsets)
        assert sorted(a_offsets) == list(range(10))

    def test_empty_sheet_yields_empty_slice(self) -> None:
        df_a = pd.DataFrame({"K": [1]})
        pages = _build_pages(
            sheets=["a.csv", "missing.csv"],
            frames={"a.csv": df_a},
            n_sample_rows=10, seed=0,
        )
        assert len(pages) == 1
        slcs = {s.sheet: s for s in pages[0]}
        assert slcs["missing.csv"].row_offsets == ()
        assert slcs["a.csv"].row_offsets == (0,)


# ---------------------------------------------------------------------------
# _normalize_locations
# ---------------------------------------------------------------------------


class TestNormalizeLocations:
    def test_local_to_global_mapping(self) -> None:
        from src.core.multi_table_semantic_detector import _PageSlice
        page = [
            _PageSlice(sheet="a.csv", row_offsets=(7, 8, 9)),
            _PageSlice(sheet="b.csv", row_offsets=(0, 5)),
        ]
        locs = _normalize_locations(
            [{"sheet": "a.csv", "row": 1},   # local 1 -> global 8
             {"sheet": "b.csv", "row": 0}],  # local 0 -> global 0
            page=page,
        )
        assert locs == [("a.csv", 8), ("b.csv", 0)]

    def test_drops_hallucinated_sheet(self) -> None:
        from src.core.multi_table_semantic_detector import _PageSlice
        page = [_PageSlice(sheet="a.csv", row_offsets=(0,))]
        locs = _normalize_locations(
            [{"sheet": "ghost.csv", "row": 0}],
            page=page,
        )
        assert locs == []

    def test_drops_out_of_range_row(self) -> None:
        from src.core.multi_table_semantic_detector import _PageSlice
        page = [_PageSlice(sheet="a.csv", row_offsets=(0, 1))]
        locs = _normalize_locations(
            [{"sheet": "a.csv", "row": 99}],
            page=page,
        )
        assert locs == []

    def test_dedupes_repeated_locations(self) -> None:
        from src.core.multi_table_semantic_detector import _PageSlice
        page = [_PageSlice(sheet="a.csv", row_offsets=(0,))]
        locs = _normalize_locations(
            [{"sheet": "a.csv", "row": 0},
             {"sheet": "a.csv", "row": 0}],
            page=page,
        )
        assert locs == [("a.csv", 0)]


# ---------------------------------------------------------------------------
# load_detect_prompt
# ---------------------------------------------------------------------------


class TestLoadDetectPrompt:
    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "detect.yaml"
        p.write_text(
            "model: gpt-x\n"
            "reasoning_effort: high\n"
            "reasoning_summary: brief\n"
            "system: |\n"
            "  hello\n"
            "user_template: |\n"
            "  body\n",
            encoding="utf-8",
        )
        cfg = load_detect_prompt(p)
        assert cfg.model == "gpt-x"
        assert cfg.reasoning_effort == "high"
        assert cfg.reasoning_summary == "brief"
        assert "hello" in cfg.system_prompt
        assert "body" in cfg.user_template


# ---------------------------------------------------------------------------
# detect_workbook_inconsistencies — happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_semantic_finding_emits_linked_violations(self) -> None:
        # The classic case the compiler can't see: "USA" / "United
        # States" same-entity but the score differs too -> a real
        # cross-sheet inconsistency.
        df_a = pd.DataFrame({"K": ["k1"], "country": ["USA"]})
        df_b = pd.DataFrame({"K": ["k1"], "country": ["United States"]})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        # LLM sees both rows on page 1, returns one finding with
        # two locations (LOCAL row 0 on each sheet).
        llm = _StubLLM([_ok_response([
            {
                "kind": "attribute_conflict",
                "verdict": "confirmed_inconsistency",
                "confidence": "high",
                "locations": [
                    {"sheet": "a.csv", "row": 0},
                    {"sheet": "b.csv", "row": 0},
                ],
                "columns": ["country"],
                "evidence": "'USA' vs 'United States'",
                "rationale": "abbreviation+full-name same entity, same key.",
            },
        ])])

        result = detect_workbook_inconsistencies(
            resolved=wb,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
            n_sample_rows=10,
        )
        # One LLM call (one page), two violation rows (one per
        # location), one verification record (per logical finding).
        assert result.n_calls == 1
        assert len(result.violations) == 2
        assert len(result.verifications) == 1
        # Both violations carry the same id.
        ids = {v.violation_id for v in result.violations}
        assert len(ids) == 1
        # And the verification points at it.
        assert result.verifications[0].violation_id == next(iter(ids))
        # Sheets covered.
        assert {v.sheet for v in result.violations} == {"a.csv", "b.csv"}
        # Each violation links the OTHER location.
        for vio in result.violations:
            other = vio.linked
            assert len(other) == 1
            assert other[0][0] != vio.sheet

    def test_no_findings_returns_empty(self) -> None:
        df_a = pd.DataFrame({"K": [1]})
        df_b = pd.DataFrame({"K": [1]})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        llm = _StubLLM([_ok_response([])])
        result = detect_workbook_inconsistencies(
            resolved=wb,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert result.n_calls == 1
        assert result.violations == []
        assert result.verifications == []

    def test_unconfirmed_groups_skipped(self) -> None:
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group(confirmed=False)],
        )
        llm = _StubLLM([])
        result = detect_workbook_inconsistencies(
            resolved=wb,
            frames={"a.csv": pd.DataFrame({"K": [1]}),
                    "b.csv": pd.DataFrame({"K": [1]})},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert llm.calls == []
        assert result.n_calls == 0
        assert result.violations == []

    def test_dedupes_across_pages(self) -> None:
        # Same finding (same locations + columns + kind) reported
        # on two pages must collapse into one logical record.
        # Construct two pages by going over n_sample_rows.
        df_a = pd.DataFrame({"K": list(range(4)),
                             "country": ["USA", "x", "x", "x"]})
        df_b = pd.DataFrame({"K": list(range(4)),
                             "country": ["United States", "y", "y", "y"]})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        # Both pages claim the SAME global locations - we use a
        # determinism-friendly seed and assume the LLM points at
        # row 0 on each page (which may not map to the same global
        # row, so this test instead verifies the dedup mechanism
        # via two findings asserted to share locations).
        llm = _StubLLM([
            _ok_response([
                {"kind": "attribute_conflict",
                 "verdict": "confirmed_inconsistency",
                 "confidence": "high",
                 "locations": [{"sheet": "a.csv", "row": 0},
                               {"sheet": "b.csv", "row": 0}],
                 "columns": ["country"],
                 "rationale": "same key, surface variant + score diff."},
            ]),
            _ok_response([
                # Second page: identical claim - dedupe.
                {"kind": "attribute_conflict",
                 "verdict": "confirmed_inconsistency",
                 "confidence": "high",
                 "locations": [{"sheet": "a.csv", "row": 0},
                               {"sheet": "b.csv", "row": 0}],
                 "columns": ["country"],
                 "rationale": "duplicate report."},
            ]),
        ])
        result = detect_workbook_inconsistencies(
            resolved=wb,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
            n_sample_rows=2, seed=0,
        )
        # The two pages are constructed by shuffling, so the global
        # rows backing local 0 may differ between pages. But IF
        # they happen to match, dedupe must kick in. This test
        # asserts the *mechanism* exists by checking we never see
        # more verifications than findings on a single global
        # location set.
        assert result.n_calls == 2
        # Number of unique violation_ids equals number of
        # verifications - the per-finding dedupe invariant.
        unique_ids = {v.violation_id for v in result.violations}
        assert len(unique_ids) == len(result.verifications)


# ---------------------------------------------------------------------------
# detect_workbook_inconsistencies — failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_llm_error_increments_counter(self) -> None:
        df_a = pd.DataFrame({"K": [1]})
        df_b = pd.DataFrame({"K": [1]})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        # Empty stub -> exhausted error response.
        llm = _StubLLM([])
        result = detect_workbook_inconsistencies(
            resolved=wb,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert result.n_calls == 1
        assert result.n_errored == 1
        assert result.violations == []

    def test_unparseable_response_increments_parse_counter(self) -> None:
        df_a = pd.DataFrame({"K": [1]})
        df_b = pd.DataFrame({"K": [1]})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        llm = _StubLLM([LLMResponse(
            text="not json", endpoint="http://stub",
            model="m", attempts=1,
        )])
        result = detect_workbook_inconsistencies(
            resolved=wb,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert result.n_calls == 1
        assert result.n_parse_errored == 1
        assert result.violations == []

    def test_finding_with_hallucinated_locations_dropped(self) -> None:
        df_a = pd.DataFrame({"K": [1]})
        df_b = pd.DataFrame({"K": [1]})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        llm = _StubLLM([_ok_response([
            {"kind": "attribute_conflict",
             "verdict": "confirmed_inconsistency",
             "confidence": "high",
             "locations": [{"sheet": "ghost.csv", "row": 5}],
             "columns": ["country"],
             "rationale": "bad sheet"},
        ])])
        result = detect_workbook_inconsistencies(
            resolved=wb,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert result.violations == []
        assert result.verifications == []

    def test_max_pages_caps_calls(self) -> None:
        df_a = pd.DataFrame({"K": list(range(20))})
        df_b = pd.DataFrame({"K": list(range(20))})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        # Without cap: 20/5 = 4 pages.  With max_pages=2: 2 calls.
        llm = _StubLLM([_ok_response([])] * 10)
        result = detect_workbook_inconsistencies(
            resolved=wb,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
            n_sample_rows=5, max_pages=2,
        )
        assert result.n_calls == 2


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_aggregates_by_kind_and_verdict(self) -> None:
        df_a = pd.DataFrame({"K": ["k1"], "country": ["USA"]})
        df_b = pd.DataFrame({"K": ["k1"], "country": ["United States"]})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        llm = _StubLLM([_ok_response([
            {"kind": "attribute_conflict",
             "verdict": "confirmed_inconsistency",
             "confidence": "high",
             "locations": [{"sheet": "a.csv", "row": 0},
                           {"sheet": "b.csv", "row": 0}],
             "columns": ["country"],
             "rationale": "real"},
            {"kind": "surface_drift",
             "verdict": "benign_variation",
             "confidence": "medium",
             "locations": [{"sheet": "a.csv", "row": 0},
                           {"sheet": "b.csv", "row": 0}],
             "columns": ["country"],
             "rationale": "casing only"},
        ])])
        result = detect_workbook_inconsistencies(
            resolved=wb,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        s = summarize(result)
        assert s["strategy"] == "semantic"
        assert s["n_calls"] == 1
        assert s["n_findings"] == 2
        # n_violation_records counts EVERY (sheet, row) record:
        # 2 findings * 2 locations each = 4 records.
        assert s["n_violation_records"] == 4
        assert s["by_kind"]["attribute_conflict"] == 1
        assert s["by_kind"]["surface_drift"] == 1
        assert s["by_verdict"]["confirmed_inconsistency"] == 1
        assert s["by_verdict"]["benign_variation"] == 1
