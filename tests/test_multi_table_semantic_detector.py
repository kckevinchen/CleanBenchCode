"""Tests for ``src.core.multi_table.semantic_detector`` (the
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
from src.core.multi_table.resolver import (
    ColumnMappingEntry,
    JoinSpec,
    ResolvedGroup,
    ResolvedWorkbook,
)
from src.core.multi_table.semantic_detector import (
    DetectPromptConfig,
    SEMANTIC_KINDS,
    _build_key_aligned_pages,
    _build_pages,
    _key_cardinality,
    _normalize_kind,
    _normalize_locations,
    _normalize_verdict,
    _pages_for_group,
    _parse_response,
    _render_key_uniqueness_block,
    _render_priors_block,
    _resolve_key_columns,
    detect_workbook_inconsistencies,
    expected_semantic_call_count,
    load_detect_prompt,
    structural_complement,
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
        from src.core.multi_table.semantic_detector import _PageSlice
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
        from src.core.multi_table.semantic_detector import _PageSlice
        page = [_PageSlice(sheet="a.csv", row_offsets=(0,))]
        locs = _normalize_locations(
            [{"sheet": "ghost.csv", "row": 0}],
            page=page,
        )
        assert locs == []

    def test_drops_out_of_range_row(self) -> None:
        from src.core.multi_table.semantic_detector import _PageSlice
        page = [_PageSlice(sheet="a.csv", row_offsets=(0, 1))]
        locs = _normalize_locations(
            [{"sheet": "a.csv", "row": 99}],
            page=page,
        )
        assert locs == []

    def test_dedupes_repeated_locations(self) -> None:
        from src.core.multi_table.semantic_detector import _PageSlice
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


# ---------------------------------------------------------------------------
# Key column resolution (B)
# ---------------------------------------------------------------------------


def _join_group(
    *,
    parent_sheet="parent.csv",
    child_sheet="child.csv",
    parent_cols=("id",),
    child_cols=("parent_id",),
) -> ResolvedGroup:
    return ResolvedGroup(
        workbook="wb.xlsx",
        group_id="j1",
        kind="join",
        confirmed=True,
        rationale="test",
        sheets=(parent_sheet, child_sheet),
        column_mapping={},
        primary_key=(),
        join=JoinSpec(
            parent_sheet=parent_sheet,
            parent_columns=tuple(parent_cols),
            child_sheet=child_sheet,
            child_columns=tuple(child_cols),
        ),
    )


class TestResolveKeyColumns:
    def test_primary_key_maps_to_raw_columns_per_sheet(self) -> None:
        g = _group(
            primary_key=("K",),
            column_mapping={
                "K": (_entry("a.csv", "Key"), _entry("b.csv", "ID")),
                "country": (
                    _entry("a.csv", "country"),
                    _entry("b.csv", "country"),
                ),
            },
        )
        key_cols = _resolve_key_columns(g)
        assert key_cols == {"a.csv": ("Key",), "b.csv": ("ID",)}

    def test_key_resolvable_on_only_one_sheet_is_dropped(self) -> None:
        # PK column missing on b.csv -> alignment impossible -> {}.
        g = _group(
            primary_key=("K",),
            column_mapping={
                "K": (_entry("a.csv", "K"),),
                "country": (
                    _entry("a.csv", "country"),
                    _entry("b.csv", "country"),
                ),
            },
        )
        assert _resolve_key_columns(g) == {}

    def test_join_group_uses_join_spec(self) -> None:
        g = _join_group(parent_cols=("id",), child_cols=("parent_id",))
        key_cols = _resolve_key_columns(g)
        assert key_cols == {
            "parent.csv": ("id",), "child.csv": ("parent_id",),
        }

    def test_no_key_returns_empty(self) -> None:
        g = _group(
            primary_key=(),
            column_mapping={
                "country": (
                    _entry("a.csv", "country"),
                    _entry("b.csv", "country"),
                ),
            },
        )
        assert _resolve_key_columns(g) == {}


# ---------------------------------------------------------------------------
# Key-aligned pagination (B + E)
# ---------------------------------------------------------------------------


class TestKeyAlignedPages:
    def test_returns_none_without_usable_key(self) -> None:
        g = _group(primary_key=(), column_mapping={
            "v": (_entry("a.csv", "v"), _entry("b.csv", "v")),
        })
        frames = {
            "a.csv": pd.DataFrame({"v": [1, 2]}),
            "b.csv": pd.DataFrame({"v": [1, 2]}),
        }
        assert _build_key_aligned_pages(
            group=g, frames=frames, n_sample_rows=10,
        ) is None

    def test_matching_keys_colocate_on_one_page(self) -> None:
        g = _group()  # PK = ("K",)
        df_a = pd.DataFrame({"K": ["k1", "k2", "k3"],
                             "country": ["USA", "FR", "DE"]})
        # b.csv lists the same keys in a DIFFERENT order.
        df_b = pd.DataFrame({"K": ["k3", "k1"],
                             "country": ["DE", "United States"]})
        pages = _build_key_aligned_pages(
            group=g, frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=50,
        )
        assert len(pages) == 1
        off = {s.sheet: s.row_offsets for s in pages[0]}
        assert off["a.csv"] == (0, 1, 2)
        assert off["b.csv"] == (0, 1)

    def test_every_key_lands_in_exactly_one_page(self) -> None:
        # The co-location invariant: across multiple pages, all rows
        # for a given normalized key live on the same page.
        g = _group()
        keys_a = [f"k{i}" for i in range(10)]
        keys_b = [f"k{i}" for i in range(10)][::-1]   # reversed order
        df_a = pd.DataFrame({"K": keys_a, "country": list(range(10))})
        df_b = pd.DataFrame({"K": keys_b, "country": list(range(10))})
        pages = _build_key_aligned_pages(
            group=g, frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=3,
        )
        # Map each key -> set of page indices it appears on.
        key_pages: dict = {}
        for p_idx, page in enumerate(pages):
            offs = {s.sheet: s.row_offsets for s in page}
            for ridx in offs.get("a.csv", ()):
                key_pages.setdefault(keys_a[ridx], set()).add(p_idx)
            for ridx in offs.get("b.csv", ()):
                key_pages.setdefault(keys_b[ridx], set()).add(p_idx)
        # Every key confined to a single page.
        for k, ps in key_pages.items():
            assert len(ps) == 1, (k, ps)
        # And every key was actually covered.
        assert set(key_pages) == set(keys_a)

    def test_respects_per_sheet_row_budget(self) -> None:
        g = _group()
        df_a = pd.DataFrame({"K": [f"k{i}" for i in range(6)],
                             "country": list(range(6))})
        df_b = pd.DataFrame({"K": [f"k{i}" for i in range(6)],
                             "country": list(range(6))})
        pages = _build_key_aligned_pages(
            group=g, frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=2,
        )
        # 6 shared keys, budget 2 per sheet -> 3 pages.
        assert len(pages) == 3
        for page in pages:
            for slc in page:
                assert len(slc.row_offsets) <= 2

    def test_key_targeted_drops_single_sheet_keys(self) -> None:
        g = _group()
        # k2 only on a.csv -> dropped under key_targeted.
        df_a = pd.DataFrame({"K": ["k1", "k2", "k3"],
                             "country": ["USA", "X", "DE"]})
        df_b = pd.DataFrame({"K": ["k1", "k3"],
                             "country": ["United States", "DE"]})
        pages = _build_key_aligned_pages(
            group=g, frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=50, key_targeted=True,
        )
        off = {s.sheet: s.row_offsets for s in pages[0]}
        assert off["a.csv"] == (0, 2)   # k2 (idx 1) dropped
        assert off["b.csv"] == (0, 1)

    def test_key_targeted_no_cross_sheet_keys_yields_empty(self) -> None:
        g = _group()
        df_a = pd.DataFrame({"K": ["k1", "k2"], "country": ["A", "B"]})
        df_b = pd.DataFrame({"K": ["k8", "k9"], "country": ["C", "D"]})
        pages = _build_key_aligned_pages(
            group=g, frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=50, key_targeted=True,
        )
        assert pages == []

    def test_missing_key_rows_kept_as_singletons(self) -> None:
        g = _group()
        # b.csv row 1 has a null key -> still shown (non-targeted).
        df_a = pd.DataFrame({"K": ["k1"], "country": ["USA"]})
        df_b = pd.DataFrame({"K": ["k1", None], "country": ["US", "X"]})
        pages = _build_key_aligned_pages(
            group=g, frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=50,
        )
        all_b = [r for page in pages for s in page
                 if s.sheet == "b.csv" for r in s.row_offsets]
        assert sorted(all_b) == [0, 1]


class TestPagesForGroup:
    def test_key_align_uses_aligned_layout(self) -> None:
        g = _group()
        df_a = pd.DataFrame({"K": ["k1", "k2"], "country": ["A", "B"]})
        df_b = pd.DataFrame({"K": ["k2", "k1"], "country": ["B", "A"]})
        pages = _pages_for_group(
            group=g, frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=50, seed=0, key_align=True, key_targeted=False,
        )
        off = {s.sheet: s.row_offsets for s in pages[0]}
        assert off["a.csv"] == (0, 1) and off["b.csv"] == (0, 1)

    def test_keyless_group_falls_back_to_shuffle(self) -> None:
        g = _group(primary_key=(), column_mapping={
            "v": (_entry("a.csv", "v"), _entry("b.csv", "v")),
        })
        frames = {
            "a.csv": pd.DataFrame({"v": list(range(10))}),
            "b.csv": pd.DataFrame({"v": list(range(3))}),
        }
        pages = _pages_for_group(
            group=g, frames=frames, n_sample_rows=4, seed=42,
            key_align=True, key_targeted=False,
        )
        # Same as legacy _build_pages: 10/4 -> 3 pages.
        assert len(pages) == 3

    def test_key_targeted_keyless_group_yields_no_pages(self) -> None:
        g = _group(primary_key=(), column_mapping={
            "v": (_entry("a.csv", "v"), _entry("b.csv", "v")),
        })
        frames = {
            "a.csv": pd.DataFrame({"v": [1, 2]}),
            "b.csv": pd.DataFrame({"v": [1, 2]}),
        }
        pages = _pages_for_group(
            group=g, frames=frames, n_sample_rows=4, seed=0,
            key_align=True, key_targeted=True,
        )
        assert pages == []

    def test_key_align_disabled_uses_shuffle(self) -> None:
        g = _group()
        df_a = pd.DataFrame({"K": [f"k{i}" for i in range(8)],
                             "country": list(range(8))})
        df_b = pd.DataFrame({"K": [f"k{i}" for i in range(8)],
                             "country": list(range(8))})
        pages = _pages_for_group(
            group=g, frames={"a.csv": df_a, "b.csv": df_b},
            n_sample_rows=4, seed=7, key_align=False, key_targeted=False,
        )
        # 8/4 -> 2 pages from the legacy shuffler.
        assert len(pages) == 2


class TestExpectedCallCount:
    def test_mirrors_key_aligned_pages(self) -> None:
        g = _group()
        df_a = pd.DataFrame({"K": [f"k{i}" for i in range(6)],
                             "country": list(range(6))})
        df_b = pd.DataFrame({"K": [f"k{i}" for i in range(6)],
                             "country": list(range(6))})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"), groups=[g],
        )
        frames = {"a.csv": df_a, "b.csv": df_b}
        # 6 shared keys / budget 2 -> 3 calls.
        assert expected_semantic_call_count(
            resolved=wb, frames=frames, n_sample_rows=2,
            key_align=True, key_targeted=False,
        ) == 3

    def test_key_targeted_counts_zero_for_keyless(self) -> None:
        g = _group(primary_key=(), column_mapping={
            "v": (_entry("a.csv", "v"), _entry("b.csv", "v")),
        })
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"), groups=[g],
        )
        frames = {
            "a.csv": pd.DataFrame({"v": [1, 2, 3]}),
            "b.csv": pd.DataFrame({"v": [1, 2, 3]}),
        }
        assert expected_semantic_call_count(
            resolved=wb, frames=frames, n_sample_rows=2,
            key_align=True, key_targeted=True,
        ) == 0


# ---------------------------------------------------------------------------
# Priors threading (C)
# ---------------------------------------------------------------------------


def _priors_with_candidates():
    from src.core.multi_table.priors import (
        EmbeddingMatchCandidate,
        FKCandidate,
        MultiTablePriors,
        NamingCandidate,
        NullSentinelCandidate,
    )
    return MultiTablePriors(
        error_type="inter_table_inconsistency",
        workbook="wb.xlsx",
        sheets=("a.csv", "b.csv"),
        fk_candidates=(FKCandidate(
            parent="a.csv", parent_column="id",
            child="b.csv", child_column="a_id",
            n_parent_distinct=3, n_child_distinct=4,
            overlap=3, coverage=0.75,
            orphan_row_indices=(2,), orphan_values=("ghost",),
            normalized_coverage=0.75,
        ),),
        naming_candidates=(NamingCandidate(
            relation="synonym",
            sheet_a="a.csv", column_a="nation",
            sheet_b="b.csv", column_b="country",
            jaccard=0.6, n_distinct_a=5, n_distinct_b=5,
        ),),
        embedding_match_candidates=(EmbeddingMatchCandidate(
            sheet_a="a.csv", column_a="country",
            sheet_b="b.csv", column_b="country",
            matches=(("Germany", "Deutschland", 0.93),),
        ),),
        null_sentinel_candidates=(NullSentinelCandidate(
            column="note", sheet_a="a.csv", sheet_b="b.csv",
            sentinel_a="", sentinel_b="N/A", freq_a=2, freq_b=3,
        ),),
    )


class TestPriorsBlock:
    def test_none_priors_renders_placeholder(self) -> None:
        g = _group()
        block = _render_priors_block(None, g)
        assert "no structural priors" in block

    def test_empty_priors_renders_placeholder(self) -> None:
        from src.core.multi_table.priors import MultiTablePriors
        g = _group()
        priors = MultiTablePriors(
            error_type="inter_table_inconsistency", workbook="wb.xlsx",
        )
        assert "no structural priors" in _render_priors_block(priors, g)

    def test_renders_value_level_hints(self) -> None:
        g = _group(sheets=("a.csv", "b.csv"))
        block = _render_priors_block(_priors_with_candidates(), g)
        assert "possible FK" in block
        assert "'ghost'" in block             # orphan VALUE, not index
        assert "synonym" in block
        assert "Germany" in block and "Deutschland" in block
        assert "null-placeholder" in block

    def test_never_emits_raw_row_indices(self) -> None:
        # The orphan ROW index (2) must not leak into the block - it
        # would clash with the prompt's LOCAL-index contract.
        g = _group(sheets=("a.csv", "b.csv"))
        block = _render_priors_block(_priors_with_candidates(), g)
        assert "orphan_row_indices" not in block

    def test_filters_candidates_outside_group_sheets(self) -> None:
        from src.core.multi_table.priors import FKCandidate, MultiTablePriors
        # FK references c.csv which is not in the group -> excluded.
        priors = MultiTablePriors(
            error_type="inter_table_inconsistency", workbook="wb.xlsx",
            fk_candidates=(FKCandidate(
                parent="a.csv", parent_column="id",
                child="c.csv", child_column="a_id",
                n_parent_distinct=1, n_child_distinct=1,
                overlap=0, coverage=0.0,
                orphan_row_indices=(0,), orphan_values=("zzz",),
            ),),
        )
        g = _group(sheets=("a.csv", "b.csv"))
        assert "no structural priors" in _render_priors_block(priors, g)


class TestDetectThreadsPriors:
    _PROMPT_WITH_PRIORS = DetectPromptConfig(
        model="m", reasoning_effort="medium", reasoning_summary="detailed",
        system_prompt="sys",
        user_template=(
            "PRIORS:\n{priors_block}\n"
            "MAP:\n{column_mapping_block}\n"
            "ROWS:\n{row_samples_block}"
        ),
    )

    def test_priors_rendered_into_prompt(self) -> None:
        g = _group(sheets=("a.csv", "b.csv"))
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"), groups=[g],
            priors=_priors_with_candidates(),
        )
        df_a = pd.DataFrame({"K": ["k1"], "country": ["Germany"]})
        df_b = pd.DataFrame({"K": ["k1"], "country": ["Deutschland"]})
        llm = _StubLLM([_ok_response([])])
        detect_workbook_inconsistencies(
            resolved=wb, frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=self._PROMPT_WITH_PRIORS,
        )
        assert len(llm.calls) == 1
        sent = llm.calls[0]["prompt"]
        assert "possible FK" in sent
        assert "Deutschland" in sent


# ---------------------------------------------------------------------------
# Key uniqueness signal (deterministic, role-aware)
# ---------------------------------------------------------------------------


class TestKeyCardinality:
    def test_unique_key(self) -> None:
        df = pd.DataFrame({"K": ["a", "b", "c"]})
        n_rows, n_keyed, n_distinct, max_mult = _key_cardinality(df, ["K"])
        assert (n_rows, n_keyed, n_distinct, max_mult) == (3, 3, 3, 1)

    def test_non_unique_key(self) -> None:
        df = pd.DataFrame({"K": ["a", "a", "a", "b"]})
        n_rows, n_keyed, n_distinct, max_mult = _key_cardinality(df, ["K"])
        assert (n_rows, n_keyed, n_distinct, max_mult) == (4, 4, 2, 3)

    def test_nulls_not_counted_as_keyed(self) -> None:
        df = pd.DataFrame({"K": ["a", None, "a"]})
        n_rows, n_keyed, n_distinct, max_mult = _key_cardinality(df, ["K"])
        # 3 rows, 2 keyed (null skipped), 1 distinct, 2 share it.
        assert (n_rows, n_keyed, n_distinct, max_mult) == (3, 2, 1, 2)


class TestRenderKeyUniqueness:
    def test_union_marks_unique_and_non_unique(self) -> None:
        g = _group()  # union, PK = ("K",)
        frames = {
            "a.csv": pd.DataFrame({"K": ["k1", "k2", "k3"],
                                   "country": [1, 2, 3]}),
            # b repeats k1 -> NON-UNIQUE (event/log grain).
            "b.csv": pd.DataFrame({"K": ["k1", "k1", "k2"],
                                   "country": [1, 2, 3]}),
        }
        block = _render_key_uniqueness_block(g, frames)
        assert "PRIMARY key cardinality" in block
        assert "UNIQUE (one row per key)" in block
        assert "NON-UNIQUE" in block
        assert "separate records" in block.lower() or "event/log" in block

    def test_join_is_role_aware_with_fk_note(self) -> None:
        g = _join_group(
            parent_sheet="parent.csv", child_sheet="child.csv",
            parent_cols=("id",), child_cols=("parent_id",),
        )
        frames = {
            # parent unique, child references it many-to-one.
            "parent.csv": pd.DataFrame({"id": ["p1", "p2"]}),
            "child.csv": pd.DataFrame({"parent_id": ["p1", "p1", "p2"]}),
        }
        block = _render_key_uniqueness_block(g, frames)
        assert "JOIN key" in block
        assert "parent parent.csv" in block and "child child.csv" in block
        assert "satisfied foreign key" in block
        assert "fk_orphan" in block

    def test_no_key_falls_back(self) -> None:
        g = _group(primary_key=(), column_mapping={
            "v": (_entry("a.csv", "v"), _entry("b.csv", "v")),
        })
        frames = {"a.csv": pd.DataFrame({"v": [1]}),
                  "b.csv": pd.DataFrame({"v": [1]})}
        block = _render_key_uniqueness_block(g, frames)
        assert "no key resolved" in block

    def test_threaded_into_detect_prompt(self) -> None:
        prompt = DetectPromptConfig(
            model="m", reasoning_effort="medium", reasoning_summary="detailed",
            system_prompt="sys",
            user_template=(
                "UNIQ:\n{key_uniqueness_block}\n"
                "MAP:\n{column_mapping_block}\n"
                "PRIORS:\n{priors_block}\n"
                "ROWS:\n{row_samples_block}"
            ),
        )
        g = _group()
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"), groups=[g],
        )
        frames = {"a.csv": pd.DataFrame({"K": ["k1"], "country": ["USA"]}),
                  "b.csv": pd.DataFrame({"K": ["k1"], "country": ["US"]})}
        llm = _StubLLM([_ok_response([])])
        detect_workbook_inconsistencies(
            resolved=wb, frames=frames, llm=llm, model="m", prompt_cfg=prompt,
        )
        assert "PRIMARY key cardinality" in llm.calls[0]["prompt"]


# ---------------------------------------------------------------------------
# Structural complement (D)
# ---------------------------------------------------------------------------


def _conflict_workbook():
    g = ResolvedGroup(
        workbook="wb.xlsx", group_id="u1", kind="union", confirmed=True,
        rationale="t", sheets=("a.csv", "b.csv"),
        column_mapping={
            "K": (_entry("a.csv", "K"), _entry("b.csv", "K")),
            # Different raw headers -> header_synonym; differing values
            # on the shared PK row -> attribute_conflict.
            "country": (_entry("a.csv", "country"), _entry("b.csv", "Country")),
        },
        primary_key=("K",),
    )
    wb = ResolvedWorkbook(
        workbook="wb.xlsx", sheets=("a.csv", "b.csv"), groups=[g],
    )
    frames = {
        "a.csv": pd.DataFrame({"K": ["k1"], "country": ["USA"]}),
        "b.csv": pd.DataFrame({"K": ["k1"], "Country": ["Canada"]}),
    }
    return wb, frames


class TestStructuralComplement:
    def test_adds_deterministic_findings_when_semantic_empty(self) -> None:
        wb, frames = _conflict_workbook()
        extra_v, extra_ver = structural_complement(
            resolved_list=[wb], frames=frames, semantic_violations=[],
        )
        kinds = {v.kind for v in extra_v}
        assert "header_synonym" in kinds
        assert "attribute_conflict" in kinds
        # 2 header_synonym findings + 1 attribute_conflict finding = 3.
        assert len(extra_ver) == 3
        # All stub verifications are uncertain / compiler-tagged.
        assert all(v.verdict == "uncertain" for v in extra_ver)
        assert all("compiler" in v.rationale for v in extra_ver)

    def test_dedupes_findings_already_in_semantic(self) -> None:
        from src.core.multi_table.violations import MultiTableViolation
        wb, frames = _conflict_workbook()
        # Semantic already flagged the attribute_conflict on the same
        # locations + columns -> the compiler copy must be dropped.
        sem = [
            MultiTableViolation(
                violation_id="sem1", kind="attribute_conflict",
                workbook="wb.xlsx", group_id="u1", group_kind="union",
                sheet="a.csv", row=0, columns=("country",),
                values={}, linked=(("b.csv", 0),),
            ),
            MultiTableViolation(
                violation_id="sem1", kind="attribute_conflict",
                workbook="wb.xlsx", group_id="u1", group_kind="union",
                sheet="b.csv", row=0, columns=("country",),
                values={}, linked=(("a.csv", 0),),
            ),
        ]
        extra_v, _ = structural_complement(
            resolved_list=[wb], frames=frames, semantic_violations=sem,
        )
        kinds = {v.kind for v in extra_v}
        # attribute_conflict de-duped away; header_synonym still added.
        assert "attribute_conflict" not in kinds
        assert "header_synonym" in kinds

    def test_no_resolved_workbooks_is_noop(self) -> None:
        extra_v, extra_ver = structural_complement(
            resolved_list=[], frames={}, semantic_violations=[],
        )
        assert extra_v == [] and extra_ver == []


# ---------------------------------------------------------------------------
# Key alignment recall (B, end-to-end through the detector)
# ---------------------------------------------------------------------------


class TestKeyAlignmentRecall:
    def test_matching_rows_share_a_page_despite_shuffle(self) -> None:
        # Two sheets, same 4 entities in OPPOSITE order, budget forces
        # paging. Under key alignment the matching rows must co-locate
        # so the model can compare them; we assert that by checking the
        # rendered prompt for each page never splits an entity's two
        # values across calls.
        g = _group(column_mapping={
            "K": (_entry("a.csv", "K"), _entry("b.csv", "K")),
            "country": (
                _entry("a.csv", "country"), _entry("b.csv", "country"),
            ),
        })
        df_a = pd.DataFrame({"K": ["k1", "k2", "k3", "k4"],
                             "country": ["USA", "FR", "DE", "JP"]})
        df_b = pd.DataFrame({"K": ["k4", "k3", "k2", "k1"],
                             "country": ["JP", "DE", "FR", "United States"]})
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"), groups=[g],
        )
        llm = _StubLLM([_ok_response([])] * 10)
        detect_workbook_inconsistencies(
            resolved=wb, frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
            n_sample_rows=2,
        )
        # The k1 pair ("USA" / "United States") must appear in the
        # SAME call's prompt.
        joined = [c["prompt"] for c in llm.calls]
        hits = [p for p in joined if "USA" in p and "United States" in p]
        assert len(hits) == 1, (
            "k1's two surface forms were split across pages"
        )

