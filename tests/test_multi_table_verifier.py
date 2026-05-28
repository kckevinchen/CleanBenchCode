"""Tests for ``src.core.multi_table.verifier`` (step 4 of the
multi-table inconsistency pipeline).

The verifier:
    * Passes through unverifiable kinds (header_synonym /
      dtype_drift / fk_orphan) as ``not_verified`` without ever
      calling the LLM.
    * For ``attribute_conflict`` / ``duplicate_row``, batches by
      group_id, fires one LLM call per batch, and parses a
      ``{"verifications": [...]}`` JSON object.
    * Maps the LLM's free-form verdict synonyms onto the canonical
      verdict enum.
    * Falls back to ``verdict="uncertain"`` (with ``error`` /
      ``parse_error`` populated) when the LLM call fails or its
      response cannot be parsed - never crashes the pipeline.
    * Deduplicates multi-row violations by ``violation_id`` so the
      LLM is asked once per logical finding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from src.core.llm_client import LLMResponse
from src.core.multi_table.resolver import (
    ColumnMappingEntry,
    ResolvedGroup,
    ResolvedWorkbook,
)
from src.core.multi_table.verifier import (
    VERIFIABLE_KINDS,
    Verification,
    VerifyPromptConfig,
    _normalize_confidence,
    _normalize_verdict,
    _parse_response,
    load_verify_prompt,
    summarize,
    verify_workbook_violations,
    write_verifications_jsonl,
)
from src.core.multi_table.violations import MultiTableViolation


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubLLM:
    """Records calls and returns scripted ``LLMResponse`` values.

    ``responses`` is consumed FIFO. If exhausted, returns an
    error LLMResponse so the verifier exercises its fail-path.
    """

    def __init__(self, responses: List[LLMResponse]) -> None:
        self.endpoints = ["http://stub"]
        self._responses = list(responses)
        self.calls: List[dict] = []

    def query(
        self, model, prompt, *,
        reasoning_effort, reasoning_summary, system_prompt,
    ):
        self.calls.append({
            "model": model,
            "prompt": prompt,
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
            "V": (_entry("a.csv", "V"), _entry("b.csv", "V")),
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


def _vio(
    *,
    vid: str,
    kind: str,
    workbook: str = "wb.xlsx",
    group_id: str = "u1",
    sheet: str = "a.csv",
    row: int = 0,
    columns=("V",),
    linked=(),
    rationale: str = "compiler",
) -> MultiTableViolation:
    return MultiTableViolation(
        violation_id=vid,
        kind=kind,
        workbook=workbook,
        group_id=group_id,
        group_kind="union",
        sheet=sheet,
        row=row,
        columns=tuple(columns),
        values={},
        linked=tuple(linked),
        rationale=rationale,
    )


_PROMPT_CFG = VerifyPromptConfig(
    model="m",
    reasoning_effort="medium",
    reasoning_summary="detailed",
    system_prompt="sys",
    user_template=(
        "wb={workbook} g={group_id} kind={group_kind} "
        "sheets={sheets} pk={primary_key}\n"
        "MAP:\n{column_mapping_block}\n"
        "VIOL:\n{violations_block}\n"
    ),
)


def _ok_response(records) -> LLMResponse:
    body = json.dumps({"verifications": records})
    return LLMResponse(
        text=body, endpoint="http://stub", model="m", attempts=1,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestNormalizeVerdict:
    def test_canonical_values_pass_through(self) -> None:
        for v in (
            "confirmed_inconsistency", "benign_variation",
            "not_a_violation", "uncertain",
        ):
            assert _normalize_verdict(v) == v

    def test_synonyms_mapped(self) -> None:
        assert _normalize_verdict("confirmed") == "confirmed_inconsistency"
        assert _normalize_verdict("yes") == "confirmed_inconsistency"
        assert _normalize_verdict("benign") == "benign_variation"
        assert _normalize_verdict("ok") == "benign_variation"
        assert _normalize_verdict("false") == "not_a_violation"
        assert _normalize_verdict("spurious") == "not_a_violation"

    def test_unknown_falls_back_to_uncertain(self) -> None:
        assert _normalize_verdict("???") == "uncertain"
        assert _normalize_verdict(None) == "uncertain"
        assert _normalize_verdict("") == "uncertain"


class TestNormalizeConfidence:
    def test_canonical_values_pass_through(self) -> None:
        for c in ("high", "medium", "low"):
            assert _normalize_confidence(c) == c

    def test_unknown_defaults_to_medium(self) -> None:
        assert _normalize_confidence("???") == "medium"
        assert _normalize_confidence(None) == "medium"


class TestParseResponse:
    def test_parses_fenced_json(self) -> None:
        body = "```json\n" + json.dumps({
            "verifications": [
                {"violation_id": "x", "verdict": "confirmed",
                 "confidence": "high", "rationale": "r"},
            ],
        }) + "\n```"
        records, err = _parse_response(body)
        assert err is None
        assert records == [
            {"violation_id": "x", "verdict": "confirmed",
             "confidence": "high", "rationale": "r"},
        ]

    def test_empty_response_errors(self) -> None:
        records, err = _parse_response("")
        assert records == []
        assert err and "empty" in err.lower()

    def test_missing_verifications_array(self) -> None:
        records, err = _parse_response(json.dumps({"foo": []}))
        assert records == []
        assert err and "verifications" in err

    def test_non_json_errors(self) -> None:
        records, err = _parse_response("not json at all")
        assert records == []
        assert err is not None


# ---------------------------------------------------------------------------
# load_verify_prompt
# ---------------------------------------------------------------------------


class TestLoadVerifyPrompt:
    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "verify.yaml"
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
        cfg = load_verify_prompt(p)
        assert cfg.model == "gpt-x"
        assert cfg.reasoning_effort == "high"
        assert cfg.reasoning_summary == "brief"
        assert "hello" in cfg.system_prompt
        assert "body" in cfg.user_template


# ---------------------------------------------------------------------------
# Pass-through behaviour
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_unverifiable_kinds_skip_llm(self) -> None:
        # Non-verifiable kinds: header_synonym, dtype_drift,
        # fk_orphan. None of these should hit the LLM.
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        violations = [
            _vio(vid="v1", kind="header_synonym"),
            _vio(vid="v2", kind="dtype_drift"),
            _vio(vid="v3", kind="fk_orphan"),
        ]
        llm = _StubLLM([])  # no scripted responses; would error if called
        out = verify_workbook_violations(
            resolved=wb, violations=violations, frames={},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert llm.calls == [], "LLM must not be called for pass-through kinds"
        assert {v.verdict for v in out} == {"not_verified"}
        assert {v.violation_id for v in out} == {"v1", "v2", "v3"}

    def test_other_workbook_violations_filtered_out(self) -> None:
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        violations = [
            _vio(vid="v1", kind="attribute_conflict",
                 workbook="OTHER_WB.xlsx"),
        ]
        llm = _StubLLM([])
        out = verify_workbook_violations(
            resolved=wb, violations=violations, frames={},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert out == []
        assert llm.calls == []


# ---------------------------------------------------------------------------
# Happy-path verification
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_attribute_conflict_verified(self) -> None:
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        df_a = pd.DataFrame({"K": [1], "V": [10]})
        df_b = pd.DataFrame({"K": [1], "V": [99]})
        violations = [
            _vio(vid="v1", kind="attribute_conflict",
                 sheet="a.csv", row=0, linked=[("b.csv", 0)]),
        ]
        llm = _StubLLM([_ok_response([
            {"violation_id": "v1", "verdict": "confirmed_inconsistency",
             "confidence": "high",
             "rationale": "different scores for the same key."},
        ])])
        out = verify_workbook_violations(
            resolved=wb, violations=violations,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert len(llm.calls) == 1, "exactly one LLM call per batch"
        assert len(out) == 1
        v = out[0]
        assert v.violation_id == "v1"
        assert v.kind == "attribute_conflict"
        assert v.verdict == "confirmed_inconsistency"
        assert v.confidence == "high"
        assert "different scores" in v.rationale.lower()

    def test_duplicate_row_dedup_by_id(self) -> None:
        # Same logical finding emits one record per affected
        # (sheet, row), all sharing violation_id. The verifier
        # must collapse these into ONE LLM-side question and
        # return ONE verification per id.
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        df_a = pd.DataFrame({"K": [1], "V": [10]})
        df_b = pd.DataFrame({"K": [1], "V": [10]})
        violations = [
            _vio(vid="v1", kind="duplicate_row",
                 sheet="a.csv", row=0, linked=[("b.csv", 0)]),
            _vio(vid="v1", kind="duplicate_row",
                 sheet="b.csv", row=0, linked=[("a.csv", 0)]),
        ]
        llm = _StubLLM([_ok_response([
            {"violation_id": "v1", "verdict": "benign_variation",
             "confidence": "medium",
             "rationale": "snapshot of the same row, expected."},
        ])])
        out = verify_workbook_violations(
            resolved=wb, violations=violations,
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert len(out) == 1, (
            "Linked rows share violation_id and must yield one verification."
        )
        assert out[0].verdict == "benign_variation"

    def test_synonym_verdict_normalized(self) -> None:
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        df_a = pd.DataFrame({"K": [1], "V": [10]})
        df_b = pd.DataFrame({"K": [1], "V": [99]})
        # LLM returns a non-canonical synonym "confirmed".
        llm = _StubLLM([_ok_response([
            {"violation_id": "v1", "verdict": "confirmed",
             "confidence": "MEDIUM", "rationale": "ok"},
        ])])
        out = verify_workbook_violations(
            resolved=wb,
            violations=[_vio(vid="v1", kind="attribute_conflict",
                             linked=[("b.csv", 0)])],
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert out[0].verdict == "confirmed_inconsistency"
        assert out[0].confidence == "medium"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_llm_error_yields_uncertain_with_error(self) -> None:
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        df_a = pd.DataFrame({"K": [1], "V": [10]})
        df_b = pd.DataFrame({"K": [1], "V": [99]})
        # Empty stub -> exhausted error response.
        llm = _StubLLM([])
        out = verify_workbook_violations(
            resolved=wb,
            violations=[_vio(vid="v1", kind="attribute_conflict",
                             linked=[("b.csv", 0)])],
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert len(out) == 1
        assert out[0].verdict == "uncertain"
        assert out[0].error is not None

    def test_unparseable_response_yields_uncertain_with_parse_error(self) -> None:
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        df_a = pd.DataFrame({"K": [1], "V": [10]})
        df_b = pd.DataFrame({"K": [1], "V": [99]})
        llm = _StubLLM([LLMResponse(
            text="not json at all", endpoint="http://stub",
            model="m", attempts=1,
        )])
        out = verify_workbook_violations(
            resolved=wb,
            violations=[_vio(vid="v1", kind="attribute_conflict",
                             linked=[("b.csv", 0)])],
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert len(out) == 1
        assert out[0].verdict == "uncertain"
        assert out[0].parse_error is not None

    def test_missing_violation_id_in_response_yields_uncertain(self) -> None:
        # LLM returns valid JSON but omits a violation_id we asked
        # about. That violation must surface as 'uncertain'.
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group()],
        )
        df_a = pd.DataFrame({"K": [1, 2], "V": [10, 20]})
        df_b = pd.DataFrame({"K": [1, 2], "V": [99, 20]})
        llm = _StubLLM([_ok_response([
            {"violation_id": "v1", "verdict": "confirmed",
             "confidence": "high", "rationale": "r"},
            # v2 missing - simulates the LLM dropping it.
        ])])
        out = verify_workbook_violations(
            resolved=wb,
            violations=[
                _vio(vid="v1", kind="attribute_conflict",
                     row=0, linked=[("b.csv", 0)]),
                _vio(vid="v2", kind="attribute_conflict",
                     row=1, linked=[("b.csv", 1)]),
            ],
            frames={"a.csv": df_a, "b.csv": df_b},
            llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        by_vid = {v.violation_id: v for v in out}
        assert by_vid["v1"].verdict == "confirmed_inconsistency"
        assert by_vid["v2"].verdict == "uncertain"

    def test_violation_against_unconfirmed_group(self) -> None:
        # Defensive: the compiler should never emit violations for
        # unconfirmed groups, but if it does, the verifier must
        # gracefully label them uncertain (no LLM call).
        wb = ResolvedWorkbook(
            workbook="wb.xlsx", sheets=("a.csv", "b.csv"),
            groups=[_group(confirmed=False)],
        )
        llm = _StubLLM([])
        out = verify_workbook_violations(
            resolved=wb,
            violations=[_vio(vid="v1", kind="attribute_conflict",
                             linked=[("b.csv", 0)])],
            frames={}, llm=llm, model="m", prompt_cfg=_PROMPT_CFG,
        )
        assert llm.calls == []
        assert len(out) == 1
        assert out[0].verdict == "uncertain"


# ---------------------------------------------------------------------------
# Persistence + summary
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_write_jsonl_round_trip(self, tmp_path: Path) -> None:
        records = [
            Verification(
                violation_id="v1", workbook="wb", group_id="u1",
                kind="attribute_conflict",
                verdict="confirmed_inconsistency",
                confidence="high", rationale="ok",
            ),
            Verification(
                violation_id="v2", workbook="wb", group_id="u1",
                kind="duplicate_row",
                verdict="benign_variation",
                confidence="medium", rationale="snapshot",
            ),
        ]
        p = tmp_path / "v.jsonl"
        n = write_verifications_jsonl(records, p)
        assert n == 2
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        loaded = [json.loads(line) for line in lines]
        assert loaded[0]["violation_id"] == "v1"
        assert loaded[0]["verdict"] == "confirmed_inconsistency"
        assert loaded[1]["kind"] == "duplicate_row"

    def test_summarize_buckets_correctly(self) -> None:
        records = [
            Verification(violation_id="v1", workbook="wb", group_id="u1",
                         kind="attribute_conflict",
                         verdict="confirmed_inconsistency",
                         confidence="high", rationale=""),
            Verification(violation_id="v2", workbook="wb", group_id="u1",
                         kind="attribute_conflict",
                         verdict="benign_variation",
                         confidence="medium", rationale=""),
            Verification(violation_id="v3", workbook="wb", group_id="u1",
                         kind="duplicate_row",
                         verdict="not_a_violation",
                         confidence="low", rationale=""),
            Verification(violation_id="v4", workbook="wb", group_id="u1",
                         kind="header_synonym", verdict="not_verified",
                         confidence="medium", rationale="",
                         error="boom"),
        ]
        s = summarize(records)
        assert s["n_verifications"] == 4
        assert s["by_verdict"]["confirmed_inconsistency"] == 1
        assert s["by_verdict"]["benign_variation"] == 1
        assert s["by_verdict"]["not_a_violation"] == 1
        assert s["by_verdict"]["not_verified"] == 1
        assert s["by_kind"]["attribute_conflict"] == 2
        assert s["by_kind"]["duplicate_row"] == 1
        assert s["by_kind"]["header_synonym"] == 1
        assert s["by_kind_verdict"]["attribute_conflict"][
            "confirmed_inconsistency"] == 1
        assert s["n_errored"] == 1


# ---------------------------------------------------------------------------
# Sanity: VERIFIABLE_KINDS coverage
# ---------------------------------------------------------------------------


class TestVerifiableKindsCoverage:
    def test_attribute_conflict_and_duplicate_row_are_verifiable(self) -> None:
        assert "attribute_conflict" in VERIFIABLE_KINDS
        assert "duplicate_row" in VERIFIABLE_KINDS

    def test_mechanical_kinds_are_pass_through(self) -> None:
        assert "header_synonym" not in VERIFIABLE_KINDS
        assert "dtype_drift" not in VERIFIABLE_KINDS
        assert "fk_orphan" not in VERIFIABLE_KINDS
