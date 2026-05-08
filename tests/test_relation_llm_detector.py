"""Tests for ``src.core.relation_llm_detector``.

Cover:

* ``_parse_violations_response`` tolerates fenced / messy JSON.
* ``detect_violations_llm`` short-circuits on missing columns / empty
  frames without firing an LLM call.
* End-to-end ``detect_violations_llm`` against a stub LLM client:
  builds the prompt, parses ``violating_rows``, and yields
  ``RelationViolation`` records with ``detection_method ==
  "llm_table_scan"`` and the LLM's reason copied verbatim.
* ``detect_violations_for_relations`` walks the verifications dict
  and only invokes the LLM for ``verifiable=False`` rules.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.core.relation_llm_detector import (
    _parse_violations_response,
    detect_violations_for_relations,
    detect_violations_llm,
)


# ---------------------------------------------------------------------------
# Stub LLM client
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, text: str, attempts: int = 1) -> None:
        self.text = text
        self.attempts = attempts


class _StubLLM:
    """Yields canned responses in FIFO order; records every call."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls = []

    def query(self, *, model, prompt, reasoning_effort=None, system_prompt="",
              **_):
        self.calls.append({
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "reasoning_effort": reasoning_effort,
        })
        if not self._responses:
            raise RuntimeError("stub LLM exhausted")
        return _StubResponse(self._responses.pop(0))


PROMPT_CFG = {
    "system_prompt": "you are a stub",
    "user_template": "scan {source} for {predicate}\n{table_md}",
    "reasoning_effort": "low",
}


@pytest.fixture
def freeform_relation() -> dict:
    return {
        "species_key": "ft_color_name",
        "source": "products.csv",
        "error_type": "free_text_inconsistency",
        "kind": "free_text_check",
        "columns": ["color"],
        "predicate": "color names are spelt consistently",
        "rationale": "Mixing 'red' and 'Red' creates duplicates.",
        "confidence": "medium",
    }


@pytest.fixture
def freeform_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "color": ["red", "Red", "blue", "RED"],
        "qty":   [1, 2, 3, 4],
    })


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_clean_json_with_violations(self) -> None:
        body = json.dumps({
            "applicable": True,
            "violations": [
                {"row": 1, "reason": "casing differs from row 0"},
                {"row": 3, "reason": "all-caps spelling"},
            ],
        })
        out = _parse_violations_response(body)
        assert out["applicable"] is True
        rows = sorted(v["row"] for v in out["violations"])
        assert rows == [1, 3]

    def test_strips_markdown_fence(self) -> None:
        body = "```json\n" + json.dumps({"applicable": True, "violations": []}) + "\n```"
        out = _parse_violations_response(body)
        assert out["applicable"] is True
        assert out["violations"] == []

    def test_handles_inapplicable_response(self) -> None:
        body = json.dumps({
            "applicable": False,
            "reason": "color column is empty",
            "violations": [],
        })
        out = _parse_violations_response(body)
        assert out["applicable"] is False
        assert out["reason"] == "color column is empty"
        assert out["violations"] == []

    def test_drops_malformed_violation_entries(self) -> None:
        body = json.dumps({
            "applicable": True,
            "violations": [
                {"row": 1, "reason": "ok"},
                {"row": "not-a-number", "reason": "bad"},  # dropped
                {"reason": "no row"},                       # dropped
                "not an object",                            # dropped
            ],
        })
        out = _parse_violations_response(body)
        assert len(out["violations"]) == 1
        assert out["violations"][0]["row"] == 1

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_violations_response("")

    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_violations_response("not json at all")


# ---------------------------------------------------------------------------
# detect_violations_llm
# ---------------------------------------------------------------------------


class TestDetectViolationsLlm:
    def test_short_circuits_on_missing_column(self, freeform_relation, freeform_frame) -> None:
        # Relation references a column that doesn't exist in the frame.
        rel = dict(freeform_relation)
        rel["columns"] = ["nope"]
        stub = _StubLLM([])  # would raise if called
        out = detect_violations_llm(
            rel, freeform_frame,
            refusal_reason="(stub)",
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert out == []
        assert stub.calls == []

    def test_short_circuits_on_empty_frame(self, freeform_relation) -> None:
        stub = _StubLLM([])
        out = detect_violations_llm(
            freeform_relation, pd.DataFrame({"color": []}),
            refusal_reason="(stub)",
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert out == []
        assert stub.calls == []

    def test_emits_records_with_llm_reason(self, freeform_relation, freeform_frame) -> None:
        stub = _StubLLM([
            json.dumps({
                "applicable": True,
                "violations": [
                    {"row": 1, "reason": "'Red' vs 'red'"},
                    {"row": 3, "reason": "'RED' all caps"},
                ],
            })
        ])
        out = detect_violations_llm(
            freeform_relation, freeform_frame,
            refusal_reason="depends on naming convention",
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert len(out) == 2
        rows = sorted(v.row for v in out)
        assert rows == [1, 3]
        for v in out:
            assert v.detection_method == "llm_table_scan"
            assert v.predicate == "color names are spelt consistently"
            assert v.peer_rows == []
            assert v.rule_pass_rate == 0.0
        # Reasons survive verbatim.
        by_row = {v.row: v.llm_reason for v in out}
        assert by_row[1] == "'Red' vs 'red'"
        assert by_row[3] == "'RED' all caps"
        # Exactly one LLM call was made (whole-table scan, not chunked).
        assert len(stub.calls) == 1
        # The user prompt embeds the predicate template token.
        assert "color names are spelt consistently" in stub.calls[0]["prompt"]

    def test_drops_out_of_range_rows(self, freeform_relation, freeform_frame) -> None:
        stub = _StubLLM([
            json.dumps({
                "applicable": True,
                "violations": [
                    {"row": 1, "reason": "ok"},
                    {"row": 99, "reason": "out of range"},
                    {"row": -1, "reason": "negative"},
                ],
            })
        ])
        out = detect_violations_llm(
            freeform_relation, freeform_frame,
            refusal_reason="(stub)",
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert len(out) == 1
        assert out[0].row == 1

    def test_caps_violation_count(self, freeform_relation) -> None:
        # Build a 60-row frame and ask the LLM to flag 60 rows; the
        # detector should cap at max_violations_per_rule=10.
        df = pd.DataFrame({"color": [f"c{i}" for i in range(60)]})
        violations = [{"row": i, "reason": f"r{i}"} for i in range(60)]
        stub = _StubLLM([json.dumps({"applicable": True, "violations": violations})])
        out = detect_violations_llm(
            freeform_relation, df,
            refusal_reason="(stub)",
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
            max_violations_per_rule=10,
        )
        assert len(out) == 10

    def test_inapplicable_response_yields_empty(self, freeform_relation, freeform_frame) -> None:
        stub = _StubLLM([
            json.dumps({"applicable": False, "reason": "no signal", "violations": []})
        ])
        out = detect_violations_llm(
            freeform_relation, freeform_frame,
            refusal_reason="(stub)",
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert out == []

    def test_llm_call_failure_returns_empty(self, freeform_relation, freeform_frame) -> None:
        class _BoomLLM:
            def query(self, **_): raise RuntimeError("boom")
        out = detect_violations_llm(
            freeform_relation, freeform_frame,
            refusal_reason="(stub)",
            llm=_BoomLLM(), model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert out == []


# ---------------------------------------------------------------------------
# detect_violations_for_relations (the dispatcher)
# ---------------------------------------------------------------------------


class TestDetectViolationsForRelations:
    def test_only_invokes_for_non_code_verifiable(
        self, freeform_relation, freeform_frame,
    ) -> None:
        # Two relations: one is code-verifiable (skipped), one isn't.
        code_rel = {
            "species_key": "code_rule",
            "source": "products.csv",
            "error_type": "fd_violation",
            "columns": ["color"],
            "predicate": "color is non-null",
            "confidence": "high",
        }
        verifications = {
            "code_rule": {"verifiable": True, "code": "def check(df): return df['color'].notna()"},
            "ft_color_name": {"verifiable": False, "reason": "naming convention"},
        }
        stub = _StubLLM([
            json.dumps({
                "applicable": True,
                "violations": [{"row": 1, "reason": "casing"}],
            })
        ])
        out = detect_violations_for_relations(
            [code_rel, freeform_relation],
            verifications=verifications,
            source_frames={"products.csv": freeform_frame},
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        # Only the freeform relation produced a violation; the code
        # relation was filtered upstream of any LLM call.
        assert len(out) == 1
        assert out[0].relation_species_key == "ft_color_name"
        assert len(stub.calls) == 1

    def test_skips_relations_without_verification(self, freeform_relation, freeform_frame) -> None:
        stub = _StubLLM([])  # would raise if called
        out = detect_violations_for_relations(
            [freeform_relation],
            verifications={},  # no entry → skipped
            source_frames={"products.csv": freeform_frame},
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert out == []
        assert stub.calls == []

    def test_skips_multi_table_relations(self, freeform_frame) -> None:
        rel = {
            "species_key": "mt_rule",
            "source": "products.csv",
            "error_type": "inter_table_inconsistency",
            "columns": ["color"],
            "predicate": "rule across sheets",
            "tables": ["sheet_a", "sheet_b"],
            "confidence": "high",
        }
        stub = _StubLLM([])
        out = detect_violations_for_relations(
            [rel],
            verifications={"mt_rule": {"verifiable": False, "reason": "needs sheets"}},
            source_frames={"products.csv": freeform_frame},
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert out == []
        assert stub.calls == []

    def test_skips_relations_with_verifier_error(self, freeform_relation, freeform_frame) -> None:
        stub = _StubLLM([])
        out = detect_violations_for_relations(
            [freeform_relation],
            verifications={
                "ft_color_name": {
                    "verifiable": False,
                    "error": "sandbox: NameError",
                    "reason": "(unused)",
                }
            },
            source_frames={"products.csv": freeform_frame},
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        # Verifier-error rules don't get LLM detection (unreliable
        # signal that the rule even makes sense).
        assert out == []
        assert stub.calls == []
