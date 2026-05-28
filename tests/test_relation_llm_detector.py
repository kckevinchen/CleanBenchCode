"""Tests for ``src.core.relations.llm_detector``.

Cover:

* ``_parse_violations_response`` tolerates fenced / messy JSON.
* ``detect_violations_llm`` short-circuits on missing columns / empty
  frames without firing an LLM call.
* End-to-end ``detect_violations_llm`` against a stub LLM client:
  builds the prompt, parses ``violating_rows``, and yields
  ``RelationViolation`` records with ``detection_method ==
  "llm_table_scan"`` and the LLM's reason copied verbatim.
* ``detect_violations_for_relations`` walks the verifications dict
  and only invokes the LLM for rules the code-path validator could
  not produce a pass-rate for: refused (``verifiable=False``) and
  errored verifications.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.core.relations.llm_detector import (
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
    "user_template": "scan {source} for {predicate}\nrefusal: {refusal_reason}\n{table_md}",
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
    def test_only_invokes_for_code_refused(
        self, freeform_relation, freeform_frame,
    ) -> None:
        # Two relations: a code-verified rule (handled by the
        # code-violation compiler, skipped here) and a code-refused
        # rule -- the LLM declined to write check(df), so the
        # dispatcher hands the rule to the whole-table LLM scan.
        code_rel = {
            "species_key": "code_rule",
            "source": "products.csv",
            "error_type": "fd_violation",
            "columns": ["color"],
            "predicate": "color is non-null",
            "confidence": "high",
        }
        verifications = {
            "code_rule": {
                "verifier_kind": "code",
                "verifiable": True,
                "code": "def check(df): return df['color'].notna()",
                "pass_rate": 0.95,
            },
            "ft_color_name": {
                "verifier_kind": "code",
                "verifiable": False,
                "reason": "free-text rule; code refused",
            },
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
        # Only the code-refused relation produced a violation; the
        # code-verified one is the code compiler's job.
        assert len(out) == 1
        assert out[0].relation_species_key == "ft_color_name"
        assert len(stub.calls) == 1
        # The validator's refusal reason should be forwarded to the
        # LLM prompt.
        assert "free-text rule; code refused" in stub.calls[0]["prompt"]

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
            verifications={"mt_rule": {
                "verifier_kind": "code",
                "verifiable": False,
                "reason": "needs sheets",
            }},
            source_frames={"products.csv": freeform_frame},
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert out == []
        assert stub.calls == []

    def test_routes_errored_verifications_to_llm_scan(
        self, freeform_relation, freeform_frame,
    ) -> None:
        # Recall-focused behaviour: a verifier error leaves the rule
        # without a pass-rate signal, but the rule itself may still be
        # real. Instead of dropping silently, the dispatcher now routes
        # errored rules through the LLM table scan so any violators it
        # finds reach the labelling queue. The verifier error string is
        # forwarded as the refusal_reason in the prompt.
        stub = _StubLLM([
            json.dumps({
                "applicable": True,
                "violations": [{"row": 1, "reason": "casing differs"}],
            })
        ])
        out = detect_violations_for_relations(
            [freeform_relation],
            verifications={
                "ft_color_name": {
                    "verifier_kind": "code",
                    "verifiable": False,
                    "error": "sandbox: ValueError",
                    "reason": "(unused)",
                }
            },
            source_frames={"products.csv": freeform_frame},
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert len(out) == 1
        assert out[0].relation_species_key == "ft_color_name"
        assert len(stub.calls) == 1
        # The verifier's error string must reach the prompt so the LLM
        # knows the rule is unvalidated.
        assert "sandbox: ValueError" in stub.calls[0]["prompt"]

    def test_errored_code_path_routed_to_llm_scan(
        self, freeform_relation, freeform_frame,
    ) -> None:
        # Symmetric fallback for the code path: when the LLM-written
        # check(df) raised in the sandbox, the deterministic compiler
        # cannot proceed. The dispatcher routes the rule to the LLM
        # scan so the labelling queue still sees it.
        stub = _StubLLM([
            json.dumps({
                "applicable": True,
                "violations": [{"row": 2, "reason": "naming drift"}],
            })
        ])
        out = detect_violations_for_relations(
            [freeform_relation],
            verifications={
                "ft_color_name": {
                    "verifier_kind": "code",
                    "verifiable": True,
                    "code": "def check(df): return df['missing'].notna()",
                    "error": "sandbox: KeyError: 'missing'",
                    "pass_rate": None,
                }
            },
            source_frames={"products.csv": freeform_frame},
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert len(out) == 1
        assert out[0].relation_species_key == "ft_color_name"
        assert "sandbox: KeyError" in stub.calls[0]["prompt"]

    def test_legacy_llm_sample_records_are_ignored(
        self, freeform_relation, freeform_frame,
    ) -> None:
        # Old runs may have ``verifier_kind="llm_sample"`` records on
        # disk. The current pipeline only runs the code-path verifier,
        # so those legacy records are skipped silently rather than
        # crashing or being re-scanned.
        stub = _StubLLM([])  # would raise if called
        out = detect_violations_for_relations(
            [freeform_relation],
            verifications={
                "ft_color_name": {
                    "verifier_kind": "llm_sample",
                    "verifiable": True,
                    "pass_rate": 0.9,
                    "n_violating": 3,
                    "reason": "legacy artifact",
                }
            },
            source_frames={"products.csv": freeform_frame},
            llm=stub, model="gpt-stub", prompt_cfg=PROMPT_CFG,
        )
        assert out == []
        assert stub.calls == []
