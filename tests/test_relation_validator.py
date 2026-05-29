"""Tests for ``src.core.relations.validator``.

Cover:

* AST sandbox accepts/rejects code as expected.
* ``_exec_check_code`` builds a callable in the restricted namespace.
* ``_coerce_to_bool_series`` normalises return values.
* ``validate_one`` calls the LLM stub, runs the code, scores the
  pass-rate, extracts sample violations, and surfaces sandbox errors
  without fabricating a pass-rate.
* End-to-end ``validate_relations`` over a small batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.relations.validator import (
    CodeValidationError,
    RelationVerification,
    _coerce_to_bool_series,
    _exec_check_code,
    _parse_llm_decision,
    _validate_code_ast,
    load_verifications_jsonl,
    validate_one,
    validate_relations,
    write_verifications_jsonl,
)


class _StubLLMResponse:
    def __init__(self, text: str, attempts: int = 1) -> None:
        self.text = text
        self.attempts = attempts


class _StubLLMClient:
    """Yields canned responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def query(self, model, prompt, *, reasoning_effort=None, system_prompt="", **_):
        self.calls.append({"model": model, "prompt": prompt})
        if not self._responses:
            raise RuntimeError("stub LLM exhausted")
        return _StubLLMResponse(self._responses.pop(0))


PROMPT_CFG = {
    "system_prompt": "you are a stub",
    "user_template": "validate {kind} on {source}",
    "reasoning_effort": "low",
}


# ---------------------------------------------------------------------------
# AST validator
# ---------------------------------------------------------------------------


class TestAstValidator:
    def test_accepts_simple_check(self) -> None:
        code = "def check(df):\n    return df['A'] > 0\n"
        _validate_code_ast(code)  # no exception

    def test_rejects_import(self) -> None:
        code = "import os\ndef check(df):\n    return df['A'] > 0\n"
        with pytest.raises(CodeValidationError, match="imports"):
            _validate_code_ast(code)

    def test_rejects_from_import(self) -> None:
        code = "from os import system\ndef check(df):\n    return df['A']\n"
        with pytest.raises(CodeValidationError, match="imports"):
            _validate_code_ast(code)

    def test_rejects_dunder_attr(self) -> None:
        code = (
            "def check(df):\n"
            "    return df.__class__\n"
        )
        with pytest.raises(CodeValidationError, match="dunder"):
            _validate_code_ast(code)

    def test_rejects_eval(self) -> None:
        code = "def check(df):\n    return eval('df.A')\n"
        with pytest.raises(CodeValidationError, match="forbidden"):
            _validate_code_ast(code)

    def test_rejects_open(self) -> None:
        code = "def check(df):\n    return open('/etc/passwd')\n"
        with pytest.raises(CodeValidationError, match="forbidden"):
            _validate_code_ast(code)

    def test_syntax_error_surfaces(self) -> None:
        with pytest.raises(CodeValidationError, match="syntax error"):
            _validate_code_ast("def check(df:\n  return")


# ---------------------------------------------------------------------------
# Sandbox exec + return-value coercion
# ---------------------------------------------------------------------------


class TestSandboxExec:
    def test_compiles_and_runs(self) -> None:
        code = "def check(df):\n    return df['A'] > 0\n"
        fn = _exec_check_code(code)
        df = pd.DataFrame({"A": [1, -1, 2]})
        out = fn(df)
        assert isinstance(out, pd.Series)
        assert list(out) == [True, False, True]

    def test_pandas_numpy_re_available(self) -> None:
        code = (
            "def check(df):\n"
            "    s = df['A'].astype(str)\n"
            "    return s.str.match(re.compile(r'^[A-Z]+$')).fillna(False)\n"
        )
        fn = _exec_check_code(code)
        df = pd.DataFrame({"A": ["AB", "ab", "C1"]})
        out = fn(df)
        assert list(out) == [True, False, False]

    def test_no_imports_at_runtime(self) -> None:
        # Even without static AST hits, runtime ``__import__`` should
        # be unreachable because builtins are restricted.
        code = (
            "def check(df):\n"
            "    return df['A']\n"
        )
        fn = _exec_check_code(code)
        # Try to reach __import__ via the function's globals; the
        # restricted builtins dict has no __import__ key.
        glb = fn.__globals__
        assert "__import__" not in glb["__builtins__"]

    def test_must_define_check(self) -> None:
        code = "def other(df):\n    return df\n"
        with pytest.raises(CodeValidationError, match="check"):
            _exec_check_code(code)


class TestCoerceToBoolSeries:
    def test_series_passthrough(self) -> None:
        s = pd.Series([True, False, True])
        out = _coerce_to_bool_series(s, 3)
        assert list(out) == [True, False, True]

    def test_array_to_series(self) -> None:
        a = np.array([True, False, True])
        out = _coerce_to_bool_series(a, 3)
        assert list(out) == [True, False, True]

    def test_list_to_series(self) -> None:
        out = _coerce_to_bool_series([1, 0, 1], 3)
        assert list(out) == [True, False, True]

    def test_scalar_broadcast(self) -> None:
        out = _coerce_to_bool_series(True, 4)
        assert list(out) == [True, True, True, True]

    def test_nan_becomes_false(self) -> None:
        s = pd.Series([True, np.nan, False])
        out = _coerce_to_bool_series(s, 3)
        assert list(out) == [True, False, False]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="returned 2"):
            _coerce_to_bool_series(pd.Series([True, False]), 3)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _coerce_to_bool_series("yes", 3)


# ---------------------------------------------------------------------------
# LLM decision parsing
# ---------------------------------------------------------------------------


class TestParseLlmDecision:
    def test_verifiable_with_code(self) -> None:
        text = '{"verifiable": true, "code": "def check(df):\\n    return df[\\"A\\"]"}'
        d = _parse_llm_decision(text)
        assert d["verifiable"] is True
        assert "def check" in d["code"]

    def test_refusal(self) -> None:
        text = '{"verifiable": false, "reason": "needs external data"}'
        d = _parse_llm_decision(text)
        assert d["verifiable"] is False
        assert "external" in d["reason"]

    def test_strips_markdown_fence(self) -> None:
        text = '```json\n{"verifiable": false, "reason": "x"}\n```'
        d = _parse_llm_decision(text)
        assert d["verifiable"] is False

    def test_missing_verifiable_field(self) -> None:
        with pytest.raises(ValueError, match="verifiable"):
            _parse_llm_decision('{"code": "def check(df): return df"}')

    def test_empty_response(self) -> None:
        with pytest.raises(ValueError):
            _parse_llm_decision("")


# ---------------------------------------------------------------------------
# validate_one (LLM stub + real pandas frame)
# ---------------------------------------------------------------------------


class TestValidateOne:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": ["x", "y", "x", "y", "x"],
        })

    def _relation(self, **overrides):
        base = {
            "species_key": "test_key",
            "source": "t.csv",
            "error_type": "fd_violation",
            "kind": "implication",
            "columns": ["A", "B"],
            "lhs": ["A"],
            "rhs": ["B"],
            "predicate": "A > 0",
            "executable": None,
            "rationale": "test",
        }
        base.update(overrides)
        return base

    def test_perfect_pass(self) -> None:
        llm = _StubLLMClient([
            json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return df['A'] > 0\n",
            })
        ])
        rec = validate_one(
            relation=self._relation(),
            df=self._frame(),
            llm=llm,
            model="m",
            prompt_cfg=PROMPT_CFG,
        )
        assert rec.verifiable is True
        assert rec.error is None
        assert rec.pass_rate == 1.0
        assert rec.n_evaluated == 5
        assert rec.n_passing == 5
        assert rec.n_violating == 0
        assert rec.sample_violations == []

    def test_partial_pass_records_violations(self) -> None:
        llm = _StubLLMClient([
            json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return df['A'] >= 3\n",
            })
        ])
        rec = validate_one(
            relation=self._relation(),
            df=self._frame(),
            llm=llm,
            model="m",
            prompt_cfg=PROMPT_CFG,
        )
        assert rec.verifiable is True
        assert rec.pass_rate == pytest.approx(3 / 5)
        assert rec.n_violating == 2
        # sample_violations should include the failing rows (A == 1, 2)
        rows = rec.sample_violations
        assert len(rows) == 2
        assert all("__row__" in r for r in rows)
        assert {r["A"] for r in rows} == {1, 2}

    def test_refusal_path(self) -> None:
        llm = _StubLLMClient([
            json.dumps({
                "verifiable": False,
                "reason": "needs external catalog",
            })
        ])
        rec = validate_one(
            relation=self._relation(),
            df=self._frame(),
            llm=llm,
            model="m",
            prompt_cfg=PROMPT_CFG,
        )
        assert rec.verifiable is False
        assert rec.reason == "needs external catalog"
        assert rec.pass_rate is None
        assert rec.error is None

    def test_sandbox_error_no_passrate_fabricated(self) -> None:
        # LLM emits illegal code -> sandbox rejects -> error recorded,
        # pass_rate stays None.
        llm = _StubLLMClient([
            json.dumps({
                "verifiable": True,
                "code": "import os\ndef check(df):\n    return df['A'] > 0\n",
            })
        ])
        rec = validate_one(
            relation=self._relation(),
            df=self._frame(),
            llm=llm,
            model="m",
            prompt_cfg=PROMPT_CFG,
        )
        assert rec.verifiable is True
        assert rec.error is not None
        assert "sandbox" in rec.error
        assert rec.pass_rate is None
        assert rec.n_evaluated == 0

    def test_runtime_error_caught(self) -> None:
        llm = _StubLLMClient([
            json.dumps({
                "verifiable": True,
                "code": (
                    "def check(df):\n"
                    "    raise ValueError('boom')\n"
                ),
            })
        ])
        rec = validate_one(
            relation=self._relation(),
            df=self._frame(),
            llm=llm,
            model="m",
            prompt_cfg=PROMPT_CFG,
        )
        assert rec.error is not None
        assert "runtime" in rec.error
        assert rec.pass_rate is None

    def test_length_mismatch_caught(self) -> None:
        llm = _StubLLMClient([
            json.dumps({
                "verifiable": True,
                "code": (
                    "def check(df):\n"
                    "    return pd.Series([True, False])\n"
                ),
            })
        ])
        rec = validate_one(
            relation=self._relation(),
            df=self._frame(),
            llm=llm,
            model="m",
            prompt_cfg=PROMPT_CFG,
        )
        assert rec.error is not None
        assert "runtime" in rec.error or "returned" in rec.error
        assert rec.pass_rate is None

    def test_llm_failure_recorded(self) -> None:
        class _Boom:
            def query(self, *a, **kw):
                raise RuntimeError("network")

        rec = validate_one(
            relation=self._relation(),
            df=self._frame(),
            llm=_Boom(),
            model="m",
            prompt_cfg=PROMPT_CFG,
        )
        assert rec.error is not None
        assert "llm-error" in rec.error
        assert rec.pass_rate is None


# ---------------------------------------------------------------------------
# Batch validation + persistence
# ---------------------------------------------------------------------------


class TestValidateRelationsBatch:
    def test_batch_round_trip(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        relations = [
            {
                "species_key": "k1",
                "source": "t.csv",
                "error_type": "et",
                "kind": "implication",
                "columns": ["A"],
                "predicate": "A > 0",
            },
            {
                "species_key": "k2",
                "source": "t.csv",
                "error_type": "et",
                "kind": "implication",
                "columns": ["B"],
                "predicate": "B is non-empty",
            },
            {
                "species_key": "k3",
                "source": "missing.csv",  # forces "source frame not loaded"
                "error_type": "et",
                "kind": "implication",
                "columns": ["A"],
                "predicate": "A > 0",
            },
        ]
        llm = _StubLLMClient([
            json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return df['A'] > 0\n",
            }),
            json.dumps({
                "verifiable": False,
                "reason": "free-text semantic constraint",
            }),
        ])
        recs = validate_relations(
            relations,
            source_frames={"t.csv": df},
            llm=llm,
            model="m",
            prompt_cfg=PROMPT_CFG,
            concurrency=1,
        )
        assert len(recs) == 3
        by_key = {r.species_key: r for r in recs}
        assert by_key["k1"].pass_rate == 1.0
        assert by_key["k2"].verifiable is False
        assert by_key["k3"].error and "source frame not loaded" in by_key["k3"].error
        # JSONL round-trip.
        out_path = tmp_path / "verifications.jsonl"
        n = write_verifications_jsonl(recs, out_path)
        assert n == 3
        loaded = load_verifications_jsonl(out_path)
        assert {r["species_key"] for r in loaded} == {"k1", "k2", "k3"}


# ---------------------------------------------------------------------------
# Multi-table validation
# ---------------------------------------------------------------------------


class TestMultiTableValidation:
    def _make_workbook(self):
        # Parent has projects P00..P09; child references P00, P01,
        # plus two orphans (Z9, Z8).
        parent = pd.DataFrame({
            "ProjectCode": [f"P{i:02d}" for i in range(10)],
            "Owner": ["a"] * 10,
        })
        child = pd.DataFrame({
            "ProjectCode": ["P00", "P01", "Z9", "P02", "Z8"],
            "Hours": [1, 2, 3, 4, 5],
        })
        return parent, child

    def _relation(self):
        return {
            "species_key": "fk_proj",
            "source": "wb1",
            "error_type": "cross_table_error",
            "kind": "foreign_key",
            "columns": ["ProjectCode"],
            "lhs": [],
            "rhs": [],
            "predicate": "child.ProjectCode references parent.ProjectCode",
            "executable": None,
            "rationale": "FK candidate",
            "tables": ["parent.csv", "child.csv"],
            "column_refs": [
                ["parent.csv", "ProjectCode"],
                ["child.csv", "ProjectCode"],
            ],
        }

    def test_multi_table_check_dfs_runs(self) -> None:
        parent, child = self._make_workbook()
        # Code asserts every child.ProjectCode exists in parent;
        # parent rows always pass.
        code = (
            "def check(dfs):\n"
            "    parent = dfs['parent.csv']\n"
            "    child = dfs['child.csv']\n"
            "    parent_set = set(parent['ProjectCode'])\n"
            "    parent_mask = pd.Series([True] * len(parent))\n"
            "    child_mask = child['ProjectCode'].isin(parent_set)\n"
            "    return pd.concat(\n"
            "        [parent_mask, child_mask.reset_index(drop=True)],\n"
            "        ignore_index=True,\n"
            "    )\n"
        )
        llm = _StubLLMClient([
            json.dumps({"verifiable": True, "code": code})
        ])
        rec = validate_one(
            relation=self._relation(),
            df=None,
            source_frames={"parent.csv": parent, "child.csv": child},
            llm=llm,
            model="m",
            prompt_cfg=PROMPT_CFG,
        )
        assert rec.verifiable is True
        assert rec.error is None
        # 10 parent rows always pass; 3 of 5 child rows pass.
        assert rec.n_evaluated == 15
        assert rec.n_passing == 13
        assert rec.n_violating == 2
        # Sample violations carry __sheet__ identifying which frame.
        assert all("__sheet__" in v for v in rec.sample_violations)
        sheets_in_violations = {v["__sheet__"] for v in rec.sample_violations}
        assert sheets_in_violations == {"child.csv"}

    def test_multi_table_missing_frame_yields_error(self) -> None:
        parent, _ = self._make_workbook()
        llm = _StubLLMClient([
            json.dumps({"verifiable": True, "code": "def check(dfs): pass"})
        ])
        rec = validate_one(
            relation=self._relation(),
            df=None,
            # ``child.csv`` deliberately absent.
            source_frames={"parent.csv": parent},
            llm=llm,
            model="m",
            prompt_cfg=PROMPT_CFG,
        )
        assert rec.verifiable is False
        assert "child.csv" in (rec.error or "")


# ---------------------------------------------------------------------------
# Adaptive (K-sample) validation entrypoint (Gap E)
# ---------------------------------------------------------------------------


class TestValidateOneAdaptive:
    """Cover the K-sample union semantics added in Gap E."""

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({"A": [1, 2, 3, 4, 5]})

    def _relation(self) -> dict:
        return {
            "species_key": "abc123",
            "source": "t.csv",
            "error_type": "fd_violation",
            "kind": "fd",
            "columns": ["A"],
            "lhs": ["A"],
            "rhs": [],
            "predicate": "A > 0",
        }

    def test_kmax_one_returns_first_run(self) -> None:
        from src.adaptive.scoped_gt import ArmConfig
        from src.core.relations.validator import validate_one_adaptive

        llm = _StubLLMClient([
            json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return df['A'] > 0\n",
            }),
        ])
        cfg = ArmConfig(
            epsilon=0.05, epsilon_chao1=0.05, stop_when="gt",
            k_min=1, k_max=1, require_two_consecutive=False,
        )
        rec = validate_one_adaptive(
            relation=self._relation(),
            df=self._frame(),
            llm=llm, model="m", prompt_cfg=PROMPT_CFG,
            arm_config=cfg,
        )
        assert rec.verifiable is True
        assert len(llm.calls) == 1

    def test_unverifiable_short_circuits(self) -> None:
        from src.adaptive.scoped_gt import ArmConfig
        from src.core.relations.validator import validate_one_adaptive

        llm = _StubLLMClient([
            json.dumps({"verifiable": False, "reason": "needs external info"}),
        ])
        cfg = ArmConfig(
            epsilon=0.05, epsilon_chao1=0.05, stop_when="gt",
            k_min=2, k_max=5, require_two_consecutive=False,
        )
        rec = validate_one_adaptive(
            relation=self._relation(),
            df=self._frame(),
            llm=llm, model="m", prompt_cfg=PROMPT_CFG,
            arm_config=cfg,
        )
        assert rec.verifiable is False
        # Should not have re-called the LLM after the first not-verifiable response.
        assert len(llm.calls) == 1

    def test_unions_violations_across_runs(self) -> None:
        from src.adaptive.scoped_gt import ArmConfig
        from src.core.relations.validator import validate_one_adaptive

        # Run 1 flags rows {0,1}; run 2 flags rows {1,2}; run 3 flags {2,3};
        # union should contain {0,1,2,3}.
        responses = [
            json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return df['A'] >= 3\n",  # rows 0,1 violate
            }),
            json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return df['A'] != 2 and df['A'] != 3 if False else (df['A'] != 2) & (df['A'] != 3)\n",
            }),
            json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return (df['A'] != 3) & (df['A'] != 4)\n",
            }),
            json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return df['A'] >= 3\n",
            }),
            json.dumps({
                "verifiable": True,
                "code": "def check(df):\n    return df['A'] >= 3\n",
            }),
        ]
        llm = _StubLLMClient(responses)
        cfg = ArmConfig(
            epsilon=0.05, epsilon_chao1=0.05, stop_when="gt",
            k_min=3, k_max=5, require_two_consecutive=False,
        )
        rec = validate_one_adaptive(
            relation=self._relation(),
            df=self._frame(),
            llm=llm, model="m", prompt_cfg=PROMPT_CFG,
            arm_config=cfg,
        )
        # Union of violations should be at least {row 0, row 1, row 2, row 3}.
        violator_rows = {v["__row__"] for v in rec.sample_violations}
        assert violator_rows >= {0, 1, 2, 3}
        # Should have made multiple LLM calls.
        assert len(llm.calls) >= 3


# ---------------------------------------------------------------------------
# Sandbox-escape hardening (review batch 3): the AST guard must reject the
# str.format / format_map / %-formatting class-graph-walk vectors where the
# dunder names live in a runtime-resolved string, not an Attribute node.
# ---------------------------------------------------------------------------


def test_rejects_str_format_attribute_escape():
    code = "def check(df):\n    return '{0.__class__}'.format(df) == ''\n"
    with pytest.raises(CodeValidationError):
        _validate_code_ast(code)


def test_rejects_format_map_call():
    # No dunder in the literal, so this exercises the .format_map rule itself.
    code = "def check(df):\n    return '{a}'.format_map({'a': 1}) == ''\n"
    with pytest.raises(CodeValidationError, match="format"):
        _validate_code_ast(code)


def test_rejects_percent_format_on_string_literal():
    code = "def check(df):\n    return ('%(x)s' % {'x': 1}) == ''\n"
    with pytest.raises(CodeValidationError):
        _validate_code_ast(code)


def test_rejects_dunder_in_string_literal():
    code = "def check(df):\n    return df['c'].astype(str).str.contains('__class__')\n"
    with pytest.raises(CodeValidationError, match="double-underscore"):
        _validate_code_ast(code)


def test_format_builtin_removed_from_sandbox():
    from src.core.relations.validator import _SAFE_BUILTINS
    assert "format" not in _SAFE_BUILTINS


def test_legitimate_modulo_arithmetic_still_allowed():
    # %-on-a-column (not a string literal) must remain valid.
    _validate_code_ast("def check(df):\n    return df['n'] % 2 == 0\n")
