"""Tests for the out-of-process check() runner (follow-up item 3).

The LLM-emitted ``check(df)`` predicate now runs in a killable,
resource-capped subprocess instead of an in-process daemon thread whose
timeout could not actually stop a runaway predicate. These tests pin the
new guarantees:

  * valid checks return the SAME coerced mask (index preserved) as the
    old in-process path -- single- and multi-table;
  * an infinite loop is *killed* and surfaces as ``TimeoutError`` within
    the wall-clock budget (the bug this change fixes);
  * an allocation bomb is contained (catchable error, not a host OOM);
  * sandbox / runtime errors keep their distinct exception types so the
    caller can label them "sandbox:" vs "runtime:".

The broader correctness of the subprocess path is already exercised by
``test_relation_validator`` / ``test_relation_violations`` (every
``validate_one`` there now routes through this runner).
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from src.core.relations._check_runner import run_check_subprocess
from src.core.relations.validator import CodeValidationError, validate_one


def _df() -> pd.DataFrame:
    return pd.DataFrame({"A": [1, -1, 2, -3, 5]})


# ---------------------------------------------------------------------------
# run_check_subprocess: happy paths
# ---------------------------------------------------------------------------


def test_returns_coerced_series_preserving_index() -> None:
    mask = run_check_subprocess("def check(df):\n    return df['A'] > 0\n", _df(), 8.0)
    assert isinstance(mask, pd.Series)
    assert list(mask) == [True, False, True, False, True]
    assert list(mask.index) == [0, 1, 2, 3, 4]
    assert mask.dtype == bool


def test_scalar_return_broadcasts() -> None:
    # Coercion happens in the child; a scalar bool broadcasts to n rows.
    mask = run_check_subprocess("def check(df):\n    return True\n", _df(), 8.0)
    assert list(mask) == [True] * 5


def test_multi_table_dict_payload() -> None:
    parent = pd.DataFrame({"K": ["a", "b", "c"]})
    child = pd.DataFrame({"K": ["a", "z"]})
    code = (
        "def check(dfs):\n"
        "    p = dfs['p']; c = dfs['c']\n"
        "    pm = pd.Series([True] * len(p))\n"
        "    cm = c['K'].isin(set(p['K'])).reset_index(drop=True)\n"
        "    return pd.concat([pm, cm], ignore_index=True)\n"
    )
    mask = run_check_subprocess(code, {"p": parent, "c": child}, 8.0)
    assert list(mask) == [True, True, True, True, False]


# ---------------------------------------------------------------------------
# run_check_subprocess: the bug this change fixes -- runaway predicates
# ---------------------------------------------------------------------------


def test_infinite_loop_is_killed_and_times_out() -> None:
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        run_check_subprocess(
            "def check(df):\n    while True:\n        pass\n", _df(), 1.0,
        )
    elapsed = time.monotonic() - start
    # Killed near the deadline -- NOT hung. (Old daemon-thread path could
    # never stop the loop; the worker ran until process exit.)
    assert elapsed < 8.0, f"timeout took {elapsed:.1f}s -- child not killed?"


def test_allocation_bomb_is_contained() -> None:
    # RLIMIT_AS turns a giant allocation into a catchable MemoryError in
    # the child (reported as a runtime error) rather than an OOM of the host.
    with pytest.raises(Exception) as ei:
        run_check_subprocess(
            "def check(df):\n    x = [0] * (10 ** 10)\n    return df['A'] > 0\n",
            _df(), 8.0,
        )
    assert not isinstance(ei.value, TimeoutError)
    assert "Memory" in str(ei.value) or "memory" in str(ei.value)


# ---------------------------------------------------------------------------
# run_check_subprocess: error classification preserved
# ---------------------------------------------------------------------------


def test_sandbox_error_raises_codevalidation() -> None:
    with pytest.raises(CodeValidationError):
        run_check_subprocess(
            "import os\ndef check(df):\n    return df['A'] > 0\n", _df(), 8.0,
        )


def test_runtime_error_raises_runtime() -> None:
    with pytest.raises(Exception) as ei:
        run_check_subprocess(
            "def check(df):\n    raise ValueError('boom')\n", _df(), 8.0,
        )
    assert not isinstance(ei.value, (TimeoutError, CodeValidationError))
    assert "boom" in str(ei.value)


# ---------------------------------------------------------------------------
# End-to-end through validate_one: the timeout is now real
# ---------------------------------------------------------------------------


class _StubLLM:
    def __init__(self, code: str) -> None:
        self._code = code

    def query(self, model, prompt, *, reasoning_effort=None, system_prompt="", **_):
        class _R:
            text = json.dumps({"verifiable": True, "code": self._code})
            attempts = 1
        return _R()


def test_validate_one_times_out_on_runaway_check() -> None:
    rec = validate_one(
        relation={
            "species_key": "k", "source": "t.csv", "error_type": "fd_violation",
            "kind": "fd", "columns": ["A"], "predicate": "loops forever",
        },
        df=_df(),
        llm=_StubLLM("def check(df):\n    while True:\n        pass\n"),
        model="m",
        prompt_cfg={"system_prompt": "s", "user_template": "{kind} {source}",
                    "reasoning_effort": "low"},
        timeout_seconds=1.0,
    )
    assert rec.error is not None
    assert "timeout" in rec.error
    assert rec.pass_rate is None
