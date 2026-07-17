"""Self-test for score_repair.py grading (run: python external/rein/_selftest_score_repair.py).

Deterministic, no benchmark I/O — exercises value-equality and the correction /
e2e grading math against hand-computed expectations. Mirrors the fixture used to
validate the scorer when it was written.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import score_repair as R  # noqa: E402


def _check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        raise AssertionError(name)


def test_value_equality():
    _check("exact", C.values_equal("abc", "abc"))
    _check("ws-collapse", C.values_equal("a  b", " a b "))
    _check("numeric-thousands", C.values_equal("1,000", "1000"))
    _check("numeric-trailing-zero", C.values_equal("1.0", "1"))
    _check("case-sensitive-default", not C.values_equal("Abc", "abc"))
    _check("casefold-opt-in", C.values_equal("Abc", "abc", casefold=True))
    _check("no-numeric-strict", not C.values_equal("1,000", "1000", numeric=False))
    _check("empty-not-equal", not C.values_equal("", "x"))


def test_correction_grading():
    gold = {("d", 0, 0): {"Apple"}, ("d", 1, 0): {"100"},
            ("d", 2, 0): {"Banana"}, ("d", 3, 0): {"Cat"}}
    E = set(gold)
    match = lambda c, v: any(C.values_equal(v, g) for g in gold[c])  # noqa: E731
    preds = {("d", 0, 0): "Apple", ("d", 1, 0): "100.0", ("d", 2, 0): "Bananna"}  # 3rd wrong, 4th abstain
    tp, fp, fn, att, _ = R._grade_correction(E, preds, match, {})
    _check("corr tp=2", tp == 2)
    _check("corr fp=1", fp == 1)
    _check("corr fn=2", fn == 2)
    _check("corr attempts=3", att == 3)


def test_accepted_set_multi():
    # A cell that accepts multiple gold variants: either should score correct.
    gold = {("d", 0, 0): {"(651) 602-4724", "651-602-4724"}}
    match = lambda c, v: any(C.values_equal(v, g) for g in gold[c])  # noqa: E731
    tp, fp, fn, att, _ = R._grade_correction(set(gold), {("d", 0, 0): "651-602-4724"}, match, {})
    _check("multi-accepted variant matches", tp == 1 and fp == 0)


def test_e2e_grading():
    gold = {("d", 0, 0): {"Apple"}, ("d", 1, 0): {"100"},
            ("d", 2, 0): {"Banana"}, ("d", 3, 0): {"Cat"}}
    E = set(gold)
    match = lambda c, v: any(C.values_equal(v, g) for g in gold[c])  # noqa: E731
    preds = {
        ("d", 0, 0): "Apple",    # error + correct -> TP
        ("d", 1, 0): "100",      # error + correct -> TP
        ("d", 2, 0): "WRONG",    # error + wrong   -> FP (located, wrong value)
        ("d", 9, 9): "whatever",  # clean cell      -> FP
        ("d", 5, 5): "",         # empty           -> ignored
    }
    tp, fp, fn, nch, tploc, _ = R._grade_e2e(E, preds, match)
    _check("e2e tp=2", tp == 2)
    _check("e2e fp=2", fp == 2)
    _check("e2e fn=2", fn == 2)
    _check("e2e n_changes=4", nch == 4)
    _check("e2e tp_location=3", tploc == 3)


if __name__ == "__main__":
    test_value_equality()
    test_correction_grading()
    test_accepted_set_multi()
    test_e2e_grading()
    print("\nALL SCORE_REPAIR SELF-TESTS PASSED")
