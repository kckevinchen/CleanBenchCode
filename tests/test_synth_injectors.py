"""Unit tests for synthetic-error injectors."""

from __future__ import annotations

import random

import pytest

from src.synth import INJECTOR_REGISTRY


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)


def test_registry_has_expected_injectors():
    expected = {
        "typo", "casing", "whitespace", "punctuation",
        "number_format", "pattern", "numeric_outlier", "missing_value",
    }
    assert expected.issubset(set(INJECTOR_REGISTRY))


def test_typo_perturbs_alpha_word(rng: random.Random):
    inj = INJECTOR_REGISTRY["typo"]()
    out = inj.try_inject("Boston", rng=rng)
    assert out is not None
    new, params = out
    assert new != "Boston"
    assert "op" in params


def test_typo_skips_numeric(rng: random.Random):
    inj = INJECTOR_REGISTRY["typo"]()
    assert inj.try_inject("12345", rng=rng) is None


def test_typo_deterministic_with_seed():
    a = INJECTOR_REGISTRY["typo"]().try_inject("Anchorage", rng=random.Random(42))
    b = INJECTOR_REGISTRY["typo"]().try_inject("Anchorage", rng=random.Random(42))
    assert a == b


def test_casing_changes_case(rng: random.Random):
    inj = INJECTOR_REGISTRY["casing"]()
    out = inj.try_inject("ACTIVE", rng=rng)
    assert out is not None
    new, _ = out
    assert new != "ACTIVE"
    assert new.lower() == "active"


def test_casing_skips_no_letters(rng: random.Random):
    inj = INJECTOR_REGISTRY["casing"]()
    assert inj.try_inject("12345", rng=rng) is None


def test_whitespace_adds_visible_anomaly(rng: random.Random):
    inj = INJECTOR_REGISTRY["whitespace"]()
    out = inj.try_inject("Active", rng=rng)
    assert out is not None
    new, params = out
    assert new != "Active"
    assert new.strip() == "Active" or "\u00A0" in new or "  " in new
    assert params["op"] in {"leading_space", "trailing_space", "nbsp", "double_space"}


def test_whitespace_skips_already_dirty(rng: random.Random):
    inj = INJECTOR_REGISTRY["whitespace"]()
    assert inj.try_inject(" Active", rng=rng) is None


def test_punctuation_swaps_ascii_to_smart(rng: random.Random):
    inj = INJECTOR_REGISTRY["punctuation"]()
    out = inj.try_inject("It's a test", rng=rng)
    assert out is not None
    new, _ = out
    assert "\u2019" in new or "\u2013" in new or "\u201D" in new


def test_punctuation_skips_clean(rng: random.Random):
    inj = INJECTOR_REGISTRY["punctuation"]()
    # No quotes, no dashes - nothing to swap.
    assert inj.try_inject("Active", rng=rng) is None


def test_number_format_perturbs_int(rng: random.Random):
    inj = INJECTOR_REGISTRY["number_format"]()
    out = inj.try_inject("1234", rng=rng)
    assert out is not None
    new, _ = out
    assert new != "1234"


def test_number_format_skips_text(rng: random.Random):
    inj = INJECTOR_REGISTRY["number_format"]()
    assert inj.try_inject("Active", rng=rng) is None


def test_pattern_changes_signature(rng: random.Random):
    inj = INJECTOR_REGISTRY["pattern"]()
    out = inj.try_inject("ABC123", rng=rng)
    assert out is not None
    new, _ = out
    assert new != "ABC123"


def test_numeric_outlier_far_from_original(rng: random.Random):
    inj = INJECTOR_REGISTRY["numeric_outlier"]()
    out = inj.try_inject("100", rng=rng)
    assert out is not None
    new, params = out
    assert abs(float(new)) >= 100 * 100  # factor >= 100
    assert params["op"] == "multiply"


def test_numeric_outlier_skips_text(rng: random.Random):
    inj = INJECTOR_REGISTRY["numeric_outlier"]()
    assert inj.try_inject("Active", rng=rng) is None


def test_missing_value_returns_blank_or_sentinel(rng: random.Random):
    inj = INJECTOR_REGISTRY["missing_value"]()
    out = inj.try_inject("Active", rng=rng)
    assert out is not None
    new, params = out
    assert params["op"] in {"empty", "NA", "null", "-", "  "}
    assert new in {"", "NA", "null", "-", "  "}


def test_missing_value_skips_blank(rng: random.Random):
    inj = INJECTOR_REGISTRY["missing_value"]()
    assert inj.try_inject("", rng=rng) is None
    assert inj.try_inject("   ", rng=rng) is None


def test_no_injector_returns_unchanged_value():
    """If a try_inject returns a value, it must differ from input."""
    rng = random.Random(123)
    for name, cls in INJECTOR_REGISTRY.items():
        inj = cls()
        # Try a few candidate inputs that *might* be applicable.
        for v in ("Active", "1234", "ABC-123", "It's a test", "100.5"):
            if not inj.applicable(v):
                continue
            out = inj.try_inject(v, rng=random.Random(7))
            if out is None:
                continue
            new, _ = out
            assert new != v, f"{name} produced no-op on {v!r}"
