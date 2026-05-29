"""Unit tests for the pure logic in ``src.labeling.correction_suggester``.

This 1100+ line module (LLM K-shot correction suggestion, used by both the
offline benchmark exporter and the live workbench endpoint) had zero direct
tests. These cover the deterministic, LLM-free units: target-id parsing,
the LLM-list parser, the cache key, the K-shot voting/consensus
(``aggregate_samples``), and the UI payload shaper. The LLM-dispatch layer
(``dispatch_groups`` / ``call_llm``) is intentionally out of scope here.
"""

from __future__ import annotations

import pytest

from src.labeling.correction_suggester import (
    ValueSuggestion,
    aggregate_samples,
    cache_key,
    parse_colclust_target_id,
    parse_llm_list,
    suggestions_to_payload,
)


# ---------------------------------------------------------------------------
# parse_colclust_target_id
# ---------------------------------------------------------------------------


def test_parse_colclust_target_id_valid():
    tid = "colclust\x00contacts.csv\x00State\x00c3"
    assert parse_colclust_target_id(tid) == ("contacts.csv", "State", "c3")


@pytest.mark.parametrize("bad", [
    "cell\x00s\x001\x00col",                 # wrong prefix
    "colclust\x00s\x00col",                   # too few parts
    "colclust\x00s\x00col\x00c1\x00extra",   # too many parts
    "not-a-target-id",
    "",
])
def test_parse_colclust_target_id_rejects(bad):
    assert parse_colclust_target_id(bad) is None


# ---------------------------------------------------------------------------
# parse_llm_list
# ---------------------------------------------------------------------------


def test_parse_llm_list_plain_object():
    assert parse_llm_list('{"possible_correct_values": ["CA", "California"]}') == [
        "CA", "California",
    ]


def test_parse_llm_list_fenced():
    text = '```json\n{"possible_correct_values": ["CA"]}\n```'
    assert parse_llm_list(text) == ["CA"]


def test_parse_llm_list_embedded_in_prose():
    text = 'Sure! {"possible_correct_values": ["CA"]} hope that helps'
    assert parse_llm_list(text) == ["CA"]


def test_parse_llm_list_strips_and_dedupes_preserving_order():
    text = '{"possible_correct_values": [" CA ", "CA", "", "California", "California"]}'
    assert parse_llm_list(text) == ["CA", "California"]


@pytest.mark.parametrize("text", [
    "",                                       # empty
    "not json at all",                        # unparseable
    '{"other_key": ["x"]}',                   # missing key -> None
    '{"possible_correct_values": "CA"}',      # not a list -> None
    '["CA"]',                                  # top-level not an object
])
def test_parse_llm_list_none_paths(text):
    assert parse_llm_list(text) is None


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


def _key(**over):
    base = dict(prompt="P", system_prompt="S", model="gpt-5.2",
                reasoning_effort="medium", sample_idx=0)
    base.update(over)
    return cache_key(**base)


def test_cache_key_stable_for_identical_inputs():
    assert _key() == _key()


@pytest.mark.parametrize("field,val", [
    ("prompt", "P2"),
    ("system_prompt", "S2"),
    ("model", "gpt-5.5"),
    ("reasoning_effort", "high"),
    ("sample_idx", 1),
])
def test_cache_key_changes_with_any_input(field, val):
    assert _key() != _key(**{field: val})


# ---------------------------------------------------------------------------
# aggregate_samples (K-shot voting)
# ---------------------------------------------------------------------------


def test_aggregate_majority_consensus():
    # 2/3 vote for A (threshold ceil(0.5*3)=2); B has 1 vote -> dropped.
    consensus, agreement, n_valid = aggregate_samples(
        [["A"], ["A"], ["B"]], threshold_frac=0.5,
    )
    assert consensus == ["A"]
    assert n_valid == 3
    assert agreement == pytest.approx(2 / 3)


def test_aggregate_drops_none_samples():
    consensus, agreement, n_valid = aggregate_samples(
        [["A"], None, ["A"]], threshold_frac=0.5,
    )
    assert consensus == ["A"]
    assert n_valid == 2
    assert agreement == pytest.approx(1.0)


def test_aggregate_all_none_returns_none():
    assert aggregate_samples([None, None], threshold_frac=0.5) == (None, 0.0, 0)


def test_aggregate_fallback_to_first_nonempty_when_no_consensus():
    # threshold_frac=0.9 over 2 samples -> threshold=2; A and B each get 1
    # vote, so nothing clears -> fall back to the FIRST non-empty sample.
    consensus, agreement, n_valid = aggregate_samples(
        [["A"], ["B"]], threshold_frac=0.9,
    )
    assert consensus == ["A"]
    assert n_valid == 2
    assert agreement == pytest.approx(0.5)


def test_aggregate_empty_lists_yield_no_consensus():
    # Valid (non-None) but empty samples: no votes, no non-empty fallback.
    assert aggregate_samples([[], []], threshold_frac=0.5) == (None, 1.0, 2)


# ---------------------------------------------------------------------------
# suggestions_to_payload
# ---------------------------------------------------------------------------


def test_suggestions_to_payload_shape_and_hides_samples():
    s = ValueSuggestion(
        source="a.csv", column="State", cluster_local_id="c1",
        value="Calif.", rows=[2, 7], suggestion=["California"],
        agreement=0.8, n_valid_samples=4, n_total_samples=5,
        samples=[["California"], None],
        would_drop_per_export_rule=False,
    )
    out = suggestions_to_payload([s])
    assert out == [{
        "value": "Calif.",
        "suggestion": ["California"],
        "agreement": 0.8,
        "n_valid_samples": 4,
        "n_total_samples": 5,
        "would_drop_per_export_rule": False,
        "rows": [2, 7],
    }]
    # The heavy per-sample raw list is intentionally not surfaced.
    assert "samples" not in out[0]


def test_suggestions_to_payload_empty():
    assert suggestions_to_payload([]) == []
