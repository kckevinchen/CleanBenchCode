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
    build_cluster_index,
    cache_key,
    load_column_cluster_records,
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


# ---------------------------------------------------------------------------
# build_cluster_index: operator-saved record merge (column_clusters.jsonl)
# ---------------------------------------------------------------------------


def _draft_doc(source, column, clusters):
    return {"source": source, "column": column, "draft_clusters": clusters}


def _record_doc(source, column, clusters, status="labeled"):
    return {
        "source": source,
        "column": column,
        "status": status,
        "clusters": clusters,
    }


def test_build_cluster_index_resolves_operator_added_cluster():
    # The LLM drafted nothing for this column (empty draft doc), but the
    # operator hand-added a "missed inconsistency" cluster c1 and gave it a
    # tp verdict. Without the record merge this column 404s in the suggester.
    src, col = "permits.csv", "Engineer Phone"
    drafts = {(src, col): _draft_doc(src, col, [])}
    target_id = f"colclust\x00{src}\x00{col}\x00c1"
    labels = {target_id: {"target_id": target_id, "status": "tp"}}
    records = {
        (src, col): _record_doc(src, col, [{
            "cluster_id": "c1",
            "kind": "numerical_format",
            "canonical_value": "(651) 602-4712",
            "llm_proposed_canonical": None,
            "values": [
                {"value": "651-602-4702", "count": 33},
                {"value": "(651) 602-4712", "count": 32},
                {"value": "(651) 602-4724", "count": 23},
            ],
        }]),
    }

    idx = build_cluster_index(drafts, labels, records=records)
    ctxs = idx[(src, col)]
    assert [c.local_id for c in ctxs] == ["c1"]
    c1 = ctxs[0]
    assert c1.target_id == target_id
    assert c1.status == "tp"
    assert c1.value_set == frozenset(
        {"651-602-4702", "(651) 602-4712", "(651) 602-4724"}
    )
    # Operator's confirmed canonical comes off the record.
    assert c1.confirmed_canonical == "(651) 602-4712"


def test_build_cluster_index_without_records_skips_operator_cluster():
    # Regression guard: the bug was that the suggester only read drafts.
    src, col = "permits.csv", "Engineer Phone"
    drafts = {(src, col): _draft_doc(src, col, [])}
    idx = build_cluster_index(drafts, {})
    assert (src, col) not in idx


def test_build_cluster_index_labeled_record_overrides_draft():
    # When a cluster id exists in BOTH a draft and a labelled record, the
    # operator's edited record wins: it is the disjoint partition the
    # workbench shows (``list_column_clusters``) and validates verdicts
    # against (``post_label``). The draft's narrower value-set must NOT
    # shadow the values the operator added.
    src, col = "geo.csv", "State"
    drafts = {(src, col): _draft_doc(src, col, [
        {"kind": "casing", "canonical": None, "values": ["CA", "ca"]},
    ])}
    records = {(src, col): _record_doc(src, col, [{
        "cluster_id": "c1",
        "kind": "casing",
        "canonical_value": "CA",
        "values": [
            {"value": "CA", "count": 5},
            {"value": "ca", "count": 1},
            {"value": "Ca", "count": 1},
        ],
    }])}

    idx = build_cluster_index(drafts, {}, records=records)
    ctxs = idx[(src, col)]
    assert [c.local_id for c in ctxs] == ["c1"]
    # The operator's edited 3-value record wins, not the draft's 2 values.
    assert ctxs[0].value_set == frozenset({"CA", "ca", "Ca"})


def test_build_cluster_index_labeled_record_replaces_all_drafts():
    # Real-world repro (run 20260610T175821Z, column "Key#2 Phone"): the
    # LLM drafted c1 (5 parenthesised phones), c2 and c3; the operator
    # merged everything they cared about into a single 7-value c1 and saved
    # a labelled record. The index must expose the operator's 7-value c1
    # and drop the obsolete draft c2/c3 -- matching what the labelled
    # column shows in the UI. Before the fix the draft's 5-value c1 shadowed
    # the record, so the two dotted-format values got no suggestion.
    src, col = "f.csv", "Key#2 Phone"
    drafts = {(src, col): _draft_doc(src, col, [
        {"kind": "pattern", "values": [
            "(405) 213-1880", "(612) 465-7305", "(703) 203-5440",
            "(703) 917-7524", "(757) 350-6138",
        ]},
        {"kind": "pattern", "values": ["202.863.9281 x820", "703-462-1538 x617"]},
        {"kind": "pattern", "values": ["202.863.9281 x820", "914.262.6981"]},
    ])}
    records = {(src, col): _record_doc(src, col, [{
        "cluster_id": "c1",
        "kind": "text",
        "canonical_value": "(405) 213-1880",
        "values": [
            {"value": "(405) 213-1880", "count": 1},
            {"value": "(612) 465-7305", "count": 1},
            {"value": "(703) 203-5440", "count": 1},
            {"value": "(703) 917-7524", "count": 1},
            {"value": "(757) 350-6138", "count": 1},
            {"value": "202.863.9281 x820", "count": 1},
            {"value": "914.262.6981", "count": 1},
        ],
    }])}

    idx = build_cluster_index(drafts, {}, records=records)
    ctxs = idx[(src, col)]
    # Only the operator's c1 survives; draft c2/c3 are gone.
    assert [c.local_id for c in ctxs] == ["c1"]
    assert len(ctxs[0].value_set) == 7
    assert "202.863.9281 x820" in ctxs[0].value_set
    assert "914.262.6981" in ctxs[0].value_set


def test_build_cluster_index_appends_record_only_ids_alongside_drafts():
    # Drafts cover c1; the operator additionally added c2 by hand. Both
    # should surface, each at its own local_id.
    src, col = "geo.csv", "City"
    drafts = {(src, col): _draft_doc(src, col, [
        {"kind": "casing", "canonical": None, "values": ["NYC", "nyc"]},
    ])}
    records = {(src, col): _record_doc(src, col, [
        {
            "cluster_id": "c1",
            "kind": "casing",
            "canonical_value": "NYC",
            "values": [
                {"value": "NYC", "count": 3},
                {"value": "nyc", "count": 1},
            ],
        },
        {
            "cluster_id": "c2",
            "kind": "pattern",
            "canonical_value": "Saint Paul",
            "values": [
                {"value": "Saint Paul", "count": 4},
                {"value": "St. Paul", "count": 2},
            ],
        },
    ])}

    idx = build_cluster_index(drafts, {}, records=records)
    by_id = {c.local_id: c for c in idx[(src, col)]}
    assert set(by_id) == {"c1", "c2"}
    # The labelled record defines both clusters; its c1 value-set happens
    # to equal the draft's, and c2 is contributed by the record only.
    assert by_id["c1"].value_set == frozenset({"NYC", "nyc"})
    assert by_id["c2"].value_set == frozenset({"Saint Paul", "St. Paul"})


def test_build_cluster_index_skips_degenerate_record_clusters():
    # A record cluster with fewer than two values is dropped, matching the
    # draft path's degenerate-cluster guard.
    src, col = "geo.csv", "Zip"
    records = {(src, col): _record_doc(src, col, [{
        "cluster_id": "c1",
        "kind": "pattern",
        "values": [{"value": "55101", "count": 1}],
    }])}
    idx = build_cluster_index({}, {}, records=records)
    assert (src, col) not in idx


def test_load_column_cluster_records_latest_wins(tmp_path):
    path = tmp_path / "column_clusters.jsonl"
    src, col = "permits.csv", "Engineer Phone"
    import json as _json
    path.write_text(
        _json.dumps(_record_doc(src, col, [{
            "cluster_id": "c1", "kind": "pattern",
            "values": [{"value": "a", "count": 1}, {"value": "b", "count": 1}],
        }])) + "\n"
        + _json.dumps(_record_doc(src, col, [{
            "cluster_id": "c1", "kind": "pattern",
            "values": [
                {"value": "a", "count": 1},
                {"value": "b", "count": 1},
                {"value": "c", "count": 1},
            ],
        }])) + "\n",
        encoding="utf-8",
    )
    recs = load_column_cluster_records(path)
    # Latest line wins: the 3-value record, not the first 2-value one.
    vals = recs[(src, col)]["clusters"][0]["values"]
    assert len(vals) == 3


def test_load_column_cluster_records_missing_file(tmp_path):
    assert load_column_cluster_records(tmp_path / "nope.jsonl") == {}
