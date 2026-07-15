"""Tests for ``src.core.row_dup.detector``.

We use a tiny fake ``LLMClient`` so the detector runs end-to-end
without touching Azure. The fake exposes the same surface
``RowDuplicationDetector`` consumes (``query``, ``endpoints``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.core.error_taxonomy import load_taxonomy
from src.core.llm_client import LLMResponse
from src.core.row_dup.detector import (
    AdjudicatorSettings,
    RowDuplicationDetector,
    _parse_verdict,
)
from src.adaptive.scoped_gt import ArmConfig
from src.core.table_digest import build_table_digest


TAXONOMY_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "error_types.yaml"
)


@dataclass
class _FakeLLM:
    """Returns one canned reply per call, in order."""

    responses: List[str] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=lambda: ["https://stub/"])

    def query(
        self,
        model,
        prompt,
        *,
        reasoning_effort="medium",
        reasoning_summary="detailed",
        system_prompt="",
    ):
        self.calls.append({"model": model, "prompt": prompt})
        text = self.responses.pop(0) if self.responses else "{}"
        return LLMResponse(
            text=text, endpoint="https://stub/", model=model, attempts=1
        )


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy(TAXONOMY_PATH)


# ---------------------------------------------------------------------------
# AdjudicatorSettings
# ---------------------------------------------------------------------------


class TestAdjudicatorSettings:
    def test_defaults(self) -> None:
        s = AdjudicatorSettings.from_config(None)
        assert s.detection_mode == "collision_adjudication"
        assert s.adjudicate_groups is True
        assert s.normalize_keys is True

    def test_overrides(self) -> None:
        s = AdjudicatorSettings.from_config(
            {
                "detection_mode": "legacy_chunk_scan",
                "max_key_subsets": 2,
                "max_groups_per_source": 5,
                "normalize_keys": False,
                "adjudicate_groups": False,
            }
        )
        assert s.detection_mode == "legacy_chunk_scan"
        assert s.max_key_subsets == 2
        assert s.max_groups_per_source == 5
        assert s.normalize_keys is False
        assert s.adjudicate_groups is False


# ---------------------------------------------------------------------------
# Verdict parser
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_round_trip(self) -> None:
        text = json.dumps(
            {"verdict": "duplicate", "confidence": "high", "explanation": "ok"}
        )
        v, err = _parse_verdict(text)
        assert err is None
        assert v["verdict"] == "duplicate"
        assert v["confidence"] == "high"

    def test_alternative_key_spelling(self) -> None:
        text = json.dumps({"label": "DUPLICATE", "explanation": "x"})
        v, _ = _parse_verdict(text)
        assert v["verdict"] == "duplicate"

    def test_missing_verdict_returns_none(self) -> None:
        text = json.dumps({"confidence": "high"})
        v, err = _parse_verdict(text)
        assert v is None
        assert "verdict" in err

    def test_unparseable_returns_none(self) -> None:
        v, err = _parse_verdict("not json at all")
        assert v is None
        assert err


# ---------------------------------------------------------------------------
# Group enumeration
# ---------------------------------------------------------------------------


class TestCollectGroups:
    def test_finds_cross_chunk_duplicate_pair(self, taxonomy) -> None:
        # 30 rows with one duplicate at indices 2 and 27 - well past
        # the typical chunk_size=10 boundary, which is exactly the
        # case the legacy chunk scan misses.
        emails = [f"user{i}@x.co" for i in range(30)]
        emails[27] = emails[2]  # pair across the table
        df = pd.DataFrame({"Email": emails, "Name": [f"N{i}" for i in range(30)]})
        digest = build_table_digest(df, source="big.csv")
        det = RowDuplicationDetector(
            llm=_FakeLLM(),
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(adjudicate_groups=False),
            digests={"big.csv": digest},
        )
        groups = det._collect_groups("big.csv", df, digest)
        assert len(groups) == 1
        assert set(groups[0].rows) == {2, 27}
        assert groups[0].key_columns == ("Email",)

    def test_max_groups_cap_applied(self, taxonomy) -> None:
        # Many duplicate pairs, but we cap at 2.
        df = pd.DataFrame({"K": ["a", "a", "b", "b", "c", "c", "d", "d"]})
        digest = build_table_digest(df, source="t.csv")
        det = RowDuplicationDetector(
            llm=_FakeLLM(),
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(
                adjudicate_groups=False, max_groups_per_source=2
            ),
        )
        groups = det._collect_groups("t.csv", df, digest)
        assert len(groups) == 2

    def test_normalised_match(self, taxonomy) -> None:
        df = pd.DataFrame(
            {
                "Org": ["Acme Corp.", "ACME CORP", "Other Inc"],
                "Id": [1, 2, 3],
            }
        )
        digest = build_table_digest(df, source="t.csv")
        det = RowDuplicationDetector(
            llm=_FakeLLM(),
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(
                adjudicate_groups=False, normalize_keys=True
            ),
        )
        groups = det._collect_groups("t.csv", df, digest)
        # Org collides under normalisation; rows 0 and 1 match.
        assert any(set(g.rows) == {0, 1} and g.evidence.get("normalized") for g in groups)


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------


class TestRun:
    def test_deterministic_emit_no_llm(self, taxonomy) -> None:
        df = pd.DataFrame(
            {
                "Email": ["a@x", "b@x", "a@x"],
                "Name": ["Ana", "Bob", "Ana"],
            }
        )
        digest = build_table_digest(df, source="t.csv")
        llm = _FakeLLM()
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(adjudicate_groups=False),
            digests={"t.csv": digest},
        )
        reports = det.run({"t.csv": df})
        assert len(reports) == 1
        assert llm.calls == []  # no LLM in deterministic mode
        r = reports[0]
        assert r.error_type == "row_duplication"
        assert r.issues
        issue = r.issues[0]
        assert sorted(issue["affected_rows"]) == [0, 2]
        assert "row_duplication" in issue["explanation"]

    def test_llm_adjudication_emits_when_duplicate(self, taxonomy) -> None:
        df = pd.DataFrame(
            {
                "Email": ["a@x", "b@x", "a@x"],
                "Name": ["Ana", "Bob", "Ana"],
            }
        )
        digest = build_table_digest(df, source="t.csv")
        llm = _FakeLLM(
            responses=[
                json.dumps(
                    {
                        "verdict": "duplicate",
                        "confidence": "high",
                        "observed": "2 duplicate rows",
                        "explanation": "Same email and name.",
                    }
                )
            ]
        )
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(),
            digests={"t.csv": digest},
            concurrency=1,
        )
        reports = det.run({"t.csv": df})
        assert len(reports) == 1
        assert len(llm.calls) == 1
        issue = reports[0].issues[0]
        assert issue["verdict"] == "duplicate"
        assert issue["confidence"] == "high"
        assert sorted(issue["affected_rows"]) == [0, 2]

    def test_llm_distinct_verdict_drops_issue(self, taxonomy) -> None:
        df = pd.DataFrame(
            {
                "Email": ["a@x", "b@x", "a@x"],
                "Name": ["Ana", "Bob", "Cy"],  # same email, different person
            }
        )
        digest = build_table_digest(df, source="t.csv")
        llm = _FakeLLM(
            responses=[
                json.dumps(
                    {
                        "verdict": "distinct",
                        "confidence": "high",
                        "explanation": "Different names.",
                    }
                )
            ]
        )
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(),
            digests={"t.csv": digest},
            concurrency=1,
        )
        reports = det.run({"t.csv": df})
        # Report exists for tracking, but no issue is emitted.
        assert len(reports) == 1
        assert reports[0].issues == []

    def test_no_groups_no_reports(self, taxonomy) -> None:
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        digest = build_table_digest(df, source="t.csv")
        det = RowDuplicationDetector(
            llm=_FakeLLM(),
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(),
            digests={"t.csv": digest},
        )
        assert det.run({"t.csv": df}) == []


# ---------------------------------------------------------------------------
# Per-column LLM canonicalisation
# ---------------------------------------------------------------------------


class TestKeyCanonicalizer:
    """Verify the canonicaliser collapses semantic variants and that
    its output flows into ``enumerate_collision_groups`` so rows with
    surface-different but semantically-equal keys end up in the same
    collision group."""

    def test_canonicalize_keys_groups_semantic_variants(self, taxonomy) -> None:
        # Three rows that share an entity ("Acme") but spell the
        # company name differently. Lexical normalisation collapses
        # case+punctuation but NOT abbreviations like "Inc." vs "Incorporated".
        df = pd.DataFrame(
            {
                "company": [
                    "Acme Corp",
                    "Acme Corporation",
                    "ACME Inc.",
                    "Globex",
                    "Globex Industries",
                    "OtherCo",
                ],
                "city": [
                    "NYC", "New York", "New York City",
                    "SF", "San Francisco", "Boston",
                ],
            }
        )
        # The canonicaliser is called twice (once per key column it
        # decides to evaluate). Provide a deterministic JSON response
        # for each. We don't assume the order - the detector dedups
        # by column name - so reuse the same mapping shape.
        # Because ``find_duplicate_subsets`` only surfaces columns
        # that ALREADY have key collisions, we set up the table so
        # ``city`` has an exact collision (NYC appears once, but to
        # induce collisions we keep keys clean) - simpler: pin the
        # canonicalisation through the column the digest selects.
        # In practice both columns are candidates; we feed canned
        # responses for both calls.
        canonical_response = json.dumps(
            {
                "groups": [
                    {
                        "canonical": "acme corporation",
                        "members": ["acme corp", "acme corporation", "acme inc"],
                    },
                    {
                        "canonical": "globex industries",
                        "members": ["globex", "globex industries"],
                    },
                    {"canonical": "otherco", "members": ["otherco"]},
                ]
            }
        )
        # Cover the city column too (consumed when find_duplicate_subsets
        # surfaces it as a candidate). If the detector doesn't ask about
        # it (because it has no exact collisions) the unused response
        # is harmless.
        city_response = json.dumps(
            {
                "groups": [
                    {
                        "canonical": "new york city",
                        "members": ["nyc", "new york", "new york city"],
                    },
                    {
                        "canonical": "san francisco",
                        "members": ["sf", "san francisco"],
                    },
                    {"canonical": "boston", "members": ["boston"]},
                ]
            }
        )
        # Two responses suffice; the detector caches per column so a
        # column visited in two subsets is only sent once.
        llm = _FakeLLM(responses=[canonical_response, city_response])
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(
                canonicalize_keys=True,
                adjudicate_groups=False,  # deterministic emit, easier to assert
                # Make sure the candidate-key search actually surfaces
                # ``company`` / ``city``: both are object dtype and
                # have collisions AFTER canonicalisation, so the
                # strict-equality scan would normally drop them. The
                # normalised pre-pass does surface them when
                # ``normalize_keys=True``.
                normalize_keys=True,
                max_key_size=1,
                max_key_subsets=4,
            ),
        )
        reports = det.run({"t.csv": df})
        # Expect at least one collision group covering 3 rows on the
        # canonicalised company name.
        sizes = sorted(len(r.issues[0]["affected_rows"]) for r in reports if r.issues)
        # Two semantic groups: company {0,1,2} (size 3) and one of
        # the others on city/company. We just assert that >=1 group
        # of size 3 exists, which is impossible without canonicalisation.
        assert any(s == 3 for s in sizes), (
            f"expected a 3-row semantic-duplicate group, got sizes={sizes}; "
            f"calls={len(llm.calls)}"
        )
        # The canonicalisation prompt should have been called at
        # least once.
        assert len(llm.calls) >= 1

    def test_canonicalize_keys_high_cardinality_skipped(self, taxonomy) -> None:
        """Columns above the cardinality cap fall back to identity."""
        # 10 distinct values but cap=5; canonicalizer should not be
        # called and detection falls back to lexical-only.
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(10)] + [f"v{i}" for i in range(10)],
                "x": list(range(20)),
            }
        )
        llm = _FakeLLM(responses=[])  # any LLM call would raise
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(
                canonicalize_keys=True,
                canonicalize_max_cardinality=5,
                adjudicate_groups=False,
                normalize_keys=True,
            ),
        )
        # Should not raise. May or may not produce groups depending on
        # whether the strict scan finds collisions on `id`.
        det.run({"t.csv": df})
        # Cardinality cap means no LLM canonicalisation calls.
        assert len(llm.calls) == 0


class TestParseCanonicalGroups:
    def test_simple_object(self) -> None:
        from src.core.row_dup.detector import _parse_canonical_groups
        groups, err = _parse_canonical_groups(
            json.dumps({"groups": [{"canonical": "a", "members": ["a", "b"]}]})
        )
        assert err is None
        assert groups == [{"canonical": "a", "members": ["a", "b"]}]

    def test_fenced_markdown(self) -> None:
        from src.core.row_dup.detector import _parse_canonical_groups
        text = '```json\n{"groups": [{"canonical": "x", "members": ["x"]}]}\n```'
        groups, err = _parse_canonical_groups(text)
        assert err is None
        assert len(groups) == 1

    def test_invalid_json(self) -> None:
        from src.core.row_dup.detector import _parse_canonical_groups
        groups, err = _parse_canonical_groups("not json at all")
        assert groups == []
        assert err is not None

    def test_missing_groups_key(self) -> None:
        from src.core.row_dup.detector import _parse_canonical_groups
        groups, err = _parse_canonical_groups(json.dumps({"foo": []}))
        assert groups == []
        assert err is not None


# ---------------------------------------------------------------------------
# Adaptive inner loop (cluster adjudication + canonicalisation)
# ---------------------------------------------------------------------------


class TestAdaptiveAdjudication:
    """The adaptive path runs each cluster's adjudication LLM call up
    to ``arm_config.k_max`` times, shuffles row order between runs, and
    unions the resulting issues. Recall-first: a species seen in any
    one run is preserved in the final report."""

    def test_adaptive_unions_issues_across_runs(self, taxonomy) -> None:
        df = pd.DataFrame(
            {
                "Email": ["a@x", "b@x", "a@x", "b@x"],
                "Name": ["Ana", "Bob", "Ana", "Bob"],
            }
        )
        digest = build_table_digest(df, source="t.csv")
        # Run 1 sees rows {0,2} as duplicate; run 2 (shuffled) ALSO
        # surfaces rows {1,3} as a separate duplicate. Union should
        # contain both issues. Provide enough responses for the worst
        # case (k_max calls per cluster, multiple clusters).
        run1 = json.dumps(
            {
                "issues": [
                    {
                        "verdict": "duplicate",
                        "affected_rows": [0, 2],
                        "explanation": "same email/name",
                    }
                ]
            }
        )
        run2 = json.dumps(
            {
                "issues": [
                    {
                        "verdict": "duplicate",
                        "affected_rows": [0, 2],
                        "explanation": "same email/name",
                    },
                    {
                        "verdict": "near_duplicate",
                        "affected_rows": [1, 3],
                        "explanation": "near match on email",
                    },
                ]
            }
        )
        # Repeat each so the two-consecutive predicate can fire.
        responses = [run1, run2, run2, run2, run2, run2, run2, run2, run2]
        llm = _FakeLLM(responses=responses)
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(),
            digests={"t.csv": digest},
            concurrency=1,
            arm_config=ArmConfig(
                k_max=3, k_min=2, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=True,
            ),
        )
        reports = det.run({"t.csv": df})
        assert reports
        # Find the cluster covering rows 0/2 and verify both species
        # surfaced. Some configurations may produce one report with
        # the unioned issues; assert via union of all issues.
        all_rows = set()
        verdicts = set()
        for r in reports:
            for issue in r.issues:
                verdicts.add(issue["verdict"])
                for row in issue["affected_rows"]:
                    all_rows.add(int(row))
                # adaptive metadata stamped onto every issue
                assert "discovery_runs" in issue
                assert "discovery_runs_total" in issue
                assert issue["discovery_runs_total"] >= 1
        # We must see the second-run-only species (rows 1,3 as
        # near_duplicate) propagated through.
        assert "near_duplicate" in verdicts
        assert {1, 3}.issubset(all_rows) or {0, 2}.issubset(all_rows)

    def test_adaptive_stops_early_when_runs_agree(self, taxonomy) -> None:
        """Identical responses across runs trip the two-consecutive stop."""
        df = pd.DataFrame(
            {"Email": ["a@x", "a@x"], "Name": ["Ana", "Ana"]}
        )
        digest = build_table_digest(df, source="t.csv")
        same = json.dumps(
            {
                "issues": [
                    {"verdict": "duplicate", "affected_rows": [0, 1]}
                ]
            }
        )
        # k_max=3 is the cap, but with require_two_consecutive=True
        # and identical species, runs 1+2 are no-new => stops at 3.
        llm = _FakeLLM(responses=[same] * 6)
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(),
            digests={"t.csv": digest},
            concurrency=1,
            arm_config=ArmConfig(
                k_max=5, k_min=2, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=True,
            ),
        )
        reports = det.run({"t.csv": df})
        # Single cluster, multiple LLM calls but bounded.
        # The two-consecutive predicate fires after 3 runs at most.
        assert len(llm.calls) <= 3
        assert reports
        for r in reports:
            for issue in r.issues:
                assert issue["discovery_runs_total"] <= 3


class TestAdaptiveCanonicalization:
    def test_adaptive_unions_partitions_via_union_find(self, taxonomy) -> None:
        """Two runs disagree on which values group; union-find
        merges classes that overlap in any single run."""
        df = pd.DataFrame(
            {
                "company": [
                    "Acme Corp",
                    "Acme Corporation",
                    "ACME Inc.",
                    "Globex",
                    "Globex Industries",
                    "OtherCo",
                ],
                "city": [
                    "NYC", "New York", "New York City",
                    "SF", "San Francisco", "Boston",
                ],
            }
        )
        # First run merges only acme corp + acme corporation;
        # second run additionally merges acme inc. Expected: union
        # produces a 3-way class for all three.
        run_a_company = json.dumps(
            {
                "groups": [
                    {
                        "canonical": "acme corporation",
                        "members": ["acme corp", "acme corporation"],
                    },
                    {"canonical": "acme inc", "members": ["acme inc"]},
                    {
                        "canonical": "globex industries",
                        "members": ["globex", "globex industries"],
                    },
                    {"canonical": "otherco", "members": ["otherco"]},
                ]
            }
        )
        run_b_company = json.dumps(
            {
                "groups": [
                    {
                        "canonical": "acme corporation",
                        "members": [
                            "acme corp", "acme corporation", "acme inc",
                        ],
                    },
                    {
                        "canonical": "globex industries",
                        "members": ["globex", "globex industries"],
                    },
                    {"canonical": "otherco", "members": ["otherco"]},
                ]
            }
        )
        run_c_company = run_b_company  # third run agrees with second
        # City column receives the same partition each run (so the
        # two-consecutive stop fires there earlier).
        city_resp = json.dumps(
            {
                "groups": [
                    {
                        "canonical": "new york city",
                        "members": ["nyc", "new york", "new york city"],
                    },
                    {
                        "canonical": "san francisco",
                        "members": ["sf", "san francisco"],
                    },
                    {"canonical": "boston", "members": ["boston"]},
                ]
            }
        )
        # Provide enough responses for both columns at full k_max.
        responses = [
            run_a_company, run_b_company, run_c_company,
            city_resp, city_resp, city_resp,
            # spare responses in case the detector visits more arms
            run_b_company, city_resp, city_resp, city_resp,
        ]
        llm = _FakeLLM(responses=responses)
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(
                canonicalize_keys=True,
                adjudicate_groups=False,  # deterministic emit
                normalize_keys=True,
                max_key_size=1,
                max_key_subsets=4,
            ),
            arm_config=ArmConfig(
                k_max=3, k_min=2, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=True,
            ),
        )
        reports = det.run({"t.csv": df})
        # Union-find should produce a 3-row collision group for the
        # three Acme spellings (rows 0, 1, 2).
        sizes = sorted(
            len(r.issues[0]["affected_rows"]) for r in reports if r.issues
        )
        assert any(s == 3 for s in sizes), (
            f"expected a 3-row Acme group from union-merged partitions, "
            f"got sizes={sizes}, calls={len(llm.calls)}"
        )


# ---------------------------------------------------------------------------
# whole_table mode (PoC: zero clustering)
# ---------------------------------------------------------------------------


class TestWholeTableMode:
    def test_single_shot_emits_duplicates(self, taxonomy) -> None:
        """k_max=1: feed the whole frame to one LLM call; emit the
        duplicate cluster verbatim."""
        df = pd.DataFrame(
            {
                "Email": ["a@x", "b@x", "a@x", "c@x"],
                "Name": ["Ana", "Bob", "Ana", "Cy"],
            }
        )
        digest = build_table_digest(df, source="t.csv")
        response = json.dumps(
            {
                "issues": [
                    {
                        "verdict": "duplicate",
                        "affected_rows": [0, 2],
                        "confidence": "high",
                        "observed": "row 0 == row 2",
                        "explanation": "same email and name",
                    }
                ]
            }
        )
        llm = _FakeLLM(responses=[response])
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(detection_mode="whole_table"),
            digests={"t.csv": digest},
            arm_config=ArmConfig(
                k_max=1, k_min=1, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=False,
            ),
        )
        reports = det.run({"t.csv": df})
        assert len(reports) == 1
        assert len(llm.calls) == 1
        # Prompt must contain every row index 0..3.
        prompt = llm.calls[0]["prompt"]
        assert "WHOLE-TABLE PASS" in prompt
        for idx in range(4):
            assert f"a@x" in prompt or "Email" in prompt
        # Rows are rendered with __row__ column carrying the index;
        # tabulate may pad whitespace, so just check substrings.
        assert "__row__" in prompt
        for idx in range(4):
            assert str(idx) in prompt
        r = reports[0]
        assert r.error_type == "row_duplication"
        assert r.prompt_name == "row_duplication.whole_table"
        assert len(r.issues) == 1
        issue = r.issues[0]
        assert sorted(issue["affected_rows"]) == [0, 2]
        assert issue["verdict"] == "duplicate"
        assert issue["discovery_strategy"] == "whole_table"
        assert issue["discovery_runs_total"] == 1

    def test_skips_when_table_exceeds_cap(self, taxonomy) -> None:
        df = pd.DataFrame({"K": list(range(10))})
        digest = build_table_digest(df, source="big.csv")
        llm = _FakeLLM()
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(
                detection_mode="whole_table", whole_table_max_rows=5,
            ),
            digests={"big.csv": digest},
            arm_config=ArmConfig(
                k_max=1, k_min=1, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=False,
            ),
        )
        reports = det.run({"big.csv": df})
        assert llm.calls == []
        assert len(reports) == 1
        r = reports[0]
        assert r.metadata.get("whole_table_skipped") is True
        assert "exceeds cap" in (r.error or "")

    def test_adaptive_unions_species_across_runs(self, taxonomy) -> None:
        """k_max>=2: two runs with overlapping but different species;
        the recall-first union must contain BOTH unique clusters."""
        df = pd.DataFrame(
            {
                "K1": ["a", "a", "b", "b", "c"],
                "K2": ["x", "x", "y", "y", "z"],
            }
        )
        digest = build_table_digest(df, source="t.csv")
        run_a = json.dumps(
            {
                "issues": [
                    {
                        "verdict": "duplicate",
                        "affected_rows": [0, 1],
                        "explanation": "K1+K2 equal",
                    }
                ]
            }
        )
        run_b = json.dumps(
            {
                "issues": [
                    {
                        "verdict": "duplicate",
                        "affected_rows": [0, 1],
                        "explanation": "K1+K2 equal",
                    },
                    {
                        "verdict": "near_duplicate",
                        "affected_rows": [2, 3],
                        "explanation": "K1+K2 equal too",
                    },
                ]
            }
        )
        # Provide spare responses for any extra adaptive runs.
        llm = _FakeLLM(responses=[run_a, run_b, run_b, run_b, run_b])
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(detection_mode="whole_table"),
            digests={"t.csv": digest},
            arm_config=ArmConfig(
                k_max=4, k_min=2, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=True,
            ),
        )
        reports = det.run({"t.csv": df})
        assert len(reports) == 1
        r = reports[0]
        species = {(i["verdict"], frozenset(i["affected_rows"])) for i in r.issues}
        assert ("duplicate", frozenset({0, 1})) in species
        assert ("near_duplicate", frozenset({2, 3})) in species
        # Both runs ran at least once.
        assert len(llm.calls) >= 2

    def test_empty_frame_returns_no_reports(self, taxonomy) -> None:
        df = pd.DataFrame({"K": []})
        digest = build_table_digest(df, source="t.csv")
        llm = _FakeLLM()
        det = RowDuplicationDetector(
            llm=llm,
            taxonomy=taxonomy,
            model="stub",
            settings=AdjudicatorSettings(detection_mode="whole_table"),
            digests={"t.csv": digest},
            arm_config=ArmConfig(
                k_max=1, k_min=1, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=False,
            ),
        )
        reports = det.run({"t.csv": df})
        assert reports == []
        assert llm.calls == []

    # -- per-CSV Good-Turing arms (whole_table adaptive support) ---------

    def _dup_response(self) -> str:
        return json.dumps({"issues": [{
            "verdict": "duplicate", "affected_rows": [0, 1],
            "explanation": "same key",
        }]})

    def test_records_saturated_per_csv_arm(self, taxonomy) -> None:
        """A multi-pass whole_table run that finds the same species every
        pass saturates -> one per-CSV arm with status 'saturated'."""
        df = pd.DataFrame({"K1": ["a", "a", "b"], "K2": ["x", "x", "y"]})
        digest = build_table_digest(df, source="t.csv")
        llm = _FakeLLM(responses=[self._dup_response()] * 5)
        det = RowDuplicationDetector(
            llm=llm, taxonomy=taxonomy, model="stub",
            settings=AdjudicatorSettings(detection_mode="whole_table"),
            digests={"t.csv": digest},
            arm_config=ArmConfig(
                k_max=4, k_min=2, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=True,
            ),
        )
        det.run({"t.csv": df})
        assert len(det.whole_table_arms) == 1
        arm = det.whole_table_arms[0]
        assert arm["scope_kind"] == "row_duplication"
        assert arm["scope_id"] == "t.csv"
        assert arm["error_type"] == "row_duplication"
        assert arm["status"] == "saturated"
        assert arm["stopped_reason"] != "k_max"
        assert arm["runs_completed"] >= 2
        assert arm["num_runs"] == arm["runs_completed"]
        # One distinct species: (duplicate, {0,1}).
        assert arm["n_species_observed"] == 1
        assert arm["n_observed"] == 1

    def test_records_exhausted_per_csv_arm(self, taxonomy) -> None:
        """k_max=1 hits the budget cap -> arm status 'exhausted'."""
        df = pd.DataFrame({"K1": ["a", "a", "b"], "K2": ["x", "x", "y"]})
        digest = build_table_digest(df, source="t.csv")
        llm = _FakeLLM(responses=[self._dup_response()])
        det = RowDuplicationDetector(
            llm=llm, taxonomy=taxonomy, model="stub",
            settings=AdjudicatorSettings(detection_mode="whole_table"),
            digests={"t.csv": digest},
            arm_config=ArmConfig(
                k_max=1, k_min=1, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=False,
            ),
        )
        det.run({"t.csv": df})
        assert len(det.whole_table_arms) == 1
        arm = det.whole_table_arms[0]
        assert arm["status"] == "exhausted"
        assert arm["stopped_reason"] == "k_max"
        assert arm["runs_completed"] == 1

    def test_collision_mode_records_no_arm(self, taxonomy) -> None:
        """The deterministic collision adjudicator never enters
        _run_whole_table, so it records no per-CSV arm."""
        df = pd.DataFrame({"Email": ["a@x", "a@x", "b@x"]})
        digest = build_table_digest(df, source="t.csv")
        det = RowDuplicationDetector(
            llm=_FakeLLM(responses=["{}"] * 5), taxonomy=taxonomy, model="stub",
            settings=AdjudicatorSettings(
                detection_mode="collision_adjudication",
                adjudicate_groups=False,
            ),
            digests={"t.csv": digest},
        )
        det.run({"t.csv": df})
        assert det.whole_table_arms == []

    def test_over_cap_records_no_arm(self, taxonomy) -> None:
        """An over-cap frame never calls run_adaptive -> no arm."""
        df = pd.DataFrame({"K": list(range(10))})
        digest = build_table_digest(df, source="big.csv")
        det = RowDuplicationDetector(
            llm=_FakeLLM(), taxonomy=taxonomy, model="stub",
            settings=AdjudicatorSettings(
                detection_mode="whole_table", whole_table_max_rows=5,
            ),
            digests={"big.csv": digest},
            arm_config=ArmConfig(
                k_max=1, k_min=1, epsilon=0.05, epsilon_chao1=0.05,
                stop_when="gt", require_two_consecutive=False,
            ),
        )
        det.run({"big.csv": df})
        assert det.whole_table_arms == []
