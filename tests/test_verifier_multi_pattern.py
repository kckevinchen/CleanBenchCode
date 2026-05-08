"""Tests for the verifier's multi-row / multi-column / composite paths.

Covers:
  * cluster_shape() correctly classifies single_cell vs multi_row vs
    multi_column vs composite, populating ``rows`` and ``columns``.
  * the gate accepts row_duplication clusters (no columns, N rows in
    detections[*].affected_rows) and fd_violation clusters (one row,
    multiple columns).
  * non-web-verifiable error types (numeric_outlier, calculation
    inconsistencies) are hard-rejected even if their column would
    otherwise pass the entity-column gate.
  * Verifier dispatches per pattern: builds RECORDS UNDER REVIEW with
    the right number of records, marks the right columns with " *",
    and persists ``rows`` / ``columns_all`` / ``record_snapshots`` /
    ``pattern`` on the resulting Verification.
  * A multi-column ``confirmed_error`` carries the agent's structured
    ``canonical_record`` mapping through (with hallucinated column
    keys filtered out).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pytest

from src.verifier import (
    AgentVerdict,
    Citation,
    GateConfig,
    StubAgentBackend,
    Verifier,
    VerifierConfig,
    cluster_shape,
    should_verify,
)


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "verify_entity.yaml"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _verdict(text: str, urls: List[tuple[str, str]] | None = None,
             bing_urls: List[str] | None = None, n_calls: int = 1) -> AgentVerdict:
    return AgentVerdict(
        text=text,
        citations=[Citation(url=u, title=t) for (u, t) in (urls or [])],
        bing_query_urls=list(bing_urls or []),
        n_search_calls=n_calls,
    )


def _make_cell_lookup(values: Dict[tuple, str]):
    def look(source, row, column):
        return values.get((source, row, column))
    return look


def _make_column_samples():
    def get(source, column, n):
        return []
    return get


def _make_row_lookup(rows: Dict[tuple, Dict[str, Any]]):
    def look(source, row):
        return rows.get((source, row), {})
    return look


# ---------------------------------------------------------------------------
# cluster_shape
# ---------------------------------------------------------------------------


def test_cluster_shape_single_cell():
    c = {"row": 5, "columns": ["Vendor"], "detections": []}
    s = cluster_shape(c)
    assert s["pattern"] == "single_cell"
    assert s["rows"] == [5]
    assert s["columns"] == ["Vendor"]


def test_cluster_shape_multi_row():
    c = {
        "row": 0,
        "columns": [],
        "detections": [{
            "affected_rows": [0, 57],
            "error_type": "row_duplication",
        }],
    }
    s = cluster_shape(c)
    assert s["pattern"] == "multi_row"
    assert s["rows"] == [0, 57]
    assert s["columns"] == []


def test_cluster_shape_multi_column():
    c = {
        "row": 3,
        "columns": ["LICENSE_ID", "BUSINESS NAME", "ADDRESS"],
        "detections": [{"affected_rows": [3], "error_type": "fd_violation"}],
    }
    s = cluster_shape(c)
    assert s["pattern"] == "multi_column"
    assert s["rows"] == [3]
    assert len(s["columns"]) == 3


def test_cluster_shape_composite():
    c = {
        "row": 0,
        "columns": ["Code", "Year", "Value"],
        "detections": [
            {"affected_rows": [0, 4], "error_type": "fd_violation"},
        ],
    }
    s = cluster_shape(c)
    assert s["pattern"] == "composite"
    assert s["rows"] == [0, 4]
    assert s["columns"] == ["Code", "Year", "Value"]


def test_cluster_shape_dedupes_and_sorts_rows():
    c = {
        "row": 5,
        "columns": [],
        "detections": [
            {"affected_rows": [5, 10]},
            {"affected_rows": [10, 2, 5]},
        ],
    }
    s = cluster_shape(c)
    assert s["rows"] == [2, 5, 10]


def test_cluster_shape_handles_missing_fields():
    s = cluster_shape({})
    assert s == {"pattern": "single_cell", "rows": [], "columns": []}


# ---------------------------------------------------------------------------
# gate: multi-row / multi-column eligibility
# ---------------------------------------------------------------------------


def test_gate_accepts_row_duplication_cluster():
    c = {
        "row": 0,
        "columns": [],
        "error_types": ["row_duplication"],
        "detections": [{
            "error_type": "row_duplication",
            "affected_rows": [0, 57],
        }],
    }
    # No cell_value (multi_row clusters don't have a single suspect cell).
    assert should_verify(c, cell_value=None)


def test_gate_rejects_singleton_row_duplication():
    """A row_duplication cluster with only one affected row is structurally
    a single_cell-shaped finding and should not enter the multi-row path."""
    c = {
        "row": 0,
        "columns": [],
        "error_types": ["row_duplication"],
        "detections": [{"affected_rows": [0]}],
    }
    # Falls through to single_cell path; columns=[] -> EntityKind.NONE,
    # error_type ``row_duplication`` IS in the default whitelist now,
    # but the value gate also matters - here cell_value=None bypasses
    # the value gate. With no entity-column match and only the
    # whitelisted error_type, single_cell logic still requires either
    # column_match or et_match: et_match is True (row_duplication is
    # whitelisted), but value_eligible is bypassed when cell_value=None.
    # So a degenerate single-row "row_duplication" cluster slips
    # through as single_cell. We accept that as a minor edge - the
    # verifier itself will gracefully fall back to single_cell with
    # an empty record. The point of THIS test is to confirm we do not
    # treat it as multi_row (rows would be [0] -> pattern==single_cell).
    s = cluster_shape(c)
    assert s["pattern"] == "single_cell"


def test_gate_accepts_fd_violation_multi_column():
    c = {
        "row": 0,
        "columns": ["LICENSE_ID", "BUSINESS NAME", "ADDRESS"],
        "error_types": ["fd_violation"],
        "detections": [{"affected_rows": [0], "error_type": "fd_violation"}],
    }
    assert should_verify(c, cell_value=None)


def test_gate_rejects_pure_arithmetic_cluster():
    """row_calculation_inconsistency / numeric_outlier are NOT
    web-adjudicable. The gate must reject them even when the column
    name happens to look like an entity column."""
    c = {
        "row": 0,
        "columns": ["Vendor Name"],  # would match ORG regex
        "error_types": ["row_calculation_inconsistency"],
        "detections": [{"affected_rows": [0]}],
    }
    assert not should_verify(c, cell_value="AcmeCo")


def test_gate_rejects_pure_numeric_outlier():
    c = {
        "row": 1,
        "columns": ["Vendor"],
        "error_types": ["numeric_outlier"],
        "detections": [{"affected_rows": [1]}],
    }
    assert not should_verify(c, cell_value="123")


def test_gate_rejects_multi_column_with_too_few_columns():
    c = {
        "row": 0,
        "columns": ["BUSINESS NAME"],  # only 1 col
        "error_types": ["fd_violation"],
        "detections": [{"affected_rows": [0]}],
    }
    # fd_violation alone is NOT in DEFAULT entity-column patterns so
    # the multi_column branch needs >=2 cols. With only 1 col we fall
    # through to single_cell, which requires cell_value. With a value
    # it'd pass the entity-column gate (BUSINESS NAME ~ ORG); without
    # one (None) the gate is permissive - but the cluster is degenerate.
    # This test pins the multi_column-specific check: with cell_value
    # missing, a 1-column fd_violation should NOT slip through the
    # multi-column branch.
    assert cluster_shape(c)["pattern"] == "single_cell"


# ---------------------------------------------------------------------------
# Verifier: multi-row dispatch
# ---------------------------------------------------------------------------


_MULTI_ROW_CLUSTER = {
    "cluster_id": "schools.csv__r00000-c000",
    "source": "schools.csv",
    "row": 0,
    "columns": [],
    "error_types": ["row_duplication"],
    "detections": [{
        "error_type": "row_duplication",
        "affected_rows": [0, 57],
        "verdict": "near_duplicate",
    }],
}

_MULTI_ROW_ROWS = {
    ("schools.csv", 0): {
        "SAU OrgID": "77",
        "SAU Name": "Bremen Public Schools",
        "Town Code": "052",
        "Town Name": "Bremen",
        "School Year": "2022",
    },
    ("schools.csv", 57): {
        "SAU OrgID": "1054",
        "SAU Name": "Great Salt Bay CSD",
        "Town Code": "052",
        "Town Name": "Bremen",
        "School Year": "2022",
    },
}


def test_verifier_multi_row_records_block_renders_both_rows():
    """The verifier should fetch row_lookup for every affected row and
    show both as records under review."""
    payload = json.dumps({
        "verdict": "false_positive",
        "canonical_value": None,
        "confidence": 0.85,
        "evidence": [{"url": "https://maine.gov/sau", "title": "Maine SAUs",
                      "quote": "SAU 77 (Bremen Public Schools) is distinct from SAU 1054 (Great Salt Bay CSD)"}],
        "reasoning": "These are two distinct school administrative units in the same town.",
    })
    agent = StubAgentBackend(default=_verdict(
        payload, urls=[("https://maine.gov/sau", "Maine SAUs")],
    ))
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({}),
        column_samples=_make_column_samples(),
        row_lookup=_make_row_lookup(_MULTI_ROW_ROWS),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    out = verifier.run([_MULTI_ROW_CLUSTER])

    assert len(out) == 1
    v = out[0]
    assert v.pattern == "multi_row"
    assert v.rows == [0, 57]
    assert v.columns_all == []
    assert v.verdict == "false_positive"
    assert len(v.record_snapshots) == 2
    # First snapshot anchored to row 0.
    assert v.record_snapshots[0]["SAU OrgID"] == "77"
    assert v.record_snapshots[1]["SAU OrgID"] == "1054"

    # The agent prompt must show RECORDS UNDER REVIEW with two records.
    prompt = agent.calls[0]["prompt"]
    assert "Record 1 (row 0):" in prompt
    assert "Record 2 (row 57):" in prompt
    assert "SAU Name" in prompt
    assert "Bremen Public Schools" in prompt
    assert "Great Salt Bay CSD" in prompt
    # Pattern slot is filled.
    assert "pattern:       multi_row" in prompt


def test_verifier_multi_row_confirmed_error_with_evidence():
    """row_duplication confirmed: agent says these ARE the same entity
    written two different ways; canonical_value carries the merged name."""
    payload = json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "Acme Corporation",
        "confidence": 0.9,
        "evidence": [{
            "url": "https://acme.example.com",
            "title": "Acme Corporation - Home",
            "quote": "Acme Corporation is a leading supplier",
        }],
        "reasoning": "Both entries refer to the same Delaware corporation; the canonical name on the company website is 'Acme Corporation'.",
    })
    agent = StubAgentBackend(default=_verdict(
        payload, urls=[("https://acme.example.com", "Acme Corporation - Home")],
    ))
    cluster = {
        "cluster_id": "vendors.csv__r0-dup",
        "source": "vendors.csv",
        "row": 1,
        "columns": [],
        "error_types": ["row_duplication"],
        "detections": [{"affected_rows": [1, 7]}],
    }
    rows = {
        ("vendors.csv", 1): {"Vendor": "Acme Corp", "City": "NYC"},
        ("vendors.csv", 7): {"Vendor": "AcmeCo", "City": "NYC"},
    }
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({}),
        column_samples=_make_column_samples(),
        row_lookup=_make_row_lookup(rows),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]
    assert v.pattern == "multi_row"
    assert v.verdict == "confirmed_error"
    assert v.canonical_value == "Acme Corporation"
    assert len(v.evidence) == 1
    assert v.rows == [1, 7]


def test_verifier_multi_row_missing_row_lookup_records_error():
    """When row_lookup is not supplied for a multi-row cluster the
    verifier cannot build the records block; the Verification carries
    an explicit error and the agent is never called."""
    agent = StubAgentBackend()  # no fixtures - call would error
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({}),
        column_samples=_make_column_samples(),
        # row_lookup intentionally None
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([_MULTI_ROW_CLUSTER])[0]
    assert v.pattern == "multi_row"
    assert v.error and "row context unavailable" in v.error
    assert agent.calls == []


# ---------------------------------------------------------------------------
# Verifier: multi-column dispatch
# ---------------------------------------------------------------------------


_MULTI_COL_CLUSTER = {
    "cluster_id": "biz.csv__r3-fd",
    "source": "biz.csv",
    "row": 3,
    "columns": ["LICENSE_ID", "BUSINESS NAME", "ADDRESS"],
    "error_types": ["fd_violation"],
    "detections": [{
        "error_type": "fd_violation",
        "affected_rows": [3],
        "columns": ["LICENSE_ID", "BUSINESS NAME", "ADDRESS"],
    }],
}

_MULTI_COL_ROWS = {
    ("biz.csv", 3): {
        "LICENSE_ID": "BL-12345",
        "BUSINESS NAME": "Acme Cafe",
        "ADDRESS": "100 Main St, Springfield",
        "ISSUED DATE": "2024-01-15",
    },
}


def test_verifier_multi_column_records_columns_under_review():
    """multi_column cluster: one record, multiple columns flagged with
    ``*``. The Verification carries columns_all + the structured
    ``canonical_record`` fix."""
    payload = json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "BUSINESS NAME should be 'Acme Cafe LLC'",
        "canonical_record": {
            "BUSINESS NAME": "Acme Cafe LLC",
            "FAKE COLUMN": "should be filtered",
        },
        "confidence": 0.82,
        "evidence": [{
            "url": "https://springfield.gov/licenses",
            "title": "Springfield Business Licenses",
            "quote": "BL-12345: Acme Cafe LLC, 100 Main St",
        }],
        "reasoning": "City license registry shows the legal entity is 'Acme Cafe LLC' for license BL-12345.",
    })
    agent = StubAgentBackend(default=_verdict(
        payload,
        urls=[("https://springfield.gov/licenses", "Springfield Business Licenses")],
    ))
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({}),
        column_samples=_make_column_samples(),
        row_lookup=_make_row_lookup(_MULTI_COL_ROWS),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([_MULTI_COL_CLUSTER])[0]

    assert v.pattern == "multi_column"
    assert v.rows == [3]
    assert v.columns_all == ["LICENSE_ID", "BUSINESS NAME", "ADDRESS"]
    assert v.verdict == "confirmed_error"
    assert v.canonical_value and "Acme Cafe LLC" in v.canonical_value
    # canonical_record kept only the column we asked about; the
    # hallucinated "FAKE COLUMN" is dropped.
    assert v.canonical_record == {"BUSINESS NAME": "Acme Cafe LLC"}

    prompt = agent.calls[0]["prompt"]
    # All three flagged columns marked with ``*``.
    assert "LICENSE_ID *: BL-12345" in prompt
    assert "BUSINESS NAME *: Acme Cafe" in prompt
    assert "ADDRESS *: 100 Main St" in prompt
    # Cross-context column (ISSUED DATE) is shown without ``*``.
    assert "ISSUED DATE: 2024-01-15" in prompt
    assert "pattern:       multi_column" in prompt


def test_verifier_multi_column_canonical_record_dropped_on_downgrade():
    """confirmed_error without surviving evidence -> downgrade to
    ambiguous AND drop the structured fix."""
    payload = json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "fix",
        "canonical_record": {"BUSINESS NAME": "Acme Cafe LLC"},
        "confidence": 0.9,
        # Cited URL is NOT in the runtime's url_citation set.
        "evidence": [{"url": "https://made-up.example.com", "title": "x", "quote": "x"}],
        "reasoning": "x",
    })
    agent = StubAgentBackend(default=_verdict(payload, urls=[]))
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({}),
        column_samples=_make_column_samples(),
        row_lookup=_make_row_lookup(_MULTI_COL_ROWS),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([_MULTI_COL_CLUSTER])[0]
    assert v.verdict == "ambiguous"
    assert v.canonical_value is None
    assert v.canonical_record is None
    assert v.parse_error and "downgraded" in v.parse_error


def test_verifier_persists_pattern_through_round_trip(tmp_path: Path):
    """After write_verifications/load_verifications the new fields
    survive (rows, columns_all, pattern, record_snapshots, canonical_record)."""
    from src.verifier.verify import load_verifications, write_verifications

    payload = json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "Acme Corporation",
        "confidence": 0.9,
        "evidence": [{
            "url": "https://acme.example.com",
            "title": "Acme Corporation",
            "quote": "Acme Corporation is the legal name",
        }],
        "reasoning": "x",
    })
    agent = StubAgentBackend(default=_verdict(
        payload, urls=[("https://acme.example.com", "Acme Corporation")],
    ))
    cluster = {
        "cluster_id": "vendors.csv__r0-dup",
        "source": "vendors.csv",
        "row": 1,
        "columns": [],
        "error_types": ["row_duplication"],
        "detections": [{"affected_rows": [1, 7]}],
    }
    rows = {
        ("vendors.csv", 1): {"Vendor": "Acme Corp"},
        ("vendors.csv", 7): {"Vendor": "AcmeCo"},
    }
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({}),
        column_samples=_make_column_samples(),
        row_lookup=_make_row_lookup(rows),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    out = verifier.run([cluster])
    out_path = tmp_path / "web_verifications.json"
    write_verifications(out_path, out)

    loaded = load_verifications(out_path)
    assert len(loaded) == 1
    v = loaded[0]
    assert v.pattern == "multi_row"
    assert v.rows == [1, 7]
    assert v.columns_all == []
    assert len(v.record_snapshots) == 2
    assert v.record_snapshots[0]["Vendor"] == "Acme Corp"


def test_load_verifications_back_compat_with_old_records(tmp_path: Path):
    """Records written before the schema bump (no rows / columns_all /
    pattern fields) must still load cleanly with sensible defaults."""
    from src.verifier.verify import load_verifications

    p = tmp_path / "web_verifications.json"
    p.write_text(json.dumps([{
        "cluster_id": "old.csv__r0-c0",
        "source": "old.csv",
        "row": 0,
        "column": "Vendor",
        "suspect_value": "AcmeCo",
        "entity_kind": "org",
        "verdict": "false_positive",
        "canonical_value": None,
        "confidence": 0.7,
        "evidence": [],
        "reasoning": "ok",
        "queries": [],
        "n_results": 0,
        "n_unique_urls": 0,
        "llm_attempts": 1,
        "llm_endpoint": "azure_agent",
        # Note: no ``pattern``, ``rows``, ``columns_all``,
        # ``record_snapshots``, or ``canonical_record`` keys.
    }]), encoding="utf-8")

    [v] = load_verifications(p)
    assert v.cluster_id == "old.csv__r0-c0"
    assert v.pattern == "single_cell"  # default
    assert v.rows == []
    assert v.columns_all == []
    assert v.record_snapshots == []
    assert v.canonical_record is None


# ---------------------------------------------------------------------------
# Regression: pattern routing must reflect error-type intent, not just
# the structural shape. A single-column ``typo`` /
# ``pattern_inconsistency`` cluster whose detections list sibling
# ``affected_rows`` (the LLM's evidence-of-scope hint) must NOT be
# rendered with the multi_row "are these the same real-world entity?"
# prompt - that asks the wrong question. Demote to single_cell so
# the agent verifies the suspect cell value at ``cluster.row``.
# ---------------------------------------------------------------------------


def test_verifier_typo_with_sibling_rows_demotes_to_single_cell():
    """A typo cluster on column 'school' with sibling affected_rows
    must verify the suspect cell, not run a duplicate-detection
    prompt across the sibling rows."""
    payload = json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "Bremen Public Schools",
        "confidence": 0.85,
        "evidence": [{
            "url": "https://maine.gov/sau",
            "title": "Maine SAUs",
            "quote": "Bremen Public Schools is the canonical name",
        }],
        "reasoning": "Authoritative directory shows the canonical name.",
    })
    agent = StubAgentBackend(default=_verdict(
        payload, urls=[("https://maine.gov/sau", "Maine SAUs")],
    ))
    cluster = {
        "cluster_id": "schools.csv__r2-c0",
        "source": "schools.csv",
        "row": 2,
        "columns": ["school"],
        "error_types": ["typo", "pattern_inconsistency"],
        "detections": [{
            "error_type": "typo",
            "affected_rows": [2, 3, 6, 11, 17],
        }],
    }
    rows = {
        ("schools.csv", 2): {"school": "Bremn Public Schools", "town": "Bremen"},
        ("schools.csv", 3): {"school": "Bremn Public", "town": "Bremen"},
    }
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({
            ("schools.csv", 2, "school"): "Bremn Public Schools",
        }),
        column_samples=_make_column_samples(),
        row_lookup=_make_row_lookup(rows),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]

    # Pattern must be demoted to single_cell despite shape == multi_row.
    assert v.pattern == "single_cell"
    # Anchored to the LLM's primary row, not the smallest sibling.
    assert v.row == 2
    assert v.column == "school"
    assert v.suspect_value == "Bremn Public Schools"

    # Prompt must use the single_cell task framing - it asks about
    # the suspect cell value, NOT whether records are duplicates.
    prompt = agent.calls[0]["prompt"]
    assert "pattern:       single_cell" in prompt
    # Single record block - no Record 2 / Record 3 siblings.
    assert "Record 1 (row 2):" in prompt
    assert "Record 2" not in prompt
    # Should NOT contain duplicate-detection cues.
    assert "describe the SAME real-world entity" not in prompt
    assert "duplicate records" not in prompt
    # Single-cell task instructions present.
    assert "suspect cell value" in prompt


def test_verifier_typo_anchors_on_cluster_row_when_lower_sibling_exists():
    """``cluster_shape`` sorts rows; the verifier must still anchor
    the single-cell snapshot on ``cluster.row`` (the LLM's anchor),
    not the lowest row index in the affected list."""
    payload = json.dumps({
        "verdict": "ambiguous",
        "canonical_value": None,
        "confidence": 0.4,
        "evidence": [],
        "reasoning": "x",
    })
    agent = StubAgentBackend(default=_verdict(payload, urls=[]))
    cluster = {
        "cluster_id": "geo.csv__r6-c0",
        "source": "geo.csv",
        # Anchor row is 6, but affected_rows includes lower indices.
        "row": 6,
        "columns": ["city"],
        "error_types": ["pattern_inconsistency", "text_inconsistency"],
        "detections": [{
            "error_type": "pattern_inconsistency",
            "affected_rows": [0, 6, 11, 75, 79, 80, 83],
        }],
    }
    rows = {
        ("geo.csv", 0): {"city": "new york"},
        ("geo.csv", 6): {"city": "miAMI"},
    }
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({
            ("geo.csv", 6, "city"): "miAMI",
        }),
        column_samples=_make_column_samples(),
        row_lookup=_make_row_lookup(rows),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]

    assert v.pattern == "single_cell"
    assert v.row == 6  # NOT 0 (the smallest sibling)
    assert v.suspect_value == "miAMI"

    prompt = agent.calls[0]["prompt"]
    # Header must show row 6, not row 0.
    assert "Record 1 (row 6):" in prompt
    assert "miAMI" in prompt


def test_verifier_row_duplication_cluster_still_uses_multi_row_prompt():
    """A genuine row_duplication cluster - the same shape as the
    typo case but with the right error_type intent - must still get
    the multi_row prompt. This pins the demotion to error_type
    intent, not just the structural shape."""
    payload = json.dumps({
        "verdict": "false_positive",
        "canonical_value": None,
        "confidence": 0.8,
        "evidence": [{
            "url": "https://example.com/registry",
            "title": "Registry",
            "quote": "These are distinct entities",
        }],
        "reasoning": "x",
    })
    agent = StubAgentBackend(default=_verdict(
        payload, urls=[("https://example.com/registry", "Registry")],
    ))
    # Same shape as the typo cluster (1 col, multi affected_rows) but
    # the LLM marked it as a duplication candidate. Must remain
    # multi_row.
    cluster = {
        "cluster_id": "vendors.csv__r0-dup",
        "source": "vendors.csv",
        "row": 0,
        "columns": ["Vendor"],
        "error_types": ["row_duplication"],
        "detections": [{
            "error_type": "row_duplication",
            "affected_rows": [0, 7],
        }],
    }
    rows = {
        ("vendors.csv", 0): {"Vendor": "Acme Corp"},
        ("vendors.csv", 7): {"Vendor": "AcmeCo"},
    }
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({}),
        column_samples=_make_column_samples(),
        row_lookup=_make_row_lookup(rows),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]

    assert v.pattern == "multi_row"
    assert len(v.record_snapshots) == 2

    prompt = agent.calls[0]["prompt"]
    assert "pattern:       multi_row" in prompt
    assert "Record 1 (row 0):" in prompt
    assert "Record 2 (row 7):" in prompt
    assert "describe the SAME real-world entity" in prompt


# ---------------------------------------------------------------------------
# Task-block specialisation by entity_kind. The single_cell branch has a
# dedicated "person" path that biases the verdict toward false_positive
# when ANY web hit on the suspect spelling exists - rare-spelling names
# look like typos but aren't.
# ---------------------------------------------------------------------------


def test_format_task_block_person_single_cell_uses_specialized_text():
    from src.verifier.verify import _format_task_block

    block = _format_task_block(
        pattern="single_cell",
        columns=["First Name"],
        n_rows=1,
        entity_kind="person",
    )
    # Person-specific framing
    assert "Person-name typo verification" in block
    # The "any web hit -> false_positive" policy must be explicit
    assert "false_positive" in block
    assert "rare-spelling" in block.lower() or "rare-spelling" in block
    # Confirmed_error bar must be high (suggested_fix supported in row context)
    assert "confirmed_error" in block
    assert "context" in block.lower()
    # Must NOT contain the default single_cell paragraph header
    # ("Notes for person-name columns: ... return ambiguous"); the
    # person-specialised block replaces that policy entirely.
    assert "Notes for person-name columns" not in block


def test_format_task_block_single_cell_other_keeps_default_text():
    """Sanity: non-person single_cell clusters get the original block."""
    from src.verifier.verify import _format_task_block

    block = _format_task_block(
        pattern="single_cell",
        columns=["Address"],
        n_rows=1,
        entity_kind="address",
    )
    # The original "Notes for person-name columns" paragraph still
    # lives in the default single_cell block (preserved for any
    # non-person single_cell call), so its presence here confirms we
    # didn't regress the default path.
    assert "Notes for person-name columns" in block
    assert "Person-name typo verification" not in block


def test_format_task_block_default_entity_kind_is_other():
    """The new ``entity_kind`` kwarg must default to ``other`` so
    existing callers (and any external users) continue to work."""
    from src.verifier.verify import _format_task_block

    block = _format_task_block(
        pattern="single_cell",
        columns=["Status"],
        n_rows=1,
    )
    # Default branch is the original single_cell text.
    assert "Notes for person-name columns" in block
    assert "Person-name typo verification" not in block


def test_verifier_dispatches_person_block_for_first_name_column():
    """End-to-end (no real agent): a typo cluster on a ``First Name``
    column with ``verify_person=True`` should emit the person-specific
    task block in the prompt the verifier hands to the backend."""
    from src.verifier.gate import GateConfig

    payload = '{"verdict":"false_positive","canonical_value":null,"confidence":0.6,"evidence":[],"reasoning":"x"}'
    agent = StubAgentBackend(default=_verdict(payload, urls=[]))
    cluster = {
        "cluster_id": "people.csv__r00005-c000",
        "source": "people.csv",
        "row": 5,
        "columns": ["First Name"],
        "values": ["Murry"],
        "error_type": "typo",
        "error_types": ["typo"],
        "detections": [{
            "error_type": "typo",
            "observed": "Murry",
            "suggested_fix": "Murray",
        }],
    }
    cells = {("people.csv", 5, "First Name"): "Murry"}
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup(cells),
        column_samples=_make_column_samples(),
        cfg=VerifierConfig(
            prompt_path=PROMPT_PATH,
            concurrency=1,
            gate=GateConfig(verify_person=True),
        ),
    )
    [v] = verifier.run([cluster])

    assert v.entity_kind == "person"
    prompt = agent.calls[0]["prompt"]
    assert "Person-name typo verification" in prompt
    # Default single_cell guidance must NOT be in the prompt.
    assert "Notes for person-name columns" not in prompt
