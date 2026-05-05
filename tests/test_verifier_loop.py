"""End-to-end Verifier loop tests with a StubAgentBackend.

Covers the contract:
  * confirmed_error verdicts pass through with canonical_value + evidence
  * fabricated URLs (not in the runtime's url_citation set) are dropped
  * confirmed_error w/o surviving evidence is downgraded to ambiguous
  * false_positive / ambiguous verdicts pass through cleanly
  * the system prompt from verify_entity.yaml reaches the backend
  * persistence (write_verifications -> load_verifications) round-trips
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from src.verifier import (
    AgentVerdict,
    Citation,
    StubAgentBackend,
    Verifier,
    VerifierConfig,
)
from src.verifier.gate import EntityKind
from src.verifier.verify import load_verifications, write_verifications


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "verify_entity.yaml"


def _make_cell_lookup(values: Dict[tuple, str]):
    def look(source, row, column):
        return values.get((source, row, column))
    return look


def _make_column_samples(samples: Dict[tuple, List[str]]):
    def get(source, column, n):
        return samples.get((source, column), [])[:n]
    return get


def _cluster(cid="c1", source="x.csv", row=0, columns=("Vendor",),
             error_types=("typo",), suggested_fix=None):
    c = {
        "cluster_id": cid, "source": source, "row": row,
        "columns": list(columns), "error_types": list(error_types),
    }
    if suggested_fix:
        c["suggested_fix"] = suggested_fix
    return c


def _verdict(text: str, urls: List[tuple[str, str]] = None,
             bing_urls: List[str] = None, n_calls: int = 1) -> AgentVerdict:
    return AgentVerdict(
        text=text,
        citations=[Citation(url=u, title=t) for (u, t) in (urls or [])],
        bing_query_urls=list(bing_urls or []),
        n_search_calls=n_calls,
    )


# ---------------------------------------------------------------------------


def test_confirmed_error_with_evidence_passes_through():
    cluster = _cluster(suggested_fix="Acme Corp")
    payload = json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "Acme Corp",
        "confidence": 0.92,
        "evidence": [{
            "url": "https://acme.example.com",
            "title": "Acme Corp - Home",
            "quote": "Acme Corp is a global supplier",
        }],
        "reasoning": "Both web hits surface the same official-site result.",
    })
    agent = StubAgentBackend(
        default=_verdict(
            payload,
            urls=[("https://acme.example.com", "Acme Corp - Home")],
            bing_urls=["https://www.bing.com/search?q=Acme+Corp"],
            n_calls=2,
        ),
    )

    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    out = verifier.run([cluster])
    assert len(out) == 1
    v = out[0]
    assert v.verdict == "confirmed_error"
    assert v.canonical_value == "Acme Corp"
    assert v.confidence == pytest.approx(0.92)
    assert len(v.evidence) == 1
    assert v.evidence[0].url == "https://acme.example.com"
    assert v.suspect_value == "AcmeCo"
    assert v.entity_kind == EntityKind.ORG.value
    assert v.queries == ["https://www.bing.com/search?q=Acme+Corp"]
    assert v.n_results == 2
    assert v.n_unique_urls == 1


def test_fabricated_url_is_dropped():
    """The agent JSON cited a URL that was NOT in the url_citation
    annotations the runtime recorded. Anti-hallucination filter must
    drop that citation. With no surviving evidence the verdict
    downgrades to ambiguous."""
    cluster = _cluster()
    payload = json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "Acme Corp",
        "confidence": 0.95,
        "evidence": [{
            "url": "https://hallucinated.example.com",  # not in citations
            "title": "Made up",
            "quote": "...",
        }],
        "reasoning": "see citation",
    })
    agent = StubAgentBackend(default=_verdict(
        payload, urls=[("https://real.example.com", "Acme Corp")],
    ))
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]
    assert v.verdict == "ambiguous"  # downgraded
    assert v.canonical_value is None
    assert v.evidence == []
    assert v.parse_error and "downgraded" in v.parse_error


def test_false_positive_passes_through():
    cluster = _cluster()
    payload = json.dumps({
        "verdict": "false_positive",
        "canonical_value": None,
        "confidence": 0.85,
        "evidence": [{
            "url": "https://acmeco.example.com",
            "title": "AcmeCo Inc.",
            "quote": "AcmeCo Inc. is the official name",
        }],
        "reasoning": "Web confirms AcmeCo is the canonical spelling.",
    })
    agent = StubAgentBackend(default=_verdict(
        payload, urls=[("https://acmeco.example.com", "AcmeCo Inc.")],
    ))
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]
    assert v.verdict == "false_positive"
    assert v.canonical_value is None
    assert len(v.evidence) == 1


def test_ambiguous_verdict_passes_through():
    cluster = _cluster()
    payload = json.dumps({
        "verdict": "ambiguous",
        "canonical_value": None,
        "confidence": 0.2,
        "evidence": [],
        "reasoning": "Web search returned nothing useful.",
    })
    agent = StubAgentBackend(default=_verdict(payload, urls=[], n_calls=0))
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]
    assert v.verdict == "ambiguous"
    assert v.evidence == []


def test_missing_cell_value_records_error_no_agent_call():
    cluster = _cluster()
    agent = StubAgentBackend()  # no fixtures -> would error if called
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({}),  # empty -> None -> error
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]
    assert v.error == "suspect cell value unavailable"
    assert agent.calls == []


def test_agent_error_propagates_to_verification():
    cluster = _cluster()
    agent = StubAgentBackend(default=AgentVerdict(error="agent run failed"))
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]
    assert v.error == "agent run failed"
    assert v.verdict == "ambiguous"  # default
    assert v.evidence == []


def test_system_prompt_reaches_backend():
    """The verifier must plumb the YAML system prompt onto the backend
    so the agent run uses the right ``instructions`` override."""
    cluster = _cluster()
    payload = json.dumps({
        "verdict": "ambiguous", "canonical_value": None,
        "confidence": 0.3, "evidence": [], "reasoning": "ok",
    })
    agent = StubAgentBackend(default=_verdict(payload))
    Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    ).run([cluster])
    assert agent.system_prompts and "data-quality reviewer" in agent.system_prompts[0]


def test_persistence_round_trip(tmp_path: Path):
    cluster = _cluster()
    payload = json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "Acme Corp",
        "confidence": 0.7,
        "evidence": [{"url": "https://u.example.com", "title": "t", "quote": "x"}],
        "reasoning": "ok",
    })
    agent = StubAgentBackend(default=_verdict(
        payload, urls=[("https://u.example.com", "t")],
    ))
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    records = verifier.run([cluster])

    out_path = tmp_path / "web_verifications.json"
    write_verifications(out_path, records)
    assert out_path.exists()

    loaded = load_verifications(out_path)
    assert len(loaded) == 1
    assert loaded[0].cluster_id == "c1"
    assert loaded[0].canonical_value == "Acme Corp"
    assert loaded[0].evidence[0].url == "https://u.example.com"


def test_row_context_passed_into_prompt():
    """The full row (column -> value) should reach the agent prompt
    when ``row_lookup`` is supplied. The flagged column must be
    marked with ' *'."""
    cluster = _cluster()
    payload = json.dumps({
        "verdict": "ambiguous", "canonical_value": None,
        "confidence": 0.3, "evidence": [], "reasoning": "ok",
    })
    agent = StubAgentBackend(default=_verdict(payload))

    rows = {
        ("x.csv", 0): {
            "Vendor": "AcmeCo",
            "Website": "https://acme.example.com",
            "City": "Springfield",
            "Phone": "555-0100",
        },
    }

    def row_lookup(source, row):
        return rows.get((source, row), {})

    Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        row_lookup=row_lookup,
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    ).run([cluster])
    assert len(agent.calls) == 1
    prompt = agent.calls[0]["prompt"]
    assert "Vendor *: AcmeCo" in prompt
    assert "Website" in prompt and "https://acme.example.com" in prompt
    assert "City" in prompt and "Springfield" in prompt
    assert "Phone" in prompt and "555-0100" in prompt


def test_row_context_optional_back_compat():
    """Verifier without ``row_lookup`` still works.

    With no row_lookup the verifier falls back to a 1-record snapshot
    that contains just the suspect column / value (so the agent at
    least sees the cell it's supposed to verify), and the rest of the
    row context is silently absent.
    """
    cluster = _cluster()
    payload = json.dumps({
        "verdict": "ambiguous", "canonical_value": None,
        "confidence": 0.3, "evidence": [], "reasoning": "ok",
    })
    agent = StubAgentBackend(default=_verdict(payload))
    Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    ).run([cluster])
    prompt = agent.calls[0]["prompt"]
    # Suspect cell still rendered in the RECORDS UNDER REVIEW block.
    assert "Vendor *: AcmeCo" in prompt
    # No extra column context (only the suspect column appears).
    assert "Website" not in prompt
    assert "City" not in prompt


def test_concurrency_runs_all_clusters():
    """Smoke: with concurrency=4 every cluster gets a verification record."""
    n = 10
    clusters = [_cluster(cid=f"c{i}") for i in range(n)]
    payload = json.dumps({
        "verdict": "ambiguous", "canonical_value": None,
        "confidence": 0.3, "evidence": [], "reasoning": "",
    })
    agent = StubAgentBackend(default=_verdict(payload))
    verifier = Verifier(
        agent=agent,
        cell_lookup=_make_cell_lookup(
            {("x.csv", 0, "Vendor"): "AcmeCo"}
        ),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=4),
    )
    out = verifier.run(clusters)
    assert len(out) == n
    assert [v.cluster_id for v in out] == sorted(c["cluster_id"] for c in clusters)
