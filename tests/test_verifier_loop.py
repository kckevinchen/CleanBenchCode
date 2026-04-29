"""End-to-end Verifier loop tests with a stubbed LLM and StubWebSearch.

Covers the contract:
  * confirmed_error verdicts pass through with canonical_value + evidence
  * fabricated URLs (not in the snippets we showed the model) are dropped
  * confirmed_error w/o surviving evidence is downgraded to ambiguous
  * false_positive / ambiguous verdicts pass through cleanly
  * persistence (write_verifications -> load_verifications) round-trips
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pytest

from src.core.llm_client import LLMResponse
from src.verifier import Verifier, VerifierConfig
from src.verifier.gate import EntityKind, GateConfig
from src.verifier.verify import load_verifications, write_verifications
from src.verifier.web_tool import SearchHit, SearchResult, StubWebSearch


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "verify_entity.yaml"


@dataclass
class StubLLM:
    """LLMClient stand-in: returns a canned text per call.

    Mirrors the public ``query()`` signature of :class:`LLMClient` so
    :class:`Verifier` cannot tell the difference. Tracks every call on
    ``calls`` for assertions.
    """

    responses: List[str]
    calls: List[Dict] = None  # type: ignore[assignment]
    endpoints: List[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.calls = []
        self.endpoints = ["https://stub/"]

    def query(self, model, prompt, *, reasoning_effort="medium",
              reasoning_summary="detailed", system_prompt=""):
        self.calls.append({"model": model, "prompt": prompt, "system": system_prompt})
        text = self.responses.pop(0) if self.responses else "{}"
        return LLMResponse(text=text, endpoint="https://stub/", model=model, attempts=1)


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


# ---------------------------------------------------------------------------


def test_confirmed_error_with_evidence_passes_through():
    cluster = _cluster(suggested_fix="Acme Corp")
    web = StubWebSearch(fixtures={
        '"AcmeCo" official site': SearchResult(query="q", hits=[
            SearchHit(title="Acme Corp - Home",
                      url="https://acme.example.com",
                      snippet="Acme Corp is a global supplier ..."),
        ]),
        '"Acme Corp" official site': SearchResult(query="q", hits=[
            SearchHit(title="Acme Corp",
                      url="https://acme.example.com",
                      snippet="Welcome to Acme Corp"),
        ]),
    })
    llm = StubLLM(responses=[json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "Acme Corp",
        "confidence": 0.92,
        "evidence": [{
            "url": "https://acme.example.com",
            "title": "Acme Corp - Home",
            "quote": "Acme Corp is a global supplier",
        }],
        "reasoning": "Both queries surface the same official-site hit.",
    })])

    verifier = Verifier(
        llm=llm, web=web,
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
    assert v.queries == ['"AcmeCo" official site', '"Acme Corp" official site']


def test_fabricated_url_is_dropped():
    """The LLM cited a URL that was NOT in the snippets we showed it.
    Anti-hallucination filter must drop that citation. With no
    surviving evidence the verdict downgrades to ambiguous."""
    cluster = _cluster()
    web = StubWebSearch(fixtures={
        '"AcmeCo" official site': SearchResult(query="q", hits=[
            SearchHit(title="Acme Corp", url="https://real.example.com",
                      snippet="..."),
        ]),
    })
    llm = StubLLM(responses=[json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "Acme Corp",
        "confidence": 0.95,
        "evidence": [{
            "url": "https://hallucinated.example.com",  # not in snippets
            "title": "Made up",
            "quote": "...",
        }],
        "reasoning": "see citation",
    })])
    verifier = Verifier(
        llm=llm, web=web,
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
    web = StubWebSearch(fixtures={
        '"AcmeCo" official site': SearchResult(query="q", hits=[
            SearchHit(title="AcmeCo Inc.", url="https://acmeco.example.com",
                      snippet="AcmeCo Inc. is the official name."),
        ]),
    })
    llm = StubLLM(responses=[json.dumps({
        "verdict": "false_positive",
        "canonical_value": None,
        "confidence": 0.85,
        "evidence": [{
            "url": "https://acmeco.example.com",
            "title": "AcmeCo Inc.",
            "quote": "AcmeCo Inc. is the official name",
        }],
        "reasoning": "Web confirms AcmeCo is the canonical spelling.",
    })])
    verifier = Verifier(
        llm=llm, web=web,
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
    web = StubWebSearch()  # no fixtures -> empty results
    llm = StubLLM(responses=[json.dumps({
        "verdict": "ambiguous",
        "canonical_value": None,
        "confidence": 0.2,
        "evidence": [],
        "reasoning": "Web search returned nothing useful.",
    })])
    verifier = Verifier(
        llm=llm, web=web,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]
    assert v.verdict == "ambiguous"
    assert v.evidence == []


def test_missing_cell_value_records_error_no_llm_call():
    cluster = _cluster()
    llm = StubLLM(responses=[])   # MUST NOT be called.
    web = StubWebSearch()
    verifier = Verifier(
        llm=llm, web=web,
        # Empty lookup -> None -> error.
        cell_lookup=_make_cell_lookup({}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    v = verifier.run([cluster])[0]
    assert v.error == "suspect cell value unavailable"
    assert llm.calls == []
    assert web.calls == []


def test_persistence_round_trip(tmp_path: Path):
    cluster = _cluster()
    web = StubWebSearch(fixtures={
        '"AcmeCo" official site': SearchResult(query="q", hits=[
            SearchHit(title="t", url="https://u.example.com", snippet="x"),
        ]),
    })
    llm = StubLLM(responses=[json.dumps({
        "verdict": "confirmed_error",
        "canonical_value": "Acme Corp",
        "confidence": 0.7,
        "evidence": [{"url": "https://u.example.com", "title": "t", "quote": "x"}],
        "reasoning": "ok",
    })])
    verifier = Verifier(
        llm=llm, web=web,
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
    """The full row (column -> value) should reach the LLM prompt
    when ``row_lookup`` is supplied. The flagged column must be
    marked with ' *' so the model knows which value is suspect."""
    cluster = _cluster()
    web = StubWebSearch(fixtures={
        '"AcmeCo" official site': SearchResult(query="q", hits=[
            SearchHit(title="t", url="https://u.example.com", snippet="x"),
        ]),
    })
    llm = StubLLM(responses=[json.dumps({
        "verdict": "ambiguous", "canonical_value": None,
        "confidence": 0.3, "evidence": [], "reasoning": "ok",
    })])

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

    verifier = Verifier(
        llm=llm, web=web,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        row_lookup=row_lookup,
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    verifier.run([cluster])
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["prompt"]
    # Suspect column flagged with " *".
    assert "Vendor *: AcmeCo" in prompt
    # Other columns present (verbatim values).
    assert "Website" in prompt and "https://acme.example.com" in prompt
    assert "City" in prompt and "Springfield" in prompt
    assert "Phone" in prompt and "555-0100" in prompt


def test_row_context_optional_back_compat():
    """Verifier without ``row_lookup`` still works (existing
    callers don't have to migrate). The prompt's row_context block
    just reads '(unavailable)'."""
    cluster = _cluster()
    web = StubWebSearch(fixtures={
        '"AcmeCo" official site': SearchResult(query="q", hits=[
            SearchHit(title="t", url="https://u.example.com", snippet="x"),
        ]),
    })
    llm = StubLLM(responses=[json.dumps({
        "verdict": "ambiguous", "canonical_value": None,
        "confidence": 0.3, "evidence": [], "reasoning": "ok",
    })])
    verifier = Verifier(
        llm=llm, web=web,
        cell_lookup=_make_cell_lookup({("x.csv", 0, "Vendor"): "AcmeCo"}),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=1),
    )
    verifier.run([cluster])
    assert "(unavailable)" in llm.calls[0]["prompt"]


def test_concurrency_runs_all_clusters():
    """Smoke: with concurrency=4 every cluster gets a verification record."""
    n = 10
    clusters = [_cluster(cid=f"c{i}") for i in range(n)]
    web = StubWebSearch(fixtures={
        '"AcmeCo" official site': SearchResult(query="q", hits=[
            SearchHit(title="t", url="https://u.example.com", snippet="x"),
        ]),
    })
    llm = StubLLM(responses=[json.dumps({
        "verdict": "ambiguous", "canonical_value": None,
        "confidence": 0.3, "evidence": [], "reasoning": "",
    })] * n)
    verifier = Verifier(
        llm=llm, web=web,
        cell_lookup=_make_cell_lookup(
            {("x.csv", 0, "Vendor"): "AcmeCo"}
        ),
        column_samples=_make_column_samples({}),
        cfg=VerifierConfig(prompt_path=PROMPT_PATH, concurrency=4),
    )
    out = verifier.run(clusters)
    assert len(out) == n
    # All ids present, sorted.
    assert [v.cluster_id for v in out] == sorted(c["cluster_id"] for c in clusters)
