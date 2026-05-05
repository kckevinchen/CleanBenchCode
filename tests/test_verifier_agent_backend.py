"""WebCache + StubAgentBackend + AzureAgentBackend parsing helpers."""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

from src.verifier import (
    AgentVerdict,
    AzureAgentBackend,
    Citation,
    StubAgentBackend,
    build_agent_backend,
)
from src.verifier.web_cache import WebCache


# ---------------------------------------------------------------------------
# WebCache (now used as the verdict cache for AzureAgentBackend)
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path):
    cache = WebCache(tmp_path)
    assert cache.get("p", "q") is None
    cache.put("p", "q", {"answer": 42})
    assert cache.get("p", "q") == {"answer": 42}


def test_cache_keys_separate_providers(tmp_path: Path):
    cache = WebCache(tmp_path)
    cache.put("a", "q", {"v": 1})
    cache.put("b", "q", {"v": 2})
    assert cache.get("a", "q") == {"v": 1}
    assert cache.get("b", "q") == {"v": 2}


def test_cache_extra_kwargs_part_of_key(tmp_path: Path):
    cache = WebCache(tmp_path)
    cache.put("p", "q", {"v": 1}, max_results=3)
    cache.put("p", "q", {"v": 2}, max_results=5)
    assert cache.get("p", "q", max_results=3) == {"v": 1}
    assert cache.get("p", "q", max_results=5) == {"v": 2}
    assert cache.get("p", "q", max_results=99) is None


def test_cache_ttl_expires(tmp_path: Path):
    cache = WebCache(tmp_path, ttl_seconds=1)
    cache.put("p", "q", {"v": 1})
    assert cache.get("p", "q") == {"v": 1}
    import json
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    rec["ts"] = time.time() - 9999
    files[0].write_text(json.dumps(rec))
    assert cache.get("p", "q") is None


# ---------------------------------------------------------------------------
# StubAgentBackend
# ---------------------------------------------------------------------------


def test_stub_agent_replays_fixtures():
    fixtures = {
        "prompt-A": AgentVerdict(
            text='{"verdict": "ambiguous"}',
            citations=[Citation(url="https://e.com", title="t")],
        ),
    }
    stub = StubAgentBackend(fixtures=fixtures)
    res = stub.verify("prompt-A")
    assert res.text == '{"verdict": "ambiguous"}'
    assert res.citations[0].url == "https://e.com"
    miss = stub.verify("nope")
    assert miss.error == "no fixture for prompt"
    assert [c["prompt"] for c in stub.calls] == ["prompt-A", "nope"]


def test_factory_builds_stub():
    backend = build_agent_backend(stub=True)
    assert backend.name == "stub_agent"


def test_factory_builds_azure_agent():
    backend = build_agent_backend(endpoint="https://e", agent_id="asst_x")
    assert isinstance(backend, AzureAgentBackend)
    assert backend.endpoint == "https://e"
    assert backend.agent_id == "asst_x"


# ---------------------------------------------------------------------------
# AzureAgentBackend annotation / run-step parsing helpers
# ---------------------------------------------------------------------------


def _make_assistant_msg(text: str, urls: list[tuple[str, str]]):
    """Mock the SDK message shape (object with .role + .text_messages,
    each having .text.value + .text.annotations)."""
    annotations = [
        SimpleNamespace(
            type="url_citation",
            url_citation=SimpleNamespace(url=u, title=t),
        )
        for (u, t) in urls
    ]
    block = SimpleNamespace(
        text=SimpleNamespace(value=text, annotations=annotations),
    )
    return SimpleNamespace(role="assistant", text_messages=[block])


def test_extract_text_and_citations_picks_assistant():
    user_msg = SimpleNamespace(role="user", text_messages=[])
    assistant_msg = _make_assistant_msg(
        "Verdict text",
        [("https://a.com", "A"), ("https://b.com", "B")],
    )
    text, cites = AzureAgentBackend._extract_text_and_citations(
        [user_msg, assistant_msg]
    )
    assert text == "Verdict text"
    assert [c.url for c in cites] == ["https://a.com", "https://b.com"]
    assert [c.title for c in cites] == ["A", "B"]


def test_extract_text_and_citations_dedupes_urls():
    msg = _make_assistant_msg(
        "x",
        [("https://a.com", "A"), ("https://a.com", "A2")],
    )
    text, cites = AzureAgentBackend._extract_text_and_citations([msg])
    assert text == "x"
    assert len(cites) == 1
    assert cites[0].url == "https://a.com"


def test_extract_text_and_citations_dict_shape():
    """SDK sometimes returns plain dicts; helper must tolerate both."""
    block = {
        "text": SimpleNamespace(
            value="hello",
            annotations=[{
                "type": "url_citation",
                "url_citation": {"url": "https://c.com", "title": "C"},
            }],
        ),
    }
    msg = {"role": "assistant", "text_messages": [SimpleNamespace(**block)]}
    text, cites = AzureAgentBackend._extract_text_and_citations([msg])
    assert text == "hello"
    assert cites[0].url == "https://c.com"


def test_extract_text_and_citations_handles_enum_role():
    """Regression: the live SDK returns ``role=MessageRole.AGENT``,
    a ``str, Enum`` whose ``.value`` is ``"assistant"``. In Python
    3.12+ ``str(enum_member)`` returns ``"MessageRole.AGENT"`` (not
    the value), so a naive ``str(role).lower() == "assistant"`` check
    silently drops every assistant message and the verifier reports
    ``agent returned empty response`` even though the run completed."""
    class _Role(str, Enum):
        USER = "user"
        AGENT = "assistant"

    user = SimpleNamespace(role=_Role.USER, text_messages=[])
    assistant = SimpleNamespace(
        role=_Role.AGENT,
        text_messages=[SimpleNamespace(
            text=SimpleNamespace(value="payload", annotations=[]),
        )],
    )
    text, _ = AzureAgentBackend._extract_text_and_citations([user, assistant])
    assert text == "payload"


def test_extract_text_and_citations_falls_back_to_content():
    """Some SDK versions populate ``.content`` only; the helper must
    extract text from there too (filtering by ``type='text'``)."""
    text_block = SimpleNamespace(
        type="text",
        text=SimpleNamespace(value="from content", annotations=[]),
    )
    image_block = SimpleNamespace(type="image_file", image_file=object())
    msg = SimpleNamespace(
        role="assistant",
        text_messages=None,            # SDK didn't populate the helper
        content=[image_block, text_block],
    )
    text, _ = AzureAgentBackend._extract_text_and_citations([msg])
    assert text == "from content"


def test_extract_search_steps_rewrites_bing_host():
    step = SimpleNamespace(
        step_details=SimpleNamespace(tool_calls=[
            SimpleNamespace(
                web_search=SimpleNamespace(
                    request_url="https://api.bing.microsoft.com/v7.0/search?q=acme",
                ),
            ),
        ]),
    )
    urls, n = AzureAgentBackend._extract_search_steps([step])
    assert n == 1
    assert urls == ["https://www.bing.com/search?q=acme"]


def test_extract_search_steps_handles_legacy_bing_grounding():
    step = SimpleNamespace(
        step_details=SimpleNamespace(tool_calls=[
            SimpleNamespace(
                bing_grounding=SimpleNamespace(
                    requesturl="https://api.bing.microsoft.com/v7.0/search?q=foo",
                ),
            ),
        ]),
    )
    urls, n = AzureAgentBackend._extract_search_steps([step])
    assert n == 1
    assert urls == ["https://www.bing.com/search?q=foo"]


# ---------------------------------------------------------------------------
# AzureAgentBackend cache + error surfaces
# ---------------------------------------------------------------------------


def test_azure_agent_returns_cached_verdict(tmp_path: Path):
    cache = WebCache(tmp_path)
    cache.put("azure_agent", "ckey", {
        "text": "{}",
        "citations": [{"url": "https://x.com", "title": "X"}],
        "bing_query_urls": ["https://www.bing.com/search?q=x"],
        "n_search_calls": 2,
    })
    backend = AzureAgentBackend(
        endpoint="https://e", agent_id="asst_x", cache=cache,
    )
    v = backend.verify("ignored", cache_key="ckey")
    assert v.cached is True
    assert v.text == "{}"
    assert v.citations[0].url == "https://x.com"
    assert v.bing_query_urls == ["https://www.bing.com/search?q=x"]
    assert v.n_search_calls == 2


def test_azure_agent_missing_id_returns_error(monkeypatch):
    monkeypatch.delenv("AZURE_AGENTS_VERIFIER_AGENT_ID", raising=False)
    backend = AzureAgentBackend(endpoint="https://e", agent_id="")
    v = backend.verify("hello")
    assert v.error == "missing agent_id"


def test_azure_agent_empty_prompt_returns_error():
    backend = AzureAgentBackend(endpoint="https://e", agent_id="asst_x")
    v = backend.verify("   ")
    assert v.error == "empty prompt"
