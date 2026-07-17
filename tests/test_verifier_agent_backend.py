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
    """When no agent_id can be resolved at all (no arg, no env, AND
    the default pool has been disabled), ``verify()`` must surface
    ``missing agent_id`` instead of crashing on a network call.

    We bypass the default-pool fallback by patching the constants
    so the test exercises the explicit ``not self.agent_id`` branch."""
    monkeypatch.delenv("AZURE_AGENTS_VERIFIER_AGENT_ID", raising=False)
    monkeypatch.delenv("AZURE_AGENTS_VERIFIER_AGENT_IDS", raising=False)
    import src.verifier.agent_backend as ab
    monkeypatch.setattr(ab, "DEFAULT_AGENT_IDS", [])
    monkeypatch.setattr(ab, "DEFAULT_AGENT_ENDPOINTS", [])
    backend = AzureAgentBackend(endpoint="https://e", agent_id="")
    v = backend.verify("hello")
    assert v.error == "missing agent_id"


def test_azure_agent_empty_prompt_returns_error():
    backend = AzureAgentBackend(endpoint="https://e", agent_id="asst_x")
    v = backend.verify("   ")
    assert v.error == "empty prompt"


# ---------------------------------------------------------------------------
# AzureAgentBackend run-status diagnostics + empty-payload cache behaviour
#
# These tests exercise the failure-mode paths that previously returned
# a vanilla ``AgentVerdict(error="agent returned empty response")``
# regardless of why the run produced no text. The fix:
#   1. Surfaces the run id + ``incomplete_details.reason`` so the
#      caller can tell content-filter blocks from real timeouts.
#   2. Skips the cache write when the agent produced no useful payload
#      (text empty AND no citations) so a transient empty result is
#      not replayed forever on subsequent runs.
# ---------------------------------------------------------------------------


def _fake_client_factory(*, run, messages=(), run_steps=(), thread_id="thr_1"):
    """Build a fake Foundry agents client object whose ``runs``,
    ``messages``, and ``run_steps`` namespaces return the values the
    test wants. Used by the run-status / cache tests below to drive
    AzureAgentBackend.verify without actually contacting Azure."""
    thread = SimpleNamespace(id=thread_id)
    return SimpleNamespace(
        threads=SimpleNamespace(create=lambda: thread),
        messages=SimpleNamespace(
            create=lambda thread_id, role, content: None,
            list=lambda thread_id, **_: list(messages),
        ),
        runs=SimpleNamespace(
            create_and_process=lambda **_: run,
        ),
        run_steps=SimpleNamespace(
            list=lambda thread_id, run_id: list(run_steps),
        ),
    )


def test_azure_agent_failed_run_includes_message_and_id():
    backend = AzureAgentBackend(endpoint="https://e", agent_id="asst_x")
    run = SimpleNamespace(
        id="run_failed",
        status="failed",
        last_error=SimpleNamespace(message="rate_limit_exceeded"),
        incomplete_details=None,
    )
    backend._client = _fake_client_factory(run=run)
    v = backend.verify("hello")
    assert v.error and "rate_limit_exceeded" in v.error
    assert "run_failed" in v.error


def test_azure_agent_cancelled_run_surfaces_status():
    backend = AzureAgentBackend(endpoint="https://e", agent_id="asst_x")
    run = SimpleNamespace(
        id="run_cancel",
        status="cancelled",
        last_error=None,
        incomplete_details=None,
    )
    backend._client = _fake_client_factory(run=run)
    v = backend.verify("hello")
    assert v.error and "cancelled" in v.error and "run_cancel" in v.error


def test_azure_agent_completed_with_incomplete_details_reports_reason():
    """Foundry returns ``status=completed`` with ``incomplete_details``
    when the model output was suppressed (e.g. content_filter, hard
    token cap). The verifier must surface the ``reason`` so empty
    responses for person names are diagnosable."""
    backend = AzureAgentBackend(endpoint="https://e", agent_id="asst_x")
    empty_msg = SimpleNamespace(role="assistant", text_messages=[])
    run = SimpleNamespace(
        id="run_blocked",
        status="completed",
        last_error=None,
        incomplete_details=SimpleNamespace(reason="content_filter"),
    )
    backend._client = _fake_client_factory(run=run, messages=[empty_msg])
    v = backend.verify("hello")
    assert v.error and "content_filter" in v.error and "run_blocked" in v.error


def test_azure_agent_does_not_cache_empty_payloads(tmp_path: Path):
    """Regression: empty payloads (no text, no citations) used to be
    cached as if they were a successful empty verdict. Once cached,
    every subsequent run for that cluster would replay the empty
    result without ever re-hitting the agent. With the fix, the
    cache is written ONLY when the verdict carries useful signal."""
    cache = WebCache(tmp_path)
    backend = AzureAgentBackend(
        endpoint="https://e", agent_id="asst_x", cache=cache,
    )
    empty_msg = SimpleNamespace(role="assistant", text_messages=[])
    run = SimpleNamespace(
        id="run_empty",
        status="completed",
        last_error=None,
        incomplete_details=None,
    )
    backend._client = _fake_client_factory(run=run, messages=[empty_msg])

    v = backend.verify("hello", cache_key="cid::deadbeef")
    assert v.error and "run_empty" in v.error
    # Empty payload must NOT be cached.
    assert cache.get(backend.name, "cid::deadbeef") is None


def test_azure_agent_caches_non_empty_payloads(tmp_path: Path):
    """Sanity-check that a real (text-bearing) verdict still gets
    cached after the empty-payload guard."""
    cache = WebCache(tmp_path)
    backend = AzureAgentBackend(
        endpoint="https://e", agent_id="asst_x", cache=cache,
    )
    msg = _make_assistant_msg('{"verdict": "ambiguous"}', [])
    run = SimpleNamespace(
        id="run_ok",
        status="completed",
        last_error=None,
        incomplete_details=None,
    )
    backend._client = _fake_client_factory(run=run, messages=[msg])

    v = backend.verify("hello", cache_key="cid::cafebabe")
    assert v.error is None
    assert v.text == '{"verdict": "ambiguous"}'
    assert cache.get(backend.name, "cid::cafebabe") is not None


# ---------------------------------------------------------------------------
# RoundRobinAgentBackend
# ---------------------------------------------------------------------------


from src.verifier import RoundRobinAgentBackend  # noqa: E402


def test_round_robin_dispatches_evenly():
    """Each call advances the index, cycling through every backend."""
    a = StubAgentBackend(default=AgentVerdict(text="from-a"))
    b = StubAgentBackend(default=AgentVerdict(text="from-b"))
    rr = RoundRobinAgentBackend(backends=[a, b])

    out = [rr.verify(f"q{i}").text for i in range(5)]
    assert out == ["from-a", "from-b", "from-a", "from-b", "from-a"]
    # Each child saw the prompts that were routed to it.
    assert [c["prompt"] for c in a.calls] == ["q0", "q2", "q4"]
    assert [c["prompt"] for c in b.calls] == ["q1", "q3"]


def test_round_robin_propagates_system_prompt():
    """Verifier writes ``system_prompt`` onto its agent attr after
    construction. RoundRobinAgentBackend must forward that write
    down to every child so they all use the same instructions
    override on their next run."""
    a = StubAgentBackend()
    b = StubAgentBackend()
    rr = RoundRobinAgentBackend(backends=[a, b])
    rr.system_prompt = "SYS"
    assert a.system_prompt == "SYS"
    assert b.system_prompt == "SYS"
    # And subsequent verify() calls record the propagated prompt.
    a.fixtures["q"] = AgentVerdict(text="x")
    b.fixtures["q"] = AgentVerdict(text="y")
    rr.verify("q")
    rr.verify("q")
    assert a.system_prompts == ["SYS"]
    assert b.system_prompts == ["SYS"]


def test_round_robin_requires_at_least_one_backend():
    import pytest
    with pytest.raises(ValueError):
        RoundRobinAgentBackend(backends=[])


def test_round_robin_thread_safe_under_concurrency():
    """Counter is guarded by a Lock; concurrent threads must each get
    a distinct backend so total dispatch counts are balanced. We
    use 2 backends and 100 calls: each backend should be hit
    exactly 50 times regardless of thread interleaving."""
    import threading

    a = StubAgentBackend(default=AgentVerdict(text="a"))
    b = StubAgentBackend(default=AgentVerdict(text="b"))
    rr = RoundRobinAgentBackend(backends=[a, b])

    def hit():
        for i in range(50):
            rr.verify(f"t-{i}")

    threads = [threading.Thread(target=hit) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 100 total calls, 2 backends -> exactly 50 each.
    assert len(a.calls) + len(b.calls) == 100
    assert len(a.calls) == 50
    assert len(b.calls) == 50


# ---------------------------------------------------------------------------
# build_agent_backend multi-endpoint factory
# ---------------------------------------------------------------------------


def test_factory_single_endpoint_returns_azure_backend():
    backend = build_agent_backend(endpoint="https://a", agent_id="asst_a")
    assert isinstance(backend, AzureAgentBackend)
    assert backend.endpoint == "https://a"


def test_factory_multi_endpoint_via_lists_returns_round_robin():
    backend = build_agent_backend(
        endpoints=["https://a", "https://b"],
        agent_ids=["asst_a", "asst_b"],
    )
    assert isinstance(backend, RoundRobinAgentBackend)
    assert len(backend.backends) == 2
    inner = backend.backends
    assert isinstance(inner[0], AzureAgentBackend)
    assert isinstance(inner[1], AzureAgentBackend)
    assert (inner[0].endpoint, inner[0].agent_id) == ("https://a", "asst_a")
    assert (inner[1].endpoint, inner[1].agent_id) == ("https://b", "asst_b")


def test_factory_multi_endpoint_via_csv_returns_round_robin():
    """Comma-separated strings on the singular CLI args are split and
    paired positionally - the natural form when a flag accepts only
    one value but the operator wants several."""
    backend = build_agent_backend(
        endpoint="https://a, https://b",
        agent_id="asst_a,asst_b",
    )
    assert isinstance(backend, RoundRobinAgentBackend)
    assert [b.endpoint for b in backend.backends] == ["https://a", "https://b"]
    assert [b.agent_id for b in backend.backends] == ["asst_a", "asst_b"]


def test_factory_mismatched_lengths_raises():
    import pytest
    with pytest.raises(ValueError):
        build_agent_backend(
            endpoints=["https://a", "https://b"],
            agent_ids=["asst_a"],
        )


def test_factory_reads_plural_env_vars(monkeypatch):
    monkeypatch.setenv(
        "AZURE_AGENTS_ENDPOINTS", "https://a, https://b",
    )
    monkeypatch.setenv(
        "AZURE_AGENTS_VERIFIER_AGENT_IDS", "asst_a,asst_b",
    )
    monkeypatch.delenv("AZURE_AGENTS_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_AGENTS_VERIFIER_AGENT_ID", raising=False)
    backend = build_agent_backend()
    assert isinstance(backend, RoundRobinAgentBackend)
    assert [b.endpoint for b in backend.backends] == ["https://a", "https://b"]


def test_factory_singular_env_vars_still_work(monkeypatch):
    """Backwards-compat: AZURE_AGENTS_ENDPOINT (singular) still
    yields a single AzureAgentBackend (no overhead vs. the legacy
    code path)."""
    monkeypatch.setenv("AZURE_AGENTS_ENDPOINT", "https://only")
    monkeypatch.setenv(
        "AZURE_AGENTS_VERIFIER_AGENT_ID", "asst_only",
    )
    monkeypatch.delenv("AZURE_AGENTS_ENDPOINTS", raising=False)
    monkeypatch.delenv("AZURE_AGENTS_VERIFIER_AGENT_IDS", raising=False)
    backend = build_agent_backend()
    assert isinstance(backend, AzureAgentBackend)
    assert backend.endpoint == "https://only"
    assert backend.agent_id == "asst_only"


def test_factory_without_configuration_returns_unconfigured_backend(monkeypatch):
    for var in (
        "AZURE_AGENTS_ENDPOINT",
        "AZURE_AGENTS_VERIFIER_AGENT_ID",
        "AZURE_AGENTS_ENDPOINTS",
        "AZURE_AGENTS_VERIFIER_AGENT_IDS",
    ):
        monkeypatch.delenv(var, raising=False)

    backend = build_agent_backend()
    assert isinstance(backend, AzureAgentBackend)
    assert backend.endpoint == ""
    assert backend.agent_id == ""


def test_azure_agent_without_configuration_stays_unconfigured(monkeypatch):
    for var in (
        "AZURE_AGENTS_ENDPOINT",
        "AZURE_AGENTS_VERIFIER_AGENT_ID",
        "AZURE_AGENTS_ENDPOINTS",
        "AZURE_AGENTS_VERIFIER_AGENT_IDS",
    ):
        monkeypatch.delenv(var, raising=False)

    backend = AzureAgentBackend()
    assert backend.endpoint == ""
    assert backend.agent_id == ""
