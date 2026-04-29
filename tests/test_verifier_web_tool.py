"""Web cache + StubWebSearch behaviour."""

from __future__ import annotations

import time
from pathlib import Path

from src.verifier.web_cache import WebCache
from src.verifier.web_tool import (
    SearchHit,
    SearchResult,
    StubWebSearch,
    build_web_search_tool,
)


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
    # Force-expire by mutating the persisted ts.
    import json
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    rec["ts"] = time.time() - 9999
    files[0].write_text(json.dumps(rec))
    assert cache.get("p", "q") is None


def test_stub_search_replays_fixtures():
    fixtures = {
        "AcmeCo": SearchResult(
            query="AcmeCo",
            hits=[SearchHit(title="t", url="https://e.com", snippet="snip")],
        ),
    }
    stub = StubWebSearch(fixtures=fixtures)
    res = stub.search("AcmeCo")
    assert res.hits[0].url == "https://e.com"
    miss = stub.search("nope")
    assert miss.hits == []
    assert stub.calls == ["AcmeCo", "nope"]


def test_factory_builds_stub():
    tool = build_web_search_tool("stub")
    assert tool.name == "stub"


def test_factory_rejects_unknown_provider():
    import pytest
    with pytest.raises(ValueError):
        build_web_search_tool("nope")
