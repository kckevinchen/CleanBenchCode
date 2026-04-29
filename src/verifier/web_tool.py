"""Web-search providers behind a uniform :class:`WebSearchTool` protocol.

We picked **Tavily** as the default provider for this workload because:

* It is purpose-built for LLM/agentic pipelines: one call returns
  snippets, plus optionally a synthesised "answer" and the extracted
  raw page content. We don't need a separate fetch_url tool round-trip.
* The API is a single POST with a JSON body; no quota juggling, no
  multi-step OAuth.
* Free tier (1k searches / month) is enough for prototyping, and per-
  call cost stays well under a US cent at production scale.

Implementations here are intentionally small and side-effect-light so
the verifier can be unit-tested with a stub provider.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Protocol

import httpx

from .web_cache import WebCache

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One result row in a :class:`SearchResult`."""

    title: str
    url: str
    snippet: str = ""
    raw_content: str = ""
    score: float = 0.0


@dataclass
class SearchResult:
    """Provider response, normalised."""

    query: str
    hits: List[SearchHit] = field(default_factory=list)
    answer: str = ""             # provider-side synthesised answer
    cached: bool = False
    error: Optional[str] = None


class WebSearchTool(Protocol):
    """Minimal interface every provider must satisfy."""

    name: str

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
    ) -> SearchResult: ...


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


@dataclass
class TavilyWebSearch:
    """Tavily Search API client.

    Reads ``TAVILY_API_KEY`` from the environment at construction. Use
    ``api_key=...`` to override (e.g. tests).
    """

    api_key: Optional[str] = None
    cache: Optional[WebCache] = None
    timeout_seconds: float = 15.0
    base_url: str = "https://api.tavily.com"
    name: str = "tavily"

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "TavilyWebSearch initialised without an API key; "
                "every call will fail until TAVILY_API_KEY is set "
                "or api_key= is supplied.",
            )

    # --- internal --------------------------------------------------

    def _http_post(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        url = self.base_url.rstrip("/") + "/search"
        resp = httpx.post(
            url,
            json=body,
            timeout=self.timeout_seconds,
            headers={"content-type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse(payload: Mapping[str, Any]) -> tuple[List[SearchHit], str]:
        hits: List[SearchHit] = []
        for r in payload.get("results") or []:
            if not isinstance(r, Mapping):
                continue
            hits.append(SearchHit(
                title=str(r.get("title") or ""),
                url=str(r.get("url") or ""),
                snippet=str(r.get("content") or ""),
                raw_content=str(r.get("raw_content") or ""),
                score=float(r.get("score") or 0.0),
            ))
        return hits, str(payload.get("answer") or "")

    # --- public ----------------------------------------------------

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
    ) -> SearchResult:
        if not query or not query.strip():
            return SearchResult(query=query, error="empty query")

        if self.cache is not None:
            cached = self.cache.get(
                self.name, query,
                max_results=max_results,
                include_raw_content=include_raw_content,
            )
            if cached is not None:
                hits, answer = self._parse(cached)
                return SearchResult(
                    query=query, hits=hits, answer=answer, cached=True,
                )

        if not self.api_key:
            return SearchResult(query=query, error="missing TAVILY_API_KEY")

        body = {
            "api_key": self.api_key,
            "query": query,
            "max_results": int(max_results),
            "include_answer": True,
            "include_raw_content": bool(include_raw_content),
            "search_depth": "basic",
        }
        try:
            payload = self._http_post(body)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Tavily search failed for %r: %s", query, exc)
            return SearchResult(query=query, error=f"{type(exc).__name__}: {exc}")

        if self.cache is not None:
            # Strip the api_key from the cached body before persisting.
            self.cache.put(
                self.name, query, payload,
                max_results=max_results,
                include_raw_content=include_raw_content,
            )
        hits, answer = self._parse(payload)
        return SearchResult(query=query, hits=hits, answer=answer)


# ---------------------------------------------------------------------------
# Stub for tests
# ---------------------------------------------------------------------------


@dataclass
class StubWebSearch:
    """Test double that replays a fixed mapping ``query -> SearchResult``.

    Unmatched queries return an empty result. Tracks the call sequence
    on ``calls`` for assertions.
    """

    fixtures: dict[str, SearchResult] = field(default_factory=dict)
    name: str = "stub"
    calls: List[str] = field(default_factory=list)

    def search(
        self, query: str, *,
        max_results: int = 5, include_raw_content: bool = False,
    ) -> SearchResult:
        self.calls.append(query)
        res = self.fixtures.get(query)
        if res is None:
            return SearchResult(query=query, hits=[], answer="")
        # Return a shallow copy so callers can't mutate the fixture.
        return SearchResult(
            query=res.query or query,
            hits=list(res.hits),
            answer=res.answer,
            cached=res.cached,
            error=res.error,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_web_search_tool(
    provider: str = "tavily",
    *,
    cache: Optional[WebCache] = None,
    api_key: Optional[str] = None,
) -> WebSearchTool:
    """Build a :class:`WebSearchTool` for ``provider``.

    Currently supports ``tavily`` (default) and ``stub`` (no-op).
    """
    p = (provider or "").strip().lower()
    if p == "tavily":
        return TavilyWebSearch(api_key=api_key, cache=cache)
    if p == "stub":
        return StubWebSearch()
    raise ValueError(
        f"unknown web search provider {provider!r}; "
        "supported: tavily, stub"
    )


__all__ = [
    "SearchHit",
    "SearchResult",
    "WebSearchTool",
    "TavilyWebSearch",
    "StubWebSearch",
    "build_web_search_tool",
]
