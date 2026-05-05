"""Web-grounded verification backend backed by Microsoft Foundry Agents.

The verifier used to chain (Tavily search) + (Azure OpenAI LLM call) per
cluster. We replaced that pipeline with a single Foundry Agents call:
the agent has the **Web Search** tool attached and decides for itself
how many searches to run. We pass the cluster context as the user
message, override the run's ``instructions`` with the system prompt
from :mod:`prompts/verify_entity.yaml`, then parse the assistant's JSON
reply plus its ``url_citation`` annotations.

Why agents-as-verifier instead of "web tool returning snippets":

* Bing-grounded tools (Web Search, Grounding with Bing) NEVER return
  raw snippet text to developers; only the agent itself sees that text.
  That makes a Tavily-style provider impossible on Azure.
* The agent's run steps include the Bing query URL it used; we surface
  those alongside the citations so the labeling UI can show
  "show me the search results" links per cluster.

Auth uses :class:`AzureCliCredential` so the deployment story matches
:mod:`src.core.llm_client` - one ``az login`` and you're done.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol

from .web_cache import WebCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend response shapes
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """One ``url_citation`` annotation from an assistant message."""

    url: str
    title: str = ""


@dataclass
class AgentVerdict:
    """Normalised response from one agent run."""

    text: str = ""                          # assistant message body
    citations: List[Citation] = field(default_factory=list)
    bing_query_urls: List[str] = field(default_factory=list)
    n_search_calls: int = 0                 # web_search tool calls in run steps
    cached: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Backend Protocol
# ---------------------------------------------------------------------------


class AgentBackend(Protocol):
    """Minimal interface every backend must satisfy.

    The verifier calls :meth:`verify` exactly once per cluster, passing
    the user-facing prompt (system prompt is set on the backend at
    construction time). The backend is responsible for any required
    threading, polling, and citation extraction.
    """

    name: str

    def verify(
        self,
        prompt: str,
        *,
        cache_key: Optional[str] = None,
    ) -> AgentVerdict: ...


# ---------------------------------------------------------------------------
# Real Foundry Agents backend
# ---------------------------------------------------------------------------


@dataclass
class AzureAgentBackend:
    """Foundry Agents-backed verifier.

    Args:
        endpoint:    Project endpoint URL, e.g.
                     ``https://<resource>.services.ai.azure.com/api/projects/<project>``.
                     Falls back to ``AZURE_AGENTS_ENDPOINT`` when blank.
        agent_id:    ID of a pre-created agent (``asst_...``) that has
                     the Web Search tool attached. Falls back to
                     ``AZURE_AGENTS_VERIFIER_AGENT_ID``.
        system_prompt: System prompt to pass as the run's ``instructions``
                     override. Set by :class:`Verifier` from
                     ``prompts/verify_entity.yaml``.
        cache:       Optional :class:`WebCache` for verdict caching so
                     repeated runs over the same cluster don't burn
                     agent calls.
        warn_after_seconds: ``runs.create_and_process`` is synchronous,
                     so we cannot cancel it mid-flight from here.
                     Once it returns we log a WARNING when wall-clock
                     time exceeded this threshold so operators see
                     slow runs in the tail. To bound real wall-clock
                     time, set the timeout on the agent itself in the
                     Foundry portal (the SDK respects it server-side).
    """

    endpoint: str = ""
    agent_id: str = ""
    system_prompt: str = ""
    cache: Optional[WebCache] = None
    warn_after_seconds: float = 120.0
    name: str = "azure_agent"
    _client: Any = field(init=False, default=None, repr=False)
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False,
    )

    def __post_init__(self) -> None:
        if not self.endpoint:
            self.endpoint = os.environ.get("AZURE_AGENTS_ENDPOINT", "").strip()
        if not self.agent_id:
            self.agent_id = os.environ.get(
                "AZURE_AGENTS_VERIFIER_AGENT_ID", "",
            ).strip()
        if not self.endpoint:
            logger.warning(
                "AzureAgentBackend initialised without an endpoint; "
                "every verify() call will fail until AZURE_AGENTS_ENDPOINT "
                "is set or endpoint= is supplied.",
            )
        if not self.agent_id:
            logger.warning(
                "AzureAgentBackend initialised without an agent id; "
                "every verify() call will fail until "
                "AZURE_AGENTS_VERIFIER_AGENT_ID is set or agent_id= is "
                "supplied.",
            )

    # --- internal --------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazy-build the AgentsClient. Raises on missing SDK / config."""
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                from azure.ai.agents import AgentsClient
                from azure.identity import AzureCliCredential
            except ImportError as exc:  # pragma: no cover  (env issue)
                raise RuntimeError(
                    "azure-ai-agents and azure-identity are required for "
                    "the Foundry Agents verifier; pip install -r "
                    "requirements.txt"
                ) from exc
            if not self.endpoint:
                raise RuntimeError(
                    "AzureAgentBackend.endpoint is empty; set "
                    "AZURE_AGENTS_ENDPOINT or pass endpoint=...",
                )
            self._client = AgentsClient(
                endpoint=self.endpoint,
                credential=AzureCliCredential(),
            )
            return self._client

    @staticmethod
    def _extract_text_and_citations(
        messages: Any,
    ) -> tuple[str, List[Citation]]:
        """Walk ``messages`` (assumed in ASCENDING order) and pull the
        last assistant reply + its ``url_citation`` annotations.

        Annotations may be SDK model objects or plain dicts depending
        on the version; we tolerate both via ``getattr`` + ``dict.get``.
        """
        text_parts: List[str] = []
        citations: List[Citation] = []
        seen_urls: set[str] = set()
        for msg in messages:
            role_obj = getattr(msg, "role", None)
            if role_obj is None and isinstance(msg, Mapping):
                role_obj = msg.get("role")
            # The SDK's ``MessageRole`` is a ``str, Enum`` whose
            # ``.value`` is ``"assistant"``. In Python 3.12+
            # ``str(enum_member)`` returns ``"MessageRole.AGENT"``
            # (not the value), so we read ``.value`` first and only
            # fall back to ``str()`` for plain strings / dicts.
            role = getattr(role_obj, "value", role_obj)
            if str(role).strip().lower() != "assistant":
                continue
            # Prefer the SDK's pre-filtered ``text_messages``, but
            # fall back to ``content`` (filtering by ``type=="text"``)
            # so we work across SDK versions and shapes.
            blocks = getattr(msg, "text_messages", None)
            if blocks is None and isinstance(msg, Mapping):
                blocks = msg.get("text_messages")
            if not blocks:
                content = getattr(msg, "content", None)
                if content is None and isinstance(msg, Mapping):
                    content = msg.get("content")
                blocks = [
                    c for c in (content or [])
                    if (getattr(c, "type", None)
                        or (c.get("type") if isinstance(c, Mapping) else None))
                    == "text"
                ]
            for block in blocks or []:
                t = getattr(block, "text", None)
                if t is None and isinstance(block, Mapping):
                    t = block.get("text")
                value = getattr(t, "value", None) if t is not None else None
                if value is None and isinstance(t, Mapping):
                    value = t.get("value")
                if isinstance(value, str) and value:
                    text_parts.append(value)
                anns = getattr(t, "annotations", None) if t is not None else None
                if anns is None and isinstance(t, Mapping):
                    anns = t.get("annotations")
                for ann in anns or []:
                    ann_type = getattr(ann, "type", None) or (
                        ann.get("type") if isinstance(ann, Mapping) else None
                    )
                    if ann_type != "url_citation":
                        continue
                    cite = getattr(ann, "url_citation", None)
                    if cite is None and isinstance(ann, Mapping):
                        cite = ann.get("url_citation") or {}
                    url = (
                        getattr(cite, "url", None)
                        or (cite.get("url") if isinstance(cite, Mapping) else None)
                        or ""
                    )
                    title = (
                        getattr(cite, "title", None)
                        or (cite.get("title") if isinstance(cite, Mapping) else None)
                        or ""
                    )
                    url = str(url).strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    citations.append(Citation(url=url, title=str(title).strip()))
        # Use the LAST assistant reply (multiple are possible if the
        # agent decided to chain reasoning + final answer).
        return (text_parts[-1] if text_parts else ""), citations

    @staticmethod
    def _extract_search_steps(run_steps: Any) -> tuple[List[str], int]:
        """Pull the Bing search request URLs out of run-step tool calls.

        Returns ``(display_urls, n_search_calls)``. The Bing API host
        is rewritten to ``www.bing.com`` per the Use & Display
        requirements before display.
        """
        display_urls: List[str] = []
        n_calls = 0
        for step in run_steps:
            details = getattr(step, "step_details", None)
            tool_calls = getattr(details, "tool_calls", None) or []
            for call in tool_calls:
                # The Web Search tool call name varies across SDK
                # versions / preview tracks. We accept any of the
                # known shapes and look for a request URL field.
                wrapper = (
                    getattr(call, "web_search", None)
                    or getattr(call, "bing_grounding", None)
                    or getattr(call, "bing_custom_search", None)
                )
                if wrapper is None:
                    continue
                n_calls += 1
                req_url = (
                    getattr(wrapper, "request_url", None)
                    or getattr(wrapper, "requesturl", None)
                )
                if not req_url:
                    continue
                display_url = str(req_url).replace(
                    "https://api.bing.microsoft.com/v7.0/search",
                    "https://www.bing.com/search",
                )
                if display_url not in display_urls:
                    display_urls.append(display_url)
        return display_urls, n_calls

    # --- public ----------------------------------------------------

    def verify(
        self,
        prompt: str,
        *,
        cache_key: Optional[str] = None,
    ) -> AgentVerdict:
        prompt = (prompt or "").strip()
        if not prompt:
            return AgentVerdict(error="empty prompt")

        # Cache lookup. The verdict cache is keyed on cluster_id +
        # a content hash supplied by the caller so re-runs over the
        # same cluster are free.
        if self.cache is not None and cache_key:
            cached = self.cache.get(self.name, cache_key)
            if isinstance(cached, Mapping):
                citations = [
                    Citation(url=str(c.get("url") or ""),
                             title=str(c.get("title") or ""))
                    for c in (cached.get("citations") or [])
                    if isinstance(c, Mapping) and c.get("url")
                ]
                return AgentVerdict(
                    text=str(cached.get("text") or ""),
                    citations=citations,
                    bing_query_urls=[
                        str(u) for u in (cached.get("bing_query_urls") or [])
                    ],
                    n_search_calls=int(cached.get("n_search_calls") or 0),
                    cached=True,
                )

        if not self.agent_id:
            return AgentVerdict(error="missing agent_id")

        try:
            client = self._ensure_client()
        except RuntimeError as exc:
            return AgentVerdict(error=str(exc))

        try:
            thread = client.threads.create()
            client.messages.create(
                thread_id=thread.id, role="user", content=prompt,
            )
            # Override the agent's baked-in instructions with our
            # task-specific system prompt; this lets us evolve
            # prompts/verify_entity.yaml without redeploying the agent.
            kwargs: Dict[str, Any] = {
                "thread_id": thread.id,
                "agent_id": self.agent_id,
            }
            if self.system_prompt:
                kwargs["instructions"] = self.system_prompt
            t0 = time.monotonic()
            run = client.runs.create_and_process(**kwargs)
            elapsed = time.monotonic() - t0
            if self.warn_after_seconds and elapsed > self.warn_after_seconds:
                logger.warning(
                    "agent run %s slow (%.1fs > %.1fs warn threshold)",
                    getattr(run, "id", "?"), elapsed, self.warn_after_seconds,
                )

            status = str(getattr(run, "status", "") or "").lower()
            if status == "failed":
                err = getattr(run, "last_error", None)
                msg = (
                    getattr(err, "message", None)
                    or (err.get("message") if isinstance(err, Mapping) else None)
                    or "agent run failed"
                )
                return AgentVerdict(error=str(msg))

            # Read messages + run steps (best-effort; messages are
            # required, run steps are decorative).
            try:
                from azure.ai.agents.models import ListSortOrder
                messages = client.messages.list(
                    thread_id=thread.id, order=ListSortOrder.ASCENDING,
                )
            except ImportError:
                messages = client.messages.list(thread_id=thread.id)

            text, citations = self._extract_text_and_citations(messages)

            bing_urls: List[str] = []
            n_calls = 0
            try:
                run_steps = client.run_steps.list(
                    thread_id=thread.id, run_id=run.id,
                )
                bing_urls, n_calls = self._extract_search_steps(run_steps)
            except Exception as exc:  # noqa: BLE001  (decorative)
                logger.debug("run_steps fetch failed: %s", exc)

            verdict = AgentVerdict(
                text=text,
                citations=citations,
                bing_query_urls=bing_urls,
                n_search_calls=n_calls,
            )

            if self.cache is not None and cache_key and not verdict.error:
                self.cache.put(
                    self.name, cache_key,
                    {
                        "text": verdict.text,
                        "citations": [
                            {"url": c.url, "title": c.title}
                            for c in verdict.citations
                        ],
                        "bing_query_urls": list(verdict.bing_query_urls),
                        "n_search_calls": verdict.n_search_calls,
                    },
                )
            return verdict

        except Exception as exc:  # noqa: BLE001
            logger.warning("agent verify failed: %s", exc)
            return AgentVerdict(error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Stub for tests
# ---------------------------------------------------------------------------


@dataclass
class StubAgentBackend:
    """Test double that replays a fixed mapping of prompt -> verdict.

    Lookup key is the entire ``prompt`` string; tests provide either a
    full prompt OR a callable that derives the verdict from the prompt.
    The call sequence is recorded on ``calls`` for assertions.
    """

    fixtures: Dict[str, AgentVerdict] = field(default_factory=dict)
    default: Optional[AgentVerdict] = None
    name: str = "stub_agent"
    calls: List[Dict[str, Any]] = field(default_factory=list)
    system_prompts: List[str] = field(default_factory=list)
    system_prompt: str = ""

    def verify(
        self,
        prompt: str,
        *,
        cache_key: Optional[str] = None,
    ) -> AgentVerdict:
        self.calls.append({"prompt": prompt, "cache_key": cache_key})
        self.system_prompts.append(self.system_prompt)
        verdict = self.fixtures.get(prompt)
        if verdict is None and self.default is not None:
            verdict = self.default
        if verdict is None:
            return AgentVerdict(error="no fixture for prompt")
        # Return a shallow copy so callers can't mutate the fixture.
        return AgentVerdict(
            text=verdict.text,
            citations=list(verdict.citations),
            bing_query_urls=list(verdict.bing_query_urls),
            n_search_calls=verdict.n_search_calls,
            cached=verdict.cached,
            error=verdict.error,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_agent_backend(
    *,
    endpoint: str = "",
    agent_id: str = "",
    system_prompt: str = "",
    cache: Optional[WebCache] = None,
    stub: bool = False,
) -> AgentBackend:
    """Build an :class:`AgentBackend`.

    Pass ``stub=True`` from tests to get an offline backend. In
    production the caller supplies (or env-supplies) ``endpoint`` and
    ``agent_id``; the agent itself must be pre-configured with the Web
    Search tool in the Foundry portal.
    """
    if stub:
        return StubAgentBackend(system_prompt=system_prompt)
    return AzureAgentBackend(
        endpoint=endpoint,
        agent_id=agent_id,
        system_prompt=system_prompt,
        cache=cache,
    )


__all__ = [
    "AgentBackend",
    "AgentVerdict",
    "AzureAgentBackend",
    "Citation",
    "StubAgentBackend",
    "build_agent_backend",
]
