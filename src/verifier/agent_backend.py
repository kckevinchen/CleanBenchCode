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

Multi-endpoint scaling: when several Foundry projects (each with its
own web-search-equipped agent) are available, wrap them in
:class:`RoundRobinAgentBackend` to distribute calls across endpoints
and raise effective throughput. See :func:`build_agent_backend`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from .web_cache import WebCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default Foundry Agents pool
# ---------------------------------------------------------------------------

# Pool of (project endpoint, agent id) pairs the verifier dispatches
# across by default when nothing else is configured (no CLI args, no
# env vars). Mirrors :data:`src.core.llm_client.DEFAULT_ENDPOINTS`:
# multi-endpoint round-robin is the out-of-the-box behaviour, not an
# opt-in. Add or remove pairs here as the agent fleet grows. The
# pairing is positional (the i-th endpoint is dispatched against the
# i-th agent id).
DEFAULT_AGENT_ENDPOINTS: List[str] = [
    "https://sc-rf-ma30hqbc-eastus2.services.ai.azure.com/api/projects/sc-rf-ma30hqbc-eastus2-project",
    "https://sc-rf-ml1gf0z3-southcentralus.services.ai.azure.com/api/projects/sc-rf-ml1gf0z3-southcen-project",
]
DEFAULT_AGENT_IDS: List[str] = [
    "asst_Hobm9YVg9eJm7848CqH4GQ4y",
    "asst_uInM055RT8OpSFC9Zgb0aPpi",
]


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
        # Final code-default fallback: when nothing was passed and no
        # singular env var was set, pick the FIRST entry from the
        # default pool so direct ``AzureAgentBackend()`` instantiation
        # (without going through ``build_agent_backend``) still works
        # out-of-the-box. ``build_agent_backend`` is the recommended
        # path - it returns a round-robin dispatcher across all
        # default pairs - but we keep this fallback so legacy callers
        # don't regress.
        if not self.endpoint and DEFAULT_AGENT_ENDPOINTS:
            self.endpoint = DEFAULT_AGENT_ENDPOINTS[0]
        if not self.agent_id and DEFAULT_AGENT_IDS:
            self.agent_id = DEFAULT_AGENT_IDS[0]
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

            run_id = str(getattr(run, "id", "") or "")
            # ``run.status`` is an ``Enum`` in newer azure-ai-agents
            # SDKs (RunStatus.COMPLETED, etc.) and a plain ``str`` in
            # older ones. ``str(enum)`` yields ``"RunStatus.COMPLETED"``
            # which historically broke the ``== "completed"`` check
            # below and tagged every successful run as errored. Pull
            # ``.value`` first when present so we always compare on the
            # SDK's documented short tokens ("queued", "in_progress",
            # "completed", ...).
            raw_status = getattr(run, "status", "")
            status = str(getattr(raw_status, "value", raw_status) or "").lower()
            if status == "failed":
                err = getattr(run, "last_error", None)
                msg = (
                    getattr(err, "message", None)
                    or (err.get("message") if isinstance(err, Mapping) else None)
                    or "agent run failed"
                )
                return AgentVerdict(error=f"failed: {msg} (run={run_id})")
            if status in ("cancelled", "expired"):
                return AgentVerdict(error=f"agent run {status} (run={run_id})")
            # ``incomplete`` is a terminal Azure agent status meaning
            # the run finished but the output was truncated (typically
            # ``max_completion_tokens`` / ``max_prompt_tokens`` /
            # ``content_filter``). Partial messages may still be on the
            # thread, and ``run.incomplete_details.reason`` explains
            # why. We treat it like ``completed`` and let the existing
            # empty-text -> error path below surface the reason; that
            # path already prefers the incomplete_reason when present.
            if status and status not in ("completed", "incomplete"):
                # queued / in_progress / requires_action shouldn't reach
                # here because create_and_process is synchronous, but
                # surface them clearly if they do.
                return AgentVerdict(
                    error=f"unexpected agent run status: {status} (run={run_id})",
                )

            # ``incomplete_details`` lives on a *completed* run when the
            # output was suppressed (content filter, max tokens, etc.).
            # Without surfacing it, the caller just sees an empty
            # response with no idea why - the dominant failure mode for
            # person-name / private-individual cells is content_filter
            # here.
            incomplete_reason: Optional[str] = None
            incomp = getattr(run, "incomplete_details", None)
            if incomp is not None:
                reason = (
                    getattr(incomp, "reason", None)
                    or (incomp.get("reason") if isinstance(incomp, Mapping) else None)
                )
                if reason:
                    incomplete_reason = str(reason)

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

            # Promote incomplete_details / empty-text into a real error
            # so it (a) shows up in the UI's "error" field with a
            # diagnostic reason, and (b) prevents us from caching the
            # failure as if it were a successful empty verdict (see
            # below).
            if not verdict.text.strip() and not verdict.citations:
                if incomplete_reason:
                    verdict.error = (
                        f"agent returned empty response "
                        f"(reason={incomplete_reason}, run={run_id})"
                    )
                else:
                    verdict.error = (
                        f"agent returned empty response (run={run_id})"
                    )

            # Cache only payloads that carry useful signal. Caching
            # empty / errored payloads poisons every subsequent retry
            # because the verifier would just replay the empty result
            # forever. Re-runs of an empty cluster should re-hit the
            # agent (it's non-deterministic; person-name content-filter
            # blocks frequently flip on retry).
            should_cache = (
                self.cache is not None
                and cache_key
                and not verdict.error
                and (verdict.text.strip() or verdict.citations)
            )
            if should_cache:
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
    endpoints: Optional[Sequence[str]] = None,
    agent_ids: Optional[Sequence[str]] = None,
) -> AgentBackend:
    """Build an :class:`AgentBackend`.

    Pass ``stub=True`` from tests to get an offline backend. In
    production the caller supplies (or env-supplies) ``endpoint`` and
    ``agent_id``; the agent itself must be pre-configured with the Web
    Search tool in the Foundry portal.

    Multi-endpoint mode
    -------------------
    To distribute load across several Foundry projects in round-robin
    fashion, supply either:

    * Lists/tuples via ``endpoints=`` and ``agent_ids=`` (same length;
      paired positionally), OR
    * Comma-separated values in the singular ``endpoint`` / ``agent_id``
      arguments (e.g. ``endpoint="https://a/...,https://b/..."``), OR
    * The plural env vars ``AZURE_AGENTS_ENDPOINTS`` and
      ``AZURE_AGENTS_VERIFIER_AGENT_IDS`` (comma-separated, paired
      positionally).

    With more than one (endpoint, agent_id) pair the factory returns
    a :class:`RoundRobinAgentBackend`. With exactly one pair it
    returns a single :class:`AzureAgentBackend` (no overhead vs. the
    legacy code path).
    """
    if stub:
        return StubAgentBackend(system_prompt=system_prompt)

    pairs = _resolve_endpoint_pairs(
        endpoint=endpoint,
        agent_id=agent_id,
        endpoints=endpoints,
        agent_ids=agent_ids,
    )
    # No multi-endpoint config → preserve the legacy single-backend
    # path so behaviour is unchanged for callers that haven't opted
    # in. AzureAgentBackend.__post_init__ falls back to the singular
    # env vars on its own.
    if len(pairs) <= 1:
        ep, aid = (pairs[0] if pairs else ("", ""))
        return AzureAgentBackend(
            endpoint=ep or endpoint,
            agent_id=aid or agent_id,
            system_prompt=system_prompt,
            cache=cache,
        )

    backends: List[AgentBackend] = [
        AzureAgentBackend(
            endpoint=ep,
            agent_id=aid,
            system_prompt=system_prompt,
            cache=cache,
        )
        for ep, aid in pairs
    ]
    logger.info(
        "agent backend: round-robin over %d endpoints (%s)",
        len(backends),
        ", ".join(_short_endpoint(ep) for ep, _ in pairs),
    )
    return RoundRobinAgentBackend(
        backends=backends, system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Multi-endpoint configuration helpers
# ---------------------------------------------------------------------------


def _split_csv(value: str) -> List[str]:
    """Split a comma-separated string into a list of non-empty trimmed
    items. ``""`` -> ``[]`` so we stay None-tolerant."""
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _resolve_endpoint_pairs(
    *,
    endpoint: str = "",
    agent_id: str = "",
    endpoints: Optional[Sequence[str]] = None,
    agent_ids: Optional[Sequence[str]] = None,
) -> List[tuple[str, str]]:
    """Resolve ``(endpoint, agent_id)`` pairs from CLI args + env vars.

    Resolution order (first non-empty wins for each axis):

    1. Explicit ``endpoints=`` / ``agent_ids=`` lists.
    2. Comma-separated values in the singular ``endpoint`` /
       ``agent_id`` strings (allows ``--agent-endpoint a,b`` from the
       CLI without needing a second flag).
    3. Plural env vars: ``AZURE_AGENTS_ENDPOINTS`` /
       ``AZURE_AGENTS_VERIFIER_AGENT_IDS``.
    4. Singular env vars: ``AZURE_AGENTS_ENDPOINT`` /
       ``AZURE_AGENTS_VERIFIER_AGENT_ID`` (legacy).
    5. Code-default pool: :data:`DEFAULT_AGENT_ENDPOINTS` /
       :data:`DEFAULT_AGENT_IDS`. Multi-endpoint round-robin is thus
       the out-of-the-box behaviour even when no .env / CLI override
       is supplied.

    Raises ``ValueError`` when the two lists are non-empty but
    different lengths - the pairing is positional and we cannot
    silently truncate.
    """
    eps: List[str] = []
    aids: List[str] = []
    used_default = False

    if endpoints:
        eps = [e.strip() for e in endpoints if e and str(e).strip()]
    elif endpoint:
        eps = _split_csv(endpoint)
    else:
        eps = _split_csv(os.environ.get("AZURE_AGENTS_ENDPOINTS", ""))
        if not eps:
            singular = os.environ.get("AZURE_AGENTS_ENDPOINT", "").strip()
            if singular:
                eps = [singular]

    if agent_ids:
        aids = [a.strip() for a in agent_ids if a and str(a).strip()]
    elif agent_id:
        aids = _split_csv(agent_id)
    else:
        aids = _split_csv(
            os.environ.get("AZURE_AGENTS_VERIFIER_AGENT_IDS", ""),
        )
        if not aids:
            singular = os.environ.get(
                "AZURE_AGENTS_VERIFIER_AGENT_ID", "",
            ).strip()
            if singular:
                aids = [singular]

    # Code-default fallback. Only kicks in when BOTH axes are still
    # empty - if the operator configured one axis explicitly, we
    # surface the length mismatch below rather than silently mixing
    # their endpoints with our default agent ids (or vice versa).
    if not eps and not aids:
        if not DEFAULT_AGENT_ENDPOINTS or not DEFAULT_AGENT_IDS:
            return []
        eps = list(DEFAULT_AGENT_ENDPOINTS)
        aids = list(DEFAULT_AGENT_IDS)
        used_default = True

    if len(eps) != len(aids):
        raise ValueError(
            "endpoint / agent_id count mismatch: "
            f"{len(eps)} endpoint(s) but {len(aids)} agent_id(s); "
            "they must pair positionally",
        )
    if used_default:
        logger.info(
            "agent backend: using built-in default pool (%d endpoint(s)). "
            "Set AZURE_AGENTS_ENDPOINTS / AZURE_AGENTS_VERIFIER_AGENT_IDS "
            "or pass --agent-endpoint / --agent-id to override.",
            len(eps),
        )
    return list(zip(eps, aids))


def _short_endpoint(url: str) -> str:
    """Display-only abbreviation for log lines."""
    if not url:
        return "<empty>"
    # Strip the long ``/api/projects/...`` suffix; keep host.
    head = url.split("/api/", 1)[0]
    return head.replace("https://", "").replace("http://", "")[:60]


# ---------------------------------------------------------------------------
# Round-robin dispatcher
# ---------------------------------------------------------------------------


@dataclass
class RoundRobinAgentBackend:
    """Round-robin dispatcher across multiple :class:`AgentBackend` instances.

    Use this when you have several Foundry projects (each with its own
    web-search-equipped agent) and want to distribute load across them
    to raise effective throughput. Each call to :meth:`verify` advances
    a shared counter and dispatches to the next backend; backends own
    their own ``AgentsClient`` + verdict cache (typically all wired to
    the same on-disk :class:`WebCache` so cache hits are shared).

    Thread-safe: the counter is guarded by a ``Lock`` so concurrent
    Verifier workers (``ThreadPoolExecutor``) get distinct backends.

    System-prompt plumbing: :class:`Verifier` writes ``system_prompt``
    onto its ``agent`` attribute after construction. We forward that
    write down to every child so they all use the same prompt for
    their per-run ``instructions`` override.
    """

    backends: List[AgentBackend] = field(default_factory=list)
    system_prompt: str = ""
    name: str = "azure_agent_pool"
    _idx: int = field(init=False, default=0, repr=False)
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False,
    )

    def __post_init__(self) -> None:
        if not self.backends:
            raise ValueError(
                "RoundRobinAgentBackend requires at least one backend",
            )
        # Propagate any system_prompt set at construction time down
        # to children so the very first call already uses the right
        # instructions override.
        if self.system_prompt:
            self._propagate_system_prompt(self.system_prompt)

    def __setattr__(self, name: str, value: Any) -> None:
        # Forward ``system_prompt`` writes to every child backend so
        # the Verifier's post-init hook (``setattr(agent,
        # "system_prompt", ...)``) reaches all of them.
        super().__setattr__(name, value)
        if name == "system_prompt" and getattr(self, "backends", None):
            self._propagate_system_prompt(value)

    def _propagate_system_prompt(self, value: str) -> None:
        for b in self.backends:
            try:
                setattr(b, "system_prompt", value)
            except (AttributeError, TypeError):  # backend doesn't support it
                continue

    def _next_backend(self) -> AgentBackend:
        with self._lock:
            backend = self.backends[self._idx % len(self.backends)]
            i = self._idx
            self._idx += 1
        # Debug-level so production logs stay quiet, but operators
        # can flip on `-v` to confirm dispatch is balanced across
        # endpoints during smoke tests.
        ep = getattr(backend, "endpoint", "")
        if ep:
            logger.debug(
                "round-robin dispatch #%d -> %s",
                i, _short_endpoint(ep),
            )
        return backend

    def verify(
        self,
        prompt: str,
        *,
        cache_key: Optional[str] = None,
    ) -> AgentVerdict:
        backend = self._next_backend()
        return backend.verify(prompt, cache_key=cache_key)


__all__ = [
    "AgentBackend",
    "AgentVerdict",
    "AzureAgentBackend",
    "Citation",
    "DEFAULT_AGENT_ENDPOINTS",
    "DEFAULT_AGENT_IDS",
    "RoundRobinAgentBackend",
    "StubAgentBackend",
    "build_agent_backend",
]
