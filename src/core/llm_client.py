"""Thin, reusable wrapper around Azure OpenAI endpoints for GPT-5 family models.

This module is a refactor of the standalone script
``combined-gpt-5-with-reasoning-summary.py``. Behaviour is preserved:

* Round-robin load balancing across a pool of endpoints
* Automatic fallback between the Responses API and the Chat Completions API
* Bounded retries with exponential backoff + jitter
* Extraction of reasoning summaries when available

Endpoints in the pool may not all carry every model deployment - and the
deployment surface drifts over time. To handle that without operator
intervention, the client tracks ``(endpoint, model)`` combinations that
return HTTP 404 and skips them on subsequent calls. Other errors still
go through the normal retry / backoff path.

The client is intentionally framework-free so it can be imported from any
experiment script.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Endpoint configuration
# ---------------------------------------------------------------------------

# Default pool mirrors the endpoints used in combined-gpt-5-with-reasoning-summary.py.
# Override via the ``AZURE_OPENAI_ENDPOINTS`` env var (comma separated URLs) when
# running in a different environment. Not every endpoint carries every model;
# the runtime 404-skip logic in :class:`LLMClient` handles that without
# operator action.
DEFAULT_ENDPOINTS: List[str] = [
    "https://dmxaz-test-aoai.openai.azure.com/",
    "https://dmxaz-test-aoai-eu2.openai.azure.com/",
    "https://yeyehe-aoai.openai.azure.com/",
    "https://sc-rf-ma30hqbc-eastus2.cognitiveservices.azure.com/",
    "https://sc-rf-m8udnqmj-swedencentral.cognitiveservices.azure.com/",
    "https://sc-rf-ml1gih82-centralus.cognitiveservices.azure.com/",
    "https://sc-rf-mcfew454-francecentral.cognitiveservices.azure.com/",
    "https://sc-rf-mbvhbia4-northcentralus.cognitiveservices.azure.com/",
    "https://sc-rf-mcfeobyx-northcentralus.cognitiveservices.azure.com/",
    "https://sc-rf-mcnngi4g-norwayeast.cognitiveservices.azure.com/",
    "https://sc-rf-ml1gf0z3-southcentralus.openai.azure.com/",
]

DEFAULT_API_VERSION = "2025-03-01-preview"

# Models that must go through the Responses API rather than Chat Completions.
RESPONSES_API_MODELS = {
    "gpt-5-pro",
    "gpt-5",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.5",
    "o1",
    "o1-pro",
    "o3",
    "o3-mini",
}

# Canonical "GPT-5 family" used as the default detection target when the
# caller asks for a family sweep instead of a single model. Order is the
# order runs are launched in.
GPT5_FAMILY: Tuple[str, ...] = ("gpt-5.2", "gpt-5.4", "gpt-5.5")


# Reasoning-effort values accepted by the Responses API for the
# GPT-5 family. ``"minimal"`` is documented elsewhere but rejected by
# at least gpt-5.2 / 5.4 / 5.5 with HTTP 400 (param='reasoning.effort',
# code='unsupported_value'). Older code paths in this repo defaulted
# to ``"minimal"``, which silently burnt the retry budget on every
# call; we coerce to ``"low"`` instead and warn once. Updates to this
# set should match what the deployed model accepts at the time.
RESPONSES_API_VALID_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
_REASONING_EFFORT_REWRITES = {"minimal": "low"}

# Per-process set of (model, original_effort) tuples we've already
# warned about, so a long-running job doesn't spam the log on every
# call.
_warned_effort_rewrites: set[Tuple[str, str]] = set()
_warned_effort_lock = Lock()


def _normalize_reasoning_effort(model: str, effort: str) -> str:
    """Coerce a Responses-API reasoning_effort to a value the model accepts.

    Returns the (possibly rewritten) effort value. Logs a warning the
    FIRST time a given (model, effort) pair is rewritten; subsequent
    calls with the same pair are silent. Unknown values are passed
    through unchanged - the API itself will surface a clearer 400
    than we could fabricate.
    """
    rewrite = _REASONING_EFFORT_REWRITES.get(effort)
    if rewrite is None:
        return effort
    key = (model, effort)
    with _warned_effort_lock:
        first_time = key not in _warned_effort_rewrites
        if first_time:
            _warned_effort_rewrites.add(key)
    if first_time:
        logger.warning(
            "Rewriting reasoning_effort=%r -> %r for model %r: %r is not "
            "accepted by the GPT-5 Responses API (valid: %s). Update "
            "the caller to silence this warning.",
            effort, rewrite, model, effort,
            sorted(RESPONSES_API_VALID_EFFORTS),
        )
    return rewrite


def _is_404(exc: Exception) -> bool:
    """Best-effort 404 detection across SDK versions.

    The Azure OpenAI SDK surfaces deployment-not-found as
    ``openai.NotFoundError`` with ``status_code == 404``. Older or
    wrapped versions may attach the status to ``response.status_code``
    instead, or only to the message text. We check all three so the
    skip logic kicks in regardless of SDK release.
    """
    sc = getattr(exc, "status_code", None)
    if sc is None:
        resp = getattr(exc, "response", None)
        sc = getattr(resp, "status_code", None) if resp is not None else None
    if sc == 404:
        return True
    if type(exc).__name__ == "NotFoundError":
        return True
    msg = str(exc).lower()
    if "deploymentnotfound" in msg.replace(" ", ""):
        return True
    return False


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Normalised response across Chat Completions and Responses APIs."""

    text: str
    reasoning_summary: str = ""
    raw: Any = None
    endpoint: str = ""
    model: str = ""
    attempts: int = 1
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


# ---------------------------------------------------------------------------
# Client pool
# ---------------------------------------------------------------------------


@dataclass
class LLMClient:
    """Round-robin pool of Azure OpenAI clients.

    Example
    -------
    >>> client = LLMClient()
    >>> resp = client.query("gpt-5.2", "Say hi", reasoning_effort="low")
    >>> print(resp.text)
    """

    endpoints: List[str] = field(default_factory=lambda: list(DEFAULT_ENDPOINTS))
    api_version: str = DEFAULT_API_VERSION
    max_retries: int = 5
    backoff_base_seconds: float = 10.0
    # Cap on per-attempt sleep regardless of attempt index. Prevents
    # the exponential schedule from running away on long retry chains
    # (e.g. attempt=5 with base=10 would otherwise sleep 160s).
    backoff_cap_seconds: float = 60.0
    # Per-call wall-clock timeout passed through to the OpenAI SDK.
    # Without this, a wedged endpoint can hang a worker thread
    # indefinitely - the retry loop only fires on RAISED exceptions,
    # not on a never-returning request. 180s comfortably covers
    # gpt-5 / o-series reasoning calls; raise for very long contexts
    # via the constructor.
    request_timeout_seconds: float = 180.0
    token_scope: str = "https://cognitiveservices.azure.com/.default"

    def __post_init__(self) -> None:
        env_endpoints = os.environ.get("AZURE_OPENAI_ENDPOINTS")
        if env_endpoints:
            self.endpoints = [e.strip() for e in env_endpoints.split(",") if e.strip()]
        if not self.endpoints:
            raise ValueError("LLMClient requires at least one endpoint")

        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(credential, self.token_scope)

        self._clients: List[AzureOpenAI] = [
            AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                timeout=self.request_timeout_seconds,
            )
            for endpoint in self.endpoints
        ]
        self._index = 0
        self._index_lock = Lock()
        # Separate counter for model rotation when ``query`` is given a
        # comma-list spec. Independent of the endpoint cursor so the
        # two rotations don't synchronise (which would always pair the
        # same (endpoint, model) tuples and undo the load-balancing
        # we want when the family is bigger than the endpoint count).
        self._model_index = 0
        self._model_index_lock = Lock()
        # (endpoint, model) tuples that returned 404. We never try
        # them again for this LLMClient lifetime.
        self._dead_combos: set[Tuple[str, str]] = set()
        self._dead_lock = Lock()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_client(self) -> "tuple[AzureOpenAI, str]":
        with self._index_lock:
            idx = self._index % len(self._clients)
            self._index += 1
        return self._clients[idx], self.endpoints[idx]

    def _next_alive_client_for_model(
        self, model: str,
    ) -> Optional[Tuple[AzureOpenAI, str]]:
        """Round-robin pick that skips known-404 (endpoint, model) combos.

        Returns ``None`` if every endpoint has been marked unsupported
        for ``model``. Each call advances the cursor by 1+ positions
        (one per dead combo skipped + one for the alive pick), so two
        consecutive callers naturally land on different endpoints under
        load.
        """
        n = len(self._clients)
        for _ in range(n):
            client, endpoint = self._next_client()
            with self._dead_lock:
                dead = (endpoint, model) in self._dead_combos
            if not dead:
                return client, endpoint
        return None

    def _mark_dead(self, endpoint: str, model: str) -> None:
        with self._dead_lock:
            self._dead_combos.add((endpoint, model))

    def _resolve_model_pool(self, spec: str) -> List[str]:
        """Split a model spec into a deduped list, preserving order.

        Accepts a single model (``"gpt-5.2"``) or a comma-separated
        sweep (``"gpt-5.2,gpt-5.4"``). Whitespace around commas is
        ignored so callers can pass the same form they'd write on
        the CLI. Empty/whitespace-only input raises - silently
        defaulting would mask config bugs in unrelated callers.
        """
        out: List[str] = []
        seen: set = set()
        for tok in spec.split(","):
            m = tok.strip()
            if m and m not in seen:
                out.append(m)
                seen.add(m)
        if not out:
            raise ValueError(
                f"LLMClient.query: model spec is empty (got {spec!r})"
            )
        return out

    def _next_model_index(self) -> int:
        """Atomic, monotonic counter for round-robin model selection.

        Used to pick a starting model from a rotation pool so two
        concurrent calls don't both hit the same family member and
        thereby waste the load-balancing benefit of the comma-list.
        """
        with self._model_index_lock:
            idx = self._model_index
            self._model_index += 1
        return idx

    def is_dead(self, endpoint: str, model: str) -> bool:
        """Public read-only view; useful for tests + diagnostics."""
        with self._dead_lock:
            return (endpoint, model) in self._dead_combos

    @staticmethod
    def _safe_model_dump(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return obj

    @classmethod
    def _extract_reasoning_summary(cls, result: Any) -> str:
        """Extract reasoning summary text from Responses API output when available."""
        try:
            payload = cls._safe_model_dump(result)
            if not isinstance(payload, dict):
                return ""

            summary_lines: List[str] = []

            reasoning_payload = payload.get("reasoning")
            if isinstance(reasoning_payload, dict):
                top_summary = reasoning_payload.get("summary")
                if isinstance(top_summary, str) and top_summary not in (
                    "auto",
                    "concise",
                    "detailed",
                ):
                    summary_lines.append(top_summary)
                elif isinstance(top_summary, list):
                    for s in top_summary:
                        if isinstance(s, dict) and s.get("text"):
                            summary_lines.append(s["text"])

            for item in payload.get("output", []) or []:
                if not isinstance(item, dict) or item.get("type") != "reasoning":
                    continue
                for summary_item in item.get("summary") or []:
                    if isinstance(summary_item, dict) and summary_item.get("text"):
                        summary_lines.append(summary_item["text"])

            return "\n".join(summary_lines)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to parse reasoning summary: %s", exc)
            return ""

    @staticmethod
    def _extract_output_text(result: Any) -> str:
        if hasattr(result, "output_text") and result.output_text:
            return result.output_text
        if hasattr(result, "output") and result.output:
            first = result.output[0]
            if hasattr(first, "content") and first.content:
                return first.content[0].text
            return str(result.output)
        return str(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _do_call(
        self,
        client: AzureOpenAI,
        endpoint: str,
        model: str,
        prompt: str,
        *,
        use_responses_api: bool,
        reasoning_effort: str,
        reasoning_summary: str,
        system_prompt: str,
    ) -> LLMResponse:
        """Single API call (no retries, no dead-combo handling).

        Used by both :meth:`query` and :meth:`probe_availability`.
        Raises on failure - the caller decides whether to retry, mark
        the combo dead, or report the error verbatim.
        """
        if use_responses_api:
            # Coerce known-bad effort values (e.g. ``"minimal"``,
            # which gpt-5.2+ rejects with HTTP 400). Warns once per
            # (model, original-value) pair so this never silently
            # masks a config bug.
            effort = _normalize_reasoning_effort(model, reasoning_effort)
            responses_kwargs: Dict[str, Any] = {
                "model": model,
                "input": prompt,
                "reasoning": {
                    "effort": effort,
                    "summary": reasoning_summary,
                },
            }
            # The Responses API uses ``instructions`` as the equivalent
            # of a system message. Without this, callers passing a
            # system_prompt (e.g. the detector's taxonomy rules) would
            # have it silently dropped for gpt-5 / o-series models.
            if system_prompt:
                responses_kwargs["instructions"] = system_prompt
            result = client.responses.create(**responses_kwargs)
            return LLMResponse(
                text=self._extract_output_text(result),
                reasoning_summary=self._extract_reasoning_summary(result),
                raw=result,
                endpoint=endpoint,
                model=model,
            )

        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return LLMResponse(
            text=result.choices[0].message.content or "",
            raw=result,
            endpoint=endpoint,
            model=model,
        )

    def query(
        self,
        model: str,
        prompt: str,
        *,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        system_prompt: str = "",
    ) -> LLMResponse:
        """Query an LLM with built-in retries and endpoint rotation.

        ``model`` may be a single deployment name (``"gpt-5.2"``) or a
        comma-separated rotation pool (``"gpt-5.2,gpt-5.4,gpt-5.5"``).
        With a pool, each call picks ONE model round-robin (via a
        per-client atomic counter) so concurrent callers spread their
        traffic across the family rather than all hammering one
        deployment - the headline win when an experiment is
        rate-limited by a single model's TPM/RPM quota.

        404 responses are treated as "this endpoint does not carry
        this model": the ``(endpoint, model)`` combo is marked dead
        and the call immediately retries against the next endpoint
        WITHOUT consuming the retry budget. When every endpoint has
        been marked dead for the picked model, we fall through to the
        next model in the rotation pool (so a globally-unavailable
        family member doesn't poison the run). The retry budget
        (``max_retries``) governs only transient failures (5xx, 429,
        timeouts, etc.).
        """
        model_pool = self._resolve_model_pool(model)
        # Pick a deterministic starting offset per call so concurrent
        # callers see different primary models. The fallback rotation
        # below cycles through the rest of the pool only when the
        # primary is globally 404.
        start = self._next_model_index() % len(model_pool)
        rotated_pool = model_pool[start:] + model_pool[:start]
        picked_model = rotated_pool[0]
        pool_idx_in_use = 0

        use_responses_api = picked_model.lower() in {
            m.lower() for m in RESPONSES_API_MODELS
        }
        last_error: Optional[str] = None
        transient_attempts = 0
        # Hard ceiling on total iterations so a pathological mix of
        # 404s + transients can't loop forever. Worst case: visit
        # every endpoint at most once for a 404 sweep PER model in
        # the pool, then up to ``max_retries`` transient retries on
        # the survivors.
        max_iterations = (
            self.max_retries + len(self._clients) * len(model_pool)
        )
        iterations = 0

        while transient_attempts < self.max_retries and iterations < max_iterations:
            iterations += 1
            picked = self._next_alive_client_for_model(picked_model)
            if picked is None:
                # Every endpoint 404s for this model. Try the next
                # family member before giving up; this only fires
                # when a model is GLOBALLY unavailable, which is the
                # exact case where the pool's redundancy pays off.
                pool_idx_in_use += 1
                if pool_idx_in_use < len(rotated_pool):
                    picked_model = rotated_pool[pool_idx_in_use]
                    use_responses_api = picked_model.lower() in {
                        m.lower() for m in RESPONSES_API_MODELS
                    }
                    last_error = (
                        f"All endpoints 404 for previous model; "
                        f"rotating to {picked_model!r}"
                    )
                    logger.info(last_error)
                    continue
                return LLMResponse(
                    text="",
                    endpoint="",
                    model=picked_model,
                    attempts=max(transient_attempts, 1),
                    error=(
                        last_error
                        or (
                            f"All {len(self._clients)} endpoints returned 404 "
                            f"for every model in pool {model_pool!r}"
                        )
                    ),
                )
            client, endpoint = picked
            attempt_no = transient_attempts + 1
            logger.debug(
                "LLM call attempt=%d model=%s endpoint=%s api=%s",
                attempt_no,
                picked_model,
                endpoint,
                "responses" if use_responses_api else "chat",
            )
            try:
                resp = self._do_call(
                    client, endpoint, picked_model, prompt,
                    use_responses_api=use_responses_api,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                    system_prompt=system_prompt,
                )
                resp.attempts = attempt_no
                return resp

            except Exception as exc:
                if _is_404(exc):
                    # Permanently route around this combo. Do NOT
                    # consume the retry budget - the caller wanted
                    # ``max_retries`` chances at TRANSIENT recovery.
                    self._mark_dead(endpoint, picked_model)
                    last_error = (
                        f"404 from {endpoint} for {picked_model!r}: marked dead, skipping"
                    )
                    logger.info(last_error)
                    continue

                transient_attempts += 1
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "LLM call failed (attempt %d/%d, endpoint=%s): %s",
                    transient_attempts,
                    self.max_retries,
                    endpoint,
                    last_error,
                )
                if transient_attempts >= self.max_retries:
                    break
                # Exponential backoff with jitter, capped. Mirrors
                # Azure's retry guidance for 429/503 responses. Full
                # jitter (Brooker, AWS): uniform on [0, delay] avoids
                # thundering-herd when several workers retry in
                # lockstep against a shared rate-limit window.
                delay = min(
                    self.backoff_cap_seconds,
                    self.backoff_base_seconds * (2 ** (transient_attempts - 1)),
                )
                delay = random.uniform(0.0, delay)
                time.sleep(delay)

        return LLMResponse(
            text="",
            endpoint="",
            model=picked_model,
            attempts=max(transient_attempts, 1),
            error=last_error or "unknown error",
        )

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def probe_availability(
        self,
        models: List[str],
        *,
        prompt: str = "ping",
        # ``low`` is the cheapest value the GPT-5 family accepts on
        # the ``reasoning.effort`` parameter (the docs include
        # ``minimal`` but at least gpt-5.2 rejects it as of 2026-05).
        reasoning_effort: str = "low",
        reasoning_summary: str = "concise",
        system_prompt: str = "",
    ) -> Dict[str, Dict[str, str]]:
        """Probe every (endpoint, model) combination once.

        Returns a nested dict mapping ``endpoint -> model -> status``,
        where status is one of:

        * ``"ok"``        - the call returned text successfully.
        * ``"404"``       - the deployment is not available; the combo
                            is marked dead so subsequent ``query`` calls
                            skip it.
        * ``"<ErrName>"`` - any other exception class name (e.g.
                            ``"AuthenticationError"``, ``"RateLimitError"``,
                            ``"APITimeoutError"``). One probe per combo;
                            transients are NOT retried so the matrix
                            reflects the right-now state.

        Side effect: combos that 404 during the probe are added to
        :attr:`_dead_combos`, so a follow-up :meth:`query` already knows
        to skip them.
        """
        matrix: Dict[str, Dict[str, str]] = {ep: {} for ep in self.endpoints}
        for model in models:
            use_responses_api = model.lower() in {m.lower() for m in RESPONSES_API_MODELS}
            for endpoint, client in zip(self.endpoints, self._clients):
                try:
                    self._do_call(
                        client, endpoint, model, prompt,
                        use_responses_api=use_responses_api,
                        reasoning_effort=reasoning_effort,
                        reasoning_summary=reasoning_summary,
                        system_prompt=system_prompt,
                    )
                    matrix[endpoint][model] = "ok"
                except Exception as exc:
                    if _is_404(exc):
                        self._mark_dead(endpoint, model)
                        matrix[endpoint][model] = "404"
                    else:
                        matrix[endpoint][model] = type(exc).__name__
                        logger.debug(
                            "probe %s @ %s -> %s: %s",
                            model, endpoint, type(exc).__name__, exc,
                        )
        return matrix
