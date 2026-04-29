"""Thin, reusable wrapper around Azure OpenAI endpoints for GPT-5 family models.

This module is a refactor of the standalone script
``combined-gpt-5-with-reasoning-summary.py``. Behaviour is preserved:

* Round-robin load balancing across a pool of endpoints
* Automatic fallback between the Responses API and the Chat Completions API
* Bounded retries with exponential-ish backoff
* Extraction of reasoning summaries when available

The client is intentionally framework-free so it can be imported from any
experiment script.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, List, Optional

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Endpoint configuration
# ---------------------------------------------------------------------------

# Default pool mirrors the endpoints used in combined-gpt-5-with-reasoning-summary.py.
# Override via the ``AZURE_OPENAI_ENDPOINTS`` env var (comma separated URLs) when
# running in a different environment.
DEFAULT_ENDPOINTS: List[str] = [
    "https://dmxaz-test-aoai.openai.azure.com/",
    "https://dmxaz-test-aoai-eu2.openai.azure.com/",
    "https://yeyehe-aoai.openai.azure.com/",
    "https://sc-rf-ma30hqbc-eastus2.cognitiveservices.azure.com/",
    "https://sc-rf-m8udnqmj-swedencentral.cognitiveservices.azure.com/",
    "https://sc-rf-ml1gih82-centralus.cognitiveservices.azure.com/",
]

DEFAULT_API_VERSION = "2025-03-01-preview"

# Models that must go through the Responses API rather than Chat Completions.
RESPONSES_API_MODELS = {
    "gpt-5-pro",
    "gpt-5",
    "gpt-5.2",
    "o1",
    "o1-pro",
    "o3",
    "o3-mini",
}


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
    >>> resp = client.query("gpt-5.2", "Say hi", reasoning_effort="minimal")
    >>> print(resp.text)
    """

    endpoints: List[str] = field(default_factory=lambda: list(DEFAULT_ENDPOINTS))
    api_version: str = DEFAULT_API_VERSION
    max_retries: int = 5
    backoff_base_seconds: float = 10.0
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
            )
            for endpoint in self.endpoints
        ]
        self._index = 0
        self._index_lock = Lock()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_client(self) -> "tuple[AzureOpenAI, str]":
        with self._index_lock:
            idx = self._index % len(self._clients)
            self._index += 1
        return self._clients[idx], self.endpoints[idx]

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

    def query(
        self,
        model: str,
        prompt: str,
        *,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        system_prompt: str = "",
    ) -> LLMResponse:
        """Query an LLM with built-in retries and endpoint rotation."""
        use_responses_api = model.lower() in {m.lower() for m in RESPONSES_API_MODELS}
        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            client, endpoint = self._next_client()
            logger.debug(
                "LLM call attempt=%d model=%s endpoint=%s api=%s",
                attempt,
                model,
                endpoint,
                "responses" if use_responses_api else "chat",
            )
            try:
                if use_responses_api:
                    responses_kwargs: dict[str, Any] = {
                        "model": model,
                        "input": prompt,
                        "reasoning": {
                            "effort": reasoning_effort,
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
                    text = self._extract_output_text(result)
                    summary_text = self._extract_reasoning_summary(result)
                    return LLMResponse(
                        text=text,
                        reasoning_summary=summary_text,
                        raw=result,
                        endpoint=endpoint,
                        model=model,
                        attempts=attempt,
                    )

                result = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = result.choices[0].message.content or ""
                return LLMResponse(
                    text=text,
                    raw=result,
                    endpoint=endpoint,
                    model=model,
                    attempts=attempt,
                )

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    last_error,
                )
                time.sleep(self.backoff_base_seconds * attempt)

        return LLMResponse(
            text="",
            endpoint="",
            model=model,
            attempts=self.max_retries,
            error=last_error or "unknown error",
        )
