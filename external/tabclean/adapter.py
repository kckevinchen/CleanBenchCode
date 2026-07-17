"""Route TabClean's single LLM seam through our Azure ``LLMClient``.

TabClean (``pipeline.py``) builds every agent's model via one factory,
``_make_llm(model, temperature) -> langchain_openai.ChatOpenAI``, and calls it
with ``llm.invoke(prompt) -> msg`` (reading ``msg.content`` plus token usage).
We don't have an ``OPENAI_API_KEY`` — auth is ``AzureCliCredential`` via our
framework-free :class:`LLMClient`. So we monkeypatch ``_make_llm`` to return a
thin shim whose ``.invoke`` delegates to ``LLMClient.query`` (endpoint rotation,
Responses/Chat fallback, dead-pair skipping all reused), and remap TabClean's
default model names (gpt-5-mini / gpt-5.2 / gpt-5.3-codex, which our Azure
endpoints do not serve) onto models they do.

Usage::

    import adapter
    adapter.patch()              # before build_pipeline()
    ...
    print(adapter.usage_summary())
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402

# --------------------------------------------------------------------------- #
# Model mapping: TabClean's per-agent defaults -> models our Azure pool serves.
# Override any of these via env (e.g. TABCLEAN_MODEL_CODE=gpt-5.4).
# --------------------------------------------------------------------------- #
_DEFAULT_MODEL = os.environ.get("TABCLEAN_MODEL", "gpt-5.5")
_MODEL_MAP = {
    "gpt-5-mini": os.environ.get("TABCLEAN_MODEL_DIAGNOSE", _DEFAULT_MODEL),
    "gpt-5.2": os.environ.get("TABCLEAN_MODEL_STRATEGIZE", "gpt-5.2"),
    "gpt-5.3-codex": os.environ.get("TABCLEAN_MODEL_CODE", _DEFAULT_MODEL),
    "gpt-5.2-codex": os.environ.get("TABCLEAN_MODEL_CODE", _DEFAULT_MODEL),
}
_REASONING_EFFORT = os.environ.get("TABCLEAN_REASONING_EFFORT", "medium")

# Approx Azure $/1M tokens for cost estimation (LLMResponse carries no usage).
_PRICING = {
    "gpt-5.5": (1.25, 10.00),
    "gpt-5.4": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_PRICE = (1.25, 10.00)

_CLIENT = LLMClient()
_LOCK = threading.Lock()
_USAGE: list[dict] = []


def map_model(model: str) -> str:
    return _MODEL_MAP.get(model, model)


def _record(model: str, in_tok: int, out_tok: int) -> None:
    pin, pout = _PRICING.get(model, _DEFAULT_PRICE)
    cost = in_tok * pin / 1e6 + out_tok * pout / 1e6
    with _LOCK:
        _USAGE.append({"model": model, "input": in_tok, "output": out_tok, "cost": cost})


class _Msg:
    """Minimal stand-in for a langchain message TabClean consumes."""

    def __init__(self, text: str, in_tok: int, out_tok: int):
        self.content = text
        self.response_metadata = {
            "token_usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok}
        }
        self.usage_metadata = {"input_tokens": in_tok, "output_tokens": out_tok}


class LLMClientChat:
    """Shim exposing the ``.invoke``/``.model_name`` surface TabClean uses."""

    def __init__(self, model: str):
        self.model = map_model(model)
        self.model_name = self.model

    def invoke(self, prompt):
        text_prompt = prompt if isinstance(prompt, str) else str(prompt)
        resp = _CLIENT.query(self.model, text_prompt, reasoning_effort=_REASONING_EFFORT)
        out = resp.text if resp.ok else ""
        in_tok = max(1, len(text_prompt) // 4)
        out_tok = max(1, len(out) // 4)
        _record(self.model, in_tok, out_tok)
        return _Msg(out, in_tok, out_tok)


def patch() -> None:
    """Replace ``pipeline._make_llm`` and remap the per-agent model constants."""
    import pipeline

    def _make_llm(model: str, temperature: float = 0.7) -> LLMClientChat:
        return LLMClientChat(model)

    pipeline._make_llm = _make_llm
    for const in ("MODEL_DIAGNOSE", "MODEL_REVIEW", "MODEL_STRATEGIZE",
                  "MODEL_CODE", "MODEL_DEBUGGER"):
        if hasattr(pipeline, const):
            setattr(pipeline, const, map_model(getattr(pipeline, const)))


def reset_usage() -> None:
    with _LOCK:
        _USAGE.clear()


def usage_snapshot() -> dict:
    with _LOCK:
        calls = list(_USAGE)
    return {
        "calls": len(calls),
        "input_tokens": sum(c["input"] for c in calls),
        "output_tokens": sum(c["output"] for c in calls),
        "cost": round(sum(c["cost"] for c in calls), 4),
    }


def usage_summary() -> str:
    s = usage_snapshot()
    return (f"{s['calls']} LLM calls | in~{s['input_tokens']:,} out~{s['output_tokens']:,} "
            f"| est ${s['cost']:.4f}")
