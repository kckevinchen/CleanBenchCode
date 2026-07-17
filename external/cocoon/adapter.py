"""Adapter for driving the authors' upstream Cocoon code with our Azure LLMClient.

Upstream Cocoon's single LLM seam is ``cocoon_data.llm.call_llm_chat`` (also
imported into ``cocoon_data.__init__``). It normally calls OpenAI/Azure OpenAI
using API-key configuration and defaults to ``OPENAI_MODEL`` or ``gpt-4-turbo``.
This adapter imports the vendored upstream package unchanged, monkeypatches that
seam, and returns old-style OpenAI dicts consumed by Cocoon's nodes.

Faithfulness: upstream default is GPT-4 class (``gpt-4-turbo``). Our served pool
includes ``gpt-4o`` but not ``gpt-4-turbo``, so the default here is ``gpt-4o`` as
the closest GPT-4-class deployment.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "cocoon_repo"

for p in (str(REPO_ROOT), str(UPSTREAM)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.core.llm_client import LLMClient  # noqa: E402

# Approx Azure/OpenAI public list prices, $/1M tokens, used only because
# LLMResponse deliberately normalizes away provider-specific usage details.
_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5.4": (1.25, 10.00),
    "gpt-5.5": (1.25, 10.00),
}
_DEFAULT_PRICE = (2.50, 10.00)

_CLIENT = LLMClient()
_LOCK = threading.Lock()
_USAGE: list[dict] = []
_MODEL = "gpt-4o"
_REASONING_EFFORT = "low"
_PATCHED = False


def _record(model: str, in_tok: int, out_tok: int) -> None:
    pin, pout = _PRICING.get(model, _DEFAULT_PRICE)
    cost = in_tok * pin / 1e6 + out_tok * pout / 1e6
    with _LOCK:
        _USAGE.append({"model": model, "input": in_tok, "output": out_tok, "cost": cost})


def _messages_to_prompt(messages: list[dict]) -> tuple[str, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(str(content))
        else:
            user_parts.append(str(content))
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def call_llm_chat(messages, temperature=0.1, top_p=0.1, use_cache=True, api_type=None,
                  return_cost=False):
    """Cocoon-compatible replacement for ``cocoon_data.llm.call_llm_chat``."""
    del temperature, top_p, use_cache, api_type
    system, prompt = _messages_to_prompt(messages)
    resp = _CLIENT.query(_MODEL, prompt, system_prompt=system, reasoning_effort=_REASONING_EFFORT)
    text = resp.text if resp.ok else ""
    in_tok = max(1, (len(system) + len(prompt)) // 4)
    out_tok = max(1, len(text) // 4)
    _record(_MODEL, in_tok, out_tok)
    old_style = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if return_cost:
        pin, pout = _PRICING.get(_MODEL, _DEFAULT_PRICE)
        return old_style, in_tok * pin / 1e6 + out_tok * pout / 1e6
    return old_style


def load_cocoon(model: str = "gpt-4o", reasoning_effort: str = "low"):
    """Import Cocoon and patch the LLM seam. Returns the ``cocoon_data`` module."""
    global _MODEL, _REASONING_EFFORT, _PATCHED
    _MODEL = model
    _REASONING_EFFORT = reasoning_effort

    import cocoon_data
    import cocoon_data.config as config
    import cocoon_data.llm as llm

    config.cocoon_main_setting["allow_cache"] = False
    config.cocoon_main_setting["DEBUG_MODE"] = False
    config.cocoon_llm_setting["api_type"] = "openai"
    config.cocoon_llm_setting["openai_model"] = model
    llm.call_llm_chat = call_llm_chat
    llm.lru_cache.cache.clear()
    cocoon_data.call_llm_chat = call_llm_chat
    _PATCHED = True
    return cocoon_data


def usage_snapshot() -> dict:
    with _LOCK:
        calls = list(_USAGE)
    return {
        "calls": len(calls),
        "input_tokens": sum(c["input"] for c in calls),
        "output_tokens": sum(c["output"] for c in calls),
        "cost": round(sum(c["cost"] for c in calls), 4),
        "models": sorted(set(c["model"] for c in calls)),
    }


def usage_summary() -> str:
    u = usage_snapshot()
    return (f"{u['calls']} LLM calls | in~{u['input_tokens']:,} out~{u['output_tokens']:,} "
            f"| est ${u['cost']:.4f} | models={','.join(u['models'])}")
