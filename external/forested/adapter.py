"""Route ForestED's OpenAI calls through our Azure ``LLMClient``.

ForestED (``decision_forest.py``) instantiates ``OpenAI(api_key="")`` at import
time and calls ``openai_client.chat.completions.create(model, messages,
temperature)`` inside ``call_llm_generate`` to induce a decision-tree spec. We
don't have an ``OPENAI_API_KEY`` (auth is ``AzureCliCredential`` via our
framework-free :class:`LLMClient`), and the explicit empty ``api_key=""``
overrides any env var -- so we must (1) stub ``openai.OpenAI`` *before* importing
the module, then (2) replace the module's ``openai_client`` with a shim whose
``.chat.completions.create`` delegates to ``LLMClient.query``.

Usage::

    import adapter
    DF = adapter.load_decision_forest(model="gpt-5.5")   # imports + patches
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402

_CLIENT = LLMClient()
_LOCK = threading.Lock()
_USAGE = {"calls": 0, "in": 0, "out": 0}
_REASONING_EFFORT = "low"


class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _Usage:
    def __init__(self, pt, ct):
        self.prompt_tokens = pt
        self.completion_tokens = ct
        self.total_tokens = pt + ct


class _Resp:
    def __init__(self, content, pt, ct):
        self.choices = [_Msg(content)]
        self.usage = _Usage(pt, ct)


class _Completions:
    def create(self, *, model, messages, temperature=1.0, **kw):
        system = ""
        user = ""
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                user = m.get("content", "")
        resp = _CLIENT.query(model, user, system_prompt=system,
                             reasoning_effort=_REASONING_EFFORT)
        text = resp.text if resp.ok else ""
        pt = max(1, (len(system) + len(user)) // 4)
        ct = max(1, len(text) // 4)
        with _LOCK:
            _USAGE["calls"] += 1
            _USAGE["in"] += pt
            _USAGE["out"] += ct
        return _Resp(text, pt, ct)


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _StubClient:
    """Stands in for ``openai.OpenAI(...)`` and routes to our LLMClient."""

    def __init__(self, *a, **k):
        self.chat = _Chat()


def load_decision_forest(model: str = "gpt-5.5", reasoning_effort: str = "low"):
    """Import ForestED's ``decision_forest`` with the OpenAI seam replaced by our
    Azure ``LLMClient``. Returns the patched module."""
    global _REASONING_EFFORT
    _REASONING_EFFORT = reasoning_effort

    import openai
    openai.OpenAI = _StubClient  # so module-level OpenAI(api_key="") is harmless

    src = Path(__file__).resolve().parent / "ForestED"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import decision_forest as DF
    # Replace the (already-constructed) module-global clients with our shim.
    DF.openai_client = _StubClient()
    if hasattr(DF, "ds_client"):
        DF.ds_client = _StubClient()
    if hasattr(DF, "nb_client"):
        DF.nb_client = _StubClient()

    # The consensus voter hardcodes ``device="cuda"`` as its default; force the
    # module's resolved DEVICE (cpu on this box) so it runs without a GPU.
    _cpu = str(DF.DEVICE)
    if hasattr(DF, "majority_voting_consensus"):
        _orig_mv = DF.majority_voting_consensus

        def _mv(partition_reports, dirty, strategy="majority", device=None):
            return _orig_mv(partition_reports, dirty, strategy, _cpu)

        DF.majority_voting_consensus = _mv
    return DF


def usage():
    with _LOCK:
        return dict(_USAGE)
