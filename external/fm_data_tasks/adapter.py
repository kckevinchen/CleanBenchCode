"""LLM seam for the vendored HazyResearch ``fm_data_tasks`` code.

Upstream drives GPT-3 through the ``manifest`` package:

    manifest = Manifest(...)
    pred = manifest.run(prompt, overwrite_cache=...)

The original models (``text-davinci-002/003``) are no longer served in our Azure
pool, so this shim preserves the upstream call shape while routing requests
through ``src.core.llm_client.LLMClient``.  The benchmark runner imports the
authors' prompt/serialization utilities unchanged and uses this Manifest-shaped
adapter for inference.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402

_CLIENT = LLMClient()
_LOCK = threading.Lock()
_USAGE = {"calls": 0, "prompt_chars": 0, "completion_chars": 0, "errors": 0}


class ManifestLLMClient:
    """Minimal ``manifest.Manifest``-compatible wrapper backed by ``LLMClient``."""

    def __init__(self, model: str = "gpt-4o-mini", reasoning_effort: str = "low"):
        self.model = model
        self.reasoning_effort = reasoning_effort

    def run(self, prompt: str, overwrite_cache: bool = False, **_: object) -> str:
        del overwrite_cache  # cache is handled by the benchmark runner, version-tagged.
        resp = _CLIENT.query(self.model, prompt, reasoning_effort=self.reasoning_effort)
        text = resp.text.strip() if resp.ok else ""
        with _LOCK:
            _USAGE["calls"] += 1
            _USAGE["prompt_chars"] += len(prompt)
            _USAGE["completion_chars"] += len(text)
            if not resp.ok:
                _USAGE["errors"] += 1
        return text


def usage() -> dict[str, int]:
    with _LOCK:
        return dict(_USAGE)
