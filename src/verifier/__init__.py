"""Web-grounded entity verification for cluster-level findings.

The cleaning detector flags suspect cells (typos in org names, malformed
addresses, etc.) but it has no ground truth for "what *should* this say".
This package adds a post-pass that takes such clusters, runs targeted
web searches, and asks an LLM to confirm or reject the suggested fix
using only retrieved evidence.

Pipeline shape (see :mod:`src.verifier.verify`):

    pick eligible clusters  ->  web search (cached)
                            ->  one LLM call w/ snippets in-context
                            ->  structured verdict + evidence URLs

The output is a JSON file under ``outputs/<run_id>/web_verifications.json``
that the labeling UI surfaces alongside each cluster.
"""

from .gate import (
    DEFAULT_ERROR_TYPES,
    EntityKind,
    GateConfig,
    classify_cluster,
    filter_eligible,
    should_verify,
)
from .web_cache import WebCache
from .web_tool import (
    SearchHit,
    SearchResult,
    TavilyWebSearch,
    WebSearchTool,
    build_web_search_tool,
)
from .verify import (
    Verification,
    Verifier,
    VerifierConfig,
)

__all__ = [
    "DEFAULT_ERROR_TYPES",
    "EntityKind",
    "GateConfig",
    "classify_cluster",
    "filter_eligible",
    "should_verify",
    "WebCache",
    "SearchHit",
    "SearchResult",
    "TavilyWebSearch",
    "WebSearchTool",
    "build_web_search_tool",
    "Verification",
    "Verifier",
    "VerifierConfig",
]
