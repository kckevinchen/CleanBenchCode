"""Web-grounded entity verification for cluster-level findings.

The cleaning detector flags suspect cells (typos in org names, malformed
addresses, etc.) but it has no ground truth for "what *should* this say".
This package adds a post-pass that takes such clusters, runs them
through a Microsoft Foundry agent equipped with the Web Search tool,
and parses the agent's structured verdict + URL citations.

Pipeline shape (see :mod:`src.verifier.verify`):

    pick eligible clusters  ->  build prompt with row + sibling context
                            ->  AgentBackend.verify(prompt) (one Foundry
                                Agents call; agent does its own web
                                searches via the Web Search tool)
                            ->  parse JSON verdict + filter citations

The output is a JSON file under ``outputs/<run_id>/web_verifications.json``
that the labeling UI surfaces alongside each cluster. Auth uses
``AzureCliCredential`` so no third-party API keys are required.
"""

from .agent_backend import (
    AgentBackend,
    AgentVerdict,
    AzureAgentBackend,
    Citation,
    StubAgentBackend,
    build_agent_backend,
)
from .gate import (
    DEFAULT_ERROR_TYPES,
    NON_WEB_VERIFIABLE_TYPES,
    EntityKind,
    GateConfig,
    classify_cluster,
    cluster_shape,
    filter_eligible,
    should_verify,
)
from .verify import (
    EvidenceCitation,
    Verification,
    Verifier,
    VerifierConfig,
    load_verifications,
    write_verifications,
)
from .web_cache import WebCache

__all__ = [
    "AgentBackend",
    "AgentVerdict",
    "AzureAgentBackend",
    "Citation",
    "DEFAULT_ERROR_TYPES",
    "EntityKind",
    "EvidenceCitation",
    "GateConfig",
    "NON_WEB_VERIFIABLE_TYPES",
    "StubAgentBackend",
    "Verification",
    "Verifier",
    "VerifierConfig",
    "WebCache",
    "build_agent_backend",
    "classify_cluster",
    "cluster_shape",
    "filter_eligible",
    "load_verifications",
    "should_verify",
    "write_verifications",
]
