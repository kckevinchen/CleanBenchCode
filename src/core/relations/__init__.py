"""Inter-column relation pipeline (multi-column / multi-row error types).

For the error types that need cross-column reasoning the detector first
*discovers* candidate relations that should hold over a table, then
verifies, filters, and detects violations of them. The stages live in
their own modules:

* :mod:`.discovery`    - the :class:`~.discovery.Relation` data model and
                         :class:`~.discovery.SchemaDiscoverer` (stage 1:
                         enumerate candidate relations from a table digest
                         + row sample).
* :mod:`.validator`    - stage 2: ask the LLM to write a ``check(df)``
                         predicate and run it in a sandbox for an empirical
                         pass-rate (``RelationVerification``).
* :mod:`.violations`   - stage 4 (code path): compile one finding per row a
                         code-verified relation flags.
* :mod:`.llm_detector` - stage 4 (LLM fallback): whole-table scan for rules
                         that refused/errored on code.
* :mod:`.llm_verifier` - LLM-only verification variant.
* :mod:`.aggregate`    - cluster discovered relations across passes.
* :mod:`.adaptive`     - drive discovery through the Good-Turing stop rule.

The most-used symbols are re-exported here so callers can keep writing
``from src.core.relations import Relation`` after the package split;
import the other stages from their submodule
(``from src.core.relations.validator import ...``).
"""

from .discovery import (
    DiscoveryPromptConfig,
    DiscoveryReport,
    Relation,
    SchemaDiscoverer,
    _is_value_anchored,
    _parse_relations_json,
    load_discovery_prompt,
    pick_outlier_indices,
    sample_rows_for_prompt,
    write_discovery_reports_json,
    write_relations_jsonl,
)

__all__ = [
    "DiscoveryPromptConfig",
    "DiscoveryReport",
    "Relation",
    "SchemaDiscoverer",
    "_is_value_anchored",
    "_parse_relations_json",
    "load_discovery_prompt",
    "pick_outlier_indices",
    "sample_rows_for_prompt",
    "write_discovery_reports_json",
    "write_relations_jsonl",
]
