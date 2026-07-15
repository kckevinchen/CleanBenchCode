"""Inconsistency rulebook prompt-optimization loop (self-contained under baseline/).

A reviewer reads dev-set results (and may see the data), proposes generalizable
suggestions, a mechanical leakage filter strips any verbatim dev values, and a
prompt editor — which never sees data — rewrites the rulebook. The orchestrator
re-scores on dev, hill-climbs, and reports once on a held-out test split.

Entry point: ``python -m baseline.inconsistency.optimize``. See the package
README. The optimization target is ``baseline/prompts/inconsistency/rulebook.md``.
"""
