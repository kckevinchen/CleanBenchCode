"""Synthetic-error injection benchmark.

Take a clean CSV (the curated, zero-empty kind), inject a controlled
mix of mechanically-reproducible errors, and persist a manifest that
maps every injection back to ``(source, row, column, error_type,
original_value, injected_value)``. Score any detection run against
the manifest to get exact per-error-type precision / recall / F1.

Only error types that can be produced *deterministically* from a
single cell value (or a small coordinated multi-cell pattern) live
here. Semantic / business-rule / cross-table errors are out of
scope - they need human-curated test cases, not perturbations.
"""

from __future__ import annotations

from .base import (
    InjectionRecord,
    Injector,
    INJECTOR_REGISTRY,
    register_injector,
    WorkbookInjector,
    WORKBOOK_INJECTOR_REGISTRY,
    register_workbook_injector,
)

from . import (  # noqa: F401  (registration side-effects)
    casing,
    formatting,
    numeric,
    typo,
    whitespace_inject,
    multi_table,
)

__all__ = [
    "InjectionRecord",
    "Injector",
    "INJECTOR_REGISTRY",
    "register_injector",
    "WorkbookInjector",
    "WORKBOOK_INJECTOR_REGISTRY",
    "register_workbook_injector",
]
