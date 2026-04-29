"""Cheap, deterministic data-quality heuristics.

This package is **isolated** from :mod:`src.core` (the LLM detection
path). It exists to provide an independent, reproducible second
opinion on what counts as an error in a CSV - useful both as a
pre-filter before the LLM runs and (the headline use case) as a
heuristic oracle to estimate the LLM's false-negative rate per
error class.

Each detector implements :class:`HeuristicDetector` and emits
:class:`HeuristicMatch` records that carry a stable
``(source, row, column)`` triple plus a taxonomy-aligned
``error_type``. Records can be lifted into the same Phase-A cluster
shape that :mod:`src.core.aggregator` produces for LLM issues, so
downstream tools (``report_builder``, the labeling UI, the
``compare_runs`` script) can consume both kinds of runs uniformly.

Design constraints:

* No LLM calls. No network. No optional heavy dependencies.
* Pure ``pandas`` + ``stdlib`` only (one optional ``rapidfuzz``
  branch is gated behind an import guard with a pure-python fallback).
* Deterministic - every detector accepts an explicit ``seed`` only
  when its output legitimately depends on randomness (none currently
  do).
* Conservative by default - precision >> recall is the rule. The
  comparison harness is responsible for combining detectors; each
  detector is judged on its own precision.
"""

from __future__ import annotations

from .base import (
    HeuristicConfig,
    HeuristicDetector,
    HeuristicMatch,
    register_detector,
    REGISTRY,
)

# Side-effect imports: every detector module decorates its class with
# @register_detector at import time. Importing them here makes the
# REGISTRY populated whenever ``import src.heuristics`` runs.
from . import (  # noqa: F401  (registration side-effects)
    casing,
    fd_violation,
    key_collision,
    numeric_outlier,
    pattern_minority,
    unicode_norm,
    whitespace,
)

__all__ = [
    "HeuristicConfig",
    "HeuristicDetector",
    "HeuristicMatch",
    "register_detector",
    "REGISTRY",
]
