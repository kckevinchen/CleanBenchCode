"""Base types for heuristic detectors.

The schema mirrors the *flat issue* shape produced by
:mod:`src.core.detector` so ``src.core.aggregator.cluster_issues``
can fold heuristic findings into the same cluster format - that's
what lets the labeling UI and ``report_builder`` consume heuristic
runs without modification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Type

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeuristicMatch:
    """One cell-level finding from a heuristic detector.

    The required fields ``source``, ``row``, ``column``,
    ``error_type`` form the stable key used by the comparison harness
    to align heuristic findings against LLM detections.

    ``observed`` is the cell value as it appears (truncated by the
    ``observed_max_chars`` config knob), ``explanation`` is a short
    human-readable rationale, and ``suggested_fix`` is optional - the
    heuristic can leave it empty when the canonical form is unclear.

    ``severity`` is a coarse self-reported confidence on
    ``{low, medium, high}``. ``meta`` carries detector-specific
    diagnostics (e.g. which signature key matched, distance score,
    etc.) and is preserved through to the issues.jsonl output for
    downstream debugging.
    """

    source: str
    row: int
    column: str
    error_type: str
    detector: str
    observed: str = ""
    explanation: str = ""
    suggested_fix: Optional[str] = None
    severity: str = "medium"
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_issue(self, *, issue_id: Optional[str] = None) -> Dict[str, Any]:
        """Serialise into the flat-issue dict shape consumed by
        :func:`src.core.aggregator.cluster_issues`. The ``error_type_*``
        fields default to the detector's chosen ``error_type`` (no
        taxonomy lookup here - the runner enriches when it has the
        :class:`ErrorTaxonomy` in hand).
        """
        return {
            "id": issue_id or self.stable_id(),
            "source": self.source,
            "row": self.row,
            "columns": [self.column] if self.column else [],
            "error_type": self.error_type,
            "error_type_name": self.error_type,
            "error_type_category": "heuristic",
            "observed": self.observed,
            "explanation": self.explanation,
            "suggested_fix": self.suggested_fix,
            "row_values": {self.column: self.observed} if self.column else {},
            "detector": self.detector,
            "severity": self.severity,
            "meta": dict(self.meta),
        }

    def stable_id(self) -> str:
        """A deterministic id; useful when the runner doesn't supply one."""
        return f"h::{self.detector}::{self.source}::{self.row}::{self.column}::{self.error_type}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HeuristicConfig:
    """Shared knobs every detector understands.

    Detectors may add their own subclasses; the runner instantiates
    detectors through the registry, passing only this base config plus
    detector-specific kwargs from the CLI.
    """

    observed_max_chars: int = 80
    # Skip cells whose value is empty / whitespace-only. Heuristics
    # should not double-report ``missing_value`` (the LLM and the
    # curate step already handle it).
    skip_blank_cells: bool = True
    # Per-column minimum non-blank count below which the column is
    # considered too sparse to draw conclusions from. 5 is a soft
    # floor that matches the conservative default of OpenRefine's
    # clustering: anything tighter and statistical detectors over-fire
    # on single-row outliers.
    min_column_size: int = 5


# ---------------------------------------------------------------------------
# Detector base + registry
# ---------------------------------------------------------------------------


class HeuristicDetector:
    """Abstract base. Subclasses implement :meth:`detect`."""

    #: Stable, snake_case identifier used in CLI flags and output
    #: ``detector`` columns. Set on each subclass.
    name: str = ""

    #: Default taxonomy ``error_type`` slug emitted by this detector.
    #: Some detectors emit several types - in that case they fill in
    #: ``error_type`` per match.
    default_error_type: str = ""

    def __init__(self, config: Optional[HeuristicConfig] = None) -> None:
        self.config = config or HeuristicConfig()

    def detect(self, frame: pd.DataFrame, *, source: str) -> List[HeuristicMatch]:
        """Return all matches for one CSV. Must be deterministic."""
        raise NotImplementedError

    # --- helpers shared across detectors ------------------------------

    def _truncate(self, value: str) -> str:
        n = self.config.observed_max_chars
        if n <= 0 or len(value) <= n:
            return value
        return value[: n - 1] + "\u2026"

    def _iter_non_blank(
        self, series: pd.Series,
    ) -> Iterable[tuple[int, str]]:
        """Yield ``(row_index, str_value)`` for non-blank cells.

        ``row_index`` is the integer DataFrame index (the runner is
        responsible for ensuring the frame is reset to 0..N-1 before
        calling ``detect``).
        """
        for idx, value in series.items():
            if value is None:
                continue
            text = "" if pd.isna(value) else str(value)
            if self.config.skip_blank_cells and not text.strip():
                continue
            yield int(idx), text


# Class-level registry. Detectors register themselves with
# ``@register_detector`` so the CLI can resolve them by ``--detectors
# whitespace,unicode_norm`` without an explicit import list.
REGISTRY: Dict[str, Type[HeuristicDetector]] = {}


def register_detector(cls: Type[HeuristicDetector]) -> Type[HeuristicDetector]:
    if not cls.name:
        raise ValueError(f"Detector {cls.__name__} must set a non-empty 'name'")
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"Detector name collision: {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls
