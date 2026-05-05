"""Base types for the synthetic-error injection package.

Each injector implements :class:`Injector.try_inject` which takes a
single cell value and a deterministic :class:`random.Random` and
returns either a ``(new_value, params)`` pair or ``None`` if the
injector doesn't apply to that cell. The benchmark builder is
responsible for picking cells, calling ``try_inject``, and writing
the resulting :class:`InjectionRecord` to the manifest.

Determinism is the headline property - every run with the same
``seed``, ``rate``, and CSV input produces byte-identical outputs.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING, Tuple, Type

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InjectionRecord:
    """One injection. Persisted to ``injection_manifest.jsonl``.

    The combination ``(source, row, column)`` is the primary key used
    by ``score_synth_benchmark`` to align detected issues with
    injected ones. ``error_type`` aligns with the existing taxonomy
    (``prompts/error_types.yaml``) so per-type recall is directly
    comparable to LLM detector outputs.
    """

    source: str
    row: int
    column: str
    error_type: str
    injector: str
    original_value: str
    injected_value: str
    params: Dict[str, Any] = field(default_factory=dict)


class Injector:
    """Abstract base. Subclasses implement :meth:`try_inject`."""

    #: Stable, snake_case identifier used in the CLI ``--error-types``
    #: flag and in the manifest's ``injector`` column.
    name: str = ""

    #: Taxonomy ``error_type`` slug emitted on the InjectionRecord.
    error_type: str = ""

    def try_inject(
        self, value: str, *, rng: random.Random,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        raise NotImplementedError

    def applicable(self, value: str) -> bool:
        """Cheap pre-filter so the builder doesn't pick cells the
        injector cannot perturb. Default: anything non-blank."""
        return bool(value and value.strip())


INJECTOR_REGISTRY: Dict[str, Type[Injector]] = {}


def register_injector(cls: Type[Injector]) -> Type[Injector]:
    if not cls.name:
        raise ValueError(f"Injector {cls.__name__} must set 'name'")
    if cls.name in INJECTOR_REGISTRY and INJECTOR_REGISTRY[cls.name] is not cls:
        raise ValueError(f"Injector name collision: {cls.name!r}")
    INJECTOR_REGISTRY[cls.name] = cls
    return cls


# ---------------------------------------------------------------------------
# Workbook-scope injectors (cross-sheet perturbations) -- Gap G
# ---------------------------------------------------------------------------


class WorkbookInjector:
    """Cross-sheet perturbation operating on a multi-sheet workbook.

    Unlike the per-cell :class:`Injector` protocol, workbook injectors
    take a mapping ``{sheet: DataFrame}`` and return a *new* mapping
    with the perturbation applied across one or more sheets, plus the
    list of :class:`InjectionRecord` describing every cell touched.

    Implementations should be deterministic given the same inputs and
    ``rng`` and must not mutate the supplied frames in place
    (callers should pass copies, but defensive ``.copy()`` is cheap).

    Subclass contract
    -----------------
    * Set the class-level ``name`` (snake_case, used in CLI flags +
      manifest) and ``error_type`` (taxonomy slug emitted on each
      :class:`InjectionRecord`).
    * Override :meth:`try_inject` to return
      ``(new_frames, [InjectionRecord, ...])`` on success or
      ``None`` when the injector cannot apply to this workbook.
    * Optionally override :meth:`applicable` for cheaper pre-checks
      (default: ``len(frames) >= min_sheets``).
    * Override ``min_sheets`` if more than 2 sibling sheets are
      required.
    """

    name: str = ""
    error_type: str = ""
    #: Minimum number of sheets in the workbook required for this
    #: injector to apply.
    min_sheets: int = 2

    def applicable(self, frames: "Mapping[str, pd.DataFrame]") -> bool:
        return len(frames) >= self.min_sheets

    def try_inject(
        self,
        frames: "Mapping[str, pd.DataFrame]",
        *,
        rng: random.Random,
    ) -> "Optional[Tuple[Dict[str, pd.DataFrame], List[InjectionRecord]]]":
        """Return ``(new_frames, [InjectionRecord, ...])`` or ``None``
        if the injector cannot apply to this workbook (e.g. no
        suitable shared column)."""
        raise NotImplementedError


WORKBOOK_INJECTOR_REGISTRY: Dict[str, Type[WorkbookInjector]] = {}


def register_workbook_injector(cls: Type[WorkbookInjector]) -> Type[WorkbookInjector]:
    if not cls.name:
        raise ValueError(f"WorkbookInjector {cls.__name__} must set 'name'")
    if cls.name in WORKBOOK_INJECTOR_REGISTRY and WORKBOOK_INJECTOR_REGISTRY[cls.name] is not cls:
        raise ValueError(f"WorkbookInjector name collision: {cls.name!r}")
    WORKBOOK_INJECTOR_REGISTRY[cls.name] = cls
    return cls
