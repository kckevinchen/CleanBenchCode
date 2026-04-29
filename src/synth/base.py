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
from typing import Any, Dict, Optional, Tuple, Type

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
