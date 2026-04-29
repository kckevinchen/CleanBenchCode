"""Casing-inconsistency injector."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, Optional, Tuple

from .base import Injector, register_injector


_HAS_LETTERS = re.compile(r"[A-Za-z]")


@register_injector
class CasingInjector(Injector):
    """Force a value into a different casing class than its current one.

    The injector picks one of {LOWER, UPPER, TITLE} that *differs*
    from the cell's current casing, so the result is always a real
    perturbation. Used together with ``--target-columns`` so the
    benchmark only flips cells in columns that should be casing-
    consistent (enums, statuses, country codes, etc.).
    """
    name = "casing"
    error_type = "casing_inconsistency"

    def applicable(self, value: str) -> bool:
        # Need at least one letter to perturb the casing.
        return bool(value) and bool(_HAS_LETTERS.search(value))

    def try_inject(
        self, value: str, *, rng: random.Random,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self.applicable(value):
            return None
        # Build the three candidate casings; drop those equal to value.
        candidates = []
        lower = value.lower()
        upper = value.upper()
        title = value.title()
        for op, candidate in (("lower", lower), ("upper", upper), ("title", title)):
            if candidate != value:
                candidates.append((op, candidate))
        if not candidates:
            return None
        op, new = rng.choice(candidates)
        return new, {"op": op}
