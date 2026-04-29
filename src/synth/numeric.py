"""Numeric-outlier and missing-value injectors."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, Optional, Tuple

from .base import Injector, register_injector


_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


@register_injector
class NumericOutlierInjector(Injector):
    """Replace a numeric cell with a value far outside the column's
    typical range. Multiplies by a random factor in ``[100, 1000]`` and
    optionally flips sign. Used together with the column-aware builder
    that checks the perturbation actually exceeds the modified-z
    threshold."""
    name = "numeric_outlier"
    error_type = "numeric_outlier"

    def applicable(self, value: str) -> bool:
        return bool(_NUMERIC_RE.match((value or "").strip()))

    def try_inject(
        self, value: str, *, rng: random.Random,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        v = value.strip()
        try:
            num = float(v)
        except ValueError:
            return None
        # Multiply by a random factor in [100, 1000]; flip sign sometimes.
        factor = rng.uniform(100.0, 1000.0)
        sign = -1.0 if rng.random() < 0.3 else 1.0
        new_num = sign * num * factor
        # Avoid trivially-zero result.
        if new_num == 0:
            new_num = factor
        if "." in v:
            new = f"{new_num:.{len(v.split('.')[1])}f}"
        else:
            new = str(int(round(new_num)))
        if new == v:
            return None
        return new, {"op": "multiply", "factor": round(factor, 3), "sign": sign}


@register_injector
class MissingValueInjector(Injector):
    """Replace a non-blank cell with one of {empty, ``NA``, ``null``,
    ``-``, whitespace-only}. Reflects the most common ways missing
    values sneak into hand-curated tables.
    """
    name = "missing_value"
    error_type = "missing_value"

    def applicable(self, value: str) -> bool:
        return bool(value and value.strip())

    def try_inject(
        self, value: str, *, rng: random.Random,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self.applicable(value):
            return None
        op = rng.choice(("empty", "NA", "null", "-", "  "))
        new = "" if op == "empty" else op
        if new == value:
            return None
        return new, {"op": op}
