"""Whitespace anomaly injectors."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from .base import Injector, register_injector


@register_injector
class WhitespaceInjector(Injector):
    """Inject one of: leading space, trailing space, NBSP, double space.

    The choice of perturbation is randomised for variety; the
    ``op`` parameter lets the scorer track which subtype was injected.
    """
    name = "whitespace"
    error_type = "inconsistency"

    def applicable(self, value: str) -> bool:
        return bool(value and value.strip()) and value == value.strip()

    def try_inject(
        self, value: str, *, rng: random.Random,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self.applicable(value):
            return None
        op = rng.choice(("leading_space", "trailing_space", "nbsp", "double_space"))
        if op == "leading_space":
            new = " " + value
        elif op == "trailing_space":
            new = value + " "
        elif op == "nbsp":
            # Replace one inner space with NBSP if there is one;
            # otherwise prepend an NBSP.
            if " " in value[1:-1] if len(value) > 2 else False:
                pos = value.index(" ")
                new = value[:pos] + "\u00A0" + value[pos + 1 :]
            else:
                new = "\u00A0" + value
        else:  # double_space
            if " " in value:
                pos = value.index(" ")
                new = value[:pos] + "  " + value[pos + 1 :]
            else:
                new = value + "  "
        if new == value:
            return None
        return new, {"op": op}
