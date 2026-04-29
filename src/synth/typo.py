"""Single-character typo injector.

Applies one of four classic single-edit perturbations chosen
uniformly at random:

* **insert** - duplicate or random letter inserted at random position.
* **delete** - random character removed.
* **swap** - adjacent characters swapped.
* **substitute** - random character replaced by an alphabetically
  adjacent letter (catches keyboard near-misses without hauling in
  a full keyboard-adjacency map).

Designed for short alphabetic identifiers (city names, status enums,
product codes). Only applies to values that contain at least 4 letters
and no digits/symbols outside ``[A-Za-z'\\.\\- ]``.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, Optional, Tuple

from .base import Injector, register_injector


_ELIGIBLE_RE = re.compile(r"^[A-Za-z'\.\- ]{4,}$")


def _adjacent_letter(ch: str, rng: random.Random) -> str:
    if not ch.isalpha():
        return ch
    base = "a" if ch.islower() else "A"
    o = ord(ch) - ord(base)
    delta = rng.choice((-1, 1))
    new_o = max(0, min(25, o + delta))
    if new_o == o:
        new_o = (o + 1) % 26
    return chr(ord(base) + new_o)


@register_injector
class TypoInjector(Injector):
    name = "typo"
    error_type = "typo"

    def applicable(self, value: str) -> bool:
        return bool(_ELIGIBLE_RE.match(value or ""))

    def try_inject(
        self, value: str, *, rng: random.Random,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self.applicable(value):
            return None
        op = rng.choice(("insert", "delete", "swap", "substitute"))
        chars = list(value)
        if op == "insert":
            i = rng.randint(0, len(chars))
            ch = chars[max(0, i - 1)] if chars else "x"
            chars.insert(i, ch)
        elif op == "delete":
            i = rng.randint(0, len(chars) - 1)
            chars.pop(i)
        elif op == "swap" and len(chars) >= 2:
            i = rng.randint(0, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        else:  # substitute
            i = rng.randint(0, len(chars) - 1)
            chars[i] = _adjacent_letter(chars[i], rng)
        new = "".join(chars)
        if new == value:
            return None
        return new, {"op": op}
