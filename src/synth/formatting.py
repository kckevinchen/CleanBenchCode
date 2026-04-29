"""Format/punctuation/number-format injectors.

* :class:`PunctuationInjector` swaps ASCII punctuation with its
  Unicode "smart" counterpart (curly quotes, en-dash, em-dash).
* :class:`NumberFormatInjector` reformats numeric cells with thousand
  separators / leading zeros / trailing zeros / sign-symbol swap.
* :class:`PatternInjector` inserts a stray punctuation character so
  the cell's pattern signature falls into the column's minority.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, Optional, Tuple

from .base import Injector, register_injector


_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


@register_injector
class PunctuationInjector(Injector):
    name = "punctuation"
    error_type = "punctuation_inconsistency"

    _SUBS = [
        ("'", "\u2019"),    # straight -> right single quote
        ('"', "\u201D"),    # straight -> right double quote
        ("-", "\u2013"),    # hyphen -> en-dash
        (" - ", " \u2014 "),  # space hyphen space -> em-dash
    ]

    def applicable(self, value: str) -> bool:
        return any(a in value for a, _ in self._SUBS)

    def try_inject(
        self, value: str, *, rng: random.Random,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        eligible = [(a, b) for a, b in self._SUBS if a in value]
        if not eligible:
            return None
        a, b = rng.choice(eligible)
        new = value.replace(a, b, 1)
        return new, {"op": f"{a!r}->{b!r}"}


@register_injector
class NumberFormatInjector(Injector):
    name = "number_format"
    error_type = "number_format_inconsistency"

    def applicable(self, value: str) -> bool:
        return bool(_NUMERIC_RE.match((value or "").strip()))

    def try_inject(
        self, value: str, *, rng: random.Random,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        v = value.strip()
        if not _NUMERIC_RE.match(v):
            return None
        # Pick one of three transforms. Skip the ones that would be no-op.
        ops: list[tuple[str, str]] = []
        # 1) Add thousands separators (only if integer part >= 4 digits).
        try:
            num = float(v)
        except ValueError:
            return None
        if "." in v:
            int_part, frac_part = v.lstrip("-").split(".", 1)
        else:
            int_part, frac_part = v.lstrip("-"), ""
        if len(int_part) >= 4:
            sign = "-" if v.startswith("-") else ""
            with_commas = sign + "{:,}".format(int(int_part))
            if frac_part:
                with_commas += "." + frac_part
            if with_commas != v:
                ops.append(("commas", with_commas))
        # 2) Leading zero pad if integer < 4 digits.
        if "." not in v and not v.startswith("-") and len(v) < 4:
            padded = v.zfill(4)
            if padded != v:
                ops.append(("zfill", padded))
        # 3) Append trailing zero in decimal cell.
        if "." in v:
            ops.append(("trailing_zero", v + "0"))
        else:
            ops.append(("trailing_dot_zero", v + ".0"))
        # 4) Replace minus with Unicode minus.
        if v.startswith("-"):
            ops.append(("unicode_minus", "\u2212" + v[1:]))
        if not ops:
            return None
        op, new = rng.choice(ops)
        return new, {"op": op}


@register_injector
class PatternInjector(Injector):
    """Insert / remove a single non-alphanumeric character to nudge
    the cell's character-class signature out of the column's modal
    pattern.
    """
    name = "pattern"
    error_type = "pattern_inconsistency"

    def applicable(self, value: str) -> bool:
        return bool(value) and len(value) >= 4

    def try_inject(
        self, value: str, *, rng: random.Random,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self.applicable(value):
            return None
        puncts = [j for j, c in enumerate(value) if not c.isalnum()]
        # Pick an op we know will succeed: drop_punct only if there's
        # something to drop, otherwise fall through to insert_punct.
        op = "drop_punct" if (puncts and rng.random() < 0.5) else "insert_punct"
        if op == "insert_punct":
            ch = rng.choice(("-", ".", "_", "/"))
            i = rng.randint(1, len(value) - 1)
            new = value[:i] + ch + value[i:]
        else:
            i = rng.choice(puncts)
            new = value[:i] + value[i + 1 :]
        if new == value:
            return None
        return new, {"op": op}
