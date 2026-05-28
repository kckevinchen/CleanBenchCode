"""Leaf helpers shared by the labelling app factory and its routers.

Kept dependency-free (stdlib only) so both ``app.py`` and the
``routers/`` modules can import from here without a circular import.
"""

from __future__ import annotations

from typing import Any


def _scalar(v: Any) -> str:
    """Coerce a (possibly ``None``) cell value to a plain string.

    Used everywhere a CSV cell is compared or rendered so equality
    matches the operator's mental model ("this value equals that").
    """
    if v is None:
        return ""
    return str(v)
