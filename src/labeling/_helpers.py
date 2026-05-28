"""Leaf helpers shared by the labelling app factory and its routers.

Kept light (stdlib + fastapi only) so both ``app.py`` and the
``routers/`` modules can import from here without a circular import.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def _scalar(v: Any) -> str:
    """Coerce a (possibly ``None``) cell value to a plain string.

    Used everywhere a CSV cell is compared or rendered so equality
    matches the operator's mental model ("this value equals that").
    """
    if v is None:
        return ""
    return str(v)


def _validate_source_name(name: str) -> None:
    """Reject source names that look like paths (traversal guard)."""
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise HTTPException(400, f"invalid source name: {name!r}")
