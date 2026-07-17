"""Inconsistency-detection RFT (v1).

Whole-column trainee: sees a column's distinct values + counts, emits the
minority values that depart from the dominant convention plus a
convention-matching correction. Shares one rulebook
(``prompts/rulebook.md``) between trainee and LLM-judge grader. See
``RFT.md``-style decisions in the project memory and the per-module
docstrings.
"""
from __future__ import annotations

from RFT.inconsistency.v1._prompt import (
    KINDS,
    PROMPT_INCONSISTENCY,
    RULEBOOK_PATH,
    build_response_format,
    build_user_message,
)

__all__ = [
    "KINDS",
    "PROMPT_INCONSISTENCY",
    "RULEBOOK_PATH",
    "build_response_format",
    "build_user_message",
]
