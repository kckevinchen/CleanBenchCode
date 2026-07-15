"""Baseline GPT-5.5 typo-detection prompts (loader).

There are exactly TWO prompts, each a complete frozen string stored as a
markdown file under ``baseline/prompts/`` (the source of truth):

* ``with_rulebook.md``     -- intro + the clean, spec-derived rulebook + output
                              footer. High-precision, proper-nouns out of scope.
* ``without_rulebook.md``  -- intro + a compact generalized rule + output footer.
                              No long rulebook; keeps only the transferable
                              token-verifiability principle.

This module only READS those files (no assembly logic) and exposes them as
constants plus their sha256, so a run can record exactly which prompt it used
and any accidental edit is visible as a hash change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


WITH_RULEBOOK_PROMPT = _load("with_rulebook.md")
WITHOUT_RULEBOOK_PROMPT = _load("without_rulebook.md")

# CLI-facing keys -> prompt string.
PROMPTS: dict[str, str] = {
    "with-rulebook": WITH_RULEBOOK_PROMPT,
    "without-rulebook": WITHOUT_RULEBOOK_PROMPT,
}

PROMPT_SHA: dict[str, str] = {
    k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in PROMPTS.items()
}


if __name__ == "__main__":  # freeze check: print length + hash for each prompt
    for key, text in PROMPTS.items():
        print(f"{key:18s} {len(text):6d} chars  sha {PROMPT_SHA[key][:12]}")
