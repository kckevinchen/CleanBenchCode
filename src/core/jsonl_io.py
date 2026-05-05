"""Shared JSONL read/write helpers.

Replaces five near-identical ``_read_jsonl`` copies in the
``experiments/`` runner scripts. Behaviour is the union of the
prior copies:

* Missing file -> empty list (no exception).
* Blank lines -> skipped silently.
* Per-line ``JSONDecodeError`` -> handled per ``on_error`` mode:
  ``"skip"`` (silent), ``"warn"`` (default; logger.warning once
  per offending line, parsing continues), ``"raise"``.
* Optional ``limit`` short-circuits as soon as that many records
  have been parsed - use this instead of ``read_jsonl(...)[:limit]``
  on multi-GB files so we don't materialise everything before
  slicing.

``iter_jsonl`` streams parsed records for callers that don't need
the whole file in memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional

logger = logging.getLogger(__name__)

OnError = Literal["skip", "warn", "raise"]


def iter_jsonl(
    path: Path,
    *,
    on_error: OnError = "warn",
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file. See module docstring."""
    if not path.is_file():
        return
    yielded = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if on_error == "raise":
                    raise
                if on_error == "warn":
                    logger.warning(
                        "skipped malformed JSONL line %d in %s: %s",
                        lineno, path, exc,
                    )
                continue
            yield obj
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def read_jsonl(
    path: Path,
    *,
    on_error: OnError = "warn",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read a whole JSONL file into a list. Thin wrapper around ``iter_jsonl``."""
    return list(iter_jsonl(path, on_error=on_error, limit=limit))


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write records as JSONL, one per line. Returns count written."""
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Shared LLM-JSON extraction patterns
# ---------------------------------------------------------------------------
#
# Both detector and discovery prompts ask the LLM for JSON. Real
# completions sometimes wrap the answer in ```json ... ``` fences,
# add prose, or emit just a bare ``{...}``. These two compiled
# regexes are the canonical patterns the parsers use; keeping them
# here means a fix to the regex propagates everywhere.

import re as _re

#: Match a fenced JSON block: ```...``` or ```json ... ```.
#: Matches both ``{...}`` objects and ``[...]`` arrays.
JSON_FENCE_RE = _re.compile(
    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", _re.DOTALL,
)

#: Match the largest ``{...}`` block in a string (greedy).
#: Used as a last-resort when no fence is present.
#:
#: NOTE: this regex is NOT bracket-aware. For text containing
#: multiple top-level objects (``{...} {...}``) it returns the
#: substring spanning both, which then fails ``json.loads``. Prefer
#: :func:`extract_json_payloads` for new callers.
JSON_OBJECT_RE = _re.compile(r"\{.*\}", _re.DOTALL)


def extract_json_payloads(text: str) -> list[str]:
    """Return JSON-shaped substrings from ``text``, bracket-balanced.

    Walks the string left-to-right and, for every top-level ``{`` or
    ``[``, captures the substring up to the matching close bracket,
    respecting string literals (so braces inside ``"..."`` don't
    confuse the counter). The result is a list of candidate JSON
    snippets, in encounter order, that the caller can hand to
    ``json.loads`` one by one.
    """
    if not text:
        return []
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c not in "{[":
            i += 1
            continue
        opener = c
        closer = "}" if opener == "{" else "]"
        depth = 0
        j = i
        in_string = False
        escape = False
        while j < n:
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        out.append(text[i:j + 1])
                        i = j + 1
                        break
            j += 1
        else:
            # Unbalanced - bail out; the remainder is not parseable.
            break
    return out
