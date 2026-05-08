"""Whitespace anomaly detector.

Catches the highest-precision class of formatting errors:

* Leading or trailing ASCII space / tab.
* Internal runs of 2+ spaces (``"foo  bar"``) - excluded when the
  cell is plausibly a free-text sentence (``space_run_min`` knob).
* Embedded tab / CR / LF inside a single cell.
* Non-ASCII whitespace: NBSP (``\\u00A0``), zero-width space
  (``\\u200B``), zero-width joiner / non-joiner, BOM (``\\uFEFF``),
  ideographic space (``\\u3000``).

Precision is essentially 100% for "this character is in the cell";
the policy question (is removing it actually correct?) is what the
labeling pass answers.
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd

from .base import HeuristicConfig, HeuristicDetector, HeuristicMatch, register_detector


# Invisible / unusual whitespace characters that almost always indicate
# a copy-paste mishap rather than intentional content. ASCII space is
# explicitly excluded - we only flag *runs* of it.
_INVISIBLE_WS = {
    "\u00A0": "NBSP",
    "\u1680": "OGHAM_SPACE",
    "\u2000": "EN_QUAD", "\u2001": "EM_QUAD",
    "\u2002": "EN_SPACE", "\u2003": "EM_SPACE",
    "\u2004": "THREE_PER_EM", "\u2005": "FOUR_PER_EM",
    "\u2006": "SIX_PER_EM", "\u2007": "FIGURE_SPACE",
    "\u2008": "PUNCTUATION_SPACE", "\u2009": "THIN_SPACE",
    "\u200A": "HAIR_SPACE",
    "\u200B": "ZERO_WIDTH_SPACE",
    "\u200C": "ZERO_WIDTH_NON_JOINER",
    "\u200D": "ZERO_WIDTH_JOINER",
    "\u2028": "LINE_SEPARATOR",
    "\u2029": "PARAGRAPH_SEPARATOR",
    "\u202F": "NARROW_NBSP",
    "\u205F": "MEDIUM_MATH_SPACE",
    "\u3000": "IDEOGRAPHIC_SPACE",
    "\uFEFF": "BOM_OR_ZWNBSP",
}
_INVISIBLE_RE = re.compile("[" + "".join(_INVISIBLE_WS) + "]")
_TRAILING_LEADING_RE = re.compile(r"^[ \t]+|[ \t]+$")
_INTERNAL_RUN_RE = re.compile(r"[ \t]{2,}")
_EMBEDDED_NEWLINE_RE = re.compile(r"[\r\n\t]")


@register_detector
class WhitespaceDetector(HeuristicDetector):
    name = "whitespace"
    default_error_type = "inconsistency"

    def __init__(
        self,
        config: HeuristicConfig | None = None,
        *,
        flag_internal_runs: bool = True,
        # Cells with average word length below this are treated as free
        # text where double-spaces between sentences are a style
        # choice, not an error. 4 keeps "Mr.  Smith" out but flags
        # "Tier  1" / "$ 100".
        free_text_avg_word_len: float = 16.0,
    ) -> None:
        super().__init__(config)
        self.flag_internal_runs = flag_internal_runs
        self.free_text_avg_word_len = free_text_avg_word_len

    def detect(self, frame: pd.DataFrame, *, source: str) -> List[HeuristicMatch]:
        out: List[HeuristicMatch] = []
        for col in frame.columns:
            for row, text in self._iter_non_blank(frame[col]):
                problems = self._inspect(text)
                if not problems:
                    continue
                out.append(HeuristicMatch(
                    source=source,
                    row=row,
                    column=str(col),
                    error_type="inconsistency",
                    detector=self.name,
                    observed=self._truncate(text),
                    explanation="; ".join(p["msg"] for p in problems),
                    suggested_fix=self._fix(text),
                    severity="high",
                    meta={"signals": [p["code"] for p in problems]},
                ))
        return out

    # --- helpers ------------------------------------------------------

    def _inspect(self, text: str) -> list[dict[str, str]]:
        problems: list[dict[str, str]] = []
        if _TRAILING_LEADING_RE.search(text):
            problems.append({"code": "edge_whitespace",
                             "msg": "leading or trailing whitespace"})
        if _EMBEDDED_NEWLINE_RE.search(text):
            problems.append({"code": "embedded_control",
                             "msg": "embedded tab/newline character"})
        m = _INVISIBLE_RE.search(text)
        if m is not None:
            ch = m.group(0)
            problems.append({
                "code": f"invisible_{_INVISIBLE_WS.get(ch, 'unknown').lower()}",
                "msg": f"invisible whitespace ({_INVISIBLE_WS.get(ch, repr(ch))})",
            })
        if self.flag_internal_runs and _INTERNAL_RUN_RE.search(text):
            # Ignore in plausible free-text cells: skip if average
            # alpha-token length is short (sentence-like) AND the cell
            # has 5+ tokens.
            tokens = [t for t in re.split(r"\s+", text.strip()) if t]
            avg_len = (sum(len(t) for t in tokens) / max(1, len(tokens)))
            if not (len(tokens) >= 5 and avg_len <= self.free_text_avg_word_len):
                problems.append({"code": "internal_double_space",
                                 "msg": "two or more consecutive spaces inside cell"})
        return problems

    def _fix(self, text: str) -> str:
        # Canonical: NFC-equivalent ASCII-space rewrite of the cell.
        out = text
        out = _INVISIBLE_RE.sub(" ", out)
        out = _EMBEDDED_NEWLINE_RE.sub(" ", out)
        out = _INTERNAL_RUN_RE.sub(" ", out)
        return out.strip()
