"""Unicode normalisation anomaly detector.

Catches:

* NFC mismatch - cell contains a combining mark that is not in
  pre-composed form (e.g. ``e\\u0301`` instead of ``\\u00E9``).
* NFKC mismatch on top of NFC - fullwidth digits / Latin letters
  (``\\uFF11`` for ``1``), ligatures (``\\uFB01``), Roman numerals,
  etc., that should be folded for downstream processing.
* Smart vs straight quotes (``\\u2018``/``\\u2019``/``\\u201C``/
  ``\\u201D``), en-dash / em-dash / minus-sign substituting for
  hyphen-minus.

Precision is high (the cell objectively contains a mark / fullwidth
char). The "is this an error?" judgement depends on the column - a
column that *should* contain Unicode prose is not an error class. The
runner exposes ``--exclude-columns`` for that.
"""

from __future__ import annotations

import unicodedata
from typing import List

import pandas as pd

from .base import HeuristicDetector, HeuristicMatch, register_detector


# Smart/curly quotes and dashes most commonly inserted by autocorrect.
# These are flagged separately because they're high-signal even when
# the rest of the cell is valid Unicode.
_SMART_QUOTES = {"\u2018", "\u2019", "\u201A", "\u201B",
                 "\u201C", "\u201D", "\u201E", "\u201F",
                 "\u2032", "\u2033"}
_FANCY_DASHES = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014",
                 "\u2015", "\u2212"}  # excluding plain hyphen-minus


@register_detector
class UnicodeNormalizationDetector(HeuristicDetector):
    name = "unicode_norm"
    default_error_type = "punctuation_inconsistency"

    def detect(self, frame: pd.DataFrame, *, source: str) -> List[HeuristicMatch]:
        out: List[HeuristicMatch] = []
        for col in frame.columns:
            for row, text in self._iter_non_blank(frame[col]):
                signals = self._inspect(text)
                if not signals:
                    continue
                # Pick the most specific signal as the primary
                # explanation; surface all of them in meta.
                primary = signals[0]
                out.append(HeuristicMatch(
                    source=source,
                    row=row,
                    column=str(col),
                    error_type=primary["error_type"],
                    detector=self.name,
                    observed=self._truncate(text),
                    explanation=primary["msg"],
                    suggested_fix=self._fix(text),
                    severity="medium",
                    meta={"signals": [s["code"] for s in signals]},
                ))
        return out

    # --- helpers ------------------------------------------------------

    def _inspect(self, text: str) -> list[dict[str, str]]:
        signals: list[dict[str, str]] = []
        if any(c in _SMART_QUOTES for c in text):
            signals.append({"code": "smart_quote",
                            "msg": "uses curly/smart quote",
                            "error_type": "punctuation_inconsistency"})
        if any(c in _FANCY_DASHES for c in text):
            signals.append({"code": "fancy_dash",
                            "msg": "uses en/em-dash or minus-sign instead of hyphen",
                            "error_type": "punctuation_inconsistency"})
        nfc = unicodedata.normalize("NFC", text)
        if nfc != text:
            signals.append({"code": "nfc_mismatch",
                            "msg": "decomposed combining marks (NFC mismatch)",
                            "error_type": "text_inconsistency"})
        nfkc = unicodedata.normalize("NFKC", nfc)
        if nfkc != nfc:
            # NFKC differs from NFC: fullwidth, ligature, compatibility.
            signals.append({"code": "nfkc_compat",
                            "msg": "compatibility characters (fullwidth / ligature / etc.)",
                            "error_type": "pattern_inconsistency"})
        return signals

    def _fix(self, text: str) -> str:
        out = unicodedata.normalize("NFKC", text)
        for q in ("\u2018", "\u2019", "\u201A", "\u201B", "\u2032"):
            out = out.replace(q, "'")
        for q in ("\u201C", "\u201D", "\u201E", "\u201F", "\u2033"):
            out = out.replace(q, '"')
        for d in _FANCY_DASHES:
            out = out.replace(d, "-")
        return out
