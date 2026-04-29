"""Casing-fingerprint outlier detector.

For each column, collapse every value into a 5-bit casing fingerprint:

* ``UPPER``        - all letters uppercase (e.g. ``USA``)
* ``LOWER``        - all letters lowercase (``usa``)
* ``TITLE``        - first letter of each word uppercase, rest lower
                     (``United States``)
* ``MIXED``        - mixed case that doesn't fit the above
                     (``iPhone`` / ``McDonalds``)
* ``NO_LETTERS``   - cell has no letters (``42``, ``$1.23``)

The column's modal fingerprint defines its "canonical" casing; cells
with a different fingerprint that share the same fingerprint key
(case-folded text) as a modal-fingerprint cell are flagged as casing
inconsistencies.

The minority threshold prevents firing on legitimately mixed columns
(free-text descriptions, product names): we only flag if the modal
fingerprint covers >= ``modal_min_share`` of values.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import List

import pandas as pd

from .base import HeuristicConfig, HeuristicDetector, HeuristicMatch, register_detector


_WORD_RE = re.compile(r"[A-Za-z]+")


def _casing_fingerprint(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "NO_LETTERS"
    if all(c.isupper() for c in letters):
        return "UPPER"
    if all(c.islower() for c in letters):
        return "LOWER"
    # Title: each whitespace-delimited word is capitalised.
    words = _WORD_RE.findall(text)
    if words and all(w[0].isupper() and w[1:].islower() for w in words if len(w) >= 2):
        # Allow single-letter words to be either case (initials like
        # "John F Kennedy" pass).
        return "TITLE"
    return "MIXED"


@register_detector
class CasingFingerprintDetector(HeuristicDetector):
    name = "casing_fingerprint"
    default_error_type = "casing_inconsistency"

    def __init__(
        self,
        config: HeuristicConfig | None = None,
        *,
        modal_min_share: float = 0.7,
        # Minimum number of distinct casefolded values needed before
        # we treat the column as enum-like enough to apply this
        # detector. Free-text columns (descriptions, notes) typically
        # have ~unique-per-row values and would over-fire.
        max_unique_share: float = 0.5,
    ) -> None:
        super().__init__(config)
        self.modal_min_share = modal_min_share
        self.max_unique_share = max_unique_share

    def detect(self, frame: pd.DataFrame, *, source: str) -> List[HeuristicMatch]:
        out: List[HeuristicMatch] = []
        for col in frame.columns:
            non_blank = list(self._iter_non_blank(frame[col]))
            n = len(non_blank)
            if n < self.config.min_column_size:
                continue
            # Free-text gate: skip if too unique.
            distinct = {text.casefold() for _, text in non_blank}
            if len(distinct) / n > self.max_unique_share:
                continue

            fps_per_value: dict[str, list[tuple[int, str]]] = defaultdict(list)
            fps_counter: Counter[str] = Counter()
            for row, text in non_blank:
                fp = _casing_fingerprint(text)
                fps_per_value[text.casefold()].append((row, text))
                fps_counter[fp] += 1
            modal_fp, modal_count = fps_counter.most_common(1)[0]
            if modal_count / n < self.modal_min_share:
                continue
            if modal_fp == "NO_LETTERS":
                continue

            # For every casefold-key with both modal-fp and non-modal-fp
            # cells, the non-modal cells are the candidates.
            for key, members in fps_per_value.items():
                if len(members) < 2:
                    continue
                fps_seen = {_casing_fingerprint(t) for _, t in members}
                if modal_fp not in fps_seen or len(fps_seen) < 2:
                    continue
                modal_example = next(t for _, t in members if _casing_fingerprint(t) == modal_fp)
                for row, text in members:
                    fp = _casing_fingerprint(text)
                    if fp == modal_fp:
                        continue
                    out.append(HeuristicMatch(
                        source=source,
                        row=row,
                        column=str(col),
                        error_type=self.default_error_type,
                        detector=self.name,
                        observed=self._truncate(text),
                        explanation=(
                            f"casing differs from column-modal {modal_fp!r} "
                            f"(saw {fp!r}; example: {modal_example!r})"
                        ),
                        suggested_fix=modal_example,
                        severity="medium",
                        meta={
                            "modal_fingerprint": modal_fp,
                            "cell_fingerprint": fp,
                            "modal_share": round(modal_count / n, 3),
                        },
                    ))
        return out
