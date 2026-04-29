"""Numeric-outlier detector based on the modified z-score (MAD).

The classic mean+sigma outlier flag is too sensitive on heavy-tailed
distributions; the modified z-score (Iglewicz & Hoaglin, 1993) uses
median + median-absolute-deviation and is robust at small N.

For each column where >= ``numeric_min_share`` of non-blank cells
parse as a finite float, flag the cells whose modified-z exceeds
``threshold`` (default 3.5 - the textbook value).
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import HeuristicConfig, HeuristicDetector, HeuristicMatch, register_detector


# Strip currency symbols, thousands separators, percent signs etc.
# before float() so columns that store "$1,234.50" still parse.
_NUMERIC_STRIP_RE = re.compile(r"[\s$%,\u00A0\u202F]")


def _coerce_float(text: str) -> Optional[float]:
    s = _NUMERIC_STRIP_RE.sub("", text)
    if not s or s in {"-", "+", "."}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not np.isfinite(v):
        return None
    return v


@register_detector
class NumericOutlierDetector(HeuristicDetector):
    name = "numeric_outlier"
    default_error_type = "numeric_outlier"

    def __init__(
        self,
        config: HeuristicConfig | None = None,
        *,
        numeric_min_share: float = 0.9,
        threshold: float = 3.5,
        # Don't fire on tiny columns - MAD is unstable at low N.
        min_n: int = 12,
    ) -> None:
        super().__init__(config)
        self.numeric_min_share = numeric_min_share
        self.threshold = threshold
        self.min_n = min_n

    def detect(self, frame: pd.DataFrame, *, source: str) -> List[HeuristicMatch]:
        out: List[HeuristicMatch] = []
        for col in frame.columns:
            non_blank = list(self._iter_non_blank(frame[col]))
            if len(non_blank) < self.min_n:
                continue
            parsed: list[tuple[int, str, float]] = []
            for row, text in non_blank:
                v = _coerce_float(text)
                if v is None:
                    continue
                parsed.append((row, text, v))
            if len(parsed) / len(non_blank) < self.numeric_min_share:
                continue
            values = np.asarray([v for _, _, v in parsed], dtype=float)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad == 0.0:
                # Degenerate (constant or near-constant): skip rather
                # than divide-by-zero; the pattern detector covers
                # these.
                continue
            for row, text, v in parsed:
                z = 0.6745 * (v - median) / mad
                if abs(z) < self.threshold:
                    continue
                out.append(HeuristicMatch(
                    source=source,
                    row=row,
                    column=str(col),
                    error_type=self.default_error_type,
                    detector=self.name,
                    observed=text,
                    explanation=(
                        f"value {v} is a robust outlier (modified z-score "
                        f"{z:.2f}, |z|>{self.threshold:g}; "
                        f"column median={median:g}, MAD={mad:g})"
                    ),
                    suggested_fix=None,
                    severity="medium" if abs(z) < 6 else "high",
                    meta={
                        "modified_z": round(z, 3),
                        "median": median,
                        "mad": mad,
                    },
                ))
        return out
