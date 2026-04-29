"""Character-class pattern-minority detector.

Replace each character with its class:

* ``A`` - uppercase letter
* ``a`` - lowercase letter
* ``9`` - digit
* keep punctuation / whitespace as-is (so ``-``, ``/``, ``.``, ``:`` etc.
  remain visible in the signature)

Then group cells in a column by signature. Signatures held by < 1% of
the column are candidates - and we only fire when the modal signature
covers >= ``modal_min_share``. This is the
Trifacta / Potter's-Wheel "domain by example" heuristic.

Designed for enum-ish format columns (phone, ZIP, date, ID,
percentage). Free-text columns are excluded by the same
``max_unique_share`` gate as the casing detector.
"""

from __future__ import annotations

from collections import Counter
from typing import List

import pandas as pd

from .base import HeuristicConfig, HeuristicDetector, HeuristicMatch, register_detector


def _signature(text: str) -> str:
    out = []
    for ch in text:
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("9")
        else:
            out.append(ch)
    return "".join(out)


def _compress(sig: str) -> str:
    """Collapse runs of a single class char to ``A+``, ``a+``, ``9+``.

    Helps when the *length* varies but the structure doesn't, e.g.
    ``9-9999`` and ``9-99999`` collapse to ``9+-9+``. Punctuation is
    preserved verbatim.
    """
    if not sig:
        return sig
    out = [sig[0]]
    last_class = sig[0] if sig[0] in "Aa9" else None
    for ch in sig[1:]:
        cls = ch if ch in "Aa9" else None
        if cls is not None and cls == last_class:
            if not out[-1].endswith("+"):
                out[-1] = out[-1] + "+"
        else:
            out.append(ch)
        last_class = cls
    return "".join(out)


@register_detector
class PatternMinorityDetector(HeuristicDetector):
    name = "pattern_minority"
    default_error_type = "pattern_inconsistency"

    def __init__(
        self,
        config: HeuristicConfig | None = None,
        *,
        modal_min_share: float = 0.8,
        minority_max_share: float = 0.15,
        max_unique_share: float = 0.5,
        # Try both raw and length-compressed signatures; flag only when
        # both versions agree (raw and compressed both place the cell
        # in the minority bucket). This kills the most common FP class
        # (variable-length free-text in an otherwise enum-ish column).
        require_compressed_agreement: bool = True,
    ) -> None:
        super().__init__(config)
        self.modal_min_share = modal_min_share
        self.minority_max_share = minority_max_share
        self.max_unique_share = max_unique_share
        self.require_compressed_agreement = require_compressed_agreement

    def detect(self, frame: pd.DataFrame, *, source: str) -> List[HeuristicMatch]:
        out: List[HeuristicMatch] = []
        for col in frame.columns:
            non_blank = list(self._iter_non_blank(frame[col]))
            n = len(non_blank)
            if n < self.config.min_column_size:
                continue
            # Note: we deliberately do NOT gate on value-uniqueness
            # here. Phone numbers / IDs / serial numbers are typically
            # all-unique values yet share one format signature; the
            # modal_min_share gate below filters out genuinely free
            # text (which produces no dominant signature).

            sigs_raw = [_signature(t) for _, t in non_blank]
            sigs_comp = [_compress(s) for s in sigs_raw]
            ctr_raw = Counter(sigs_raw)
            ctr_comp = Counter(sigs_comp)
            modal_raw, modal_raw_count = ctr_raw.most_common(1)[0]
            modal_comp, _ = ctr_comp.most_common(1)[0]
            if modal_raw_count / n < self.modal_min_share:
                continue

            modal_example = next(
                t for (_, t), s in zip(non_blank, sigs_raw) if s == modal_raw
            )
            for (row, text), s_raw, s_comp in zip(non_blank, sigs_raw, sigs_comp):
                if s_raw == modal_raw:
                    continue
                share_raw = ctr_raw[s_raw] / n
                share_comp = ctr_comp[s_comp] / n
                if share_raw > self.minority_max_share:
                    continue
                if self.require_compressed_agreement:
                    if s_comp == modal_comp:
                        # Only the raw signature differs - usually a
                        # length variation; let it pass.
                        continue
                    if share_comp > self.minority_max_share:
                        continue
                out.append(HeuristicMatch(
                    source=source,
                    row=row,
                    column=str(col),
                    error_type=self.default_error_type,
                    detector=self.name,
                    observed=self._truncate(text),
                    explanation=(
                        f"format signature {s_raw!r} (share {share_raw:.0%}) "
                        f"differs from column-modal {modal_raw!r} "
                        f"(share {modal_raw_count / n:.0%}; example {modal_example!r})"
                    ),
                    suggested_fix=None,
                    severity="medium",
                    meta={
                        "signature": s_raw,
                        "signature_compressed": s_comp,
                        "modal_signature": modal_raw,
                        "modal_share": round(modal_raw_count / n, 3),
                        "minority_share": round(share_raw, 3),
                    },
                ))
        return out
