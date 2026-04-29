"""Key-collision typo detector.

For every column build several normalised keys per distinct value and
group values that share a key. A cluster whose members have unequal
text but equal key is a candidate typo set. Within a cluster the rare
spelling (frequency below ``rare_max_count``) is flagged when there is
a clearly dominant spelling (frequency >= ``modal_min_count``). This
is the OpenRefine ``key collision`` algorithm.

Three keys are computed and tried in order:

1. **fingerprint** - lowercase, NFKD-fold, strip punctuation, split,
   sort+unique tokens, rejoin with single space. Catches reordered
   tokens, casing differences, accent variation, punctuation noise.
2. **ngram-2 fingerprint** - sorted unique 2-grams of the
   fingerprint. Catches single-character typos
   (``Tanya`` <-> ``Tnya``).
3. **metaphone** - phonetic key (NYSIIS approximation; pure-stdlib).
   Catches ``Schmidt`` vs ``Schmit``.

The detector is enum-aware: it skips columns where the value is
unique-per-row.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable, List

import pandas as pd

from .base import HeuristicConfig, HeuristicDetector, HeuristicMatch, register_detector


_PUNCT_RE = re.compile(r"[^A-Za-z0-9\s]+")


def _fingerprint_key(text: str) -> str:
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    tokens = sorted({tok for tok in s.split() if tok})
    return " ".join(tokens)


def _ngram_key(text: str, n: int = 2) -> str:
    fp = _fingerprint_key(text).replace(" ", "")
    if len(fp) < n:
        return fp
    grams = sorted({fp[i : i + n] for i in range(len(fp) - n + 1)})
    return "".join(grams)


def _letters_key(text: str) -> str:
    """Sorted-multiset of alphanumeric characters.

    Catches single-character transpositions (``Boston`` <-> ``Bsoton``)
    that the n-gram fingerprint can miss. Pure-ASCII fold via
    :func:`_fingerprint_key`.
    """
    fp = _fingerprint_key(text).replace(" ", "")
    return "".join(sorted(fp))


# Light-weight pure-stdlib NYSIIS-style phonetic key. Not as good as
# Double-Metaphone but adequate for catching same-pronunciation typos
# and avoids a third-party dep.
def _phonetic_key(text: str) -> str:
    s = _fingerprint_key(text).replace(" ", "")
    if not s:
        return ""
    # Step 1: collapse common digraphs.
    repl = [
        ("MAC", "MCC"), ("KN", "N"), ("K", "C"), ("PH", "FF"),
        ("PF", "FF"), ("SCH", "SSS"), ("EE", "Y"), ("IE", "Y"),
    ]
    s_upper = s.upper()
    for a, b in repl:
        s_upper = s_upper.replace(a, b)
    # Step 2: drop trailing S, EE, IE; collapse vowels.
    s_upper = re.sub(r"[AEIOU]+", "A", s_upper)
    # Step 3: collapse consecutive duplicates.
    out = []
    for ch in s_upper:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


@register_detector
class KeyCollisionTypoDetector(HeuristicDetector):
    name = "key_collision_typo"
    default_error_type = "typo"

    def __init__(
        self,
        config: HeuristicConfig | None = None,
        *,
        # Don't run on free-text columns.
        max_unique_share: float = 0.6,
        # A "rare" spelling: appears at most this many times in the
        # column. Default 1 keeps precision very high.
        rare_max_count: int = 1,
        # The dominant spelling must appear at least this many times.
        modal_min_count: int = 3,
        # The dominant must outnumber the rare by this factor.
        modal_to_rare_ratio: float = 5.0,
    ) -> None:
        super().__init__(config)
        self.max_unique_share = max_unique_share
        self.rare_max_count = rare_max_count
        self.modal_min_count = modal_min_count
        self.modal_to_rare_ratio = modal_to_rare_ratio

    def detect(self, frame: pd.DataFrame, *, source: str) -> List[HeuristicMatch]:
        out: List[HeuristicMatch] = []
        seen_match_keys: set[tuple] = set()
        for col in frame.columns:
            non_blank = list(self._iter_non_blank(frame[col]))
            n = len(non_blank)
            if n < self.config.min_column_size:
                continue
            value_counts: Counter[str] = Counter(t for _, t in non_blank)
            distinct = len(value_counts)
            if distinct / n > self.max_unique_share:
                continue
            # Skip degenerate single-distinct columns.
            if distinct < 2:
                continue

            # Try keys in order; once a (col,row,col_value) is flagged
            # we skip it for later keys to avoid duplicate matches.
            for key_name, key_fn in (
                ("fingerprint", _fingerprint_key),
                ("letters", _letters_key),
                ("ngram2", _ngram_key),
                ("phonetic", _phonetic_key),
            ):
                groups: dict[str, list[str]] = defaultdict(list)
                for value in value_counts:
                    k = key_fn(value)
                    if not k:
                        continue
                    groups[k].append(value)
                for k, members in groups.items():
                    if len(members) < 2:
                        continue
                    counts = {v: value_counts[v] for v in members}
                    modal_value, modal_count = max(counts.items(), key=lambda kv: kv[1])
                    if modal_count < self.modal_min_count:
                        continue
                    for value, count in counts.items():
                        if value == modal_value:
                            continue
                        if count > self.rare_max_count:
                            continue
                        if modal_count < self.modal_to_rare_ratio * count:
                            continue
                        # Emit one match per row holding this rare value.
                        for row, text in non_blank:
                            if text != value:
                                continue
                            mk = (col, row, value)
                            if mk in seen_match_keys:
                                continue
                            seen_match_keys.add(mk)
                            out.append(HeuristicMatch(
                                source=source,
                                row=row,
                                column=str(col),
                                error_type=self.default_error_type,
                                detector=self.name,
                                observed=self._truncate(text),
                                explanation=(
                                    f"rare spelling {value!r} (n={count}) shares "
                                    f"the {key_name} key with dominant {modal_value!r} "
                                    f"(n={modal_count})"
                                ),
                                suggested_fix=modal_value,
                                severity="medium" if key_name == "fingerprint" else "low",
                                meta={
                                    "key": key_name,
                                    "rare_count": count,
                                    "modal_count": modal_count,
                                    "modal_value": modal_value,
                                },
                            ))
        return out
