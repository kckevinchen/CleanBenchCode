"""Classical (non-LLM) cell-level error detectors for benchmark/merged_v2.

Three zero-shot, label-free paradigms that the existing roster lacked, run over
the entire CSV and emitting ``(row, col)`` cells exactly like every other baseline
(scored by ``external/rein/score.py``):

* ``pattern``    -- per-column character-class *format* outliers (minority format
  in a column with a dominant pattern). Targets ``inconsistency`` /
  ``typo-formatting``. The "Auto-Detect / Potter's Wheel" pattern-profiling idea.
* ``openrefine`` -- OpenRefine-style key-collision clustering (fingerprint +
  n-gram fingerprint). Within a cluster of values sharing a canonical key, the
  minority raw variants are flagged as inconsistent representations. Targets
  ``inconsistency`` and casing/punctuation/token-order ``semantic`` variants.
* ``pyod``       -- per numeric column, a PyOD ensemble (ECOD/COPOD/KNN/LOF) flags
  consensus outlier cells. Targets ``numeric_outlier``.

All are deterministic and dependency-light (pandas; ``pyod`` only for ``pyod``).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd

Cell = tuple[int, int]


# --------------------------------------------------------------------------- #
# 1) Pattern / format-outlier detector
# --------------------------------------------------------------------------- #
_RUN_RE = [(re.compile("A+"), "A"), (re.compile("a+"), "a"),
           (re.compile("9+"), "9"), (re.compile(" +"), " ")]


def char_class_signature(value: str, collapse: bool = True) -> str:
    """Map a value to a character-class signature: uppercase->A, lowercase->a,
    digit->9, whitespace->space, punctuation kept literally. Consecutive runs of
    a class are collapsed (so values differ only by *format*, not length)."""
    out = []
    for ch in value:
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("9")
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(ch)
    s = "".join(out)
    if collapse:
        for rx, rep in _RUN_RE:
            s = rx.sub(rep, s)
    return s


def detect_pattern(df: pd.DataFrame, *, dom_frac: float = 0.75,
                   minority_frac: float = 0.08, min_rows: int = 8,
                   min_distinct: int = 15, max_sig_cardinality: float = 0.5) -> set[Cell]:
    """Flag cells whose format signature is a clear minority in a column that has
    a dominant format. Skips (a) free-text / high-cardinality columns with no
    dominant pattern and (b) low-cardinality categorical *enums* (few distinct
    values), where a minority value is a legitimate category, not a format error
    -- those casing/punctuation variants are handled by the clustering detector.
    Targets format-regular columns (dates, codes, numbers, units)."""
    out: set[Cell] = set()
    for j, col in enumerate(df.columns):
        vals = df[col].astype(str).tolist()
        nonempty = [(i, v) for i, v in enumerate(vals) if v.strip()]
        if len(nonempty) < min_rows:
            continue
        # Skip categorical enums: few distinct values -> minorities are real
        # categories, not format outliers (e.g. a rare Status label).
        if len({v for _, v in nonempty}) < min_distinct:
            continue
        sigs = {i: char_class_signature(v) for i, v in nonempty}
        counts = Counter(sigs.values())
        total = len(nonempty)
        dom_sig, dom_n = counts.most_common(1)[0]
        if dom_n / total < dom_frac:
            continue  # no dominant format (free text)
        if len(counts) / total > max_sig_cardinality:
            continue
        thresh = max(1, int(minority_frac * total))
        for i, sig in sigs.items():
            if sig != dom_sig and counts[sig] <= thresh:
                out.add((i, j))
    return out


# --------------------------------------------------------------------------- #
# 2) OpenRefine key-collision clustering
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_NONWORD_RE = re.compile(r"[^\w]", re.UNICODE)


def fingerprint(value: str) -> str:
    """OpenRefine 'fingerprint' key: trim, lowercase, drop punctuation, split to
    tokens, dedupe + sort, rejoin. Collapses casing/punctuation/token-order
    variants of the same value to one key."""
    s = _PUNCT_RE.sub(" ", value.strip().lower())
    toks = sorted(set(s.split()))
    return " ".join(toks)


def ngram_fingerprint(value: str, n: int = 2) -> str:
    """OpenRefine 'n-gram fingerprint': lowercase, drop non-word chars, sorted
    unique char n-grams. Catches near-duplicates / small typos as collisions."""
    s = _NONWORD_RE.sub("", value.strip().lower())
    if len(s) < n:
        return s
    grams = sorted(set(s[i:i + n] for i in range(len(s) - n + 1)))
    return "".join(grams)


def _cluster_minority(values_by_row: list[tuple[int, str]], keyfn) -> set[int]:
    """Group rows by ``keyfn(value)``; within each cluster of >=2 DISTINCT raw
    values, flag rows whose raw value is not the (strict) majority canonical form."""
    by_key: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for i, v in values_by_row:
        k = keyfn(v)
        if k:
            by_key[k].append((i, v))
    flagged: set[int] = set()
    for k, members in by_key.items():
        raw_counts = Counter(v for _, v in members)
        if len(raw_counts) < 2:
            continue  # no variation -> nothing inconsistent
        (canon, canon_n), = raw_counts.most_common(1)
        # Require a strict majority canonical form; skip ambiguous ties.
        if list(raw_counts.values()).count(canon_n) > 1:
            continue
        for i, v in members:
            if v != canon:
                flagged.add(i)
    return flagged


_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def detect_openrefine(df: pd.DataFrame, *, min_rows: int = 8,
                      use_ngram: bool = True, min_letter_frac: float = 0.5) -> set[Cell]:
    """Flag minority raw variants that collapse to a shared canonical key
    (fingerprint, and optionally n-gram fingerprint). Restricted to *textual*
    columns (casing/punctuation/token-order normalization, OpenRefine's intent):
    columns that are mostly numeric/date are skipped and non-letter values are
    ignored, so genuinely-distinct numbers/dates aren't merged by token-sorting."""
    out: set[Cell] = set()
    for j, col in enumerate(df.columns):
        vals = df[col].astype(str).tolist()
        nonempty = [(i, v) for i, v in enumerate(vals) if v.strip()]
        if len(nonempty) < min_rows:
            continue
        # Textual columns only: clustering is for name/label normalization, not
        # for merging distinct numbers/dates (token-sorting would collide them).
        with_letters = [(i, v) for i, v in nonempty if _HAS_LETTER_RE.search(v)]
        if len(with_letters) / len(nonempty) < min_letter_frac:
            continue
        # Skip near-unique columns (free text): no meaningful clusters.
        if len({v for _, v in with_letters}) / len(with_letters) > 0.95:
            continue
        flagged = _cluster_minority(with_letters, fingerprint)
        if use_ngram:
            flagged |= _cluster_minority(with_letters, ngram_fingerprint)
        for i in flagged:
            out.add((i, j))
    return out


# --------------------------------------------------------------------------- #
# 3) PyOD numeric-outlier ensemble
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _parse_numeric(v: str):
    s = str(v).strip().replace(",", "")
    return float(s) if _NUM_RE.match(s) else None


def detect_pyod(df: pd.DataFrame, *, min_rows: int = 20, num_frac: float = 0.8,
                contamination: float = 0.03, vote: int = 3) -> set[Cell]:
    """Per numeric column, run a PyOD ensemble (ECOD, COPOD, KNN, LOF) and flag
    cells that >= ``vote`` detectors agree are outliers. Conservative consensus +
    low contamination keep false positives down on near-clean numeric columns."""
    import numpy as np
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF

    out: set[Cell] = set()
    for j, col in enumerate(df.columns):
        vals = df[col].astype(str).tolist()
        parsed = [(i, _parse_numeric(v)) for i, v in enumerate(vals) if v.strip()]
        nums = [(i, x) for i, x in parsed if x is not None]
        nonempty = sum(1 for v in vals if v.strip())
        if len(nums) < min_rows or nonempty == 0:
            continue
        if len(nums) / nonempty < num_frac:
            continue  # not a numeric column
        rows = [i for i, _ in nums]
        X = np.array([[x] for _, x in nums], dtype=float)
        if float(np.std(X)) == 0.0:
            continue  # constant column
        votes = np.zeros(len(X))
        n_ok = 0
        for Model in (ECOD, COPOD, KNN, LOF):
            try:
                m = Model(contamination=contamination)
                m.fit(X)
                votes += m.labels_
                n_ok += 1
            except Exception:
                continue
        need = min(vote, n_ok) if n_ok else vote + 1
        for k, i in enumerate(rows):
            if votes[k] >= need:
                out.add((i, j))
    return out


DETECTORS = {
    "pattern": detect_pattern,
    "openrefine": detect_openrefine,
    "pyod": detect_pyod,
}
