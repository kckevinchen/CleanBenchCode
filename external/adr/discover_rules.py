#!/usr/bin/env python3
"""Mine approximate FD/DC rules from a dirty CSV only.

The ADR'24 constraint repair methods expect hand-authored rule files.  Our
benchmark has none, so this miner emits conservative, unsupervised approximate
functional dependencies from the dirty table itself in both ADR formats:

* Horizon/RelativeTrust/SCARE/Unified FD: ``LHS ⇒ RHS``
* HoloClean/Holistic/BigDansing/Daisy DC: ``t1&t2&EQ(t1.LHS,t2.LHS)&IQ(t1.RHS,t2.RHS)``

No clean data, labels, or benchmark ground truth are read.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ApproxFD:
    lhs: tuple[str, ...]
    rhs: str
    confidence: float
    violations: int
    rows: int
    duplicate_mass: int
    lhs_cardinality: int

    @property
    def lhs_text(self) -> str:
        return ",".join(self.lhs)


def _read_dirty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False).fillna("")


def _fd_score(df: pd.DataFrame, lhs: tuple[str, ...], rhs: str) -> ApproxFD | None:
    if rhs in lhs or df.empty:
        return None
    cols = list(lhs) + [rhs]
    sub = df[cols].astype(str).fillna("")
    n = len(sub)
    grouped = sub.groupby(list(lhs), dropna=False, sort=False)[rhs]
    lhs_cardinality = int(grouped.ngroups)
    duplicate_mass = int(grouped.size().loc[lambda s: s > 1].sum()) if lhs_cardinality else 0
    if lhs_cardinality == 0:
        return None
    kept = int(grouped.value_counts(sort=False).groupby(level=list(range(len(lhs)))).max().sum())
    violations = n - kept
    confidence = kept / n if n else 0.0
    return ApproxFD(lhs, rhs, confidence, violations, n, duplicate_mass, lhs_cardinality)


def discover_fds(
    df: pd.DataFrame,
    *,
    min_confidence: float = 0.95,
    max_violations: int | None = None,
    min_duplicate_mass: float = 0.10,
    max_lhs_uniqueness: float = 0.98,
    max_lhs_size: int = 2,
    max_rules: int = 50,
) -> list[ApproxFD]:
    """Return high-confidence approximate FDs, preferring single-column LHSs.

    Candidate keys that are almost unique are filtered out because they produce
    vacuous FDs that over-repair arbitrary values.  The default still admits
    useful identifiers repeated across rows (e.g. hospital name -> zip code).
    """

    n = len(df)
    if n == 0 or df.shape[1] < 2:
        return []
    if max_violations is None:
        max_violations = max(0, int(round((1.0 - min_confidence) * n)))

    cols = [str(c) for c in df.columns]

    # Columns that are constant/nearly constant produce tautological rules such
    # as X -> empty_address_line.  They are not useful for repair and swamp the
    # authored hospital FDs, so exclude them as RHSs and as antecedents.
    usable_cols: list[str] = []
    for col in cols:
        vc = df[col].astype(str).fillna("").value_counts(dropna=False)
        nunique = int(vc.size)
        dominant = int(vc.iloc[0]) / n if nunique else 1.0
        if nunique >= 2 and dominant < 0.98:
            usable_cols.append(col)

    candidates: list[ApproxFD] = []
    for lhs_size in range(1, max_lhs_size + 1):
        for lhs in itertools.combinations(usable_cols, lhs_size):
            for rhs in usable_cols:
                fd = _fd_score(df, lhs, rhs)
                if fd is None:
                    continue
                if fd.confidence < min_confidence or fd.violations > max_violations:
                    continue
                if fd.duplicate_mass < max(2, int(min_duplicate_mass * n)):
                    continue
                if fd.lhs_cardinality > max(1, int(max_lhs_uniqueness * n)):
                    continue
                candidates.append(fd)

    candidates.sort(
        key=lambda f: (
            len(f.lhs),
            -f.confidence,
            f.violations,
            -f.duplicate_mass,
            f.lhs_text.lower(),
            f.rhs.lower(),
        )
    )

    # De-duplicate RHSs that have both unary and composite explanations; keep the
    # simpler/stronger antecedent first, but allow multiple rules for a RHS when
    # they have distinct single-column antecedents.
    selected: list[ApproxFD] = []
    seen: set[tuple[tuple[str, ...], str]] = set()
    for fd in candidates:
        key = (fd.lhs, fd.rhs)
        if key in seen:
            continue
        selected.append(fd)
        seen.add(key)
        if len(selected) >= max_rules:
            break
    return selected


def write_horizon_fd(fds: Iterable[ApproxFD], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fd in fds:
            if len(fd.lhs) == 1:
                f.write(f"{fd.lhs[0]} ⇒ {fd.rhs}\n")


def write_holoclean_dc(fds: Iterable[ApproxFD], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fd in fds:
            predicates = [f"EQ(t1.{col},t2.{col})" for col in fd.lhs]
            predicates.append(f"IQ(t1.{fd.rhs},t2.{fd.rhs})")
            f.write("t1&t2&" + "&".join(predicates) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirty_csv", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--horizon-fd", type=Path, default=None)
    ap.add_argument("--holoclean-dc", type=Path, default=None)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--min-confidence", type=float, default=0.95)
    ap.add_argument("--max-violations", type=int, default=None)
    ap.add_argument("--min-duplicate-mass", type=float, default=0.10)
    ap.add_argument("--max-lhs-uniqueness", type=float, default=0.98)
    ap.add_argument("--max-lhs-size", type=int, default=2)
    ap.add_argument("--max-rules", type=int, default=50)
    args = ap.parse_args()

    out_dir = args.out_dir or args.dirty_csv.parent
    horizon_fd = args.horizon_fd or out_dir / "dc_rules-validate-fd-horizon.txt"
    holoclean_dc = args.holoclean_dc or out_dir / "dc_rules_holoclean.txt"
    metadata = args.metadata or out_dir / "discovered_rules.json"

    df = _read_dirty(args.dirty_csv)
    fds = discover_fds(
        df,
        min_confidence=args.min_confidence,
        max_violations=args.max_violations,
        min_duplicate_mass=args.min_duplicate_mass,
        max_lhs_uniqueness=args.max_lhs_uniqueness,
        max_lhs_size=args.max_lhs_size,
        max_rules=args.max_rules,
    )
    write_horizon_fd(fds, horizon_fd)
    write_holoclean_dc(fds, holoclean_dc)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps([asdict(fd) for fd in fds], indent=2), encoding="utf-8")

    print(f"rows={len(df)} cols={df.shape[1]} rules={len(fds)}")
    for fd in fds[:20]:
        print(
            f"{fd.lhs_text} => {fd.rhs} "
            f"conf={fd.confidence:.4f} violations={fd.violations} duplicate_mass={fd.duplicate_mass}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
