"""Approximate functional-dependency violation detector.

For every column pair (A, B) with B's distinct count <= ``max_codomain``,
test whether each value of A maps to a single value of B in
>= ``min_support`` of the rows. Cells in the minority mapping for a
valid FD are flagged.

Concretely: for FD ``A -> B`` with rows grouped by A-value:

* If the modal B-value within an A-group covers >= ``rule_modal_share``
  of the A-group's rows AND the A-group has >= ``min_group_size`` rows,
  the modal is treated as the canonical mapping.
* Rows in that A-group whose B-value differs are flagged on column
  ``B`` with ``error_type = "fd_violation"``.

This is the canonical "city -> state" / "country -> country_code" /
"product_id -> price_tier" check. Precision is high when
``min_support`` is large (default 0.95).
"""

from __future__ import annotations

from typing import List

import pandas as pd

from .base import HeuristicConfig, HeuristicDetector, HeuristicMatch, register_detector


@register_detector
class FdViolationDetector(HeuristicDetector):
    name = "fd_violation"
    default_error_type = "fd_violation"

    def __init__(
        self,
        config: HeuristicConfig | None = None,
        *,
        # B's domain must be small enough to look enum-y. Free-text
        # columns blow this gate immediately.
        max_codomain: int = 50,
        # The FD A->B is "approximately valid" if at least this share
        # of rows agree with the per-group modal mapping.
        min_support: float = 0.9,
        # Per A-value group, modal B must cover at least this share
        # before we accuse minority rows of violating the FD.
        rule_modal_share: float = 0.8,
        min_group_size: int = 3,
    ) -> None:
        super().__init__(config)
        self.max_codomain = max_codomain
        self.min_support = min_support
        self.rule_modal_share = rule_modal_share
        self.min_group_size = min_group_size

    def detect(self, frame: pd.DataFrame, *, source: str) -> List[HeuristicMatch]:
        cols = list(frame.columns)
        out: List[HeuristicMatch] = []
        n_rows = len(frame)
        if n_rows < max(self.min_group_size, self.config.min_column_size):
            return out

        # Pre-stringify + strip every column once. Vectorised pandas
        # ops below are ~100x faster than the previous Python double
        # loop on a 50-column / 50k-row workbook.
        stripped: dict[str, pd.Series] = {}
        distinct_counts: dict[str, int] = {}
        for col in cols:
            ser = frame[col]
            s = ser.where(ser.notna(), "").astype(str).str.strip()
            # Caller resets index to 0..N-1 per the base contract; we
            # enforce here so reported ``row`` stays aligned.
            s.index = pd.RangeIndex(n_rows)
            stripped[col] = s
            distinct_counts[col] = int(s[s != ""].nunique())

        for b in cols:
            b_distinct = distinct_counts[b]
            if b_distinct < 2 or b_distinct > self.max_codomain:
                continue
            b_ser = stripped[b]
            for a in cols:
                if a == b:
                    continue
                a_distinct = distinct_counts[a]
                # Avoid checking trivially-unique LHS or constant LHS.
                if not (2 <= a_distinct <= n_rows - 1):
                    continue

                a_ser = stripped[a]
                mask = (a_ser != "") & (b_ser != "")
                if not mask.any():
                    continue
                sub = pd.DataFrame({"_a": a_ser[mask], "_b": b_ser[mask]})

                # Per-A group sizes; restrict to qualifying groups.
                group_sizes = sub.groupby("_a", sort=False).size()
                qualifying = group_sizes[group_sizes >= self.min_group_size]
                if qualifying.empty:
                    continue
                sub_q = sub[sub["_a"].isin(qualifying.index)]

                # Per-(A, B) counts; pick the modal B per A. Sort with
                # deterministic tie-break (lex on B) so equally-frequent
                # B values pick the same modal across runs.
                pair_counts = (
                    sub_q.groupby(["_a", "_b"], sort=False)
                    .size()
                    .rename("c")
                    .reset_index()
                    .sort_values(["_a", "c", "_b"], ascending=[True, False, True])
                )
                modals = pair_counts.drop_duplicates(subset=["_a"], keep="first")
                modals = modals.set_index("_a")
                modals["group_size"] = qualifying

                total = int(modals["group_size"].sum())
                modal_total = int(modals["c"].sum())
                if total == 0 or modal_total / total < self.min_support:
                    continue
                fd_support = round(modal_total / total, 3)

                # Per-group modal share; emit only on groups passing
                # rule_modal_share AND with at least one minority row.
                modals["share"] = modals["c"] / modals["group_size"]
                flagged = modals[
                    (modals["share"] >= self.rule_modal_share)
                    & (modals["c"] < modals["group_size"])
                ]
                if flagged.empty:
                    continue

                modal_map = flagged["_b"].to_dict()
                count_map = flagged["c"].to_dict()
                size_map = flagged["group_size"].to_dict()

                sub_f = sub_q[sub_q["_a"].isin(flagged.index)]
                mismatch = sub_f["_b"] != sub_f["_a"].map(modal_map)
                mismatches = sub_f.loc[mismatch]

                a_str, b_str = str(a), str(b)
                for row_idx, av, bv in zip(
                    mismatches.index.to_numpy(),
                    mismatches["_a"].to_numpy(),
                    mismatches["_b"].to_numpy(),
                ):
                    modal = modal_map[av]
                    modal_count = int(count_map[av])
                    group_size = int(size_map[av])
                    out.append(HeuristicMatch(
                        source=source,
                        row=int(row_idx),
                        column=b_str,
                        error_type=self.default_error_type,
                        detector=self.name,
                        observed=str(bv),
                        explanation=(
                            f"value disagrees with FD {a!r} -> {b!r}: "
                            f"row's {a}={av!r} usually maps to {modal!r} "
                            f"({modal_count}/{group_size} rows)"
                        ),
                        suggested_fix=modal,
                        severity="medium",
                        meta={
                            "lhs_column": a_str,
                            "lhs_value": str(av),
                            "fd_support": fd_support,
                            "group_size": group_size,
                            "modal_value": modal,
                            "modal_count": modal_count,
                        },
                    ))
        return out
