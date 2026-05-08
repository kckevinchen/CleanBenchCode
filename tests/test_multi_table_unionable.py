"""Tests for ``find_unionable_groups`` (the new cross-sheet
unionable-group detector in ``src.core.multi_table_priors``).

The detector clusters sibling sheets with high normalized-header
Jaccard. The two real datasets that motivate this code are:

    * ``curated_bing21_min_filtered/52409d4c.../*.csv`` - 3 sheets
      with near-identical schemas; one column has surface drift
      (``Reading Quiz 1 %`` / ``(%)`` / ``(/10%)2``).
    * ``curated_bing21_min_filtered/e5739481.../*.csv`` - 2 sheets
      with the same 5 columns in different order.

Both should land in a unionable group with high avg_jaccard.
"""

from __future__ import annotations

import pandas as pd

from src.core.multi_table_priors import (
    UnionableGroupCandidate,
    _normalize_header,
    find_unionable_groups,
)


class TestNormalizeHeader:
    def test_surface_drift_collapses(self) -> None:
        # The motivating real-world drift: '%' and '(%)' should be
        # the same canonical header after strip-punct + collapse-ws.
        assert _normalize_header("Reading Quiz 1 %") == _normalize_header(
            "Reading Quiz 1 (%)"
        )

    def test_trailing_token_distinguishes(self) -> None:
        # The trailing '2' in '(/10%)2' is real, not punctuation -
        # it must NOT collapse with '(/10%)'.
        a = _normalize_header("Reading Quiz 1 (/10%)")
        b = _normalize_header("Reading Quiz 1 (/10%)2")
        assert a != b

    def test_case_and_whitespace(self) -> None:
        assert _normalize_header("  Town  Code  ") == _normalize_header(
            "town code"
        )

    def test_none_and_empty(self) -> None:
        assert _normalize_header(None) == ""
        assert _normalize_header("") == ""


class TestFindUnionableGroups:
    def test_three_sheets_same_subject_cluster(self) -> None:
        # The 52409d4c.xlsx case: 3 sheets with student grades.
        df_a = pd.DataFrame({
            "Name": [1, 2, 3],
            "Reading Quiz 1 (/30)": [26, 22, 27],
            "Reading Quiz 1 %": [86.7, 73.3, 90.0],
            "Reading Quiz 1 (/10%)": [8.6, 7.3, 9.0],
        })
        df_b = pd.DataFrame({
            "Name": [4, 5],
            "Reading Quiz 1 (/30)": [27, 27],
            "Reading Quiz 1 (%)": [90.0, 90.0],     # surface drift
            "Reading Quiz 1 (/10%)": [9.0, 9.0],
        })
        df_c = pd.DataFrame({
            "Name": [6, 7],
            "Reading Quiz 1 (/30)": [20, 27],
            "Reading Quiz 1 (/10%)2": [6.6, 9.0],   # trailing-2 typo
            "Reading Quiz 1 (/10%)": [6.6, 9.0],
        })
        frames = {"a.csv": df_a, "b.csv": df_b, "c.csv": df_c}
        groups = find_unionable_groups(
            sheets=list(frames), frames=frames,
        )
        assert len(groups) == 1
        g = groups[0]
        assert isinstance(g, UnionableGroupCandidate)
        assert set(g.sheets) == {"a.csv", "b.csv", "c.csv"}
        assert g.n_sheets == 3
        # Name + Reading Quiz 1 (/30) + Reading Quiz 1 (/10%) are the
        # three intersection canonicals. The '%' header collapses
        # in a/b but C has '(/10%)2' instead so it counts as a
        # separate canonical that is NOT in the intersection.
        assert g.n_shared_headers == 3
        # Union: name, reading quiz 1 30, reading quiz 1, reading
        # quiz 1 10, reading quiz 1 10 2 -> 5 canonicals.
        assert g.n_canonical_headers == 5
        assert g.avg_jaccard > 0.5

    def test_two_sheets_column_reorder_only(self) -> None:
        # The e5739481.xlsx case: same 5 columns, reordered.
        df_a = pd.DataFrame({
            "SAU OrgID": [77, 94],
            "SAU Name": ["Bremen", "Brunswick"],
            "Town Code": ["052", "063"],
            "Town Name": ["Bremen", "Brunswick"],
            "School Year": [2022, 2022],
        })
        df_b = pd.DataFrame({
            "Town Code": ["015", "021"],
            "Town Name": ["Arrowsic", "Augusta"],
            "SAU OrgID": [1438, 28],
            "SAU Name": ["RSU 01", "Augusta"],
            "School Year": [2022, 2022],
        })
        frames = {"a.csv": df_a, "b.csv": df_b}
        groups = find_unionable_groups(sheets=list(frames), frames=frames)
        assert len(groups) == 1
        g = groups[0]
        # Identical column sets -> Jaccard = 1.0 exactly.
        assert g.avg_jaccard == 1.0
        assert g.n_shared_headers == 5
        assert g.n_canonical_headers == 5

    def test_unrelated_sheets_dropped(self) -> None:
        # Two related sheets + one unrelated sheet: the unrelated
        # sheet must not cluster with the others.
        df_a = pd.DataFrame({"Name": [1], "Score": [10]})
        df_b = pd.DataFrame({"Name": [2], "Score": [20]})
        df_c = pd.DataFrame({"Country": ["US"], "Currency": ["USD"]})
        frames = {"a.csv": df_a, "b.csv": df_b, "c.csv": df_c}
        groups = find_unionable_groups(sheets=list(frames), frames=frames)
        assert len(groups) == 1
        assert set(groups[0].sheets) == {"a.csv", "b.csv"}
        assert "c.csv" not in groups[0].sheets

    def test_below_threshold_no_group(self) -> None:
        # With only 1 of 5 union columns shared the Jaccard is 1/5 = 0.2,
        # below the 0.25 default - no group is emitted.
        df_a = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df_b = pd.DataFrame({"a": [1], "x": [9], "y": [8]})
        frames = {"a.csv": df_a, "b.csv": df_b}
        groups = find_unionable_groups(sheets=list(frames), frames=frames)
        assert groups == []

    def test_single_sheet_returns_empty(self) -> None:
        df = pd.DataFrame({"Name": [1]})
        frames = {"a.csv": df}
        assert find_unionable_groups(sheets=["a.csv"], frames=frames) == []

    def test_partial_unionability_clusters(self) -> None:
        # Two sheets, mostly-shared headers + extras. Jaccard ~ 0.6
        # -> still clusters; the LLM resolver would emit a partial
        # column_mapping at confirmation time.
        df_a = pd.DataFrame({
            "ID": [1, 2], "Name": ["a", "b"], "Score": [10, 20],
            "ExtraInA": ["x", "y"],
        })
        df_b = pd.DataFrame({
            "ID": [3, 4], "Name": ["c", "d"], "Score": [30, 40],
            "ExtraInB": ["m", "n"],
        })
        frames = {"a.csv": df_a, "b.csv": df_b}
        groups = find_unionable_groups(sheets=list(frames), frames=frames)
        assert len(groups) == 1
        g = groups[0]
        # 3 shared, 5 canonical -> Jaccard 3/5 = 0.6.
        assert g.n_shared_headers == 3
        assert g.n_canonical_headers == 5
        assert abs(g.avg_jaccard - 0.6) < 1e-9

    def test_to_dict_round_trip(self) -> None:
        df_a = pd.DataFrame({"A": [1], "B": [2]})
        df_b = pd.DataFrame({"A": [3], "B": [4]})
        groups = find_unionable_groups(
            sheets=["a.csv", "b.csv"],
            frames={"a.csv": df_a, "b.csv": df_b},
        )
        d = groups[0].to_dict()
        assert d["group_id"] == "u1"
        assert d["sheets"] == ["a.csv", "b.csv"]
        assert d["n_shared_headers"] == 2
        assert d["n_canonical_headers"] == 2
        assert d["avg_jaccard"] == 1.0
        assert {entry["sheet"] for entry in d["sheet_headers"]} == {
            "a.csv", "b.csv",
        }
