"""Tests for the deterministic group-level plausibility guard
(``src.core.multi_table.group_guard``).

The guard collapses a confirmed group's row-pairing findings into ONE
note when the group is implausible: a positional/ordinal key (pairs
unrelated rows) or a union that disagrees on a high fraction of its keys
(periods/snapshots, not replicas). Schema-level kinds and small/plausible
groups are left untouched.
"""

from __future__ import annotations

import pandas as pd

from src.core.multi_table.group_guard import (
    _looks_ordinal,
    evaluate_group,
    filter_scope_leak_dicts,
    guard_finding_dicts,
    is_non_identifying_key,
    pairable_key_count,
)
from src.core.multi_table.resolver import ColumnMappingEntry, ResolvedGroup


def _entry(s, c):
    return ColumnMappingEntry(sheet=s, column=c)


def _union(sheets=("a.csv", "b.csv"), pk=("K",), keycol="K"):
    return ResolvedGroup(
        workbook="wb", group_id="u1", kind="union", confirmed=True,
        rationale="", sheets=tuple(sheets),
        column_mapping={
            keycol: tuple(_entry(s, keycol) for s in sheets),
            "v": tuple(_entry(s, "v") for s in sheets),
        },
        primary_key=tuple(pk),
    )


# ---------------------------------------------------------------------------
# _looks_ordinal
# ---------------------------------------------------------------------------


class TestLooksOrdinal:
    def test_consecutive_ints_true(self) -> None:
        assert _looks_ordinal([str(i) for i in range(1, 30)])

    def test_consecutive_with_gaps_true(self) -> None:
        # row-dropping leaves gaps but it's still dense + starts near 1.
        assert _looks_ordinal(["1", "3", "4", "5", "6", "7", "9", "10"])

    def test_sparse_ids_false(self) -> None:
        assert not _looks_ordinal(["1001", "5500", "98231", "42", "777777"])

    def test_non_numeric_false(self) -> None:
        assert not _looks_ordinal(["alice", "bob", "carol", "dave", "eve"])

    def test_decimals_false(self) -> None:
        assert not _looks_ordinal(["1.5", "2.5", "3.5", "4.5", "5.5"])


# ---------------------------------------------------------------------------
# is_non_identifying_key
# ---------------------------------------------------------------------------


class TestNonIdentifyingKey:
    def test_ordinal_named_key(self) -> None:
        g = _union(pk=("No.",), keycol="No.")
        frames = {
            "a.csv": pd.DataFrame({"No.": ["1", "2", "3"], "v": [1, 2, 3]}),
            "b.csv": pd.DataFrame({"No.": ["1", "2", "3"], "v": [4, 5, 6]}),
        }
        bad, reason = is_non_identifying_key(g, frames)
        assert bad and "positional/ordinal" in reason

    def test_ordinal_valued_key_even_if_named_id(self) -> None:
        # Column named 'ID' but values are 1..N -> still positional.
        g = _union(pk=("ID",), keycol="ID")
        frames = {
            "a.csv": pd.DataFrame({"ID": [str(i) for i in range(1, 21)],
                                   "v": list(range(20))}),
            "b.csv": pd.DataFrame({"ID": [str(i) for i in range(1, 21)],
                                   "v": list(range(20))}),
        }
        bad, _ = is_non_identifying_key(g, frames)
        assert bad

    def test_real_key_is_identifying(self) -> None:
        g = _union(pk=("K",), keycol="K")
        frames = {
            "a.csv": pd.DataFrame({"K": ["acct-9001", "acct-5550"], "v": [1, 2]}),
            "b.csv": pd.DataFrame({"K": ["acct-9001", "acct-5550"], "v": [1, 9]}),
        }
        bad, _ = is_non_identifying_key(g, frames)
        assert not bad


# ---------------------------------------------------------------------------
# evaluate_group
# ---------------------------------------------------------------------------


def _keyed_frames(keycol="K", n=20):
    keys = [f"k{i}" for i in range(n)]
    return {
        "a.csv": pd.DataFrame({keycol: keys, "v": list(range(n))}),
        "b.csv": pd.DataFrame({keycol: keys, "v": list(range(n))}),
    }


class TestEvaluateGroup:
    def test_ordinal_key_collapses(self) -> None:
        g = _union(pk=("No.",), keycol="No.")
        frames = {
            "a.csv": pd.DataFrame({"No.": [str(i) for i in range(1, 21)],
                                   "v": list(range(20))}),
            "b.csv": pd.DataFrame({"No.": [str(i) for i in range(1, 21)],
                                   "v": list(range(20))}),
        }
        d = evaluate_group(g, frames, 15, density_threshold=0.4, min_findings=10)
        assert d.collapse and "ordinal" in d.reason

    def test_high_density_collapses(self) -> None:
        g = _union()
        frames = _keyed_frames(n=20)
        d = evaluate_group(g, frames, 15, density_threshold=0.4, min_findings=10)
        assert d.collapse and "disagree" in d.reason

    def test_low_density_kept(self) -> None:
        g = _union()
        frames = _keyed_frames(n=100)
        # 12 findings / 100 keys = 0.12 < 0.4.
        d = evaluate_group(g, frames, 12, density_threshold=0.4, min_findings=10)
        assert not d.collapse

    def test_below_min_findings_kept(self) -> None:
        g = _union()
        frames = _keyed_frames(n=20)
        # density would be 8/20=0.4 but below min_findings=10.
        d = evaluate_group(g, frames, 8, density_threshold=0.4, min_findings=10)
        assert not d.collapse

    def test_pairable_key_count(self) -> None:
        g = _union()
        frames = _keyed_frames(n=20)
        assert pairable_key_count(g, frames) == 20


# ---------------------------------------------------------------------------
# guard_finding_dicts (dict-level collapse)
# ---------------------------------------------------------------------------


def _viol(vid, kind, sheet="a.csv", row=0, gid="u1"):
    return {
        "violation_id": vid, "kind": kind, "workbook": "wb", "group_id": gid,
        "group_kind": "union", "sheet": sheet, "row": row,
        "columns": ["v"], "values": {}, "linked": [],
    }


class TestGuardFindingDicts:
    def _group_dict(self, gid="u1", pk=("K",), keycol="K", sheets=("a.csv", "b.csv")):
        return {
            "group_id": gid, "kind": "union", "confirmed": True,
            "rationale": "", "sheets": list(sheets),
            "column_mapping": {
                keycol: [{"sheet": s, "column": keycol} for s in sheets],
                "v": [{"sheet": s, "column": "v"} for s in sheets],
            },
            "primary_key": list(pk),
        }

    def test_flood_group_collapses_keeps_structural(self) -> None:
        # 15 attribute_conflicts (flood) + 1 header_synonym (structural).
        viols = [_viol(f"c{i}", "attribute_conflict") for i in range(15)]
        viols.append(_viol("h1", "header_synonym", row=-1))
        vers = [{"violation_id": v["violation_id"], "verdict": "uncertain"}
                for v in viols]
        frames = _keyed_frames(n=20)
        gl = {("wb", "u1"): self._group_dict()}
        nv, nver, stats = guard_finding_dicts(
            violation_dicts=viols, verification_dicts=vers,
            group_lookup=gl, load_frames=lambda wb, sh: frames,
            density_threshold=0.4, min_findings=10,
        )
        kinds = {r["kind"] for r in nv}
        assert "attribute_conflict" not in kinds   # flood dropped
        assert "header_synonym" in kinds            # structural kept
        # A collapse is NOT an error: no labelable finding is emitted.
        assert "suspect_grouping" not in kinds
        assert stats["groups_collapsed"] == 1
        assert stats["findings_suppressed"] == 15
        assert stats["notes_added"] == 0
        # the rejected group is recorded in stats, not the queue.
        assert stats["collapsed_groups"][0]["group_id"] == "u1"
        assert stats["collapsed_groups"][0]["suppressed"] == 15
        # no suspect_grouping verification either; the flood's vers are gone.
        assert all(v.get("kind") != "suspect_grouping" for v in nver)
        assert any(v.get("violation_id") == "h1" for v in nver)

    def test_small_group_untouched(self) -> None:
        viols = [_viol("c1", "attribute_conflict"), _viol("c2", "duplicate_row")]
        frames = _keyed_frames(n=20)
        gl = {("wb", "u1"): self._group_dict()}
        nv, _nver, stats = guard_finding_dicts(
            violation_dicts=viols, verification_dicts=[],
            group_lookup=gl, load_frames=lambda wb, sh: frames,
            density_threshold=0.4, min_findings=10,
        )
        assert stats["groups_collapsed"] == 0
        assert len(nv) == 2  # nothing collapsed

    def test_ordinal_key_group_collapses(self) -> None:
        viols = [_viol(f"c{i}", "attribute_conflict") for i in range(15)]
        frames = {
            "a.csv": pd.DataFrame({"No.": [str(i) for i in range(1, 21)],
                                   "v": list(range(20))}),
            "b.csv": pd.DataFrame({"No.": [str(i) for i in range(1, 21)],
                                   "v": list(range(20))}),
        }
        gl = {("wb", "u1"): self._group_dict(pk=("No.",), keycol="No.")}
        _nv, _nver, stats = guard_finding_dicts(
            violation_dicts=viols, verification_dicts=[],
            group_lookup=gl, load_frames=lambda wb, sh: frames,
            density_threshold=0.4, min_findings=10,
        )
        assert stats["groups_collapsed"] == 1
        assert "ordinal" in stats["collapsed_groups"][0]["reason"]


def _join_group(sheets=("a.csv", "b.csv"), pk=("K",), keycol="K"):
    return ResolvedGroup(
        workbook="wb", group_id="j1", kind="join", confirmed=True,
        rationale="", sheets=tuple(sheets),
        column_mapping={
            keycol: tuple(_entry(s, keycol) for s in sheets),
            "v": tuple(_entry(s, "v") for s in sheets),
        },
        primary_key=tuple(pk),
    )


class TestJoinReason:
    def test_join_density_collapse_has_join_reason(self) -> None:
        g = _join_group()
        frames = _keyed_frames(n=20)
        d = evaluate_group(g, frames, 15, density_threshold=0.4, min_findings=10)
        assert d.collapse
        assert "join group" in d.reason
        # union language must NOT leak into a join collapse.
        assert "periods" not in d.reason

    def test_union_density_collapse_keeps_snapshot_reason(self) -> None:
        g = _union()
        frames = _keyed_frames(n=20)
        d = evaluate_group(g, frames, 15, density_threshold=0.4, min_findings=10)
        assert d.collapse
        assert "join group" not in d.reason
        assert "periods" in d.reason


# ---------------------------------------------------------------------------
# filter_scope_leak_dicts (scope filter)
# ---------------------------------------------------------------------------


def _pair(vid, kind, sheet="a.csv", other="b.csv", row=0, other_row=5):
    """A row-pairing finding spanning ``sheet`` and ``other``."""
    return {
        "violation_id": vid, "kind": kind, "workbook": "wb", "group_id": "u1",
        "sheet": sheet, "row": row, "columns": ["v"], "values": {},
        "linked": [{"sheet": other, "row": other_row}],
    }


class TestFilterScopeLeak:
    def test_same_sheet_pair_dropped(self) -> None:
        viols = [_pair("c1", "duplicate_row", other="a.csv")]  # both a.csv
        nv, _nver, stats = filter_scope_leak_dicts(
            violation_dicts=viols, verification_dicts=[],
        )
        assert nv == []
        assert stats["dropped_same_sheet"] == 1
        assert stats["findings_after"] == 0

    def test_cross_sheet_pair_kept(self) -> None:
        viols = [_pair("c1", "duplicate_row", other="b.csv")]  # a vs b
        nv, _nver, stats = filter_scope_leak_dicts(
            violation_dicts=viols, verification_dicts=[],
        )
        assert len(nv) == 1
        assert stats["dropped_same_sheet"] == 0

    def test_fk_orphan_dropped(self) -> None:
        viols = [{
            "violation_id": "o1", "kind": "fk_orphan", "workbook": "wb",
            "group_id": "j1", "sheet": "a.csv", "row": 3, "columns": ["FK"],
            "values": {}, "linked": [],
        }]
        nv, _nver, stats = filter_scope_leak_dicts(
            violation_dicts=viols, verification_dicts=[],
        )
        assert nv == []
        assert stats["dropped_fk_orphan"] == 1

    def test_fk_orphan_kept_when_disabled(self) -> None:
        viols = [{
            "violation_id": "o1", "kind": "fk_orphan", "workbook": "wb",
            "group_id": "j1", "sheet": "a.csv", "row": 3, "columns": ["FK"],
            "values": {}, "linked": [],
        }]
        nv, _nver, stats = filter_scope_leak_dicts(
            violation_dicts=viols, verification_dicts=[], drop_fk_orphan=False,
        )
        assert len(nv) == 1
        assert stats["dropped_fk_orphan"] == 0

    def test_same_sheet_kept_when_disabled(self) -> None:
        viols = [_pair("c1", "attribute_conflict", other="a.csv")]
        nv, _nver, _stats = filter_scope_leak_dicts(
            violation_dicts=viols, verification_dicts=[],
            drop_same_sheet_pairs=False,
        )
        assert len(nv) == 1

    def test_structural_kinds_untouched(self) -> None:
        viols = [
            {"violation_id": "h1", "kind": "header_synonym", "sheet": "a.csv",
             "row": -1, "linked": [], "columns": ["x"], "values": {}},
            {"violation_id": "d1", "kind": "dtype_drift", "sheet": "a.csv",
             "row": -1, "linked": [], "columns": ["x"], "values": {}},
        ]
        nv, _nver, stats = filter_scope_leak_dicts(
            violation_dicts=viols, verification_dicts=[],
        )
        assert len(nv) == 2
        assert stats["dropped_same_sheet"] == 0
        assert stats["dropped_fk_orphan"] == 0

    def test_verifications_dropped_with_findings(self) -> None:
        viols = [
            _pair("c1", "duplicate_row", other="a.csv"),   # dropped
            _pair("c2", "duplicate_row", other="b.csv"),   # kept
        ]
        vers = [
            {"violation_id": "c1", "verdict": "confirmed_inconsistency"},
            {"violation_id": "c2", "verdict": "confirmed_inconsistency"},
        ]
        _nv, nver, _stats = filter_scope_leak_dicts(
            violation_dicts=viols, verification_dicts=vers,
        )
        kept_vids = {v["violation_id"] for v in nver}
        assert kept_vids == {"c2"}
