"""Tests for :class:`src.labeling.row_verdict_store.RowVerdictStore`.

Per-row verdict overrides let an operator label one row of an
fd_violation conflict group independently of the cluster's fanned-out
verdict. Latest-wins per (source, row, column); ``unlabeled`` is a
tombstone that reverts the cell to the cluster default.
"""

from __future__ import annotations

from pathlib import Path

from src.labeling.row_verdict_store import RowVerdictStore
from src.labeling.schema import LabelStatus, RowVerdictIn


def _in(row: int, status: LabelStatus, **kw) -> RowVerdictIn:
    return RowVerdictIn(source="a.csv", row=row, column="state", status=status, **kw)


def test_upsert_latest_wins(tmp_path: Path) -> None:
    s = RowVerdictStore(tmp_path / "row_verdicts.jsonl")
    s.upsert(_in(58, LabelStatus.TP), annotator="t")
    s.upsert(_in(58, LabelStatus.FP), annotator="t")
    ov = s.active_overrides()
    assert ov[("a.csv", 58, "state")].status == LabelStatus.FP


def test_active_overrides_drops_unlabeled_tombstone(tmp_path: Path) -> None:
    s = RowVerdictStore(tmp_path / "row_verdicts.jsonl")
    s.upsert(_in(58, LabelStatus.FP), annotator="t")
    assert ("a.csv", 58, "state") in s.active_overrides()
    # Tombstone reverts the cell to the cluster default.
    s.upsert(_in(58, LabelStatus.UNLABELED), annotator="t")
    assert ("a.csv", 58, "state") not in s.active_overrides()
    # The tombstone is still on disk for the audit trail.
    assert s.latest_for("a.csv", 58, "state").status == LabelStatus.UNLABELED


def test_key_is_source_row_column(tmp_path: Path) -> None:
    s = RowVerdictStore(tmp_path / "row_verdicts.jsonl")
    s.upsert(RowVerdictIn(source="a.csv", row=3, column="state", status=LabelStatus.FP), annotator="t")
    s.upsert(RowVerdictIn(source="a.csv", row=3, column="zip", status=LabelStatus.TP), annotator="t")
    ov = s.active_overrides()
    assert ov[("a.csv", 3, "state")].status == LabelStatus.FP
    assert ov[("a.csv", 3, "zip")].status == LabelStatus.TP


def test_persists_across_reload(tmp_path: Path) -> None:
    p = tmp_path / "row_verdicts.jsonl"
    s = RowVerdictStore(p)
    s.upsert(_in(58, LabelStatus.TP, corrected_value="CA", note="fixed"), annotator="t")
    s2 = RowVerdictStore(p)
    rec = s2.latest_for("a.csv", 58, "state")
    assert rec.status == LabelStatus.TP
    assert rec.corrected_value == "CA"
    assert rec.note == "fixed"
