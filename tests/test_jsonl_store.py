"""Shared-contract tests for the JsonlStore base (follow-up item 2).

The six labelling stores were collapsed onto :class:`JsonlStore`, which
gives all of them the out-of-band-write self-healing that previously only
``LabelStore`` / ``ColumnClusterStore`` had (batch 7). These tests pin that
contract for the four stores that *gained* it here -- ``CellNotesStore``,
``ExclusionStore``, ``MultiTableLabelStore``, ``RelationReviewStore`` --
which also had no direct store-level tests before.

(``LabelStore``'s own out-of-band test lives in ``test_label_store.py``.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Set

import pytest

from src.labeling.cell_notes_store import CellNotesStore
from src.labeling.exclusions_store import ExclusionStore
from src.labeling.multi_table_label_store import MultiTableLabelStore
from src.labeling.relation_review_store import (
    RelationReviewStore,
    ReviewDecision,
    ReviewIn,
)
from src.labeling.schema import (
    CellNoteIn,
    ExclusionIn,
    ExclusionScope,
    MultiTableLabel,
    MultiTableLabelStatus,
)


class _Adapter:
    """Uniform write/read surface over one concrete store, so the
    out-of-band-write contract can be exercised identically across all."""

    fname: str

    def __init__(self, path: Path) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def write(self, i: int):  # append i-th distinct record, return its key
        raise NotImplementedError

    def keys(self) -> Set:  # keys currently visible to THIS instance
        raise NotImplementedError


class _CellNotes(_Adapter):
    fname = "cell_notes.jsonl"

    def __init__(self, path: Path) -> None:
        self.s = CellNotesStore(path)

    def write(self, i: int):
        self.s.upsert(
            CellNoteIn(source="a.csv", row=i, column="c0", note=f"n{i}"),
            annotator="t",
        )
        return ("a.csv", i, "c0")

    def keys(self) -> Set:
        return set(self.s.active_overrides())


class _Exclusions(_Adapter):
    fname = "exclusions.jsonl"

    def __init__(self, path: Path) -> None:
        self.s = ExclusionStore(path)

    def write(self, i: int):
        rec = self.s.add(
            ExclusionIn(scope=ExclusionScope.CSV, source=f"s{i}.csv"),
            annotator="t",
        )
        return rec.id

    def keys(self) -> Set:
        return {r.id for r in self.s.active()}


class _MultiTable(_Adapter):
    fname = "multi_table_labels.jsonl"

    def __init__(self, path: Path) -> None:
        self.s = MultiTableLabelStore(path)

    def write(self, i: int):
        vid = f"v{i}"
        self.s.append(
            MultiTableLabel(
                violation_id=vid,
                status=MultiTableLabelStatus.CONFIRMED_INCONSISTENCY,
                annotator="t",
            )
        )
        return vid

    def keys(self) -> Set:
        return set(self.s.latest_by_violation())


class _RelationReview(_Adapter):
    fname = "relation_reviews.jsonl"

    def __init__(self, path: Path) -> None:
        self.s = RelationReviewStore(path)

    def write(self, i: int):
        rid = f"r{i}"
        self.s.append(
            ReviewIn(relation_id=rid, decision=ReviewDecision.APPROVED, author="t")
        )
        return rid

    def keys(self) -> Set:
        return set(self.s.latest_by_relation())


_ADAPTERS = [
    pytest.param(_CellNotes, id="cell_notes"),
    pytest.param(_Exclusions, id="exclusions"),
    pytest.param(_MultiTable, id="multi_table"),
    pytest.param(_RelationReview, id="relation_review"),
]


@pytest.mark.parametrize("adapter_cls", _ADAPTERS)
def test_reloads_on_out_of_band_write(adapter_cls, tmp_path: Path) -> None:
    """An external writer (offline CLI / second worker) appending to the
    same file must become visible to a cached store on the next read."""
    path = tmp_path / adapter_cls.fname
    a1 = adapter_cls(path)
    k1 = a1.write(1)
    assert a1.keys() == {k1}

    # Out-of-band append via a *separate* instance on the same file.
    a2 = adapter_cls(path)
    k2 = a2.write(2)

    # a1 never wrote k2, but must pick it up on its next read.
    assert a1.keys() == {k1, k2}


@pytest.mark.parametrize("adapter_cls", _ADAPTERS)
def test_own_writes_not_lost(adapter_cls, tmp_path: Path) -> None:
    """Our own appends refresh the (mtime,size) signature, so a self-write
    never spuriously discards in-memory state on the next read."""
    path = tmp_path / adapter_cls.fname
    a = adapter_cls(path)
    k1 = a.write(1)
    k2 = a.write(2)
    assert a.keys() == {k1, k2}
