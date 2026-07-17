"""Export-overlay tests for per-row FD verdict overrides.

An fd_violation conflict is ONE cluster anchored at a row, fanning its
verdict out across every conflicting ``affected_row`` (peer rows usually
have no cluster of their own, so the fan-out is the only thing that puts
them in the benchmark). A per-row override lets the operator give one
row a different verdict WITHOUT dropping any row from the benchmark.
"""

from __future__ import annotations

from pathlib import Path

from src.labeling.export import materialize_label_benchmark
from src.labeling.schema import Label, LabelStatus, LabelTarget, RowVerdict
from src.labeling.store import LabelStore


def _fd_cluster() -> dict:
    # FD "zip -> state": rows 3 & 58 share zip=30301 but disagree on state.
    return {
        "cluster_id": "fd1",
        "source": "a.csv",
        "row": 3,
        "columns": ["zip", "state"],
        "row_values": {"zip": "30301", "state": "GA"},
        "error_types": ["fd_violation"],
        "detections": [
            {"issue_id": "rel__r1#a.csv#r00003", "affected_rows": [3, 58]},
        ],
    }


def _cluster_label() -> Label:
    return Label(
        target=LabelTarget.CLUSTER, target_id="fd1",
        source="a.csv", row=3, column=None,
        columns=["zip", "state"], status=LabelStatus.TP, annotator="t",
    )


def _ov(row: int, col: str, status: LabelStatus, **kw) -> RowVerdict:
    return RowVerdict(source="a.csv", row=row, column=col, status=status, annotator="t", **kw)


def _materialize(store, overrides):
    return {
        (c.row, c.column): c
        for c in materialize_label_benchmark(
            [_fd_cluster()], store, row_verdict_overrides=overrides,
        )
    }


def test_peer_override_keeps_all_rows_and_flips_only_its_cell(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(_cluster_label())
    by_key = _materialize(store, {("a.csv", 58, "state"): _ov(58, "state", LabelStatus.FP)})
    # Coverage preserved: all 4 cells still emitted (peer row 58 NOT dropped).
    assert set(by_key) == {(3, "zip"), (3, "state"), (58, "zip"), (58, "state")}
    # Anchor row keeps the cluster verdict.
    assert by_key[(3, "zip")].label == "error"
    assert by_key[(3, "state")].label == "error"
    # The override flips ONLY (58, state) -> clean (not "ambiguous").
    assert by_key[(58, "state")].label == "clean"
    assert by_key[(58, "state")].corrected_value is None
    # Peer's other column has no override -> stays the cluster default.
    assert by_key[(58, "zip")].label == "error"


def test_peer_override_carries_its_own_corrected_value(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(_cluster_label())
    by_key = _materialize(
        store, {("a.csv", 58, "state"): _ov(58, "state", LabelStatus.TP, corrected_value="CA")},
    )
    assert by_key[(58, "state")].label == "error"
    assert by_key[(58, "state")].corrected_value == "CA"
    # The anchor row is not contaminated by the peer's correction.
    assert by_key[(3, "state")].corrected_value in (None, "")


def test_no_overrides_is_identical_to_pre_feature(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(_cluster_label())
    base = [c.model_dump() for c in materialize_label_benchmark([_fd_cluster()], store)]
    empty = [
        c.model_dump()
        for c in materialize_label_benchmark([_fd_cluster()], store, row_verdict_overrides={})
    ]
    assert base == empty


def test_skip_override_leaves_cluster_default(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(_cluster_label())
    by_key = _materialize(store, {("a.csv", 58, "state"): _ov(58, "state", LabelStatus.SKIP)})
    # SKIP carries no benchmark verdict -> peer keeps the fanned-out default.
    assert by_key[(58, "state")].label == "error"
