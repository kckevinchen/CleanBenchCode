"""Tests for ``src.labeling.store.LabelStore``: append durability,
malformed-line recovery, and concurrent-append safety.

Pinned by Pass 6c MED gap.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List

import pytest

from src.labeling.schema import Label, LabelStatus, LabelTarget, cell_target_id
from src.labeling.store import LabelStore


def _label(target_id: str, *, status: LabelStatus = LabelStatus.TP,
            annotator: str = "tester", note: str = "") -> Label:
    return Label(
        target=LabelTarget.CLUSTER,
        target_id=target_id,
        source="a.csv",
        row=0,
        column="c0",
        status=status,
        annotator=annotator,
        note=note,
    )


# ---------------------------------------------------------------------------
# Basic append + reload + history
# ---------------------------------------------------------------------------


class TestAppendAndReload:
    def test_append_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        s1 = LabelStore(path)
        s1.append(_label("t1"))
        s1.append(_label("t2"))
        # File on disk has exactly two JSONL lines.
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        # Fresh store reads the same records back.
        s2 = LabelStore(path)
        ids = sorted(r.target_id for r in s2.all())
        assert ids == ["t1", "t2"]

    def test_latest_wins_on_target_collision(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        s = LabelStore(path)
        s.append(_label("t1", status=LabelStatus.TP, note="first"))
        s.append(_label("t1", status=LabelStatus.FP, note="second"))
        latest = s.latest_for("t1")
        assert latest is not None
        assert latest.status == LabelStatus.FP
        assert latest.note == "second"
        # History keeps both rows in append order.
        h = s.history("t1")
        assert [r.note for r in h] == ["first", "second"]

    def test_creates_parent_directory_on_first_append(
        self, tmp_path: Path
    ) -> None:
        nested = tmp_path / "deep" / "down" / "labels.jsonl"
        s = LabelStore(nested)
        s.append(_label("t1"))
        assert nested.is_file()
        assert _label("t1").target_id in {r.target_id for r in s.all()}

    def test_history_filters_by_target(self, tmp_path: Path) -> None:
        s = LabelStore(tmp_path / "labels.jsonl")
        s.append(_label("t1", note="a"))
        s.append(_label("t2", note="b"))
        s.append(_label("t1", note="c"))
        notes = [r.note for r in s.history("t1")]
        assert notes == ["a", "c"]

    def test_iter_returns_all_records(self, tmp_path: Path) -> None:
        s = LabelStore(tmp_path / "labels.jsonl")
        s.append(_label("t1"))
        s.append(_label("t2"))
        assert sorted(r.target_id for r in iter(s)) == ["t1", "t2"]


# ---------------------------------------------------------------------------
# Malformed-line recovery
# ---------------------------------------------------------------------------


class TestMalformedRecovery:
    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        s1 = LabelStore(path)
        s1.append(_label("t1"))
        # Inject blank lines between records.
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n\n   \n")
        s1.append(_label("t2"))
        s2 = LabelStore(path)
        assert sorted(r.target_id for r in s2.all()) == ["t1", "t2"]

    def test_corrupt_json_line_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        # Manually craft a file with a bad line + a good line.
        good = _label("ok").model_dump_json()
        path.write_text(good + "\n{not real json}\n" + good + "\n")
        s = LabelStore(path)
        # Both good lines reload; bad line dropped silently.
        assert [r.target_id for r in s.all()] == ["ok", "ok"]

    def test_schema_mismatch_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        # Valid JSON object but missing required ``target_id`` field.
        path.write_text('{"foo": "bar"}\n')
        s = LabelStore(path)
        assert s.all() == []

    def test_partial_then_recover(self, tmp_path: Path) -> None:
        # Simulate a write that was truncated mid-record.
        path = tmp_path / "labels.jsonl"
        path.write_text('{"target_id": "x", "sour')  # no newline, truncated
        s = LabelStore(path)
        # Bad line dropped; the store still works for new appends.
        assert s.all() == []
        s.append(_label("after"))
        assert [r.target_id for r in s.all()] == ["after"]


# ---------------------------------------------------------------------------
# Concurrent append safety
# ---------------------------------------------------------------------------


class TestConcurrentAppend:
    def test_threaded_appends_preserve_all_records(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "labels.jsonl"
        s = LabelStore(path)
        n_threads = 8
        per_thread = 25
        errors: List[BaseException] = []

        def worker(tid: int) -> None:
            try:
                for i in range(per_thread):
                    s.append(_label(f"t-{tid}-{i}"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # Every append landed exactly once.
        assert len(s.all()) == n_threads * per_thread
        # All target_ids are unique → no append-overwrite races.
        ids = {r.target_id for r in s.all()}
        assert len(ids) == n_threads * per_thread
        # On-disk record count matches in-memory record count.
        assert len(path.read_text().splitlines()) == n_threads * per_thread


# ---------------------------------------------------------------------------
# Batched append (single fsync)
# ---------------------------------------------------------------------------


class TestAppendMany:
    def test_persists_and_indexes_all(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        s = LabelStore(path)
        s.append_many([_label(f"b-{i}") for i in range(5)])
        # On-disk: one line per record.
        assert len(path.read_text().splitlines()) == 5
        # In-memory: latest_by_target is updated for every record.
        latest = s.latest_by_target()
        assert {f"b-{i}" for i in range(5)} <= set(latest)
        # Reloading from disk yields the same set.
        s2 = LabelStore(path)
        ids = sorted(r.target_id for r in s2.all())
        assert ids == [f"b-{i}" for i in range(5)]

    def test_empty_input_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        s = LabelStore(path)
        s.append_many([])
        assert not path.exists() or path.read_text() == ""
        assert s.all() == []

    def test_latest_collision_resolves_to_last_in_batch(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "labels.jsonl"
        s = LabelStore(path)
        s.append_many([
            _label("dup", status=LabelStatus.TP, note="first"),
            _label("dup", status=LabelStatus.FP, note="second"),
        ])
        latest = s.latest_for("dup")
        assert latest is not None
        assert latest.status == LabelStatus.FP
        assert latest.note == "second"

    def test_interleaves_with_append(self, tmp_path: Path) -> None:
        # Mixing single + batched appends preserves order on disk.
        path = tmp_path / "labels.jsonl"
        s = LabelStore(path)
        s.append(_label("a"))
        s.append_many([_label("b"), _label("c")])
        s.append(_label("d"))
        s2 = LabelStore(path)
        assert [r.target_id for r in s2.all()] == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# cell_target_id helper
# ---------------------------------------------------------------------------


class TestCellTargetId:
    def test_stable_format_uses_nul_separators(self) -> None:
        tid = cell_target_id("file.csv", 5, "name")
        # NUL separators avoid collision with cluster:/issue: ID schemas.
        assert tid == "cell\x00file.csv\x005\x00name"

    def test_int_coerced(self) -> None:
        # The signature accepts ints; explicitly coerced inside.
        tid = cell_target_id("file.csv", 5, "c")
        assert "\x005\x00" in tid


def test_reloads_on_out_of_band_write(tmp_path: Path) -> None:
    """Review batch 7: an external writer (offline CLI / second worker)
    appending to the same labels.jsonl must become visible to a cached
    store on the next read -- without an explicit reload() and without
    rebuilding the enclosing RunContext."""
    path = tmp_path / "labels.jsonl"
    s1 = LabelStore(path)
    s1.append(_label("t1"))
    assert set(s1.latest_by_target()) == {"t1"}

    # Out-of-band append via a *separate* store instance bound to the
    # same file (simulates a second process / offline CLI writing).
    s2 = LabelStore(path)
    s2.append(_label("t2", annotator="other"))

    # s1 never wrote t2, but must pick it up on its next read.
    latest = s1.latest_by_target()
    assert set(latest) == {"t1", "t2"}
    assert latest["t2"].annotator == "other"
    assert s1.latest_for("t2") is not None


def test_own_writes_do_not_lose_in_memory_state(tmp_path: Path) -> None:
    """The (mtime,size) signature is refreshed on our own appends, so a
    self-write never spuriously discards in-memory state on the next read."""
    path = tmp_path / "labels.jsonl"
    s = LabelStore(path)
    s.append(_label("t1", note="first"))
    s.append(_label("t1", status=LabelStatus.FP, note="override"))
    latest = s.latest_by_target()
    assert latest["t1"].note == "override"
    assert latest["t1"].status == LabelStatus.FP
