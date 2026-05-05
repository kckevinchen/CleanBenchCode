"""Tests for src.core.jsonl_io: shared JSONL read/write helpers."""

from __future__ import annotations

import json

import pytest

from src.core.jsonl_io import iter_jsonl, read_jsonl, write_jsonl


class TestReadJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        assert read_jsonl(tmp_path / "no.jsonl") == []

    def test_round_trip(self, tmp_path):
        path = tmp_path / "out.jsonl"
        recs = [{"a": 1}, {"b": "two"}, {"c": [1, 2, 3]}]
        n = write_jsonl(path, recs)
        assert n == 3
        assert read_jsonl(path) == recs

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blank.jsonl"
        path.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
        assert read_jsonl(path) == [{"a": 1}, {"b": 2}]

    def test_malformed_line_warn_default(self, tmp_path, caplog):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"a": 1}\nNOT_JSON\n{"b": 2}\n', encoding="utf-8")
        with caplog.at_level("WARNING"):
            recs = read_jsonl(path)
        assert recs == [{"a": 1}, {"b": 2}]
        assert any("malformed" in m for m in caplog.messages)

    def test_malformed_line_skip_silent(self, tmp_path, caplog):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"a": 1}\nNOT_JSON\n', encoding="utf-8")
        with caplog.at_level("WARNING"):
            recs = read_jsonl(path, on_error="skip")
        assert recs == [{"a": 1}]
        assert not any("malformed" in m for m in caplog.messages)

    def test_malformed_line_raise(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('NOT_JSON\n', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            read_jsonl(path, on_error="raise")

    def test_limit_short_circuits(self, tmp_path):
        path = tmp_path / "many.jsonl"
        write_jsonl(path, [{"i": i} for i in range(100)])
        recs = read_jsonl(path, limit=5)
        assert recs == [{"i": i} for i in range(5)]


class TestIterJsonl:
    def test_streams_records(self, tmp_path):
        path = tmp_path / "s.jsonl"
        write_jsonl(path, [{"i": i} for i in range(3)])
        out = list(iter_jsonl(path))
        assert out == [{"i": 0}, {"i": 1}, {"i": 2}]

    def test_missing_file_yields_nothing(self, tmp_path):
        assert list(iter_jsonl(tmp_path / "absent.jsonl")) == []


class TestWriteJsonl:
    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.jsonl"
        n = write_jsonl(path, [{"x": 1}])
        assert n == 1
        assert path.is_file()

    def test_unicode_preserved(self, tmp_path):
        path = tmp_path / "u.jsonl"
        write_jsonl(path, [{"name": "你好"}])
        # ensure_ascii=False -> raw UTF-8 on disk.
        assert "你好" in path.read_text(encoding="utf-8")

    def test_empty_iterable(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        n = write_jsonl(path, [])
        assert n == 0
        assert path.is_file()
        assert path.read_text() == ""
