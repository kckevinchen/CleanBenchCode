"""Regression tests for the launcher path-traversal validators.

Pinned by code-review Pass 5b HIGH#1: ``_path_within`` /
``_glob_within`` / ``_validate_payload_paths`` enforce that every
file path in a launcher payload resolves into the workspace root.
Without these checks an unauthenticated POST to ``/api/launcher/jobs``
would let a caller redirect detection runs at arbitrary host paths
(``/etc/passwd``, ``~/.ssh/`` etc.) and persist outputs anywhere the
service has write access.

These tests poke the validators directly so a future refactor that
removes/relaxes the check fails loudly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.labeling.runs import (
    _DETECTION_GLOB_KEYS,
    _DETECTION_PATH_KEYS,
    _glob_within,
    _path_within,
    _validate_payload_paths,
)


# ---------------------------------------------------------------------------
# _path_within
# ---------------------------------------------------------------------------


class TestPathWithin:
    def test_relative_path_under_root_resolves(self, tmp_path: Path) -> None:
        target = tmp_path / "data" / "x.csv"
        target.parent.mkdir(parents=True)
        target.write_text("a,b\n")
        out = _path_within("data/x.csv", tmp_path)
        assert out == target.resolve()

    def test_absolute_path_under_root_accepted(self, tmp_path: Path) -> None:
        target = tmp_path / "data" / "x.csv"
        target.parent.mkdir(parents=True)
        target.write_text("a,b\n")
        out = _path_within(str(target), tmp_path)
        assert out == target.resolve()

    def test_dotdot_traversal_rejected(self, tmp_path: Path) -> None:
        # Even if the parent of tmp_path is benign, a ".." segment
        # MUST escape the workspace and trip the check.
        with pytest.raises(ValueError, match="escapes project root"):
            _path_within("../../../etc/passwd", tmp_path)

    def test_absolute_path_outside_root_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="escapes project root"):
            _path_within("/etc/passwd", tmp_path)

    def test_empty_string_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="path is required"):
            _path_within("", tmp_path)

    def test_symlink_pointing_outside_root_rejected(self, tmp_path: Path) -> None:
        # Construct: tmp_path/escape -> /tmp/<unique>/secret. Resolving
        # tmp_path/escape must follow the link and detect the escape.
        outside = tmp_path.parent / f"_outside_{os.getpid()}"
        outside.mkdir(exist_ok=True)
        try:
            (outside / "secret.txt").write_text("hi")
            link = tmp_path / "escape"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                pytest.skip("symlinks unsupported on this platform")
            with pytest.raises(ValueError, match="escapes project root"):
                _path_within("escape/secret.txt", tmp_path)
        finally:
            # Clean up; tmp_path itself is auto-cleaned by pytest.
            for child in outside.glob("**/*"):
                if child.is_file():
                    child.unlink()
            outside.rmdir()

    def test_nonexistent_path_under_root_still_resolves(self, tmp_path: Path) -> None:
        # Validator must work BEFORE the file exists (the launcher
        # creates output_dir on first write). resolve(strict=False)
        # handles this; the relative_to check still catches escapes.
        out = _path_within("outputs/notyet/x.json", tmp_path)
        assert str(out).startswith(str(tmp_path.resolve()))


# ---------------------------------------------------------------------------
# _glob_within
# ---------------------------------------------------------------------------


class TestGlobWithin:
    def test_relative_glob_accepted(self, tmp_path: Path) -> None:
        assert _glob_within("data/*.csv", tmp_path) == "data/*.csv"
        assert _glob_within("*.csv", tmp_path) == "*.csv"
        assert _glob_within("a/b/c/*.parquet", tmp_path) == "a/b/c/*.parquet"

    def test_empty_glob_passes_through(self, tmp_path: Path) -> None:
        assert _glob_within("", tmp_path) == ""

    def test_dotdot_segment_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            _glob_within("../*.csv", tmp_path)
        with pytest.raises(ValueError, match=r"\.\."):
            _glob_within("data/../../*.csv", tmp_path)

    def test_absolute_glob_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="absolute glob"):
            _glob_within("/etc/*.csv", tmp_path)

    def test_windows_separator_dotdot_rejected(self, tmp_path: Path) -> None:
        # Validator splits on both POSIX and Windows separators so a
        # mixed-separator payload can't sneak past.
        with pytest.raises(ValueError, match=r"\.\."):
            _glob_within("data\\..\\*.csv", tmp_path)


# ---------------------------------------------------------------------------
# _validate_payload_paths
# ---------------------------------------------------------------------------


class TestValidatePayloadPaths:
    def test_all_path_keys_canonicalised(self, tmp_path: Path) -> None:
        (tmp_path / "data").mkdir()
        (tmp_path / "outputs").mkdir()
        (tmp_path / "config.yaml").write_text("a: 1\n")
        payload = {
            "csv_dir": "data",
            "output_dir": "outputs",
            "config": "config.yaml",
            "csv_glob": "*.csv",
            "model": "gpt-4",  # non-path keys untouched
        }
        _validate_payload_paths(
            payload,
            project_root=tmp_path,
            path_keys=_DETECTION_PATH_KEYS,
            glob_keys=_DETECTION_GLOB_KEYS,
        )
        assert payload["csv_dir"] == str((tmp_path / "data").resolve())
        assert payload["output_dir"] == str((tmp_path / "outputs").resolve())
        assert payload["config"] == str((tmp_path / "config.yaml").resolve())
        assert payload["csv_glob"] == "*.csv"  # globs not resolved
        assert payload["model"] == "gpt-4"  # untouched

    def test_traversal_in_any_path_key_blocks_launch(self, tmp_path: Path) -> None:
        for evil_key in _DETECTION_PATH_KEYS:
            payload = {evil_key: "../../../etc/passwd"}
            with pytest.raises(ValueError, match=f"invalid {evil_key}"):
                _validate_payload_paths(
                    payload,
                    project_root=tmp_path,
                    path_keys=_DETECTION_PATH_KEYS,
                    glob_keys=_DETECTION_GLOB_KEYS,
                )

    def test_traversal_in_glob_key_blocks_launch(self, tmp_path: Path) -> None:
        payload = {"csv_glob": "../*.csv"}
        with pytest.raises(ValueError, match="invalid csv_glob"):
            _validate_payload_paths(
                payload,
                project_root=tmp_path,
                path_keys=_DETECTION_PATH_KEYS,
                glob_keys=_DETECTION_GLOB_KEYS,
            )

    def test_missing_keys_are_ignored(self, tmp_path: Path) -> None:
        # Empty / missing path values should not raise.
        payload = {"csv_dir": "", "output_dir": None, "model": "gpt-4"}
        _validate_payload_paths(
            payload,
            project_root=tmp_path,
            path_keys=_DETECTION_PATH_KEYS,
            glob_keys=_DETECTION_GLOB_KEYS,
        )
        assert payload["csv_dir"] == ""
        assert payload["output_dir"] is None
        assert payload["model"] == "gpt-4"
