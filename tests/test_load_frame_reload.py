"""RunContext.load_frame must serve the TRUE current CSV cell value, not a
stale memoized frame. When the dataset is normalized on disk AFTER a label run
(e.g. excess whitespace removed), the next ``load_frame`` has to pick up the
edit instead of re-serving the cached pre-normalization frame -- otherwise the
trivial taggers flag whitespace that no longer exists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.labeling.app import RunContext


def _build_run(tmp_path: Path, cell: str) -> tuple[Path, Path]:
    project = tmp_path / "proj"
    (project / "data").mkdir(parents=True)
    csv = project / "data" / "vendors.csv"
    csv.write_text(f'id,name\n1,"{cell}"\n', encoding="utf-8")
    rd = project / "outputs" / "20260101T000000Z_run"
    rd.mkdir(parents=True)
    (rd / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    (rd / "clusters.json").write_text("[]")
    (rd / "issues.json").write_text("[]")
    return rd, csv


def test_unchanged_file_serves_cached_frame(tmp_path: Path) -> None:
    rd, _csv = _build_run(tmp_path, "Acme Co")
    ctx = RunContext(rd)
    first = ctx.load_frame("vendors.csv")
    second = ctx.load_frame("vendors.csv")
    # Same object: the memoized frame is reused when the file is unchanged.
    assert first is second


def test_frame_reloads_after_csv_normalized_on_disk(tmp_path: Path) -> None:
    # Pre-normalization value carries a double internal space.
    rd, csv = _build_run(tmp_path, "Fallen Below 18  Days")
    ctx = RunContext(rd)
    before = ctx.load_frame("vendors.csv")
    assert before.iat[0, before.columns.get_loc("name")] == "Fallen Below 18  Days"

    # Normalize the dataset on disk (collapse the double space). Bump mtime
    # explicitly so the change is detectable even on coarse clocks (the size
    # also shrinks by one byte, which alone is enough).
    csv.write_text('id,name\n1,"Fallen Below 18 Days"\n', encoding="utf-8")
    st = csv.stat()
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    after = ctx.load_frame("vendors.csv")
    # The reloaded frame reflects the TRUE current cell value (single space),
    # so a whitespace tagger scanning it would no longer flag the cell.
    assert after.iat[0, after.columns.get_loc("name")] == "Fallen Below 18 Days"
    assert after is not before


def test_whitespace_anomaly_clears_after_reload(tmp_path: Path) -> None:
    from src.labeling._helpers import _has_whitespace_anomaly, _scalar

    rd, csv = _build_run(tmp_path, "Fallen Below 18  Days")
    ctx = RunContext(rd)
    frame = ctx.load_frame("vendors.csv")
    val = _scalar(frame.iat[0, frame.columns.get_loc("name")])
    assert _has_whitespace_anomaly(val) is True

    csv.write_text('id,name\n1,"Fallen Below 18 Days"\n', encoding="utf-8")
    st = csv.stat()
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    frame2 = ctx.load_frame("vendors.csv")
    val2 = _scalar(frame2.iat[0, frame2.columns.get_loc("name")])
    assert _has_whitespace_anomaly(val2) is False
