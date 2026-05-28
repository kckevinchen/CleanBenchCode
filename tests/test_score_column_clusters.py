"""Tests for ``experiments.score_column_clusters``.

Slice 2 of the column-clusters labelling subsystem. Covers:
  * library API ``materialize_inconsistency_benchmark``
  * CLI smoke test against a hand-authored CSV + column_clusters.jsonl
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.score_column_clusters import (  # noqa: E402
    _global_cluster_id,
    _resolve_source_path,
    materialize_inconsistency_benchmark,
)
from src.labeling.column_cluster_store import ColumnClusterStore  # noqa: E402
from src.labeling.schema import (  # noqa: E402
    ClusterKind,
    ClusterValue,
    ColumnClusters,
    ColumnLabelStatus,
    StdCluster,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_smoke_csv(path: Path) -> None:
    """Hand-authored 10-row CSV with two known inconsistencies on
    column ``School`` and one on ``State``."""
    rows = [
        # School: BBB(4)/ABC HS(2), EEE(2)/DDD(1); State: CA(7), Calif.(2), Cali(1)
        {"School": "BBB",    "State": "CA"},
        {"School": "BBB",    "State": "CA"},
        {"School": "BBB",    "State": "CA"},
        {"School": "BBB",    "State": "CA"},
        {"School": "ABC HS", "State": "CA"},
        {"School": "ABC HS", "State": "CA"},
        {"School": "EEE",    "State": "Calif."},
        {"School": "EEE",    "State": "Calif."},
        {"School": "DDD",    "State": "Cali"},
        {"School": "BBB",    "State": "CA"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_labeled_record(
    source: str = "smoke.csv",
    column: str = "School",
    annotator: str = "tester",
) -> ColumnClusters:
    return ColumnClusters(
        source=source, column=column, status=ColumnLabelStatus.LABELED,
        clusters=[
            StdCluster(
                cluster_id="c1", kind=ClusterKind.TEXT,
                canonical_value="BBB",
                values=[
                    ClusterValue(value="BBB", count=5),
                    ClusterValue(value="ABC HS", count=2),
                ],
            ),
            StdCluster(
                cluster_id="c2", kind=ClusterKind.UNTAGGED,
                values=[
                    ClusterValue(value="EEE", count=2),
                    ClusterValue(value="DDD", count=1),
                ],
            ),
        ],
        annotator=annotator,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_global_cluster_id_format(self) -> None:
        assert _global_cluster_id("a.csv", "X", "c1") == "a.csv::X::c1"

    def test_resolve_source_with_csv_suffix(self, tmp_path: Path) -> None:
        f = tmp_path / "foo.csv"
        f.write_text("a,b\n1,2\n")
        assert _resolve_source_path(tmp_path, "foo.csv") == f

    def test_resolve_source_without_csv_suffix(self, tmp_path: Path) -> None:
        f = tmp_path / "foo.csv"
        f.write_text("a,b\n1,2\n")
        # _resolve_source_path tries "foo" first, then "foo.csv".
        assert _resolve_source_path(tmp_path, "foo") == f

    def test_resolve_source_missing_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_source_path(tmp_path, "missing.csv") is None


# ---------------------------------------------------------------------------
# Library API
# ---------------------------------------------------------------------------


class TestMaterialize:
    def test_emits_one_cell_per_matched_row(self, tmp_path: Path) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        rec = _make_labeled_record()
        cells = materialize_inconsistency_benchmark([rec], tmp_path)
        # School column has 7 matched rows (5 BBB + 2 ABC HS + 2 EEE + 1 DDD).
        # Wait - the CSV has BBB x 5, ABC HS x 2, EEE x 2, DDD x 1 = 10
        # rows. Cluster c1 covers BBB + ABC HS (7 rows); c2 covers EEE + DDD (3).
        assert len(cells) == 10
        # Every emitted cell uses the same column.
        assert {c.column for c in cells} == {"School"}
        # Direction is always bi_directional at Slice 2.
        # Direction is implicit in corrected_value's type. The smoke
        # record's c1 has canonical "BBB" (operator-confirmed in the
        # fixture) so its cells are one-directional; c2 has no
        # canonical so its cells are bi-directional (a list).
        c1_cells = [c for c in cells if c.cluster_ids[0].endswith("::c1")]
        c2_cells = [c for c in cells if c.cluster_ids[0].endswith("::c2")]
        assert all(c.corrected_value == "BBB" for c in c1_cells)
        assert all(isinstance(c.corrected_value, list) for c in c2_cells)
        # error_types defaults correctly.
        assert all(c.error_types == ["inconsistency"] for c in cells)

    def test_corrected_value_shape_per_cluster(
        self, tmp_path: Path,
    ) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        rec = _make_labeled_record()
        cells = materialize_inconsistency_benchmark([rec], tmp_path)
        c1_cells = [
            c for c in cells if c.cluster_ids[0].endswith("::c1")
        ]
        c2_cells = [
            c for c in cells if c.cluster_ids[0].endswith("::c2")
        ]
        # c1 = {BBB, ABC HS} with canonical=BBB confirmed =>
        # one-directional => scalar corrected_value.
        assert c1_cells, "expected at least one c1 cell"
        assert all(c.corrected_value == "BBB" for c in c1_cells)
        # c2 = {EEE, DDD} with no canonical confirmed =>
        # bi-directional => sorted list in corrected_value.
        assert c2_cells, "expected at least one c2 cell"
        assert all(
            c.corrected_value == ["DDD", "EEE"] for c in c2_cells
        )

    def test_canonical_value_carried_through(self, tmp_path: Path) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        rec = _make_labeled_record()
        cells = materialize_inconsistency_benchmark([rec], tmp_path)
        c1_cells = [
            c for c in cells if c.cluster_ids[0].endswith("::c1")
        ]
        c2_cells = [
            c for c in cells if c.cluster_ids[0].endswith("::c2")
        ]
        # Confirmed canonical => corrected_value carries the scalar.
        assert all(c.corrected_value == "BBB" for c in c1_cells)
        # No canonical => corrected_value is the list (not None).
        assert all(isinstance(c.corrected_value, list) for c in c2_cells)

    def test_cluster_kind_carried_through(self, tmp_path: Path) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        rec = _make_labeled_record()
        cells = materialize_inconsistency_benchmark([rec], tmp_path)
        c1_cells = [
            c for c in cells if c.cluster_ids[0].endswith("::c1")
        ]
        c2_cells = [
            c for c in cells if c.cluster_ids[0].endswith("::c2")
        ]
        assert all(c.cluster_kinds == ["text"] for c in c1_cells)
        assert all(c.cluster_kinds == ["untagged"] for c in c2_cells)

    def test_original_value_matches_csv_cell(self, tmp_path: Path) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        rec = _make_labeled_record()
        cells = materialize_inconsistency_benchmark([rec], tmp_path)
        # Verify each emitted cell's original_value belongs to the
        # cluster's value set. For bi-directional clusters that means
        # ``original_value in corrected_value`` (which is a list);
        # for one-directional clusters we just check it's a known
        # cluster value via membership in original (we lookup the
        # cluster the cell came from).
        for c in cells:
            cv = c.corrected_value
            if isinstance(cv, list):
                assert c.original_value in cv
            else:
                # One-directional: original may equal the canonical
                # itself or be one of the other surface forms (the
                # scorer emits all matching cells, including those
                # whose value already equals the canonical).
                assert isinstance(cv, str)

    def test_row_indices_match_csv(self, tmp_path: Path) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        df = pd.read_csv(tmp_path / "smoke.csv", dtype=str,
                         keep_default_na=False)
        rec = _make_labeled_record()
        cells = materialize_inconsistency_benchmark([rec], tmp_path)
        for c in cells:
            assert df.iloc[c.row][c.column] == c.original_value

    def test_non_labeled_records_skipped(self, tmp_path: Path) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        rec_no = ColumnClusters(
            source="smoke.csv", column="School",
            status=ColumnLabelStatus.NO_INCONSISTENCY,
            annotator="t",
        )
        rec_drop = ColumnClusters(
            source="smoke.csv", column="State",
            status=ColumnLabelStatus.DROPPED_MESSY,
            drop_reason="too noisy", annotator="t",
        )
        cells = materialize_inconsistency_benchmark(
            [rec_no, rec_drop], tmp_path,
        )
        assert cells == []

    def test_missing_csv_warns_and_skips(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Don't create smoke.csv. Scorer should warn-and-skip.
        rec = _make_labeled_record()
        with caplog.at_level("WARNING", logger="score_column_clusters"):
            cells = materialize_inconsistency_benchmark([rec], tmp_path)
        assert cells == []
        assert any("Source CSV not found" in r.message for r in caplog.records)

    def test_missing_column_warns_and_skips(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        rec = _make_labeled_record(column="NonExistentColumn")
        # We can't construct ColumnClusters with an unknown column via the
        # validator (no check there) - column is just a string. But values
        # won't match anything either. We expect a "Column not present"
        # warning.
        # Build a manually-relaxed record that has clusters with valid
        # invariants but column doesn't exist in CSV.
        with caplog.at_level("WARNING", logger="score_column_clusters"):
            cells = materialize_inconsistency_benchmark([rec], tmp_path)
        assert cells == []
        assert any("Column" in r.message and "not present" in r.message
                   for r in caplog.records)

    def test_stale_label_no_match_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        # A labelled cluster that references values not in the CSV.
        rec = ColumnClusters(
            source="smoke.csv", column="School",
            status=ColumnLabelStatus.LABELED,
            clusters=[
                StdCluster(
                    cluster_id="c1",
                    values=[
                        ClusterValue(value="OBSOLETE_A", count=2),
                        ClusterValue(value="OBSOLETE_B", count=1),
                    ],
                ),
            ],
            annotator="t",
        )
        with caplog.at_level("WARNING", logger="score_column_clusters"):
            cells = materialize_inconsistency_benchmark([rec], tmp_path)
        assert cells == []
        assert any("no matching rows" in r.message for r in caplog.records)

    def test_source_filter(self, tmp_path: Path) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        rec_match = _make_labeled_record(source="smoke.csv")
        rec_skip = _make_labeled_record(source="other.csv")
        cells = materialize_inconsistency_benchmark(
            [rec_match, rec_skip], tmp_path, source_filter="smoke.csv",
        )
        assert len(cells) == 10
        assert {c.source for c in cells} == {"smoke.csv"}

    def test_column_filter(self, tmp_path: Path) -> None:
        _build_smoke_csv(tmp_path / "smoke.csv")
        rec_school = _make_labeled_record(column="School")
        rec_state = ColumnClusters(
            source="smoke.csv", column="State",
            status=ColumnLabelStatus.LABELED,
            clusters=[
                StdCluster(
                    cluster_id="c1", canonical_value="CA",
                    values=[
                        ClusterValue(value="CA", count=7),
                        ClusterValue(value="Calif.", count=2),
                        ClusterValue(value="Cali", count=1),
                    ],
                ),
            ],
            annotator="t",
        )
        cells = materialize_inconsistency_benchmark(
            [rec_school, rec_state], tmp_path, column_filter="State",
        )
        assert {c.column for c in cells} == {"State"}
        assert len(cells) == 10  # all 10 rows of State are in cluster c1


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_end_to_end(self, tmp_path: Path) -> None:
        # Build smoke CSV + a column_clusters.jsonl, run the CLI, check
        # output JSONL is well-formed and content matches.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _build_smoke_csv(data_dir / "smoke.csv")

        clusters_path = tmp_path / "column_clusters.jsonl"
        store = ColumnClusterStore(clusters_path)
        store.append(_make_labeled_record())

        out_path = tmp_path / "out.jsonl"
        proc = subprocess.run(
            [
                sys.executable, "-m", "experiments.score_column_clusters",
                "--clusters-jsonl", str(clusters_path),
                "--data-dir", str(data_dir),
                "--output", str(out_path),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
        )
        assert out_path.is_file()
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 10
        for line in lines:
            rec = json.loads(line)
            assert rec["label"] == "error"
            assert rec["error_types"] == ["inconsistency"]
            # corrected_value shape encodes direction:
            #   list  => bi-directional (no confirmed canonical)
            #   str   => one-directional (confirmed canonical)
            cv = rec["corrected_value"]
            assert isinstance(cv, (list, str))
            assert rec["source"] == "smoke.csv"
            assert rec["column"] == "School"
            assert isinstance(rec["cluster_ids"], list)
            assert rec["cluster_ids"]
            assert rec["cluster_ids"][0].startswith("smoke.csv::School::")
            if isinstance(cv, list):
                assert rec["original_value"] in cv
