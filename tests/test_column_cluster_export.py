"""Tests for Slice 5: ``column_cluster_export`` and the CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.column_cluster_export import (  # noqa: E402
    benchmark_output_filename,
    export_inconsistency_benchmark,
    load_records_for_export,
    synthesize_records_from_drafts_and_labels,
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


def _make_state_record(
    *,
    annotator: str = "reviewer",
    note: str = "test",
    canonical: str = "California",
    counts: List[int] = (18, 6, 4),  # type: ignore[assignment]
    confirmed: bool = False,
) -> ColumnClusters:
    """Build a labelled record for the State column with 3 surface forms.

    By default the record holds an UN-confirmed LLM proposal
    (``llm_proposed_canonical=canonical``, ``canonical_value=None``)
    so the cluster stays bi-directional in the scorer's output and
    the threshold-flip path is what drives any one-directional fix.
    Set ``confirmed=True`` to model the operator endorsing the
    canonical (then the scorer emits one-directional unconditionally).
    """
    return ColumnClusters(
        source="smoke.csv",
        column="State",
        status=ColumnLabelStatus.LABELED,
        clusters=[
            StdCluster(
                cluster_id="c1",
                kind=ClusterKind.TEXT,
                canonical_value=(canonical if confirmed else None),
                llm_proposed_canonical=canonical,
                values=[
                    ClusterValue(value="CA", count=counts[0]),
                    ClusterValue(value="Calif.", count=counts[1]),
                    ClusterValue(value="California", count=counts[2]),
                ],
            ),
        ],
        annotator=annotator,
        ts="2026-05-12T00:00:00Z",
        note=note,
    )


@pytest.fixture
def smoke_dir(tmp_path: Path) -> Path:
    """Build a tiny data dir with 28 State-column rows.

    Distribution: 18x 'CA', 6x 'Calif.', 4x 'California'. Matches the
    fixture cluster counts so all rows are covered.
    """
    rows = (
        ["CA"] * 18 + ["Calif."] * 6 + ["California"] * 4
    )
    df = pd.DataFrame({"State": rows})
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_csv(data_dir / "smoke.csv", index=False)
    return data_dir


# ---------------------------------------------------------------------------
# Filename convention
# ---------------------------------------------------------------------------


class TestBenchmarkFilename:
    def test_with_tag(self) -> None:
        out = benchmark_output_filename("20260512T010203", "smoke-run")
        assert out == "benchmark_inconsistency_20260512T010203Z_smoke-run.jsonl"

    def test_without_tag(self) -> None:
        out = benchmark_output_filename("20260512T010203", None)
        assert out == "benchmark_inconsistency_20260512T010203Z.jsonl"


# ---------------------------------------------------------------------------
# Bi-directional default
# ---------------------------------------------------------------------------


class TestBiDirectional:
    def test_default_emits_all_cells_bi_directional(
        self, smoke_dir: Path,
    ) -> None:
        rec = _make_state_record()
        cells, decisions = export_inconsistency_benchmark(
            [rec], smoke_dir, threshold=None,
        )
        assert len(cells) == 28  # every row
        for c in cells:
            # Bi-directional: corrected_value is the sorted list of
            # cluster surface forms (no canonical confirmed).
            assert c.corrected_value == ["CA", "Calif.", "California"]
            assert c.label == "error"
            assert c.error_types == ["inconsistency"]
            assert c.cluster_kinds == ["text"]
        assert len(decisions) == 1
        d = decisions[0]
        assert d.flipped is False
        assert d.canonical_fraction == pytest.approx(4 / 28)


# ---------------------------------------------------------------------------
# Threshold flip
# ---------------------------------------------------------------------------


class TestThresholdFlip:
    def test_below_threshold_stays_bidirectional(self, smoke_dir: Path) -> None:
        rec = _make_state_record()  # canonical fraction = 4/28 ≈ 0.143
        cells, decisions = export_inconsistency_benchmark(
            [rec], smoke_dir, threshold=0.5,
        )
        assert len(cells) == 28
        # Stays bi-directional: corrected_value is still the list.
        assert all(
            c.corrected_value == ["CA", "Calif.", "California"]
            for c in cells
        )
        assert decisions[0].flipped is False

    def test_at_or_above_threshold_flips_to_one_directional(
        self, tmp_path: Path,
    ) -> None:
        # Build a column where canonical dominates: 8x 'California',
        # 1x 'CA', 1x 'Calif.'. Canonical fraction = 0.8.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rows = ["California"] * 8 + ["CA"] + ["Calif."]
        pd.DataFrame({"State": rows}).to_csv(
            data_dir / "smoke.csv", index=False,
        )
        rec = _make_state_record(counts=[1, 1, 8])  # CA=1, Calif.=1, Cal=8
        cells, decisions = export_inconsistency_benchmark(
            [rec], data_dir, threshold=0.7,
        )
        # Canonical rows dropped, only 2 non-canonical cells remain.
        assert len(cells) == 2
        original_values = sorted(c.original_value for c in cells)
        assert original_values == ["CA", "Calif."]
        for c in cells:
            # One-directional: corrected_value is the scalar canonical.
            assert c.corrected_value == "California"
            assert c.label == "error"
        assert decisions[0].flipped is True
        assert decisions[0].canonical_fraction == pytest.approx(0.8)

    def test_exact_threshold_match_flips(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rows = ["X"] * 6 + ["x"] * 4  # canonical fraction = 0.6
        pd.DataFrame({"C": rows}).to_csv(data_dir / "s.csv", index=False)
        rec = ColumnClusters(
            source="s.csv", column="C",
            status=ColumnLabelStatus.LABELED,
            clusters=[
                StdCluster(
                    cluster_id="c1",
                    kind=ClusterKind.CASING,
                    # Unconfirmed LLM hint => threshold path is what
                    # promotes this cluster to one-directional.
                    canonical_value=None,
                    llm_proposed_canonical="X",
                    values=[
                        ClusterValue(value="X", count=6),
                        ClusterValue(value="x", count=4),
                    ],
                ),
            ],
            annotator="reviewer", ts="2026-01-01T00:00:00Z",
        )
        cells, decisions = export_inconsistency_benchmark(
            [rec], data_dir, threshold=0.6,
        )
        assert decisions[0].flipped is True
        assert len(cells) == 4
        assert all(c.corrected_value == "X" for c in cells)

    def test_mixed_clusters_flip_independently(
        self, tmp_path: Path,
    ) -> None:
        """One cluster flips, another stays bi-directional."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rows = (
            ["AAA"] * 8 + ["aaa"] * 2 +  # cluster 1: canon 0.8 -> flip
            ["BBB"] * 5 + ["bbb"] * 5    # cluster 2: canon 0.5 -> stay
        )
        pd.DataFrame({"C": rows}).to_csv(data_dir / "s.csv", index=False)
        rec = ColumnClusters(
            source="s.csv", column="C",
            status=ColumnLabelStatus.LABELED,
            clusters=[
                StdCluster(
                    cluster_id="c1",
                    kind=ClusterKind.CASING,
                    # Unconfirmed LLM hint => threshold path drives
                    # the one-directional flip.
                    canonical_value=None,
                    llm_proposed_canonical="AAA",
                    values=[
                        ClusterValue(value="AAA", count=8),
                        ClusterValue(value="aaa", count=2),
                    ],
                ),
                StdCluster(
                    cluster_id="c2",
                    kind=ClusterKind.CASING,
                    canonical_value=None,
                    llm_proposed_canonical="BBB",
                    values=[
                        ClusterValue(value="BBB", count=5),
                        ClusterValue(value="bbb", count=5),
                    ],
                ),
            ],
            annotator="reviewer", ts="2026-01-01T00:00:00Z",
        )
        cells, decisions = export_inconsistency_benchmark(
            [rec], data_dir, threshold=0.7,
        )
        by_id = {d.cluster_id: d for d in decisions}
        assert by_id["c1"].flipped is True
        assert by_id["c2"].flipped is False
        # c1 contributes 2 one-directional cells, c2 contributes 10
        # bi-directional cells.
        c1_cells = [
            c for c in cells
            if any("c1" in cid for cid in c.cluster_ids)
        ]
        c2_cells = [
            c for c in cells
            if any("c2" in cid for cid in c.cluster_ids)
        ]
        assert len(c1_cells) == 2
        # One-directional => scalar canonical in corrected_value.
        assert all(c.corrected_value == "AAA" for c in c1_cells)
        assert len(c2_cells) == 10
        # Bi-directional => list of distinct surface forms.
        assert all(
            c.corrected_value == ["BBB", "bbb"] for c in c2_cells
        )


class TestThresholdValidation:
    @pytest.mark.parametrize("t", [-0.1, 0.0, 1.1, 2.0])
    def test_invalid_threshold_raises(
        self, smoke_dir: Path, t: float,
    ) -> None:
        rec = _make_state_record()
        with pytest.raises(ValueError, match="threshold"):
            export_inconsistency_benchmark([rec], smoke_dir, threshold=t)


# ---------------------------------------------------------------------------
# Filter passthrough
# ---------------------------------------------------------------------------


class TestFilters:
    def test_source_filter_excludes_other_records(
        self, smoke_dir: Path,
    ) -> None:
        rec_state = _make_state_record()
        rec_other = ColumnClusters(
            source="other.csv", column="C",
            status=ColumnLabelStatus.LABELED,
            clusters=[
                StdCluster(
                    cluster_id="c1",
                    kind=ClusterKind.TEXT,
                    canonical_value="X",
                    values=[
                        ClusterValue(value="X", count=1),
                        ClusterValue(value="Y", count=1),
                    ],
                ),
            ],
            annotator="reviewer", ts="2026-01-01T00:00:00Z",
        )
        cells, decisions = export_inconsistency_benchmark(
            [rec_state, rec_other], smoke_dir,
            source_filter="smoke.csv",
        )
        assert {d.source for d in decisions} == {"smoke.csv"}
        assert all(c.source == "smoke.csv" for c in cells)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCLI:
    def _build_run_dir(self, tmp_path: Path) -> Path:
        # Build data + run with one labelled column.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rows = ["California"] * 8 + ["CA"] * 1 + ["Calif."] * 1
        pd.DataFrame({"State": rows}).to_csv(
            data_dir / "smoke.csv", index=False,
        )
        run_dir = tmp_path / "outputs" / "smoke_run"
        run_dir.mkdir(parents=True)
        store = ColumnClusterStore(run_dir / "column_clusters.jsonl")
        store.append(_make_state_record(counts=[1, 1, 8]))
        return tmp_path

    def test_cli_default_bi_directional(self, tmp_path: Path) -> None:
        ws = self._build_run_dir(tmp_path)
        label_dir = ws / "label_result"
        proc = subprocess.run(
            [
                sys.executable, "-m",
                "experiments.export_inconsistency_benchmark",
                "--run-dir", str(ws / "outputs" / "smoke_run"),
                "--data-dir", str(ws / "data"),
                "--label-result-dir", str(label_dir),
                "--tag", "smoke",
                "--timestamp", "20260512T000000",
            ],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        out_path = label_dir / (
            "benchmark_inconsistency_20260512T000000Z_smoke.jsonl"
        )
        assert out_path.exists(), proc.stderr
        cells = [
            json.loads(line) for line in out_path.read_text().splitlines()
        ]
        assert len(cells) == 10
        # Bi-directional: corrected_value is the sorted distinct
        # surface-form list.
        for c in cells:
            assert isinstance(c["corrected_value"], list)
            assert c["corrected_value"] == ["CA", "Calif.", "California"]

        # Sidecar flips report
        flips_path = label_dir / (
            "benchmark_inconsistency_20260512T000000Z_smoke.flips.json"
        )
        assert flips_path.exists()
        body = json.loads(flips_path.read_text())
        assert body["threshold"] is None
        assert body["summary"]["n_flipped"] == 0
        assert body["summary"]["n_clusters"] == 1

    def test_cli_threshold_flip(self, tmp_path: Path) -> None:
        ws = self._build_run_dir(tmp_path)
        label_dir = ws / "label_result"
        proc = subprocess.run(
            [
                sys.executable, "-m",
                "experiments.export_inconsistency_benchmark",
                "--run-dir", str(ws / "outputs" / "smoke_run"),
                "--data-dir", str(ws / "data"),
                "--label-result-dir", str(label_dir),
                "--tag", "smoke",
                "--threshold", "0.6",
                "--timestamp", "20260512T000000",
            ],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        out_path = label_dir / (
            "benchmark_inconsistency_20260512T000000Z_smoke.jsonl"
        )
        cells = [
            json.loads(line) for line in out_path.read_text().splitlines()
        ]
        # 2 non-canonical rows after flip
        assert len(cells) == 2
        for c in cells:
            # One-directional: corrected_value is the scalar canonical.
            assert c["corrected_value"] == "California"
        body = json.loads(
            (label_dir / (
                "benchmark_inconsistency_20260512T000000Z_smoke.flips.json"
            )).read_text(),
        )
        assert body["summary"]["n_flipped"] == 1

    def test_cli_missing_clusters_file_fails(self, tmp_path: Path) -> None:
        proc = subprocess.run(
            [
                sys.executable, "-m",
                "experiments.export_inconsistency_benchmark",
                "--run-dir", str(tmp_path / "nope"),
                "--data-dir", str(tmp_path),
                "--label-result-dir", str(tmp_path / "out"),
            ],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 2
        # Either the legacy message (column_clusters.jsonl alone) or
        # the new one (neither file found) is acceptable; we only
        # require the run-dir basename to be cited so the operator
        # can identify which path was probed.
        stderr = proc.stderr.lower()
        assert "found" in stderr
        assert "nope" in stderr


# ---------------------------------------------------------------------------
# Per-cluster-label synthesis (drafts + labels.jsonl)
# ---------------------------------------------------------------------------


def _write_drafts_jsonl(
    drafts_path: Path,
    *,
    source: str,
    column: str,
    clusters: List[dict],
) -> None:
    """Write one column's worth of LLM-proposed draft clusters."""
    drafts_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "source": source,
        "column": column,
        "draft_clusters": clusters,
    }
    with drafts_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc) + "\n")


def _write_labels_jsonl(
    labels_path: Path,
    *,
    entries: List[dict],
) -> None:
    """Append labels with column_cluster targets."""
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def _column_cluster_target_id(
    source: str, column: str, local_id: str,
) -> str:
    return f"colclust\x00{source}\x00{column}\x00{local_id}"


class TestSynthesizeFromDraftsAndLabels:
    """The export path used when the operator labelled clusters
    individually from the queue (per-cluster verdict on LLM drafts)
    rather than finalising a partition via the column board."""

    def test_only_tp_and_partial_clusters_are_synthesized(
        self, tmp_path: Path,
    ) -> None:
        """Drafts with verdict tp/partial become single-cluster
        records; fp/unsure/skip/unlabeled drafts are excluded."""
        drafts = tmp_path / "column_cluster_drafts.jsonl"
        labels = tmp_path / "labels.jsonl"
        _write_drafts_jsonl(
            drafts,
            source="smoke.csv", column="State",
            clusters=[
                {"kind": "text", "canonical": "California",
                 "values": ["CA", "Calif.", "California"]},
                {"kind": "casing", "canonical": None,
                 "values": ["nyc", "NYC"]},
                {"kind": "text", "canonical": "TX",
                 "values": ["TX", "Texas"]},
                {"kind": "text", "canonical": None,
                 "values": ["A", "B"]},
            ],
        )
        _write_labels_jsonl(
            labels,
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c1"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp", "annotator": "reviewer"},
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c2"),
                 "source": "smoke.csv", "column": "State",
                 "status": "fp", "annotator": "reviewer"},
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c3"),
                 "source": "smoke.csv", "column": "State",
                 "status": "partial", "annotator": "reviewer"},
                # c4 has no label entry -> excluded.
            ],
        )
        records = synthesize_records_from_drafts_and_labels(drafts, labels)
        # c1 (tp) and c3 (partial) -> 2 records, one cluster each.
        assert len(records) == 2
        assert all(len(r.clusters) == 1 for r in records)
        cluster_ids = {r.clusters[0].cluster_id for r in records}
        assert cluster_ids == {"c1", "c3"}
        # The OLD behaviour folded the LLM's draft.canonical into the
        # cluster's canonical_value. The NEW behaviour reserves
        # canonical_value for the operator's confirmed pick (none of
        # these labels carry confirmed_canonical) and stashes the
        # LLM hint on llm_proposed_canonical instead.
        canonicals = {r.clusters[0].canonical_value for r in records}
        assert canonicals == {None}
        llm_proposals = {
            r.clusters[0].llm_proposed_canonical for r in records
        }
        assert llm_proposals == {"California", "TX"}

    def test_confirmed_canonical_promotes_to_canonical_value(
        self, tmp_path: Path,
    ) -> None:
        """When a label carries ``confirmed_canonical``, the
        synthesised cluster's ``canonical_value`` is set to that
        value (operator-confirmed) and the cluster becomes
        one-directional in the downstream export. The LLM's draft
        canonical is preserved separately on
        ``llm_proposed_canonical``.
        """
        drafts = tmp_path / "column_cluster_drafts.jsonl"
        labels = tmp_path / "labels.jsonl"
        _write_drafts_jsonl(
            drafts,
            source="smoke.csv", column="State",
            clusters=[
                # c1: LLM proposed "California" but operator confirms
                # "CA" instead (legitimate disagreement).
                {"kind": "text", "canonical": "California",
                 "values": ["CA", "Calif.", "California"]},
                # c2: LLM proposed "TX"; operator confirms "TX".
                {"kind": "text", "canonical": "TX",
                 "values": ["TX", "Texas"]},
                # c3: LLM proposed nothing; operator confirms.
                {"kind": "text", "canonical": None,
                 "values": ["NY", "N.Y."]},
                # c4: TP without any canonical pick (stays bi).
                {"kind": "text", "canonical": "FL",
                 "values": ["FL", "Florida"]},
            ],
        )
        _write_labels_jsonl(
            labels,
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c1"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp", "confirmed_canonical": "CA",
                 "annotator": "reviewer"},
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c2"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp", "confirmed_canonical": "TX",
                 "annotator": "reviewer"},
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c3"),
                 "source": "smoke.csv", "column": "State",
                 "status": "partial", "confirmed_canonical": "NY",
                 "annotator": "reviewer"},
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c4"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp",  # no confirmed_canonical => bi-directional
                 "annotator": "reviewer"},
            ],
        )
        records = synthesize_records_from_drafts_and_labels(drafts, labels)
        assert len(records) == 4
        by_id = {r.clusters[0].cluster_id: r.clusters[0] for r in records}
        # c1: confirmed=CA, llm proposed California (split preserved).
        assert by_id["c1"].canonical_value == "CA"
        assert by_id["c1"].llm_proposed_canonical == "California"
        # c2: confirmed agrees with LLM.
        assert by_id["c2"].canonical_value == "TX"
        assert by_id["c2"].llm_proposed_canonical == "TX"
        # c3: confirmed without LLM hint.
        assert by_id["c3"].canonical_value == "NY"
        assert by_id["c3"].llm_proposed_canonical is None
        # c4: TP without confirm => stays bi-directional.
        assert by_id["c4"].canonical_value is None
        assert by_id["c4"].llm_proposed_canonical == "FL"

    def test_stale_confirmed_canonical_is_nulled(
        self, tmp_path: Path,
    ) -> None:
        """A confirmed_canonical that doesn't match any cluster value
        (e.g. drafts were re-clustered after the operator's vote)
        is silently nulled rather than raising."""
        drafts = tmp_path / "column_cluster_drafts.jsonl"
        labels = tmp_path / "labels.jsonl"
        _write_drafts_jsonl(
            drafts,
            source="smoke.csv", column="State",
            clusters=[
                {"kind": "text", "canonical": "California",
                 "values": ["CA", "California"]},
            ],
        )
        _write_labels_jsonl(
            labels,
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c1"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp",
                 # Stale: operator confirmed "Calif." but it's no
                 # longer a value of the (re-clustered) cluster.
                 "confirmed_canonical": "Calif.",
                 "annotator": "reviewer"},
            ],
        )
        records = synthesize_records_from_drafts_and_labels(drafts, labels)
        assert len(records) == 1
        # canonical_value gracefully nulled, llm hint preserved.
        assert records[0].clusters[0].canonical_value is None
        assert (
            records[0].clusters[0].llm_proposed_canonical
            == "California"
        )

    def test_latest_label_wins_per_target(self, tmp_path: Path) -> None:
        """Multiple verdicts on the same target_id: last line wins
        (matches the labels store's append-order semantics)."""
        drafts = tmp_path / "column_cluster_drafts.jsonl"
        labels = tmp_path / "labels.jsonl"
        _write_drafts_jsonl(
            drafts,
            source="smoke.csv", column="State",
            clusters=[
                {"kind": "text", "canonical": "California",
                 "values": ["CA", "California"]},
            ],
        )
        tid = _column_cluster_target_id("smoke.csv", "State", "c1")
        _write_labels_jsonl(
            labels,
            entries=[
                {"target": "column_cluster", "target_id": tid,
                 "source": "smoke.csv", "column": "State",
                 "status": "tp", "annotator": "k"},
                {"target": "column_cluster", "target_id": tid,
                 "source": "smoke.csv", "column": "State",
                 "status": "fp", "annotator": "k"},
            ],
        )
        records = synthesize_records_from_drafts_and_labels(drafts, labels)
        # The most-recent verdict was fp -> excluded.
        assert records == []

    def test_overlapping_drafts_become_separate_records(
        self, tmp_path: Path,
    ) -> None:
        """LLM may emit a coarse superset cluster + fine-grained
        subset clusters; both can be TP'd. Each becomes its own
        record so the disjointness check (within a record) still
        passes."""
        drafts = tmp_path / "column_cluster_drafts.jsonl"
        labels = tmp_path / "labels.jsonl"
        _write_drafts_jsonl(
            drafts,
            source="smoke.csv", column="School",
            clusters=[
                {"kind": "casing", "canonical": "BBB",
                 "values": ["BBB", "ABC HS", "EEE", "DDD"]},
                {"kind": "casing", "canonical": "BBB",
                 "values": ["BBB", "ABC HS"]},
            ],
        )
        _write_labels_jsonl(
            labels,
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "School", "c1"),
                 "source": "smoke.csv", "column": "School",
                 "status": "tp", "annotator": "k"},
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "School", "c2"),
                 "source": "smoke.csv", "column": "School",
                 "status": "tp", "annotator": "k"},
            ],
        )
        records = synthesize_records_from_drafts_and_labels(drafts, labels)
        assert len(records) == 2
        # Both records target the same (source, column) but each
        # carries its own single-cluster definition.
        assert all(r.source == "smoke.csv" and r.column == "School"
                   for r in records)
        assert {r.clusters[0].cluster_id for r in records} == {"c1", "c2"}
        # Overlapping values are fine across records (the disjoint
        # check is per-record, not across the export).
        assert {v.value for v in records[0].clusters[0].values} == {
            "BBB", "ABC HS", "EEE", "DDD",
        }
        assert {v.value for v in records[1].clusters[0].values} == {
            "BBB", "ABC HS",
        }

    def test_missing_drafts_returns_empty_list(self, tmp_path: Path) -> None:
        records = synthesize_records_from_drafts_and_labels(
            tmp_path / "missing_drafts.jsonl",
            tmp_path / "missing_labels.jsonl",
        )
        assert records == []

    def test_drafts_present_no_labels_returns_empty_list(
        self, tmp_path: Path,
    ) -> None:
        drafts = tmp_path / "column_cluster_drafts.jsonl"
        _write_drafts_jsonl(
            drafts,
            source="smoke.csv", column="State",
            clusters=[
                {"kind": "text", "canonical": "California",
                 "values": ["CA", "California"]},
            ],
        )
        records = synthesize_records_from_drafts_and_labels(
            drafts, tmp_path / "missing_labels.jsonl",
        )
        assert records == []

    def test_skip_on_any_draft_excludes_whole_column(
        self, tmp_path: Path,
    ) -> None:
        """Mirror of the cell-level skip rule: a single ``skip``
        verdict on ANY draft for a (source, column) drops the entire
        column from the export, including sibling TP/Partial verdicts
        on that same column."""
        drafts = tmp_path / "column_cluster_drafts.jsonl"
        labels = tmp_path / "labels.jsonl"
        _write_drafts_jsonl(
            drafts,
            source="smoke.csv", column="State",
            clusters=[
                # c1 looks legit, would normally be exported.
                {"kind": "text", "canonical": "California",
                 "values": ["CA", "California"]},
                # c2 is the operator's "this column is unusable" flag.
                {"kind": "text", "canonical": None,
                 "values": ["x", "y", "z"]},
            ],
        )
        _write_labels_jsonl(
            labels,
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c1"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp", "annotator": "k"},
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c2"),
                 "source": "smoke.csv", "column": "State",
                 "status": "skip", "annotator": "k"},
            ],
        )
        records = synthesize_records_from_drafts_and_labels(drafts, labels)
        # c1 (tp) is suppressed because the column is skip-flagged.
        assert records == []

    def test_skip_only_affects_its_own_column_not_others(
        self, tmp_path: Path,
    ) -> None:
        """Skip is column-scoped: a skip on column A does not affect
        TP verdicts on column B (or on a different source's column A)."""
        drafts = tmp_path / "column_cluster_drafts.jsonl"
        labels = tmp_path / "labels.jsonl"
        # Two columns in the same source.
        with drafts.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "source": "smoke.csv", "column": "State",
                "draft_clusters": [
                    {"kind": "text", "canonical": "CA",
                     "values": ["CA", "California"]},
                ],
            }) + "\n")
            fh.write(json.dumps({
                "source": "smoke.csv", "column": "School",
                "draft_clusters": [
                    {"kind": "text", "canonical": "ABC",
                     "values": ["ABC", "abc"]},
                ],
            }) + "\n")
        _write_labels_jsonl(
            labels,
            entries=[
                # State is skipped wholesale.
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c1"),
                 "source": "smoke.csv", "column": "State",
                 "status": "skip", "annotator": "k"},
                # School has a TP — must survive.
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "School", "c1"),
                 "source": "smoke.csv", "column": "School",
                 "status": "tp", "annotator": "k"},
            ],
        )
        records = synthesize_records_from_drafts_and_labels(drafts, labels)
        assert len(records) == 1
        assert records[0].column == "School"

    def test_skip_supersedes_via_latest_label_wins(
        self, tmp_path: Path,
    ) -> None:
        """If the operator first FP'd a draft then later SKIP'd it,
        the latest-wins rule promotes it to skip and drops the column."""
        drafts = tmp_path / "column_cluster_drafts.jsonl"
        labels = tmp_path / "labels.jsonl"
        _write_drafts_jsonl(
            drafts,
            source="smoke.csv", column="State",
            clusters=[
                {"kind": "text", "canonical": "CA",
                 "values": ["CA", "California"]},
                {"kind": "casing", "canonical": None,
                 "values": ["nyc", "NYC"]},
            ],
        )
        c2_tid = _column_cluster_target_id("smoke.csv", "State", "c2")
        _write_labels_jsonl(
            labels,
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c1"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp", "annotator": "k"},
                # Earlier verdict on c2 was fp.
                {"target": "column_cluster", "target_id": c2_tid,
                 "source": "smoke.csv", "column": "State",
                 "status": "fp", "annotator": "k"},
                # Operator changed their mind: this column is unusable.
                {"target": "column_cluster", "target_id": c2_tid,
                 "source": "smoke.csv", "column": "State",
                 "status": "skip", "annotator": "k"},
            ],
        )
        records = synthesize_records_from_drafts_and_labels(drafts, labels)
        # c2's latest is skip -> column State is excluded entirely.
        assert records == []


class TestLoadRecordsForExport:
    """End-to-end: ``load_records_for_export`` merges board-finalised
    records with per-cluster-labelled drafts, with the board taking
    precedence on overlapping (source, column) pairs."""

    def test_drafts_only_path(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        drafts = run_dir / "column_cluster_drafts.jsonl"
        labels = run_dir / "labels.jsonl"
        _write_drafts_jsonl(
            drafts,
            source="smoke.csv", column="State",
            clusters=[
                {"kind": "text", "canonical": "California",
                 "values": ["CA", "California"]},
            ],
        )
        _write_labels_jsonl(
            labels,
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c1"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp", "annotator": "k"},
            ],
        )
        records = load_records_for_export(run_dir)
        assert len(records) == 1
        assert records[0].source == "smoke.csv"
        assert records[0].column == "State"

    def test_board_record_supersedes_drafts_for_same_column(
        self, tmp_path: Path,
    ) -> None:
        """Column-board save for (source, column) hides any draft-
        based synthesised records for that same pair so cells aren't
        double-counted in the benchmark."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Board-saved record for (smoke.csv, State).
        store = ColumnClusterStore(run_dir / "column_clusters.jsonl")
        store.append(_make_state_record())
        # Drafts + labels for the same column.
        _write_drafts_jsonl(
            run_dir / "column_cluster_drafts.jsonl",
            source="smoke.csv", column="State",
            clusters=[
                {"kind": "casing", "canonical": "CA",
                 "values": ["CA", "ca", "Ca"]},
            ],
        )
        _write_labels_jsonl(
            run_dir / "labels.jsonl",
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c1"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp", "annotator": "k"},
            ],
        )
        records = load_records_for_export(run_dir)
        # Only the board record; the synthesised draft for the same
        # (source, column) is filtered out.
        assert len(records) == 1
        assert records[0].annotator == "reviewer"
        # Three values from the board record, NOT the draft's three.
        values = {v.value for v in records[0].clusters[0].values}
        assert values == {"CA", "Calif.", "California"}

    def test_board_and_drafts_for_different_columns_both_load(
        self, tmp_path: Path,
    ) -> None:
        """Board record for column A + per-cluster-labelled drafts
        for column B both contribute to the export."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        store = ColumnClusterStore(run_dir / "column_clusters.jsonl")
        store.append(_make_state_record())  # State column
        # Drafts + label for a DIFFERENT column.
        _write_drafts_jsonl(
            run_dir / "column_cluster_drafts.jsonl",
            source="smoke.csv", column="School",
            clusters=[
                {"kind": "text", "canonical": "ABC",
                 "values": ["ABC", "ABC HS", "abc"]},
            ],
        )
        _write_labels_jsonl(
            run_dir / "labels.jsonl",
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "School", "c1"),
                 "source": "smoke.csv", "column": "School",
                 "status": "tp", "annotator": "k"},
            ],
        )
        records = load_records_for_export(run_dir)
        assert len(records) == 2
        cols = sorted(r.column for r in records)
        assert cols == ["School", "State"]


class TestExportFromDraftsViaCli:
    """End-to-end CLI test: per-cluster verdicts in labels.jsonl
    drive the per-cell benchmark when no column_clusters.jsonl
    exists."""

    def _build_drafts_only_run(self, tmp_path: Path) -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rows = ["California"] * 4 + ["CA"] * 3 + ["Calif."] * 3
        pd.DataFrame({"State": rows}).to_csv(
            data_dir / "smoke.csv", index=False,
        )
        run_dir = tmp_path / "outputs" / "smoke_drafts"
        run_dir.mkdir(parents=True)
        _write_drafts_jsonl(
            run_dir / "column_cluster_drafts.jsonl",
            source="smoke.csv", column="State",
            clusters=[
                {"kind": "text", "canonical": "California",
                 "values": ["CA", "Calif.", "California"]},
            ],
        )
        _write_labels_jsonl(
            run_dir / "labels.jsonl",
            entries=[
                {"target": "column_cluster",
                 "target_id": _column_cluster_target_id(
                     "smoke.csv", "State", "c1"),
                 "source": "smoke.csv", "column": "State",
                 "status": "tp", "annotator": "reviewer"},
            ],
        )
        return tmp_path

    def test_cli_falls_back_to_drafts_when_no_record_file(
        self, tmp_path: Path,
    ) -> None:
        ws = self._build_drafts_only_run(tmp_path)
        label_dir = ws / "label_result"
        proc = subprocess.run(
            [
                sys.executable, "-m",
                "experiments.export_inconsistency_benchmark",
                "--run-dir", str(ws / "outputs" / "smoke_drafts"),
                "--data-dir", str(ws / "data"),
                "--label-result-dir", str(label_dir),
                "--tag", "drafts",
                "--timestamp", "20260512T010000",
            ],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        out_path = label_dir / (
            "benchmark_inconsistency_20260512T010000Z_drafts.jsonl"
        )
        assert out_path.exists(), proc.stderr
        cells = [
            json.loads(line) for line in out_path.read_text().splitlines()
        ]
        # All 10 cells in the column belong to the labelled cluster.
        assert len(cells) == 10
        for c in cells:
            assert isinstance(c["corrected_value"], list)
