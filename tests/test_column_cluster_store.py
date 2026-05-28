"""Tests for ``src.labeling.column_cluster_store.ColumnClusterStore``
and the validators on the related Pydantic schemas.

Pinned by Slice 1 of the column-clusters labelling subsystem.
Mirrors ``tests/test_label_store.py`` style for the per-cluster store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.labeling.column_cluster_store import ColumnClusterStore
from src.labeling.schema import (
    ClusterKind,
    ClusterValue,
    ColumnClusters,
    ColumnClustersIn,
    ColumnLabelStatus,
    StdCluster,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cluster(cluster_id: str, *values_counts, canonical: str | None = None,
             kind: ClusterKind = ClusterKind.UNTAGGED) -> StdCluster:
    return StdCluster(
        cluster_id=cluster_id,
        kind=kind,
        canonical_value=canonical,
        values=[ClusterValue(value=v, count=c) for v, c in values_counts],
    )


def _record(source: str, column: str, *clusters: StdCluster,
            status: ColumnLabelStatus = ColumnLabelStatus.LABELED,
            annotator: str = "tester", note: str = "",
            drop_reason: str = "") -> ColumnClusters:
    return ColumnClusters(
        source=source, column=column, status=status,
        drop_reason=drop_reason, clusters=list(clusters),
        annotator=annotator, note=note,
    )


# ---------------------------------------------------------------------------
# Schema invariants (5 + adjuncts)
# ---------------------------------------------------------------------------


class TestSchemaInvariants:
    def test_single_value_cluster_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 2"):
            StdCluster(
                cluster_id="c1",
                values=[ClusterValue(value="A", count=1)],
            )

    def test_intra_cluster_duplicate_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate value"):
            StdCluster(
                cluster_id="c1",
                values=[
                    ClusterValue(value="A", count=1),
                    ClusterValue(value="A", count=2),
                ],
            )

    def test_canonical_not_in_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="canonical_value"):
            StdCluster(
                cluster_id="c1",
                canonical_value="Z",
                values=[
                    ClusterValue(value="A", count=2),
                    ClusterValue(value="B", count=2),
                ],
            )

    def test_canonical_in_values_accepted(self) -> None:
        sc = StdCluster(
            cluster_id="c1", canonical_value="B",
            values=[ClusterValue(value="A", count=2),
                    ClusterValue(value="B", count=5)],
        )
        assert sc.canonical_value == "B"

    def test_disjoint_clusters_required(self) -> None:
        with pytest.raises(ValueError, match="appears in two clusters"):
            ColumnClustersIn(
                source="s.csv", column="Name",
                status=ColumnLabelStatus.LABELED,
                clusters=[
                    _cluster("c1", ("A", 2), ("B", 2)),
                    _cluster("c2", ("A", 2), ("C", 2)),
                ],
            )

    def test_unique_cluster_ids_required(self) -> None:
        with pytest.raises(ValueError, match="duplicate cluster_id"):
            ColumnClustersIn(
                source="s.csv", column="Name",
                status=ColumnLabelStatus.LABELED,
                clusters=[
                    _cluster("c1", ("A", 2), ("B", 2)),
                    _cluster("c1", ("X", 2), ("Y", 2)),
                ],
            )

    def test_labeled_requires_clusters(self) -> None:
        with pytest.raises(ValueError, match="labeled requires"):
            ColumnClustersIn(
                source="s.csv", column="Name",
                status=ColumnLabelStatus.LABELED,
                clusters=[],
            )

    @pytest.mark.parametrize(
        "status",
        [
            ColumnLabelStatus.NO_INCONSISTENCY,
            ColumnLabelStatus.DROPPED_OVERLAPPING,
            ColumnLabelStatus.DROPPED_MESSY,
            ColumnLabelStatus.UNLABELED,
        ],
    )
    def test_non_labeled_must_be_empty(self, status: ColumnLabelStatus) -> None:
        with pytest.raises(ValueError, match="requires no clusters"):
            ColumnClustersIn(
                source="s.csv", column="Name",
                status=status,
                clusters=[_cluster("c1", ("A", 2), ("B", 2))],
            )

    def test_no_inconsistency_empty_accepted(self) -> None:
        rec = ColumnClustersIn(
            source="s.csv", column="Name",
            status=ColumnLabelStatus.NO_INCONSISTENCY,
        )
        assert rec.clusters == []

    def test_kind_defaults_to_untagged(self) -> None:
        sc = _cluster("c1", ("A", 1), ("B", 1))
        assert sc.kind == ClusterKind.UNTAGGED


# ---------------------------------------------------------------------------
# Store: append + reload + history
# ---------------------------------------------------------------------------


class TestAppendAndReload:
    def test_append_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "column_clusters.jsonl"
        s1 = ColumnClusterStore(path)
        s1.append(_record("a.csv", "X", _cluster("c1", ("a", 2), ("A", 1))))
        s1.append(_record("a.csv", "Y", _cluster("c1", ("p", 3), ("P", 2))))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        s2 = ColumnClusterStore(path)
        keys = sorted((r.source, r.column) for r in s2.all())
        assert keys == [("a.csv", "X"), ("a.csv", "Y")]

    def test_latest_wins_on_column_collision(self, tmp_path: Path) -> None:
        path = tmp_path / "column_clusters.jsonl"
        s = ColumnClusterStore(path)
        s.append(_record(
            "a.csv", "X", _cluster("c1", ("a", 2), ("A", 1)),
            note="first",
        ))
        s.append(_record(
            "a.csv", "X", _cluster("c1", ("a", 2), ("A", 1)),
            _cluster("c2", ("b", 3), ("B", 2)),
            note="second",
        ))
        latest = s.latest_for("a.csv", "X")
        assert latest is not None
        assert latest.note == "second"
        assert len(latest.clusters) == 2
        hist = s.history("a.csv", "X")
        assert [r.note for r in hist] == ["first", "second"]

    def test_overwrite_to_no_inconsistency(self, tmp_path: Path) -> None:
        """User may flip a previously-labelled column to no_inconsistency
        by writing a new record with empty clusters."""
        path = tmp_path / "column_clusters.jsonl"
        s = ColumnClusterStore(path)
        s.append(_record("a.csv", "X", _cluster("c1", ("a", 2), ("A", 1))))
        s.append(_record(
            "a.csv", "X",
            status=ColumnLabelStatus.NO_INCONSISTENCY,
        ))
        latest = s.latest_for("a.csv", "X")
        assert latest is not None
        assert latest.status == ColumnLabelStatus.NO_INCONSISTENCY
        assert latest.clusters == []
        # History keeps both.
        assert len(s.history("a.csv", "X")) == 2

    def test_creates_parent_directory_on_first_append(
        self, tmp_path: Path,
    ) -> None:
        nested = tmp_path / "deep" / "down" / "column_clusters.jsonl"
        s = ColumnClusterStore(nested)
        s.append(_record("a.csv", "X", _cluster("c1", ("a", 1), ("A", 1))))
        assert nested.is_file()
        assert s.latest_for("a.csv", "X") is not None

    def test_history_filters_by_column(self, tmp_path: Path) -> None:
        s = ColumnClusterStore(tmp_path / "column_clusters.jsonl")
        s.append(_record("a.csv", "X", _cluster("c1", ("a", 1), ("A", 1)), note="a"))
        s.append(_record("a.csv", "Y", _cluster("c1", ("b", 1), ("B", 1)), note="b"))
        s.append(_record("a.csv", "X", _cluster("c1", ("a", 1), ("A", 1)), note="c"))
        notes = [r.note for r in s.history("a.csv", "X")]
        assert notes == ["a", "c"]

    def test_iter_returns_all_records(self, tmp_path: Path) -> None:
        s = ColumnClusterStore(tmp_path / "column_clusters.jsonl")
        s.append(_record("a.csv", "X", _cluster("c1", ("a", 1), ("A", 1))))
        s.append(_record("b.csv", "Y", _cluster("c1", ("b", 1), ("B", 1))))
        keys = sorted((r.source, r.column) for r in iter(s))
        assert keys == [("a.csv", "X"), ("b.csv", "Y")]

    def test_latest_by_column_returns_dict(self, tmp_path: Path) -> None:
        s = ColumnClusterStore(tmp_path / "column_clusters.jsonl")
        s.append(_record("a.csv", "X", _cluster("c1", ("a", 1), ("A", 1))))
        s.append(_record("a.csv", "Y", _cluster("c1", ("b", 1), ("B", 1))))
        d = s.latest_by_column()
        assert set(d.keys()) == {("a.csv", "X"), ("a.csv", "Y")}


# ---------------------------------------------------------------------------
# Malformed-line recovery
# ---------------------------------------------------------------------------


class TestMalformedRecovery:
    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "column_clusters.jsonl"
        s1 = ColumnClusterStore(path)
        s1.append(_record("a.csv", "X", _cluster("c1", ("a", 1), ("A", 1))))
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n\n   \n")
        s1.append(_record("a.csv", "Y", _cluster("c1", ("b", 1), ("B", 1))))
        s2 = ColumnClusterStore(path)
        keys = sorted((r.source, r.column) for r in s2.all())
        assert keys == [("a.csv", "X"), ("a.csv", "Y")]

    def test_corrupt_json_line_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "column_clusters.jsonl"
        good = _record(
            "a.csv", "X", _cluster("c1", ("a", 1), ("A", 1))
        ).model_dump_json()
        path.write_text(good + "\n{not real json}\n" + good + "\n")
        s = ColumnClusterStore(path)
        assert [(r.source, r.column) for r in s.all()] == [
            ("a.csv", "X"), ("a.csv", "X"),
        ]

    def test_schema_mismatch_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "column_clusters.jsonl"
        path.write_text('{"foo": "bar"}\n')
        s = ColumnClusterStore(path)
        assert s.all() == []

    def test_invariant_violation_in_jsonl_skipped(self, tmp_path: Path) -> None:
        """A JSONL line that parses but fails validators is dropped."""
        path = tmp_path / "column_clusters.jsonl"
        # Hand-craft a record where one value appears in two clusters.
        # We can't construct this via the model, so write raw JSON.
        bad = (
            '{"source": "a.csv", "column": "X", "status": "labeled",'
            '"drop_reason": "", "annotator": "t", "ts": "2026-01-01T00:00:00",'
            '"note": "",'
            '"clusters": ['
            ' {"cluster_id": "c1", "kind": "untagged", "canonical_value": null,'
            '  "values": [{"value": "A", "count": 2}, {"value": "B", "count": 2}]},'
            ' {"cluster_id": "c2", "kind": "untagged", "canonical_value": null,'
            '  "values": [{"value": "A", "count": 2}, {"value": "C", "count": 2}]}'
            ']}'
        )
        path.write_text(bad + "\n")
        s = ColumnClusterStore(path)
        assert s.all() == []


# ---------------------------------------------------------------------------
# JSONL round-trip via model_dump_json / model_validate_json
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_round_trip_preserves_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "column_clusters.jsonl"
        rec_in = _record(
            "a.csv", "School",
            _cluster(
                "c1", ("BBB", 47), ("ABC HS", 3),
                canonical="BBB", kind=ClusterKind.TEXT,
            ),
            _cluster(
                "c2", ("EEE", 19), ("DDD", 1),
                kind=ClusterKind.UNTAGGED,
            ),
            annotator="alice", note="reviewed twice",
        )
        s1 = ColumnClusterStore(path)
        s1.append(rec_in)
        s2 = ColumnClusterStore(path)
        rec_out = s2.latest_for("a.csv", "School")
        assert rec_out is not None
        assert rec_out.source == "a.csv"
        assert rec_out.column == "School"
        assert rec_out.status == ColumnLabelStatus.LABELED
        assert rec_out.annotator == "alice"
        assert rec_out.note == "reviewed twice"
        assert len(rec_out.clusters) == 2
        c1 = rec_out.clusters[0]
        assert c1.cluster_id == "c1"
        assert c1.canonical_value == "BBB"
        assert c1.kind == ClusterKind.TEXT
        assert [(v.value, v.count) for v in c1.values] == [("BBB", 47), ("ABC HS", 3)]
        c2 = rec_out.clusters[1]
        assert c2.cluster_id == "c2"
        assert c2.canonical_value is None
        assert c2.kind == ClusterKind.UNTAGGED


# ---------------------------------------------------------------------------
# Legacy ClusterKind compatibility.
#
# Old persisted records / draft files use the previous kind taxonomy
# (``entity_variant`` / ``template_shape`` / ``style_convention`` /
# ``encoding_convention``). The schema validator must remap them to
# the new taxonomy on load so they keep parsing while still being
# normalised to the canonical post-migration tag space.
# ---------------------------------------------------------------------------


class TestLegacyKindMapping:
    @pytest.mark.parametrize("legacy,expected", [
        ("entity_variant", ClusterKind.TEXT),
        ("template_shape", ClusterKind.PATTERN),
        ("style_convention", ClusterKind.CASING),
        ("encoding_convention", ClusterKind.NUMERICAL_FORMAT),
        ("untagged", ClusterKind.UNTAGGED),
        ("text", ClusterKind.TEXT),
        ("casing", ClusterKind.CASING),
        ("pattern", ClusterKind.PATTERN),
        ("punctuation", ClusterKind.PUNCTUATION),
        ("numerical_format", ClusterKind.NUMERICAL_FORMAT),
        ("UNKNOWN_TAG", ClusterKind.UNTAGGED),
        ("", ClusterKind.UNTAGGED),
        (None, ClusterKind.UNTAGGED),
        (42, ClusterKind.UNTAGGED),
    ])
    def test_from_string_handles_all_taxonomies(
        self, legacy: object, expected: ClusterKind,
    ) -> None:
        assert ClusterKind.from_string(legacy) is expected

    def test_std_cluster_accepts_legacy_tag_via_validator(self) -> None:
        # Pydantic field_validator(mode="before") must remap legacy
        # strings before the enum constructor sees them; otherwise
        # this would raise.
        sc = StdCluster(
            cluster_id="c1",
            kind="entity_variant",  # type: ignore[arg-type]
            canonical_value="BBB",
            values=[
                ClusterValue(value="BBB", count=5),
                ClusterValue(value="ABC HS", count=2),
            ],
        )
        assert sc.kind is ClusterKind.TEXT

    def test_std_cluster_round_trip_preserves_new_tag(self) -> None:
        # New-taxonomy strings round-trip through JSON.
        sc = StdCluster.model_validate({
            "cluster_id": "c1",
            "kind": "punctuation",
            "canonical_value": None,
            "values": [
                {"value": "a-b", "count": 3},
                {"value": "a–b", "count": 1},
            ],
        })
        assert sc.kind is ClusterKind.PUNCTUATION
