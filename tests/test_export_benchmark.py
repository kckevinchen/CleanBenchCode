"""Tests for ``src.labeling.export.materialize_label_benchmark``.

Focus: the new label-side ``columns`` / ``corrected_values`` fields and
back-compat with old labels that only carry the scalar ``column`` /
``corrected_value`` fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.labeling.export import materialize_label_benchmark
from src.labeling.schema import Label, LabelStatus, LabelTarget
from src.labeling.store import LabelStore


def _label(
    *,
    target_id: str,
    column: str | None,
    columns: list[str] | None = None,
    status: LabelStatus = LabelStatus.TP,
    corrected_value: str | None = None,
    corrected_values: dict[str, str] | None = None,
) -> Label:
    return Label(
        target=LabelTarget.CLUSTER,
        target_id=target_id,
        source="a.csv",
        row=0,
        column=column,
        columns=columns or [],
        status=status,
        corrected_value=corrected_value,
        corrected_values=corrected_values or {},
        annotator="tester",
    )


def _cluster(cid: str, columns: list[str], **extra) -> dict:
    base = {
        "cluster_id": cid,
        "source": "a.csv",
        "row": 0,
        "columns": columns,
        "row_values": {c: f"val_{c}" for c in columns},
        "error_types": ["typo"],
        "detections": [],
    }
    base.update(extra)
    return base


@pytest.fixture
def store(tmp_path: Path) -> LabelStore:
    return LabelStore(tmp_path / "labels.jsonl")


class TestColumnsField:
    def test_label_columns_overrides_cluster_columns(self, store: LabelStore) -> None:
        # Cluster currently advertises 3 columns, but the operator's
        # frozen verdict was on just one. Export should respect the
        # label's frozen scope, not the (potentially re-clustered)
        # cluster's current columns.
        cluster = _cluster("c1", ["a", "b", "c"])
        store.append(_label(
            target_id="c1",
            column="a",
            columns=["a"],
            status=LabelStatus.TP,
            corrected_value="fixed",
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert {c.column for c in cells} == {"a"}
        assert cells[0].label == "error"

    def test_falls_back_to_cluster_columns_when_label_columns_empty(
        self, store: LabelStore,
    ) -> None:
        # Old-format label (no ``columns`` set) - exporter falls back to
        # the cluster's columns so existing JSONL still materialises.
        cluster = _cluster("c1", ["a", "b"])
        store.append(_label(
            target_id="c1",
            column="a",
            columns=[],         # legacy label
            status=LabelStatus.TP,
            corrected_value="fixed",
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert {c.column for c in cells} == {"a", "b"}


class TestCorrectedValuesField:
    def test_per_column_corrections_applied(self, store: LabelStore) -> None:
        cluster = _cluster("c1", ["first_name", "last_name"])
        store.append(_label(
            target_id="c1",
            column="first_name",
            columns=["first_name", "last_name"],
            status=LabelStatus.TP,
            corrected_value=None,
            corrected_values={"first_name": "Alice", "last_name": "Smith"},
        ))
        cells = materialize_label_benchmark([cluster], store)
        by_col = {c.column: c.corrected_value for c in cells}
        assert by_col == {"first_name": "Alice", "last_name": "Smith"}

    def test_falls_back_to_scalar_corrected_value(self, store: LabelStore) -> None:
        # No corrected_values map => every column gets the scalar.
        cluster = _cluster("c1", ["a"])
        store.append(_label(
            target_id="c1",
            column="a",
            columns=["a"],
            status=LabelStatus.TP,
            corrected_value="fixed",
            corrected_values={},
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert cells[0].corrected_value == "fixed"

    def test_corrected_value_ignored_when_verdict_is_clean(
        self, store: LabelStore,
    ) -> None:
        cluster = _cluster("c1", ["a"])
        store.append(_label(
            target_id="c1",
            column="a",
            columns=["a"],
            status=LabelStatus.FP,
            corrected_value="ignored",
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert cells[0].label == "clean"
        assert cells[0].corrected_value is None


class TestSingleColumnRoundtrip:
    """The single-column PoC path: one column per cluster, scalar
    corrected_value, ``columns=[column]`` and
    ``corrected_values={column: corrected_value}`` mirrored from the UI."""

    def test_roundtrip_matches_legacy_output(self, store: LabelStore) -> None:
        cluster = _cluster("c1", ["name"])
        store.append(_label(
            target_id="c1",
            column="name",
            columns=["name"],
            status=LabelStatus.TP,
            corrected_value="Alice",
            corrected_values={"name": "Alice"},
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert len(cells) == 1
        assert cells[0].column == "name"
        assert cells[0].label == "error"
        assert cells[0].corrected_value == "Alice"


class TestRowLevelFindings:
    """Row-level error types (e.g. ``row_duplication``) cluster with
    ``columns=[]`` because the verdict applies to the whole row, not
    any specific cell. The exporter must still emit one record per
    such cluster, using ``column=""`` as the row-level sentinel."""

    def test_row_level_cluster_emits_sentinel_record(
        self, store: LabelStore,
    ) -> None:
        cluster = _cluster("c1", [], error_types=["row_duplication"])
        store.append(_label(
            target_id="c1",
            column=None,
            columns=[],
            status=LabelStatus.TP,
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert len(cells) == 1
        assert cells[0].column == ""
        assert cells[0].label == "error"
        assert cells[0].error_types == ["row_duplication"]
        assert cells[0].cluster_ids == ["c1"]


# ---------------------------------------------------------------------------
# Per-error-type bundle export
# ---------------------------------------------------------------------------


import json
import zipfile

from src.labeling.export import (
    UNCATEGORIZED_BUCKET,
    materialize_label_benchmark_by_error_type,
    write_benchmark_by_error_type,
)
from src.labeling.schema import Label, LabelStatus, LabelTarget


class TestPerErrorTypeBundle:
    def test_buckets_split_by_error_type(self, store: LabelStore) -> None:
        # Two clusters, three rows total, two distinct error types.
        cluster_typo = _cluster(
            "c1", ["name"], error_types=["typo"],
        )
        cluster_inc = {
            **_cluster("c2", ["city"], error_types=["pattern_inconsistency"]),
            "row": 1,
        }
        store.append(_label(
            target_id="c1", column="name", columns=["name"],
            status=LabelStatus.TP, corrected_value="Alice",
        ))
        store.append(Label(
            target=LabelTarget.CLUSTER, target_id="c2",
            source="a.csv", row=1, column="city", columns=["city"],
            status=LabelStatus.TP, corrected_value="Miami",
            annotator="tester",
        ))

        buckets = materialize_label_benchmark_by_error_type(
            [cluster_typo, cluster_inc], store,
        )
        assert set(buckets.keys()) == {"typo", "pattern_inconsistency"}
        assert len(buckets["typo"]) == 1
        assert buckets["typo"][0].column == "name"
        assert len(buckets["pattern_inconsistency"]) == 1
        assert buckets["pattern_inconsistency"][0].column == "city"

    def test_cell_with_multiple_error_types_is_replicated(
        self, store: LabelStore,
    ) -> None:
        cluster = _cluster(
            "c1", ["school"],
            error_types=["typo", "pattern_inconsistency"],
        )
        store.append(_label(
            target_id="c1", column="school", columns=["school"],
            status=LabelStatus.TP, corrected_value="Bremen Public Schools",
        ))
        buckets = materialize_label_benchmark_by_error_type([cluster], store)
        # Same cell appears in both buckets - downstream cleaners want
        # "every typo" and "every pattern_inconsistency" independently.
        assert "typo" in buckets and "pattern_inconsistency" in buckets
        assert len(buckets["typo"]) == 1
        assert len(buckets["pattern_inconsistency"]) == 1
        assert buckets["typo"][0].column == "school"
        assert buckets["pattern_inconsistency"][0].column == "school"

    def test_fp_and_fn_cells_with_no_error_types_go_to_uncategorized(
        self, store: LabelStore,
    ) -> None:
        # FP cluster - error_types still populated on the cluster
        # but the benchmark cell will carry them too. To exercise
        # the uncategorized bucket cleanly, use an empty error_types
        # cluster + an FN cell label.
        cluster_fp_no_types = _cluster("c1", ["x"], error_types=[])
        store.append(_label(
            target_id="c1", column="x", columns=["x"],
            status=LabelStatus.FP,
        ))
        # FN cell: no enclosing cluster, no error_type info available.
        store.append(Label(
            target=LabelTarget.CELL,
            target_id="cell:a.csv:5:y",
            source="a.csv", row=5, column="y",
            status=LabelStatus.FN,
            corrected_value="real value",
            annotator="tester",
        ))
        buckets = materialize_label_benchmark_by_error_type(
            [cluster_fp_no_types], store,
        )
        assert UNCATEGORIZED_BUCKET in buckets
        # Both the FP cell (no types on cluster) and the FN cell
        # land in _uncategorized.
        cells = buckets[UNCATEGORIZED_BUCKET]
        cols = sorted(c.column for c in cells)
        assert cols == ["x", "y"]

    def test_writes_one_jsonl_per_bucket(
        self, store: LabelStore, tmp_path: Path,
    ) -> None:
        cluster = _cluster(
            "c1", ["school"],
            error_types=["typo", "pattern_inconsistency"],
        )
        store.append(_label(
            target_id="c1", column="school", columns=["school"],
            status=LabelStatus.TP, corrected_value="x",
        ))
        out_dir = tmp_path / "by_type"
        written = write_benchmark_by_error_type(
            out_dir, [cluster], store,
        )
        assert set(written.keys()) == {"typo", "pattern_inconsistency"}
        for path in written.values():
            assert path.exists()
            lines = [
                json.loads(L) for L in path.read_text(encoding="utf-8").splitlines()
                if L.strip()
            ]
            assert len(lines) == 1
            assert lines[0]["column"] == "school"
            assert lines[0]["label"] == "error"

    def test_skipped_rows_are_excluded_from_all_buckets(
        self, store: LabelStore,
    ) -> None:
        # Two clusters on the same row; one is TP, one is SKIP. The
        # skip cascades and excludes the entire row from the benchmark
        # (existing semantics) - per-type export must respect that.
        cluster_tp = _cluster(
            "c1", ["name"], error_types=["typo"],
        )
        cluster_skip = {
            **_cluster("c2", ["city"], error_types=["pattern_inconsistency"]),
        }
        store.append(_label(
            target_id="c1", column="name", columns=["name"],
            status=LabelStatus.TP, corrected_value="Alice",
        ))
        store.append(_label(
            target_id="c2", column="city", columns=["city"],
            status=LabelStatus.SKIP,
        ))
        buckets = materialize_label_benchmark_by_error_type(
            [cluster_tp, cluster_skip], store,
        )
        assert buckets == {}

    def test_endpoint_returns_zip_with_per_type_files(
        self, store: LabelStore, tmp_path: Path,
    ) -> None:
        # Smoke-test the on-disk shape: a ZIP with one entry per
        # bucket, each readable as JSONL.
        cluster_a = _cluster("c1", ["name"], error_types=["typo"])
        cluster_b = {
            **_cluster("c2", ["city"], error_types=["pattern_inconsistency"]),
            "row": 1,
        }
        store.append(_label(
            target_id="c1", column="name", columns=["name"],
            status=LabelStatus.TP, corrected_value="Alice",
        ))
        store.append(Label(
            target=LabelTarget.CLUSTER, target_id="c2",
            source="a.csv", row=1, column="city", columns=["city"],
            status=LabelStatus.TP, corrected_value="Miami",
            annotator="tester",
        ))
        out_dir = tmp_path / "by_type"
        write_benchmark_by_error_type(out_dir, [cluster_a, cluster_b], store)

        # Mimic the endpoint: zip the directory.
        import os
        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.glob("*.jsonl")):
                zf.write(p, arcname=p.name)
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(zf.namelist())
            assert names == ["pattern_inconsistency.jsonl", "typo.jsonl"]
            with zf.open("typo.jsonl") as fh:
                rec = json.loads(fh.read().decode("utf-8").splitlines()[0])
                assert rec["column"] == "name"
                assert rec["corrected_value"] == "Alice"

    def test_non_canonical_error_type_is_slug_safe(
        self, store: LabelStore,
    ) -> None:
        # Defensive: a custom detector emits a non-canonical
        # error_type with characters that wouldn't be filename-safe.
        cluster = _cluster(
            "c1", ["x"],
            error_types=["weird/type with spaces"],
        )
        store.append(_label(
            target_id="c1", column="x", columns=["x"],
            status=LabelStatus.TP, corrected_value="ok",
        ))
        buckets = materialize_label_benchmark_by_error_type([cluster], store)
        # The bucket key is sanitised but the data is preserved.
        assert len(buckets) == 1
        slug = next(iter(buckets.keys()))
        assert "/" not in slug and " " not in slug
        # The original error_type string is preserved on the cell.
        assert buckets[slug][0].error_types == ["weird/type with spaces"]


class TestValueResolverPopulatesOriginalValue:
    """``original_value`` must be populated for every benchmark row,
    regardless of whether the cluster carries a row_values snapshot.
    A ``value_resolver`` (typically backed by the live source CSV)
    is consulted first; cluster row_values is only used as a fallback
    when the resolver returns ``None``.

    Motivation: same-value FN propagation (and cluster pruning) can
    leave clusters with empty/sparse ``row_values``. Downstream
    consumers of an exported benchmark should not have to re-load
    the source CSV to recover the cell's original value.
    """

    def test_resolver_fills_in_empty_row_values(
        self, store: LabelStore,
    ) -> None:
        # Cluster has empty row_values - mimics propagated/pruned
        # clusters seen in real runs.
        cluster = _cluster("c1", ["SAU Name"])
        cluster["row_values"] = {}
        store.append(_label(
            target_id="c1", column="SAU Name", columns=["SAU Name"],
            status=LabelStatus.FP,
        ))

        # Resolver returns the canonical CSV value.
        resolver_calls = []

        def resolver(src: str, row: int, col: str):
            resolver_calls.append((src, row, col))
            return "Education in Unorganized Terr"

        cells = materialize_label_benchmark(
            [cluster], store, value_resolver=resolver,
        )
        assert len(cells) == 1
        assert cells[0].original_value == "Education in Unorganized Terr"
        assert cells[0].label == "clean"
        assert resolver_calls == [("a.csv", 0, "SAU Name")]

    def test_resolver_overrides_row_values_snapshot(
        self, store: LabelStore,
    ) -> None:
        # If row_values is stale (cluster snapshot from before a
        # workbook reload), the resolver wins so the export reflects
        # the live CSV.
        cluster = _cluster("c1", ["x"])
        cluster["row_values"] = {"x": "stale_snapshot"}
        store.append(_label(
            target_id="c1", column="x", columns=["x"],
            status=LabelStatus.FP,
        ))
        cells = materialize_label_benchmark(
            [cluster], store, value_resolver=lambda *_: "live_value",
        )
        assert cells[0].original_value == "live_value"

    def test_resolver_none_falls_back_to_row_values(
        self, store: LabelStore,
    ) -> None:
        # If the resolver returns None (e.g. source unavailable), the
        # exporter falls back to the cluster's row_values snapshot
        # before defaulting to ``""``.
        cluster = _cluster("c1", ["x"])
        cluster["row_values"] = {"x": "snapshot"}
        store.append(_label(
            target_id="c1", column="x", columns=["x"],
            status=LabelStatus.FP,
        ))
        cells = materialize_label_benchmark(
            [cluster], store, value_resolver=lambda *_: None,
        )
        assert cells[0].original_value == "snapshot"

    def test_resolver_exception_does_not_break_export(
        self, store: LabelStore,
    ) -> None:
        # Any exception from the resolver is swallowed - the export
        # path must never crash because of operator code that throws.
        cluster = _cluster("c1", ["x"])
        cluster["row_values"] = {"x": "snapshot"}
        store.append(_label(
            target_id="c1", column="x", columns=["x"],
            status=LabelStatus.FP,
        ))

        def boom(*_):
            raise RuntimeError("boom")

        cells = materialize_label_benchmark(
            [cluster], store, value_resolver=boom,
        )
        assert cells[0].original_value == "snapshot"

    def test_resolver_fills_fn_cell_original_value(
        self, store: LabelStore,
    ) -> None:
        # FN cells were never enclosed in a cluster, so they have no
        # row_values snapshot. The resolver is the only way to get
        # the live CSV value into the exported benchmark.
        store.append(Label(
            target=LabelTarget.CELL,
            target_id="cell:a.csv:5:email",
            source="a.csv", row=5, column="email",
            status=LabelStatus.FN,
            corrected_value="real@example.com",
            annotator="tester",
        ))
        cells = materialize_label_benchmark(
            [], store,
            value_resolver=lambda s, r, c: "live@example.com",
        )
        assert len(cells) == 1
        assert cells[0].original_value == "live@example.com"
        assert cells[0].corrected_value == "real@example.com"
        assert cells[0].cluster_ids == []

    def test_no_resolver_keeps_legacy_behaviour(
        self, store: LabelStore,
    ) -> None:
        # Backwards compatible: without a resolver, the exporter
        # mirrors the old behaviour (row_values snapshot, empty
        # string for FN cells).
        cluster = _cluster("c1", ["x"])
        cluster["row_values"] = {"x": "snapshot"}
        store.append(_label(
            target_id="c1", column="x", columns=["x"],
            status=LabelStatus.FP,
        ))
        store.append(Label(
            target=LabelTarget.CELL,
            target_id="cell:a.csv:5:y",
            source="a.csv", row=5, column="y",
            status=LabelStatus.FN,
            annotator="tester",
        ))
        cells = materialize_label_benchmark([cluster], store)
        by_col = {c.column: c.original_value for c in cells}
        assert by_col == {"x": "snapshot", "y": ""}


class TestLLMSuggestedValue:
    """Verify that ``llm_suggested_value`` is populated from the cluster's
    ``synthesis.canonical_fix`` (preferred) or from the longest non-empty
    per-detection ``suggested_fix`` (fallback). FN cell labels (no
    enclosing cluster) get ``None``.
    """

    def test_canonical_fix_propagates_to_every_column(
        self, store: LabelStore,
    ) -> None:
        cluster = _cluster("c1", ["a", "b"])
        cluster["synthesis"] = {"canonical_fix": "Boston"}
        store.append(_label(
            target_id="c1", column="a", columns=["a", "b"],
            status=LabelStatus.TP, corrected_value="Boston",
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert {c.column: c.llm_suggested_value for c in cells} == {
            "a": "Boston", "b": "Boston",
        }

    def test_falls_back_to_longest_detection_suggested_fix_per_column(
        self, store: LabelStore,
    ) -> None:
        # No synthesis.canonical_fix; expect column-keyed fallback to
        # the detection's suggested_fix.
        cluster = _cluster("c1", ["a", "b"])
        cluster["detections"] = [
            {"issue_id": "i1", "columns": ["a"], "suggested_fix": "Boston"},
            {"issue_id": "i2", "columns": ["b"], "suggested_fix": "MA"},
        ]
        store.append(_label(
            target_id="c1", column="a", columns=["a", "b"],
            status=LabelStatus.TP, corrected_value="Boston",
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert {c.column: c.llm_suggested_value for c in cells} == {
            "a": "Boston", "b": "MA",
        }

    def test_longest_wins_when_multiple_detections_target_same_column(
        self, store: LabelStore,
    ) -> None:
        cluster = _cluster("c1", ["a"])
        cluster["detections"] = [
            {"issue_id": "i1", "columns": ["a"], "suggested_fix": "Bos"},
            {"issue_id": "i2", "columns": ["a"], "suggested_fix": "Boston"},
        ]
        store.append(_label(
            target_id="c1", column="a", columns=["a"],
            status=LabelStatus.TP, corrected_value="Boston",
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert cells[0].llm_suggested_value == "Boston"

    def test_null_or_empty_suggestions_yield_none(
        self, store: LabelStore,
    ) -> None:
        cluster = _cluster("c1", ["a"])
        cluster["synthesis"] = {"canonical_fix": None}
        cluster["detections"] = [
            {"issue_id": "i1", "columns": ["a"], "suggested_fix": ""},
            {"issue_id": "i2", "columns": ["a"], "suggested_fix": "null"},
        ]
        store.append(_label(
            target_id="c1", column="a", columns=["a"],
            status=LabelStatus.FP,
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert cells[0].llm_suggested_value is None

    def test_fp_clusters_still_record_llm_suggestion(
        self, store: LabelStore,
    ) -> None:
        # An FP verdict means the operator REJECTED the LLM's claim,
        # but we still record what the LLM proposed so downstream
        # consumers can A/B compare human vs LLM corrections without
        # re-loading clusters.json.
        cluster = _cluster("c1", ["a"])
        cluster["synthesis"] = {"canonical_fix": "Boston"}
        store.append(_label(
            target_id="c1", column="a", columns=["a"],
            status=LabelStatus.FP,
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert cells[0].label == "clean"
        assert cells[0].corrected_value is None
        assert cells[0].llm_suggested_value == "Boston"

    def test_pure_fn_cell_has_none(self, store: LabelStore) -> None:
        # Cell-level FN labels are filed without an enclosing cluster,
        # so there is no LLM suggestion to record.
        store.append(Label(
            target=LabelTarget.CELL,
            target_id="cell:a.csv:5:x",
            source="a.csv", row=5, column="x",
            status=LabelStatus.FN, corrected_value="Boston",
            annotator="tester",
        ))
        cells = materialize_label_benchmark([], store)
        assert len(cells) == 1
        assert cells[0].label == "error"
        assert cells[0].corrected_value == "Boston"
        assert cells[0].llm_suggested_value is None

    def test_conflict_takes_longest_suggestion_across_clusters(
        self, store: LabelStore,
    ) -> None:
        # Two clusters touching the same cell; both confirmed TP. The
        # exported cell should record the longer of the two LLM
        # suggestions (same heuristic the auto-synthesiser uses).
        c1 = _cluster("c1", ["a"])
        c1["synthesis"] = {"canonical_fix": "Bos"}
        c2 = _cluster("c2", ["a"])
        c2["synthesis"] = {"canonical_fix": "Boston"}
        store.append(_label(
            target_id="c1", column="a", columns=["a"],
            status=LabelStatus.TP, corrected_value="Boston",
        ))
        store.append(_label(
            target_id="c2", column="a", columns=["a"],
            status=LabelStatus.TP, corrected_value="Boston",
        ))
        cells = materialize_label_benchmark([c1, c2], store)
        assert len(cells) == 1
        assert cells[0].llm_suggested_value == "Boston"


