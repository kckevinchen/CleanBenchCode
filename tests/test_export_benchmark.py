"""Tests for ``src.labeling.export.materialize_label_benchmark``.

Focus: the new label-side ``columns`` / ``corrected_values`` fields and
back-compat with old labels that only carry the scalar ``column`` /
``corrected_value`` fields.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.labeling._cells import (
    _deduplicate_benchmark_cells,
    _materialize_inconsistency_cells_for_export,
)
from src.labeling._helpers import _is_low_value_fix
from src.labeling.export import (
    materialize_label_benchmark,
    materialize_skipped_cells,
)
from src.labeling.schema import BenchmarkCell, Label, LabelStatus, LabelTarget
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


class TestCrossPipelineDeduplication:
    def test_merges_same_cell_provenance(self) -> None:
        typo = BenchmarkCell(
            source="a.csv", row=2, column="name", original_value="Ailce",
            label="error", corrected_value="Alice", error_types=["typo"],
            cluster_ids=["typo-1"], annotators=["reviewer"], note="spelling",
        )
        inconsistency = BenchmarkCell(
            source="a.csv", row=2, column="name", original_value="Ailce",
            label="error", corrected_value="Alice",
            error_types=["inconsistency"], cluster_ids=["inc-1"],
            annotators=["auditor"], note="minority form",
        )

        merged = _deduplicate_benchmark_cells([typo, inconsistency])

        assert len(merged) == 1
        assert merged[0].label == "error"
        assert merged[0].corrected_value == "Alice"
        assert merged[0].error_types == ["typo", "inconsistency"]
        assert merged[0].cluster_ids == ["typo-1", "inc-1"]
        assert merged[0].annotators == ["reviewer", "auditor"]
        assert merged[0].note == "spelling | minority form"

    def test_conflicting_labels_degrade_to_ambiguous(self) -> None:
        error = BenchmarkCell(
            source="a.csv", row=2, column="name", original_value="Alice",
            label="error", corrected_value="Alicia", error_types=["typo"],
        )
        clean = BenchmarkCell(
            source="a.csv", row=2, column="name", original_value="Alice",
            label="clean", error_types=["inconsistency"], low_value=True,
        )

        merged = _deduplicate_benchmark_cells([error, clean])

        assert len(merged) == 1
        assert merged[0].label == "ambiguous"
        assert merged[0].corrected_value is None
        assert merged[0].low_value is False

    def test_inconsistency_failure_is_not_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fail(*args, **kwargs):
            raise RuntimeError("broken inconsistency cache")

        monkeypatch.setattr(
            "src.labeling._cells.materialize_inconsistency_benchmark", _fail,
        )
        ctx = SimpleNamespace(run_dir=tmp_path / "run")

        with pytest.raises(RuntimeError, match="broken inconsistency cache"):
            _materialize_inconsistency_cells_for_export(ctx)


class TestAffectedRowsPropagation:
    """One cluster verdict must propagate to every row the cluster
    covers — the primary ``row`` plus every entry in any detection's
    ``affected_rows``. Relation rules (and dup groups) collapse N
    failing rows into ONE cluster after per-incident aggregation, so
    without this fan-out the benchmark would only carry the verdict
    for the representative row and silently drop the rest."""

    def test_fans_out_to_every_affected_row_for_fd_violation(
        self, store: LabelStore,
    ) -> None:
        # FD violations fan out to all affected rows.
        cluster = _cluster(
            "c1",
            ["zip", "state"],
            error_types=["fd_violation"],
            detections=[
                {
                    "issue_id": "fd__r1#a.csv#r00005",
                    "affected_rows": [5, 23, 26, 32],
                },
            ],
            row=5,
        )
        store.append(Label(
            target=LabelTarget.CLUSTER, target_id="c1",
            source="a.csv", row=5, column=None,
            columns=["zip", "state"],
            status=LabelStatus.TP,
            annotator="tester",
        ))

        cells = materialize_label_benchmark([cluster], store)
        # 4 rows × 2 columns = 8 emitted cells.
        keys = {(c.row, c.column) for c in cells}
        assert keys == {
            (5, "zip"), (5, "state"),
            (23, "zip"), (23, "state"),
            (26, "zip"), (26, "state"),
            (32, "zip"), (32, "state"),
        }
        # Every emitted cell carries the cluster's verdict + cluster id.
        for c in cells:
            assert c.label == "error"
            assert c.cluster_ids == ["c1"]
            assert c.error_types == ["fd_violation"]

    def test_calculation_inconsistency_does_not_fan_out_to_affected_rows(
        self, store: LabelStore,
    ) -> None:
        # Calculation inconsistency types (row_calculation_inconsistency,
        # string_calculation_inconsistency) apply ONLY to the anchor row,
        # NOT to affected_rows. This differs from FD violations and row
        # duplications which do fan out.
        cluster = _cluster(
            "c1",
            ["qty", "price", "total"],
            error_types=["row_calculation_inconsistency"],
            detections=[
                {
                    "issue_id": "calc__r1#a.csv#r00005",
                    "affected_rows": [5, 23, 26, 32],
                },
            ],
            row=5,
        )
        store.append(Label(
            target=LabelTarget.CLUSTER, target_id="c1",
            source="a.csv", row=5, column=None,
            columns=["qty", "price", "total"],
            status=LabelStatus.TP,
            annotator="tester",
        ))

        cells = materialize_label_benchmark([cluster], store)
        # Only 1 row (anchor) × 3 columns = 3 emitted cells.
        keys = {(c.row, c.column) for c in cells}
        assert keys == {
            (5, "qty"), (5, "price"), (5, "total"),
        }
        # Verify no cells from affected rows are emitted.
        assert all(c.row == 5 for c in cells)
        # Every emitted cell carries the cluster's verdict + cluster id.
        for c in cells:
            assert c.label == "error"
            assert c.cluster_ids == ["c1"]
            assert c.error_types == ["row_calculation_inconsistency"]

    def test_no_affected_rows_emits_only_primary_row(
        self, store: LabelStore,
    ) -> None:
        # Singleton clusters (no detections.affected_rows) must keep
        # their pre-fan-out behaviour: one cell per column at the
        # primary row only.
        cluster = _cluster("c1", ["name"])
        store.append(_label(
            target_id="c1", column="name", columns=["name"],
            status=LabelStatus.TP, corrected_value="Alice",
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert {(c.row, c.column) for c in cells} == {(0, "name")}

    def test_primary_row_in_affected_rows_is_not_duplicated(
        self, store: LabelStore,
    ) -> None:
        # Compiled detections include the primary row in
        # ``affected_rows`` (the list is the full failing-row set);
        # the exporter must de-dup so we don't emit two cells for the
        # same (row, column).
        cluster = _cluster(
            "c1",
            ["city"],
            detections=[{"affected_rows": [0, 7, 9]}],
            row=0,
        )
        store.append(_label(
            target_id="c1", column="city", columns=["city"],
            status=LabelStatus.TP, corrected_value="Miami",
        ))
        cells = materialize_label_benchmark([cluster], store)
        assert {(c.row, c.column) for c in cells} == {
            (0, "city"), (7, "city"), (9, "city"),
        }

    def test_peer_rows_do_not_inherit_primary_row_correction(
        self, store: LabelStore,
    ) -> None:
        # Regression for cluster contamination: two per-row clusters
        # fired on overlapping ``affected_rows`` (rule-level fan-out),
        # but each cluster's ``corrected_value`` is row-specific. The
        # exporter must NOT propagate a cluster's correction to peer
        # rows -- those rows have their own cluster + label.
        c1 = _cluster(
            "c1",
            ["district"],
            row=0,
            row_values={"district": "EAST VALLEY  (SPOKANE)"},
            detections=[{"affected_rows": [0, 1]}],
        )
        c2 = _cluster(
            "c2",
            ["district"],
            row=1,
            row_values={"district": "EAST VALLEY  (YAKIMA)"},
            detections=[{"affected_rows": [0, 1]}],
        )
        store.append(_label(
            target_id="c1", column="district", columns=["district"],
            status=LabelStatus.TP, corrected_value="EAST VALLEY (SPOKANE)",
        ))
        store.append(_label(
            target_id="c2", column="district", columns=["district"],
            status=LabelStatus.TP, corrected_value="EAST VALLEY (YAKIMA)",
        ))
        cells = materialize_label_benchmark([c1, c2], store)
        by_row = {c.row: c for c in cells}
        assert by_row[0].corrected_value == "EAST VALLEY (SPOKANE)"
        assert by_row[1].corrected_value == "EAST VALLEY (YAKIMA)"

    def test_large_fanout_does_not_contaminate_peer_corrections(
        self, store: LabelStore,
    ) -> None:
        # Regression for the buggy export file
        # ``benchmark_typo_20260512T213238Z_gpt5-family (10).jsonl``:
        # a single rule fired on N rows (here 9) producing N per-row
        # clusters that ALL list the same ``affected_rows`` peer set.
        # Each cluster has its own row-specific ``corrected_value``.
        # Pre-fix, every cell ended up stamped with the FIRST cluster's
        # correction (e.g. all 9 SCHOOL/SITE rows became
        # ``Burroughs Elementary``). The exporter must keep each row's
        # own correction regardless of cluster iteration order.
        fixes = {
            33: "Burroughs Elementary",
            34: "Columbia Elementary",
            35: "Ewing Elementary",
            36: "Greenberg Elementary",
            37: "Holland Elementary",
            38: "Lincoln Elementary",
            39: "Mayfair Elementary",
            40: "Powers Elementary",
            41: "Pyle Elementary",
        }
        peer_rows = sorted(fixes.keys())
        clusters = []
        for r, fix in fixes.items():
            cluster = _cluster(
                f"c{r}",
                ["SCHOOL/SITE"],
                row=r,
                row_values={"SCHOOL/SITE": f"orig_{r}"},
                detections=[{"affected_rows": peer_rows}],
            )
            clusters.append(cluster)
            store.append(_label(
                target_id=f"c{r}", column="SCHOOL/SITE",
                columns=["SCHOOL/SITE"],
                status=LabelStatus.TP, corrected_value=fix,
            ))

        # Forward order: primary clusters in row-ascending order.
        cells = materialize_label_benchmark(clusters, store)
        by_row = {c.row: c.corrected_value for c in cells}
        assert by_row == fixes

        # Reverse order: primary clusters in row-descending order.
        # Conflict-resolution uses ``existing or corrected_for_emit``,
        # so peer-row emits (which carry ``None``) must never clobber
        # a peer's own primary-row correction regardless of which
        # cluster the exporter visits first.
        store2 = LabelStore(store.path.parent / "labels_rev.jsonl")
        for r, fix in fixes.items():
            store2.append(_label(
                target_id=f"c{r}", column="SCHOOL/SITE",
                columns=["SCHOOL/SITE"],
                status=LabelStatus.TP, corrected_value=fix,
            ))
        cells_rev = materialize_label_benchmark(
            list(reversed(clusters)), store2,
        )
        by_row_rev = {c.row: c.corrected_value for c in cells_rev}
        assert by_row_rev == fixes

    def test_overbroad_error_fanout_does_not_make_peer_ambiguous(
        self, store: LabelStore,
    ) -> None:
        # Regression for the phantom "ambiguous" verdict on
        # ``f5f0334b.csv`` row 2 (run 20260601T193136Z): a TP cluster
        # anchored at row 10 carried an aggregate "examples: ..."
        # detection whose over-broad ``affected_rows`` fanned its
        # "error" verdict onto peer rows. Those peers have their OWN
        # FP (clean) anchor cluster -- but the clean anchor's verdict
        # MATCHED other clean fan-outs first, so the cell was never
        # marked authoritative and the later error fan-out downgraded
        # it to "ambiguous". The peer's own anchor must win: "clean".
        peers = [2, 5, 16]
        clean_clusters = []
        for r in peers:
            cl = _cluster(
                f"clean{r}",
                ["VENDOR NAME"],
                row=r,
                row_values={"VENDOR NAME": f"vendor_{r}"},
                # Each clean cluster also advertises an aggregate
                # detection that fans onto its clean peers.
                detections=[{"affected_rows": peers}],
            )
            clean_clusters.append(cl)
            store.append(_label(
                target_id=f"clean{r}", column="VENDOR NAME",
                columns=["VENDOR NAME"], status=LabelStatus.FP,
            ))
        # TP cluster anchored elsewhere whose aggregate detection
        # over-reaches onto rows 2 and 16 (but NOT row 5).
        err = _cluster(
            "err10",
            ["VENDOR NAME"],
            row=10,
            row_values={"VENDOR NAME": "XEROX BUSINESS  SOLU"},
            detections=[{"affected_rows": [2, 10, 16]}],
        )
        store.append(_label(
            target_id="err10", column="VENDOR NAME",
            columns=["VENDOR NAME"], status=LabelStatus.TP,
        ))

        cells = materialize_label_benchmark(clean_clusters + [err], store)
        by_row = {c.row: c.label for c in cells}
        assert by_row[2] == "clean"
        assert by_row[5] == "clean"
        assert by_row[16] == "clean"
        assert by_row[10] == "error"


# ---------------------------------------------------------------------------
# Per-error-type bundle export
# ---------------------------------------------------------------------------


from src.labeling.export import (
    UNCATEGORIZED_BUCKET,
    materialize_label_benchmark_by_error_type,
    write_benchmark_by_error_type,
)


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

    def test_skip_does_not_drop_sibling_clusters_on_same_row(
        self, store: LabelStore,
    ) -> None:
        # Two clusters on the same row; one is TP, one is SKIP. The
        # skip applies to its own cluster only - the sibling TP must
        # still appear in the exported benchmark. Row- or CSV-wide
        # removal is the exclusions API's job, not the export's.
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
        # TP survives, SKIP is silently dropped.
        assert set(buckets) == {"typo"}
        rows = buckets["typo"]
        assert len(rows) == 1
        assert rows[0].column == "name"
        assert rows[0].label == "error"

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

    def test_latest_unsure_cell_claim_exports_as_ambiguous(
        self, store: LabelStore,
    ) -> None:
        target_id = "cell:a.csv:5:duration"
        store.append(Label(
            target=LabelTarget.CELL,
            target_id=target_id,
            source="a.csv", row=5, column="duration",
            status=LabelStatus.FN,
            annotator="tester",
        ))
        store.append(Label(
            target=LabelTarget.CELL,
            target_id=target_id,
            source="a.csv", row=5, column="duration",
            status=LabelStatus.UNSURE,
            annotator="tester",
        ))

        cells = materialize_label_benchmark(
            [], store, value_resolver=lambda *_: "4.5 Innings",
        )

        assert len(cells) == 1
        assert cells[0].original_value == "4.5 Innings"
        assert cells[0].label == "ambiguous"
        assert cells[0].corrected_value is None
        assert cells[0].cluster_ids == []

    def test_standalone_partial_cell_claim_exports_as_error(
        self, store: LabelStore,
    ) -> None:
        store.append(Label(
            target=LabelTarget.CELL,
            target_id="cell:a.csv:5:duration",
            source="a.csv", row=5, column="duration",
            status=LabelStatus.PARTIAL,
            corrected_value="4.5",
            annotator="tester",
        ))

        cells = materialize_label_benchmark(
            [], store, value_resolver=lambda *_: "4.5 Innings",
        )

        assert len(cells) == 1
        assert cells[0].label == "error"
        assert cells[0].corrected_value == "4.5"
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


# ---------------------------------------------------------------------------
# Pre-export conflict detection (find_label_conflicts)
# ---------------------------------------------------------------------------

from src.labeling.export import find_label_conflicts


def _bcell(
    *, source: str = "a.csv", row: int, column: str = "name",
    original_value: str = "Foo", label: str = "clean",
    corrected_value=None, cluster_ids=None, note: str = "",
) -> BenchmarkCell:
    """Factory for a minimal :class:`BenchmarkCell`."""
    return BenchmarkCell(
        source=source, row=row, column=column,
        original_value=original_value, label=label,
        corrected_value=corrected_value,
        cluster_ids=list(cluster_ids or []),
        note=note,
    )


class TestFindLabelConflicts:
    def test_no_conflicts_when_all_labels_agree(self) -> None:
        cells = [
            _bcell(row=0, label="clean"),
            _bcell(row=1, label="clean"),
            _bcell(row=2, label="clean"),
        ]
        assert find_label_conflicts(cells) == []

    def test_distinct_labels_on_same_value_surface_as_conflict(self) -> None:
        cells = [
            _bcell(row=0, label="clean", cluster_ids=["c0"]),
            _bcell(row=7, label="error", cluster_ids=["c7"],
                   corrected_value="Foobar"),
            _bcell(row=12, label="clean", cluster_ids=["c12"]),
        ]
        groups = find_label_conflicts(cells)
        assert len(groups) == 1
        g = groups[0]
        assert g["source"] == "a.csv"
        assert g["column"] == "name"
        assert g["original_value"] == "Foo"
        assert g["labels"] == ["clean", "error"]
        # Cells listed in row order for stable presentation.
        assert [c["row"] for c in g["cells"]] == [0, 7, 12]
        # Per-cell payload carries enough to round-trip to the UI.
        e = next(c for c in g["cells"] if c["label"] == "error")
        assert e["corrected_value"] == "Foobar"
        assert e["cluster_ids"] == ["c7"]

    def test_distinct_sources_or_columns_are_independent_groups(
        self,
    ) -> None:
        cells = [
            # Group A: a.csv/name -- conflict.
            _bcell(source="a.csv", row=0, column="name",
                   original_value="X", label="clean"),
            _bcell(source="a.csv", row=1, column="name",
                   original_value="X", label="error"),
            # Same value but DIFFERENT column -> independent (no conflict).
            _bcell(source="a.csv", row=2, column="city",
                   original_value="X", label="clean"),
            # Same value but DIFFERENT source -> independent (no conflict).
            _bcell(source="b.csv", row=2, column="name",
                   original_value="X", label="clean"),
        ]
        groups = find_label_conflicts(cells)
        assert len(groups) == 1
        assert groups[0]["source"] == "a.csv"
        assert groups[0]["column"] == "name"

    def test_three_distinct_labels_collapse_into_one_group(self) -> None:
        cells = [
            _bcell(row=0, label="clean"),
            _bcell(row=1, label="error", corrected_value="Foobar"),
            _bcell(row=2, label="ambiguous"),
        ]
        groups = find_label_conflicts(cells)
        assert len(groups) == 1
        assert groups[0]["labels"] == ["ambiguous", "clean", "error"]
        assert len(groups[0]["cells"]) == 3

    def test_empty_column_and_empty_value_cells_are_ignored(self) -> None:
        # Row-level findings (column="") share no column-value semantics
        # with each other, so they should NEVER trigger a conflict
        # warning. Empty original_value cells likewise re-use the same
        # key trivially and would generate noise.
        cells = [
            _bcell(row=0, column="", original_value="X", label="clean"),
            _bcell(row=1, column="", original_value="X", label="error"),
            _bcell(row=2, original_value="", label="clean"),
            _bcell(row=3, original_value="", label="error"),
        ]
        assert find_label_conflicts(cells) == []

    def test_output_is_deterministic_across_runs(self) -> None:
        # Stable sort key: (source, column, original_value). Operators
        # comparing benchmarks across runs shouldn't have to re-read
        # a conflict report whose rows shuffled.
        cells_a = [
            _bcell(source="z.csv", row=0, column="c", original_value="V",
                   label="clean"),
            _bcell(source="z.csv", row=1, column="c", original_value="V",
                   label="error"),
            _bcell(source="a.csv", row=2, column="c", original_value="V",
                   label="clean"),
            _bcell(source="a.csv", row=3, column="c", original_value="V",
                   label="error"),
        ]
        cells_b = list(reversed(cells_a))
        out_a = find_label_conflicts(cells_a)
        out_b = find_label_conflicts(cells_b)
        keys_a = [(g["source"], g["column"], g["original_value"]) for g in out_a]
        keys_b = [(g["source"], g["column"], g["original_value"]) for g in out_b]
        assert keys_a == keys_b == [
            ("a.csv", "c", "V"), ("z.csv", "c", "V"),
        ]




class TestTrivialPerCellFix:
    """A multi-row TRIVIAL whitespace cluster must give each affected row
    its OWN deterministic whitespace fix, not the anchor's correction
    fanned out via ``affected_rows``."""

    def test_peer_rows_get_own_normalized_fix(self, store: LabelStore) -> None:
        vals = {
            (0, "name"): "DRY SODA  - TPR 11/18",   # double space
            (1, "name"): "SEA TANGLE  - TPR",
            (2, "name"): "LIGHTLIFE  - TPR",
        }
        cluster = _cluster(
            "c1", ["name"],
            synthesis={"canonical_fix": "DRY SODA - TPR 11/18"},
            detections=[{"affected_rows": [0, 1, 2], "columns": ["name"]}],
            row=0,
        )
        store.append(Label(
            target=LabelTarget.CLUSTER, target_id="c1", source="a.csv",
            row=0, column="name", columns=["name"], status=LabelStatus.TP,
            note="(auto: spacing anomaly) TRIVIAL", annotator="tester",
        ))
        cells = materialize_label_benchmark(
            [cluster], store,
            value_resolver=lambda s, r, c: vals.get((r, c), ""),
        )
        by_row = {c.row: c.corrected_value for c in cells}
        assert by_row[0] == "DRY SODA - TPR 11/18"
        # Peer rows get THEIR OWN fix, not the anchor's "DRY SODA ...".
        assert by_row[1] == "SEA TANGLE - TPR"
        assert by_row[2] == "LIGHTLIFE - TPR"
        for c in cells:
            assert c.label == "error"
            assert "TRIVIAL" in (c.note or "")

    def test_own_anchor_fix_not_overridden(self, store: LabelStore) -> None:
        # An explicit (operator/LLM) fix on a TRIVIAL-noted cell must be
        # preserved -- the deterministic recompute only fills EMPTY fixes.
        cluster = _cluster("c1", ["name"], row=0)
        store.append(Label(
            target=LabelTarget.CLUSTER, target_id="c1", source="a.csv",
            row=0, column="name", columns=["name"], status=LabelStatus.TP,
            corrected_value="Operator's better fix",
            note="spelling+space TRIVIAL", annotator="tester",
        ))
        cells = materialize_label_benchmark(
            [cluster], store, value_resolver=lambda s, r, c: "foo  bar",
        )
        assert cells[0].corrected_value == "Operator's better fix"

    def test_non_trivial_peer_unaffected(self, store: LabelStore) -> None:
        # A non-TRIVIAL multi-row cluster keeps the existing fan-out /
        # llm-suggestion behaviour (peer rows are NOT whitespace-normalised).
        cluster = _cluster(
            "c1", ["name"],
            synthesis={"canonical_fix": "Canonical"},
            detections=[{"affected_rows": [0, 1], "columns": ["name"]}],
            row=0,
        )
        store.append(Label(
            target=LabelTarget.CLUSTER, target_id="c1", source="a.csv",
            row=0, column="name", columns=["name"], status=LabelStatus.TP,
            annotator="tester",
        ))
        cells = materialize_label_benchmark(
            [cluster], store, value_resolver=lambda s, r, c: "weird  value",
        )
        # Peer row falls back to the cluster's llm suggestion, unchanged.
        by_row = {c.row: c.corrected_value for c in cells}
        assert by_row[1] == "Canonical"

    def test_trivial_anchor_note_survives_earlier_fanout(
        self, store: LabelStore,
    ) -> None:
        # A non-anchor fan-out (cluster c1, plain note) creates the cell
        # for row 1 BEFORE its own anchor (cluster c2, TRIVIAL note) is
        # processed. The agreeing anchor's note must supersede the
        # fanned-out plain note, otherwise the deterministic
        # whitespace-fix backfill (which keys on "TRIVIAL" in the note)
        # skips the cell and the error exports with no correction.
        c1 = {
            "cluster_id": "c1", "source": "a.csv", "row": 0,
            "columns": ["name"], "row_values": {"name": "X"},
            "error_types": ["typo"],
            "detections": [{"affected_rows": [1], "columns": ["name"]}],
        }
        c2 = {
            "cluster_id": "c2", "source": "a.csv", "row": 1,
            "columns": ["name"], "row_values": {"name": "Acme  Co"},
            "error_types": ["typo"], "detections": [],
        }
        # c1 plain note, c2 TRIVIAL note; both TP, neither carries a fix.
        store.append(Label(
            target=LabelTarget.CLUSTER, target_id="c1", source="a.csv",
            row=0, column="name", columns=["name"], status=LabelStatus.TP,
            note="(real typo)", annotator="tester",
        ))
        store.append(Label(
            target=LabelTarget.CLUSTER, target_id="c2", source="a.csv",
            row=1, column="name", columns=["name"], status=LabelStatus.TP,
            note="(auto: spacing anomaly) TRIVIAL", annotator="tester",
        ))
        vals = {(0, "name"): "X", (1, "name"): "Acme  Co"}
        # c1 (the fan-out) is processed BEFORE c2 (the TRIVIAL anchor).
        cells = materialize_label_benchmark(
            [c1, c2], store,
            value_resolver=lambda s, r, c: vals.get((r, c), ""),
        )
        by_key = {(c.row, c.column): c for c in cells}
        row1 = by_key[(1, "name")]
        assert "TRIVIAL" in (row1.note or "")
        # The deterministic whitespace fix is applied (double space -> one).
        assert row1.corrected_value == "Acme Co"


class TestIsLowValueFix:
    """Spacing-or-single-punctuation rule used to flag 'low value' TP cells."""

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("New  Haven", "New Haven", True),       # internal whitespace
            (" Apple ", "Apple", True),              # padding whitespace
            ("Strubenvalley", "Struben Valley", True),   # word-boundary spacing
            ("log anberries", "loganberries", True),     # OCR word-split
            ("St Louis", "St. Louis", True),         # one punctuation added
            ("Pty,) Ltd", "Pty) Ltd", True),         # one punctuation removed
            ("apple", "Apple", True),                # casing-only
            ("Pacfic", "Pacific", False),            # real letter typo
            ("CENTENNIEL", "CENTENNIAL", False),     # real letter typo
            ("NX PRESTIEGE", "NX PRESTIGE", False),  # real letter typo
            ("foo", "foo", False),                   # no-op (no fix)
        ],
    )
    def test_classification(self, a, b, expected):
        assert _is_low_value_fix(a, b) is expected


class TestSkippedCells:
    """``materialize_skipped_cells`` captures SKIP-cluster cells separately."""

    def test_skip_cluster_emitted_with_skip_label(self, store):
        store.append(_label(target_id="c1", column="name", status=LabelStatus.SKIP))
        skipped = materialize_skipped_cells([_cluster("c1", ["name"])], store)
        assert len(skipped) == 1
        assert skipped[0].label == "skip"
        assert skipped[0].original_value == "val_name"
        assert skipped[0].cluster_ids == ["c1"]
        assert skipped[0].corrected_value is None

    def test_non_skip_cluster_not_emitted(self, store):
        store.append(_label(target_id="c1", column="name", status=LabelStatus.TP))
        assert materialize_skipped_cells([_cluster("c1", ["name"])], store) == []

    def test_exclude_keys_drops_benchmark_cells(self, store):
        # A real verdict on the same cell outranks the skip -> omit it.
        store.append(_label(target_id="c1", column="name", status=LabelStatus.SKIP))
        skipped = materialize_skipped_cells(
            [_cluster("c1", ["name"])], store,
            exclude_keys={("a.csv", 0, "name")},
        )
        assert skipped == []

    def test_fanout_across_affected_rows(self, store):
        store.append(_label(target_id="c1", column="name", status=LabelStatus.SKIP))
        cluster = _cluster("c1", ["name"], detections=[{"affected_rows": [0, 5]}])
        skipped = materialize_skipped_cells([cluster], store)
        assert sorted(c.row for c in skipped) == [0, 5]
        assert all(c.label == "skip" for c in skipped)


class TestLowValueFlag:
    """low_value marks ONLY rejected (FP / clean) LLM flags whose fix is
    spacing-or-single-punctuation. Confirmed typos (TP / error) are never
    low value, however trivial their fix."""

    def _flagged(self, store, status, original, fix):
        store.append(_label(
            target_id="c1", column="name", columns=["name"], status=status,
        ))
        return _cluster(
            "c1", ["name"],
            row_values={"name": original},
            synthesis={"canonical_fix": fix},
        )

    def test_fp_spacing_or_punct_fix_is_low_value(self, store):
        cl = self._flagged(store, LabelStatus.FP, "Pty,) Ltd", "Pty) Ltd")
        cells = materialize_label_benchmark([cl], store)
        assert cells[0].label == "clean"
        assert cells[0].low_value is True

    def test_tp_spacing_fix_not_low_value(self, store):
        # A confirmed typo is a real error even with a trivial fix.
        cl = self._flagged(store, LabelStatus.TP, "New  Haven", "New Haven")
        cells = materialize_label_benchmark([cl], store)
        assert cells[0].label == "error"
        assert cells[0].low_value is False

    def test_fp_letter_fix_not_low_value(self, store):
        # FP, but a real letter change -> not a spacing/punct flag.
        cl = self._flagged(store, LabelStatus.FP, "Pacfic", "Pacific")
        cells = materialize_label_benchmark([cl], store)
        assert cells[0].label == "clean"
        assert cells[0].low_value is False

    def test_clean_without_llm_fix_not_low_value(self, store):
        store.append(_label(
            target_id="c1", column="name", columns=["name"],
            status=LabelStatus.FP,
        ))
        cl = _cluster("c1", ["name"], row_values={"name": "Acme Ltd"})
        cells = materialize_label_benchmark([cl], store)
        assert cells[0].low_value is False


# ---------------------------------------------------------------------------
# Instruction-vs-value detection + lean benchmark projection
# ---------------------------------------------------------------------------

from src.labeling._helpers import _derive_trivial_fix, _looks_like_instruction
from src.labeling.export import (
    LEAN_BENCHMARK_FIELDS,
    LEAN_EXPORT_ERROR_TYPES,
    benchmark_record_for_export,
    full_benchmark_record,
    lean_benchmark_record,
)


class TestLooksLikeInstruction:
    @pytest.mark.parametrize("text", [
        'Change "Strubenvalley" to "Struben Valley".',
        'set Primary Target Firm to "Pacific Breeze (Pty) Ltd"',
        'Correct to "Property Protection Management SA (Pty) Ltd"',
        'Change "Ptd" to "Ltd"',
        'Replace “Smart phones” with “Smartphones”',
        "Replace Quard-Core with Quad-Core",
        # Markerless instructions: verb + editing-OBJECT keyword, no quote/arrow.
        "Remove the trailing comma",
        "Remove the trailing comma.",
        "Remove the space before the comma",
        "Remove the space before the colon.",
        "Remove the trailing apostrophe",
        "Add closing parenthesis",
        "Remove the extra whitespace",
        "Delete the leading space",
    ])
    def test_instruction_prose_detected(self, text):
        assert _looks_like_instruction(text) is True

    @pytest.mark.parametrize("text", [
        "Struben Valley",
        "Pacific Breeze (Pty) Ltd",
        "Smartphones & PDAs",
        "Change Healthcare Inc",       # verb prefix but no instruction marker
        "Set Design Studio",            # verb prefix but no marker
        "Remove Wall Anchors",          # verb prefix, no editing-object keyword
        "Add Value Consulting Group",   # verb prefix, no editing-object keyword
        "Trim Kitchen Cabinets",        # verb prefix, no editing-object keyword
        "10 to 20",                      # marker but no verb prefix
        "",
        None,
    ])
    def test_plain_values_not_instructions(self, text):
        assert _looks_like_instruction(text) is False


class TestDeriveTrivialFix:
    """``_derive_trivial_fix`` turns a trivial edit INSTRUCTION into the literal
    result of that specific edit (faithful by construction), or ``None`` when
    the instruction is out of scope."""

    @pytest.mark.parametrize("original,instruction,expected", [
        # Class 1: remove a stray space before a named mark.
        ("PMP , Machine learning with R",
         "Remove the space before the comma",
         "PMP, Machine learning with R"),
        ("Petroleum (Price) Violation Escrow : Exxon 1986 Restitution",
         "Remove the space before the colon.",
         "Petroleum (Price) Violation Escrow: Exxon 1986 Restitution"),
        # Class 2: remove a trailing named mark.
        ("MATTHEW R. DEKOATZ,", "Remove the trailing comma", "MATTHEW R. DEKOATZ"),
        ("RUBEN P. HERNANDEZ,", "Remove the trailing comma.", "RUBEN P. HERNANDEZ"),
        ("Elliott Nevertire Trust'", "Remove the trailing apostrophe",
         "Elliott Nevertire Trust"),
        # Class 3: purely-whitespace collapse (no punctuation object named).
        ("Acme  Ltd", "Remove the extra whitespace", "Acme Ltd"),
    ])
    def test_derives_faithfully(self, original, instruction, expected):
        assert _derive_trivial_fix(original, instruction) == expected

    def test_trailing_removal_does_not_touch_other_punctuation(self):
        # The ampersand must survive: only the trailing comma is removed.
        out = _derive_trivial_fix(
            "Group Insurance Acct Appd&Cont Isf,", "Remove the trailing comma."
        )
        assert out == "Group Insurance Acct Appd&Cont Isf"

    @pytest.mark.parametrize("original,instruction", [
        # Ambiguous placement -> not derivable.
        ("CLUBS-CNTRY,MBRSHIP(ATHLET", "Add closing parenthesis"),
        # Blank-a-cell semantics -> not a concrete value.
        ("NM", "set SA3_Code to null"),
        # Replace with an unknown value -> concrete value unavailable here.
        ('Change "x" to "y"', 'Change "x" to "y"'),
        # A trailing-comma instruction on a value with no trailing comma is a no-op.
        ("MATTHEW R. DEKOATZ", "Remove the trailing comma"),
    ])
    def test_out_of_scope_returns_none(self, original, instruction):
        assert _derive_trivial_fix(original, instruction) is None

    def test_non_instruction_input_returns_none(self):
        # A concrete value that is not an instruction is never "derived".
        assert _derive_trivial_fix("Acme Ltd", "Pacific Breeze (Pty) Ltd") is None

    def test_empty_original_returns_none(self):
        assert _derive_trivial_fix("", "Remove the trailing comma") is None


class TestLeanBenchmarkRecord:
    def _cell(self, **kw) -> BenchmarkCell:
        base = {
            "source": "a.csv", "row": 0, "column": "Firm",
            "original_value": "Pacfic Breeze (Pty) Ltd",
            "label": "error",
            "corrected_value": None,
            "llm_suggested_value": None,
            "error_types": ["typo"],
            "cluster_ids": ["a.csv__r00000-c000"] * 40,
            "detection_ids": ["r0001-i0000"],
            "annotators": ["tester"],
            "note": "(auto: spacing anomaly) TRIVIAL",
        }
        base.update(kw)
        return BenchmarkCell(**base)

    def test_lean_record_has_only_lean_fields(self):
        rec = lean_benchmark_record(self._cell())
        assert set(rec.keys()) == set(LEAN_BENCHMARK_FIELDS)
        # Provenance bloat is dropped.
        for dropped in ("cluster_ids", "detection_ids", "annotators",
                        "note", "llm_suggested_value"):
            assert dropped not in rec

    def test_prose_corrected_value_resolves_to_llm_value(self):
        cell = self._cell(
            corrected_value='set Firm to "Pacific Breeze (Pty) Ltd"',
            llm_suggested_value="Pacific Breeze (Pty) Ltd",
        )
        rec = lean_benchmark_record(cell)
        assert rec["corrected_value"] == "Pacific Breeze (Pty) Ltd"

    def test_concrete_operator_value_is_kept(self):
        cell = self._cell(
            corrected_value="Pacific Breeze (Pty) Ltd",
            llm_suggested_value="something else",
        )
        rec = lean_benchmark_record(cell)
        assert rec["corrected_value"] == "Pacific Breeze (Pty) Ltd"

    def test_underivable_prose_is_dropped(self):
        cell = self._cell(
            corrected_value='Change "x" to "y"',
            llm_suggested_value=None,
        )
        rec = lean_benchmark_record(cell)
        # No concrete value AND the instruction is not a trivially derivable
        # class -> drop it (null) rather than ship an edit-instruction sentence.
        assert rec["corrected_value"] is None

    def test_derivable_prose_instruction_is_resolved(self):
        cell = self._cell(
            original_value="MATTHEW R. DEKOATZ,",
            corrected_value="Remove the trailing comma",
            llm_suggested_value=None,
        )
        rec = lean_benchmark_record(cell)
        # A trivial, unambiguous instruction is applied to the original.
        assert rec["corrected_value"] == "MATTHEW R. DEKOATZ"

    def test_clean_cell_has_null_corrected_value(self):
        cell = self._cell(label="clean", corrected_value=None,
                          llm_suggested_value="Cape Gate (Pty) Ltd")
        rec = lean_benchmark_record(cell)
        assert rec["label"] == "clean"
        assert rec["corrected_value"] is None

    def test_benchmark_record_for_export_lean_only_for_lean_types(self):
        cell = self._cell()
        assert "typo" in LEAN_EXPORT_ERROR_TYPES
        # typo -> lean
        typo_rec = benchmark_record_for_export(cell, "typo")
        assert set(typo_rec.keys()) == set(LEAN_BENCHMARK_FIELDS)
        # a non-lean type -> full schema (provenance retained)
        full_rec = benchmark_record_for_export(cell, "fd_violation")
        assert "cluster_ids" in full_rec and "note" in full_rec

    def test_non_lean_export_is_also_instruction_safe(self):
        # The unified non-lean path resolves corrected_value too: an edit
        # instruction never ships verbatim, while full provenance is kept.
        cell = self._cell(
            corrected_value='set Firm to "Pacific Breeze (Pty) Ltd"',
            llm_suggested_value="Pacific Breeze (Pty) Ltd",
        )
        rec = benchmark_record_for_export(cell, "fd_violation")
        assert rec["corrected_value"] == "Pacific Breeze (Pty) Ltd"
        assert "cluster_ids" in rec and "note" in rec


class TestFullBenchmarkRecord:
    """``full_benchmark_record`` keeps every model_dump field but resolves
    ``corrected_value`` to a concrete literal for ``error`` cells (so the
    full ``benchmark.jsonl`` never ships an edit-instruction), exactly like
    the lean/merged exports."""

    def _cell(self, **kw) -> BenchmarkCell:
        base = {
            "source": "a.csv", "row": 0, "column": "Firm",
            "original_value": "Pacfic Breeze (Pty) Ltd",
            "label": "error",
            "corrected_value": None,
            "llm_suggested_value": None,
            "error_types": ["typo"],
            "cluster_ids": ["a.csv__r00000-c000"],
            "detection_ids": ["r0001-i0000"],
            "annotators": ["tester"],
            "note": "n",
        }
        base.update(kw)
        return BenchmarkCell(**base)

    def test_prose_corrected_value_resolves_to_llm_literal(self):
        cell = self._cell(
            corrected_value='set Firm to "Pacific Breeze (Pty) Ltd"',
            llm_suggested_value="Pacific Breeze (Pty) Ltd",
        )
        rec = full_benchmark_record(cell)
        assert rec["corrected_value"] == "Pacific Breeze (Pty) Ltd"
        # Full schema retained, incl. the raw suggestion alongside.
        assert rec["llm_suggested_value"] == "Pacific Breeze (Pty) Ltd"
        assert "cluster_ids" in rec and "note" in rec and "annotators" in rec

    def test_operator_literal_is_kept_over_suggestion(self):
        cell = self._cell(
            corrected_value="Pacific Breeze (Pty) Ltd",
            llm_suggested_value="something else",
        )
        rec = full_benchmark_record(cell)
        assert rec["corrected_value"] == "Pacific Breeze (Pty) Ltd"

    def test_clean_cell_dumped_unchanged(self):
        cell = self._cell(label="clean", corrected_value="whatever",
                          llm_suggested_value="x")
        rec = full_benchmark_record(cell)
        # Non-error cells are dumped verbatim (corrected_value untouched).
        assert rec["corrected_value"] == "whatever"
        assert rec == cell.model_dump()

    def test_matches_lean_corrected_value(self):
        # The full and lean exports must agree on the resolved corrected_value.
        cell = self._cell(
            corrected_value='Change "Pacfic" to "Pacific"',
            llm_suggested_value="Pacific Breeze (Pty) Ltd",
        )
        assert (
            full_benchmark_record(cell)["corrected_value"]
            == lean_benchmark_record(cell)["corrected_value"]
        )


# ---------------------------------------------------------------------------
# Merged immunity stream (skip cells + exclusion sentinels)
# ---------------------------------------------------------------------------

from src.labeling.export import materialize_ignored_cells
from src.labeling.schema import Exclusion, ExclusionScope, IgnoreRecord


def _exclusion(scope, source, *, rows=None, columns=None) -> Exclusion:
    return Exclusion(
        id=f"x-{source}-{scope.value}",
        scope=scope,
        source=source,
        rows=rows or [],
        columns=columns or [],
        annotator="tester",
    )


class TestMaterializeIgnoredCells:
    def _skip_cell(self, source="a.csv", row=3, column="Firm") -> BenchmarkCell:
        return BenchmarkCell(
            source=source, row=row, column=column,
            original_value="Acme  Ltd", label="skip",
        )

    def test_skip_cells_become_skip_records(self):
        recs = materialize_ignored_cells([self._skip_cell()], [])
        assert len(recs) == 1
        r = recs[0]
        assert r.reason == "skip"
        assert (r.source, r.row, r.column) == ("a.csv", 3, "Firm")
        assert r.original_value == "Acme  Ltd"
        assert r.scope is None

    def test_csv_exclusion_is_one_wildcard_sentinel(self):
        recs = materialize_ignored_cells(
            [], [_exclusion(ExclusionScope.CSV, "big.csv")]
        )
        assert len(recs) == 1
        r = recs[0]
        assert (r.reason, r.scope) == ("exclusion", "csv")
        assert r.source == "big.csv"
        assert r.row is None and r.column is None  # whole-file wildcard

    def test_rows_exclusion_expands_per_row_column_wildcard(self):
        recs = materialize_ignored_cells(
            [], [_exclusion(ExclusionScope.ROWS, "b.csv", rows=[4, 9])]
        )
        assert {(r.row, r.column) for r in recs} == {(4, None), (9, None)}
        assert all(r.reason == "exclusion" and r.scope == "rows" for r in recs)

    def test_columns_exclusion_expands_per_column_row_wildcard(self):
        recs = materialize_ignored_cells(
            [], [_exclusion(ExclusionScope.COLUMNS, "c.csv", columns=["Notes"])]
        )
        assert len(recs) == 1
        r = recs[0]
        assert (r.row, r.column, r.scope) == (None, "Notes", "columns")

    def test_merge_orders_skips_then_exclusions(self):
        recs = materialize_ignored_cells(
            [self._skip_cell()],
            [
                _exclusion(ExclusionScope.COLUMNS, "c.csv", columns=["X"]),
                _exclusion(ExclusionScope.CSV, "big.csv"),
                _exclusion(ExclusionScope.ROWS, "b.csv", rows=[1]),
            ],
        )
        reasons = [r.reason for r in recs]
        # skip first, then exclusions grouped csv -> rows -> columns.
        assert reasons[0] == "skip"
        assert [r.scope for r in recs[1:]] == ["csv", "rows", "columns"]

    def test_ignore_record_is_serializable(self):
        recs = materialize_ignored_cells(
            [self._skip_cell()], [_exclusion(ExclusionScope.CSV, "big.csv")]
        )
        for r in recs:
            assert isinstance(r, IgnoreRecord)
            json.dumps(r.model_dump())  # must not raise
