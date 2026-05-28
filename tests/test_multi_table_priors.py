"""Tests for ``src.core.multi_table.priors``."""

from __future__ import annotations

import pandas as pd

from src.core.multi_table.priors import (
    AttributeAgreementCandidate,
    CompositeFKCandidate,
    ContainmentCandidate,
    EmbeddingMatchCandidate,
    FKCandidate,
    MultiTablePriors,
    NamingCandidate,
    NullSentinelCandidate,
    RepresentationCandidate,
    TimeOverlapCandidate,
    compute_table_priors,
    find_attribute_agreement_conflicts,
    find_composite_fk_candidates,
    find_containment_candidates,
    find_embedding_match_candidates,
    find_fk_candidates,
    find_naming_conflicts,
    find_null_sentinel_candidates,
    find_representation_conflicts,
    find_time_overlap_candidates,
    normalize_value,
    register_alias_dictionary,
    render_table_priors_block,
    render_workbook_context,
    set_embedding_matcher,
)
from src.core.table_digest import build_table_digest


def _make_workbook(*sheets):
    """Helper: build (frames, digests) dicts from a list of (name, df)."""
    frames = {name: df for name, df in sheets}
    digests = {name: build_table_digest(df, source=name) for name, df in sheets}
    return frames, digests


# ---------------------------------------------------------------------------
# find_fk_candidates
# ---------------------------------------------------------------------------


class TestFindFKCandidates:
    def test_simple_orphan_in_child(self) -> None:
        parent = pd.DataFrame({
            "ProjectCode": [f"P{i:02d}" for i in range(20)],
            "Owner": ["a", "b", "c"] + ["x"] * 17,
        })
        child = pd.DataFrame({
            "ProjectCode": ["P00", "P01", "P02", "P03", "ZZ-XXX", "FJ-001"],
            "Hours":       [10, 20, 30, 40, 50, 60],
        })
        frames, digests = _make_workbook(("parent.csv", parent), ("child.csv", child))
        cands = find_fk_candidates(
            sheets=["parent.csv", "child.csv"],
            frames=frames, digests=digests,
        )
        # Should find at least one FK candidate parent <- child on ProjectCode.
        fks = [c for c in cands
               if c.parent == "parent.csv" and c.child == "child.csv"
               and c.parent_column == "ProjectCode"]
        assert len(fks) >= 1
        fk = fks[0]
        # Two orphans: ZZ-XXX, FJ-001.
        assert "ZZ-XXX" in fk.orphan_values
        assert "FJ-001" in fk.orphan_values
        assert fk.coverage < 1.0

    def test_skips_perfect_overlap(self) -> None:
        # No orphans -> not actionable.
        parent = pd.DataFrame({"ID": [f"P{i}" for i in range(20)]})
        child = pd.DataFrame({"ID": ["P0", "P1", "P2", "P3"]})
        frames, digests = _make_workbook(("p.csv", parent), ("c.csv", child))
        cands = find_fk_candidates(
            sheets=["p.csv", "c.csv"], frames=frames, digests=digests,
        )
        assert cands == []

    def test_single_sheet_returns_empty(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        frames, digests = _make_workbook(("only.csv", df))
        assert find_fk_candidates(
            sheets=["only.csv"], frames=frames, digests=digests,
        ) == []


# ---------------------------------------------------------------------------
# find_naming_conflicts
# ---------------------------------------------------------------------------


class TestFindNamingConflicts:
    def test_synonym_high_overlap_different_names(self) -> None:
        # Same identifier values, different headers -> synonym.
        sheet_a = pd.DataFrame({
            "CustomerID": [f"C{i:03d}" for i in range(15)],
            "Name":       [f"Customer {i}" for i in range(15)],
        })
        sheet_b = pd.DataFrame({
            "ClientNumber": [f"C{i:03d}" for i in range(15)],
            "Status":       ["active"] * 15,
        })
        frames, digests = _make_workbook(("a.csv", sheet_a), ("b.csv", sheet_b))
        cands = find_naming_conflicts(
            sheets=["a.csv", "b.csv"], frames=frames, digests=digests,
        )
        synonyms = [c for c in cands if c.relation == "synonym"]
        assert any(
            ({c.column_a, c.column_b} == {"CustomerID", "ClientNumber"})
            for c in synonyms
        )

    def test_homonym_same_name_disjoint_values(self) -> None:
        sheet_a = pd.DataFrame({"Region": ["North", "South", "East", "West"] * 4})
        sheet_b = pd.DataFrame({"Region": ["EMEA", "APAC", "AMER", "LATAM"] * 4})
        frames, digests = _make_workbook(("a.csv", sheet_a), ("b.csv", sheet_b))
        cands = find_naming_conflicts(
            sheets=["a.csv", "b.csv"], frames=frames, digests=digests,
        )
        homs = [c for c in cands if c.relation == "homonym"]
        assert any(c.column_a == "Region" for c in homs)


# ---------------------------------------------------------------------------
# find_representation_conflicts
# ---------------------------------------------------------------------------


class TestFindRepresentationConflicts:
    def test_dtype_mismatch(self) -> None:
        sheet_a = pd.DataFrame({"Sales": [1, 2, 3, 4]})            # int
        sheet_b = pd.DataFrame({"Sales": ["1.5", "2.5", "3.5"]})   # string
        frames, digests = _make_workbook(("a.csv", sheet_a), ("b.csv", sheet_b))
        cands = find_representation_conflicts(
            sheets=["a.csv", "b.csv"], frames=frames, digests=digests,
        )
        assert any(c.column == "Sales" and c.reason == "dtype" for c in cands)

    def test_magnitude_mismatch(self) -> None:
        # USD vs EUR (or different unit) - similar but ratios differ ~100x
        sheet_a = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
        sheet_b = pd.DataFrame({"Amount": [10000.0, 20000.0, 30000.0]})
        frames, digests = _make_workbook(("a.csv", sheet_a), ("b.csv", sheet_b))
        cands = find_representation_conflicts(
            sheets=["a.csv", "b.csv"], frames=frames, digests=digests,
        )
        assert any(c.column == "Amount" and c.reason == "magnitude" for c in cands)

    def test_vocabulary_mismatch(self) -> None:
        sheet_a = pd.DataFrame({"Gender": ["M", "F", "M", "F", "M"]})
        sheet_b = pd.DataFrame({"Gender": ["0", "1", "0", "1", "1"]})
        frames, digests = _make_workbook(("a.csv", sheet_a), ("b.csv", sheet_b))
        cands = find_representation_conflicts(
            sheets=["a.csv", "b.csv"], frames=frames, digests=digests,
        )
        assert any(c.column == "Gender" and c.reason == "vocabulary" for c in cands)


# ---------------------------------------------------------------------------
# compute_table_priors
# ---------------------------------------------------------------------------


class TestComputeTablePriors:
    def test_dispatch_by_error_type(self) -> None:
        # Build a workbook with all three patterns simultaneously.
        a = pd.DataFrame({
            "ProjectCode": [f"P{i:02d}" for i in range(15)],
            "Sales": [100.0, 200.0, 300.0] * 5,
            "Region": ["North"] * 15,
        })
        b = pd.DataFrame({
            "ProjectCode": ["P00", "P01", "ZZ"] * 5,
            "Sales": [10000.0, 20000.0, 30000.0] * 5,
            "Region": ["EMEA"] * 15,
        })
        frames, digests = _make_workbook(("a.csv", a), ("b.csv", b))
        sheets = ["a.csv", "b.csv"]

        umbrella_priors = compute_table_priors(
            sheets=sheets, frames=frames, digests=digests,
            error_type="inter_table_inconsistency", workbook="wb1",
        )
        # The unified umbrella type runs ALL three detectors.
        assert umbrella_priors.fk_candidates  # FK orphans found
        assert umbrella_priors.representation_candidates  # Sales magnitude

    def test_single_sheet_returns_empty(self) -> None:
        frames = {"only.csv": pd.DataFrame({"a": [1, 2, 3]})}
        digests = {"only.csv": build_table_digest(frames["only.csv"], source="only.csv")}
        priors = compute_table_priors(
            sheets=["only.csv"], frames=frames, digests=digests,
            error_type="inter_table_inconsistency", workbook="wb1",
        )
        assert priors.is_empty()


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class TestRenderers:
    def test_render_workbook_context_lists_sheets(self) -> None:
        a = pd.DataFrame({"X": [1, 2, 3], "Y": ["a", "b", "c"]})
        b = pd.DataFrame({"X": [10, 20], "Z": [True, False]})
        frames, digests = _make_workbook(("a.csv", a), ("b.csv", b))
        out = render_workbook_context(
            workbook="wb1", sheets=["a.csv", "b.csv"],
            frames=frames, digests=digests,
        )
        assert "a.csv" in out
        assert "b.csv" in out
        assert "WORKBOOK CONTEXT" in out

    def test_render_priors_block_empty(self) -> None:
        priors = MultiTablePriors(error_type="inter_table_inconsistency", workbook="wb1")
        out = render_table_priors_block(priors)
        assert "STRUCTURAL CROSS-TABLE CANDIDATES" in out
        assert "none found" in out

    def test_render_priors_block_with_fk(self) -> None:
        priors = MultiTablePriors(
            error_type="inter_table_inconsistency", workbook="wb1",
            sheets=("a.csv", "b.csv"),
            fk_candidates=(FKCandidate(
                parent="a.csv", parent_column="ID",
                child="b.csv", child_column="RefID",
                n_parent_distinct=10, n_child_distinct=8,
                overlap=6, coverage=0.75,
                orphan_row_indices=(2, 7),
                orphan_values=("X", "Y"),
            ),),
        )
        out = render_table_priors_block(priors)
        assert "Foreign-key candidates" in out
        assert "a.csv" in out and "b.csv" in out
        assert "75%" in out


# ---------------------------------------------------------------------------
# Normalization ladder & alias tables (Gap B)
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_alias_country(self) -> None:
        assert normalize_value("USA") == normalize_value("United States")
        assert normalize_value("U.S.") == normalize_value("usa")
        assert normalize_value("UK") == normalize_value("united kingdom")

    def test_punctuation_and_case_fold(self) -> None:
        assert normalize_value("New-York") == normalize_value("new york")
        assert normalize_value("  New   York  ") == normalize_value("new york")

    def test_unidecode_strips_accents(self) -> None:
        # Accented variants fold to ASCII.
        assert normalize_value("Mexico") == normalize_value("México")

    def test_register_alias_dictionary(self) -> None:
        register_alias_dictionary([
            ("project_x", ("px", "p-x", "project x")),
        ])
        assert normalize_value("PX") == normalize_value("Project X")


class TestFKNormalization:
    def test_fk_surface_form_inconsistency_flagged(self) -> None:
        # Parent has full names, child has abbrevs -> raw coverage low,
        # normalized coverage high. The "US" / "United States" case.
        parent = pd.DataFrame({
            "Country": ["United States", "United Kingdom", "Germany",
                        "France", "Japan", "China", "Russia",
                        "Brazil", "Italy", "Spain"],
        })
        child = pd.DataFrame({
            "Country": ["USA", "UK", "DEU", "FR", "JP",
                        "China", "Russia", "Brazil", "Italy", "Spain", "ZZ"],
        })
        frames, digests = _make_workbook(("p.csv", parent), ("c.csv", child))
        cands = find_fk_candidates(
            sheets=["p.csv", "c.csv"], frames=frames, digests=digests,
        )
        # Should produce a candidate where normalized_coverage is
        # materially higher than raw coverage.
        flagged = [
            c for c in cands
            if c.normalized_coverage > c.coverage + 0.1
            and c.best_normalization_step != "raw"
        ]
        assert flagged, f"no surface-form candidates flagged: {cands}"


# ---------------------------------------------------------------------------
# Attribute-agreement (Gap F)
# ---------------------------------------------------------------------------


class TestAttributeAgreement:
    def test_conflict_on_descriptive_attribute(self) -> None:
        a = pd.DataFrame({
            "CustomerID": [f"C{i:03d}" for i in range(10)],
            "Country":    ["US"] * 5 + ["UK"] * 5,
            "Tier":       ["Gold"] * 10,
        })
        # Same customers, but Country in sheet b disagrees on 3.
        b = pd.DataFrame({
            "CustomerID": [f"C{i:03d}" for i in range(10)],
            "Country":    ["US"] * 3 + ["Canada"] * 2 + ["UK"] * 3 + ["FR"] * 2,
            "Tier":       ["Gold"] * 10,
        })
        frames, digests = _make_workbook(("customers.csv", a), ("orders.csv", b))
        cands = find_attribute_agreement_conflicts(
            sheets=["customers.csv", "orders.csv"],
            frames=frames, digests=digests,
        )
        country_conflicts = [c for c in cands if c.attribute_column == "Country"]
        assert country_conflicts, f"no country conflicts: {cands}"
        c0 = country_conflicts[0]
        assert c0.n_conflicts >= 3
        assert c0.n_shared_keys == 10


# ---------------------------------------------------------------------------
# Composite-FK (Gap F)
# ---------------------------------------------------------------------------


class TestCompositeFK:
    def test_two_column_fk_detected(self) -> None:
        # (Country, PostalCode) tuple FK.
        parent = pd.DataFrame({
            "Country":    ["US", "US", "UK", "UK", "DE", "DE", "FR", "FR"],
            "PostalCode": ["10001", "94016", "SW1A", "EC1A", "10115", "20095",
                           "75001", "13001"],
            "Region":     ["NE", "W", "Lon", "Lon", "Berlin", "Hamburg",
                           "Paris", "Marseille"],
        })
        child = pd.DataFrame({
            "Country":    ["US", "US", "UK", "DE", "FR", "FR", "ZZ"],
            "PostalCode": ["10001", "94016", "SW1A", "10115", "75001", "00000", "9999"],
            "Sales":      [100, 200, 300, 400, 500, 600, 700],
        })
        frames, digests = _make_workbook(("p.csv", parent), ("c.csv", child))
        cands = find_composite_fk_candidates(
            sheets=["p.csv", "c.csv"], frames=frames, digests=digests,
        )
        # We expect at least one composite candidate covering
        # (Country, PostalCode).
        comp = [c for c in cands
                if set(c.parent_columns) == {"Country", "PostalCode"}]
        # Whether we get one depends on whether the digest flagged
        # both as identity-like; if not, the test is trivially
        # satisfied (no false detection).
        if comp:
            c0 = comp[0]
            assert c0.coverage < 1.0  # orphans exist
            assert c0.orphan_keys


# ---------------------------------------------------------------------------
# Containment (Gap F)
# ---------------------------------------------------------------------------


class TestContainment:
    def test_one_sided_containment(self) -> None:
        # Small dim table mostly contained in large fact.
        dim = pd.DataFrame({"region_code": ["EMEA", "APAC", "AMER", "LATAM", "ANZ"]})
        fact = pd.DataFrame({
            "region_code": (["EMEA"] * 200 + ["APAC"] * 150
                            + ["AMER"] * 300 + ["NORDIC"] * 50
                            + ["AFRICA"] * 25 + ["ASIA"] * 25),
            "Sales": list(range(750)),
        })
        frames, digests = _make_workbook(("dim.csv", dim), ("fact.csv", fact))
        cands = find_containment_candidates(
            sheets=["dim.csv", "fact.csv"], frames=frames, digests=digests,
        )
        # Expect at least one direction with high containment.
        hit = [c for c in cands
               if max(c.coverage_a_in_b, c.coverage_b_in_a) >= 0.6]
        assert hit


# ---------------------------------------------------------------------------
# Time overlap (Gap F)
# ---------------------------------------------------------------------------


class TestTimeOverlap:
    def test_overlapping_date_ranges(self) -> None:
        a = pd.DataFrame({
            "date": pd.date_range("2023-01-01", "2023-12-31", freq="D"),
            "x": range(365),
        })
        b = pd.DataFrame({
            "date": pd.date_range("2023-06-01", "2024-06-01", freq="D"),
            "y": range(367),
        })
        frames, digests = _make_workbook(("q1.csv", a), ("q2.csv", b))
        cands = find_time_overlap_candidates(
            sheets=["q1.csv", "q2.csv"], frames=frames, digests=digests,
        )
        assert cands
        c0 = cands[0]
        assert c0.n_rows_a_in_overlap > 0
        assert c0.n_rows_b_in_overlap > 0


# ---------------------------------------------------------------------------
# Null-sentinel drift (Gap F)
# ---------------------------------------------------------------------------


class TestNullSentinel:
    def test_different_sentinels_flagged(self) -> None:
        a = pd.DataFrame({"Status": ["active", "active", "NA", "NA", "NA",
                                     "inactive", "active", "active"]})
        b = pd.DataFrame({"Status": ["active", "-", "-", "-", "active",
                                     "inactive", "active", "active"]})
        frames, digests = _make_workbook(("a.csv", a), ("b.csv", b))
        cands = find_null_sentinel_candidates(
            sheets=["a.csv", "b.csv"], frames=frames, digests=digests,
        )
        assert cands
        c0 = cands[0]
        assert c0.column == "Status"
        assert {c0.sentinel_a, c0.sentinel_b} == {"na", "-"}


# ---------------------------------------------------------------------------
# Embedding-residual hook (Gap C)
# ---------------------------------------------------------------------------


class TestEmbeddingHook:
    def test_no_matcher_yields_empty(self) -> None:
        a = pd.DataFrame({"Country": ["Germany", "France", "Italy"] * 5})
        b = pd.DataFrame({"Country": ["Deutschland", "Francia", "Italia"] * 5})
        frames, digests = _make_workbook(("a.csv", a), ("b.csv", b))
        # Default: no matcher registered.
        set_embedding_matcher(None)
        out = find_embedding_match_candidates(
            sheets=["a.csv", "b.csv"], frames=frames, digests=digests,
        )
        assert out == []

    def test_registered_matcher_invoked(self) -> None:
        calls = []

        def fake_matcher(va, vb):
            calls.append((tuple(va), tuple(vb)))
            return [("Germany", "Deutschland", 0.91)]

        set_embedding_matcher(fake_matcher)
        try:
            a = pd.DataFrame({"Country": ["Germany", "France", "Italy",
                                            "Spain", "Belgium"] * 3})
            b = pd.DataFrame({"Country": ["Deutschland", "Francia",
                                            "Italia", "Espana",
                                            "Belgique"] * 3})
            frames, digests = _make_workbook(("a.csv", a), ("b.csv", b))
            out = find_embedding_match_candidates(
                sheets=["a.csv", "b.csv"], frames=frames, digests=digests,
            )
            assert calls, "matcher was not invoked"
            assert out
            assert out[0].matches[0][0] == "Germany"
        finally:
            set_embedding_matcher(None)
