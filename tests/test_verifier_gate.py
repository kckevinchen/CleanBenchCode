"""Unit tests for the verifier gate (entity-column heuristic)."""

from __future__ import annotations

from src.verifier.gate import (
    DEFAULT_ERROR_TYPES,
    EntityKind,
    GateConfig,
    classify_cluster,
    filter_eligible,
    should_verify,
)


def _cluster(**kw):
    base = {
        "cluster_id": "x__r0-c0",
        "source": "x.csv",
        "row": 0,
        "columns": [],
        "error_types": [],
    }
    base.update(kw)
    return base


def test_org_column_is_eligible():
    c = _cluster(columns=["Vendor name"], error_types=["typo"])
    assert classify_cluster(c) == EntityKind.ORG
    assert should_verify(c, cell_value="AcmeCo")


def test_address_column_is_eligible():
    c = _cluster(columns=["Mailing Address"], error_types=["text_inconsistency"])
    assert classify_cluster(c) == EntityKind.ADDRESS
    assert should_verify(c, cell_value="1 Main St")


def test_phone_url_kinds():
    assert classify_cluster(_cluster(columns=["Phone"])) == EntityKind.PHONE
    assert classify_cluster(_cluster(columns=["Website"])) == EntityKind.URL


def test_person_off_by_default_for_pii():
    c = _cluster(columns=["First Name"], error_types=["typo"])
    assert classify_cluster(c) == EntityKind.NONE
    # Opt in.
    cfg = GateConfig(verify_person=True)
    assert classify_cluster(c, cfg=cfg) == EntityKind.PERSON
    assert should_verify(c, cell_value="Marcus", cfg=cfg)


def test_person_column_not_verified_via_error_type_fallthrough():
    """Regression: a PERSON column with a whitelisted error_type
    (e.g. typo) used to leak through ``should_verify`` because
    ``classify_cluster`` downgraded PERSON->NONE, after which the
    ``error_type`` branch happily admitted it. The PII reject must
    win over the error_type branch."""
    c = _cluster(columns=["First Name"], error_types=["typo"])
    assert not should_verify(c, cell_value="Matt")  # default cfg
    c2 = _cluster(columns=["Last Name"], error_types=["text_inconsistency"])
    assert not should_verify(c2, cell_value="Smith")
    c3 = _cluster(columns=["Contact Name"], error_types=["typo"])
    assert not should_verify(c3, cell_value="Jane Doe")
    # But with opt-in, it goes through.
    cfg = GateConfig(verify_person=True)
    assert should_verify(c, cell_value="Matt", cfg=cfg)


def test_numeric_only_value_rejected():
    c = _cluster(columns=["Amount"], error_types=["typo"])
    # Column doesn't match any entity regex but error_type is in
    # whitelist -> kind=OTHER. The numeric value gate should still
    # reject.
    assert classify_cluster(c) == EntityKind.OTHER
    assert not should_verify(c, cell_value="$1,234.56")
    assert not should_verify(c, cell_value="42")


def test_too_long_value_rejected():
    c = _cluster(columns=["Description"], error_types=["typo"])
    long = "x" * 500
    # Description has no entity-regex match; falls through to OTHER on
    # error_type whitelist; value gate rejects on length.
    assert not should_verify(c, cell_value=long)


def test_irrelevant_error_type_rejected():
    c = _cluster(columns=["Notes"], error_types=["numeric_outlier"])
    assert not should_verify(c, cell_value="text here")


def test_multiline_header_still_matches():
    """LLM column-mangling occasionally produces multi-line headers
    like ``"Contractor\\nName"``. The entity regex must still classify
    those after whitespace folding."""
    c = _cluster(columns=["Contractor\nName"], error_types=["typo"])
    assert classify_cluster(c) == EntityKind.ORG


def test_filter_eligible_uses_value_lookup():
    clusters = [
        _cluster(cluster_id="a", columns=["Vendor"], error_types=["typo"]),
        _cluster(cluster_id="b", columns=["Vendor"], error_types=["typo"]),
        _cluster(cluster_id="c", columns=["Amount"], error_types=["typo"]),
    ]
    values = {"a": "AcmeCo", "b": "", "c": "$100"}

    def lookup(c):
        return values[c["cluster_id"]]

    out = filter_eligible(clusters, value_lookup=lookup)
    ids = [c["cluster_id"] for c in out]
    assert ids == ["a"]   # b is empty, c is numeric


def test_default_error_types_include_typo():
    assert "typo" in DEFAULT_ERROR_TYPES
    assert "text_inconsistency" in DEFAULT_ERROR_TYPES
