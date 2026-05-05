"""Structural tests for the ``number_format_inconsistency`` prompt.

These tests validate that the prompt YAML still carries the
guardrails introduced after the kmax-sweep singleton diagnosis:

* The legacy ``affected_rows_cap`` knob has been removed.
* R6 mandates listing EVERY affected row (no truncation).
* The NFI notes contain a "QUANTITATIVE-COLUMN GATE" that fast-skips
  code / identifier columns (zip, FIPS, county/town code, SKU, SSN,
  account number, ...).

They are pure-Python and run without any LLM calls.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.core.error_taxonomy import load_taxonomy

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "error_types.yaml"


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy(PROMPT_PATH)


# ---------------------------------------------------------------------------
# affected_rows_cap removal + R6 completeness
# ---------------------------------------------------------------------------


class TestAffectedRowsCompleteness:
    def test_default_cap_is_gone(self):
        raw = PROMPT_PATH.read_text(encoding="utf-8")
        # Comment line about which keys live in defaults still references
        # historical names; the active YAML key should be absent.
        assert "affected_rows_cap:" not in raw, (
            "defaults.affected_rows_cap should be removed; we want "
            "complete affected_rows lists (every error captured)."
        )

    def test_user_template_has_no_unresolved_placeholders(self, taxonomy):
        # If a template still references ${affected_rows_cap} the
        # taxonomy loader would refuse to load. This is a belt-and-
        # suspenders check that nothing dangling remains.
        unresolved = re.findall(r"\$\{[a-zA-Z_]+\}", taxonomy.user_template)
        assert unresolved == [], unresolved

    def test_r6_requires_listing_every_row(self, taxonomy):
        tmpl = taxonomy.user_template
        # The new R6 must mention completeness, no truncation.
        assert re.search(r"R6\b", tmpl), "R6 rule missing from user_template"
        r6_block = tmpl[tmpl.find("R6") : tmpl.find("R7")]
        assert "EVERY" in r6_block or "every" in r6_block.lower(), r6_block
        assert "truncate" in r6_block.lower(), r6_block
        # No leftover phrasing about a cap.
        assert "AT MOST" not in r6_block.upper().replace("AT MOST 1", "")
        assert "cap" not in r6_block.lower() or "do not truncate" in r6_block.lower()


# ---------------------------------------------------------------------------
# number_format_inconsistency code/identifier guard
# ---------------------------------------------------------------------------


class TestNFIQuantitativeGate:
    def test_nfi_type_exists(self, taxonomy):
        assert "number_format_inconsistency" in taxonomy.keys()

    def test_notes_have_quantitative_gate_header(self, taxonomy):
        nfi = taxonomy.get("number_format_inconsistency")
        assert "QUANTITATIVE-COLUMN GATE" in nfi.notes, nfi.notes[:400]

    @pytest.mark.parametrize(
        "term",
        [
            "code",
            "identifier",
            "zip",
            "FIPS",
            "postal",
            "SKU",
            "SSN",
            "account",
            "leading zero",
        ],
    )
    def test_notes_call_out_known_code_column_kinds(self, taxonomy, term):
        nfi = taxonomy.get("number_format_inconsistency")
        assert term.lower() in nfi.notes.lower(), (
            f"NFI notes should mention '{term}' so the model fast-skips "
            f"code/identifier columns instead of fabricating singletons."
        )

    def test_notes_route_padding_to_pattern_inconsistency(self, taxonomy):
        nfi = taxonomy.get("number_format_inconsistency")
        # Zero-padding / fixed-width differences must be redirected to
        # pattern_inconsistency, not flagged here.
        assert "pattern_inconsistency" in nfi.notes
        assert (
            "zero-padding" in nfi.notes.lower()
            or "leading zero" in nfi.notes.lower()
        )

    def test_notes_require_value_preserving_normalisation(self, taxonomy):
        nfi = taxonomy.get("number_format_inconsistency")
        # The dominant and minority forms must denote the SAME numeric
        # value when normalised. This is the core test that distinguishes
        # genuine NFI from code-column false positives.
        # Collapse whitespace so multi-line phrases match.
        flat = re.sub(r"\s+", " ", nfi.notes.lower())
        assert (
            "same numeric value when normalised" in flat
            or "same numeric value" in flat
            or "numeric meaning is preserved" in flat
        ), nfi.notes

    def test_notes_emit_empty_for_non_quantitative(self, taxonomy):
        nfi = taxonomy.get("number_format_inconsistency")
        # The fast-skip instruction on a code column.
        assert '{"issues": []}' in nfi.notes
