"""Structural tests for the merged ``inconsistency`` prompt's
number-format sub-pattern guardrails.

Originally these tests covered a standalone
``number_format_inconsistency`` type. After the five legacy
``*_inconsistency`` types were folded into a single ``inconsistency``
umbrella, the same NFI-specific guardrails now live inside the
merged prompt under its number_format sub-pattern. The tests here
validate that the YAML still carries:

* the affected_rows_cap removal + R6 "list every row" completeness
  rule (this part is type-agnostic, lives in base.user_template);
* the QUANTITATIVE-COLUMN GATE that fast-skips code / identifier
  columns (zip, FIPS, county/town code, SKU, SSN, account number,
  ...) for the number_format sub-pattern;
* the routing rule sending zero-padding / fixed-width differences to
  the pattern sub-pattern instead of the number_format sub-pattern;
* the value-preserving-normalisation guard that distinguishes
  genuine number_format inconsistency from code-column false
  positives.

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
# affected_rows_cap removal + R6 completeness (lives in base.user_template,
# applies to every type, including the merged ``inconsistency``).
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

    def test_r9_truncation_guard_present(self, taxonomy):
        """R9 must instruct the model to FAST-SKIP a column whose
        values are pervasively truncated. Without this, every
        per-type signal on a truncated column produces noise: a
        cut-off place name reads as a typo, a cut-off URL reads
        as a pattern inconsistency, a cut-off composed string
        reads as a string_calculation defect, etc.
        """
        tmpl = taxonomy.user_template
        assert re.search(r"R9\b", tmpl), "R9 rule missing from user_template"
        # Locate the R9 block. R9 is currently the last rule before
        # OUTPUT FORMAT.
        r9_start = tmpl.find("R9")
        output_start = tmpl.find("OUTPUT FORMAT")
        r9_block = tmpl[r9_start:output_start]
        # Must explicitly say "truncation" and "FAST-SKIP" / fast-skip
        # so the carve-out is greppable.
        assert "truncation" in r9_block.lower(), r9_block
        assert "fast-skip" in r9_block.lower() or "FAST-SKIP" in r9_block
        # Must reference the empty-issues JSON literal so the model
        # has a copyable instruction.
        assert '{"issues": []}' in r9_block, r9_block

    @pytest.mark.parametrize(
        "signal_phrase",
        [
            # Mid-word / mid-token endings.
            "mid-word",
            # Truncation-marker suffixes.
            "...",
            # Uniform-length cap heuristic. Fold whitespace so
            # "SUSPICIOUSLY UNIFORM length" still matches.
            "uniform",
        ],
    )
    def test_r9_lists_truncation_heuristics(self, taxonomy, signal_phrase):
        """The truncation guard must enumerate concrete heuristics
        the model can use to detect truncation - mid-word endings,
        ellipsis markers, uniform-length caps. Without enumeration
        the rule is too abstract to apply consistently.
        """
        tmpl = re.sub(r"\s+", " ", taxonomy.user_template.lower())
        assert signal_phrase.lower() in tmpl, (
            f"R9 should mention the truncation-detection signal "
            f"{signal_phrase!r}; without explicit heuristics the "
            f"guard is too abstract."
        )


# ---------------------------------------------------------------------------
# number_format sub-pattern code/identifier guard inside the merged
# ``inconsistency`` umbrella.
# ---------------------------------------------------------------------------


class TestNFIQuantitativeGate:
    def test_inconsistency_type_exists(self, taxonomy):
        # Post-merge, the standalone ``number_format_inconsistency``
        # key is gone; its guardrails must live inside the merged
        # ``inconsistency`` type's notes.
        assert "inconsistency" in taxonomy.keys()
        assert "number_format_inconsistency" not in taxonomy.keys()

    def test_notes_have_quantitative_gate_header(self, taxonomy):
        ic = taxonomy.get("inconsistency")
        assert "QUANTITATIVE-COLUMN GATE" in ic.notes, ic.notes[:400]

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
        ic = taxonomy.get("inconsistency")
        assert term.lower() in ic.notes.lower(), (
            f"inconsistency notes should mention '{term}' so the model "
            f"fast-skips the number_format sub-pattern on "
            f"code/identifier columns instead of fabricating singletons."
        )

    def test_notes_route_padding_to_pattern_subpattern(self, taxonomy):
        ic = taxonomy.get("inconsistency")
        # Zero-padding / fixed-width differences must be redirected to
        # the pattern sub-pattern, not flagged as number_format.
        assert "pattern" in ic.notes.lower()
        assert (
            "zero-padding" in ic.notes.lower()
            or "leading zero" in ic.notes.lower()
        )

    def test_notes_require_value_preserving_normalisation(self, taxonomy):
        ic = taxonomy.get("inconsistency")
        # The dominant and minority forms must denote the SAME numeric
        # value when normalised. This is the core test that
        # distinguishes genuine number_format inconsistency from
        # code-column false positives. Collapse whitespace so
        # multi-line phrases match.
        flat = re.sub(r"\s+", " ", ic.notes.lower())
        assert (
            "same numeric value when normalised" in flat
            or "same numeric value" in flat
            or "numeric meaning is preserved" in flat
        ), ic.notes

    def test_notes_emit_empty_for_non_quantitative(self, taxonomy):
        ic = taxonomy.get("inconsistency")
        # The fast-skip instruction on a code column.
        assert '{"issues": []}' in ic.notes


# ---------------------------------------------------------------------------
# Identity-column carve-out (person names, emails, URLs, phones)
# ---------------------------------------------------------------------------
#
# Both the `typo` and the `inconsistency` prompts must instruct the
# model to FAST-SKIP these columns. Variation in unique-per-subject
# identifiers (each person, each inbox, each URL) is content, not a
# representation defect, and the model cannot adjudicate it. Flagging
# here only manufactures false positives.


class TestIdentityColumnFastSkip:
    @pytest.fixture(scope="class")
    def typo_notes_lower(self, taxonomy):
        # Collapse any internal whitespace so multi-line phrases like
        # "first name, last\n  name" still match a literal "last name"
        # check. Without this, the YAML's hard-wrap at 60 chars would
        # spuriously hide phrases from the test.
        return re.sub(r"\s+", " ", taxonomy.get("typo").notes.lower())

    @pytest.fixture(scope="class")
    def inconsistency_notes_lower(self, taxonomy):
        return re.sub(
            r"\s+", " ", taxonomy.get("inconsistency").notes.lower(),
        )

    def test_typo_notes_have_semantic_view_section(self, taxonomy):
        notes = taxonomy.get("typo").notes
        assert "SEMANTIC VIEW OF THE COLUMN" in notes, (
            "The typo prompt must instruct the model to inspect the "
            "column's semantic meaning before flagging - otherwise it "
            "will manufacture typos in person-name / email columns."
        )

    @pytest.mark.parametrize(
        "term",
        [
            "email", "url", "phone",
        ],
    )
    def test_typo_notes_call_out_identity_columns(
        self, typo_notes_lower, term,
    ):
        assert term.lower() in typo_notes_lower, (
            f"typo notes should mention '{term}' as an identity-column "
            f"kind to FAST-SKIP. Without this, the model flags spelling "
            f"variants of emails / URLs as typos even though they "
            f"cannot be verified."
        )

    def test_typo_notes_explicitly_fast_skip_identity_columns(
        self, taxonomy,
    ):
        notes = taxonomy.get("typo").notes
        # The fast-skip JSON literal must appear, signalling that the
        # model returns an empty result rather than down-weighting.
        assert '{"issues": []}' in notes
        # And the prose must say "FAST-SKIP" or "fast-skip" so the
        # behaviour is unambiguous.
        assert "fast-skip" in notes.lower() or "FAST-SKIP" in notes

    @pytest.mark.parametrize(
        "term",
        [
            "email", "url",
        ],
    )
    def test_inconsistency_notes_call_out_identity_columns(
        self, inconsistency_notes_lower, term,
    ):
        assert term.lower() in inconsistency_notes_lower, (
            f"inconsistency notes should mention '{term}' as an "
            f"identity-column kind. Without this, the model flags "
            f"casing / pattern / text variations of unique-per-subject "
            f"identifiers."
        )

    def test_inconsistency_notes_have_identity_fast_skip_section(
        self, inconsistency_notes_lower,
    ):
        # The phrase "IDENTITY COLUMNS" (any case) must appear so the
        # carve-out is structurally visible in the prompt.
        assert "identity column" in inconsistency_notes_lower, (
            "inconsistency notes should have a labelled IDENTITY "
            "COLUMNS section so the carve-out is greppable / "
            "auditable."
        )

    def test_every_error_type_has_a_semantic_view_section(self, taxonomy):
        """Cross-cutting invariant: every error type's notes must
        carry a SEMANTIC VIEW section that tells the model to
        consider the column's (or table's, for row_duplication;
        slice's, for inter_table_inconsistency) semantic role
        BEFORE flagging.

        Without this, the model defaults to syntactic checks and
        flags spurious typos / inconsistencies / outliers / FDs in
        columns where the type doesn't actually apply.
        """
        missing: list[str] = []
        for key in taxonomy.keys():
            notes = taxonomy.get(key).notes
            if "SEMANTIC VIEW" not in notes:
                missing.append(key)
        assert not missing, (
            f"These error types are missing a SEMANTIC VIEW section: "
            f"{missing}. Every type needs explicit guidance on which "
            f"column kinds it applies to and which to fast-skip."
        )

    def test_every_error_type_emits_empty_for_inapplicable_columns(
        self, taxonomy,
    ):
        """Counterpart to the SEMANTIC VIEW invariant: every type's
        notes must contain the literal ``{"issues": []}`` JSON so
        the LLM has an unambiguous, copyable example of how to
        return-no-findings on a fast-skip case.

        We allow exceptions for types where fast-skip lives in the
        sibling base.user_template (the shared ``R1`` rule already
        explains the empty form). This test pins the per-type
        instruction wherever the SEMANTIC VIEW section says
        FAST-SKIP.
        """
        missing: list[str] = []
        for key in taxonomy.keys():
            notes = taxonomy.get(key).notes
            # All types whose SEMANTIC VIEW uses the words
            # "FAST-SKIP" or "fast-skip" must show the JSON literal.
            if (
                "fast-skip" in notes.lower()
                and '{"issues": []}' not in notes
            ):
                missing.append(key)
        assert not missing, (
            f"These types reference FAST-SKIP in their SEMANTIC VIEW "
            f"but don't show the literal {{\"issues\": []}} JSON: "
            f"{missing}. Add the literal so the LLM has a copyable "
            f"example."
        )

    def test_typo_explicitly_excludes_abbreviations(
        self, typo_notes_lower, taxonomy,
    ):
        """Standard abbreviations / acronyms / initialisms must be
        called out as NOT typos. Without this, the model classifies
        short forms ('Ave', 'Inc', 'Dr', 'USA') as 'obvious
        misspellings of a common domain word' and emits high-
        confidence false positives.
        """
        # The notes must have an explicit ABBREVIATIONS section.
        notes = taxonomy.get("typo").notes
        assert "ABBREVIATIONS ARE NOT TYPOS" in notes, (
            "The typo prompt must have a labelled ABBREVIATIONS ARE "
            "NOT TYPOS section so the carve-out is structurally "
            "visible and greppable."
        )
        # And the rule must be described as a "do not flag" / "not"
        # case, not merely a down-weight.
        assert "do not flag" in typo_notes_lower or (
            "abbreviation" in typo_notes_lower and "not" in typo_notes_lower
        )

    @pytest.mark.parametrize(
        "term",
        [
            "number",
            "numeric",
            "serial",
            "version",
            "uuid",
            "file name",
            "file path",
            ".doc",
            ".png",
        ],
    )
    def test_typo_notes_call_out_non_word_columns(
        self, typo_notes_lower, term,
    ):
        """Beyond identity columns, the model must also fast-skip
        columns whose values are numbers, opaque serial / version
        codes, or file names. "Typo" reasoning only applies to
        spelled-out WORDS; treating "1234" / "report.doc" / "v1.2.3"
        as candidates for typo finding produces nothing but false
        positives.
        """
        assert term.lower() in typo_notes_lower, (
            f"typo notes should mention '{term}' as a non-typo-eligible "
            f"column / value kind. Without this, the model flags "
            f"digit / serial / file-name differences as typos."
        )

    @pytest.mark.parametrize(
        "non_typo_example",
        [
            "report.doc",
            "1234",
            "ABC-123-XYZ",
            "v1.2.3",
        ],
    )
    def test_typo_example_block_calls_out_non_word_values(
        self, taxonomy, non_typo_example,
    ):
        """The DO NOT FLAG block in the typo example must show
        canonical non-word values (number / serial / file name) so
        the carve-out is concrete and copyable."""
        flat_example = re.sub(
            r"\s+", " ", taxonomy.get("typo").example,
        )
        assert non_typo_example in flat_example, (
            f"The typo example block should reference {non_typo_example!r} "
            f"as a DO NOT FLAG case. Without canonical examples the "
            f"model treats the carve-out as advisory and still flags."
        )

    @pytest.mark.parametrize(
        "abbreviation",
        [
            # Country / region
            "USA",
            # Corporate suffixes
            "Inc",
            "Corp",
            "Ltd",
            # Street types
            "Ave",
            "St",
            # Place prefixes
            "Mt",
            "Ft",
            # Titles
            "Dr",
            # Org-name initialisms
            "MoE",
        ],
    )
    def test_typo_examples_call_out_canonical_abbreviations(
        self, taxonomy, abbreviation,
    ):
        """The DO NOT FLAG block in the typo example must show
        canonical abbreviations the model is most likely to
        misclassify as typos. Pinning these avoids regressions where
        a future edit drops the example list.
        """
        flat_example = re.sub(
            r"\s+", " ", taxonomy.get("typo").example,
        )
        assert abbreviation in flat_example, (
            f"The typo example block should reference the canonical "
            f"abbreviation {abbreviation!r} as a DO NOT FLAG case. "
            f"Without this, the model treats short forms as typos."
        )

    def test_typo_does_not_silently_downweight_identity_columns(
        self, typo_notes_lower,
    ):
        # The DOWN-WEIGHT section USED to enumerate "person names"
        # (capping their confidence at low). Operator policy is now
        # that name columns follow the NORMAL confidence ladder so
        # genuine name typos are not silently demoted; only the
        # remaining identity shapes (email / URL / file name) stay
        # capped. This test pins that the down-weight list no longer
        # mentions person names.
        assert "person name" not in typo_notes_lower, (
            "The typo notes must NOT list 'person name' under the "
            "DOWN-WEIGHT identity-shape carve-out. Operator policy "
            "is to let name columns follow the normal confidence "
            "ladder so legitimate name typos are not capped at low."
        )
