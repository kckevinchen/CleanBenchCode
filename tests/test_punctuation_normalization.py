"""Unit tests for the Tier-1 trivial-PUNCTUATION normalizer in
``src.labeling._helpers`` (``_normalize_punctuation`` / ``_has_punctuation_anomaly``).

The punctuation analogue of ``test_*`` coverage for the whitespace pair: each
Tier-1 rule fires on a true anomaly, every guard leaves its target untouched,
the fix is idempotent, and pure-whitespace defects are left to the whitespace
rule (the two taggers must stay orthogonal).
"""

from __future__ import annotations

import pytest

from src.labeling._helpers import (
    _contains_email_or_url,
    _has_punctuation_anomaly,
    _has_whitespace_anomaly,
    _normalize_punctuation,
    _punctuation_column_findings,
    _punctuation_rules_fired,
)

# (input, expected) pairs where a Tier-1 rule SHOULD fire.
FIRES = [
    # 1. Unicode -> ASCII fold.
    ("\u201cJames\u201d", '"James"'),       # curly double quotes
    ("\u2018a\u2019", "'a'"),               # curly single quotes
    ("O\u2019Brien", "O'Brien"),            # smart apostrophe in a name
    ("a\u2026b", "a...b"),                   # ellipsis glyph -> three dots
    ("\u22125", "-5"),                       # U+2212 minus -> hyphen-minus
    # 2. Doubled comma / semicolon.
    ("a,,b", "a, b"),                        # also gains the rule-7 space
    ("a;;b", "a;b"),
    ("a;;;b", "a;b"),
    # 3 + 4. Bracket padding / space before closing punctuation.
    ("( x )", "(x)"),
    ("[ x ]", "[x]"),
    ("word ,", "word,"),
    ("word ;", "word;"),
    ("word :", "word:"),
    ("word !", "word!"),
    ("word ?", "word?"),
    ("word )", "word)"),
    ("Inc .", "Inc."),
    # 5. Missing space before "(" (the motivating example).
    ("word(text)", "word (text)"),
    ("Mississippi(gulf coast)", "Mississippi (gulf coast)"),
    # 6. Missing space after ")".
    ("(text)word", "(text) word"),
    # 7. Missing space after a comma.
    ("Doe,Jane", "Doe, Jane"),
    ("a,b,c", "a, b, c"),
    # 8. Asymmetric slash spacing -> tighten to "A/B".
    ("A/ B", "A/B"),                              # space after slash only
    ("A /B", "A/B"),                              # space before slash only
    ("and/ or", "and/or"),
    ("N/ A", "N/A"),
    ("ORNAMENTAL IRON/ BLOCK", "ORNAMENTAL IRON/BLOCK"),
    ("Northleaf /InfraRed/Macquarie", "Northleaf/InfraRed/Macquarie"),
    ("A /B/ C", "A/B/C"),                         # both directions in one cell
    # 9. Word-joining ampersand -> add spaces; acronyms keep the tight form.
    ("Black&Decker", "Black & Decker"),
    ("milk&more", "milk & more"),
    ("Town&Country", "Town & Country"),
    ("Black &Decker", "Black & Decker"),          # asymmetric (space before)
    ("Black& Decker", "Black & Decker"),          # asymmetric (space after)
    ("a&b", "a & b"),
    ("a&b&c", "a & b & c"),                        # chain, one pass
]

# Inputs no Tier-1 rule may touch (guards + out-of-scope shapes).
GUARDS = [
    "1,000",            # grouped number, not a list
    "3,14",             # comma-decimal (unknown locale) -> leave alone
    "f(x)",             # single letter before paren (function)
    "g(y)",
    "varchar(255)",     # numeric arg after paren (size)
    "Smith(2020)",      # citation year
    "H2O(l)",           # formula: digit inside the pre-token
    "U.S.",             # abbreviation periods
    "a.b",              # missing-space-after-period is Tier 2, not Tier 1
    "1990-95",          # ascii hyphen range untouched (en-dash NOT folded)
    "max",              # plain word
    "",                 # empty
    "(already) spaced", # nothing to do
    "Doe, Jane",        # already correct
    # symmetric slash spacing is a deliberate style, not a typo.
    "A / B",
    "Sales / Marketing",
    "5 / 5",
    "M4217 / M4219",
    "A/B",              # clean slash, no adjacent space
    "and / or",
    # optional-plural / inflection idiom: the parenthetical is a grammatical
    # suffix, never a missing space.
    "recommendation(s)",
    "POUND(S)",
    "box(es)",
    "categor(ies)",
    "Form(S)",
    # ampersand acronyms / initialisms / codes -> tight ``&`` is conventional.
    "AT&T",
    "R&D",
    "CF&CE",
    "H&M",
    "Q&A",
    "S&P",
    "BB&T",
    "MS&E",
    "1&1",               # numeric, not a word-join
    "1320C&E",           # code, not a word-join
    "Smith&CO",          # half-acronym (one side all-caps)
    "J&Johnson",         # half-acronym (single initial)
    "Black & Decker",    # already correctly spaced
]


@pytest.mark.parametrize("raw, expected", FIRES)
def test_rule_fires_and_fixes(raw: str, expected: str) -> None:
    assert _normalize_punctuation(raw) == expected
    assert _has_punctuation_anomaly(raw) is True


@pytest.mark.parametrize("raw, expected", FIRES)
def test_fix_is_idempotent(raw: str, expected: str) -> None:
    once = _normalize_punctuation(raw)
    assert _normalize_punctuation(once) == once
    assert _has_punctuation_anomaly(once) is False


@pytest.mark.parametrize("raw", GUARDS)
def test_guards_leave_value_untouched(raw: str) -> None:
    assert _normalize_punctuation(raw) == raw
    assert _has_punctuation_anomaly(raw) is False


@pytest.mark.parametrize(
    "raw",
    [
        " James",          # leading space
        "James ",          # trailing space
        "James  Smith",    # 2+ internal spaces
        "a\tb",            # tab
        "a\u00a0b",        # non-breaking space
    ],
)
def test_pure_whitespace_is_not_a_punctuation_anomaly(raw: str) -> None:
    """Whitespace-only defects belong to the whitespace rule, so the
    punctuation rule must ignore them (orthogonal taggers)."""
    assert _has_whitespace_anomaly(raw) is True
    assert _has_punctuation_anomaly(raw) is False
    assert _normalize_punctuation(raw) == raw


def test_detector_agrees_with_fix() -> None:
    """``_has_punctuation_anomaly`` is True iff the fix changes the value."""
    for raw, _ in FIRES:
        assert _has_punctuation_anomaly(raw) == (_normalize_punctuation(raw) != raw)
    for raw in GUARDS:
        assert _has_punctuation_anomaly(raw) is False


def test_skip_aware_normalize_omits_named_rules() -> None:
    """``skip`` drops the named rule from BOTH the fix and the detection."""
    assert _normalize_punctuation("Doe,Jane", skip={"missing_space_after_comma"}) == "Doe,Jane"
    # Other rules still apply when only one is skipped.
    assert _normalize_punctuation(
        "Doe,Jane(x)", skip={"missing_space_after_comma"}
    ) == "Doe,Jane (x)"


def test_rules_fired_reports_contributing_rules() -> None:
    assert _punctuation_rules_fired("Doe,Jane") == frozenset({"missing_space_after_comma"})
    assert _punctuation_rules_fired("word(text)") == frozenset({"missing_space_before_paren"})
    assert _punctuation_rules_fired("clean value") == frozenset()
    # recommendation(s): the (s) idiom is guarded, so NO rule fires.
    assert _punctuation_rules_fired("recommendation(s)") == frozenset()
    assert _punctuation_rules_fired("A/ B") == frozenset({"slash_spacing"})
    assert _punctuation_rules_fired("Black&Decker") == frozenset({"ampersand_spacing"})


class TestSlashSpacing:
    """Rule 8: asymmetric space around a slash is a typo (``A/ B`` -> ``A/B``);
    symmetric spacing (``A / B``) is a deliberate style and is left alone."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("A/ B", "A/B"),
            ("A /B", "A/B"),
            ("and/ or", "and/or"),
            ("w/ o", "w/o"),
            ("24/ 7", "24/7"),
            ("c/ o", "c/o"),
            ("HC Svcs Supv/ Assistant Mgr", "HC Svcs Supv/Assistant Mgr"),
            ("A /B/ C", "A/B/C"),
            ("A/  B", "A/B"),                 # multiple spaces
            ("A/\tB", "A/B"),                 # tab counts like a space here
        ],
    )
    def test_asymmetric_is_tightened(self, raw: str, expected: str) -> None:
        assert _normalize_punctuation(raw) == expected
        assert _has_punctuation_anomaly(raw) is True

    @pytest.mark.parametrize(
        "raw",
        [
            "A / B",            # symmetric -> deliberate style
            "Sales / Marketing",
            "5 / 5",
            "M4217 / M4219",
            "and / or",
            "A/B",              # no adjacent space
            "N/A",
            "/leading",         # nothing valid on the left
            "trailing /",       # nothing valid on the right
        ],
    )
    def test_symmetric_and_clean_are_untouched(self, raw: str) -> None:
        assert _normalize_punctuation(raw) == raw
        assert _has_punctuation_anomaly(raw) is False

    def test_url_with_slash_space_is_exempt(self) -> None:
        # The e-mail/URL guard wins: a URL's slashes are structural.
        assert _normalize_punctuation("http://a.com/ x") == "http://a.com/ x"
        assert _has_punctuation_anomaly("www.foo.com/a/ b") is False


class TestAmpersandSpacing:
    """Rule 9: an ampersand joining two WORDS gets spaces (``Black&Decker`` ->
    ``Black & Decker``); a short all-caps acronym keeps its tight ``&``."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Black&Decker", "Black & Decker"),
            ("Barnes&Noble", "Barnes & Noble"),
            ("Smith&Jones", "Smith & Jones"),
            ("milk&more", "milk & more"),
            ("Town&Country", "Town & Country"),
            ("Month&Year", "Month & Year"),
            ("RENTAL&LEASING&MORE", "RENTAL&LEASING&MORE"),  # all-caps: not words
            ("a&b", "a & b"),
            ("Black &Decker", "Black & Decker"),   # asymmetric: space before
            ("Black& Decker", "Black & Decker"),   # asymmetric: space after
            ("a&b&c", "a & b & c"),                # chain spaced in one pass
            ("New Town&Country Club", "New Town & Country Club"),
        ],
    )
    def test_word_join_is_spaced(self, raw: str, expected: str) -> None:
        assert _normalize_punctuation(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "AT&T", "R&D", "CF&CE", "E&I", "H&M", "M&S", "B&B", "O&M",
            "MS&E", "Q&A", "S&P", "BB&T", "J&J",   # acronyms / initialisms
            "1&1", "1320C&E",                       # codes (digits, not words)
            "Smith&CO", "J&Johnson",                # half-acronym -> leave alone
            "Black & Decker",                       # already spaced (idempotent)
            "AT & T",
            "milk  &  more",                        # multi-space -> whitespace rule
        ],
    )
    def test_acronyms_and_codes_are_untouched(self, raw: str) -> None:
        assert _normalize_punctuation(raw) == raw
        assert _has_punctuation_anomaly(raw) is False

    def test_url_query_ampersand_is_exempt(self) -> None:
        # The e-mail/URL guard wins: ``&`` in a query string is structural.
        assert _normalize_punctuation("foo.com/p?a=1&b=2") == "foo.com/p?a=1&b=2"
        assert _has_punctuation_anomaly("jane&doe@example.com") is False

    def test_uniform_ampersand_column_is_suppressed(self) -> None:
        # Every cell is a ``Word&Word`` join -> the column's convention.
        col = ["milk&more", "rock&roll", "salt&pepper", "fish&chips", "law&order"]
        findings = _punctuation_column_findings(col)
        assert all(not anom for anom, _ in findings)


class TestColumnConventionGuard:
    """A punctuation pattern shared by ~all of a column's cells is the
    column's convention, not a defect, so it must not be flagged."""

    def test_uniform_comma_column_is_suppressed(self) -> None:
        # Every cell is `word,word` (comma-joined pairs) -> convention.
        col = [
            "people,analytics", "analytics,world", "looking,forward",
            "use,code", "list,speakers",
        ]
        findings = _punctuation_column_findings(col)
        assert all(not anom for anom, _ in findings)

    def test_uniform_space_before_colon_is_suppressed(self) -> None:
        col = [f"Total : Region {i}" for i in range(6)]
        findings = _punctuation_column_findings(col)
        assert all(not anom for anom, _ in findings)

    def test_isolated_anomaly_in_clean_column_is_flagged(self) -> None:
        col = ["Acme Corp", "Beta LLC", "Gamma Inc", "Delta Co", "Acme(Holdings)"]
        findings = _punctuation_column_findings(col)
        flagged = [(i, fix) for i, (anom, fix) in enumerate(findings) if anom]
        assert flagged == [(4, "Acme (Holdings)")]

    def test_convention_aware_fix_preserves_shared_pattern(self) -> None:
        # Comma is the column convention (5/5); only the last cell ALSO has a
        # genuine missing-space-before-paren. It is flagged, but its fix must
        # NOT add a space after the convention comma.
        col = ["a,b", "c,d", "e,f", "g,h", "Acme(X),y"]
        findings = _punctuation_column_findings(col)
        flagged = [(i, fix) for i, (anom, fix) in enumerate(findings) if anom]
        assert flagged == [(4, "Acme (X),y")]

    def test_small_column_cannot_establish_convention(self) -> None:
        # Below the min-cell floor, fall back to the per-cell rule.
        col = ["a,b", "c,d", "e,f"]
        findings = _punctuation_column_findings(col)
        assert all(anom for anom, _ in findings)

    def test_mixed_column_below_threshold_still_flags(self) -> None:
        # Half the cells use the comma pattern (50% < 90%) -> not a
        # convention, so they are flagged.
        col = ["a,b", "c,d", "clean one", "clean two"]
        findings = _punctuation_column_findings(col)
        flagged = [i for i, (anom, _) in enumerate(findings) if anom]
        assert flagged == [0, 1]

    def test_empty_and_blank_cells_ignored(self) -> None:
        col = ["people,analytics", "", "   ", "use,code", "list,speakers", "a,b"]
        findings = _punctuation_column_findings(col)
        # convention (comma) across the 4 non-empty cells -> nothing flagged.
        assert all(not anom for anom, _ in findings)


# Cells containing an e-mail address or a web URL must be exempt: their dots,
# commas, slashes and parentheses are structural, not defects.
EMAIL_WEB = [
    "john.doe@example.com,jane@example.com",   # comma-joined emails
    "Contact,info@acme.com",                    # comma before an email
    "plain@email.com",                          # plain email
    "en.wikipedia.org/wiki/Foo(bar)",           # bare-host URL w/ paren
    "https://site.com/Path(x)",                 # scheme URL w/ paren
    "www.foo.com/a,b",                          # www URL w/ comma
    "example.com,example.org",                  # comma-joined domains
    "see http://a.com/x(y) now",                # URL embedded in prose
    "mycompany.co.uk/contact",                  # ccTLD host + path
    "COMPANY.CO.UK",                            # bare ccTLD host, upper-case
]

# Look-alikes that are NOT emails / URLs and must stay flaggable / inert.
NOT_EMAIL_WEB = [
    "Doe,Jane",          # name list -> a real anomaly
    "Acme(Holdings)",    # missing-space paren -> a real anomaly
    "U.S.",              # abbreviation, not a domain
    "e.g.",
    "Mr.Smith",          # ".smith" is not a TLD
    "data.csv",          # filename, ".csv" is not a TLD
    "9.99",              # decimal
    "version 1.2.3",
]


class TestEmailWebGuard:
    @pytest.mark.parametrize("raw", EMAIL_WEB)
    def test_email_or_url_is_detected(self, raw: str) -> None:
        assert _contains_email_or_url(raw) is True

    @pytest.mark.parametrize("raw", EMAIL_WEB)
    def test_email_or_url_is_never_modified_or_flagged(self, raw: str) -> None:
        assert _normalize_punctuation(raw) == raw
        assert _has_punctuation_anomaly(raw) is False
        assert _punctuation_rules_fired(raw) == frozenset()

    @pytest.mark.parametrize("raw", NOT_EMAIL_WEB)
    def test_lookalikes_are_not_treated_as_email_or_url(self, raw: str) -> None:
        assert _contains_email_or_url(raw) is False

    def test_real_anomalies_among_lookalikes_still_fire(self) -> None:
        assert _normalize_punctuation("Doe,Jane") == "Doe, Jane"
        assert _normalize_punctuation("Acme(Holdings)") == "Acme (Holdings)"

    def test_email_column_yields_no_findings(self) -> None:
        col = [
            "a@x.com,b@x.com", "c@y.com,d@y.com", "e@z.com",
            "f@z.com,g@z.com",
        ]
        findings = _punctuation_column_findings(col)
        assert all(not anom for anom, _ in findings)


