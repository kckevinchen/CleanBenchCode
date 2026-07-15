"""v9 grader variants -- v7 reward shapes with the v9 rulebook fork.

v9 forks the precision sub-grader (`score_model_extras`) from v5/v7 in
two ways:

  1. It loads the v9 rulebook (``RFT/typo/v9/prompts/rulebook.md``)
     instead of the v5 rulebook. The v9 rulebook puts proper nouns
     OUT OF SCOPE for character-level typo corrections (sec.3).
  2. The judge's developer message is rewritten so that proposed
     corrections that change a proper-noun value via a character-level
     slip are classified as ``fp`` (reward 0.0), while structural
     defects on proper-noun values (adjacent duplicated word, doubled
     punctuation) remain ``valid`` (reward 1.0).

Refinement (2026-05-28): the judge decides scope from the VERIFIABILITY of the
TOKEN the correction changes, not from the value or column. A misspelled COMMON
word -- ordinary dictionary word, domain term, or generic geographic descriptor
(Mountain, North, ...) -- is `valid` EVEN inside a place / person / company name
(`Pilot Mountatin` -> `Pilot Mountain`, `Toowoomba Nour` -> `Toowoomba North`);
single-letter spacing and structural defects stay in scope; only a SPECIFIC,
UNVERIFIABLE name token (unusual surname, settlement/county coinage, brand) is
`fp` (`Calavares` -> `Calaveras`, `O'Donell` -> `O'Donnell`). This took the
gold-answer reward from 36% -> 75% pass / mean precision 0.37 -> 0.76 (checked by
``outputs/grader_check/grader_accuracy.py``). Residual misses are ~half
genuinely-correct name rejections (a gold-reconciliation matter -- those cells
should leave the gold set) and ~half gpt-4o limits on structural/punctuation
defects + temp-0 non-determinism, not the proper-noun over-rejection this fixed.

Everything else (trivial-FP gate, column-cap gate, python recall
sub-grader, reward shapes r1 / r2 / r3, pass thresholds) is inherited
from v7 unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from RFT._common import typo_task as _v5  # shared primitives (no peer-version dep)

V9_VARIANTS: tuple[str, ...] = ("r1", "r2", "r3")


# ---------------------------------------------------------------------------
# Rulebook -- loaded from the v9 fork on disk
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TYPO_V9_RULEBOOK_PATH = (
    _PROJECT_ROOT / "RFT" / "typo" / "v9" / "prompts" / "rulebook.md"
)


def _load_rulebook_v9() -> str:
    if not TYPO_V9_RULEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"v9 rulebook missing at {TYPO_V9_RULEBOOK_PATH}; the v9 "
            f"grader cannot be built without it."
        )
    return TYPO_V9_RULEBOOK_PATH.read_text(encoding="utf-8")


TYPO_V9_RULEBOOK = _load_rulebook_v9()


# ---------------------------------------------------------------------------
# v9 precision sub-grader -- rulebook + judge instructions both updated
# ---------------------------------------------------------------------------

def build_score_model_extras_grader_v9(
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """v9 fork of the precision sub-grader.

    Returns a float in [0, 1] reflecting how rulebook-justified the
    model's extras are. Differences vs the v5/v7 variant:

      * Embeds the v9 rulebook (proper nouns out of scope for
        character-level slips).
      * The ``Rulebook reminders`` and ``Scoring procedure`` sections
        explicitly classify character-level proper-noun corrections
        as ``fp``, with structural-defect carve-outs (adjacent
        duplicated word, doubled punctuation) staying ``valid``.
    """
    if grader_model not in _v5.TYPO_V5_GRADER_MODEL_WHITELIST:
        raise ValueError(
            f"grader_model {grader_model!r} not in Azure RFT score_model "
            f"whitelist {_v5.TYPO_V5_GRADER_MODEL_WHITELIST!r}"
        )

    developer = (
        "You are evaluating the PRECISION of typo predictions from a "
        "column-audit model. The audit model is tuned for RECALL and "
        "deliberately over-proposes; your job is to be PRECISE -- judge "
        "each predicted typo on its own merits and mark the wrong ones, "
        "so the model learns where it over-reached. You use ONLY the v9 "
        "rulebook and the column data you "
        "are shown. You are NOT given any gold set, label set, or "
        "list of known-clean values, and you must not assume one "
        "exists -- every prediction is evaluated against the rulebook "
        "and the column evidence alone.\n\n"
        f"{TYPO_V9_RULEBOOK}\n\n"
        "Inputs (plain-text labelled fields):\n"
        "  - column:             CSV column header.\n"
        "  - source:             source table id.\n"
        "  - column_values:      JSON array of [value, count] pairs "
        "for the column (your only evidence about what values "
        "actually appear and how often).\n"
        "  - model_output:       the audit model's raw JSON reply "
        "(an object with a 'typos' array of "
        "{value, correction, flavour, rationale}).\n\n"
        "Rulebook reminders that govern every classification below:\n"
        "  * Casing-only edits, padding-whitespace-only edits "
        "(adding or stripping leading/trailing whitespace only), and "
        "no-op edits (correction equals value) are NOT typos. A "
        "prediction whose edit reduces to one of these violates the "
        "rulebook and must be scored as a false positive -- it must "
        "NOT be excused on the grounds that the rulebook would "
        "ignore such an edit.\n"
        "  * Spelling, punctuation, and spacing-tokenisation typos "
        "ARE in scope on ORDINARY (non-proper-noun) values: "
        "inserting a missing comma in a list, splitting a run-"
        "together compound, joining a wrongly-split compound, fixing "
        "a misspelled common word, and similar substantive edits "
        "ARE typos under the rulebook.\n"
        "  * **Decide scope from the TOKEN the correction CHANGES, "
        "not from the whole value or the column name.** Most values "
        "in Name / Vendor / Agency / Supplier / School columns are "
        "multi-token, and the proper-noun rule protects ONLY the "
        "name-specific token itself -- never the whole value.\n"
        "      - If the edited token is a COMMON, VERIFIABLE word -- "
        "an ordinary dictionary word in any language, an established "
        "domain term, OR a generic geographic / positional descriptor "
        "(Mountain, North, South, East, West, Lake, River, Creek, "
        "Valley, County, City, Town, Street, Avenue, Road, ...) -- "
        "then a grounded fix to its standard spelling is `valid`, "
        "EVEN when the value as a whole names a specific place, "
        "person, company, school, or agency. The misspelling of a "
        "generic word is a real, verifiable typo no matter what the "
        "value names (`Eagle Mountian` -> `Eagle Mountain` "
        "[Mountain], `Brookdale Nrth` -> `Brookdale North` [North], "
        "`RIVERTON ENVIROMENTAL` -> `RIVERTON ENVIRONMENTAL`, "
        "`Ashford Country Council` -> `Ashford County Council`, "
        "`MERIDIAN INDEPENDANT` -> `MERIDIAN INDEPENDENT`, "
        "`Maple Cottge Lane Inc.` -> `Maple Cottage Lane Inc.` "
        "[Cottage], `Hillcrest Developmental Agency` -> `Hillcrest "
        "Development Agency` [Development]). Do NOT reject a "
        "common-word fix merely because the surrounding value names a "
        "place, company, school, or agency.\n"
        "      - ONLY when the edited token is itself a SPECIFIC, "
        "UNVERIFIABLE name word -- an unusual surname, a specific "
        "settlement / county / region name, or a brand / product / "
        "organisation coinage whose correct spelling you could not "
        "confirm without external world knowledge about that "
        "particular entity -- is a character-level change to it OUT "
        "OF SCOPE and scored `fp`, however obvious the fix looks "
        "(`Cincinatti` -> `Cincinnati` [a specific place name], "
        "`Microsft` -> `Microsoft` [a brand], `Kowalczek` -> "
        "`Kowalczyk` [a surname], `Brennon Middle` -> `Brennan "
        "Middle` [surname token inside an org name], "
        "`Felis silvestrix` -> `Felis silvestris` [scientific "
        "binomial]). The SINGLE "
        "decision is the VERIFIABILITY of the corrected token: if it "
        "is a real word you can confirm from general or domain "
        "knowledge (including a generic geographic descriptor like "
        "Mountain or North), score `valid`; only if confirming it "
        "would require knowing a specific name's spelling, score "
        "`fp`. Decide from the TOKEN alone -- never from the column "
        "header or from whether the whole value is a name.\n"
        "  * **Spacing and structural defects are IN SCOPE even on "
        "proper-noun values** -- they are recognisable from the "
        "string alone without knowing the canonical name, so score "
        "them `valid`:\n"
        "      - a run of single-letter tokens spaced apart, closed "
        "up, keeping any `&` / punctuation (`I B M` -> `IBM`, "
        "`U S A` -> `USA`);\n"
        "      - a wrongly-split or run-together compound "
        "(`OpenInvoices` -> `Open Invoices`, `Eastgate "
        "ShoppingCentre` -> `Eastgate Shopping Centre`);\n"
        "      - an adjacent duplicated word (`Acme Trading Trading` "
        "-> `Acme Trading`);\n"
        "      - doubled / stray punctuation that does not change a "
        "word (`St.. Louis` -> `St. Louis`, `( Region 4 )` -> "
        "`(Region 4)`).\n"
        "  * For ordinary words, you may and should use your own "
        "knowledge of standard language and established domain "
        "terminology when judging whether a proposed correction is "
        "grounded. Do NOT use that knowledge to second-guess a "
        "proper-noun spelling -- per the bullet above, such a "
        "prediction is `fp` in v9 regardless.\n"
        "  * A proposed correction on an ordinary word is ungrounded "
        "when the resulting form is not an attested standard form -- "
        "i.e. the edit produces another arbitrary surface variant "
        "rather than a word recognisable from general or domain "
        "knowledge. Ungrounded corrections are false positives.\n"
        "  * Pure numeric values, dates, and fixed-length identifier "
        "codes do not have spelling typos; substantive edits to them "
        "(changing a digit, reformatting a date) are normally false "
        "positives.\n\n"
        "Using column convention as evidence:\n"
        "  The column_values list is your primary source of evidence "
        "about what this column normally looks like. When the other "
        "values in the column share a consistent punctuation, "
        "spacing, casing, or formatting pattern, a single value "
        "that breaks that pattern is a signal for a possible real "
        "typo (typically spacing-tokenisation or punctuation), "
        "even when the textual difference looks small. If the "
        "proposed correction brings the value INTO line with the "
        "prevailing convention of the column, you may score it as "
        "valid (subject to the proper-noun rule above). Conversely, "
        "if a proposed correction would move a value AWAY from the "
        "prevailing convention, or if no clear prevailing convention "
        "is visible in column_values, the convention argument does "
        "NOT apply and the prediction must be judged on the "
        "rulebook reminders alone. Casing-only, padding-whitespace-"
        "only, and no-op edits remain false positives regardless of "
        "column convention.\n\n"
        "Scoring procedure:\n"
        "  Step 1 -- Parse model_output as JSON. If parsing fails, "
        "or 'typos' is missing or not an array, return Score: 0.0.\n"
        "  Step 2 -- If the 'typos' array is empty, return "
        "Score: 1.0 (the model made no positive claims, so there is "
        "nothing to penalise on precision).\n"
        "  Step 3 -- For each entry in model_output.typos, classify "
        "into ONE of:\n"
        "    * trivial_fp -- (value -> correction) is a no-op (value "
        "equals correction), or differs only in letter case, or "
        "differs only in leading/trailing whitespace. The rulebook "
        "forbids these as typo proposals; treat as a FALSE POSITIVE. "
        "Reward 0.0.\n"
        "    * valid     -- the value IS a typo per the v9 rulebook "
        "AND the proposed correction is a plausible, grounded fix. "
        "Two paths qualify: (a) the corrected TOKEN is an ORDINARY "
        "word and "
        "the correction is the attested standard form (general "
        "dictionary spelling, established domain term, unambiguous "
        "formatting fix) -- this holds even when other tokens in the "
        "value are a proper name (`Saint Marys Hopsital` -> "
        "`Saint Mary's Hospital`); or (b) the value is a PROPER NOUN and the "
        "defect is STRUCTURAL or SPACING (adjacent duplicated word; "
        "doubled / stray punctuation that does not change a word; a "
        "run of single-letter tokens spaced apart, e.g. `I B M` -> "
        "`IBM`; a wrongly-split or run-together compound, e.g. "
        "`OpenInvoices` -> `Open Invoices`) -- the correction removes "
        "the defect without re-spelling the underlying name. "
        "Reward 1.0.\n"
        "    * ambiguous -- the value is an ordinary word that "
        "plausibly might be a typo but the proposed correction is "
        "not uniquely determined (two or more equally defensible "
        "fixes exist), OR the value is a marginal case where "
        "rulebook scope is genuinely unclear. Reward 0.5. Do NOT "
        "use this bucket as a soft landing for proper-noun "
        "character-level corrections -- those are `fp` in v9.\n"
        "    * fp        -- per the rulebook the value is CLEAN "
        "(in-scope content with no typo), OR the corrected TOKEN is "
        "itself a NAME-SPECIFIC word (surname; place / region "
        "coinage; brand / product / org coinage; scientific genus / "
        "species -- NOT a common dictionary or domain word) and the "
        "correction is a character-level edit (see rulebook sec.3 and "
        "the reminder above), OR the proposed correction on an "
        "ordinary word is ungrounded, OR the value is out of scope "
        "for typo detection (pure number, date, identifier code, "
        "etc.) and the edit is substantive. Reward 0.0.\n"
        "  Step 4 -- Return Score: mean(rewards) over all 'typos' "
        "entries. Clamp to [0.0, 1.0].\n\n"
        "Respond ONLY with two lines:\n"
        "Score: <0.0-1.0>\n"
        "Reasoning: <one short sentence per typos entry, comma-"
        "separated>"
    )

    user = (
        "Column: {{item.column}}\n"
        "Source: {{item.source}}\n"
        "Column values (JSON): {{item.column_values_json}}\n"
        "\n"
        "Model output:\n"
        "{{sample.output_text}}"
    )

    return {
        "name": "typo_v9_score_model_extras",
        "type": "score_model",
        "model": grader_model,
        "input": [
            {"role": "developer", "content": developer},
            {"role": "user", "content": user},
        ],
        "range": [0.0, 1.0],
    }


# ---------------------------------------------------------------------------
# Reward shapes -- identical to v7's r1 / r2 / r3 with v9 sub-grader names
# ---------------------------------------------------------------------------

def _variant_formula_v9(variant: str) -> tuple[str, float]:
    """Return (calculate_output expression, pass_threshold) for a variant.

    Identical algebra to v7's `_variant_formula`, with the precision
    sub-grader renamed `typo_v9_score_model_extras` so v9 runs are
    distinguishable in telemetry.

    Notation:
      P = typo_v9_score_model_extras   (LLM precision under v9 rulebook)
      R = typo_v5_python_gold          (gold-set recall)
      T = typo_v5_python_trivial       (mechanical trivial-FP gate)
      C = typo_v5_python_column_cap    (mechanical column-cap gate)
    """
    if variant == "r1":
        return (
            "2.5 * typo_v5_python_trivial * typo_v5_python_column_cap * "
            "(typo_v9_score_model_extras + typo_v5_python_gold)",
            4.0,
        )
    if variant == "r2":
        return (
            "5 * typo_v5_python_trivial * typo_v5_python_column_cap * "
            "2 * typo_v9_score_model_extras * typo_v5_python_gold / "
            "(typo_v9_score_model_extras + typo_v5_python_gold + 1e-6)",
            3.5,
        )
    if variant == "r3":
        return (
            "5 * typo_v5_python_trivial * typo_v5_python_column_cap * "
            "(0.6 * typo_v9_score_model_extras + 0.4 * typo_v5_python_gold)",
            4.0,
        )
    raise ValueError(
        f"unknown v9 reward variant {variant!r}; expected one of "
        f"{V9_VARIANTS!r}"
    )


def build_grader_v9(
    variant: str,
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """Multigrader = (v9 P precision + v5 R recall + trivial + column_cap)."""
    extras = build_score_model_extras_grader_v9(grader_model=grader_model)
    trivial = _v5.build_python_trivial_grader()
    column_cap = _v5.build_python_column_cap_grader()
    gold = _v5.build_python_grader()
    formula, pass_threshold = _variant_formula_v9(variant)
    return {
        "type": "multi",
        "name": f"typo_v9_multigrader_{variant}",
        "graders": {
            "typo_v9_score_model_extras": extras,
            "typo_v5_python_trivial": trivial,
            "typo_v5_python_column_cap": column_cap,
            "typo_v5_python_gold": gold,
        },
        "calculate_output": formula,
        "pass_threshold": pass_threshold,
        "range": [0.0, 5.0],
    }
