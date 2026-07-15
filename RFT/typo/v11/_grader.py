"""v11 grader -- fully GOLD-FREE: two LLM judges (precision + recall).

v11 exists so the trainee can be trained on UNLABELLED columns. The whole
human benchmark is reserved for eval (no train/valid is carved out of it), so
the grader has no gold set to lean on. The deterministic gold-recall leg that
every prior version used (``typo_v5_python_gold``) is therefore GONE.

The recall signal it provided -- and, critically, the counter-pressure that
stops the model collapsing to "always predict the empty list" -- is replaced
by a second LLM ``score_model`` judge:

  * PRECISION judge (``typo_v11_score_model_precision``): the v9/v10 precision
    judge verbatim -- grades each PROPOSED typo against the rulebook, returns
    mean reward over the proposals (vacuous 1.0 on an empty prediction). It
    only ever sees what the model proposed, so on its own an empty prediction
    scores the maximum -- the degeneracy we must defeat.
  * RECALL judge (``typo_v11_score_model_recall``, NEW): scans EVERY distinct
    value in the column, re-derives the set G of genuine in-scope typos from
    the SAME rulebook, and returns ``caught / |G|`` -- the fraction of G that
    appears in the model's ``typos`` array. An empty prediction on a column
    that has real typos scores ~0, so proposing nothing is no longer safe;
    on a genuinely clean column G is empty and the judge returns 1.0, so
    correct abstention is still rewarded. No gold set is consulted -- the
    truth set is derived from the column evidence under the rulebook.

The recall judge's scope is kept IDENTICAL to the precision judge (same
rulebook, same "verifiability of the corrected token" test) so the two never
contradict, and it is told to be CONSERVATIVE about G (count a value as a
missed typo only when it is clearly in scope with an unambiguous fix) so its
errors bias toward UNDER-counting real typos rather than inventing them --
inventing a typo would punish correct abstention and erode precision, the
exact failure the gold leg used to prevent.

The two deterministic, gold-free gates are KEPT and still multiply in: the
trivial-FP gate (``typo_v5_python_trivial``) and the column-cap gate
(``typo_v5_python_column_cap``, re-added vs the v10 nogold ablations) -- the
latter is cheap insurance against the model spamming proposals to satisfy a
hallucinating recall judge.

Reward shapes mirror the v-series grid with R = LLM-recall instead of
gold-recall: r1 additive (the historical winner), r2 harmonic/F1, r3
precision-weighted. The precision judge and rulebook are v10's verbatim.

NOTE (Azure constraint): the training grader_model must be in the Azure RFT
``score_model`` whitelist (``gpt-4o-2024-08-06`` or ``o3-mini-2025-01-31``).
gpt-5.x can only stand in OFFLINE for behavioural validation
(``RFT/typo/v11/eval/judge_check.py``); it is not a legal training grader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from RFT._common import typo_task as _v5  # shared primitives (no peer-version dep)

V11_VARIANTS: tuple[str, ...] = ("r1", "r2", "r3")


# ---------------------------------------------------------------------------
# Rulebook -- loaded from the v11 fork on disk
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TYPO_V11_RULEBOOK_PATH = (
    _PROJECT_ROOT / "RFT" / "typo" / "v11" / "prompts" / "rulebook.md"
)


def _load_rulebook_v11() -> str:
    if not TYPO_V11_RULEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"v11 rulebook missing at {TYPO_V11_RULEBOOK_PATH}; the v11 "
            f"grader cannot be built without it."
        )
    return TYPO_V11_RULEBOOK_PATH.read_text(encoding="utf-8")


TYPO_V11_RULEBOOK = _load_rulebook_v11()


# ---------------------------------------------------------------------------
# v11 PRECISION judge -- v10's precision judge verbatim (rulebook embedded)
# ---------------------------------------------------------------------------

def build_precision_judge_v11(
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """v11 precision judge (v10's verbatim, v11 rulebook).

    Returns a float in [0, 1] reflecting how rulebook-justified the
    model's extras are. Differences vs the v5/v7 variant:

      * Embeds the v11 rulebook (proper nouns out of scope for
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
        "so the model learns where it over-reached. You use ONLY the v11 "
        "rulebook and the column data you "
        "are shown. You are NOT given any gold set, label set, or "
        "list of known-clean values, and you must not assume one "
        "exists -- every prediction is evaluated against the rulebook "
        "and the column evidence alone.\n\n"
        f"{TYPO_V11_RULEBOOK}\n\n"
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
        "prediction is `fp` in v11 regardless.\n"
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
        "    * valid     -- the value IS a typo per the v11 rulebook "
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
        "character-level corrections -- those are `fp` in v11.\n"
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
        "name": "typo_v11_score_model_precision",
        "type": "score_model",
        "model": grader_model,
        "input": [
            {"role": "developer", "content": developer},
            {"role": "user", "content": user},
        ],
        "range": [0.0, 1.0],
    }


# ---------------------------------------------------------------------------
# v11 RECALL judge -- gold-free; the leg that makes an empty prediction unsafe
# ---------------------------------------------------------------------------

def build_recall_judge_v11(
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """Gold-free RECALL judge: scan the column for typos the model MISSED.

    Replaces the deterministic ``typo_v5_python_gold`` leg, which v11 cannot
    use because the training columns are UNLABELLED. An LLM re-derives the set
    G of genuine in-scope typos in the column from the rulebook + the column
    evidence and returns ``recall = caught / |G|`` (the fraction of G that the
    model proposed). Proposing nothing on a column that has real typos scores
    ~0, so the empty-output policy collapses; on a genuinely clean column G is
    empty and the judge returns 1.0, so correct abstention is rewarded.

    Scope is IDENTICAL to the precision judge (same rulebook, same
    "verifiability of the corrected token" test), so the two judges agree on
    what a typo is. The judge is told to be CONSERVATIVE about G -- count a
    value only when it is clearly in scope with an unambiguous fix -- so its
    errors bias toward under-counting real typos rather than inventing them
    (inventing one would punish correct abstention and erode precision).

    "Caught" is by VALUE only (not correction quality): a value in G counts as
    caught if it appears as the ``value`` of any ``typos`` entry. Whether the
    proposed *correction* is right is the precision judge's job -- keeping the
    two legs orthogonal.
    """
    if grader_model not in _v5.TYPO_V5_GRADER_MODEL_WHITELIST:
        raise ValueError(
            f"grader_model {grader_model!r} not in Azure RFT score_model "
            f"whitelist {_v5.TYPO_V5_GRADER_MODEL_WHITELIST!r}"
        )

    developer = (
        "You check the RECALL of a typo-audit model: given a CSV column's "
        "COMPLETE distinct values and the model's predicted typos, find the "
        "CLEAR in-scope typos the model MISSED. You use ONLY the v11 rulebook "
        "and the column shown. You are NOT given any gold set or label set -- "
        "you decide which values are genuine typos from the rulebook and the "
        "column evidence alone.\n\n"
        f"{TYPO_V11_RULEBOOK}\n\n"
        "Inputs: column / source identifiers; column_values (JSON array of "
        "[value, count] pairs -- the complete distinct values); model_output "
        "(the audit model's JSON reply with a 'typos' array of {value, "
        "correction, flavour, rationale} -- every value NOT named there is one "
        "the model implicitly called CLEAN).\n\n"
        "Put a value in the missed-set G ONLY when a TOKEN in it is clearly "
        "one of:\n"
        "  * a misspelled COMMON word -- a real dictionary word in any "
        "language, an established domain term, or a generic geographic / "
        "positional descriptor (Environmental, Independent, Development, "
        "Hospital, Mountain, North, County, Witness, Capital, Project, ...) "
        "rendered with an obvious spelling slip; OR\n"
        "  * a STRUCTURAL / SPACING defect visible from the string -- an "
        "adjacent duplicated word, doubled / stray punctuation, a missing "
        "list separator, spaced single-letter tokens (`I B M`), or a "
        "run-together / wrongly-split compound.\n\n"
        "A NAME does not exempt a misspelled COMMON token. Most name / vendor / "
        "agency values are several tokens; inspect EACH token. If one is a "
        "misspelled common word the value IS in G, even though the value as a "
        "whole names something: `GRAHAM INDEPENDANT` (-> Independent), `AQUATIC "
        "ENVIROMENTAL` (-> Environmental), `Pilot Mountatin` (-> Mountain). Do "
        "NOT dismiss a value as 'proper noun, out of scope' without checking "
        "its tokens for a misspelled common word.\n\n"
        "BE CONSERVATIVE -- it is EXPECTED and FINE to leave these OUT of G "
        "(under-flagging is acceptable; INVENTING a typo is not):\n"
        "  * a character slip in a SPECIFIC, unverifiable name token you cannot "
        "confirm -- an unusual surname, a settlement / county / region / brand "
        "coinage, or a scientific (Latin binomial) name (`Cincinatti`, "
        "`Microsft`, `Villareal`, `Felis silvestrix`). Missing these "
        "proper-noun and scientific cases is acceptable by design -- do NOT "
        "flag them even when the slip looks obvious.\n"
        "  * a value mechanically TRUNCATED at a fixed width -- cut off "
        "mid-word so a trailing word is incomplete (`NORTHSIDE CONSTRUC`, "
        "`Metro Office Sup`, `RIVERSIDE UTILIT`). This is a display / "
        "field-width artifact, not a spelling typo -- never flag it.\n"
        "  * anything differing ONLY by casing, padding whitespace, an "
        "apostrophe, an accent / diacritic, an encoding artifact (e.g. "
        "`_x000D_`), or a pure number / date / id-code edit -- NEVER flag "
        "these.\n"
        "  * an abbreviation or shortened form that is not a misspelling "
        "(`Dept`, `Mgmt Svcs`, `Equip Resv`) -- expanding an abbreviation is "
        "NOT a typo fix; leave it out.\n"
        "  * any value you are not confident contains a clearly misspelled "
        "common word or a clear structural defect -- when in doubt, leave it "
        "out.\n\n"
        "Scoring: let P = the values the model predicted (its 'typos' array, "
        "matched trim / casefold / collapsed-whitespace). A value g in G is "
        "CAUGHT iff its text appears in P (correction text is irrelevant -- "
        "detection only). recall = caught / |G|; if G is empty (no clear typos "
        "in the column) recall = 1.0 -- an empty / minimal prediction is the "
        "CORRECT behaviour there.\n\n"
        "Inspect every value's tokens silently, then output EXACTLY two lines "
        "and NOTHING else -- no analysis, no headers, no step text. The final "
        "line must be the labelled Score:\n"
        "Missed: <in-scope typos in G that are NOT in P (value -> fix), or "
        "'none'>\n"
        "Score: <0.0-1.0>"
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
        "name": "typo_v11_score_model_recall",
        "type": "score_model",
        "model": grader_model,
        "input": [
            {"role": "developer", "content": developer},
            {"role": "user", "content": user},
        ],
        "range": [0.0, 1.0],
    }


# ---------------------------------------------------------------------------
# FUSED precision+recall judge -- ONE score_model call.
#
# Every v11 RFT job hit Azure's NON-RETRIABLE limit: "Runtime exceptions in
# Python grader ... ~20% ... FileNotFoundError: /mnt/data/<uuid>_data.json" --
# Azure failing to STAGE a python grader's per-invocation data file (training
# reward itself was healthy, ~4.3-4.8). Two fixes were applied:
#
#   1. Fuse the two LLM judges (precision + recall) into ONE score_model call
#      (``build_pr_judge_v11``, F1(P,R)) -- the proven v10 ``nogoldrec`` LLM
#      recipe. This ALONE did NOT fix it: the failing component is a *python*
#      grader, whose ``grade()`` is fully defensive (try/except -> default,
#      cannot raise ``FileNotFoundError``); the run carrying the fused judge but
#      still TWO python gates failed identically. The LLM-judge count was a
#      misdiagnosis -- it is not the failure source.
#   2. Fuse the two python gates (trivial + column-cap) into ONE subprocess
#      (``_v5.build_python_gates_grader``) so Azure stages ONE /mnt/data file
#      per rollout instead of two. v10 ``nogold``/``nogoldrec`` -- which trained
#      + deployed -- used exactly ONE python grader; v11's extra column-cap
#      subprocess was the controlled difference. Reward math is unchanged (the
#      fused gate returns trivial * cap).
#
# The judge is built by patching the v11 precision judge to add a recall pass
# and return F1(precision, recall). Patch anchors are the precision judge's own
# text.
# ---------------------------------------------------------------------------

_INTRO_PRECISION_ONLY = (
    "The audit model is tuned for RECALL and "
    "deliberately over-proposes; your job is to be PRECISE -- judge "
    "each predicted typo on its own merits and mark the wrong ones, "
    "so the model learns where it over-reached."
)
_INTRO_PRECISION_RECALL = (
    "Judge the model on BOTH precision and recall. For PRECISION, judge "
    "each predicted typo on its own merits and mark the wrong ones. For "
    "RECALL, you must ALSO scan every distinct value in column_values and "
    "find genuine typos (per the rulebook) that the model did NOT include "
    "in its 'typos' array -- a value the model omitted is implicitly "
    "claimed CLEAN, and omitting a real typo is a recall error that must "
    "be penalised. Proposing nothing is therefore NOT safe: it is scored "
    "as missing every true typo in the column."
)
_STEP2_EMPTY_SHORTCUT = (
    "  Step 2 -- If the 'typos' array is empty, return "
    "Score: 1.0 (the model made no positive claims, so there is "
    "nothing to penalise on precision).\n"
)
_STEP2_NO_SHORTCUT = (
    "  Step 2 -- Do NOT short-circuit on an empty 'typos' array. An empty "
    "array means the model claims the whole column is clean; you must "
    "still run the recall check in Step 5, which will penalise it for "
    "every genuine typo present.\n"
)
_STEP4_PRECISION_ONLY = (
    "  Step 4 -- Return Score: mean(rewards) over all 'typos' "
    "entries. Clamp to [0.0, 1.0].\n\n"
)
_STEP4_PRECISION_RECALL = (
    "  Step 4 -- PRECISION. Let precision = mean(rewards) over the "
    "'typos' entries from Step 3. If 'typos' is empty, set precision = "
    "1.0 (vacuous -- recall in Step 5 carries the empty case).\n"
    "  Step 5 -- RECALL. Independently scan EVERY distinct value in "
    "column_values and, using the SAME rulebook scope as above, "
    "identify the set G of values that ARE genuine in-scope typos (a "
    "value is in G only if a correction to it would score 'valid' under "
    "Step 3 -- so name-specific character slips, casing-only, padding-"
    "whitespace-only, pure numbers / dates / id codes are NOT in G). For "
    "each value in G, the model CAUGHT it iff that value appears as the "
    "'value' of a 'typos' entry classified valid or ambiguous in Step 3. "
    "Let caught = number of G values caught and missed = number of G "
    "values not caught. Set recall = caught / (caught + missed). If G is "
    "empty (the column has no genuine typos), set recall = 1.0.\n"
    "  Step 6 -- Return Score: the F1 of precision and recall, "
    "2 * precision * recall / (precision + recall + 1e-6), clamped to "
    "[0.0, 1.0]. If precision and recall are both vacuous (no 'typos' "
    "entries AND G empty), return Score: 1.0.\n\n"
)


def build_pr_judge_v11(
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """Single fused precision+recall judge: returns F1(P, R) in [0, 1].

    Reuses the v11 precision judge verbatim (same rulebook + per-prediction
    classification) and patches its developer message to add a recall pass
    over all ``column_values`` -- so omitting a genuine typo is penalised and
    an empty prediction collapses -- in ONE ``score_model`` call. Same recipe
    as v10 ``nogoldrec`` (which trained + deployed).
    """
    base = build_precision_judge_v11(grader_model=grader_model)
    developer = base["input"][0]["content"]
    for old, new in (
        (_INTRO_PRECISION_ONLY, _INTRO_PRECISION_RECALL),
        (_STEP2_EMPTY_SHORTCUT, _STEP2_NO_SHORTCUT),
        (_STEP4_PRECISION_ONLY, _STEP4_PRECISION_RECALL),
    ):
        if old not in developer:
            raise RuntimeError(
                f"v11 precision judge text drifted; cannot patch anchor:\n{old!r}"
            )
        developer = developer.replace(old, new)
    base["name"] = "typo_v11_score_model_pr"
    base["input"][0]["content"] = developer
    return base


def build_grader_v11(
    variant: str,
    grader_model: str = _v5.TYPO_V5_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """Gold-free multigrader = ONE fused P+R LLM judge + ONE fused python gate.

    Both the LLM judges (precision + recall -> ``build_pr_judge_v11``) and the
    python gates (trivial-FP + column-cap -> ``_v5.build_python_gates_grader``)
    are fused into a single grader of each kind, so Azure runs exactly ONE
    score_model call and ONE python subprocess per rollout -- matching the v10
    ``nogoldrec`` recipe that trained + deployed and staying within Azure's
    grader-runtime-exception limit (see the module note above). The reward math
    is unchanged: the fused python gate returns trivial * cap, so
    ``calculate_output`` is still ``5 * trivial * cap * pr``.

    All reward variants share this grader; ``variant`` only tags telemetry.
    ``build_precision_judge_v11`` / ``build_recall_judge_v11`` are kept for the
    offline judge_check harness.
    """
    if variant not in V11_VARIANTS:
        raise ValueError(
            f"unknown v11 reward variant {variant!r}; expected one of "
            f"{V11_VARIANTS!r}"
        )
    pr = build_pr_judge_v11(grader_model=grader_model)
    gates = _v5.build_python_gates_grader()
    return {
        "type": "multi",
        "name": f"typo_v11_multigrader_{variant}",
        "graders": {
            "typo_v11_score_model_pr": pr,
            "typo_v5_python_gates": gates,
        },
        "calculate_output": (
            "5 * typo_v5_python_gates * typo_v11_score_model_pr"
        ),
        "pass_threshold": 4.0,
        "range": [0.0, 5.0],
    }
