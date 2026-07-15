"""Typo-detection RFT task assets.

This module bundles everything specific to the typo-detection task so
``experiments.demo_rft`` stays focused on the generic RFT lifecycle.
``experiments.synth_typo_llm`` and ``experiments.demo_rft`` both import
from here.

The trainee is asked to classify ONE candidate cell value (with the
column header and a few sibling values for context) into
``typo`` / ``clean`` / ``ambiguous``, returning a strict-schema JSON
object. The grader is a ``score_model`` that applies the project
rulebook (``docs/Typo labeling criteria.md``) and rewards both the
right label and, when the cell is a typo, a correction that matches
the gold up to trivial normalisation.

Why single-cell (and not whole-column like
``prompts/precision_baseline_typo.yaml`` does)?

* RFT score-model graders are most stable with one well-defined output
  per input. The whole-column variant would require set-based
  precision/recall in the grader, which is harder to make consistent
  across reasoning effort levels.
* Single-cell outputs compose: at inference we run the trained model
  over each distinct value of a column and re-assemble a column-level
  detector if needed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

# Filenames used by both ``prepare`` (which stages train/valid JSONL into
# the demo state dir) and ``submit`` (which uploads them). Kept in this
# module so the synth script and the RFT lifecycle agree on the names.
TYPO_TRAIN_FILE = "typo_train.jsonl"
TYPO_VALID_FILE = "typo_valid.jsonl"

# Allowed labels (also enforced via the trainee's JSON schema). Mirrors
# the benchmark's ``label`` field exactly so gold rows can flow through
# without remapping.
TYPO_LABELS = ("typo", "clean", "ambiguous")

# Allowed typo flavours. These mirror the v3.3 detector prompt plus
# ``punctuation`` (which the benchmark contains TPs for, e.g.
# 'Administrative,Planning' -> 'Administrative, Planning').
TYPO_FLAVOURS = ("spelling", "whitespace", "punctuation")


# ---------------------------------------------------------------------------
# Rulebook -- single source of truth for trainee + grader
# ---------------------------------------------------------------------------
#
# Distilled from docs/Typo labeling criteria.md and cross-checked against
# benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl (label
# counts: 1304 clean / 456 ambiguous / 243 error). Notable adjustments
# from the raw rulebook:
#
# * Punctuation is included as a typo flavour because the benchmark has
#   labelled TPs for it (TDHCA 'Administrative,Planning' ->
#   'Administrative, Planning'; '88.1,88.3,88.5 88.7' -> proper commas).
# * Casing-only changes ('28TH' -> '28th') are NOT promoted as a flavour
#   here because the canonical detector prompt v3.3 explicitly dropped
#   them; the corresponding benchmark column was also dropped.
# * Numeric and date-shaped values are explicitly NOT typos -- the
#   rulebook says so, and benchmark violations (3 total) are noise to
#   be ignored rather than learned.

TYPO_RULEBOOK = (
    "A cell is a TYPO only when, judging from the cell value and "
    "column header alone, you can name the specific defect AND the "
    "intended correction is unambiguous. Three flavours qualify:\n"
    "  - spelling: a one- or two-character distortion of a word that "
    "    plausibly belongs in this column (e.g., 'Acomodate' -> "
    "    'Accommodate'; 'Pilot Mountatin' -> 'Pilot Mountain').\n"
    "  - whitespace: internal repeated spaces, embedded tab or newline "
    "    (e.g., 'New  York'). Leading/trailing whitespace is stripped "
    "    in preprocessing and must NOT be flagged.\n"
    "  - punctuation: missing comma+space between tokens, doubled "
    "    commas, stray hyphens (e.g., 'Address,State' -> "
    "    'Address, State'; '(575-) 894-8350' -> '(575) 894-8350').\n"
    "Use AMBIGUOUS when the cell looks suspicious but the rulebook "
    "cannot uniquely resolve it from cell + header alone:\n"
    "  - Truncated values whose full form you cannot recover "
    "    ('CLARION SAFETY SYSTE', 'Education in Unorganized Terr').\n"
    "  - Contraction-style abbreviations ('Twp', 'Gl Hearn Limited').\n"
    "  - Multiple equally plausible corrections.\n"
    "Use CLEAN for everything else, especially:\n"
    "  - Recognised proper nouns, brand names, person names.\n"
    "  - Fixed-length codes (zip, SKU, FIPS, ISBN, version strings).\n"
    "  - Pure numeric values -- never typos by our definition.\n"
    "  - Date-shaped strings -- never typos by our definition.\n"
    "  - Standard idioms with elided punctuation ('New Years Day').\n"
    "When in doubt, prefer AMBIGUOUS over TYPO."
)


# ---------------------------------------------------------------------------
# Trainee prompt assembly
# ---------------------------------------------------------------------------

TYPO_SYSTEM_INSTRUCTION = (
    "You are a data-quality reviewer evaluating one cell of a CSV "
    "column. Decide whether the cell is a typo, clean, or ambiguous "
    "under the rulebook below. Reply with a single JSON object only "
    "-- no Markdown fences, no surrounding text.\n\n"
    f"{TYPO_RULEBOOK}\n\n"
    "Output JSON keys (strict):\n"
    "  label:      'typo' | 'clean' | 'ambiguous'\n"
    "  correction: intended spelling as a string (only when "
    "              label=='typo'); otherwise null.\n"
    "  flavour:    'spelling' | 'whitespace' | 'punctuation' (only "
    "              when label=='typo'); otherwise null.\n"
    "  rationale:  one short sentence naming the defect or stating "
    "              why the cell is clean / ambiguous."
)


def build_user_message(
    column: str,
    source: str,
    candidate_value: str,
    sibling_values: Iterable[str],
    *,
    max_siblings: int = 20,
) -> str:
    """Format one judgment prompt.

    ``sibling_values`` are other distinct values from the same column
    -- they give the model a sense of the column's value distribution
    so a typo-shaped divergence stands out. Capped to ``max_siblings``
    so the prompt stays bounded for wide columns (longer lists would
    dilute the signal anyway).
    """
    sib = list(sibling_values)[:max_siblings]
    sib_lines = "\n".join(f"  - {v!r}" for v in sib) or "  (none provided)"
    return (
        f"Column header: {column}\n"
        f"Source CSV: {source}\n"
        f"Other distinct values in this column (sample of "
        f"{len(sib)}):\n"
        f"{sib_lines}\n\n"
        f"Candidate value to judge:\n"
        f"  {candidate_value!r}\n\n"
        "Return the JSON object."
    )


def build_record(item: dict[str, Any]) -> dict[str, Any]:
    """Shape one (column, value, gold) example as an RFT JSONL row.

    Required keys in ``item``:
      column, source, candidate_value, sibling_values (list[str]),
      gold_label  ('typo' | 'clean' | 'ambiguous'),
      gold_correction (str | None -- required when gold_label=='typo').

    Top-level fields outside ``messages`` are exposed to the grader
    via the ``item.*`` template namespace. We pass strings only (no
    JSON-typed values) because Azure's grader template substitution
    inserts raw text rather than parsed values.
    """
    if item["gold_label"] not in TYPO_LABELS:
        raise ValueError(
            f"gold_label must be one of {TYPO_LABELS}, "
            f"got {item['gold_label']!r}"
        )
    if item["gold_label"] == "typo" and not item.get("gold_correction"):
        raise ValueError(
            "gold_correction is required when gold_label=='typo' "
            f"(candidate_value={item.get('candidate_value')!r})"
        )

    user_msg = build_user_message(
        item["column"],
        item.get("source", ""),
        item["candidate_value"],
        item.get("sibling_values", []),
    )
    return {
        "messages": [
            {"role": "developer", "content": TYPO_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_msg},
        ],
        # Grader-visible (item.*). All values must be JSON-encodable as
        # strings: the grader template performs textual substitution,
        # not JSON injection, so the surrounding {"...": {{item.x}}}
        # template works only when item.x is itself a quoted-JSON-string.
        "gold_label": json.dumps(item["gold_label"]),
        "gold_correction": json.dumps(item.get("gold_correction")),
        "candidate_value": json.dumps(item["candidate_value"]),
        "column": json.dumps(item["column"]),
        "source": json.dumps(item.get("source", "")),
    }


# ---------------------------------------------------------------------------
# Grader (score_model)
# ---------------------------------------------------------------------------


def build_grader(grader_model: str) -> dict[str, Any]:
    """Score-model grader applying the typo rulebook.

    The grader sees:
      * ``item.column``, ``item.candidate_value``,
        ``item.gold_label``, ``item.gold_correction`` -- the gold row.
      * ``sample.output_text`` -- the trainee's JSON reply.

    Scoring intentionally awards partial credit when the trainee picks
    ``ambiguous`` against a definitive gold (or vice versa) because
    the rulebook itself acknowledges fuzzy boundaries between typo and
    ambiguous. The opposite-label case (typo<->clean) is the only one
    we penalise heavily.
    """
    developer = (
        "You are evaluating a data-quality model's JSON output "
        "against a gold label.\n\n"
        f"{TYPO_RULEBOOK}\n\n"
        "You will receive:\n"
        "  - column:           CSV column header (JSON string).\n"
        "  - candidate_value:  cell value the model judged (JSON string).\n"
        "  - gold_label:       'typo' | 'clean' | 'ambiguous' "
        "(JSON string).\n"
        "  - gold_correction:  intended correction (JSON-encoded; "
        "may be null).\n"
        "  - model_output:     the model's JSON reply (raw text).\n\n"
        "Parse model_output as JSON. Then score 0-5:\n"
        "  5: model.label == gold_label AND, when gold_label=='typo', "
        "model.correction equals gold_correction up to case and "
        "collapsed whitespace.\n"
        "  4: model.label == gold_label but, when gold_label=='typo', "
        "the correction is missing, partially wrong, or off by one "
        "character.\n"
        "  3: model.label is 'ambiguous' when gold is 'typo' or "
        "'clean' (or vice versa) AND the rationale is consistent "
        "with the rulebook (e.g., model cites truncation when value "
        "is truncated).\n"
        "  2: model.label is the opposite of gold_label "
        "('typo' vs 'clean').\n"
        "  1: JSON parses but required fields are missing or "
        "malformed.\n"
        "  0: output is empty or not valid JSON.\n\n"
        "Respond ONLY with two lines:\n"
        "Score: <0-5>\n"
        "Reasoning: <one short sentence>"
    )
    # NOTE: no whitespace inside {{ ... }} braces. Azure's grader
    # template parser does not strip them, and treats
    # ``{{ item.x }}`` as a variable literally named " item.x " --
    # which it then rejects with HTTP 400.
    user = (
        '{"column": {{item.column}}, '
        '"candidate_value": {{item.candidate_value}}, '
        '"gold_label": {{item.gold_label}}, '
        '"gold_correction": {{item.gold_correction}}, '
        '"model_output": {{sample.output_text}}}'
    )
    return {
        "name": "typo_grader",
        "type": "score_model",
        "model": grader_model,
        "input": [
            {"role": "developer", "content": developer},
            {"role": "user", "content": user},
        ],
        # Pass at >=4 so the trainee gets full reward only when it
        # nails both label and correction (or label without correction
        # for non-typos).
        "pass_threshold": 4,
        "range": [0.0, 5.0],
    }


def build_response_format() -> dict[str, Any]:
    """Strict JSON schema for the trainee's reply.

    Strict mode (per OpenAI docs) requires every property to appear in
    ``required`` and ``additionalProperties: false``; nullability is
    expressed via union types ``["string", "null"]``.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "typo_judgment",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": list(TYPO_LABELS),
                    },
                    "correction": {
                        "type": ["string", "null"],
                        "description": (
                            "Intended spelling when label=='typo'; "
                            "null otherwise."
                        ),
                    },
                    "flavour": {
                        "type": ["string", "null"],
                        "description": (
                            "'spelling' | 'whitespace' | 'punctuation' "
                            "when label=='typo'; null otherwise."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One short sentence.",
                    },
                },
                "required": ["label", "correction", "flavour", "rationale"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


# ---------------------------------------------------------------------------
# Local grader simulation -- used by the ``calibrate-grader`` subcommand
# ---------------------------------------------------------------------------


def render_grader_user_message(
    *,
    column: str,
    candidate_value: str,
    gold_label: str,
    gold_correction: str | None,
    model_output_text: str,
) -> str:
    """Build the grader's user-turn content as the training service does.

    The training service performs textual substitution into the grader's
    user-message template (which is itself a JSON-shaped string).
    Mirroring that locally lets ``calibrate-grader`` run the grader
    model directly via the standard chat API and inspect its output
    BEFORE we burn $$$ on a training run.
    """
    payload = {
        "column": column,
        "candidate_value": candidate_value,
        "gold_label": gold_label,
        "gold_correction": gold_correction,
        "model_output": model_output_text,
    }
    return json.dumps(payload, ensure_ascii=False)


def grader_developer_message() -> str:
    """Developer-turn content of the grader, for local replay."""
    return build_grader("ignored")["input"][0]["content"]


_SCORE_RE = re.compile(r"score\s*[:=]\s*([0-5](?:\.\d+)?)", re.IGNORECASE)


def parse_grader_score(text: str) -> float | None:
    """Pull the ``Score: X`` value out of a grader reply, else None."""
    if not text:
        return None
    m = _SCORE_RE.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(5.0, v))


# ---------------------------------------------------------------------------
# Synthesizing trainee responses for calibration
# ---------------------------------------------------------------------------


def synthesize_correct_response(gold: dict[str, Any]) -> dict[str, Any]:
    """Build a model-reply JSON that the grader should score ~5.

    The fake reply mirrors the gold exactly. For ``ambiguous`` gold we
    leave correction null (matching the benchmark convention -- all
    benchmark ambiguous rows have ``corrected_value == null``).
    """
    label = gold["label"]
    if label == "typo":
        return {
            "label": "typo",
            "correction": gold.get("corrected_value"),
            "flavour": _infer_flavour(
                gold.get("original_value", ""),
                gold.get("corrected_value", ""),
            ),
            "rationale": (
                "Matches the gold correction; defect is "
                f"{_infer_flavour(gold.get('original_value', ''), gold.get('corrected_value', ''))} "
                "under the rulebook."
            ),
        }
    if label == "ambiguous":
        return {
            "label": "ambiguous",
            "correction": None,
            "flavour": None,
            "rationale": (
                "Cell looks defective but cell+header alone cannot "
                "uniquely determine the intended value."
            ),
        }
    # clean
    return {
        "label": "clean",
        "correction": None,
        "flavour": None,
        "rationale": "No identifiable defect under the rulebook.",
    }


def synthesize_incorrect_response(gold: dict[str, Any]) -> dict[str, Any]:
    """Build a model-reply JSON that the grader should score low.

    Opposite-label flip (typo <-> clean) where possible. When gold is
    ``ambiguous`` we pick ``typo`` with a fabricated correction so the
    grader has to decide whether the model is overclaiming.
    """
    label = gold["label"]
    if label == "clean":
        return {
            "label": "typo",
            "correction": gold.get("original_value", "") + "_x",
            "flavour": "spelling",
            "rationale": "Asserts a defect that the rulebook does not support.",
        }
    if label == "typo":
        return {
            "label": "clean",
            "correction": None,
            "flavour": None,
            "rationale": "Misses a defect that the rulebook clearly identifies.",
        }
    # ambiguous -> swing to typo (overclaim)
    return {
        "label": "typo",
        "correction": gold.get("original_value", "") + "_x",
        "flavour": "spelling",
        "rationale": "Overclaims a defect for an inherently ambiguous cell.",
    }


def _infer_flavour(original: str, corrected: str | None) -> str:
    """Best-effort flavour inference from (original, corrected) pair.

    Cheap heuristic used only to populate synthetic 'correct' responses
    for calibration; not a substitute for the model's own classification.
    """
    if not isinstance(original, str) or not isinstance(corrected, str):
        return "spelling"
    if original.replace("  ", " ") == corrected or "\t" in original or "\n" in original:
        return "whitespace"
    if _strip_punct(original) == _strip_punct(corrected) and original != corrected:
        return "punctuation"
    return "spelling"


def _strip_punct(s: str) -> str:
    """Collapse punctuation + repeated whitespace for the flavour heuristic."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip().lower()
