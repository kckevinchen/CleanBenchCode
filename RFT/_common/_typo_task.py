"""Typo-detection RFT task assets (v2 -- binary task).

This module bundles everything specific to the typo-detection task so
``experiments.demo_rft`` stays focused on the generic RFT lifecycle.
``experiments.synth_typo_llm`` and ``experiments.demo_rft`` both import
from here.

v2 changes (relative to the frozen v1 in ``RFT/typo/v1/scripts/``):

* **Binary trainee output.** The model emits one of two labels --
  ``typo`` or ``clean`` -- and never ``ambiguous``. The benchmark
  still carries a third gold label ``ambiguous``; when scoring, EITHER
  trainee answer counts as correct for an ambiguous gold cell
  (because the rulebook itself says those cells are unresolvable
  from cell+header alone, so we don't penalise either choice).
* **No trivial-edit training data.** Gold typos whose ``corrected_value``
  differs from ``original_value`` only by whitespace, casing, or
  punctuation are excluded from train/valid (they are catchable by a
  one-line regex and don't exercise the model's spelling judgment).
  See ``is_trivial_typo_correction`` below.

The trainee is asked to classify ONE candidate cell value (with the
column header and a few sibling values for context) into
``typo`` / ``clean``, returning a strict-schema JSON object. The grader
is a ``score_model`` that applies the project rulebook
(``docs/Typo labeling criteria.md``) and rewards both the right label
and, when the cell is a typo, a correction that matches the gold up
to trivial normalisation.

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

# Gold-label space (what the benchmark may carry, what eval rows look
# like). The model is NOT trained on or asked to emit ``ambiguous`` --
# but rows with this gold label remain useful for evaluation, where
# either trainee answer is credited.
TYPO_LABELS = ("typo", "clean", "ambiguous")

# Trainee output space (enforced via the JSON schema and the system
# message). Strictly binary -- the trainee must commit to one of these
# two labels for every input.
TYPO_OUTPUT_LABELS = ("typo", "clean")

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
    "column header alone, BOTH of the following hold:\n"
    "  (a) you can name the specific defect, AND\n"
    "  (b) the intended correction is unambiguous and requires a real "
    "      character-level edit (not merely a layout fix).\n"
    "The two flavours that qualify are SPELLING and PUNCTUATION:\n"
    "  - spelling: a one- or two-character distortion of a word that "
    "    plausibly belongs in this column (e.g., 'Acomodate' -> "
    "    'Accommodate'; 'Pilot Mountatin' -> 'Pilot Mountain').\n"
    "  - punctuation: a wrong or missing punctuation token that "
    "    changes how the value is read (e.g., 'Address,State' -> "
    "    'Address, State'; '(575-) 894-8350' -> '(575) 894-8350').\n"
    "Do NOT flag values whose only difference from the intended form "
    "is whitespace, casing, or trivial punctuation -- those are "
    "normalisation issues caught upstream, not typos. Examples that "
    "are CLEAN under this rulebook even if they look 'off':\n"
    "  - 'New  York' (double space)  -- whitespace normalisation.\n"
    "  - 'STRATFORD-UPON-AVON' vs 'Stratford-upon-Avon' -- casing.\n"
    "  - 'Du Page' vs 'DuPage' (pure spacing) -- not a typo.\n"
    "Use CLEAN for everything else, especially:\n"
    "  - Recognised proper nouns, brand names, person names.\n"
    "  - Fixed-length codes (zip, SKU, FIPS, ISBN, version strings).\n"
    "  - Pure numeric values -- never typos by our definition.\n"
    "  - Date-shaped strings -- never typos by our definition.\n"
    "  - Standard idioms with elided punctuation ('New Years Day').\n"
    "  - Truncated / contracted values where you cannot recover the "
    "    full form ('CLARION SAFETY SYSTE', 'Twp'). The benchmark may "
    "    score these as 'ambiguous' but you must commit to CLEAN here.\n"
    "When in doubt, prefer CLEAN over TYPO."
)


# ---------------------------------------------------------------------------
# Trainee prompt assembly
# ---------------------------------------------------------------------------

TYPO_SYSTEM_INSTRUCTION = (
    "You are a data-quality reviewer evaluating one cell of a CSV "
    "column. Decide whether the cell is a typo or clean under the "
    "rulebook below. You MUST commit to one of those two labels -- "
    "'ambiguous' is not a valid choice. Reply with a single JSON "
    "object only -- no Markdown fences, no surrounding text.\n\n"
    f"{TYPO_RULEBOOK}\n\n"
    "Output JSON keys (strict):\n"
    "  label:      'typo' | 'clean'\n"
    "  correction: intended spelling as a string (only when "
    "              label=='typo'); otherwise null.\n"
    "  flavour:    'spelling' | 'punctuation' (only when "
    "              label=='typo'); otherwise null.\n"
    "  rationale:  one short sentence naming the defect or stating "
    "              why the cell is clean."
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
    if item["gold_label"] not in TYPO_OUTPUT_LABELS:
        # Training records must be in the binary output space -- the
        # benchmark's 'ambiguous' rows are held out for evaluation
        # only. Filter callers (synth, split) must drop them before
        # invoking build_record.
        raise ValueError(
            f"gold_label must be one of {TYPO_OUTPUT_LABELS} for "
            f"training; got {item['gold_label']!r}. "
            "Ambiguous-gold rows are eval-only in this task."
        )
    if item["gold_label"] == "typo" and not item.get("gold_correction"):
        raise ValueError(
            "gold_correction is required when gold_label=='typo' "
            f"(candidate_value={item.get('candidate_value')!r})"
        )
    if item["gold_label"] == "typo" and is_trivial_typo_correction(
        item.get("candidate_value", "") or "",
        item.get("gold_correction", "") or "",
    ):
        # Defence-in-depth: synthesis is supposed to filter these out
        # already (see ``is_trivial_typo_correction``), but emitting a
        # training row whose 'correction' is a regex-fixable layout
        # change just teaches the model bad habits.
        raise ValueError(
            "gold_correction is trivial (whitespace / casing / "
            "punctuation only) and is excluded from training in "
            f"v2: candidate_value={item.get('candidate_value')!r}, "
            f"correction={item.get('gold_correction')!r}"
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
    """Score-model grader applying the binary typo rulebook.

    The grader sees:
      * ``item.column``, ``item.candidate_value``,
        ``item.gold_label``, ``item.gold_correction`` -- the gold row.
      * ``sample.output_text`` -- the trainee's JSON reply
        (binary: ``typo`` | ``clean``).

    Scoring rules:
      * On ``ambiguous`` gold, EITHER trainee label scores 5
        (the cell cannot be uniquely resolved from cell+header alone,
        so we don't penalise either choice).
      * On ``typo`` gold, the trainee gets 5 when both the label and
        the correction match; partial credit when only the label is
        right; full penalty when the label flips to ``clean``.
      * On ``clean`` gold, exact label match scores 5; ``typo`` flip
        scores 2.
    """
    developer = (
        "You are evaluating a data-quality model's JSON output "
        "against a gold label. The model is BINARY -- it must "
        "commit to either 'typo' or 'clean'.\n\n"
        f"{TYPO_RULEBOOK}\n\n"
        "You will receive:\n"
        "  - column:           CSV column header (JSON string).\n"
        "  - candidate_value:  cell value the model judged (JSON string).\n"
        "  - gold_label:       'typo' | 'clean' | 'ambiguous' "
        "(JSON string).\n"
        "  - gold_correction:  intended correction (JSON-encoded; "
        "may be null).\n"
        "  - model_output:     the model's JSON reply (raw text).\n\n"
        "Parse model_output as JSON. The model.label must be "
        "'typo' or 'clean'; any other value (including 'ambiguous') "
        "is a schema violation. Score 0-5:\n"
        "  5: gold_label=='ambiguous' (either model.label is "
        "accepted; the rulebook says these cells are unresolvable);\n"
        "     OR gold_label=='clean' and model.label=='clean';\n"
        "     OR gold_label=='typo' and model.label=='typo' AND "
        "model.correction equals gold_correction up to case and "
        "collapsed whitespace.\n"
        "  4: gold_label=='typo' and model.label=='typo' but the "
        "correction is missing, partially wrong, or off by one "
        "character.\n"
        "  2: gold_label and model.label are opposites "
        "(typo vs clean).\n"
        "  1: JSON parses but required fields are missing, "
        "malformed, OR model.label is 'ambiguous' (schema violation).\n"
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
                        "enum": list(TYPO_OUTPUT_LABELS),
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
                            "'spelling' | 'punctuation' when "
                            "label=='typo'; null otherwise."
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

    For ``typo`` and ``clean`` gold we mirror the gold label exactly.
    For ``ambiguous`` gold (eval-only) we pick ``clean`` -- the rubric
    accepts either binary answer, and clean is the conservative one.
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
        # Binary task: pick the conservative answer. The grader accepts
        # either, so this only needs to be a well-formed binary reply.
        return {
            "label": "clean",
            "correction": None,
            "flavour": None,
            "rationale": (
                "Cell looks defective but cell+header alone cannot "
                "uniquely determine the intended value; defaulting "
                "to clean."
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

    Opposite-label flip (typo <-> clean). For ``ambiguous`` gold the
    grader credits either binary label, so a 'low' reply must instead
    violate the schema -- we emit ``label=='ambiguous'`` which the
    new binary rubric scores 1 (schema violation).
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
    # ambiguous -> grader credits either binary label, so emit a
    # schema-violating 'ambiguous' label to drive the score down.
    return {
        "label": "ambiguous",
        "correction": None,
        "flavour": None,
        "rationale": (
            "Refuses to commit to a binary label (schema violation)."
        ),
    }


def _infer_flavour(original: str, corrected: str | None) -> str:
    """Best-effort flavour inference from (original, corrected) pair.

    v2 binary task only emits two flavours (``spelling`` and
    ``punctuation``). Whitespace-only edits are no longer considered
    typos, so a pair that differs only in whitespace falls back to
    ``spelling`` (which is what the model will say if it still flags
    the cell).
    """
    if not isinstance(original, str) or not isinstance(corrected, str):
        return "spelling"
    if _strip_punct(original) == _strip_punct(corrected) and original != corrected:
        return "punctuation"
    return "spelling"


def _strip_punct(s: str) -> str:
    """Collapse punctuation + repeated whitespace for the flavour heuristic."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip().lower()


# ---------------------------------------------------------------------------
# Trivial-correction filter (v2)
# ---------------------------------------------------------------------------


def is_trivial_typo_correction(original: str, correction: str | None) -> bool:
    """Return True iff (original -> correction) is a pure layout edit.

    A correction is *trivial* when it differs from the original only
    by whitespace, casing, or pure punctuation -- i.e. the alphanumeric
    content is identical case-folded. Such corrections are catchable by
    a one-line regex and don't exercise the model's spelling judgment,
    so v2 excludes them from train/valid (see ``build_record``).

    Examples that ARE trivial (returns True):
      * 'New  York' -> 'New York'         (double space)
      * 'STRATFORD-UPON-AVON' -> 'Stratford-upon-Avon'  (case)
      * 'Du Page'  -> 'DuPage'            (space removal)
      * 'A T & T'  -> 'AT&T'              (whitespace+punct)

    Examples that are NOT trivial (returns False):
      * 'Balnk cell' -> 'Blank cell'      (real character substitution)
      * 'Seacost'    -> 'Seacoast'        (missing letter)
      * 'Address,State' -> 'Address, State' (alphanumeric content
        identical BUT the missing space changes how the value parses;
        we treat punctuation-only with a space change as substantive)

    The ``Address,State`` case is the tricky one: alphanumeric content
    is identical, so by the strict rule it would count as trivial.
    However, the rulebook explicitly calls comma+space restoration a
    'punctuation' typo, so we make an exception: pairs whose only
    diff is the insertion of a separator punct (``,``, ``;``, ``:``,
    ``/``) followed by a space are NOT trivial.
    """
    if not isinstance(original, str) or not isinstance(correction, str):
        return False
    if original == correction:
        return False
    # Carve out the 'Address,State' -> 'Address, State' pattern: only
    # punctuation-space restoration. Detect by stripping spaces from
    # both sides -- if the punct-bearing strings differ, this is a
    # spacing-around-punct edit which the rulebook calls a typo.
    o_no_space = re.sub(r"\s+", "", original)
    c_no_space = re.sub(r"\s+", "", correction)
    o_alnum_lc = "".join(ch for ch in original if ch.isalnum()).lower()
    c_alnum_lc = "".join(ch for ch in correction if ch.isalnum()).lower()
    if o_alnum_lc != c_alnum_lc:
        return False  # real character-level edit
    # Alphanumeric content matches. Now decide whether the difference
    # is purely layout (trivial) or punctuation-changing (non-trivial).
    if o_no_space == c_no_space:
        # Identical once spaces are removed -- pure whitespace/casing.
        return True
    # Some non-whitespace, non-alphanumeric chars differ. If both
    # sides contain the same multiset of punctuation, the only
    # diff is positional whitespace, still trivial. Otherwise the
    # punctuation set changed -- punctuation typo, NOT trivial.
    o_punct = sorted(ch for ch in original if not ch.isalnum() and not ch.isspace())
    c_punct = sorted(ch for ch in correction if not ch.isalnum() and not ch.isspace())
    if o_punct == c_punct:
        return True
    return False
