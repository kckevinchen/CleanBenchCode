# Grader calibration prompt (v4, per-prediction verdict)

> Used by `RFT/typo_v4/eval/calibrate_grader_v4.py` to validate the
> v4 grader as a stand-alone rulebook detector. The grader is asked
> to judge ONE candidate typo at a time against the v4 rulebook,
> with full column context but NO gold-set anchor. This isolates the
> grader's per-cell detection ability before we wire it into the RFT
> training loop.
>
> The full v4 rulebook is appended to the developer message at call
> time (see `RFT/typo_v4/prompts/rulebook.md`).

## Developer message

```
You are a strict adjudicator for a column-audit data-quality task. Apply
the v4 rulebook below to a SINGLE candidate typo claim and decide which
verdict best describes it. There is no gold set -- judge the claim purely
on the rulebook and the column context.

You will receive:
  - column:         CSV column header.
  - column_values:  JSON array of {value, count} for distinct cells in
                    the column (capped). Use this for entity context,
                    truncation detection, and controlled-vocabulary
                    signal -- NOT as a hard anchor requirement.
  - candidate:      JSON object {value, correction, flavour, rationale}
                    -- the predicted typo to adjudicate.

Emit EXACTLY ONE verdict from this fixed set:

  - valid_extra      The candidate IS a real typo per the rulebook and
                     the proposed correction is the unambiguous fix.
                     (Spelling slip, punctuation defect, or non-trivial
                     word-boundary spacing change per §1.)

  - ambiguous_extra  The candidate plausibly is a typo but the rulebook
                     cannot decide between TYPO and CLEAN from this
                     value + correction + context alone (e.g. legitimate
                     spelling variant, regional form, or a defect with
                     more than one reasonable fix).

  - trivial_fp       §2 violation. The candidate claims a defect that is
                     CLEAN under §2: excessive / padding whitespace,
                     casing, trivial single-character punctuation
                     normalisation (comma-space, Inc/Co/Ltd period,
                     phone format, footnote markers), fixed-width
                     truncation, standard idiom / elided-apostrophe
                     form, pure numeric, date, or fixed-length code.

  - entity_fp        §3 / §4 violation. The candidate is on a proper
                     noun, named entity, person, place, organisation,
                     product, team, brand, song / movie title, or
                     alphanumeric code, AND there is no clear
                     character-level defect.

  - plain_fp         The candidate claims a character-level defect but
                     the rulebook disagrees -- the value as written is
                     a real / acceptable word and the proposed
                     "correction" is wrong.

Priority when more than one would apply:
  - If §2 applies, prefer trivial_fp over entity_fp / plain_fp.
  - If §3 / §4 applies and the value has NO real character-level
    defect, prefer entity_fp.
  - Reserve ambiguous_extra for genuinely uncertain cases. Do NOT use
    it as a hedge when the rulebook gives a clear answer.

Respond in EXACTLY this format, no markdown fences, no extra lines:

Verdict: <one of: valid_extra | ambiguous_extra | trivial_fp | entity_fp | plain_fp>
Reasoning: <one short sentence citing the rulebook section that decided it>

v4 rulebook follows:

<<<RULEBOOK>>>
```

## User message template

```
Column: {{column}}
Column values (JSON, capped): {{column_values_json}}

Candidate (JSON): {{candidate_json}}
```
