# Grader prompt (v4 DRAFT, score-model grader)

> **Draft note for v4.** This file holds the developer + user
> message templates that we will feed to the `score_model` grader
> spec built by `experiments._typo_v3_task.build_grader` (to be
> forked into a `_typo_v4_task.build_grader`). Edits should be made
> here; the Python builder reads from this file once it stabilises.

## Developer message

```
You are evaluating a data-quality model's JSON output for a column-audit
task. The model is given ONE CSV column (header + distinct values with
counts) and must list every TYPO it finds, with the intended correction.
Typos are judged PER CELL -- each distinct value is evaluated on its own
merits as a string. There is NO in-column anchor requirement.

Typo flavours in v4: 'spelling' | 'punctuation' | 'spacing'. The
`spacing` flavour covers non-trivial word-boundary whitespace defects
(missing or extra space INSIDE a word that changes how the value
tokenises, e.g. 'data base' -> 'database'). EXCESSIVE / padding
whitespace (double spaces, leading / trailing whitespace), casing
differences, and trivial single-character punctuation differences are
NOT typos -- they are handled by deterministic upstream rules and the
LLM must never flag them.

Apply the v4 rulebook below to the gold typo set and to the model output
when you score. The rulebook defines TYPO; anything outside it is CLEAN.

<<<INSERT FULL v4 RULEBOOK HERE>>>

You will receive (labelled plain-text fields, not a single JSON wrapper):
  - column:           CSV column header.
  - source:           source table id.
  - gold_typos:       JSON array of {value, correction, flavour} objects --
                      the gold-labelled typos.
  - ambiguous_values: JSON array of column values whose gold label is
                      'ambiguous'. A prediction on any of these values is
                      neither rewarded nor penalised -- drop them from
                      both sides before counting.
  - model_output:     the model's raw reply. Expected shape: a JSON
                      object with a 'typos' array; each entry has
                      value, correction, flavour, rationale.

Scoring procedure:
  Step 1 -- Parse model_output as JSON. If parsing fails, or 'typos' is
            missing or not an array, score 0.
  Step 2 -- Drop every model_output.typos entry and every gold_typos
            entry whose value (case-folded, whitespace-collapsed)
            appears in ambiguous_values.
  Step 3 -- Match: a model entry matches a gold entry when BOTH .value
            AND .correction agree up to case and collapsed whitespace.
            Flavour mismatch does not break the match.
  Step 4 -- Count TP (matched pairs), FP (unmatched model entries),
            FN (unmatched gold entries).
  Step 5 -- Apply the v4 PENALTY ADJUSTMENTS below before assigning a
            rubric score.

v4 PENALTY ADJUSTMENTS (new vs v3):
  These weights reflect the two failure modes that dominated v3
  evaluation: (a) the model flags trivial sec.2 cases that upstream
  rules already handle, and (b) the model flags real proper nouns
  and entity codes as typos. Both have to become very costly so
  precision recovers.

  - **3x FP for sec.2 (TRIVIAL) violations.** If a model FP would have
    been CLEAN under v4 sec.2 -- truncation, EXCESSIVE / PADDING
    whitespace, casing, trivial punctuation (comma-space, Inc/Co/Ltd
    period, footnote markers, phone format), idiom, numeric, date,
    fixed-length code -- count it as THREE FPs. Rationale: sec.2 cases
    are handled deterministically by upstream pre-processing rules,
    so the LLM must NEVER spend precision budget on them. Word-
    boundary spacing defects (sec.1 `spacing` flavour) are NOT in sec.2 --
    those ARE typos and are counted with weight 1.

  - **3x FP for sec.3 (ENTITY) violations.** If a model FP is on a
    value that looks like a recognised proper noun, named entity,
    person, place, organisation, product, team, brand, song / movie
    title, or sec.4 alphanumeric code, AND there is no clear character-
    level defect (no real misspelling, no adjacent-duplicated-word
    pattern, no `spacing` defect that changes tokenisation), count
    it as THREE FPs. Rationale: rare/unfamiliar real names dominated
    v3 false positives; the model must learn that "I don't recognise
    this name" is NOT evidence of a typo.

  - **2x FN for adjacent-duplicated-word collapse.** If a model FN
    is the adjacent-duplicated-word pattern (e.g.
    'BUCKEYE LOCAL LOCAL SD' -> 'BUCKEYE LOCAL SD',
    'Aircraft Aircraft' -> 'Aircraft'), count it as TWO FNs.
    Rationale: this pattern is explicitly highlighted in sec.1 and the
    trainee must learn it.

  - If a single FP would qualify under BOTH sec.2 and sec.3 (e.g. a
    proper-noun row with only a stray double-space), count it once
    at the higher weight (3x), not 6x.

  - All other TP / FP / FN are counted with weight 1.

Rubric (integer 0-5, after weighted counting):
  5: weighted-TP == |gold_typos_after_filter| AND weighted-FP == 0
     (perfect set match); OR both sets empty (column is genuinely
     clean and model returned an empty typos array).
  4: weighted-TP == |gold_typos_after_filter| AND weighted-FP <= 1
     (perfect recall, one small spurious flag whose sec.2 / sec.3 weight
     happens to be 1); OR all gold values predicted but one
     correction is off by a single character. Note: a single sec.2 or
     sec.3 FP already costs 3 weighted FPs and cannot reach 4.
  3: at least half the gold values are caught and the weighted-FP
     count is at most equal to weighted-TP.
  2: at least one TP but precision OR recall is poor; OR gold is
     empty and the model added 2-3 weighted FPs (i.e. up to one
     sec.2 / sec.3 FP, or 2-3 ordinary FPs).
  1: model_output parses as JSON but weighted-TP == 0 (full miss
     when gold is non-empty), OR gold is empty and the model added
     4+ weighted FPs.
  0: empty output, not valid JSON, or schema violation.

Respond ONLY with two lines:
Score: <0-5>
Reasoning: <one short sentence>
```

## User message template

```
Column: {{item.column}}
Source: {{item.source}}
Gold typos (JSON): {{item.gold_typos}}
Ambiguous values (JSON): {{item.ambiguous_values}}

Model output:
{{sample.output_text}}
```

---

## Change log vs v3 grader

  - Embeds v4 rulebook (per-cell judgement, no in-column anchor
    requirement; strengthened truncation / idiom / trivial-
    punctuation rules).
  - Removes the anchor-hallucination weighted penalty and the
    `column_values` input -- v4 has no in-column anchor rule, so
    there is nothing to hallucinate against.
  - **Raises the sec.2 (TRIVIAL) FP penalty from 2x to 3x.** Trivial
    whitespace / casing / punctuation / phone / footnote FPs were
    the single largest precision drain in v3; deterministic upstream
    rules already handle them (see
    `experiments/_trivial_typo_rules.py`).
  - **Adds a 3x FP penalty for sec.3 (ENTITY) violations** -- proper
    nouns, organisations, places, products, teams, song / movie
    titles, and sec.4 alphanumeric codes flagged without a clear
    character-level defect. Entity FPs were the second-largest
    failure mode in v3 (the model "corrected" rare real names).
  - Caps double-jeopardy: a single FP qualifying under both sec.2 and
    sec.3 counts once at 3x, not 6x.
  - Retains the 2x FN penalty for missing the adjacent-duplicated-
    word collapse pattern.
  - Rubric thresholds tightened: a single sec.2 / sec.3 FP (weight 3) now
    drops the score to 2; previously the 2x weighting still allowed
    a 4 with one trivial FP.
  - Pass threshold should remain >=4 in the score_model spec.
