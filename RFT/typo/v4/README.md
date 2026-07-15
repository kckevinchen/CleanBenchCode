# typo_v4 — column-audit RFT with per-cell judgement

v4 of the typo-detection RFT task. Mirrors `RFT/typo_v3/` layout but
with:

* **Per-cell judgement** — the in-column anchor requirement from v3 is
  removed. Each value is evaluated on its own merits as a string.
* **Three flavours** — `spelling`, `punctuation`, and a new `spacing`
  flavour for word-boundary tokenisation defects
  (`web site` ↔ `website`, `peanutbutter` ↔ `peanut butter`).
* **v4-specific trivial filter** — only pure padding-whitespace and
  pure casing edits are excluded. Spacing-tokenisation edits are now
  substantive (handled under the `spacing` flavour) and are KEPT in
  both gold and predictions.
* **Re-exported benchmark** — built from
  `benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl`, the
  post-cluster-contamination-fix export.

## Layout

```
RFT/typo_v4/
  prompts/
    rulebook.md                # canonical v4 rulebook (single source of truth)
    system_instruction.md      # dumped trainee developer prompt
    user_message_template.md   # shape of the per-column user message
    response_format.json       # strict JSON schema for trainee replies
    labels.json                # flavour enum
    grader_prompt.md           # earlier draft (kept for diff history)
    grader_prompt_calibration.md
  grader/
    grader.json                # uploadable score_model grader spec
  scripts/
    split_typo_benchmark_v4.py # benchmark → eval.jsonl + demo.jsonl
    synth_typo_llm_v4.py       # synth → train.jsonl + valid.jsonl + injections.jsonl
    _dump_assets_v4.py         # regenerate prompts/* + grader/grader.json from _typo_v4_task
  data/
    eval.jsonl                 # 199 per-column records from the fixed benchmark
    demo.jsonl                 # 20 held-out columns (stratified) for synth few-shot
    train.jsonl                # synthesised
    valid.jsonl                # synthesised
    injections.jsonl           # per-synth-row provenance
    manifest.json              # split manifest
    synth_manifest.json        # synthesis manifest
  eval/
    calibrate_grader_v4.py     # earlier rulebook-only calibration harness
```

The task module itself lives at `experiments/_typo_v4_task.py`. That
module is the single source of truth for the rulebook (loaded from
`prompts/rulebook.md` at import time), the trainee system instruction,
the response-format schema, and the grader spec. Everything in
`prompts/` and `grader/grader.json` is dumped from it.

## Regenerating the data

From the project root, with the venv activated:

```bash
# 1. Aggregate the fixed benchmark into per-column records.
python RFT/typo_v4/scripts/split_typo_benchmark_v4.py

# 2. Synthesise training + validation data (deterministic mode by default).
python RFT/typo_v4/scripts/synth_typo_llm_v4.py --mode deterministic --n-columns 120

#    Or LLM-driven for better diversity (uses gpt-5.x ensemble, cached on disk):
python RFT/typo_v4/scripts/synth_typo_llm_v4.py --mode llm --n-columns 120

# 3. Refresh dumped prompts + grader (idempotent).
python RFT/typo_v4/scripts/_dump_assets_v4.py
```

## Current data sizes (post initial regeneration)

| File              | Rows | With gold typos |
|-------------------|------|------------------|
| `eval.jsonl`      | 199  | 50               |
| `demo.jsonl`      | 20   | 4                |
| `train.jsonl`     | 96   | 59               |
| `valid.jsonl`     | 24   | 15               |
| `injections.jsonl`| 120  | n/a              |

Split-time trivial-correction filter dropped 121 v3-trivial gold entries
that survive in v4 only if they were spacing-tokenisation edits.
See `data/manifest.json` for the full breakdown.

## Grader

Single `score_model` grader (same upload SHAPE as v3) with v4 rulebook
+ 5-verdict rubric embedded:

* `gold_match`     — value in gold AND correction matches.
* `gold_wrong_fix` — value in gold but correction differs.
* `valid_extra`    — out-of-gold but rulebook-valid typo.
* `ambiguous_extra`— out-of-gold and rulebook-silent / borderline.
* `fp`             — out-of-gold and rulebook says CLEAN.

Per-prediction reward `+1.0 / +0.3 / +0.5 / 0 / −1.0`, plus per-missed-
gold recall penalty of `−1.0`, linearly mapped to `[0, 5]`.

Azure RFT `score_model` only accepts `gpt-4o-2024-08-06` or
`o3-mini-2025-01-31`. v4 defaults to the former.

A future upgrade to multigrader (Python gold-bookkeeping +
`score_model` extras-judgement) is tracked separately — the current
single-grader shape was chosen to match the v3 upload contract and
unblock first uploads.

## Notes for v3 removal

The v3 directory (`RFT/typo_v3/`) and the v3 task module
(`experiments/_typo_v3_task.py`) remain in place. They will be removed
once v4 has been validated end-to-end on Azure RFT.
