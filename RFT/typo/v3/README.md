# Typo RFT — v3 bundle (per-column audit, English-only)

Self-contained snapshot of the third iteration of the typo
detection RFT experiment. Created **2026-05-19**.

## What changed vs [`RFT/typo/v2/`](../v2)

| Aspect | v2 | v3 |
|---|---|---|
| Granularity | Per-cell (one API call per candidate value) | **Per-column** (one API call per `(csv, column)` pair, audits all distinct values together) |
| Trainee output | `label ∈ {typo, clean}` + optional `correction` | `{typos: [{value, correction, flavour, rationale}, ...]}` — full set of typos in the column (empty array = clean column) |
| Grader | Score-model LLM judging label match | **Python grader** — deterministic set-F1, score = F1 × 5 |
| Ambiguous credit | Either binary label accepted | Predictions on ambiguous values removed from BOTH pred and gold (credited neither TP nor FP) |
| Language scope | Multilingual (whatever was in v2's benchmark) | **English-only** — non-ASCII headers and predominantly non-ASCII columns are dropped at split and synth time |
| Training corpus | 150 cell-level rows | 80 train + 20 valid columns |
| Eval corpus | 1827 cells (95 typo / 1283 clean / 449 ambiguous) | 199 columns (43 has-typo / 86 clean / 70 ambiguous-only) |

The v2 bundle is **not** modified — `experiments/_typo_task.py`
remains the live v2 definition. v3 lives in
`experiments/_typo_v3_task.py` and the new sibling scripts
`split_typo_benchmark_v3.py`, `synth_typo_llm_v3.py`, and
`eval_typo_finetune_v3.py`.

## Why these changes

**1. Per-column granularity.** v2 sends the column header + a
sibling sample with every cell call. For an N-cell column the same
context is re-tokenised N times. The Bing21 eval corpus has ~220
distinct `(csv, column)` pairs spread across ~1857 cells, so v3
makes ~8× fewer API calls at inference. Per-column also gives the
trainee the natural signal — "this value looks misspelled relative
to the column's *other* values" — directly, with frequencies, so
rare typos stand out against common canonical forms.

**2. Python grader.** v2's score-model grader is an LLM that
introduces variance and burns budget on every training step. The v3
task is fully deterministic once the trainee output is parsed —
set comparison with a fixed normalisation rule — so a Python
grader is both cheaper and less noisy. The grader is ~80 lines of
code in `grader/grader_source.py`; it never raises (errors return
`0.0`) and always returns a number in `[0.0, 5.0]`.

**3. English-only.** The v3 rulebook is written for English typos
(spelling distortions, missing punctuation in English idioms). On
multilingual columns the model is asked to distinguish "real typo"
from "legitimate non-English glyph" with zero signal. The filter
is applied at:

* `split_typo_benchmark_v3.py` — columns whose header has any
  non-ASCII alphabetic character, or whose values are less than
  98% ASCII-alphabetic (by character count), are skipped.
* `synth_typo_llm_v3.py` — same filter is applied during CSV
  scan, so the synth corpus is constrained to English columns.

This is a heuristic (no `langdetect` dependency). It tolerates the
occasional foreign name in an otherwise-English column ("São Paulo"
in a US-state column passes) but rejects Vietnamese (Latin-with-
heavy-diacritics), Cyrillic, CJK, and similar outright. From the
220 v2-eval columns, exactly 1 was dropped; from the v3 synth
corpus scan, many CSVs were skipped (83 scanned vs 53 without the
filter for the same `n_columns=100` budget).

## Layout

```
RFT/typo/v3/
├── README.md                       # this file
├── __init__.py                     # marker only
├── _dump_assets.py                 # regenerates prompts/ + grader/
│
├── prompts/                        # what the trainee sees
│   ├── rulebook.md                 # TYPO_V3_RULEBOOK
│   ├── system_instruction.md       # full developer-turn text
│   ├── user_message_template.md    # build_user_message example
│   ├── response_format.json        # strict JSON schema {typos: [...]}
│   └── labels.json                 # flavour enum + scoring rules
│
├── grader/                         # Python grader (NOT a score-model)
│   ├── grader.json                 # full grader spec {type: python, ...}
│   ├── grader_source.py            # the actual sandboxed grade() code
│   └── scoring_rubric.md           # set-F1 rubric (score = F1 * 5)
│
├── data/
│   ├── train.jsonl                 # 80 columns (22 has_typo / 58 clean)
│   ├── valid.jsonl                 # 20 columns ( 2 has_typo / 18 clean)
│   ├── injections.jsonl            # 100 rows -- provenance of every synth column
│   ├── synth_manifest.json         # what synth_typo_llm_v3 did
│   ├── demo.jsonl                  #  20 columns (real benchmark, used as few-shot pool)
│   ├── eval.jsonl                  # 199 columns held-out benchmark
│   ├── benchmark_split_manifest.json
│   └── example_train_row.json      # pretty-printed first typo-bearing train row
│
└── scripts/                        # frozen snapshots used to build this bundle
    ├── _typo_v3_task.py
    ├── _rft_common.py              # shared RFT plumbing (client, state, upload)
    ├── synth_typo_llm_v3.py
    ├── split_typo_benchmark_v3.py
    ├── eval_typo_finetune_v3.py
    ├── typo_rft.py                 # typo-specific RFT CLI (use this for v3)
    └── demo_rft.py                 # generic Countdown reference demo
```

## How this bundle was produced

All commands run from the repo root (LLM steps assume `az login`
has been completed):

```bash
# 1. Aggregate v2 cell-level eval into v3 per-column rows
#    (English-only filter is ON by default).
python -m experiments.split_typo_benchmark_v3 \
    --output-dir outputs/typo_rft/benchmark_split_v3 \
    --seed 0

# 2. LLM-synthesize per-column training data matching real-world
#    distribution (~22% columns have typos; 1-2 typos per typo
#    column). English-only filter is ON by default.
python -m experiments.synth_typo_llm_v3 \
    --mode llm \
    --n-columns 100 \
    --p-inject-typos 0.22 \
    --min-typos-per-column 1 \
    --max-typos-per-column 2 \
    --output-dir outputs/typo_rft/synth_v3 \
    --seed 0

# 3. Refresh prompts/ and grader/ in this bundle
python -m RFT.typo.v3._dump_assets

# (the cp lines that staged the JSONL and frozen scripts into
#  RFT/typo/v3/data and RFT/typo/v3/scripts are not repeated here.)
```

### Distribution targets vs achieved

The v3 split exposes the real-world Bing21 typo distribution:

| Stratum | Real-world (v2 eval) | v3 synth target | v3 synth achieved |
|---|---|---|---|
| has_typo columns | 21.8% (48 / 220) | 22% | 24% (24 / 100) |
| Mean typos per typo column | 1.71 | 1.5 (uniform [1,2]) | 1.50 |
| Modal typos per typo column | 1 (62%) | 1 (50%) | 1 (50%) |

The mean of 1.71 in real-world is dragged up by ~6 outlier columns
with 5–9 typos each. v3 synth caps K at 2 to bias toward the modal
case; the model can still generalise to higher K at inference because
the grader is set-F1 and rewards full recall.

## Fine-tuning

Use `experiments.typo_rft` (the typo-specific RFT CLI; `experiments.demo_rft`
remains scoped to the generic Countdown reference demo). Pass `--task typo_v3`:

```bash
python -m experiments.typo_rft prepare --task typo_v3 \
    --train-jsonl RFT/typo/v3/data/train.jsonl \
    --valid-jsonl RFT/typo/v3/data/valid.jsonl \
    --output-dir outputs/rft_typo_v3

python -m experiments.typo_rft upload --task typo_v3 \
    --output-dir outputs/rft_typo_v3

python -m experiments.typo_rft submit --task typo_v3 \
    --output-dir outputs/rft_typo_v3 \
    --suffix typo-rft-v3
```

`build_grader()` and `build_response_format()` are picked up from
the live `experiments._typo_v3_task`. The grader is a Python grader
(`type: python`); no grader-model deployment is needed.

## Evaluating

```bash
python -m experiments.eval_typo_finetune_v3 \
    --eval-jsonl RFT/typo/v3/data/eval.jsonl \
    --deployment <v3-deployment-name> \
    --output-dir outputs/typo_rft/eval_v3_ft
```

The eval script reports:

* **Column-level** macro/micro precision/recall/F1 over the typo
  sets (the primary v3 metric).
* **Per-cell projection** (binary `typo` vs `clean`) for direct
  comparability with v2. The ambiguous-either-correct rule is
  applied (any prediction on an ambiguous gold cell is credited).

## Caveats and known limitations

* **Training corpus is small** (100 columns total). For a head-to-
  head with v2 we scoped to a comparable data volume; if the v3
  fine-tune underperforms, re-run step 2 with larger `--n-columns`
  (the LLM-synth cost scales roughly linearly with
  `n_columns * p_inject_typos`).
* **English-only is a hard filter.** Disable with
  `--no-english-only` at both the split and synth CLIs if you want
  a multilingual v3 variant; you'll then also need to broaden the
  v3 rulebook (`TYPO_V3_RULEBOOK` in `_typo_v3_task.py`) to cover
  non-English typo flavours.
* **Synth is biased toward spelling typos.** The LLM strongly
  prefers `flavour == "spelling"` over `"punctuation"`; the
  injection set is 100% spelling. Real-world data has a few
  punctuation typos (`"Address,State"` → `"Address, State"`).
  If recall on punctuation is a target, augment training with a
  deterministic-mode pass.
* **K=2 max is a cap.** Real-world has rare columns with 5–9
  typos. The grader is set-F1, so the model can learn to emit
  larger sets at inference if prompted with a high-typo column,
  but it has no training signal for K > 2. If high-K columns
  matter, re-run synth with a wider `--max-typos-per-column`
  range and a long-tailed sampler (current is uniform).
