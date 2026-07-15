# Typo RFT (v1)

Frozen snapshot of the per-cell **typo / clean / ambiguous** Reinforcement
Fine-Tuning run. The trained model is the Azure OpenAI deployment
`o4-mini-2025-04-16-typo-rft-v1` (base `o4-mini`, region `dmxaz-test-aoai`).

Everything you need to understand, reproduce, or audit the run lives in
this folder. Live files that are still being written by the eval run
(`eval/predictions.jsonl`, `eval/summary.json`, `eval/eval_v1_ft.log`)
are symlinks back into `outputs/typo_rft/eval_v1_ft/`; everything else
is a hard copy.

## TL;DR

| Component                | Where                                       |
|--------------------------|---------------------------------------------|
| Rulebook                 | [prompts/rulebook.md](prompts/rulebook.md)  |
| Trainee system message   | [prompts/system_instruction.md](prompts/system_instruction.md) |
| Trainee user template    | [prompts/user_message_template.md](prompts/user_message_template.md) |
| Trainee response schema  | [prompts/response_format.json](prompts/response_format.json) |
| Grader spec (full)       | [grader/grader.json](grader/grader.json)    |
| Grader developer message | [grader/grader_developer.md](grader/grader_developer.md) |
| Grader user template     | [grader/grader_user_template.md](grader/grader_user_template.md) |
| Scoring rubric           | [grader/scoring_rubric.md](grader/scoring_rubric.md) |
| Grader calibration       | [grader/calibration_v1.json](grader/calibration_v1.json) |
| Train set                | [data/train.jsonl](data/train.jsonl) (80 rows) |
| Valid set                | [data/valid.jsonl](data/valid.jsonl) (20 rows) |
| Synth source records     | [data/injections.jsonl](data/injections.jsonl) (100 rows) |
| Held-out demo split      | [data/demo.jsonl](data/demo.jsonl) (30 rows from benchmark) |
| Held-out eval split      | [data/eval.jsonl](data/eval.jsonl) (1973 rows from benchmark) |
| Training log             | [training/rft_typo_v1.log](training/rft_typo_v1.log) |
| Eval predictions (live)  | [eval/predictions.jsonl](eval/predictions.jsonl) |
| Eval summary (live)      | [eval/summary.json](eval/summary.json)      |

## What the trainee does

For a single CSV cell, decide whether the value is

- `typo` — defect named & corrected unambiguously,
- `clean` — no defect under the rulebook,
- `ambiguous` — suspicious but cannot be uniquely resolved,

and, when `typo`, return both the intended `correction` and one of three
`flavour`s (`spelling`, `whitespace`, `punctuation`). The reply is a
single JSON object validated against
[prompts/response_format.json](prompts/response_format.json).

Each prompt shows the column header, source CSV, up to 20 sibling values
from the same column, and the candidate value to judge. The format is
fixed by `experiments._typo_task.build_user_message`; see
[prompts/user_message_template.md](prompts/user_message_template.md).

## The grader

A `score_model` grader (developer + user messages in `grader/`) reads
the gold row via `item.*` template substitution and the trainee's reply
via `sample.output_text`, then writes a 0-5 score. `pass_threshold = 4`.
See [grader/scoring_rubric.md](grader/scoring_rubric.md) for the rubric.

Before kicking off training we sanity-checked the grader against
synthetic correct/incorrect trainee replies: 1.0 agreement,
mean score 5.0 (correct) vs 2.1 (incorrect); see
[grader/calibration_v1.json](grader/calibration_v1.json).

## Data lineage

```
benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl
        │   (2003 rows, label counts {clean 1304, ambiguous 456, error 243})
        │
        ▼
experiments/split_typo_benchmark.py   ── stratified 30 / 1973 split, seed 0
        │
        ├─► data/demo.jsonl              (30 rows, for synth few-shot)
        └─► data/eval.jsonl              (1973 rows, held-out eval split)

corpus: data/Excel50KTables (Bing CSVs)
denylisted sources: all 137 CSVs touched by the benchmark
        │
        ▼
experiments/synth_typo_llm.py --mode llm
        │   (gpt-5.2 / gpt-5.4 / gpt-5.5 producing injection seeds;
        │    20 columns × 5 examples; mix p_typo=0.40 / p_amb=0.15 / p_clean=0.45)
        │
        ├─► data/injections.jsonl        (100 raw injection seeds)
        ├─► data/train.jsonl             (80 RFT-formatted rows; 35 typo / 30 clean / 15 ambiguous)
        └─► data/valid.jsonl             (20 RFT-formatted rows; 5 typo / 9 clean / 6 ambiguous)
```

Per-record schema for the RFT JSONLs (`data/train.jsonl`,
`data/valid.jsonl`):

```jsonc
{
  "messages": [
    {"role": "developer", "content": "<TYPO_SYSTEM_INSTRUCTION>"},
    {"role": "user",      "content": "<column / source / siblings / candidate>"}
  ],
  // top-level fields are exposed to the grader as ``item.*``
  // (JSON-encoded strings -- template substitution is textual)
  "gold_label":      "\"typo\" | \"clean\" | \"ambiguous\"",
  "gold_correction": "<json-encoded string or null>",
  "candidate_value": "<json-encoded string>",
  "column":          "<json-encoded string>",
  "source":          "<json-encoded string>"
}
```

A fully expanded example is in [data/example_train_row.json](data/example_train_row.json).

## Training

Submitted via `experiments/demo_rft.py submit --task typo` against
`o4-mini` base. The full training log (token usage, per-step rewards,
final job status) is in [training/rft_typo_v1.log](training/rft_typo_v1.log).
Resulting Azure deployment: **`o4-mini-2025-04-16-typo-rft-v1`**.

## Eval

`experiments/eval_typo_finetune.py` (snapshot at
[scripts/eval_typo_finetune.py](scripts/eval_typo_finetune.py)) runs the
fine-tuned deployment over [data/eval.jsonl](data/eval.jsonl) (1973 rows
from the held-out benchmark split), looks up sibling values from the
source CSVs in `data/source_inconsistency_bing21/`, and writes one
prediction per row plus a summary. Concurrency 8, reasoning effort
medium, api-version `2025-03-01-preview`, AzureCliCredential.

The `eval/` symlinks point at the live output directory
`outputs/typo_rft/eval_v1_ft/` so this folder stays current as the run
finishes.

## Reproducing the bundle

The prompts/grader files are extracted from the source-of-truth module
`experiments/_typo_task.py`. To refresh them after any edit there:

```bash
python -m RFT.typo._dump_assets
```

(This is a freeze, not a regen of the training data -- the data files
under `data/` are hand-frozen copies from `outputs/typo_rft/`.)

## Folder map

```
RFT/typo/
├── README.md
├── _dump_assets.py           # extracts prompts/ and grader/ from _typo_task.py
├── __init__.py
├── prompts/                  # what the trainee sees
│   ├── rulebook.md
│   ├── system_instruction.md
│   ├── user_message_template.md
│   ├── response_format.json
│   └── labels.json
├── grader/                   # what the score_model grader sees
│   ├── grader.json
│   ├── grader_developer.md
│   ├── grader_user_template.md
│   ├── scoring_rubric.md
│   ├── calibration_v1.json
│   └── calibration_v1.log
├── data/                     # frozen train / valid / demo / eval JSONLs + manifests
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── injections.jsonl
│   ├── synth_manifest.json
│   ├── demo.jsonl
│   ├── eval.jsonl
│   ├── benchmark_split_manifest.json
│   └── example_train_row.json
├── scripts/                  # snapshot of code that produced this run
│   ├── _typo_task.py
│   ├── synth_typo_llm.py
│   ├── split_typo_benchmark.py
│   ├── demo_rft.py
│   └── eval_typo_finetune.py
├── training/
│   └── rft_typo_v1.log
└── eval/                     # symlinks into outputs/typo_rft/eval_v1_ft/
    ├── predictions.jsonl  -> ../../../outputs/typo_rft/eval_v1_ft/predictions.jsonl
    ├── summary.json       -> ../../../outputs/typo_rft/eval_v1_ft/summary.json
    └── eval_v1_ft.log     -> ../../../outputs/typo_rft/eval_v1_ft.log
```
