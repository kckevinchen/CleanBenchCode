# Typo RFT (v1)

Frozen source snapshot of the per-cell **typo / clean / ambiguous**
Reinforcement Fine-Tuning run against an `o4-mini` base model. This folder
preserves the prompts, grader, and scripts needed to understand the run.
Generated datasets and all execution artifacts live in ignored local
directories and are intentionally not versioned.

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
| Data generation scripts  | [scripts](scripts)                          |
| Runtime artifacts        | `outputs/typo_rft/` (local, ignored)        |

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
mean score 5.0 (correct) vs 2.1 (incorrect). The generated calibration
transcript is not versioned.

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

Generated RFT JSONLs use this per-record schema:

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

## Training

Submitted via `experiments/demo_rft.py submit --task typo` against
an `o4-mini` base. Training logs, per-step rewards, job IDs, and deployment
names are runtime state and remain under ignored output directories.

## Eval

`experiments/eval_typo_finetune.py` (snapshot at
[scripts/eval_typo_finetune.py](scripts/eval_typo_finetune.py)) runs the
fine-tuned deployment over `data/eval.jsonl` (1973 rows
from the held-out benchmark split), looks up sibling values from the
source CSVs in `data/source_inconsistency_bing21/`, and writes one
prediction per row plus a summary. Concurrency 8, reasoning effort
medium, api-version `2025-03-01-preview`, AzureCliCredential. Generated
predictions and summaries remain in the ignored output directory.

## Reproducing the bundle

The prompts/grader files are extracted from the source-of-truth module
`experiments/_typo_task.py`. To refresh them after any edit there:

```bash
python -m RFT.typo._dump_assets
```

This refreshes source assets only. Training and evaluation data must be
regenerated into ignored local directories.

## Folder map

```
RFT/typo/v1/
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
│   └── scoring_rubric.md
├── data/                     # generated locally; ignored by Git
├── scripts/                  # snapshot of code that produced this run
│   ├── _typo_task.py
│   ├── synth_typo_llm.py
│   ├── split_typo_benchmark.py
│   ├── demo_rft.py
│   └── eval_typo_finetune.py
└── outputs/                  # runtime artifacts live at repo root; ignored by Git
```
