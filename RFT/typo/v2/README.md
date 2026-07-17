# Typo RFT — v2 bundle (binary task, no trivial errors)

Reproducible source snapshot of the second iteration of the typo detection RFT
experiment. Generated datasets and runtime artifacts are local and ignored.
Created **2026-05-19**, after the v1 eval
revealed that 62% of the v1 "typo" benchmark rows were trivial
whitespace / casing / punctuation rewrites that a one-line regex
could fix without any model.

## What changed vs [`RFT/typo/v1/`](../v1)

| Aspect | v1 | v2 |
|---|---|---|
| Trainee output | `typo` \| `clean` \| `ambiguous` | `typo` \| `clean` (BINARY) |
| Training corpus | All gold typos | Substantive typos only (drops 146 / 243 trivial pairs) |
| Eval scoring on `ambiguous` gold | Whatever the model said had to match exactly (3 × 3 matrix) | Either binary answer is credited correct |
| Per-class F1 | Reported on 3 classes | Reported on 2 classes (`typo`, `clean`); `ambiguous` reported as a coverage rate |
| Whitespace flavour | A valid typo flavour (`whitespace`) | Removed; never a typo in v2 |
| LLM-synth rulebook | Allowed `ambiguous` and whitespace mutations | Excludes both |

The v1 bundle is **not** modified — it remains the frozen record of
the v1 fine-tune. The live source in `experiments/_typo_task.py` is
now the v2 definition; the v1 task definition can still be read from
[`../v1/scripts/_typo_task.py`](../v1/scripts/_typo_task.py).

## Why these two changes

**1. No trivial errors in training/eval.** The v1 post-mortem found:

- 121 / 239 (50.6%) of gold typos were *whitespace only* — and 117
  of those were a single regex rule: `re.sub(r"\s+", " ", x)`.
- 22 more (9.2%) were punctuation-only (`'Du Page' → 'DuPage'`).
- 1 was casing-only, 4 were a casing+spacing combo.
- Only 91 / 239 (38.1%) were substantive spelling edits.

Training the model on the trivial pairs (a) wastes capacity on a
problem a regex solves better, and (b) creates a calibration bias
toward "over-firing typo" that hurt v1's overall accuracy
(48.3% on a benchmark where 65% of rows were clean).

The new helper `experiments._typo_task.is_trivial_typo_correction`
formalises the filter: a `(original, correction)` pair is *trivial*
when the case-folded alphanumeric content is identical AND the
punctuation multiset is unchanged. The first condition catches
`'Balnk' → 'Blank'` as non-trivial; the second carves out
`'Address,State' → 'Address, State'` as a real punctuation typo.

**2. Binary trainee output.** The benchmark's `ambiguous` label is a
useful annotator escape hatch ("the cell is suspicious but the rule
cannot resolve it") but it's not a useful *model* output: at deploy
time the downstream user wants a commit-to-action label, not a
deferral. v2 forces the trainee to choose between `typo` and `clean`,
and the grader / eval credits *either* answer on `ambiguous` gold
rows (the rulebook itself says those cells are unresolvable, so
penalising one binary answer over the other is arbitrary).

## Layout

```
RFT/typo/v2/
├── README.md                       # this file
├── __init__.py                     # marker only
├── _dump_assets.py                 # regenerates prompts/ + grader/
│
├── prompts/                        # what the trainee sees
│   ├── rulebook.md                 # TYPO_RULEBOOK (v2, no whitespace flavour)
│   ├── system_instruction.md       # full developer-turn text
│   ├── user_message_template.md    # build_user_message example
│   ├── response_format.json        # strict JSON schema (label enum=[typo,clean])
│   └── labels.json                 # gold space vs output space + scoring note
│
├── grader/                         # what the score_model judges
│   ├── grader.json                 # full grader spec (pass_threshold=4)
│   ├── grader_developer.md         # grader developer message
│   ├── grader_user_template.md     # textual {{item.x}} / {{sample.output_text}} template
│   └── scoring_rubric.md           # 0-5 rubric (ambiguous → either-correct)
│
├── data/                            # generated locally; ignored by Git
│   ├── train.jsonl                 # 120 rows  (52 typo / 68 clean)
│   ├── valid.jsonl                 #  30 rows  (10 typo / 20 clean)
│   ├── injections.jsonl            # 150 rows  (provenance: column + mutator metadata)
│   ├── synth_manifest.json         # what synth_typo_llm did
│   ├── demo.jsonl                  #  30 rows  benchmark few-shot pool (drives synth)
│   ├── eval.jsonl                  # 1827 rows held-out benchmark (95 typo / 1283 clean / 449 ambiguous)
│   ├── benchmark_split_manifest.json
│   └── example_train_row.json      # pretty-printed first train row
│
└── scripts/                        # frozen snapshots used to build this bundle
    ├── _typo_task.py
    ├── synth_typo_llm.py
    ├── split_typo_benchmark.py
    ├── demo_rft.py
    └── eval_typo_finetune.py
```

When generated, `data/train.jsonl` and `data/valid.jsonl` are RFT-ready: the
JSONL is exactly the shape Azure expects for `o4-mini` reinforcement
fine-tuning (`messages`, plus item-level grader fields).

## How this bundle was produced

All commands run from the repo root (and the LLM steps assume
`az login` has been completed so `AzureCliCredential` can mint
tokens):

```bash
# 1. Re-split the benchmark with the trivial-typo filter enabled
python -m experiments.split_typo_benchmark \
    --output-dir outputs/typo_rft/benchmark_split_v2 \
    --exclude-trivial-typos

# 2. LLM-synthesize new train/valid in the binary task shape
python -m experiments.synth_typo_llm \
    --mode llm \
    --demo-path outputs/typo_rft/benchmark_split_v2/demo.jsonl \
    --manifest-path outputs/typo_rft/benchmark_split_v2/manifest.json \
    --output-dir outputs/typo_rft/synth_v2 \
    --n-columns 30 --examples-per-column 5 \
    --p-typo 0.55 --p-clean 0.45 --p-ambiguous 0.0 \
    --seed 0

# 3. Refresh prompts/ and grader/ in this bundle
python -m RFT.typo.v2._dump_assets

# (the cp lines that staged the JSONL and frozen scripts into
#  RFT/typo/v2/data and RFT/typo/v2/scripts are not repeated here;
#  see the bundle creation transcript.)
```

## Fine-tuning

Use the same `experiments.demo_rft` lifecycle as v1, but point it at
the v2 data and pick a new RFT job name (don't recycle v1's job ID):

```bash
python -m experiments.demo_rft prepare --task typo \
    --train RFT/typo/v2/data/train.jsonl \
    --valid RFT/typo/v2/data/valid.jsonl \
    --state-dir state/typo_rft_v2

python -m experiments.demo_rft submit  --task typo \
    --state-dir state/typo_rft_v2 \
    --grader-model "<grader-deployment-name>"
```

`build_grader()` and `build_response_format()` are picked up from
the **live** `experiments._typo_task` (which is now v2), so the
submitted job uses the binary schema and the ambiguous-either-correct
grader rubric automatically.

## Evaluating

The eval script (`experiments.eval_typo_finetune`) was updated for v2:

- Predictions outside `{typo, clean}` count as parse failures.
- Per-row accuracy: `pred == gold` OR `gold == 'ambiguous'`.
- Per-class P/R/F1 reported on the binary axis only.
- Ambiguous gold rows reported separately as `ambiguous_coverage`.

```bash
python -m experiments.eval_typo_finetune \
    --eval-jsonl RFT/typo/v2/data/eval.jsonl \
    --deployment <v2-deployment-name> \
    --output-dir outputs/typo_rft/eval_v2_ft
```

Once the v2 fine-tune is trained and evaluated, symlink the
artifacts into `RFT/typo/v2/eval/` to freeze the result (mirroring
the v1 bundle's `RFT/typo/eval/` layout).

## Caveats and known limitations

- **Training corpus is small** (150 examples). v2 was scoped to mirror
  v1's data volume for a like-for-like comparison; if accuracy is
  still limiting, re-run step 2 with larger `--n-columns` and
  `--examples-per-column`.
- **LLM-synth diversity** depends on the demo few-shot pool. With
  146 trivial typos filtered out, the demo set has only 2 substantive
  typos to use as exemplars (the rest are `clean`/`ambiguous`). This
  is a known thin-pool risk — re-running synthesis with the original
  benchmark plus a hand-curated set of substantive typo exemplars
  may produce a more varied training set.
- **`RFT/typo/v1/` bundle's `_dump_assets.py`** imports from the
  live `experiments._typo_task`, which is now v2. If you re-run
  `python -m RFT.typo._dump_assets` it will overwrite v1's
  `prompts/` and `grader/` JSONs with v2 content. Don't do that —
  the v1 prompts/grader on disk are the frozen record of what was
  used for the v1 fine-tune.
