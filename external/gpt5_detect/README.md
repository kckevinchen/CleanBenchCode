# `external/gpt5_detect/` — GPT-5.5 single-prompt cell-error detector

A holistic LLM baseline: instead of the flagship pipeline's one call per
(chunk × error type), this asks **GPT-5.5 to detect ALL four cell-level error
types in a SINGLE prompt** per table — `typo`, `inconsistency`,
`semantic_incompatible`, `numeric_outlier`.

```
external/gpt5_detect/
  prompt.py       builds ONE unified system prompt from prompts/error_types.yaml
  run.py          chunk -> gpt-5.5 (LLMClient) -> parse issues -> predictions JSONL
  tuning.py       deterministic split, dossier, leak scan, prompt validation
  tune_prompt.py  three-agent tuning orchestrator + state monitor
  cache/          per-request raw responses (gitignored; resumable)
```

## Design

- **Unified prompt** (`prompt.py`): the per-type DEFINITION / EXAMPLE / NOTES
  blocks for the four cell types are pulled **verbatim** from
  `prompts/error_types.yaml` (the same taxonomy the flagship pipeline uses) and
  concatenated into one system prompt. Output is a JSON `issues` array, one entry
  per affected **cell** → unambiguous (row, column) mapping for cell scoring.
- **Faithful rendering**: tables are loaded with the REIN-faithful loader and
  rendered with `src.core.data_loader` (`__row__` index + whitespace markers), so
  the model's `row` values map straight back to benchmark rows.
- **Whole-table prompt**: `--chunk-size 100` ≥ max table size (100 rows), so each
  table is **one prompt / one call** (use a smaller size to split big tables).
- **Concurrent + cached**: parallel calls across the Azure endpoint pool
  (`--concurrency`), each chunk cached under `cache/` (resumable; re-runs assemble
  instantly). The cache key includes the model, full system and user prompts, and
  reasoning effort.

## Run (main venv, `az login`)

```bash
python external/rein/prepare_datasets.py            # materialize datasets (once)

python external/gpt5_detect/run.py --chunk-size 100 --concurrency 10
#   --model gpt-5.5            deployment (default)
#   --reasoning-effort medium  low|medium|high
#   --min-confidence low       drop issues below this confidence
#   --limit N                  smoke test on N datasets
#   --dataset-list FILE        exact JSON allowlist (frozen split)
#   --system-prompt-file FILE  use a versioned prompt
#   --tag NAME                 isolate result artifacts

python external/rein/score.py --predictions external/rein/results/predictions_gpt5_detect_gpt55.jsonl
```

## Leakage-gated agentic prompt tuning

`tune_prompt.py` implements a three-agent state machine:

1. The **prompt writer** sees only the current prompt and judge-approved generic
   suggestions.
2. The target **GPT-5.5 detector** runs that prompt on a frozen development split.
3. The **evaluator** may inspect development tables, labels, predictions, and
   metrics, but must emit only high-level, domain-independent suggestions.
4. The **judge** compares those suggestions with the development dossier and
   rejects any source/table/file identifier, row or column location, header,
   value, domain fact, exact format, example, metric, count, or other benchmark
   information.
5. A deterministic scanner and the judge must both pass. Only then does the
   writer receive the suggestions for the next candidate.

The writer never receives a development dossier. No agent receives any test
identifier, table, label, prediction, or metric. Candidate selection uses
development F1; the selected prompt is SHA-256 locked before one final run over
the untouched test split.

The v7 split is deterministic and table-disjoint:

- development: 202 tables (20%, rounded);
- test: 806 tables;
- error-bearing and no-error tables are stratified;
- every error type represented by at least two tables occurs on both sides.

The benchmark fingerprint covers the manifest, every `gt_cells.jsonl`, and every
`dirty.csv`. Resume fails if the benchmark, code, scoring configuration, split,
prompt, predictions, or score artifact changes.

```bash
# Create and inspect the split/call budget without an LLM call.
python external/gpt5_detect/tune_prompt.py \
  --dry-run --rounds 5 --chunk-size 100 \
  --run-dir outputs/prompt_tuning/v7_agentic_gpt55

# Start or resume the five-candidate loop.
python external/gpt5_detect/tune_prompt.py \
  --resume --rounds 5 --chunk-size 100 \
  --run-dir outputs/prompt_tuning/v7_agentic_gpt55

# Monitor without touching the running process.
python external/gpt5_detect/tune_prompt.py \
  --status --run-dir outputs/prompt_tuning/v7_agentic_gpt55
```

Use `--run-full` only if a post-lock all-table artifact is needed. Its score is
not the headline result because it includes the development tables.

### Audit artifacts

Each run directory contains:

- `split.json`, exact development/test allowlists, and benchmark hash;
- `state.json` and timestamped stage transitions;
- per-round prompt, prompt hash, detector output, score, and command logs;
- development-only evaluator dossier;
- every raw/parsed agent response and validation result;
- deterministic scanner and judge verdicts;
- `approved_feedback.json` (the only evaluator output forwarded to the writer);
- `prompt_lock.json` and `locked_prompt.txt`;
- final untouched-test predictions and score.

The dossiers and raw agent traces contain development data and remain under
gitignored `outputs/`.

### Completed v7 agentic run

Run: `outputs/prompt_tuning/v7_agentic_gpt55/`

| candidate | dev P | dev R | dev F1 |
|---:|---:|---:|---:|
| 0 | 0.237 | 0.514 | **0.324** |
| 1 | 0.228 | 0.377 | 0.284 |
| 2 | 0.260 | 0.388 | 0.311 |
| 3 | **0.270** | 0.333 | 0.298 |
| 4 | 0.226 | 0.257 | 0.240 |

Candidate 0 was locked as
`923987b017fd141216ea0733935e5252e56cb4c3d8bf322320d7e78e7e11f595`.
On the untouched 806-table test split:

| prompt | coverage | eligible GT | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| fixed direct prompt | 806 | 1,163 | 0.181 | **0.684** | 0.286 |
| locked GATE-Prompt candidate | 806 | 1,163 | **0.310** | 0.607 | **0.411** |

Test recall by type: typo-spelling 0.921 (441), typo-formatting 0.732
(179), inconsistency 0.379 (601), semantic-incompatible 0.900 (20), and
numeric-outlier 0.000 (1).

GATE-Prompt improves held-out F1 over the pre-specified fixed prompt by 0.124
(43.4% relative), with higher precision and lower recall. The iterative-feedback
result is nevertheless negative: all four feedback-derived candidates scored
below the first candidate on development.

The fixed comparator was produced without new model calls by filtering the
pre-existing full-universe direct-prompt predictions to the frozen test set.
Hashes and provenance are recorded in
`outputs/prompt_tuning/v7_agentic_gpt55/fixed_prompt_test/metadata.json`.

Audit: all 1,816 target calls and 40 agent responses succeeded; seven feedback
attempts were rejected and preserved; all forwarded feedback and candidate
prompts pass the final scanner; no test identifier appears in an evaluator
dossier; and prompt lock preceded final-test creation.

## Historical result (pre-v7; not for the current paper table)

| detector | cov | P | R | F1 | typo | inconsist. | semantic | numeric |
|---|---|---|---|---|---|---|---|---|
| **GPT-5.5 single-prompt** | 214 | 0.403 | 0.773 | **0.530** | 0.843 | 0.693 | 0.808 | 1.000 |
| raha (reference) | 186 | 0.785 | 0.690 | 0.735 | 0.693 | 0.672 | 1.000 | 0.500 |

These values predate `merged_v7` and the full-universe refresh. Do not copy them
into the current paper table; use fresh v7 score artifacts.

## Notes / caveats

- **Coverage 214/215**: one table (`a8df…f6da8663`, 100×14) repeatedly hit the
  180s request timeout across endpoints and was **excluded from scope** (not
  counted as misses), per the "remove the CSV we can't run" decision. All other
  214 returned. Re-running will retry it (cache resumes the rest instantly).
- **LLM**: Azure `gpt-5.5` via the repo's `LLMClient`, `reasoning_effort=medium`.
- Scope = whole CSV; positives = labeled `error` cells; `ambiguous`/`skipped`/
  `excluded` neutralized (see `external/rein/README.md`).
- This detector is cell-only. All `benchmark_row.json` relation findings are out
  of coverage and reported as N/A, not as false negatives.
