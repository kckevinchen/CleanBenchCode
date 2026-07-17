# `external/baseline/inconsistency/` — rulebook prompt-optimization loop

A self-contained multi-agent loop that optimizes the GPT-5.5 **inconsistency
rulebook** prompt against the ground-truth benchmark, with a hard information
barrier so the prompt never overfits to the eval data. Vendors the few helpers
it needs (decoupled from `RFT/`), like the rest of `external/baseline/`.

## The idea

```
  eval(seed rulebook) on DEV  ──►  precision / recall / F1  (vs gold)
        │
        ▼
  REVIEWER (sees dev data) ──►  generalizable suggestions (JSON)
        │
   LEAKAGE FILTER  ──►  strips any verbatim dev value           ← mechanical barrier
        │
        ▼
  PROMPT EDITOR (never sees data) ──►  rewrites the rulebook rules
        │
  candidate = edited rules + FROZEN Output section               ← schema preserved by construction
        │
        ▼
  eval(candidate) on DEV  ──►  accept iff objective improves > margin (hill-climb)

  …repeat…  then eval the best prompt on the held-out TEST split ONCE, per dataset.
```

Three properties make it sound rather than just runnable. They address two
failure modes observed during development: prompt-vs-benchmark overfitting and
*verbatim* eval-cell leakage.

1. **The barrier is data-flow, not trust.** The editor's input is *constructed*
   to contain only the rulebook + filtered suggestions. On top of that,
   `leakage.py` builds a blocklist from the concrete dev values the reviewer saw
   and **redacts/drops** any suggestion that quotes one — so a leaked value can
   never ride a suggestion into the prompt.
2. **The output schema is frozen by surgery, not instruction.** `_task.split_rulebook`
   peels everything from `## 4. Output` onward (the JSON contract the parser
   depends on) off the editable rules and re-attaches it verbatim every
   iteration. The editor literally cannot change the response format.
3. **Train/test discipline.** Split **by source** (no CSV straddles dev/test).
   The reviewer only ever sees **dev**; **test** is scored once at the end and
   reported per dataset for a pbi-vs-bing generalization read.

## Data (auto-resolved)

- Gold: `benchmark/merged_v2/benchmark_inconsistency_cells.jsonl` (per-cell,
  `label` ∈ {error, ambiguous}, `dataset` ∈ {pbi_500, bing21_500}).
- Source CSVs resolved from `data/curated_pbi_500_dedup` + `data/curated_bing21_500_dedup`.
- Scoring (vendored from `RFT.inconsistency.v1.eval.run_baseline`): per distinct
  `(column, value)`; gold-error = positive; **ambiguous = FP-immune**; clean
  columns of the split's sources are audited so over-flagging counts as FP.

## Run

```bash
az login   # required before any live call

# full loop, pooled pbi+bing, optimize F1, 6 iterations
python -m external.baseline.inconsistency.optimize --iters 6

# optimize recall-leaning (F-beta, beta=2) since we have ground truth
python -m external.baseline.inconsistency.optimize --objective fbeta --fbeta 2.0

# out-of-distribution test: optimize on bing, hold ALL pbi out for test
python -m external.baseline.inconsistency.optimize --dev-dataset bing21_500

# smoke-test the whole loop with NO API calls (stub model + stub agents)
python -m external.baseline.inconsistency.optimize --dry-run --iters 2
```

Key flags: `--objective {f1,f1_labeled,precision,precision_labeled,recall,fbeta}`
(default `f1`), `--fbeta`, `--margin` (noise guard — require the dev objective to
improve by more than this), `--dev-frac`/`--seed` (the split), `--dev-dataset`
(OOD), `--model`/`--effort`/`--temperature` (detector; default GPT-5.5 /
`none` / `0` — the proven deterministic setting), `--agent-model`/`--agent-effort`
(reviewer + editor), `--endpoints`, `--concurrency`, `--leakage-min-len`.

## Outputs

`outputs/inconsistency_promptopt/<ts>/`:

- `split.json` — the dev/test source lists + per-dataset stats.
- `iter_NN/` — `critique_raw.json`, `critique_filtered.json`,
  `leakage_report.json` (what was redacted/dropped), `prompt.md` (the candidate),
  `dev_metrics.json`, `decision.json` (accept/reject + objective delta).
- `best_prompt.md`, `test_metrics.json`, `loop_summary.json`.

## Prompts (source of truth)

- `external/baseline/prompts/inconsistency/rulebook.md` — the seed / optimization target.
- `external/baseline/prompts/inconsistency/reviewer.md` — reviewer system prompt.
- `external/baseline/prompts/inconsistency/prompt_editor.md` — editor system prompt.

## Layout

| file | role |
|---|---|
| `_task.py` | user message, response schema, `parse_findings`, **rulebook surgery** |
| `data.py` | gold loading (merged_v2), CSV lookup, by-source stratified split |
| `leakage.py` | blocklist + suggestion scrubber (the mechanical barrier) |
| `evaluator.py` | pure scoring (`score_column`/`aggregate`) + injected detector + live `Gpt55Detector` |
| `agents.py` | `review()` / `edit_prompt()` over an injectable `AgentBackend` |
| `loop.py` | the orchestrator (hill-climb + artifacts + final test) |
| `optimize.py` | CLI entry point (+ `--dry-run` stubs) |

The deterministic core (split, scoring, leakage, schema freeze) is covered by
`tests/test_inconsistency_promptopt.py` with no network.
