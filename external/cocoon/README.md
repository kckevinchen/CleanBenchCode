# Cocoon (ICDEW'25) on `benchmark/merged_v8` (cell-level)

Cocoon — "Data cleaning using large language models" (Zhang, Huang, Wu, *ICDEW
2025*; project: <https://github.com/Cocoon-Data-Transformation/cocoon>) — is a
**zero-shot**, column-oriented LLM cleaner. For each column it separates the
dominant/"normal" values from the "unusual" ones and synthesizes *executable
cleaning logic* (regex normalizations and `CASE WHEN` value mappings) validated
against the column's values. It is one of the four baselines the TabClean paper
compares against (the LLM-based, zero-human-input one).

## We run the authors' upstream package (not a reimplementation)

This integration drives the **published `cocoon_data` source** unmodified. The
upstream package normally exposes cleaning through an interactive Streamlit /
Jupyter-widget flow, so there is no batch CLI. We reach the underlying method
programmatically by calling its headless graph node
**`cocoon_data.CleanUnusual.run`** directly — the node that proposes per-column
value-cleaning mappings — over each benchmark table.

Provenance (per `files/integration_contract.md`):

- **`cocoon_repo/`** — an unmodified clone of the upstream repo, pinned at commit
  **`410a4f7`** (gitignored). Zero edits, zero injected files.
- **`adapter.py`** — lives *outside* the clone. Its only job is to monkeypatch
  upstream's single LLM seam `cocoon_data.llm.call_llm_chat` (legacy `openai` 0.x
  `engine=` calls) so requests route through our Azure `src.core.llm_client.
  LLMClient` (the 11-endpoint round-robin pool). No method logic is touched.
- **`run_cocoon.py`** — the benchmark runner (also outside the clone): materializes
  each table's `extract` input the way upstream's `CleanUnusual.extract` does,
  calls `CleanUnusual().run(...)`, and records which cells the proposed mappings
  rewrite.

**Detection = cells a proposed mapping changes** (`df[col] == from` and
`to != from`). No labels or ground truth are used, so Cocoon is **zero-shot** and
scored over the full 1,007-CSV universe (no held-out sidecar).

## Faithful model

Upstream defaults to **`gpt-4-turbo`**. That deployment returns empty strings on
our pool (effectively unusable), so we substitute the closest served frontier
chat model, **`gpt-4o`** (`--model`, default `gpt-4o`). This is the single
documented deviation; everything else is upstream code.

## Known limitation (honest reporting)

`CleanUnusual` parses the LLM's YAML reply with upstream `yaml.load(...,
SafeLoader)`. With the `gpt-4o` substitution, roughly **~22% of tables** produce
YAML that upstream cannot parse (e.g. an unterminated quoted scalar), raising a
`ScannerError` *inside upstream code*. Those tables are recorded as **zero
detections** (an honest recall floor, no false positives injected) rather than
being retried or dropped, and the parse-failure rate is footnoted in the writeup.
We do **not** patch upstream to work around it.

## Layout

```
external/cocoon/
  adapter.py        # patches cocoon_data.llm.call_llm_chat -> LLMClient (outside clone)
  run_cocoon.py     # concurrent runner: cocoon_data.CleanUnusual.run -> predictions_cocoon.jsonl
  README.md
  cocoon_repo/      # upstream clone @410a4f7, UNMODIFIED (gitignored)
  cache/v8/         # per-table record cache, version+commit tagged (gitignored)
  venv/             # isolated environment (gitignored)
```

## Run

First clone the pinned upstream source with `bash external/setup_upstreams.sh`
and provision the package-local environment described by the upstream project.

```bash
az login
# datasets are the materialized REIN ones (see external/rein/prepare_datasets.py)
external/cocoon/venv/bin/python external/cocoon/run_cocoon.py --limit 5            # smoke test
external/cocoon/venv/bin/python external/cocoon/run_cocoon.py --model gpt-4o --concurrency 12

# zero-shot: scored over the full 1,007-CSV universe, no held-out
source venv/bin/activate
python external/rein/score.py --predictions external/rein/results/predictions_cocoon.jsonl
```

`--model` (default `gpt-4o`), `--reasoning-effort` (default `low`), and
`--concurrency` (default `12`, >= the 11-endpoint pool) tune the run. Per-table
records are cached under `cache/v8/` keyed by dataset + model + upstream commit
(resumable; a benchmark/version tag prevents stale reuse).

## Notes / fidelity

- **Zero-shot, full universe.** Cocoon belongs on the *zero-shot* leaderboard
  with the GPT-5.5 single-prompt detector, SymSpell/NeuSpell, and CleanAgent.
- **Upstream code, substituted model.** We run the authors' `cocoon_data`
  package unmodified; the only deviation is `gpt-4-turbo` -> `gpt-4o` (see above).
  The paper notes Cocoon's regex / `CASE WHEN` rules "cover only a small fraction
  of heterogeneous errors", so modest recall is expected.
