# fm_data_tasks (Narayan et al., VLDB'23) on `benchmark/merged_v8`

["Can Foundation Models Wrangle Your Data?"](https://arxiv.org/abs/2205.09911)
(HazyResearch/[fm_data_tasks](https://github.com/HazyResearch/fm_data_tasks),
VLDB'23) is the seminal study of **foundation models for data wrangling**. Its
error-detection task frames cell-level ED as a **per-row, per-attribute binary
judgment** over a characteristic serialization: a tuple is rendered as
`col: val. col: val. ...` and the model is asked *"Is there an error in
`col: val`? Yes or No"*, optionally preceded by `k` in-context examples.

## We run the authors' upstream code (not a reimplementation)

This integration drives the **published `fm_data_tasks` source** unmodified. The
benchmark runner imports the authors' own serialization / sampling utilities and
uses them directly:

- **Serialization** — `fm_data_tasks.utils.data_utils.serialize_error_detection`
  renders each `(row, attribute)` query exactly as upstream does.
- **Few-shot sampling** — `data_utils.sample_train_data` selects the in-context
  examples, upstream code.

Provenance (per `files/integration_contract.md`):

- **`fm_data_tasks_repo/`** — an unmodified clone of the upstream repo, pinned at
  commit **`c954cfd`** (gitignored). Zero edits.
- **`adapter.py`** — lives *outside* the clone. Upstream drives inference through
  the `manifest` package (`manifest.run(prompt)`); the original models
  (`text-davinci-002/003`) are decommissioned, so `ManifestLLMClient` provides a
  `Manifest`-shaped `.run()` that routes each call through our Azure
  `src.core.llm_client.LLMClient` (11-endpoint round-robin pool). The upstream
  call shape is preserved; no recipe logic is reimplemented.
- **`run.py`** — the benchmark runner (outside the clone): loops every
  `(row, attribute)` of each table, builds the query with upstream
  `serialize_error_detection`, and records the cells the model answers "Yes".

**This is the faithful per-cell recipe** (one call per `(row, attribute)`), which
is what fm_data_tasks's ED task actually does — not a per-row batching
approximation. Corpus-wide that is a few hundred thousand calls, which is why a
cheap model + high concurrency are used (see below).

## Faithful model

Upstream targets **GPT-3 `text-davinci-002/003`**, which are no longer served.
We substitute **`gpt-4o-mini`** (`--model`, default) — the recipe is about
in-context learning, not raw model power, so a small model keeps the per-cell call
volume affordable. This model substitution is the single documented deviation.

## Zero-shot vs. few-shot (honest)

- **Zero-shot** (default, `--shots 0`) — instruction only -> zero-shot leaderboard.
- **Few-shot** (`--shots k`) — `k` in-context examples are built from a fixed
  **synthetic** injection (`external/synth_fewshot/inject.py`, `SEED=20240601`),
  so **no real benchmark label is ever revealed**. The synthetic cells are written
  to a `.synthetic_cells.json` sidecar and neutralized by the scorer, and the
  labeled rows are held out. This is the honest counterpart to fm_data_tasks's
  in-context-learning mode.

## Run

First clone the pinned upstream source with `bash external/setup_upstreams.sh`
and provision a package-local environment containing `manifest-ml`, pandas, and
the repository's Azure client dependencies.

```bash
az login   # Azure LLMClient auth
# datasets are the materialized REIN ones (see external/rein/prepare_datasets.py)
external/fm_data_tasks/venv/bin/python external/fm_data_tasks/run.py --limit 10 --no-cache   # smoke
external/fm_data_tasks/venv/bin/python external/fm_data_tasks/run.py --model gpt-4o-mini --concurrency 24            # zero-shot, full corpus
external/fm_data_tasks/venv/bin/python external/fm_data_tasks/run.py --model gpt-4o-mini --shots 3 --tag fm_fewshot  # few-shot (synthetic)

source venv/bin/activate
python external/rein/score.py --predictions external/rein/results/predictions_fm_data_tasks.jsonl
```

`--model` (default `gpt-4o-mini`), `--reasoning-effort` (default `low`),
`--concurrency` (default `12`; bump to 24+ for the per-cell full-corpus run since
the model is cheap). Per-table responses cache under `cache/` keyed by dataset +
model + upstream commit + `v8` tag (resumable). Predictions land in
`external/rein/results/` (gitignored).

## Layout / notes

```
external/fm_data_tasks/
  adapter.py            # ManifestLLMClient: manifest.run(prompt) -> LLMClient (outside clone)
  run.py                # per-(row,attribute) runner using upstream serialization
  README.md
  fm_data_tasks_repo/   # upstream clone @c954cfd, UNMODIFIED (gitignored)
  cache/                # per-table LLM responses, version+commit tagged (gitignored)
  venv/                 # isolated environment (gitignored)
```

- **Upstream code, substituted model.** We run the authors' `fm_data_tasks`
  serialization + binary per-attribute judgment unmodified; the only deviation is
  `text-davinci-002/003` -> `gpt-4o-mini` (see above).
- **Expected behavior.** Per-cell judgment without full column context tends to
  **over-flag** (high recall, low precision) on free-form fields — consistent with
  the TabClean paper's note that per-cell LLM prompting "over-edits free-form
  bibliographic and metadata fields." In-context examples (`--shots`) calibrate it.
- **Placement.** Zero-shot runs go on the zero-shot leaderboard; `--shots k` runs
  are few-shot (synthetic, honest) and reported in the few-shot section.
