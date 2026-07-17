# `external/baseline/` — GPT-5.5 typo-detection baseline

A self-contained external baseline for the typo benchmark. It labels a CSV
dataset with **GPT-5.5** (one Responses-API call per textual column,
`reasoning_effort=none` by default) and writes a standard, workbench-loadable
detection-run directory under `outputs/`. It is decoupled from the `RFT/` tree
(it vendors the few helpers it needs), so RFT can change without affecting it.

## The two prompts

Exactly two, each a complete frozen string in [`prompts/`](prompts/) (the source
of truth; read verbatim and sent as the Responses-API `instructions`):

| `--prompt` | file | what it is |
|---|---|---|
| `with-rulebook` (default) | [`prompts/typo/with_rulebook.md`](prompts/typo/with_rulebook.md) | intro + the clean, spec-derived rulebook + output footer. Precision-first; proper-noun character-level slips out of scope. **7,574 chars, sha256 `2fd244b3…`** |
| `without-rulebook` | [`prompts/typo/without_rulebook.md`](prompts/typo/without_rulebook.md) | intro + a compact *generalized* rule + output footer. No long rulebook — keeps a per-token NAME-vs-generic-word scope rule (names strictly out of scope for char-level fixes). **3,827 chars, sha256 `054cafd8…`** |

[`prompt.py`](prompt.py) just loads these files and exposes them as
`WITH_RULEBOOK_PROMPT` / `WITHOUT_RULEBOOK_PROMPT` (+ `PROMPTS`, `PROMPT_SHA`).
To edit a prompt, edit its `.md` file; the recorded sha256 makes any drift
visible. Freeze check: `python -m external.baseline.prompt`.

## Run

The repository does not ship benchmark or source data. Prepare a local
directory of CSV files under `data/` (or another ignored location) and pass it
with `--data-dir`. The command exits if the directory is missing or contains no
CSV files.

```bash
az login   # required when AZURE_OPENAI_KEY is not set
export AZURE_OPENAI_ENDPOINTS="https://<your-resource>.openai.azure.com/"

# with-rulebook (default)
python -m external.baseline.label_gpt55 --data-dir <dataset-dir>

# without-rulebook
python -m external.baseline.label_gpt55 --data-dir <dataset-dir> \
    --prompt without-rulebook

# smoke test (one CSV; still one LLM call per textual column)
python -m external.baseline.label_gpt55 --data-dir <dataset-dir> --limit 1
```

Authentication normally uses `AzureCliCredential`. If `AZURE_OPENAI_KEY` is
explicitly set, this baseline uses that key instead; unset it to force Entra
authentication.

Key flags: `--prompt {with-rulebook,without-rulebook}`, `--model` (default
`gpt-5.5`), `--endpoint`, `--effort` (default `none`), `--limit`,
`--concurrency`, `--tag`. Output run id is
`outputs/<ts>_<model>_<datadir>_<tag>/`, where `tag` defaults to
`<prompt>_<effort>` (e.g. `with_rulebook_none`). Per-column responses are cached
under the run's `gpt55_cache/`, keyed by prompt sha + model + effort + column.

## Score

Scoring is **shared**, not part of `external/baseline/` — the same scorer grades RFT
model runs. The current scorer uses the historical local typo-export layout and
does not consume the workbench eval bundle directly. Prepare an ignored
benchmark directory containing:

```text
<legacy-benchmark-dir>/
    typo_benchmark.jsonl
    typo_skipped.jsonl          # optional skip immunity
    typo_exclusions.jsonl       # optional CSV/column exclusions
```

Then point it at the run directory produced above:

```bash
TYPO_BENCH_DIR=<legacy-benchmark-dir> \
    python -m experiments.score_typo_dir_benchmark <run_dir> [<run_dir> ...]
```

Do not assume `benchmark/typo_excel_500` exists in a clean checkout; benchmark
data is intentionally untracked.
