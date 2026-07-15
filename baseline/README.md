# `baseline/` — GPT-5.5 typo-detection baseline

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
| `with-rulebook` (default) | [`prompts/with_rulebook.md`](prompts/with_rulebook.md) | intro + the clean, spec-derived rulebook + output footer. Precision-first; proper-noun character-level slips out of scope. **7,574 chars, sha256 `2fd244b3…`** |
| `without-rulebook` | [`prompts/without_rulebook.md`](prompts/without_rulebook.md) | intro + a compact *generalized* rule + output footer. No long rulebook — keeps a per-token NAME-vs-generic-word scope rule (names strictly out of scope for char-level fixes). **3,827 chars, sha256 `054cafd8…`** |

[`prompt.py`](prompt.py) just loads these files and exposes them as
`WITH_RULEBOOK_PROMPT` / `WITHOUT_RULEBOOK_PROMPT` (+ `PROMPTS`, `PROMPT_SHA`).
To edit a prompt, edit its `.md` file; the recorded sha256 makes any drift
visible. Freeze check: `python -m baseline.prompt`.

## Run

```bash
az login   # required before any live call

# with-rulebook (default)
python -m baseline.label_gpt55 --data-dir data/curated_bing21_500_dedup

# without-rulebook
python -m baseline.label_gpt55 --data-dir data/curated_bing21_500_dedup \
    --prompt without-rulebook

# smoke test (2 CSVs)
python -m baseline.label_gpt55 --data-dir data/curated_bing21_500_dedup --limit 2
```

Key flags: `--prompt {with-rulebook,without-rulebook}`, `--model` (default
`gpt-5.5`), `--endpoint`, `--effort` (default `none`), `--limit`,
`--concurrency`, `--tag`. Output run id is
`outputs/<ts>_<model>_<datadir>_<tag>/`, where `tag` defaults to
`<prompt>_<effort>` (e.g. `with_rulebook_none`). Per-column responses are cached
under the run's `gpt55_cache/`, keyed by prompt sha + model + effort + column.

## Score

Scoring is **shared**, not part of `baseline/` — the same scorer grades RFT
model runs. Point it at the run dir produced above:

```bash
TYPO_BENCH_DIR=benchmark/typo_excel_500 \
    python -m experiments.score_typo_dir_benchmark <run_dir> [<run_dir> ...]
```
