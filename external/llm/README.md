# Project-native LLM baselines

These runners provide independent LLM comparison methods without using candidate
clusters from the main detector.

## Blind inconsistency baseline

`blind_inconsistency.py` shows one raw column vocabulary at a time and asks the
model for normalization groups. The benchmark is used only after prediction for
scoring.

```bash
python -m external.llm.blind_inconsistency --help
```

Prompt: `prompts/inconsistency.yaml`. The optional judge prompt is
`prompts/inconsistency_judge.yaml`.

## Conservative precision baseline

`precision.py` makes one call per `(CSV, column, error_type)` with conservative,
precision-first prompts for typo, inconsistency, numeric outlier, and semantic
incompatibility.

```bash
python -m external.llm.precision --help
python -m external.llm.score_precision --help
```

Predictions and scoring outputs are written below ignored `outputs/` and require
local source data and benchmark JSONL files.

Both methods use `src.core.llm_client.LLMClient`; configure
`AZURE_OPENAI_ENDPOINTS` and authenticate with `az login` before live calls.
