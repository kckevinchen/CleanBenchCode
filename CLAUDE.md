# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A modular research harness for data-quality experiments against Azure-hosted GPT-5-family LLMs. The flagship experiment scans a CSV **one narrow error type at a time** and merges findings. See `README.md` for the user-facing quickstart and dashboard tour; this file covers the conventions and architecture you need to be productive in the code.

## Commands

```bash
# Environment (Azure OpenAI auth is via AzureCliCredential, NOT API keys)
./create_venv.sh && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt   # -dev for ruff/pytest/pre-commit
az login                                                  # required before any live LLM call

# Tests (config in pyproject.toml; testpaths=tests)
pytest                                  # full suite
pytest tests/test_aggregator.py         # single file
pytest tests/test_relations.py::test_x  # single test
pytest -m "not integration"             # skip live-service tests (integration + slow markers exist)

# Lint / format (line-length 100, double quotes)
ruff check .            # add --fix to autofix
ruff format .
pre-commit run --all-files

# Smoke test the pipeline without spending LLM calls
python -m experiments.run_cleaning_detection --dry-run
python -m experiments.run_cleaning_detection --list-error-types

# Web workbench (launches runs, browses outputs, labels clusters); survives SSH disconnect
./scripts/run_server.sh start --port 8000   # also: stop | restart | status
# -> http://127.0.0.1:8000/  (logs/workbench.log, logs/workbench.pid)
```

## Architecture

The codebase is three components forming one pipeline:

1. **Labeling method** — the automated LLM detection pipeline that *generates* candidate data-quality error labels (the two detection pipelines below + adaptive Good-Turing mode + the web-grounded verifier). `src/core/`, `src/adaptive/`, `src/verifier/`; run via `experiments/run_*.py`.
2. **Labeling UI** — the FastAPI workbench where humans review/curate those candidates into a gold **benchmark**. `src/labeling/`, `frontend/`; run via `experiments/workbench.py`. Exports per-cell benchmark JSONL.
3. **RFT** — reinforcement fine-tuning of a model on that benchmark to improve data-cleaning performance. `RFT/` + `RFT.md`.

The throughline: **method generates candidates → UI curates them into a benchmark → RFT trains on the benchmark.** The sections below detail each.

Three things are separated **on purpose** so the taxonomy can be iterated without touching Python: `src/` (code), `prompts/*.yaml` (taxonomy + LLM prompt templates), `config/experiment.yaml` (settings). Every YAML key under `config/experiment.yaml` has a matching CLI flag on the entry points.

**Layering.** `experiments/*.py` are thin CLI entry points (argparse + orchestration, zero core logic) invoked as `python -m experiments.<name>`. All real logic lives in `src/`. The web UI launches these same scripts as child processes.

### Two detection pipelines, split by error-type scope

The 9-type taxonomy (`prompts/error_types.yaml`) is grouped by Scope, and scope decides which pipeline runs and whether `column_mode` must be `all`:

1. **Single-column types** (`typo`, `inconsistency`, `semantic_incompatible`, `numeric_outlier`) → the **per-chunk detector** (`src/core/detector.py::PerTypeCleaningDetector`). One LLM call per (chunk × error_type). Works in `column_mode=single`.
   - **Phase A** (`src/core/aggregator.py`): deterministic union-find clustering — detections sharing a row with overlapping columns merge into one `Cluster`.
   - **Phase B** (optional, default on): one extra synthesis LLM call per cluster picks a `primary_error_type` and emits a canonical explanation/fix/confidence/FP-flag.

2. **Multi-column / multi-row / multi-table types** (`fd_violation`, `row_calculation_inconsistency`, `string_calculation_inconsistency`, `row_duplication`, `inter_table_inconsistency`) → the **inter-column relation pipeline** (`config: inter_column_pipeline`). Requires `column_mode=all`. The whole pipeline lives in the `src/core/relations/` package. Four stages: **DISCOVER** candidate relations (`relations/discovery.py`, `llm_detector.py`) → **VERIFY** by having the LLM write a `check(df)` predicate run in a sandbox (`relations/validator.py`) → **FILTER** on empirical pass-rate (recall-first: drop only certainly-useless rules; code-refused/errored rules bypass to detect) → **DETECT** violating rows (`relations/violations.py`). `from src.core.relations import Relation` still works (the package `__init__` re-exports the core symbols from `discovery`). `row_duplication` additionally has its own collision-adjudication detector in the `src/core/row_dup/` package (`detector.py` + `clusterers.py` + `priors.py`) using key collision + MinHash + sentence-embedding blocking. Cross-table (`inter_table_inconsistency`) logic lives in the `src/core/multi_table/` package.

   The relation pipeline can be run as discrete steps for review-in-the-loop: `run_relation_discovery` → review in UI → `run_relation_validation` → `run_relation_detection`, with `build_relation_report` rendering its dashboard.

### Adaptive (Good-Turing) mode

`experiment.mode=adaptive` (default) runs up to K perturbed passes, recomputing a Good-Turing / Chao1 missing-mass estimate after each and stopping when undiscovered-defect mass drops below threshold. Perturbation = LLM prompt-ensemble rewrites (`src/adaptive/prompt_variants.py`) and/or row/column shuffles (`perturbations.py`). Each `(scope, error_type)` arm saturates independently (`src/adaptive/scoped_gt.py`, `cell_gt.py`, `_chao1.py`). Final clusters/report are built from the **union** of detections across all passes. `mode=single` does one un-perturbed pass.

### LLM client

`src/core/llm_client.py::LLMClient` is framework-free and shared by everything. It round-robins across a pool of Azure endpoints (`DEFAULT_ENDPOINTS`, override via `AZURE_OPENAI_ENDPOINTS`), tracks dead `(endpoint, model)` 404 pairs and skips them, and falls back between the Responses API (`RESPONSES_API_MODELS`) and Chat Completions. `GPT5_FAMILY` is the default model rotation when a run requests a family sweep. Reasoning effort is coerced through `RESPONSES_API_VALID_EFFORTS` (`minimal`→`low`).

### Web-grounded verifier (optional post-pass)

`src/verifier/` runs flagged entity clusters through a Microsoft Foundry Agent with the Web Search tool to check "what should this value say" against the web (`run_web_verify`). Has a built-in round-robin agent pool; override via `AZURE_AGENTS_*` env vars. `StubAgentBackend` is used in tests.

### Labeling workbench

`src/labeling/` is a FastAPI app (`app.py` factory, `routers/`, `runs.py` job manager, `store.py`/`*_store.py` JSONL-backed label stores). Entry point `experiments/workbench.py`. Frontend is static HTML/JS in `frontend/` (`report_template.html` rendered by `src/core/report_builder.py` via the `/*__REPORT_DATA_JSON__*/` placeholder; `frontend/labeling/` is the annotation UI; `relations_template.html` for relation runs). Labels persist per-run and export to per-cell benchmark JSONL.

### Output convention

Each run writes `outputs/<timestamp>_<model>/` (gitignored) with `report.html` (self-contained dashboard), `report.md`, `summary.json`, `clusters.json(l)`, `issues.json(l)`, `chunk_reports.json(l)`, `raw_responses/`, and `good_turing/` (adaptive mode). Relation runs and `run_relation_detection` reuse this exact artifact shape so the labeling UI picks them up unchanged.

## Other directories

- `src/data_prep/` — offline CSV curation library (`curate.py`, `subsample.py`, bing21 reconstruction). Not part of the runtime detection path.
- `src/heuristics/` & `src/synth/` — deterministic heuristic detectors and synthetic-error injectors (for benchmark construction / baselines).
- `RFT/` + `RFT.md` — the typo-detection reinforcement-fine-tuning effort. `RFT.md` is the single source of truth for its eval protocol (cell-level micro-F1 against `benchmark/...jsonl`; eval split must never be seen in training). Read it before touching anything RFT-related.
- `data/`, `outputs/`, `benchmark/`, `logs/`, `docs/`, `notebooks/` are all gitignored.
