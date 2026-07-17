# Copilot instructions for data-quality-llm

A modular research harness for data-quality experiments against Azure-hosted
GPT-5-family LLMs. The flagship experiment scans a CSV **one narrow error type
at a time** and merges the findings. `README.md` is the user-facing quickstart
and dashboard tour; `CLAUDE.md` mirrors much of the guidance below. This file
covers the conventions and architecture you need to be productive in the code.

## Commands

```bash
# Environment — Azure OpenAI auth is via AzureCliCredential, NOT API keys
./create_venv.sh && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt   # -dev adds ruff/pytest/pre-commit
az login                                                   # required before any live LLM call
cp .env.example .env                                       # add your Azure resources
set -a; source .env; set +a                                # loads AZURE_OPENAI_ENDPOINTS / AZURE_AGENTS_*

# Tests (config in pyproject.toml; testpaths=tests)
pytest                                   # full suite
pytest tests/test_aggregator.py          # single file
pytest tests/test_relations.py::test_x   # single test
pytest -m "not integration"              # skip live-service tests (markers: integration, slow)

# Lint / format (line-length 100, double quotes)
ruff check .            # add --fix to autofix
ruff format .
pre-commit run --all-files

# Smoke-test the pipeline without spending LLM calls
python -m experiments.run_cleaning_detection --dry-run
python -m experiments.run_cleaning_detection --list-error-types

# Web workbench (launch runs, browse outputs, label clusters); survives SSH disconnect
./scripts/run_server.sh start --port 8000   # also: stop | restart | status
# -> http://127.0.0.1:8000/  (logs/workbench.log, logs/workbench.pid)

# RFT — one generic engine driven by per-task specs (read RFT.md first)
python -m RFT.cli --list-tasks                          # tasks grouped by error type
python -m RFT.cli --task typo_v10 all --reward-variant r1 --endpoint <ep>  # prepare→upload→submit→follow
python -m RFT.cli --task typo_v10 status --follow       # monitor; deploy / run (eval) are subcommands too
```

## Big picture

Three components form one pipeline — **method generates candidates → UI curates
them into a benchmark → RFT trains on the benchmark**:

1. **Labeling method** — the automated LLM detection pipeline that *generates*
   candidate error labels (the two detection pipelines below + adaptive
   Good-Turing mode + the web-grounded verifier). `src/core/`, `src/adaptive/`,
   `src/verifier/`; run via `experiments/run_*.py`.
2. **Labeling UI** — the FastAPI workbench where humans curate those candidates
   into a gold benchmark. `src/labeling/`, `frontend/`; run via
   `experiments/workbench.py`. Exports per-cell benchmark JSONL.
3. **RFT** — reinforcement fine-tuning of a model on that benchmark. `RFT/` +
   `RFT.md`. **`RFT.md` is the single source of truth for the eval protocol
   (cell-level micro-F1; the eval split must never be seen in training) — read
   it before touching anything RFT-related.**

### Two detection pipelines, split by error-type scope

The 9-type taxonomy (`prompts/error_types.yaml`) is grouped by Scope, and scope
decides which pipeline runs and whether `column_mode` must be `all`:

1. **Single-column types** (`typo`, `inconsistency`, `semantic_incompatible`,
   `numeric_outlier`) → the **per-chunk detector**
   (`src/core/detector.py::PerTypeCleaningDetector`). One LLM call per
   (chunk × error_type); works in `column_mode=single`.
   - **Phase A** (`src/core/aggregator.py`): deterministic union-find clustering
     — detections sharing a row with overlapping columns merge into one `Cluster`.
   - **Phase B** (optional, default on): one synthesis LLM call per cluster picks
     a `primary_error_type` and emits a canonical explanation/fix/confidence/FP-flag.

2. **Multi-column / multi-row / multi-table types** (`fd_violation`,
   `row_calculation_inconsistency`, `string_calculation_inconsistency`,
   `row_duplication`, `inter_table_inconsistency`) → the **inter-column relation
   pipeline** (`config: inter_column_pipeline`, package `src/core/relations/`).
   Requires `column_mode=all`. Four stages: **DISCOVER** candidate relations →
   **VERIFY** by having the LLM write a `check(df)` predicate run in a sandbox →
   **FILTER** on empirical pass-rate (recall-first: drop only certainly-useless
   rules; code-refused/errored rules bypass to detect) → **DETECT** violating
   rows. Run as discrete review-in-the-loop steps: `run_relation_discovery` →
   review in UI → `run_relation_validation` → `run_relation_detection`, with
   `build_relation_report` rendering the dashboard. `row_duplication` has its own
   collision-adjudication detector in `src/core/row_dup/`; cross-table logic
   lives in `src/core/multi_table/`.

### Adaptive (Good-Turing) mode

`experiment.mode=adaptive` (default) runs up to K perturbed passes, recomputing a
Good-Turing / Chao1 missing-mass estimate after each and stopping when
undiscovered-defect mass drops below threshold. Perturbation = LLM prompt-ensemble
rewrites (`src/adaptive/prompt_variants.py`) and/or row/column shuffles
(`perturbations.py`). Each `(scope, error_type)` arm saturates independently
(`src/adaptive/scoped_gt.py`, `cell_gt.py`, `_chao1.py`). Final clusters/report
are built from the **union** of detections across all passes. `mode=single` does
one un-perturbed pass.

### LLM client

`src/core/llm_client.py::LLMClient` is framework-free and shared by everything. It
round-robins across the Azure endpoints configured via `AZURE_OPENAI_ENDPOINTS`,
tracks dead `(endpoint, model)` 404 pairs and skips
them, and falls back between the Responses API (`RESPONSES_API_MODELS`) and Chat
Completions. The web-grounded verifier (`src/verifier/`) instead hits a Microsoft
Foundry Agent with the Web Search tool (configured via `AZURE_AGENTS_*`;
`StubAgentBackend` in tests).

## Key conventions

- **Three-way separation, on purpose:** `src/` (code), `prompts/*.yaml` (taxonomy
  + LLM prompt templates), `config/experiment.yaml` (settings). Iterate on the
  taxonomy or thresholds without touching Python. To add an error type, add a key
  under `types:` in `prompts/error_types.yaml` (`name`, `category`, `description`,
  `example`, `notes` with both *WHAT TO FLAG* and *WHAT NOT TO FLAG*) — it is
  picked up automatically. The template shared by every type lives in
  `base.system` / `base.user_template` in the same file; tunable numeric
  thresholds are substituted into the templates at load time (under `defaults:`
  in the `error_types_focused.yaml` variant).
- **Thin entry points:** `experiments/*.py` are argparse + orchestration only
  (zero core logic), invoked as `python -m experiments.<name>`. All real logic
  lives in `src/`. The web UI launches these same scripts as child processes.
- **YAML ↔ CLI parity:** every key under `config/experiment.yaml` has a matching
  CLI flag on the entry points (the YAML documents the override flag inline).
- **Output convention:** each run writes `outputs/<timestamp>_<model>/`
  (gitignored) with `report.html` (self-contained dashboard), `report.md`,
  `summary.json`, `clusters.json(l)`, `issues.json(l)`, `chunk_reports.json(l)`,
  `raw_responses/`, and `good_turing/` (adaptive mode). Relation runs reuse this
  exact shape so the labeling UI picks them up unchanged.
- **Dashboard:** edit `frontend/report_template.html`; the
  `/*__REPORT_DATA_JSON__*/` placeholder is filled by `src/core/report_builder.py`.
- **Ruff:** line-length 100, double quotes; many `PL*` checks and `E501` are
  intentionally ignored for research code (see `pyproject.toml`). `prompts/` is
  excluded from `check-yaml` because templates use Jinja-style `{{var}}`.
- **Gitignored (do not expect in VCS):** `data/`, `outputs/`, `benchmark/`,
  `logs/`, `docs/`, `notebooks/`, `venv/`.

## Other directories

- `src/data_prep/` — offline CSV curation library; not part of the runtime path.
- `src/heuristics/` & `src/synth/` — deterministic heuristic detectors and
  synthetic-error injectors (benchmark construction / baselines).
- `baseline/` — self-contained external GPT-5.5 typo-detection baseline
  (`python -m baseline.label_gpt55`); vendors its own helpers so it is decoupled
  from `RFT/`. Its two frozen prompts in `baseline/prompts/` are the source of truth.
- `external/` — ad-hoc normalization/baseline experiments and scratch data;
  excluded from ruff (and most tooling). Not on the runtime path.
- `scripts/` — one-off operational helpers (`run_server.sh`, label/benchmark utilities).
- `experiments/` holds the few flagship entry points (`run_cleaning_detection`,
  `run_relation_*`, `workbench`, `build_report`) alongside ~60 one-off labeling /
  scoring / migration scripts — most are not part of the core pipeline.
