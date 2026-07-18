# TabClean on CleanBench (cell-level)

[TabClean](https://github.com/Wangyibo321/TabClean) (Wang, Zhang, He, Su,
Bhargava, Liu — arXiv:2606.25388) is a **few-shot, multi-agent** LLM pipeline
that *synthesizes a reusable Python cleaning program*:

```
profile → diagnose → strategize → code → test_dev → eval_dev → review → decide → apply_full
```

It builds an adaptive **development sample** from dirty/ground-truth pairs,
iterates (keeping the best dev-scoring code), then applies that program to the
full table. Because it learns from ground-truth labels on the dev rows, it is
**few-shot** and is evaluated **held-out** — exactly like raha and Baran in
`external/rein/` (the TabClean paper itself reports "cleaning quality … on
held-out rows").

This harness runs the *upstream* pipeline unchanged except for two seams:

1. **LLM backend.** Upstream builds every agent via one factory,
   `pipeline._make_llm(model) → langchain_openai.ChatOpenAI`, keyed on an
   `OPENAI_API_KEY`. We don't have one — auth is `AzureCliCredential` via our
   framework-free `src/core/llm_client.py::LLMClient`. `adapter.py` monkeypatches
   `_make_llm` to a thin shim whose `.invoke()` delegates to `LLMClient.query`
   (endpoint rotation, Responses/Chat fallback, dead-pair skipping all reused)
   and remaps TabClean's default per-agent models (`gpt-5-mini` / `gpt-5.2` /
   `gpt-5.3-codex`, which our Azure pool does not serve) onto ones it does
   (default: **gpt-5.5** for every agent; override via `TABCLEAN_MODEL*` env).
2. **Dev-set size.** Our benchmark tables are ~100 rows; TabClean's native
   adaptive sizer floors the dev set at 100 rows, which would label the *entire*
   table and leave nothing to hold out. `--dev-min-rows` (default **20**, the
   same labeling budget as raha/Baran) caps it so held-out test rows remain.

**Detector semantics.** TabClean is a *repair* system; we score it as a detector
via **changed cells** — a cell is a detection iff its value in `final_cleaned.csv`
differs from the dirty value (same convention as CleanAgent). The dev rows it
learned from are written to a `.labeled_rows.json` sidecar so `score.py` holds
them out of both predictions and ground truth.

## Layout

```
external/tabclean/
  adapter.py          # route pipeline._make_llm -> our Azure LLMClient (+ cost est.)
  run_tabclean.py     # per-dataset runner: dev capture, pipeline, changed-cell detection
  README.md
  TabClean/           # upstream clone (gitignored)
  venv/               # py3.12: langgraph + langchain-openai + pandas + our client deps (gitignored)
  cache/  work/       # per-table result cache + pipeline scratch (gitignored)
```

## Setup

```bash
bash external/setup_upstreams.sh
cd external/tabclean
python3.12 -m venv venv
./venv/bin/pip install langgraph langchain-openai pandas python-dateutil openai "azure-identity>=1.14.0"
# datasets are the materialized REIN ones:
python ../rein/prepare_datasets.py --include-clean-universe   # if not already done
az login
```

## Run

```bash
# pilot one labeled table
external/tabclean/venv/bin/python external/tabclean/run_tabclean.py --scope labeled --limit 1

# headline run: error-bearing tables + a clean sample to check false positives
external/tabclean/venv/bin/python external/tabclean/run_tabclean.py \
    --scope sample --clean-sample 100 --dev-min-rows 20 --workers 8

# score (held-out is the default — a .labeled_rows.json sidecar is present)
python external/rein/score.py --predictions external/rein/results/predictions_tabclean.jsonl
python external/rein/analyze_by_type.py            # consolidated per-type P/R/F1
```

`--scope {labeled,all,sample}` selects the 158 error-bearing tables, all 1002, or
error-bearing + N clean. Per-table results are cached under `cache/` (resumable).
`TABCLEAN_REASONING_EFFORT` (default `medium`) and `TABCLEAN_MODEL` (default
`gpt-5.5`) tune the agents.

## Notes / fidelity

- **Few-shot, held-out.** Report TabClean in the *few-shot* section alongside
  raha/Baran, never the zero-shot leaderboard.
- **Small-table caveat.** TabClean was designed for large, error-dense tables
  (the paper's benchmarks are 1k–200k rows); our tables are ~100 rows and
  sparsely corrupted, so each dev set (20 rows) carries only a few error
  examples. Results reflect that regime.
- **Cost.** ~$0.1–0.9 per table depending on how many refine iterations a table
  needs (early-stop patience 3, F1 threshold 0.95, ≤10 iterations).
