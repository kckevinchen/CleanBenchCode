# ForestED (LLM-induced decision-tree ensemble) on CleanBench

[ForestED](https://github.com/mengqiwang1/ForestED) — *"Ensembling LLM-Induced
Decision Trees for Explainable and Robust Error Detection"* — is a **few-shot**
cell-level error detector. From a clean/dirty pair it:

1. profiles the table into a **data catalog** (`build_data_catalog`),
2. **GP-uncertainty-samples ~5%** of rows as labeled examples (`gp_sampling`),
3. prompts an **LLM to synthesize a decision-tree spec** of `rule` (executable
   Python) and `gnn` nodes per partition (`call_llm_generate`),
4. trains any GNN nodes, runs the forest over the full table, and
5. **ensembles partitions by majority vote** into a consensus prediction.

It is the public-code substitute for the (code-less) Zhang DASFAA baseline.

## Honest, leakage-free evaluation

ForestED needs labels (the GP-sampled rows). To avoid the value-memorization that
inflates raha/Baran/TabClean, we run it on a **synthetic** signal exactly like
those: the "dirty" table is the real table with a fixed synthetic injection
(`external/synth_fewshot/`) and the "clean" oracle is the original, so ForestED's
internal ground truth = the **synthetic** errors only — never a real benchmark
label. Detection = cells the consensus marks erroneous. The GP-sampled rows are
held out and the synthetic cells neutralized at scoring time (sidecars).

**Key result:** unlike raha/Baran/TabClean (which collapse to F1 ≈ 0.01 under
synthetic training because they *memorize* dirty→clean value pairs), ForestED does
**not** collapse — it induces *general validation rules* (e.g. "GPA ∈ [0, 4]") that
transfer from synthetic to real errors. This is the method's headline advantage.

## Adapting it (what `adapter.py` does)

The upstream repo hardcodes `OpenAI(api_key="")` (the empty key overrides any env
var) and calls `chat.completions.create` to induce the tree. `adapter.py`:

1. stubs `openai.OpenAI` **before** importing `decision_forest`, then replaces the
   module's `openai_client` with a shim whose `.chat.completions.create` delegates
   to our Azure `LLMClient.query`;
2. forces the majority-vote consensus onto **CPU** (it defaults to `device="cuda"`).

`run_forested.py` additionally caps `pca_components` for tiny tables and coerces
numpy types in the catalog JSON.

## Setup

```bash
bash external/setup_upstreams.sh
cd external/forested
python3.12 -m venv venv
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install torch_geometric
./venv/bin/pip install torch_scatter -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
./venv/bin/pip install pandas numpy scikit-learn scipy openai "azure-identity>=1.14.0"
az login
```

## Run

```bash
external/forested/venv/bin/python external/forested/run_forested.py --limit 1   # smoke
external/forested/venv/bin/python external/forested/run_forested.py \
    --scope sample --clean-sample 100 --workers 6 --model gpt-5.5

python external/rein/score.py --predictions external/rein/results/predictions_forested.jsonl
python external/rein/analyze_by_type.py
```

`--scope {labeled,all,sample}`, `--model` (default `gpt-5.5`; ~1 LLM call/table),
`--workers` (CPU-bound GNN training; 16 cores here). Per-table results cache under
`cache/` (resumable). Predictions + `.labeled_rows.json` (held-out GP rows) +
`.synthetic_cells.json` land in `external/rein/results/`.

## Layout / notes

```
external/forested/
  adapter.py        # openai seam -> Azure LLMClient; CPU consensus patch
  run_forested.py   # per-table runner: synthetic inject, catalog, GP sample, pipeline, consensus
  README.md
  ForestED/         # upstream clone (gitignored)
  venv/             # py3.12 torch+torch_geometric+torch_scatter CPU (gitignored)
  cache/  work/     # per-table cache + pipeline scratch (gitignored)
```

- **Few-shot, synthetic, held-out.** Report in the few-shot section, not the
  zero-shot leaderboard.
- **Cost / speed.** ~1 LLM call per table (small tables = 1 partition) but
  ~60–110s/table from CPU GNN training; run on a tractable scope (labeled + clean
  sample), like TabClean.
- **Faithful, with a CPU caveat.** We run the upstream pipeline unchanged except
  the LLM seam and forcing CPU; the GNN nodes train on CPU (slower than the paper's
  GPU setup, same algorithm).
