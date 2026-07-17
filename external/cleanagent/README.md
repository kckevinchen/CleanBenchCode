# `external/cleanagent/` — CleanAgent on `benchmark/merged_v2`

[CleanAgent](https://github.com/sfu-db/CleanAgent) (Qi, Miao, Wang,
*Automating Data Standardization with LLM-based Agents*, arXiv:2403.08291)
repurposed as a cell-level **error detector** on the cell benchmark, reusing the
REIN harness so it scores next to `raha` and the typo specialists.

## What CleanAgent actually is (and how we adapt it)

CleanAgent is a Chainlit/AutoGen app whose pipeline is:
1. a **`col_annotator`** LLM agent labels each column with one of **exactly five
   standardizable types** — `email, phone, date, country, IP Address`;
2. a **`coder`** agent calls the matching **`dataprep.clean.clean_<type>`**
   function to standardize each typed column.

We don't run the interactive web/agent loop over 215 CSVs. Instead we replicate
its two steps programmatically and turn standardization into a **detector**:

> a cell is flagged when `clean_<type>` **changes or cannot parse** it — i.e. its
> raw value is non-standard for the column's declared type.

CleanAgent's exact `COL_ANNOTATOR_SYSTEM_MESSAGE` and type list are loaded by
`prompts.py` from the pinned, ignored upstream clone's `CleanAgent/main.py`.
Prompt text is not redistributed by this wrapper. Run `setup.sh` before the
annotation phase.

## Two-venv architecture (on purpose)

The two phases need incompatible environments, so they're decoupled via a file:

| phase | script | env | needs |
|---|---|---|---|
| 1. annotate | `annotate.py` | **main venv** (py3.12) | `src.core.llm_client` (Azure) |
| 2. detect | `detect.py` | **`external/cleanagent/venv`** (py3.10) | `dataprep` |
| 3. score | `external/rein/score.py` | any | — |

`dataprep` pins an old pandas (1.5.3) that **does not build on Python 3.12**, so
`setup.sh` provisions a standalone CPython 3.10 with `uv` just for phase 2. The
LLM phase stays in the main venv (Azure `LLMClient`). They communicate through
`annotations.json` (`{dataset: {columns: {col: cleaner_key|null}}}`).

## LLM used

The CleanAgent paper hard-codes **`gpt-4o-2024-08-06`** (OpenAI). We drive the
annotator with **Azure `gpt-4o`** via the repo's `LLMClient` — the closest
available match (the exact `...-2024-08-06` suffix is not an Azure deployment
name; plain `gpt-4o` is, and is served by **all 11** endpoints in the pool, so
calls round-robin/parallelize freely). Override with `--model`.

## Setup & run

```bash
bash external/cleanagent/setup.sh          # clone pinned commit + uv py3.10 venv + dataprep

# datasets are materialized by the REIN adapter (once):
python external/rein/prepare_datasets.py

# phase 1 — annotate (main venv, az login required):
source venv/bin/activate
python external/cleanagent/annotate.py --model gpt-4o          # 215 calls, ~13 min
#   --heuristic   LLM-free fallback (dataprep parse-rate; run in the 3.10 venv)

# phase 2 — detect (3.10 venv):
external/cleanagent/venv/bin/python external/cleanagent/detect.py --mode changed
external/cleanagent/venv/bin/python external/cleanagent/detect.py --mode unparseable

# score (+ neutral set):
python external/rein/score.py --predictions external/rein/results/predictions_cleanagent_changed.jsonl
```

### Detection modes
- `changed` (default) — flag cells where normalized standardized value != original
  **or** unparseable. Recall-oriented (but date reformatting inflates FPs).
- `unparseable` — flag only cells `clean_<type>` set to NaN. Precision-oriented.

## Results (full benchmark, gpt-4o annotation, neutral set applied)

186 columns were typed (123 date, 24 phone, 21 email, 16 country, 2 IP).

| detector | P | R | F1 | typo R | inconsist. R | **semantic R** |
|---|---|---|---|---|---|---|
| `cleanagent_changed` | 0.008 | 0.076 | 0.014 | 0.016 | 0.129 | **0.615** (16/26) |
| `cleanagent_unparseable` | 0.012 | 0.013 | 0.012 | 0.012 | 0.000 | 0.308 (8/26) |
| `raha` (reference) | 0.785 | 0.690 | 0.735 | 0.693 | 0.672 | 1.000 |

**Interpretation — a faithful, narrow result.** CleanAgent only knows five
standardizable types, which on this spreadsheet benchmark map mostly to **date**
columns. Its genuine niche is **`semantic_incompatible`**: values that can't be
standardized to their column's declared type (e.g. a non-date in a date column) —
it recovers **62%** of those. Elsewhere precision is low because standardizing
dates rewrites every cell's format (FP flood in `changed`) and the benchmark is
typo/inconsistency-heavy, outside CleanAgent's five-type scope. This is the
honest ceiling of a standardization tool used as a general error detector, and a
useful contrast to `raha` (broad) and the typo specialists (recall-heavy).

## Limitations / notes

- Narrow by design: only email/phone/date/country/IP columns are ever inspected.
- `changed` mode conflates "reformatted" with "erroneous" for dates; `unparseable`
  is the higher-precision read.
- Annotation quality depends on the LLM; `gpt-4o` and `gpt-5.4` gave near-identical
  type counts here.
- Faithful to CleanAgent's *method* (LLM annotate -> `dataprep.clean`), not its
  Chainlit/AutoGen UI or its standardization-accuracy evaluation.
