# RFT — Decisions log and canonical evaluation protocol

This is the single source of truth for the typo-detection RFT effort.
All future evaluations must follow the protocol in §3.

---

## 1. Data sources

| Purpose | Source | Path |
|---|---|---|
| **Gold benchmark** (the *only* thing eval is allowed to score against) | Hand-curated, post-cluster-contamination fix | `benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl` |
| Source tables | Per-column CSV samples (≤500 distinct values) | `data/benchmark_sources_v6/` |
| Train split (RFT only) | 60% of benchmark columns (by-CSV stratified) | `RFT/typo/v9/data/train.jsonl` (99 rows, 78 gold cells) |
| Valid split (RFT only) | 10% of benchmark columns | `RFT/typo/v9/data/valid.jsonl` (20 rows, 6 gold cells) |
| **Eval split (REQUIRED for all reporting)** | 30% of benchmark columns, *never seen by any FT job* | `RFT/typo/v9/data/eval.jsonl` (64 rows, 31 gold cells, 15 typo cols / 20 amb-only / 29 clean) |
| Demo / few-shot | 8 held-out columns | `RFT/typo/v9/data/demo.jsonl` |
| Trainee dev message | **Versioned** — fork of v8 with names put out of scope for char-level fixes; multiple `PROMPT_V9_*` variants live side-by-side and are gated through the §7 recall harness before any RFT job consumes them | [RFT/typo/v9/_prompt.py](RFT/typo/v9/_prompt.py) (current export: `PROMPT_V9`) |
| Precision rulebook | Fork of v7 rulebook with proper-noun char-level slips → fp | [RFT/typo/v9/prompts/rulebook.md](RFT/typo/v9/prompts/rulebook.md) |
| Gold reconciliation | Deterministic offline classifier moving proper-noun char-level gold → ambiguous | [RFT/typo/v9/_gold_reconcile.py](RFT/typo/v9/_gold_reconcile.py); audit trail in `RFT/typo/v9/data/gold_reconciliation_report.json` |
| Frozen prior split (kept for back-compat only) | v6 splits, no longer used for headline numbers | `RFT/typo/v6/data/` |

Data, benchmark, and output paths in this document are local, gitignored artifacts.

### Hard rule

> **All evaluation is against the human benchmark.**
> No synthetic data, no LLM-judged data, no leftover v5 synth. The
> only allowed eval file is `RFT/typo/v9/data/eval.jsonl`, which is
> derived 1:1 from the benchmark file above (verified — same column
> set as v8 modulo two gold typo cells reconciled from `gold_typos`
> into `ambiguous_values` by [RFT/typo/v9/_gold_reconcile.py](RFT/typo/v9/_gold_reconcile.py);
> zero CSV leakage across train/valid/eval).

---

## 2. Models under comparison

| ID | Kind | Notes |
|---|---|---|
| `base` | `o4-mini-2025-04-16` (no FT) | reference; run under v8 trainee prompt |
| `v8-r1-low` | RFT `o4-mini` | r1 additive reward, n_epochs=2, reasoning=low |
| `v8-slim` | RFT `o4-mini` | slim 2-grader reward, n_epochs=2 |
| `gpt-5.2` | base model | external baseline |
| `gpt-5.5` | base model | external baseline (current SOTA on this bench) |

FT models can only be deployed on the same Azure account where the
job ran, so historical runs used more than one private resource.
Deprecated v6-era RFTs (`longer`, `sharper`, `baseline`, `deep`) are no
longer benchmarked; the v8 reward grid (`r1`/`r2`/`r3`) supersedes them.
v9 RFT runs are pending — they will be added here once trained.

### 2.1 Pinned model identifiers

| Role | Model | Source of truth |
|---|---|---|
| **RFT base / trainee** | `o4-mini-2025-04-16` | [RFT/typo/v7/scripts/typo_rft_v7.py](RFT/typo/v7/scripts/typo_rft_v7.py) (`DEFAULT_BASE_MODEL`); v8 inherits via `--base-model` default |
| **Training-time grader** (inside the RFT job) | `gpt-4o-2024-08-06` | [RFT/typo/v5/_task.py](RFT/typo/v5/_task.py) (`TYPO_V5_DEFAULT_GRADER_MODEL`); v7/v8/v9 inherit |
| **Eval-time rollout judges** (§4 comparison harness only) | `gpt-5.2`, `gpt-5.4`, `gpt-5.5` (ensemble) | [RFT/typo/v8/eval/compare_v8.py](RFT/typo/v8/eval/compare_v8.py) (`grader_models=`) |
| **Recall-bench trainee** (§7) | `o4-mini` @ `reasoning_effort=low` | [RFT/typo/v9/eval/recall_bench.py](RFT/typo/v9/eval/recall_bench.py) |

The training-time grader (`gpt-4o`) and the eval-time judges
(`gpt-5.{2,4,5}`) are deliberately different models: the grader is the
reward signal the RFT job actually optimises against, while the eval
judges are an independent reference used post-hoc in §4 to score
rollouts on the held-out eval split. Treating them as the same model
would conflate "what we trained for" with "what we measure" and would
be circular.

---

## 3. Canonical evaluation protocol (USE THIS, nothing else)

### 3.1 Unit of evaluation

**Cell, not column.**

Discard column identity entirely. Pool every predicted value across
every column in the eval set into one global set. Compare to the
global set of gold typo cells.

### 3.2 Definitions

For each model `m`:

* $\hat{Y}_m = \bigcup_{c \in \text{eval}} \text{pred}(m, c)$ — all cells the model flagged anywhere
* $G = \bigcup_{c \in \text{eval}} \text{gold}(c)$ — all benchmark-gold typo cells (33 cells in v6 eval)
* $A = \bigcup_{c \in \text{eval}} \text{ambiguous}(c)$ — benchmark-ambiguous cells (FP immunity)

> **FP immunity for `A`**: a cell labelled ambiguous in the benchmark
> is *not* a false positive when flagged, because the human labeller
> wasn't sure either. It is also not a true positive (no recall
> credit). It is simply excluded from scoring.

### 3.3 Metric

$$
\text{TP} = |\hat{Y}_m \cap G| \qquad
\text{FP} = |\hat{Y}_m \setminus (G \cup A)| \qquad
\text{FN} = |G \setminus \hat{Y}_m|
$$

$$
P = \frac{\text{TP}}{\text{TP}+\text{FP}} \qquad
R = \frac{\text{TP}}{\text{TP}+\text{FN}} \qquad
F_1 = \frac{2PR}{P+R}
$$

**The headline number is the micro $F_1$.** Always report TP/FP/FN
counts alongside, because with only ~33 gold cells a single FP swings
$P$ visibly and totals communicate calibration better than ratios.

### 3.4 What is explicitly NOT used

* **Per-column macro F1.** Inflated by the 48/64 (75%) columns where
  abstaining gives a "free" $F_1 = 1$. Hides the actual signal.
* **The "abstain on a typo column ⇒ P=0" rule** from the original
  grader. It mixes precision with recall and produces nonsensical
  "low precision" numbers for over-cautious models that in fact made
  zero false positives.
* **Synthetic eval data.** No exceptions.

### 3.5 Reference implementation

The existing rollouts written by
[RFT/typo/v6/eval/compare_ft_vs_base_v6.py](RFT/typo/v6/eval/compare_ft_vs_base_v6.py)
already record per-column `tp/fp/fn` (with ambiguous immunity baked
in). To get the canonical metric, sum those three columns across all
rows of each model's `*.rollouts.jsonl`:

```python
TP=FP=FN=0
for line in open(f"{out_dir}/{model}.rollouts.jsonl"):
    r = json.loads(line)
    TP += r["tp"]; FP += r["fp"]; FN += r["fn"]
P = TP/(TP+FP) if TP+FP else 0.0
R = TP/(TP+FN) if TP+FN else 1.0
F1 = 2*P*R/(P+R) if P+R else 0.0
```

That's the entire metric. Anything more elaborate is a distraction.

---

## 4. Current results (v8 eval, 64 cols, 33 gold cells)

Run: `outputs/v8_compare/` plus the three GPT-5.5 prompt-sweep folders
(`outputs/v8_compare_gpt55_min/`, `outputs/v8_compare_gpt55_v7prompt/`,
`outputs/v8_compare_gpt55_r4/`). All numbers
computed by summing per-row `tp/fp/fn` from `*.rollouts.jsonl` exactly per §3.5.

| model | trainee prompt | preds | TP | FP | FN | micro P | micro R | micro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base `o4-mini` | v8 | 276 | 28 | 168 | 5 | 0.143 | 0.848 | 0.245 |
| RFT `v8-r1-low` | v8 | 36 | 24 | 4 | 9 | 0.857 | 0.727 | **0.787** |
| RFT `v8-slim` | v8 | 11 | 11 | 0 | 22 | **1.000** | 0.333 | 0.500 |
| `gpt-5.5` | v8 (min eff) | 94 | 31 | 39 | 2 | 0.443 | **0.939** | 0.602 |
| `gpt-5.5` | v7 | 30 | 22 | 3 | 11 | 0.880 | 0.667 | 0.759 |
| `gpt-5.5` | r4 | 91 | 30 | 41 | 3 | 0.423 | 0.909 | 0.577 |

### Findings

1. **The v8 trainee prompt is far more aggressive than v6's.** Base
   `o4-mini` jumps from 13 FP (v6 prompt) to 168 FP (v8 prompt) while
   recall holds (28/33 vs 26/33). The headline RFT precision gain is
   *relative to that aggressive prompt*, not to v6 baseline behaviour.
2. **RFT `v8-r1-low` is the current best F1 (0.787)** and the only
   model that keeps both legs alive: 24/33 TP at only 4 FP. It beats
   every GPT-5.5 prompt configuration tried so far on F1.
3. **`v8-slim` over-abstained again** — same failure mode as v6
   `longer`/`sharper`: zero FP but only 11/33 TP. The slim 2-grader
   reward concentrates pressure on precision too hard.
4. **GPT-5.5 prompt sensitivity is enormous.** Same model, three
   prompts → preds in {30, 91, 94}, F1 in {0.602, 0.577, 0.759}.
   The v7 prompt is the only one that approaches RFT F1, but with
   recall 0.667 vs RFT's 0.727. The v8/r4 prompts make GPT-5.5
   trigger-happy (39–41 FP).
5. The grader's precision pressure is still the dominant force in
   RFT training; `v8-r1` (additive) preserves more recall than the
   slim variant, which is why it currently leads.

---

## 4b. v9 results (2026-05-29 — new benchmark relabel, reconciliation-OFF gold, GPT-5 @ none)

Run: `outputs/v9_compare*`. Eval `RFT/typo/v9/data/eval.jsonl`
(64 cols, **19 gold cells** after further benchmark relabels; **reconciliation
disabled** — gold = the human benchmark's typo cells as-is, the LLM grader judges
proper-noun scope at grading time, see D13/D14). Cell-level **micro** (§3.5).
o4-mini base + RFT models at `reasoning_effort=medium` (trained effort);
**GPT-5 baselines at `reasoning_effort=none`** (D13). Each model runs the prompt
it was trained on, baked into the eval rows; GPT-5 runs whichever prompt is named.

| model | prompt | effort | TP | FP | FN | micro P | micro R | micro F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `gpt-5.5` | grader-detector (de-leaked) | none | 16 | 1 | 3 | 0.941 | 0.842 | **0.889** |
| `gpt-5.5` | grader-detector (cell-leaked, discard) | none | 16 | 0 | 3 | 1.000 | 0.842 | _0.914_ |
| RFT `v9-r1-e3` | v9 | med | 17 | 5 | 2 | 0.773 | 0.895 | 0.829 |
| RFT `v8-r1-low` | v8 | med | 17 | 6 | 2 | 0.739 | 0.895 | 0.810 |
| `gpt-5.5` | v7 | none | 14 | 2 | 5 | 0.875 | 0.737 | 0.800 |
| RFT `v9-r1-e2` | v9 | med | 15 | 6 | 4 | 0.714 | 0.789 | 0.750 |
| RFT `v9-r1-lr2` | v9 | med | 13 | 7 | 6 | 0.650 | 0.684 | 0.667 |
| RFT `v9-r3-base` | v9 | med | 18 | 22 | 1 | 0.450 | 0.947 | 0.610 |
| RFT `v9-r2-base` | v9 | med | 18 | 29 | 1 | 0.383 | 0.947 | 0.545 |
| RFT `v9-r1-base` | v9 | med | 18 | 29 | 1 | 0.383 | 0.947 | 0.545 |
| `gpt-5.5` | v9 | none | 17 | 27 | 2 | 0.386 | 0.895 | 0.540 |
| base `o4-mini` | v9 | med | 19 | 76 | 0 | 0.200 | 1.000 | 0.333 |
| `gpt-5.2` | v9 | none | 15 | 114 | 4 | 0.116 | 0.789 | 0.203 |

### Findings

1. **The grader-detector's verbatim cell leakage was minor (0.914 → de-leaked
   0.889).** The contaminated prompt embedded **3 of the 19 eval-gold cells
   word-for-word** (`GRAHAM INDEPENDANT`, `AQUATIC ENVIROMENTAL`, `Pilot
   Mountatin`). After replacing every benchmark string in the grader + rulebook
   with invented examples (verified 0 overlap, 2026-05-29) and rebuilding, gpt-5.5
   grader-detector @none scores **0.889** (P 0.941 / R 0.842) — only 1 new FP. So
   the prompt's lift comes from the RULES (verifiability criteria), not the
   memorized cells. **But those criteria were still authored against this
   benchmark this session** (criteria-level overfit) — 0.889 is *not* held-out;
   only new CSVs settle it (D10/D14).
2. **A well-prompted GPT-5.5 leads even de-leaked.** Clean grader-detector
   **0.889** > RFT `v8-r1-low` **0.810** ≈ gpt-5.5 + v7 **0.800** (both 0/19
   leakage) > `v9-r1-e2` 0.750. Prompting the strong base model still beats
   RFT-o4-mini here by ~0.08 — margin pending held-out confirmation.
3. **Prompt + effort still matter hugely for GPT-5.5** (clean prompts, @none):
   v9 recall-first 0.540 → v7 rulebook 0.800; and `none` > `medium` on the
   precision prompts (v7 0.743→0.800). The recall-first v9 prompt cripples
   precision (27 FP). → D13.
4. **Among RFT models, `v8-r1-low` (0.810) beats all v9** (best `v9-r1-e2`
   0.750): v9's proper-noun/diacritic descoping over-corrected, costing recall
   (v8 17/19 vs v9-r1-e2 15/19 at ~equal precision). RFT lifted base (0.333)
   sharply on precision *without* the old over-abstention, but didn't beat v8.
5. **`v9-r1-base` (epoch-1, default lr) = 0.545**, identical to `v9-r2-base`
   (both 18 TP / 29 FP / 1 FN: high recall, low precision = the untuned-reward
   profile). Its epoch-2 sibling `v9-r1-e2` reaches 0.750 by pruning FP 29→6
   (precision 0.383→0.714) at the cost of 3 TP (18→15) — a clean illustration of
   D11: RFT calibrates precision by *removing* candidates, it does not grow
   recall. So more epochs help only while the FP cut outweighs the TP loss.

### Caveats

- **Overfitting / leakage (finding 1):** the grader-detector 0.914 is
  contaminated; the trustworthy "well-prompted GPT-5.5" point is v7 ≈ 0.80,
  a tie with `v8-r1-low`. Validate any prompt claim on held-out CSVs (D10).
- 19 gold cells, single stochastic rollout per model — F1 gaps under ~0.1 are
  within o4-mini/GPT-5 run-to-run noise. v8-vs-v9 and variant ordering need
  multi-sampling.

---

## 5. Decisions and rationale

| # | Decision | Why |
|---|---|---|
| D1 | Use only `benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl` for all eval | Single point of truth, hand-curated, post-cluster-fix |
| D2 | Cell-level micro F1 (§3) is the only headline | Per-column macro inflated by 75% free-F1 rows; cell-level micro is what the user actually cares about |
| D3 | Ambiguous cells grant FP immunity but no recall credit | Matches the benchmark's labelling intent |
| D4 | v9 eval set is 64 cols, 31 gold cells (was 33 in v6/v8 — 2 reconciled into ambiguous by [RFT/typo/v9/_gold_reconcile.py](RFT/typo/v9/_gold_reconcile.py)) | Stratified by-CSV split, never seen during RFT training; reconciliation aligns the recall reward with the v9 precision rulebook so the trainee stops being penalised for *not* proposing proper-noun char-level fixes the rulebook would reject |
| D5 | Train RFT from human benchmark, not LLM-synth (v5) | v5 synth inflated train/eval similarity; v9 trains on a real benchmark slice that is disjoint from eval at the CSV level |
| D6 | v8 reward grid is `r1` (additive), `r2` (harmonic), `r3` (weighted) — `v8-r1-low` won (§4); `v8-slim` shows the 2-grader collapse | Ablate reward shape to characterise the abstention failure mode |
| D7 | Always report alongside base `o4-mini` and `gpt-5.{2,5}` | Without a non-RFT anchor, "RFT improved precision" is meaningless |
| D8 | FT models pin to the account that ran their job | Azure constraint; explains the multi-endpoint deployment matrix |
| D9 | v9 forks both *trainee prompt* (names out of scope for char-level fixes) AND *gold set* (proper-noun char-level gold → ambiguous) | The v8 trainee was being pushed to propose proper-noun char-level corrections that the precision sub-grader then rejected; the gap fed the abstention failure. v9 closes the loop on both sides. See [RFT/typo/v9/__init__.py](RFT/typo/v9/__init__.py) for the full rationale. |
| D10 | Do NOT iterate the trainee prompt against this benchmark to chase recall numbers; validate generalisation on a held-out CSV set | Single-run deltas of ~5 cells on a 33-cell benchmark are within the o4-mini-low noise floor (one column-level decision e.g. a 5-cell `Fuel Type` block flips the whole count). Prose-tuning against the same eval is the overfitting failure mode, not the cure. Enumerated concrete defect classes (`Foo Bar Bar Baz`, `I B M → IBM`) operationalise better for small models than abstract principles, but neither is validated without out-of-distribution evidence. |
| D11 | **The trainee prompt entering RFT must START high-recall**, even at the cost of precision — RFT reliably lifts precision, not recall. | Empirical pattern across every variant in §4: v6 `longer`/`sharper` (R=0.33/0.30, P=1.00) and v8 `slim` (R=0.33, P=1.00) all collapsed recall toward zero; the grader's precision pressure is asymmetric. `v8-r1-low` survived because its starting prompt was aggressive (base o4-mini under v8 trainee = R 0.85 with 168 FP), giving RFT signal to *prune* rather than *grow* the candidate set. Lesson: pick prompts that over-propose, then let RFT calibrate. A prompt that under-proposes pre-training will under-propose post-training too. |
| D12 | **Trainee prompts are versioned and iterated; multiple variants coexist.** Promotion from "variant" to "RFT input" goes through the §7 recall harness as the gate. | We expect to keep evolving the trainee surface (defect classes in scope, proper-noun rules, response schema hints). The recall harness (cheap, no grader) is the fast filter — a variant that doesn't clear a recall floor on the full benchmark (e.g. NT recall ≥ 0.88) is not worth burning RFT compute on. Variants live under [RFT/typo/v9/_prompt.py](RFT/typo/v9/_prompt.py) and are tracked by `prompt_hash` in `outputs/v9_recall/<tag>_summary_<hash>.json`. |
| D13 | **GPT-5 baselines (gpt-5.5, gpt-5.2) are evaluated at `reasoning_effort=none`**, not medium; o4-mini base + RFT models stay at their trained effort (medium). Run GPT-5 in a separate `--reasoning-effort none --only gpt-5.5,gpt-5.2` pass. | `none` is the intended operating point for gpt-5.5 and empirically gives it *higher* F1 on the precision prompts (v7 0.743→0.800, grader-det 0.865→0.914) — no-reasoning is more disciplined. |
| D14 | **Any prompt that was iterated against the benchmark (e.g. the §4b "grader-detector") is OVERFIT and its number is not load-bearing** — see §4b finding 1 (3/19 eval cells embedded verbatim). Reuses D10's discipline: prompt claims must be validated on held-out CSVs, not this benchmark. | The grader-detector 0.914 was contaminated by this-session tuning against the gold; the clean signal is the un-tuned v7 prompt (0.800), which ties `v8-r1-low`. |

---

## 6. Open questions

* Does increasing recall pressure in the grader (or adding a recall
  reward term) close the remaining recall gap of `v8-r1-low` (R=0.727)
  without re-introducing FPs?
* Is RFT on `gpt-5.x` (rather than `o4-mini`) a better starting
  point? GPT-5.5 zero-shot already sits at R=0.939 under the v8 prompt.
* The benchmark is small (31–33 gold cells). Confidence intervals on
  $F_1$ are wide — a follow-up should add bootstrap CIs to §4.
* Does v9's gold reconciliation + forked trainee prompt + r1 reward
  recover *both* legs (P and R together)? v9 RFT training run pending.

---

## 7. Recall-only harness (the promotion gate for new trainee prompts)

[RFT/typo/v9/eval/recall_bench.py](RFT/typo/v9/eval/recall_bench.py)
measures per-cell **recall only** of a `PROMPT_V9` variant on the full
benchmark (no grader, no precision call, ~30s end-to-end):

```bash
python -m RFT.typo.v9.eval.recall_bench --concurrency 12 --tag <T>
```

**Role.** Per D11, the trainee entering RFT must start high-recall. Per
D12, prompts are versioned and iterated. This harness is the cheap
gate between those two facts: every new prompt variant is run through
it before any RFT compute is committed. A variant that doesn't clear
the current minimum (working number: NT recall ≥ 0.88, i.e. no worse
than iter2) is rejected at this stage, not at RFT time. The minimum
is a floor, not a target — per F4 the achievable ceiling under the
current rulebook is closer to ~0.96 NT and ~8 NT cells of headroom
are still in-scope and recoverable.

Details:

* Reconstructs full columns from `data/benchmark_sources_v6/`
  (137 sources, all resolve); 67 cols have ≥1 gold typo cell, 233 gold typo cells total.
* Reports pool / non-trivial / trivial recall, where "trivial" =
  whitespace-only corrections that the official scoring excludes
  (`RFT/typo/v5/_task.is_v5_trivial_correction`). **Non-trivial recall
  (112 cells) is the metric to compare**, not pool recall.
* Per-column response cache keyed on `sha(<prompt>|source|column)` —
  editing the prompt invalidates it automatically. Variants are
  identified by `prompt_hash` in the summary JSON, so multiple
  candidates can be compared side-by-side without re-running shared
  columns.
* Current `PROMPT_V9` (hash `fdcd2cd6d836`): **non-trivial recall 0.893
  (100/112), pool recall 0.927 (216/233)** — above the minimum,
  eligible as the trainee starting point for the next RFT run, but
  with ~8 in-scope NT cells of recoverable headroom per F4.
* **Caveat (D10).** o4-mini @ `reasoning_effort=low` is stochastic;
  the ~12–17 NT-miss set shifts ±2–4 cells run-to-run. Use this
  harness as a *negative* signal (large recall drop = real regression),
  not for ranking prompts on small deltas — and never replace the
  §3 cell-level micro F1 with it for headline reporting.

---

## 8. PROMPT_V9 iteration log

Tracked per-hash recall on the §7 harness (o4-mini @ `reasoning_effort=low`).
Each iteration changed *only* the proper-noun-handling paragraph; the
rest of the prompt body is held constant. Summaries on disk under
`outputs/v9_recall/`.

| iter | prompt_hash | change | NT recall | pool recall |
|---|---|---|---:|---:|
| base | `f48a1d0f8bd9` | v9 forked from v8 (names out of scope for char-level fixes) | 0.875 (98/112) | 0.897 (209/233) |
| iter1 | `6c5da43b02d6` | + "common dictionary word inside a name with a clear slip is in scope" | 0.875 (98/112) | 0.901 (210/233) |
| iter2 | `fdcd2cd6d836` | + "a run of spaced single-letter tokens is a spacing defect, close it up (`I B M → IBM`, `U S A → USA`)" | **0.893 (100/112)** | **0.927 (216/233)** |
| iter3 | `673dc63190c1` | Replaced enumerated clauses with one abstract principle ("defect detectable from FORM alone vs needing name knowledge") | 0.848 (95/112) | 0.884 (206/233) |
| iter4 | `cd483415d21d` | Hybrid: abstract principle + four generic bullet examples | 0.830 (93/112) | 0.884 (206/233) |
| current | `fdcd2cd6d836` | **Reverted to iter2** after iter3/iter4 regressed | 0.893 (100/112) | 0.927 (216/233) |

### Findings recorded from this cycle

**F1. Enumerated concrete defect classes beat abstract principles for small models.**
The iter2→iter3 rewrite kept the same *coverage* in prose ("dup-words,
doubled punctuation, spaced single letters, common-word-in-name") but
recast it as one general discriminator. o4-mini-low stopped reliably
applying it: iter3 lost the dup-word case (`Remote-Controlled Aircraft
Aircraft`), a plain-dictionary misspelling (`AQUATIC ENVIROMENTAL`), and
the spelling slip `Tuggeranon→Tuggeranong`. iter4 (hybrid) regressed
further and lost `A T & T→AT&T` again. **Enumeration is instruction;
abstraction is hope.**

**F2. Concrete examples of general defect *classes* are NOT benchmark
overfitting.** iter2's anchors are all generic strings (`Foo Bar Bar Baz`,
`I B M`, `U S A`, `Bright Hosptial Trust`), not benchmark values. The
proof: when the abstract rewrite "generalized" the wording, it actually
*lost* generally-in-scope cases. Generic examples for general classes
transfer; benchmark-string examples (e.g. literally `A T & T → AT&T`)
would not. Encode general classes, not specific instances.

**F3. Single-run deltas of ~5 NT cells are noise.** The iter2→iter4
delta (100→93 NT hits) is dominated by a *single* column decision:
`Fuel Type` has 5 identical-value rows, so one model-level flip on that
value moves 5 cells of the count. Combined with binomial Latin-name
toggling, that's most of the apparent drop. Rank prompts only on miss
*composition* (which cells, which classes), never on the headline NT count
alone. This formalises the basis for D10.

**F4. There is real headroom above iter2's 0.89 NT — it is NOT a
design ceiling.** Of the 12 NT misses at iter2, only **~3–4 are truly
out of reach by v9 design** (scientific binomials in `Nom commun
d'animal` / `Common animal name`, possibly the surname char-slip
`VILLAREAL→VILLARREAL`). The other **~8 are in-scope by the current
prompt's own rules** but get dropped, partly stochastically and partly
because the prompt operationalises them weakly:

* dup-word: `HURON CITY CITY SD → HURON CITY SD`
* dup-segment: `North Bennington/Bennington → North Bennington`
* common dictionary word in a name: `Inland Valley Developmental →
  Development Agency`
* word-boundary: `Hannah's Foodmart → Food Mart`, `Yellow Fin Tuna →
  Yellowfin Tuna` (×2), `Midpeninsula` spacing
* French dictionary slip: `laquaiches argentée → laquaiche argentée`

The achievable ceiling is therefore closer to ~0.96 NT (108/112)
under the current rulebook, not 0.89. The `gpt-5.5 (v8 prompt)` row
in §4 — R=0.939 on the eval split — corroborates that the ceiling
sits well above 0.89; it just costs precision the v9 rulebook then
rejects. Treat 0.89 as "where iter2 currently sits," not "as good as
it gets."

**F5. Generalisation is validated OOD, not on this benchmark.** Three
prompt rewrites against this same eval produced one improvement
(iter2's initialism clause, anchored by a real failure case) and two
regressions (iter3, iter4 — abstract rewrites). Further prose-tuning
against the same 67 columns is the overfitting risk D10 warns about.
Validate the next prompt variant on a held-out CSV set (e.g. one of the
other curated sources under `data/`) — that's the real test of
the "generalize, don't memorize" goal.
