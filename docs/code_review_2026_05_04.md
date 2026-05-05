# Pre-demo Code Review — 2026-05-04

Scope: every Python module under `src/`, every CLI under `experiments/`, and
every prompt YAML under `prompts/`. 826 tests pass on `main`.

This document is the **single source of truth** for what we found and what
we propose to do before the demo. Each finding has:

- **File / line range** (verified against the actual repo at this date)
- **Severity**: `critical` / `high` / `medium` / `low`
- **Why it matters** for the demo
- **Recommended fix**

Findings the upstream sub-agent reports flagged but that are actually
**documented design choices** (e.g. "GT denominator uses raw_hits") have
been removed — they are safe.

---

## 1. Critical — fix before demo

### 1.1 EmbeddingClusterer reloads the SentenceTransformer on every CSV
- **File**: [src/core/row_dup_clusterers.py](../src/core/row_dup_clusterers.py#L734-L760)
- **Why**: `RowDuplicationDetector` instantiates **one** `EmbeddingClusterer`
  but calls `cluster()` once per source CSV. Inside `cluster()`, when the
  caller did not preload an embedder, we lazy-import `sentence_transformers`
  and call `SentenceTransformer(model_name)` **every invocation** — that's a
  ~80 MB model download/load per CSV. On a `--csv-dir` run with 30 CSVs the
  detector loads the model 30 times: minutes of latency, GBs of churn.
- **Fix**: cache the loaded model and the `_encode` callable on the
  dataclass after first construction. Implemented in this review (see
  §6 below).

### 1.2 LLM retry uses linear backoff that ignores 429 / 503 hints
- **File**: [src/core/llm_client.py](../src/core/llm_client.py#L260-L275)
- **Why**: `time.sleep(self.backoff_base_seconds * attempt)` is linear
  (10s, 20s, 30s, 40s, 50s with `backoff_base_seconds=10`). Azure OpenAI
  429s and the GPT-5 family rate-limit windows want **exponential** backoff
  with jitter; under demo load (multi-CSV mode + Phase B + relation
  validation all in flight) we will get long stalls instead of fast
  recovery, and identical retries fan out to the same hot endpoint
  because `_next_client()` is round-robin, not "skip the one that just
  failed".
- **Fix**:
  1. Switch to `min(cap, base * 2**(attempt-1)) + jitter` (exponential
     with cap, e.g. cap=60s).
  2. On exception, advance the round-robin index by 1 again so we don't
     immediately retry on the same client.
  3. Implemented in §6.

### 1.3 `_coerce_to_bool_series` silently turns NaN into `False`
- **File**: [src/core/relation_validator.py](../src/core/relation_validator.py#L188-L194)
- **Why**: The relation validator runs `executable` on a frame and gets a
  Series. NaN almost always means *the rule did not apply to this row*
  (e.g. an `if Cadence == 'Monthly' then …` rule on a Quarterly row).
  Today every NaN becomes `False` → counts as a violation → pass-rate
  collapses → high-quality rules are deleted by `min_pass_rate`. We have
  no way to recover the rule afterwards.
- **Fix**: keep NaN as-is in the mask, exclude NaN from the denominator
  when computing `pass_rate`, and emit `n_evaluated` separately in the
  report. The dashboard already has room for the field.

### 1.4 Multi-table relation extraction can mis-attribute violations on
non-default frame indices
- **File**: [src/core/relation_validator.py](../src/core/relation_validator.py#L834-L842)
- **Why**: We re-index the per-sheet mask to the source frame's index
  (`sub_mask.index = df_s.index`) and then call `df_s.loc[fail, keep]`.
  If `df_s.index` has duplicates (which happens when callers concatenate
  Excel sheets without `reset_index`), `.loc` returns multiple rows per
  position and the `__row__` value emitted with each violator becomes
  ambiguous. The labeling UI will then point reviewers at the wrong row.
- **Fix**: assert a unique RangeIndex at the boundary
  (`assert df_s.index.is_unique`) **and** prefer positional indexing
  (`df_s.iloc[positions]`) so the result is index-agnostic. See §6.

---

## 2. High severity — fix before demo if at all possible

### 2.1 `_fold_predicate` does not normalize operator variants
- **File**: [src/core/relations.py](../src/core/relations.py#L177-L185)
- **Why**: NFKC turns `≤`/`≥` into `<=`/`>=` but does not touch `!=` vs `<>` 
  or `==` vs `=`. Two semantically identical rules emitted by different
  passes therefore land in different species buckets, inflating the
  Good-Turing singleton count and (worse) producing duplicate rules in
  the dashboard.
- **Fix**: normalize `!=`, `≠`, `<>` → `!=`; `==` → `=` before tokenising.

### 2.2 Discovery prompt is huge (2 000 lines) and instruction-following
degrades across long contexts
- **File**: [prompts/discover_relations.yaml](../prompts/discover_relations.yaml)
- **Why**: GPT-5.2 ignores late instructions when the system prompt
  exceeds ~1500 lines (we observed this in the gpt-5.2 sweep on
  2026-05-01). Aggregation/generalize passes routinely re-emit
  value-keyed paraphrases that the discovery prompt explicitly bans.
- **Fix (post-demo OK)**: extract the long galleries to a separate
  `_appendix` section that we *don't* include in the system message.
  Keep the red-lines as a tight 5-bullet list. Tracked separately —
  this is a multi-day refactor, **not** a one-day fix.

### 2.3 Empty-cell semantics in base prompt are undefined
- **File**: [prompts/error_types.yaml](../prompts/error_types.yaml#L80-L100)
- **Why**: The base prompt says "do not flag empty cells unless the type
  explicitly cares" without defining empty. Today the model treats
  pandas-NaN, `""`, and `"   "` differently across types. This is a
  precision issue: each variant produces different findings on the same
  data and the demo will show non-determinism that is actually a prompt
  bug.
- **Fix**: append a one-paragraph definition to `base.system` (NaN, `""`,
  whitespace-only). Also clarify that empty cells are excluded from peer
  counts and dominant-baseline denominators. Implemented in §6.

### 2.4 `R4` majority guard says "roughly every non-empty row"
- **File**: [prompts/error_types.yaml](../prompts/error_types.yaml#L185-L189)
- **Why**: "roughly" is not a number. Two runs disagree on identical data.
- **Fix**: replace with `>=80% of the non-empty rows in the column`. Same
  threshold used in the dominant-baseline rule (R3). Implemented in §6.

### 2.5 `verify_entity` near-verbatim definition admits abbreviation
expansion
- **File**: [prompts/verify_entity.yaml](../prompts/verify_entity.yaml#L97-L99)
- **Why**: The rule says "case- and punctuation-tolerant" which the
  verifier reads as license to match "FBI" against "Federal Bureau of
  Investigation" via punctuation-strip. Web verification then accepts
  weak evidence and stamps clean values as suspicious.
- **Fix**: spell out the allowed transforms (case, whitespace, ASCII
  punctuation) and explicitly forbid abbreviation expansion, synonym
  substitution, and paraphrase. Implemented in §6.

### 2.6 `canonicalize_values` selection is subjective
- **File**: [prompts/canonicalize_values.yaml](../prompts/canonicalize_values.yaml#L48-L58)
- **Why**: The model is told to "pick the most explicit/full-length
  spelling preferred". On the same input it picks "USA" once and
  "United States" the next time. The downstream collision scan then
  groups rows under whichever token appears in the canonical map this
  pass — which means the same dataset emits different duplicate clusters
  on rerun.
- **Fix**: give a deterministic tiebreaker: longest non-abbreviated form
  → most frequent in input → first occurrence. Implemented in §6.

---

## 3. Medium severity — fix after demo

### 3.1 `enumerate_collision_groups` drops rows when any key column
normalizes to `""`
- **File**: [src/core/multi_row_priors.py](../src/core/multi_row_priors.py#L495-L500)
- **Why**: A key column whose value is `"---"` or `"."` normalizes to `""`
  and the entire row is excluded from collision groups. Two rows that
  share a real ID but have such a placeholder for a *secondary* key
  column never collide.
- **Fix**: convert empty post-normalization values to NaN per-column and
  re-use the `notna().all(axis=1)` filter. Easy 5-line change but it
  changes test output in `test_multi_row_priors.py` so it ships
  post-demo.

### 3.2 Rewriter prompt has no fallback when JSON parse fails
- **File**: [src/core/relation_aggregate.py](../src/core/relation_aggregate.py#L300-L320),
  [src/core/relation_generalize.py](../src/core/relation_generalize.py#L290-L307)
- **Why**: Both modules attempt naive outermost-brace extraction when
  `json.loads` fails. A response like
  `{"x":1} The reasoning: {note}` parses the right slice incorrectly
  and the rewrite is silently dropped (`error: …` returned but caller
  ignores it).
- **Fix**: switch to `re.finditer(r'\{(?:[^{}]|(?R))*\}', text)` (regex
  with recursion via `regex` module) or, simpler, instruct the LLM to
  emit JSON between sentinel tokens (e.g. `<<JSON>>{…}<<END>>`) and
  parse the slice between them. Tracked.

### 3.3 `_safe_recall` treats empty arms as recall=1.0
- **File**: [experiments/analyze_stop_thresholds.py](../experiments/analyze_stop_thresholds.py#L255-L270)
- **Why**: Threshold sweeps look artificially good when many arms are
  empty, because empty arms contribute neutral 1.0. The recommended
  threshold is biased toward stopping early.
- **Fix**: add a `skip_empty_arms` flag (default True) and exclude arms
  with `denom == 0` from both numerator and denominator.

### 3.4 `inner_loop.py` silently clamps `k_min > k_max`
- **File**: [src/adaptive/inner_loop.py](../src/adaptive/inner_loop.py#L55-L60)
- **Why**: A misconfigured arm config saturates after one pass and the
  user has no idea.
- **Fix**: log a warning when clamping; keep behavior so existing tests
  pass.

### 3.5 `EndpointPool` round-robin does not deprioritize a failing endpoint
- **File**: [src/core/llm_client.py](../src/core/llm_client.py#L162-L182)
- **Why**: Same as 1.2 but more general — even after a successful retry,
  the next worker still sends to the same hot endpoint.
- **Fix**: piggy-back on §1.2 — advance index on failure. Implemented.

### 3.6 `iqr_multiple` was bumped from 3 → 5 but the change is documented
inline and not in the changelog
- **File**: [prompts/error_types.yaml](../prompts/error_types.yaml#L57-L60), L408-L412
- **Why**: Anyone re-running an old experiment will get different
  numeric_outlier counts and not know why.
- **Fix**: add a `# bumped to 5 on 2026-04-30 to suppress heavy-tail FPs`
  marker in both places. Implemented in §6.

### 3.7 Stray `test.py` at repo root is unrelated demo code
- **File**: [test.py](../test.py)
- **Why**: It is **not** part of the test suite (pytest collects only
  `tests/`) but it exists at the root with hard-coded endpoint and
  agent IDs. Running `python test.py` (which the workspace's terminal
  history shows the user did) errors because the agent ID is dead. It
  also looks like the answer to "is this an example or a test?" is
  "neither" and reviewers will be confused.
- **Fix**: move under `examples/azure_agents_demo.py` or remove. We do
  **not** delete it without confirmation (operational-safety rule).

---

## 4. Low severity — backlog

| Finding | File | Notes |
|---|---|---|
| Dead code in `_WS_TRANSLATE` second-pass loop | [src/core/text_normalize.py](../src/core/text_normalize.py#L69-L77) | All values have `len(v) <= 1`, so the `if len(dst) > 1` branch is unreachable. Implemented removal in §6. |
| `LLMClient._extract_reasoning_summary` catches bare `Exception` | [src/core/llm_client.py](../src/core/llm_client.py#L173) | Cosmetic. |
| `parse_llm_json` aggregates parse errors but reports a generic message | [src/core/detector.py](../src/core/detector.py#L96-L105) | Add per-candidate detail at debug log level. |
| `Cluster` ID format `c{idx:03d}` overflows past 1000 | [src/core/aggregator.py](../src/core/aggregator.py#L345-L357) | Theoretical only; raise to `:04d`. |
| `discover_relations` does not specify `tables` field in example output | [prompts/discover_relations.yaml](../prompts/discover_relations.yaml) | Add the field. |
| `aggregate_relations` lacks expected-output guidance | [prompts/aggregate_relations.yaml](../prompts/aggregate_relations.yaml) | Add a paragraph on `canonical: null`. |
| `row_duplication` "identical" not defined | [prompts/error_types.yaml](../prompts/error_types.yaml) | Define equality precisely. |
| `BlockingClusterer` defaults docstring vague | [src/core/row_dup_clusterers.py](../src/core/row_dup_clusterers.py#L345-L370) | Documentation. |
| Magic numbers across `table_digest`, `llm_client`, `data_loader` | various | Hoist to `src/core/constants.py`. |

---

## 5. Items the upstream sub-agent flagged that we **kept** (no change)

These either misread the code or quoted an outdated version. They are
explicitly **not** bugs:

1. **"Chao1 GT denominator violates i.i.d."** — the docstring at
   [src/adaptive/_chao1.py](../src/adaptive/_chao1.py#L92-L165) explicitly
   adopts Wikipedia's Interpretation B, where each emission is one
   "individual" and the textbook GT identity holds:
   `N = Σ_r r·n_r = Σ_x R_x`. The estimator is mathematically
   self-consistent under that interpretation; the surprise is purely
   pedagogical. We keep it.

2. **"FN-rate CI is mathematically invalid"** — the report claimed
   `(ci - s_obs)/ci` is wrong; in fact, `FN(total) = (total - S_obs) / total`
   is **strictly monotonic** in `total`, so transforming the log-normal
   CI bounds on `total` is a valid 95% CI on FN-rate by the standard
   "monotonic-transform" property of CIs. Keep.

3. **"affected_rows accepted without validation when frame=None"** —
   [src/core/report_builder.py](../src/core/report_builder.py#L257-L268)
   already logs `dropped_oob` at debug level when a frame is provided.
   The `frame=None` path is reachable only in unit tests; production
   always passes the source frames map. Keep, but add a log line in §6.

4. **"`whole_table` token-budget undocumented"** — the `whole_table_max_rows=200`
   default works comfortably on every table in `data/curated_pbi/` (we
   measured 12 k tokens worst-case). Documentation can wait.

---

## 6. What this PR changes

The accompanying patch implements the high-confidence fixes that are
safe to ship today:

- §1.1 EmbeddingClusterer caches the loaded model.
- §1.2 LLMClient uses exponential backoff with jitter and skips
  the failing endpoint on retry.
- §1.3 `_coerce_to_bool_series` preserves NaN; `pass_rate` excludes
  unevaluated rows.
- §1.4 multi-table validator asserts unique index and uses positional
  indexing.
- §2.1 `_fold_predicate` normalizes operator variants.
- §2.3 base prompt empty-cell definition.
- §2.4 R4 majority guard now says `>=80%`.
- §2.5 verify_entity tightens "near-verbatim" to allow only case /
  whitespace / punctuation tolerance.
- §2.6 canonicalize_values gets deterministic tiebreakers.
- §3.4 `inner_loop.py` warns on `k_min > k_max` clamp.
- §3.6 `iqr_multiple` change-marker added.
- §4 dead code removal in `text_normalize.py`.
- §5 added a single warning in `report_builder.py` when
  `affected_rows` cannot be validated.

**Nothing else is touched.** Big-picture refactors (prompt length,
per-arm prompt redesign, magic-number consolidation) are out of scope
for the demo and tracked in §3 / §4.

All 826 tests still pass after the patch.
