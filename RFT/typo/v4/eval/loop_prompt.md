# Grader-prompt iteration loop (headless)

You are continuing an iteration loop that tunes the score-model developer prompt at `RFT/typo_v4/grader/grader.json` (path: `graders.typo_v4_score_model_extras.input[0].content`). Goal: minimise `clean@1.0` and `ambiguous@1.0` and `error_trivial@1.0` while keeping `error_nontrivial@1.0` at or above 90 on the cell-level test.

## State on entry

The best run so far is **v6**: clean=153, ambiguous=117, error_nontrivial=93, error_trivial=7, prompt length ~6.4k chars. v7 regressed (error_nontrivial dropped to 87 because an aggressive `correction.startswith(value)` rule fired on real typos). Treat v6 as the baseline to beat.

If `RFT/typo_v4/grader/grader.json` is currently v7 (verify by looking for `"## Test 1 -- Prefix-expansion"`), restore v6 first: it is backed up at `RFT/typo_v4/grader/grader.json.v6_bak` if present, otherwise copy from `outputs/grader_sm_cell_level_gpt4o_v6/grader.json` if you preserved it, otherwise reconstruct from your transcript. The first thing to do is `cp RFT/typo_v4/grader/grader.json RFT/typo_v4/grader/grader.json.<current-label>_bak` before any edit.

## Procedure per iteration (label as v8, v9, ...)

1. **Examine errors** from the most-recent run's `per_cell.jsonl`:
   - clean cells scored 1.0 (false positives)
   - ambiguous cells scored 1.0 (false positives)
   - error_nontrivial cells scored 0.0 (gold recall misses)
   - Group by shape: CASING_ONLY (`v.lower()==c.lower()`), APOS_ONLY, HYPH_SPACE, EXPANSION (correction has value as prefix or close to it), SPELLING_SMALL, OTHER.
2. **Hypothesise ONE focused change** that targets the largest category that has room to move. Avoid:
   - Long worked-example dumps (gpt-4o memorises rather than generalises).
   - Aggressive blanket rules like "correction.startswith(value) -> fp" — v7 showed these break gold recall.
   - Adding length without clear targeting. Cap the developer prompt at 7,000 chars.
3. **Edit the prompt** by reading the current JSON, replacing `input[0].content`, and writing back. Use Python with a heredoc; do not paste the prompt into the tool's `str_replace` because it's >1k tokens.
4. **Run the test:**
   ```bash
   source venv/bin/activate
   python RFT/typo_v4/eval/test_grader_sm_cell_level.py \
       --out-dir outputs/grader_sm_cell_level_gpt4o_v<N> \
       --concurrency 20 \
       2>&1 | tee logs/grader_sm_cell_level_gpt4o_v<N>.log
   ```
   Wall time ~3 min.
5. **Compare** the new summary against v6:
   - If `error_nontrivial.score_1.0 < 90`, the change broke gold recall — REVERT (restore the prior grader.json) and try a different hypothesis next iteration.
   - Otherwise log the deltas in `logs/loop_progress.md` (append a row to a Markdown table) and continue.
6. **Stop conditions** (any one terminates the loop):
   - 3 consecutive iterations with no improvement in (clean@1.0 + ambiguous@1.0 + error_trivial@1.0) and no regression in error_nontrivial.
   - Developer prompt length about to exceed 7,000 chars and no further gains in last 2 iterations.
   - Hard cap: 8 iterations total (v8 through v15).

## Hard constraints

- Do NOT modify the Python grader (`typo_v4_python_gold`) -- only `typo_v4_score_model_extras.input[0].content`.
- Do NOT touch the benchmark file or the corrections cache.
- Do NOT change the test script's CLI; pass options only as documented above.
- Keep the prompt <= 7000 chars.

## Deliverables

- A best-performing `RFT/typo_v4/grader/grader.json` (whichever iteration won).
- `logs/loop_progress.md` showing every iteration's hypothesis + result + delta.
- A final summary message saying which version is best and on what metric.

Begin by reading the latest `per_cell.jsonl` (the highest-numbered `outputs/grader_sm_cell_level_gpt4o_v*` directory) and proposing your v8 hypothesis.
