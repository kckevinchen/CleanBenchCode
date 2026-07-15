# Grader scoring rubric v3 (set-F1; Python grader)

Per-column score, computed deterministically by the Python grader in ``grader_source.py``:

```
1. Decode item.gold_typos and item.ambiguous_values from
   their JSON-encoded strings.
2. Decode sample.output_text as JSON; take ``typos`` list
   (any parse failure -> 0.0).
3. Remove entries whose normalised value is in
   ambiguous_values from BOTH pred and gold (ambiguous-
   either-correct rule).
4. Build maps {norm(value) -> norm(correction)} for pred
   and gold. Normalisation: case-fold + collapse whitespace.
5. TP: keys in both maps with matching corrections.
   FP: pred-only keys + correction-mismatch keys.
   FN: gold-only keys + correction-mismatch keys.
   (Correction-mismatch counts as both FP and FN.)
6. Both empty -> 5.0 (column is genuinely clean, full credit).
   TP == 0 -> 0.0.
   Otherwise -> F1 * 5.0, clamped to [0.0, 5.0].
```

There is no ``pass_threshold`` / ``range`` field for a Python grader -- only ``type``, ``name``, ``source`` are accepted by the Azure RFT GA API.

The local helper ``experiments._typo_v3_task.compute_column_f1`` mirrors this logic for offline eval scripts (used by ``experiments.eval_typo_finetune_v3``).
