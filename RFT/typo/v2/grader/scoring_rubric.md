# Grader scoring rubric v2 (binary; ambiguous either-way)

```
5: gold_label=='ambiguous' (either binary label accepted);
   OR gold_label=='clean'  and model.label=='clean';
   OR gold_label=='typo'   and model.label=='typo' AND
      model.correction equals gold_correction up to case and
      collapsed whitespace.
4: gold_label=='typo' and model.label=='typo' but the
   correction is missing, partially wrong, or off by one
   character.
2: gold_label and model.label are opposites
   ('typo' vs 'clean').
1: JSON parses but required fields are missing, malformed,
   OR model.label is 'ambiguous' (schema violation).
0: output is empty or not valid JSON.
```

`pass_threshold` = **4**, `range` = **[0.0, 5.0]**.
