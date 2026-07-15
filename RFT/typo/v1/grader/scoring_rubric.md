# Grader scoring rubric (verbatim from the developer message)

```
5: model.label == gold_label AND, when gold_label=='typo',
   model.correction equals gold_correction up to case and
   collapsed whitespace.
4: model.label == gold_label but, when gold_label=='typo',
   the correction is missing, partially wrong, or off by one
   character.
3: model.label is 'ambiguous' when gold is 'typo' or 'clean'
   (or vice versa) AND the rationale is consistent with the
   rulebook.
2: model.label is the opposite of gold_label
   ('typo' vs 'clean').
1: JSON parses but required fields are missing or malformed.
0: output is empty or not valid JSON.
```

`pass_threshold` = **4**, `range` = **[0.0, 5.0]**.
