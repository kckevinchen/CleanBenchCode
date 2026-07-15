# Grader user template

Template substitution happens server-side: ``{{item.x}}`` and ``{{sample.output_text}}`` are textually replaced. All ``item.*`` values are JSON-encoded strings so the resulting message is valid JSON.

---

```
{"column": {{item.column}}, "candidate_value": {{item.candidate_value}}, "gold_label": {{item.gold_label}}, "gold_correction": {{item.gold_correction}}, "model_output": {{sample.output_text}}}
```
