# Trainee user message template

Built by ``experiments._typo_task.build_user_message``. The sibling list is a sample of up to 20 distinct values from the same column (capped to keep the prompt bounded for wide columns).

---

```
Column header: <COLUMN_HEADER>
Source CSV: <SOURCE_CSV>
Other distinct values in this column (sample of 3):
  - '<SIBLING_1>'
  - '<SIBLING_2>'
  - '<SIBLING_3>'

Candidate value to judge:
  '<CANDIDATE_VALUE>'

Return the JSON object.
```
