# New RFT run — 3-step checklist

This folder is a working skeleton. A new run changes only **data, prompt, and
grader**; the whole Azure lifecycle is inherited from `RFT/_common/`.

```bash
# 0. Scaffold a fresh copy (renames identity for you) -> RFT/mytask/v1/
python -m RFT.cli new mytask v1 --findings-key findings
```

Then edit, in the new `RFT/mytask/v1/` folder:

1. **PROMPT** — `_prompt.py`: replace `TRAINEE_PROMPT` with your rulebook and
   adjust `build_response_format()` / `KINDS` if your finding shape differs.
2. **GRADER** — `_grader.py`: set `TASK_DESCRIPTION`. The default shared
   gold-free `5*F1` judge usually needs nothing else. For typo-style reward
   variants / python sub-graders, see `RFT/_common/typo_task.py` (shared
   primitives) and `RFT/typo/v9/_grader_variants.py` (multigrader example).
3. **DATA** — `scripts/build_splits.py`: point `--sources` at your CSVs (and
   add gold labels if you have them); it writes `data/{train,valid,eval}.jsonl`.

Finally register the spec (one line in `RFT/registry.py`):

```python
from RFT.mytask.v1.spec import SPEC as MYTASK_V1
_ALL = (..., MYTASK_V1)
```

Run it (the spec's registry name is `<error_type>_<version>`):

```bash
python -m RFT.cli --task mytask_v1 all --reward-variant default \
   --endpoint <azure-openai-endpoint> --suffix mytask-v1
python -m RFT.cli --task mytask_v1 deploy --deployment-name mytask-v1 --resource-group <rg>
python -m RFT.cli --task mytask_v1 run --deployment mytask-v1
```
