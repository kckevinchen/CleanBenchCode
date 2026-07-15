"""Template for a new RFT task version.

Copy this folder to ``RFT/<task>/`` (``python -m RFT.cli new <task>
--error-type <type>`` does it for you) and edit only the THREE things that
change between runs:

  1. DATA   -> ``scripts/build_splits.py`` writes ``data/{train,valid,eval}.jsonl``
  2. PROMPT -> ``_prompt.py`` (trainee developer message + output schema)
  3. GRADER -> ``_grader.py`` (defaults to the shared gold-free 5*F1 judge)

``spec.py`` wires them into the shared lifecycle; you only set identity there.
The entire Azure lifecycle (upload/submit/status/supervise/deploy/run/delete)
is inherited from ``RFT._common.engine`` -- you write no launcher.
"""
