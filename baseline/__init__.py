"""GPT-5.5 typo-detection baseline.

A self-contained external baseline for the typo benchmark: label a CSV dataset
with GPT-5.5 (per column, Responses API, reasoning_effort=none) using one of two
frozen prompts -- WITH a rulebook or WITHOUT one. Decoupled from the RFT tree.

Entry point: ``python -m baseline.label_gpt55``. See ``baseline/README.md``.
"""
