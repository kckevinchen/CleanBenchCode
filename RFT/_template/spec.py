"""Task spec -- wires data + prompt + grader into the shared lifecycle.

You normally only set the identity (name / error_type / version) and the
field-name hooks here; ``make_spec`` fills the rest with sensible defaults
(o4-mini base, the Azure score_model grader whitelist, 2 epochs).
"""

from __future__ import annotations

from pathlib import Path

from RFT._common.spec import make_spec
from RFT._template._grader import build_grader
from RFT._template._prompt import FINDINGS_KEY, build_response_format

_DIR = Path(__file__).resolve().parent

SPEC = make_spec(
    name="example_v1",
    error_type="example",
    version="v1",
    data_dir=_DIR / "data",  # holds train/valid/eval.jsonl from build_splits.py
    response_format=build_response_format,
    build_grader=build_grader,
    findings_key=FINDINGS_KEY,  # -> prediction parser for `rft run`
    default_suffix="example-v1",
    # Gold-anchored eval? Add gold_field=... (+ ambiguous_field=...) to get a
    # cell-level micro-F1 scorer for `rft run`. Omit for gold-free eval splits.
    # gold_field="gold_values",
    # ambiguous_field="ambiguous_values",
)
