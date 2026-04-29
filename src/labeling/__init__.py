"""Labelling web-app MVP for turning detector output into a benchmark.

This package is imported by :mod:`experiments.label_server` - the CLI
entry point users run to spin up the annotation UI. It deliberately
contains **no** knowledge of the detection pipeline: it consumes the
JSON artifacts that ``src.core.report_builder`` emits (``summary.json``,
``clusters.json``, ``issues.json``) plus the source CSVs referenced by
``summary.config.csv_paths``.

Append-only persistence (``labels.jsonl``) lives next to those
artifacts, inside the run directory. Nothing is written anywhere else.
"""

from .schema import Label, LabelStatus
from .store import LabelStore

__all__ = ["Label", "LabelStatus", "LabelStore"]
