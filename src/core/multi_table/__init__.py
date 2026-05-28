"""Multi-table (cross-source) inconsistency detection subsystem.

The ``inter_table_inconsistency`` error type needs to reason across
several CSVs at once (a workbook). That logic is large enough to live
in its own package rather than alongside the single-table detectors in
``src.core``. Stage order mirrors the relation pipeline:

* :mod:`.priors`            - cheap deterministic candidate signals
                              (unionable groups, join keys) per workbook.
* :mod:`.resolver`          - resolve a workbook into aligned column
                              mappings + join specs (LLM-assisted).
* :mod:`.semantic_detector` - scan resolved groups for cross-table
                              semantic inconsistencies.
* :mod:`.verifier`          - code/LLM verification of candidate
                              violations.
* :mod:`.violations`        - the ``MultiTableViolation`` record + helpers.
* :mod:`.pipeline`          - orchestration entry points
                              (``group_workbooks`` -> ``resolve_groups_pass``
                              -> ``detect_violations_pass``).

Import submodules directly, e.g.
``from src.core.multi_table.pipeline import resolve_groups_pass``.
"""
