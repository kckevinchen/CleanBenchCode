"""Row-duplication (intra-table, multi-row) detection subsystem.

The ``row_duplication`` error type asks whether two or more rows in a
single table describe the same entity. The logic is large and has
several pluggable candidate generators, so it lives in its own package
rather than alongside the single-table cell detectors in ``src.core``.

* :mod:`.priors`      - deterministic multi-row priors: identity-key
                        collision groups, value normalisation, and the
                        prompt blocks that summarise them. Also consumed
                        by the relation-discovery pipeline
                        (``src.core.relations``) for ``group``-scoped
                        types.
* :mod:`.clusterers`  - recall-first candidate generators
                        (``KeyCollisionClusterer``, ``BlockingClusterer``,
                        ``MinHashLSHClusterer``, ``EmbeddingClusterer``).
* :mod:`.detector`    - :class:`~.detector.RowDuplicationDetector`:
                        runs the clusterers, optionally LLM-adjudicates
                        each collision group, and emits findings in the
                        same shape as the per-chunk cell detector.

Import submodules directly, e.g.
``from src.core.row_dup.detector import RowDuplicationDetector``.
"""
