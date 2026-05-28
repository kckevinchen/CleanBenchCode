"""Column value-clustering subsystem.

Groups the distinct surface forms in a single column into equivalence
classes ("US" / "USA" / "United States" -> one canonical value). Used
by the column-clustering workflow (``experiments.run_column_clustering``)
that feeds the inconsistency-labeling UI.

* :mod:`.proposer` - :class:`~.proposer.ValueClusterProposer`: one LLM
                     call that proposes clusters over a column's value
                     vocabulary, plus JSON parsing helpers.
* :mod:`.adaptive` - :class:`~.adaptive.ColumnClusteringAdaptive`: drives
                     the proposer through the Good-Turing / Chao1 adaptive
                     stop rule (multi-pass, recall-first union).

Import submodules directly, e.g.
``from src.core.value_cluster.proposer import ValueClusterProposer``.
"""
