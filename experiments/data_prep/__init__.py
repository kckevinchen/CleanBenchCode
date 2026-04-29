"""CLI wrappers for the data-prep utilities living in ``src/data_prep/``.

``src/`` is library code; every user-facing entry point is surfaced
through ``experiments/`` so users can invoke them uniformly as
``python -m experiments.<name>``. The modules in this package are
deliberately thin - they just delegate to the corresponding ``main()``
function in ``src.data_prep``.
"""
