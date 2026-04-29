"""CLI entry point: ``python -m experiments.data_prep.subsample``.

Thin wrapper around :func:`src.data_prep.subsample.main`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.subsample import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
