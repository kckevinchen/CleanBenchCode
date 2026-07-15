"""On-disk record of one RFT run (``<output_dir>/state.json``).

A superset of the legacy ``_rft_common.DemoState`` (it still reads old
state files -- unknown keys are ignored and missing keys default), extended
with the endpoint/model/deployment fields the deploy + supervise verbs need.
Folding the old ``supervise_state.json`` into this single record means one
file is the full history of a run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

STATE_FILE = "state.json"


@dataclass
class RunState:
    # File + job identity (legacy DemoState fields) -------------------------
    train_file_id: str | None = None
    valid_file_id: str | None = None
    job_id: str | None = None

    # Where/what the job ran (new) -----------------------------------------
    endpoint: str | None = None
    base_model: str | None = None
    suffix: str | None = None
    fine_tuned_model: str | None = None

    # Deployment (new) ------------------------------------------------------
    deployment_name: str | None = None

    # Supervisor bookkeeping (folds in old supervise_state.json) -----------
    attempts: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, output_dir: Path) -> "RunState":
        path = Path(output_dir) / STATE_FILE
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / STATE_FILE).write_text(json.dumps(asdict(self), indent=2))
