"""Registry of RFT task specs, grouped by error type.

Adding a new version is one import + one entry here. ``RFT.cli`` drives any
registered spec through the full Azure lifecycle.
"""

from __future__ import annotations

from collections import defaultdict

from RFT._common.spec import RftTaskSpec
from RFT.inconsistency.v1.spec import SPEC as INCONSISTENCY_V1
from RFT.typo.v9.spec import SPEC as TYPO_V9
from RFT.typo.v10.spec import SPEC as TYPO_V10
from RFT.typo.v11.spec import SPEC as TYPO_V11

_ALL: tuple[RftTaskSpec, ...] = (TYPO_V9, TYPO_V10, TYPO_V11, INCONSISTENCY_V1)

REGISTRY: dict[str, RftTaskSpec] = {s.name: s for s in _ALL}


def get(name: str) -> RftTaskSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"Unknown task {name!r}. Known tasks: {', '.join(sorted(REGISTRY))}"
        ) from None


def by_error_type() -> dict[str, list[RftTaskSpec]]:
    grouped: dict[str, list[RftTaskSpec]] = defaultdict(list)
    for spec in REGISTRY.values():
        grouped[spec.error_type].append(spec)
    for specs in grouped.values():
        specs.sort(key=lambda s: s.version)
    return dict(grouped)


def format_task_list() -> str:
    lines = ["Registered RFT tasks (grouped by error type):"]
    for error_type, specs in sorted(by_error_type().items()):
        lines.append(f"  {error_type}:")
        for s in specs:
            variants = ", ".join(s.reward_variants)
            lines.append(f"    {s.name:<20} base={s.base_model}  variants=[{variants}]")
    return "\n".join(lines)
