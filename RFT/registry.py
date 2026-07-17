"""Registry of current RFT task specs, grouped by error type."""

from __future__ import annotations

from collections import defaultdict

from RFT._common.spec import RftTaskSpec
from RFT.inconsistency.spec import SPEC as INCONSISTENCY
from RFT.typo.spec import SPEC as TYPO

_ALL: tuple[RftTaskSpec, ...] = (TYPO, INCONSISTENCY)

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
    return dict(grouped)


def format_task_list() -> str:
    lines = ["Registered RFT tasks (grouped by error type):"]
    for error_type, specs in sorted(by_error_type().items()):
        lines.append(f"  {error_type}:")
        for s in specs:
            variants = ", ".join(s.reward_variants)
            lines.append(f"    {s.name:<20} base={s.base_model}  variants=[{variants}]")
    return "\n".join(lines)
