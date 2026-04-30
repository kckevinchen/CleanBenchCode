"""Error-type taxonomy loader and per-type prompt renderer.

This module is the counterpart to ``prompts/error_types.yaml``. It exposes:

* :class:`ErrorType` - metadata for a single error class
* :class:`ErrorTaxonomy` - full catalogue (shared base prompt + per-type entries)
* :func:`load_taxonomy` - parse the YAML file into an :class:`ErrorTaxonomy`

The exhaustive detection pipeline issues one LLM call per
``(csv_chunk, error_type)`` pair. Keeping the per-type prompt data here (as
opposed to hard-coding it inside the detector) means the taxonomy can be
edited, reviewed, and versioned purely through YAML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from .data_loader import CsvChunk
from .table_digest import TableDigest

logger = logging.getLogger(__name__)


# Placeholder rendered into the user prompt when the caller did not
# supply a :class:`TableDigest` (e.g. legacy single-column runs that
# don't need cross-row context). Kept short so it does not balloon
# token counts in the common case.
_NO_DIGEST_PLACEHOLDER = "(table overview unavailable for this run)"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorType:
    """One entry in the error-type catalogue."""

    key: str
    name: str
    category: str
    description: str
    example: str = ""
    notes: str = ""

    def summary_line(self) -> str:
        return f"[{self.category}] {self.name}"


@dataclass(frozen=True)
class ErrorTaxonomy:
    """The full taxonomy: shared base prompt + ordered list of error types."""

    system_prompt: str
    user_template: str
    types: Sequence[ErrorType]
    source_path: Optional[Path] = None
    _index: Dict[str, ErrorType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_index", {et.key: et for et in self.types}
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def keys(self) -> List[str]:
        return [et.key for et in self.types]

    def get(self, key: str) -> ErrorType:
        if key not in self._index:
            raise KeyError(
                f"Unknown error_type '{key}'. Known: {sorted(self._index)}"
            )
        return self._index[key]

    def select(self, keys: Optional[Iterable[str]]) -> List[ErrorType]:
        """Return the types matching ``keys`` in taxonomy order.

        ``None`` (or an empty iterable) selects the full taxonomy.
        """
        if not keys:
            return list(self.types)
        wanted = list(keys)
        missing = [k for k in wanted if k not in self._index]
        if missing:
            raise KeyError(
                f"Unknown error_type keys: {missing}. Known: {sorted(self._index)}"
            )
        wanted_set = set(wanted)
        return [et for et in self.types if et.key in wanted_set]

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def render_user_prompt(
        self,
        error_type: ErrorType,
        chunk: CsvChunk,
        *,
        digest: Optional[TableDigest] = None,
    ) -> str:
        """Render the per-type user prompt for a specific CSV chunk.

        ``digest`` is the optional whole-table summary built by
        :func:`src.core.table_digest.build_table_digest`. When the
        ``base.user_template`` carries a ``{table_overview}``
        placeholder, the digest is rendered into that slot; the
        chunk row table itself is unchanged. Templates that do not
        reference ``{table_overview}`` ignore the argument
        (``str.format`` silently drops unused kwargs), keeping the
        single-column path backward-compatible.
        """
        try:
            return self.user_template.format(
                type_key=error_type.key,
                type_name=error_type.name,
                type_category=error_type.category,
                type_description=error_type.description.strip(),
                type_example=error_type.example.strip() or "(no reference example provided)",
                type_notes=error_type.notes.strip() or "(no additional guidance)",
                headers=", ".join(chunk.headers),
                rows=chunk.to_markdown(),
                start_row=chunk.start_row,
                end_row=chunk.end_row,
                table_overview=(
                    digest.to_prompt_block() if digest is not None
                    else _NO_DIGEST_PLACEHOLDER
                ),
            )
        except KeyError as exc:
            raise KeyError(
                f"error_types.yaml base.user_template references undefined "
                f"placeholder: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"error_types.yaml: missing required field '{key}' in {where}")
    return mapping[key]


def load_taxonomy(path: Path) -> ErrorTaxonomy:
    """Parse ``prompts/error_types.yaml`` (or an override) into a taxonomy."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Error-type taxonomy not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top level must be a mapping")

    base = _require(raw, "base", "<root>")
    if not isinstance(base, Mapping):
        raise ValueError(f"{path}: 'base' must be a mapping")
    system_prompt = str(_require(base, "system", "base")).strip()
    user_template = str(_require(base, "user_template", "base"))

    # Optional ``defaults:`` block - a single source of truth for the
    # numeric thresholds referenced inside the GLOBAL RULES of
    # base.user_template (and base.system). Substituted via
    # ``string.Template`` (``${name}`` syntax) so it cannot collide
    # with the ``str.format`` placeholders used at runtime
    # (``{type_name}`` etc.).
    defaults_raw = raw.get("defaults")
    if defaults_raw is not None:
        if not isinstance(defaults_raw, Mapping):
            raise ValueError(f"{path}: 'defaults' must be a mapping if present")
        defaults = {str(k): str(v) for k, v in defaults_raw.items()}
        try:
            system_prompt = Template(system_prompt).substitute(defaults)
            user_template = Template(user_template).substitute(defaults)
        except KeyError as exc:
            raise ValueError(
                f"{path}: ${{{exc.args[0]}}} referenced in base prompts but "
                f"missing from 'defaults' block"
            ) from exc

    types_raw = _require(raw, "types", "<root>")
    if not isinstance(types_raw, Mapping):
        raise ValueError(f"{path}: 'types' must be a mapping of key -> {{...}}")

    types: List[ErrorType] = []
    for key, body in types_raw.items():
        if not isinstance(body, Mapping):
            raise ValueError(f"{path}: types.{key} must be a mapping")
        types.append(
            ErrorType(
                key=str(key),
                name=str(_require(body, "name", f"types.{key}")),
                category=str(_require(body, "category", f"types.{key}")),
                description=str(_require(body, "description", f"types.{key}")),
                example=str(body.get("example", "") or ""),
                notes=str(body.get("notes", "") or ""),
            )
        )

    if not types:
        raise ValueError(f"{path}: 'types' must contain at least one entry")

    logger.info(
        "Loaded error-type taxonomy from %s (%d types)", path, len(types)
    )
    return ErrorTaxonomy(
        system_prompt=system_prompt,
        user_template=user_template,
        types=tuple(types),
        source_path=path,
    )
