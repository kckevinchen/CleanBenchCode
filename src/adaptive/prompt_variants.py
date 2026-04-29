"""LLM-authored per-pass variants of error-type prompts.

Used by the adaptive-mode Good-Turing loop in
``experiments.run_cleaning_detection``. Instead of (or in addition to)
shuffling row/column order - which barely perturbs the detector for
cell-local error types - each pass can ask a **generator LLM** to
REWRITE every selected error type's ``description`` / ``notes`` /
``example`` so that the downstream detection LLM sees a semantically
equivalent but linguistically different definition of the same error
class.

Why this helps the estimator
----------------------------

Good-Turing's ``n_1`` (singleton) count is meaningful only if the K
runs are genuinely independent draws from the same defect population.
Row shuffling does not produce independence for single-cell errors
(the LLM's verdict on "is this a typo?" is nearly identical
regardless of row order). Rewritten prompts inject REAL variance into
what the detector looks for, so singletons / doubletons start
reflecting actual coverage uncertainty rather than parsing noise.

Design rules
------------

* We rewrite only the content-level fields (``description``,
  ``notes``, ``example``). The shared ``system_prompt`` and
  ``user_template`` - which carry the output-schema contract -
  are held constant across passes so raw detections remain
  comparable and the species canonicalisation stays valid.
* Generation failures fall back to the seed definition for that
  ``(pass, type)`` pair, with a logged warning. An occasional pass
  running a subset of seed-prompt types is preferable to crashing.
* Variants are persisted to disk per pass so humans can review what
  the LLM came up with.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from ..core.error_taxonomy import ErrorTaxonomy, ErrorType
from ..core.llm_client import LLMClient

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n(.*?)\n?```\s*$", re.DOTALL)


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if the LLM slips them in."""
    text = text.strip()
    if text.startswith("```"):
        m = _FENCE_RE.match(text)
        if m:
            return m.group(1).strip()
    return text


class PromptVariantGenerator:
    """Rewrites (description, notes, example) for an ErrorType via the LLM.

    Stateless apart from the shared :class:`LLMClient`, the loaded
    generator template, and the model-level configuration. Safe to
    call concurrently via :meth:`generate_for_taxonomy` which fans out
    across selected types with a thread pool.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        model: str,
        template_path: Path,
        reasoning_effort: str = "low",
        reasoning_summary: str = "detailed",
        concurrency: Optional[int] = None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.concurrency = int(concurrency) if concurrency else 4
        self._system, self._user_template = self._load_template(Path(template_path))

    @staticmethod
    def _load_template(path: Path) -> Tuple[str, str]:
        if not path.is_file():
            raise FileNotFoundError(f"Variant-generator template not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: top level must be a mapping")
        base = raw.get("base")
        if not isinstance(base, Mapping):
            raise ValueError(
                f"{path}: missing `base:` block (need `system` + `user_template`)"
            )
        try:
            system = str(base["system"]).strip()
            user_template = str(base["user_template"])
        except KeyError as exc:
            raise ValueError(
                f"{path}: missing required field {exc!s} "
                "(need base.system and base.user_template)"
            ) from exc
        return system, user_template

    def _render_user_prompt(self, seed: ErrorType, pass_index: int) -> str:
        return self._user_template.format(
            type_key=seed.key,
            type_name=seed.name,
            type_category=seed.category,
            description=seed.description.strip(),
            notes=(seed.notes.strip() or "(no additional guidance)"),
            example=(seed.example.strip() or "(no reference example provided)"),
            pass_index=pass_index,
        )

    def generate_for_type(
        self, seed: ErrorType, pass_index: int
    ) -> Dict[str, Any]:
        """Produce a variant for a single type. Never raises.

        Returns a dict with::

            {
              "variant":            ErrorType (generated, or the seed on fallback),
              "ok":                 bool,
              "pass_index":         int,
              "type_key":           str,
              "seed_description":   str,
              "seed_notes":         str,
              "seed_example":       str,
              "variant_description":str,
              "variant_notes":      str,
              "variant_example":    str,
              "raw_text":           str  (LLM reply, possibly empty),
              "reasoning_summary":  str,
              "error":              Optional[str],
              "prompt":             str  (user prompt sent to the LLM),
            }
        """
        user_prompt = self._render_user_prompt(seed, pass_index)
        record: Dict[str, Any] = {
            "pass_index": pass_index,
            "type_key": seed.key,
            "seed_description": seed.description,
            "seed_notes": seed.notes,
            "seed_example": seed.example,
            "prompt": user_prompt,
            "raw_text": "",
            "reasoning_summary": "",
            "error": None,
        }

        response = self.llm.query(
            model=self.model,
            prompt=user_prompt,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            system_prompt=self._system,
        )
        record["raw_text"] = response.text or ""
        record["reasoning_summary"] = response.reasoning_summary or ""

        if response.error:
            logger.warning(
                "Variant LLM error (type=%s pass=%d): %s - falling back to seed",
                seed.key, pass_index, response.error,
            )
            record["error"] = response.error
            return _finalise_record(record, seed, seed, ok=False)

        try:
            payload = json.loads(_strip_json_fences(response.text or ""))
            if not isinstance(payload, Mapping):
                raise ValueError("top-level JSON value is not an object")
            desc = str(payload.get("description", "")).strip()
            notes = str(payload.get("notes", "")).strip()
            example = str(payload.get("example", "")).strip()
            if not desc:
                raise ValueError("empty or missing `description` field")
        except Exception as exc:
            logger.warning(
                "Variant JSON parse failed (type=%s pass=%d): %s - falling back to seed",
                seed.key, pass_index, exc,
            )
            record["error"] = f"parse: {exc}"
            return _finalise_record(record, seed, seed, ok=False)

        variant = replace(
            seed,
            description=desc,
            notes=notes or seed.notes,
            example=example or seed.example,
        )
        return _finalise_record(record, seed, variant, ok=True)

    def generate_for_taxonomy(
        self,
        base_taxonomy: ErrorTaxonomy,
        seed_types: Sequence[ErrorType],
        pass_index: int,
    ) -> Tuple[ErrorTaxonomy, List[ErrorType], List[Dict[str, Any]]]:
        """Generate variants for every seed type in parallel.

        Returns ``(variant_taxonomy, variant_types, records)``:

        * ``variant_taxonomy`` - a new :class:`ErrorTaxonomy` with the
          variant types substituted in for their seed counterparts.
          Non-selected types are carried through unchanged so
          ``taxonomy.select(...)`` keeps working elsewhere.
        * ``variant_types``    - the ordered list of ErrorType objects
          the caller should hand to the detector (i.e. the seed order,
          with rewrites applied).
        * ``records``          - per-type generation metadata suitable
          for persisting via :func:`save_variant_records`.
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(seed_types)
        if not seed_types:
            variant_taxonomy = ErrorTaxonomy(
                system_prompt=base_taxonomy.system_prompt,
                user_template=base_taxonomy.user_template,
                types=tuple(base_taxonomy.types),
                source_path=base_taxonomy.source_path,
            )
            return variant_taxonomy, [], []

        max_workers = max(1, min(self.concurrency, len(seed_types)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_to_idx = {
                pool.submit(self.generate_for_type, et, pass_index): i
                for i, et in enumerate(seed_types)
            }
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                results[i] = fut.result()

        # At this point every slot is populated (generate_for_type never
        # raises). The type-checker needs a cast to drop the Optional.
        records: List[Dict[str, Any]] = [r for r in results if r is not None]

        variant_types = [r["variant"] for r in records]
        overrides = {vt.key: vt for vt in variant_types}
        rebuilt_types = tuple(
            overrides.get(t.key, t) for t in base_taxonomy.types
        )
        variant_taxonomy = ErrorTaxonomy(
            system_prompt=base_taxonomy.system_prompt,
            user_template=base_taxonomy.user_template,
            types=rebuilt_types,
            source_path=base_taxonomy.source_path,
        )
        ok_count = sum(1 for r in records if r["ok"])
        logger.info(
            "Variant generation pass=%d: %d/%d types rewritten (rest fell back to seed)",
            pass_index, ok_count, len(records),
        )
        return variant_taxonomy, variant_types, records


def _finalise_record(
    record: Dict[str, Any],
    seed: ErrorType,
    variant: ErrorType,
    *,
    ok: bool,
) -> Dict[str, Any]:
    record["ok"] = ok
    record["variant"] = variant
    record["variant_description"] = variant.description
    record["variant_notes"] = variant.notes
    record["variant_example"] = variant.example
    return record


def save_variant_records(
    records: Sequence[Mapping[str, Any]],
    *,
    run_subdir: Path,
) -> None:
    """Persist per-type variant records under ``run_subdir/<key>.json``."""
    run_subdir = Path(run_subdir)
    run_subdir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        serialisable: Dict[str, Any] = {
            k: v for k, v in rec.items() if k != "variant"
        }
        vt: ErrorType = rec["variant"]
        serialisable["variant"] = {
            "key": vt.key,
            "name": vt.name,
            "category": vt.category,
            "description": vt.description,
            "notes": vt.notes,
            "example": vt.example,
        }
        path = run_subdir / f"{rec['type_key']}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, ensure_ascii=False, indent=2)


__all__ = [
    "PromptVariantGenerator",
    "save_variant_records",
]
