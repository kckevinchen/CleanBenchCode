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
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
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
    "ClusterPromptVariantGenerator",
    "ClusterPromptVariant",
    "save_variant_records",
    "save_cluster_variant_records",
]


# =========================================================================
# Column-clustering prompt-variant generator
# =========================================================================
#
# The detection-side ``PromptVariantGenerator`` above is tightly bound
# to ``ErrorType`` (it rewrites the per-error-type description / notes
# / example). The column-clustering proposer prompt
# (``prompts/cluster_column_values.yaml``) has a different shape - one
# flat ``{system, user_template}`` pair with three placeholders
# (``{source}``, ``{column}``, ``{value_list}``) - so it needs its own
# minimal rewriter. The two generators share the LLM client and the
# JSON-fence stripper helper but nothing else.


@dataclass(frozen=True)
class ClusterPromptVariant:
    """One paraphrased version of the column-clustering proposer prompt.

    ``ok=False`` means the LLM call or the JSON parse failed for this
    slot and the caller should fall back to the seed prompt. The
    ``system`` / ``user_template`` fields are then a copy of the seed,
    so callers can always use the variant unconditionally if they
    don't need to distinguish ok vs fallback.
    """

    pass_index: int
    system: str
    user_template: str
    ok: bool = True
    raw_text: str = ""
    reasoning_summary: str = ""
    error: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "pass_index": self.pass_index,
            "ok": self.ok,
            "system": self.system,
            "user_template": self.user_template,
            "raw_text": self.raw_text,
            "reasoning_summary": self.reasoning_summary,
            "error": self.error,
        }


# Placeholders that MUST survive the rewrite. If any is dropped or
# duplicated by the LLM the variant is rejected and the caller falls
# back to the seed -- a malformed user_template would crash
# ``str.format(source=..., column=..., value_list=...)`` at proposer
# call time.
_REQUIRED_CLUSTER_PLACEHOLDERS = ("{source}", "{column}", "{value_list}")


def _placeholders_preserved(text: str) -> bool:
    """Each required placeholder must appear exactly once in ``text``."""
    return all(text.count(p) == 1 for p in _REQUIRED_CLUSTER_PLACEHOLDERS)


def _template_format_failure(text: str) -> Optional[str]:
    """Return a short diagnostic if ``text`` can NOT be safely
    ``.format()``ed at call time, else ``None``.

    The proposer invokes
    ``user_template.format(source=..., column=..., value_list=...)``;
    any of:
      * an unescaped literal ``{`` / ``}`` (e.g. a JSON example block
        such as ``{"clusters": ...}`` that should be doubled to
        ``{{...}}``);
      * a stray named placeholder like ``{foo}`` beyond the three
        required ones;
      * a malformed format-spec inside one of the placeholders;
    will raise at proposer call time and silently skip the column.
    Run a dry ``.format`` with empty probe values to catch this at
    variant-build time and route the offender to the seed fallback.

    The string returned is suitable for embedding in the audit-trail
    ``error`` field so operators can tell the two failure classes
    apart without re-running.
    """
    try:
        text.format(source="", column="", value_list="")
    except KeyError as exc:
        # KeyError fires for BOTH unescaped ``{...}`` literals
        # (Python reads ``{\n  "clusters"`` as a placeholder name)
        # AND stray named placeholders like ``{foo}``. The argument
        # to KeyError is the offending name -- echo it so the
        # operator can tell which.
        return (
            f"unknown placeholder or unescaped brace in user_template "
            f"(missing key: {exc!r}); JSON example blocks must use "
            "``{{`` / ``}}``"
        )
    except IndexError:
        return "positional placeholder like `{0}` in user_template"
    except ValueError as exc:
        return f"malformed format spec in user_template: {exc}"
    return None


def _template_format_safe(text: str) -> bool:
    """Thin boolean wrapper around :func:`_template_format_failure`.

    Kept for the existing unit tests in
    ``tests/test_cluster_prompt_variants.py`` (the test cases are
    parametrised against this name).
    """
    return _template_format_failure(text) is None


@dataclass
class ClusterPromptVariantGenerator:
    """Paraphrases the column-clustering proposer prompt via the LLM.

    Stateless apart from the loaded generator template and the shared
    :class:`LLMClient`. Safe to call ``generate`` concurrently via
    ``ThreadPoolExecutor`` -- the constructor wraps the same pattern
    around ``generate_one``.
    """

    llm: LLMClient
    model: str
    template_path: Path
    reasoning_effort: str = "low"
    reasoning_summary: str = "detailed"
    concurrency: int = 4
    _system: str = field(init=False, default="")
    _user_template: str = field(init=False, default="")

    def __post_init__(self) -> None:
        path = Path(self.template_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Cluster prompt variant template not found: {path}"
            )
        with path.open("r", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: top level must be a mapping")
        base = raw.get("base")
        if not isinstance(base, Mapping):
            raise ValueError(
                f"{path}: missing ``base:`` block (need ``system`` and "
                "``user_template``)"
            )
        try:
            self._system = str(base["system"]).strip()
            self._user_template = str(base["user_template"])
        except KeyError as exc:
            raise ValueError(
                f"{path}: missing required field {exc!s} "
                "(need base.system and base.user_template)"
            ) from exc

    # ------------------------------------------------------------ API

    def generate_one(
        self,
        *,
        base_system: str,
        base_user_template: str,
        pass_index: int,
    ) -> ClusterPromptVariant:
        """Generate one paraphrased prompt; returns the seed on failure."""
        user_prompt = self._user_template.format(
            base_system=base_system,
            base_user_template=base_user_template,
            pass_index=pass_index,
        )
        try:
            response = self.llm.query(
                model=self.model,
                prompt=user_prompt,
                reasoning_effort=self.reasoning_effort,
                reasoning_summary=self.reasoning_summary,
                system_prompt=self._system,
            )
        except Exception as exc:  # noqa: BLE001 - report and fall back
            logger.warning(
                "ClusterPromptVariant LLM call failed at pass=%d: %s "
                "- falling back to seed", pass_index, exc,
            )
            return ClusterPromptVariant(
                pass_index=pass_index,
                system=base_system, user_template=base_user_template,
                ok=False, error=f"llm: {exc}",
            )

        raw_text = getattr(response, "text", "") or ""
        reasoning = getattr(response, "reasoning_summary", "") or ""
        err = getattr(response, "error", None)
        if err:
            logger.warning(
                "ClusterPromptVariant LLM error at pass=%d: %s "
                "- falling back to seed", pass_index, err,
            )
            return ClusterPromptVariant(
                pass_index=pass_index,
                system=base_system, user_template=base_user_template,
                ok=False, raw_text=raw_text,
                reasoning_summary=reasoning, error=err,
            )

        try:
            payload = json.loads(_strip_json_fences(raw_text))
            if not isinstance(payload, Mapping):
                raise ValueError("top-level JSON value is not an object")
            new_system = str(payload.get("system", "")).strip()
            new_user_template = str(payload.get("user_template", ""))
            if not new_system:
                raise ValueError("empty or missing `system` field")
            if not new_user_template:
                raise ValueError("empty or missing `user_template` field")
            if not _placeholders_preserved(new_user_template):
                raise ValueError(
                    "rewritten user_template missing or duplicating one of "
                    "{source}, {column}, {value_list}"
                )
            _fmt_err = _template_format_failure(new_user_template)
            if _fmt_err is not None:
                raise ValueError(_fmt_err)
        except Exception as exc:
            logger.warning(
                "ClusterPromptVariant JSON parse failed at pass=%d: %s "
                "- falling back to seed", pass_index, exc,
            )
            return ClusterPromptVariant(
                pass_index=pass_index,
                system=base_system, user_template=base_user_template,
                ok=False, raw_text=raw_text,
                reasoning_summary=reasoning, error=f"parse: {exc}",
            )

        return ClusterPromptVariant(
            pass_index=pass_index,
            system=new_system, user_template=new_user_template,
            ok=True, raw_text=raw_text, reasoning_summary=reasoning,
        )

    def generate(
        self,
        *,
        base_system: str,
        base_user_template: str,
        n_variants: int,
    ) -> List[ClusterPromptVariant]:
        """Generate ``n_variants`` paraphrases in parallel.

        Pass indices are 1..n_variants (the 0-th index is reserved
        for the identity / seed prompt and is the caller's
        responsibility -- this function never returns the identity).
        """
        if n_variants <= 0:
            return []
        results: List[Optional[ClusterPromptVariant]] = [None] * n_variants
        max_workers = max(1, min(self.concurrency, n_variants))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self.generate_one,
                    base_system=base_system,
                    base_user_template=base_user_template,
                    pass_index=i + 1,
                ): i
                for i in range(n_variants)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
        return [r for r in results if r is not None]


def save_cluster_variant_records(
    variants: Sequence[ClusterPromptVariant],
    *,
    run_dir: Path,
) -> Path:
    """Persist a JSONL trail at ``<run_dir>/cluster_prompt_variants.jsonl``.

    Returns the path written. Each line is one ``ClusterPromptVariant``
    record (``to_record()``). Existing files are overwritten so a
    re-run replaces the audit trail cleanly.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "cluster_prompt_variants.jsonl"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for v in variants:
            fh.write(json.dumps(v.to_record(), ensure_ascii=False) + "\n")
    # ``os.replace`` is the portable atomic-rename primitive; matches
    # the pattern used by ``experiments.run_column_clustering`` and
    # ``experiments.score_column_clusters`` so all column-cluster
    # outputs land via the same call.
    os.replace(tmp, path)
    return path
