"""Single-LLM-call standardization-cluster proposer.

Given the distinct values of one CSV column, asks the LLM to identify
multi-value standardization groups (e.g. ``{"ABC HS", "BBB"}`` -> same
school) and returns them as a list of :class:`ProposedCluster` records.

This is the inner half of the column-clustering pipeline; the
multi-sample adaptive loop in
:mod:`src.core.value_cluster.adaptive` calls this once per perturbation
and then aggregates the recall-first union of clusters across runs.

Failure modes (returns an empty list rather than raising):
  * LLM call returned an error or empty text.
  * Response is not valid JSON or doesn't match the expected schema.
  * Every proposed cluster fails post-parse validation (single-value,
    contains values not in the input, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import yaml

from src.core.llm_client import LLMClient
from src.labeling.schema import ClusterKind

logger = logging.getLogger(__name__)


# Normalisation of LLM-emitted kind tags happens via
# ``ClusterKind.from_string`` (legacy-aware: maps known aliases to
# the canonical enum value and returns UNTAGGED for anything else).


@dataclass(frozen=True)
class ProposedCluster:
    """One standardization cluster proposed by the LLM in a single call.

    ``values`` is a sorted tuple of the cluster's distinct values
    (sorted so two clusters with the same value set always hash the
    same; the adaptive driver uses this as a species key).

    ``target_style`` is a free-text hint emitted by the LLM that
    describes the column's COMPETING MAJORITY style for type-2
    (shared-minority-style) groups -- i.e. what the group's values
    should be rewritten toward. It is ``None`` for type-1 alias
    groups, where ``canonical`` already names the target form.
    """

    values: Tuple[str, ...]
    kind: ClusterKind = ClusterKind.UNTAGGED
    canonical: Optional[str] = None
    target_style: Optional[str] = None

    def as_species_key(self) -> Tuple[str, ...]:
        return self.values


@dataclass
class ValueClusterProposer:
    """LLM-driven proposer for column standardization clusters.

    Stateless across calls; safe to re-use a single instance per
    proposer/adaptive driver.

    ``variants`` (optional) is a list of paraphrased prompts, each
    a mapping with ``system`` and ``user_template`` keys -- e.g.
    produced by
    :class:`src.adaptive.prompt_variants.ClusterPromptVariantGenerator`.
    When set, callers pass ``variant_idx`` to :meth:`propose_once` to
    pick a variant. ``variant_idx=0`` always uses the seed prompt
    loaded from ``prompt_path``; ``variant_idx >= 1`` indexes into
    ``variants`` (mod len, so callers don't need to clamp).
    """

    llm: LLMClient
    model: str
    prompt_path: Path
    reasoning_effort: str = "medium"
    reasoning_summary: str = "detailed"
    variants: Optional[Sequence[Mapping[str, str]]] = None
    _prompt: Mapping[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._prompt = self._load_prompt(self.prompt_path)

    @staticmethod
    def _load_prompt(path: Path) -> Mapping[str, str]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"cluster_column_values prompt not found: {path}"
            )
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: top level must be a mapping")
        system = str(raw.get("system", "")).strip()
        template = str(raw.get("user_template", ""))
        if not system or not template:
            raise ValueError(f"{path}: 'system' and 'user_template' required")
        return {"system": system, "user_template": template}

    def _select_prompt(self, variant_idx: int) -> Mapping[str, str]:
        """Pick the prompt slot for this call.

        ``variant_idx=0`` always returns the seed (loaded from
        ``prompt_path``). ``variant_idx >= 1`` indexes into
        ``self.variants`` modulo its length so callers can hand off
        ``run_idx`` directly without clamping. When ``variants`` is
        empty or None, every index falls back to the seed.
        """
        if not self.variants or variant_idx <= 0:
            return self._prompt
        v = self.variants[(variant_idx - 1) % len(self.variants)]
        sys_prompt = str(v.get("system", "")).strip()
        user_template = str(v.get("user_template", ""))
        if not sys_prompt or not user_template:
            # Malformed variant entry -- defensive fallback to seed
            # so the proposer never crashes mid-loop.
            return self._prompt
        return {"system": sys_prompt, "user_template": user_template}

    # ------------------------------------------------------------------ API

    def propose_once(
        self,
        *,
        source: str,
        column: str,
        value_counts: Sequence[Tuple[str, int]],
        value_order: Optional[Sequence[str]] = None,
        variant_idx: int = 0,
    ) -> List[ProposedCluster]:
        """Make a single LLM call and parse the returned clusters.

        ``value_counts`` is ``[(value, count), ...]`` for the column's
        distinct non-empty values (count > 0). ``value_order``, when
        given, controls the order in which values appear in the
        prompt; the adaptive driver shuffles this between runs to
        perturb the model. When ``value_order`` is ``None``, values
        are presented in descending-count order.

        ``variant_idx`` picks a paraphrased prompt slot from
        ``self.variants`` (0 = seed). The adaptive driver passes the
        run index here so prompt-ensemble perturbation kicks in on
        run 1, 2, ... while run 0 always uses the seed.
        """
        distinct = {v for v, _ in value_counts}
        if len(distinct) < 2:
            return []
        if value_order is None:
            order = [v for v, _ in sorted(
                value_counts, key=lambda kv: (-kv[1], kv[0])
            )]
        else:
            order = [v for v in value_order if v in distinct]
        counts_map = dict(value_counts)
        value_list = "\n".join(
            f'- "{v}" (count={counts_map.get(v, 0)})' for v in order
        )

        active_prompt = self._select_prompt(variant_idx)
        try:
            prompt = active_prompt["user_template"].format(
                source=source, column=column, value_list=value_list,
            )
        except KeyError as exc:
            logger.warning(
                "value_cluster_proposer: prompt missing placeholder %s; "
                "skipping %s.%s (variant_idx=%d)",
                exc, source, column, variant_idx,
            )
            return []

        try:
            response = self.llm.query(
                self.model,
                prompt,
                reasoning_effort=self.reasoning_effort,
                reasoning_summary=self.reasoning_summary,
                system_prompt=active_prompt["system"],
            )
        except Exception as exc:  # noqa: BLE001 - report-and-skip
            logger.warning(
                "value_cluster_proposer: LLM call failed for %s.%s: %s",
                source, column, exc,
            )
            return []

        if not getattr(response, "ok", True):
            logger.warning(
                "value_cluster_proposer: LLM error for %s.%s: %s",
                source, column, getattr(response, "error", "<no error>"),
            )
            return []

        clusters, parse_err = parse_proposed_clusters(
            response.text, distinct=distinct,
        )
        if parse_err is not None:
            logger.warning(
                "value_cluster_proposer: parse failed for %s.%s: %s",
                source, column, parse_err,
            )
            return []
        return clusters


# ---------------------------------------------------------------------------
# Response parsing (separable, testable without an LLM)
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    s = (text or "").strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


def parse_proposed_clusters(
    text: str,
    *,
    distinct: set,
) -> Tuple[List[ProposedCluster], Optional[str]]:
    """Parse an LLM response into a list of :class:`ProposedCluster`.

    Returns ``(clusters, err)``. ``err`` is ``None`` on success and a
    human-readable string when the top-level shape is wrong. The
    caller is expected to log + skip. Per-cluster malformed entries
    are SILENTLY DROPPED (with a debug log line); only top-level
    shape errors propagate to ``err``.
    """
    stripped = _strip_markdown_fences(text or "")
    if not stripped:
        return [], "empty response"
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return [], f"JSON decode error: {exc}"
    if not isinstance(obj, Mapping):
        return [], "top-level JSON is not an object"
    raw = obj.get("clusters")
    if not isinstance(raw, list):
        return [], "missing 'clusters' list"

    out: List[ProposedCluster] = []
    seen_values: set = set()
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        values_raw = entry.get("values") or []
        if not isinstance(values_raw, list):
            continue
        values: List[str] = []
        for v in values_raw:
            if not isinstance(v, str):
                continue
            if v not in distinct:
                # LLM hallucinated a value not in the input. Drop it
                # rather than the whole cluster, but log at debug.
                logger.debug(
                    "value_cluster_proposer: dropping hallucinated value %r",
                    v,
                )
                continue
            if v in seen_values:
                # Already claimed by an earlier cluster in this same
                # response; drop the dup so the per-record disjoint
                # invariant holds before we even reach the schema
                # validator.
                continue
            values.append(v)
        # Need >=2 distinct values after the cleanup.
        if len(values) < 2:
            continue
        # Mark these values as claimed for this response's scope.
        for v in values:
            seen_values.add(v)
        # Kind normalisation: unknown / missing / legacy tags are
        # mapped via ``ClusterKind.from_string`` (returns UNTAGGED
        # for unknown). Legacy taxonomy tags (entity_variant, etc.)
        # are remapped to the new taxonomy so prompts that drift
        # back to the old vocabulary still parse.
        kind = ClusterKind.from_string(entry.get("kind"))
        # Canonical normalisation: drop when not present in the kept
        # values (LLM may emit a canonical that referred to a
        # hallucinated value we just stripped).
        canonical_raw = entry.get("canonical")
        canonical: Optional[str] = None
        if isinstance(canonical_raw, str) and canonical_raw in values:
            canonical = canonical_raw
        # Target-style hint: free-text description of the competing
        # majority style. Trimmed; empty or non-string -> None so
        # downstream consumers can rely on a truthy/None contract.
        target_style_raw = entry.get("target_style")
        target_style: Optional[str] = None
        if isinstance(target_style_raw, str):
            stripped_hint = target_style_raw.strip()
            if stripped_hint:
                target_style = stripped_hint

        out.append(
            ProposedCluster(
                values=tuple(sorted(values)),
                kind=kind,
                canonical=canonical,
                target_style=target_style,
            )
        )
    return out, None
