"""Post-discovery aggregation of paraphrased rules.

A multi-pass discovery run emits many paraphrases of the same
underlying constraint - the species_key() canonicalization at
src.core.relations folds duplicates at the token level, but the
LLM can rephrase a rule with no overlapping distinctive tokens
("Dossard uniquely identifies a runner" vs "Dossard (bib number)
deterministically determines participant attributes") and the
canonicalizer cannot tell they are the same rule.

This module clusters discovered rules by ``(source, error_type,
sorted column tuple)`` and asks the LLM to PARTITION each cluster
into the smallest set of canonical rules that capture the
underlying structural constraints. Two kinds of merge happen here:

* Paraphrases - rules with identical logical content - collapse
  into one canonical.
* Nested rules - where one is logically stronger than another
  (its violations are a subset) - also collapse, with the
  WEAKER / MORE GENERAL rule becoming the canonical. The strict
  form is recorded in ``aggregated_from`` but does NOT become the
  canonical (see ``prompts/aggregate_relations.yaml`` for the
  full policy and examples).

Multiple canonical rules per cluster are expected and supported -
one cluster does not collapse to one rule.

Originals are NEVER discarded - they remain in ``relations.jsonl``.
The output of this pass is written to ``relations_aggregated.jsonl``
where each row is a canonical rule plus an ``aggregated_from`` list
of input species_keys; the report UI surfaces the canonical rule
as the primary card with the originals tucked under expand.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from src.core.llm_client import LLMClient
from src.core.relations import Relation, sample_rows_for_prompt
from src.core.table_digest import TableDigest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _cluster_key(rel: Relation) -> Tuple[str, str, Tuple[str, ...]]:
    """Cluster identity: same source, same error_type, same column
    set (order-insensitive). Different ``kind`` values are folded
    together because the LLM frequently rephrases the same FD as
    "fd" in one pass and "composite_fd" in another."""
    return (rel.source, rel.error_type, tuple(sorted(rel.columns)))


def cluster_relations_by_arm(
    rels: Sequence[Relation],
    *,
    min_cluster_size: int = 2,
) -> List[List[Relation]]:
    """Group rules into ``(source, error_type, columns)`` clusters.

    Only clusters of size ``>= min_cluster_size`` are returned -
    singletons cannot be aggregated. The order of rules within
    each cluster matches their order in ``rels`` so the LLM-side
    1-indexed ``covers`` positions are stable.
    """
    by_key: Dict[Tuple[str, str, Tuple[str, ...]], List[Relation]] = {}
    for r in rels:
        by_key.setdefault(_cluster_key(r), []).append(r)
    return [grp for grp in by_key.values() if len(grp) >= min_cluster_size]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregationPromptConfig:
    system_prompt: str
    user_template: str

    def render(
        self,
        *,
        cluster: Sequence[Relation],
        digest: TableDigest,
        df: pd.DataFrame,
        n_row_samples: int = 6,
    ) -> str:
        first = cluster[0]
        bullets: List[str] = []
        for i, r in enumerate(cluster, start=1):
            kind = r.kind or "?"
            pred = (r.predicate or "").replace("\n", " ").strip()
            ex = r.executable
            ex_part = f"  executable: {ex}" if ex else ""
            bullets.append(
                f"  {i}. kind={kind}  conf={r.confidence}\n"
                f"     predicate: {pred}{ex_part}"
            )
        rows_md = sample_rows_for_prompt(
            df,
            n_samples=n_row_samples,
            total_row_indices=digest.total_row_indices,
        )
        return self.user_template.format(
            source=first.source,
            error_type=first.error_type,
            cluster_size=len(cluster),
            columns="\n".join(f"  - {c}" for c in first.columns),
            cluster_rules="\n".join(bullets),
            sample_rows=rows_md,
        )


def load_aggregation_prompt(path: Path | str) -> AggregationPromptConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AggregationPromptConfig(
        system_prompt=str(data["system"]).strip(),
        user_template=str(data["user_template"]),
    )


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


@dataclass
class CanonicalRule:
    """One canonical rule output by the aggregator, paired with the
    1-indexed positions of the input cluster it covers."""

    relation: Relation
    covers: List[int]  # 1-indexed positions into the input cluster


def _coerce_canonical_rule(
    item: Any,
    template: Relation,
    cluster_size: int,
) -> Tuple[Optional[CanonicalRule], str]:
    """Lift one entry from the LLM's ``canonical: [...]`` list into
    a ``CanonicalRule``. Returns (None, reason) on rejection."""
    if not isinstance(item, dict):
        return None, "canonical entry was not an object"
    cols_raw = item.get("columns") or list(template.columns)
    if not isinstance(cols_raw, list) or not cols_raw:
        return None, "canonical entry missing `columns`"
    cols_t = tuple(str(c) for c in cols_raw)
    for c in cols_t:
        if c not in template.columns:
            return None, f"canonical referenced unknown column {c!r}"
    covers_raw = item.get("covers")
    if not isinstance(covers_raw, list) or not covers_raw:
        return None, "canonical entry missing `covers`"
    covers: List[int] = []
    for x in covers_raw:
        try:
            i = int(x)
        except (TypeError, ValueError):
            return None, f"`covers` entry not an int: {x!r}"
        if i < 1 or i > cluster_size:
            return None, f"`covers` index {i} out of range [1,{cluster_size}]"
        covers.append(i)
    lhs_raw = item.get("lhs") or []
    rhs_raw = item.get("rhs") or []
    lhs = tuple(str(c) for c in lhs_raw) if isinstance(lhs_raw, list) else ()
    rhs = tuple(str(c) for c in rhs_raw) if isinstance(rhs_raw, list) else ()
    rel = Relation(
        source=template.source,
        error_type=template.error_type,
        kind=str(item.get("kind") or template.kind),
        columns=cols_t,
        predicate=str(item.get("predicate") or "").strip(),
        rationale=str(item.get("rationale") or "").strip(),
        confidence=str(item.get("confidence") or "medium").strip(),
        lhs=lhs,
        rhs=rhs,
        executable=(
            str(item.get("executable")).strip()
            if item.get("executable") else None
        ),
        raw=dict(item),
    )
    if not rel.predicate:
        return None, "canonical entry had empty predicate"
    return CanonicalRule(relation=rel, covers=sorted(set(covers))), ""


def _coerce_aggregation(
    payload: Any,
    cluster: Sequence[Relation],
) -> Tuple[List[CanonicalRule], str]:
    """Lift the LLM's ``{"canonical": [...], "reason": "..."}``
    response into a list of CanonicalRules. Returns ([], reason)
    when the LLM declines OR when validation fails (every input
    rule must appear in exactly one ``covers`` list)."""
    if not isinstance(payload, dict):
        return [], "LLM response was not a JSON object"
    reason = str(payload.get("reason") or "")
    canonical = payload.get("canonical")
    if canonical in (None, [], {}):
        return [], reason or "LLM declined"
    if not isinstance(canonical, list):
        return [], f"`canonical` was not a list: {type(canonical).__name__}"
    out: List[CanonicalRule] = []
    template = cluster[0]
    n = len(cluster)
    for item in canonical:
        cr, err = _coerce_canonical_rule(item, template, n)
        if cr is None:
            return [], err or "could not coerce canonical entry"
        out.append(cr)
    if not out:
        return [], reason or "no canonical rules emitted"
    # Validate the partition: union of covers == {1..n}, no overlaps.
    seen: Dict[int, int] = {}  # input_idx -> canonical_idx
    for cidx, cr in enumerate(out):
        for i in cr.covers:
            if i in seen:
                return [], (
                    f"input rule {i} assigned to multiple canonical "
                    f"buckets ({seen[i]+1} and {cidx+1})"
                )
            seen[i] = cidx
    missing = [i for i in range(1, n + 1) if i not in seen]
    if missing:
        return [], f"input rules not covered: {missing}"
    # Reject the no-op partition (every canonical covers exactly one
    # input - that's just renaming, not aggregation).
    if all(len(cr.covers) == 1 for cr in out):
        return [], "no merges proposed (every canonical covers 1 input)"
    return out, reason


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


@dataclass
class AggregationOutcome:
    """One LLM result for one cluster.

    ``replacements`` is the list of canonical Relations to surface;
    when empty, the LLM declined or validation failed and the
    originals are kept untouched. ``original_keys`` is the species_key
    list of the input cluster so persistence can stitch originals
    back to the canonical card."""

    cluster: List[Relation]
    replacements: List[CanonicalRule] = field(default_factory=list)
    reason: str = ""
    error: Optional[str] = None
    raw_text: str = ""

    @property
    def original_keys(self) -> List[str]:
        return [r.species_key() for r in self.cluster]

    @property
    def replaced(self) -> bool:
        return bool(self.replacements)


def _aggregate_one(
    cluster: List[Relation],
    *,
    digest: TableDigest,
    df: pd.DataFrame,
    prompt_cfg: AggregationPromptConfig,
    llm: LLMClient,
    model: str,
    reasoning_effort: str,
    reasoning_summary: str,
    n_row_samples: int,
) -> AggregationOutcome:
    user = prompt_cfg.render(
        cluster=cluster,
        digest=digest,
        df=df,
        n_row_samples=n_row_samples,
    )
    try:
        resp = llm.query(
            model=model,
            prompt=user,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            system_prompt=prompt_cfg.system_prompt,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return AggregationOutcome(
            cluster=cluster,
            error=f"{type(exc).__name__}: {exc}",
        )
    if resp.error:
        return AggregationOutcome(
            cluster=cluster, error=resp.error, raw_text=resp.text,
        )
    try:
        obj = json.loads(resp.text)
    except json.JSONDecodeError:
        start = resp.text.find("{")
        end = resp.text.rfind("}")
        obj = None
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(resp.text[start: end + 1])
            except json.JSONDecodeError:
                obj = None
    if obj is None:
        return AggregationOutcome(
            cluster=cluster, error="could not parse JSON",
            raw_text=resp.text,
        )
    replacements, reason = _coerce_aggregation(obj, cluster)
    return AggregationOutcome(
        cluster=cluster,
        replacements=replacements,
        reason=reason,
        raw_text=resp.text,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@dataclass
class AggregationResult:
    outcomes: List[AggregationOutcome] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return len(self.outcomes)

    @property
    def n_aggregated_clusters(self) -> int:
        return sum(1 for o in self.outcomes if o.replaced)

    @property
    def n_declined(self) -> int:
        return sum(
            1 for o in self.outcomes
            if not o.replaced and not o.error
        )

    @property
    def n_errored(self) -> int:
        return sum(1 for o in self.outcomes if o.error)

    @property
    def n_canonical_rules(self) -> int:
        return sum(len(o.replacements) for o in self.outcomes if o.replaced)

    @property
    def n_originals_collapsed(self) -> int:
        """Total originals covered by emitted canonical rules - i.e.
        the number of paraphrase duplicates we just eliminated."""
        return sum(
            len(o.cluster) for o in self.outcomes if o.replaced
        )

    @property
    def covered_species_keys(self) -> set[str]:
        """Species keys of all originals subsumed by an emitted
        canonical rule. Used by downstream pipeline to know which
        originals the report UI should hide from the main list."""
        keys: set[str] = set()
        for o in self.outcomes:
            if o.replaced:
                for r in o.cluster:
                    keys.add(r.species_key())
        return keys

    def canonical_relations(self) -> List[Relation]:
        """Flat list of all canonical Relations across clusters."""
        out: List[Relation] = []
        for o in self.outcomes:
            for cr in o.replacements:
                out.append(cr.relation)
        return out

    def post_aggregation_relations(
        self,
        original_relations: Sequence[Relation],
    ) -> List[Relation]:
        """Return the relation set the rest of the pipeline should
        consume: originals NOT covered by any canonical rule, plus
        the canonical rules themselves."""
        covered = self.covered_species_keys
        out: List[Relation] = [
            r for r in original_relations
            if r.species_key() not in covered
        ]
        out.extend(self.canonical_relations())
        return out


def aggregate_relations(
    relations: Sequence[Relation],
    *,
    digests: Mapping[str, TableDigest],
    source_frames: Mapping[str, pd.DataFrame],
    prompt_cfg: AggregationPromptConfig,
    llm: LLMClient,
    model: str,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "detailed",
    n_row_samples: int = 6,
    min_cluster_size: int = 2,
    concurrency: int = 4,
) -> AggregationResult:
    """Cluster relations by arm and ask the LLM to merge paraphrases
    within each cluster. See module docstring."""
    clusters = cluster_relations_by_arm(
        relations, min_cluster_size=min_cluster_size,
    )
    if not clusters:
        logger.info("Aggregation: no clusters of size >= %d.", min_cluster_size)
        return AggregationResult(outcomes=[])
    logger.info(
        "Aggregation: %d cluster(s) covering %d original relation(s).",
        len(clusters), sum(len(c) for c in clusters),
    )
    outcomes: List[AggregationOutcome] = []
    if concurrency <= 1 or len(clusters) == 1:
        for cluster in clusters:
            src = cluster[0].source
            outcomes.append(_aggregate_one(
                cluster,
                digest=digests[src],
                df=source_frames[src],
                prompt_cfg=prompt_cfg,
                llm=llm,
                model=model,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                n_row_samples=n_row_samples,
            ))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as exe:
            futures = [
                exe.submit(
                    _aggregate_one,
                    cluster,
                    digest=digests[cluster[0].source],
                    df=source_frames[cluster[0].source],
                    prompt_cfg=prompt_cfg,
                    llm=llm,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                    n_row_samples=n_row_samples,
                )
                for cluster in clusters
            ]
            for fut in as_completed(futures):
                outcomes.append(fut.result())
    return AggregationResult(outcomes=outcomes)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_aggregated_jsonl(
    result: AggregationResult,
    path: Path,
) -> int:
    """Persist successful aggregations to ``relations_aggregated.jsonl``.

    Each record is a canonical ``Relation.as_dict()`` plus
    ``aggregated_from`` (species_keys of the originals it covers),
    ``n_originals``, and ``aggregation_reason``. Declined / errored
    clusters are NOT written here - originals stay authoritative in
    ``relations.jsonl``.
    """
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for outcome in result.outcomes:
            if not outcome.replaced:
                continue
            for cr in outcome.replacements:
                # Map this canonical's 1-indexed covers back to the
                # original Relations, then to their species keys.
                covered_keys = [
                    outcome.cluster[i - 1].species_key()
                    for i in cr.covers
                ]
                rec = cr.relation.as_dict()
                rec["aggregated_from"] = covered_keys
                rec["n_originals"] = len(covered_keys)
                rec["aggregation_reason"] = outcome.reason
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def write_aggregation_report_json(
    result: AggregationResult,
    path: Path,
) -> None:
    """Persist a per-cluster diagnostic so reasons / errors are
    auditable even when no canonical was emitted."""
    rows: List[Dict[str, Any]] = []
    for outcome in result.outcomes:
        rows.append({
            "source": outcome.cluster[0].source,
            "error_type": outcome.cluster[0].error_type,
            "columns": list(outcome.cluster[0].columns),
            "n_originals": len(outcome.cluster),
            "original_keys": outcome.original_keys,
            "replaced": outcome.replaced,
            "n_canonical": len(outcome.replacements),
            "reason": outcome.reason,
            "error": outcome.error,
        })
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
