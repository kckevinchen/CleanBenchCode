"""Post-discovery generalization of value-keyed rule families.

Some discovery passes still emit families of value-keyed rules (one
rule per observed cell value of a free-text column) despite the
prompt's PARAMETRIC ABSTRACTION clause:

    "If Description == 'Exemption 10 MW' then Capacity (kW) <= 10000"
    "If Description == 'Exemption 5 MW'  then Capacity (kW) <= 5000"
    "If Description == 'Exemption 1 MW'  then Capacity (kW) <= 1000"

These are unverifiable at the schema level and bloat the rule set.
This module clusters such families by ``(source, error_type, sorted
column tuple)`` and asks the LLM to rewrite each cluster as ONE
parametric rule, OR explicitly decline.

Originals are NEVER discarded - they remain in ``relations.jsonl``.
The output of this pass is written to ``relations_generalized.jsonl``
with a ``generalized_from`` list of the input species_keys, and the
report UI surfaces the generalization as the primary card with the
original family tucked under expand.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from src.core.llm_client import LLMClient
from src.core.relations import (
    Relation,
    sample_rows_for_prompt,
)
from src.core.table_digest import TableDigest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic: is this a value-keyed rule?
# ---------------------------------------------------------------------------

# A predicate is value-keyed when it pins one of its referenced
# columns to a literal cell value (single- or double-quoted) - this
# is the LLM's typical surface form for the kind of family we want
# to generalize. The literal also has to *actually appear* as a
# value in that column according to the digest, otherwise the
# quoted text is just an example and the rule is generic.
_QUOTED_LITERAL_RE = re.compile(r"""(['"])([^'"]{1,80})\1""")


def is_value_conditional(rel: Relation, digest: Optional[TableDigest]) -> bool:
    """Return True when the rule conditions on a literal cell value
    that's actually present in one of its columns.
    """
    pred = rel.predicate or ""
    if not pred:
        return False
    quoted = [m.group(2) for m in _QUOTED_LITERAL_RE.finditer(pred)]
    if not quoted:
        return False
    if digest is None:
        # Without a digest we can't verify cell membership. Fall back
        # to assuming any quoted literal that's clearly not a column
        # name signals a value-keyed rule. The prompt-side guard
        # already discourages these, so a soft heuristic is fine.
        col_names = {c.lower() for c in rel.columns}
        return any(q.strip().lower() not in col_names for q in quoted)
    # Tight check: the literal appears as a top-value, sample value,
    # or any preview-row value of one of the relation's columns.
    for col in rel.columns:
        cd = digest.columns.get(col)
        if cd is None:
            continue
        haystack: set[str] = set()
        for v, _ in cd.top_values:
            haystack.add(str(v))
        for v in cd.sample_values:
            haystack.add(str(v))
        for q in quoted:
            if q in haystack:
                return True
    return False


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _cluster_key(rel: Relation) -> Tuple[str, str, Tuple[str, ...]]:
    """Cluster identity: same source, same error type, same column
    set (order-insensitive). Different `kind`s are deliberately
    folded together because the LLM frequently splits the same rule
    across "implication" / "cap" / "factual"."""
    return (rel.source, rel.error_type, tuple(sorted(rel.columns)))


def cluster_value_conditional(
    rels: Sequence[Relation],
    digests: Mapping[str, TableDigest],
    *,
    min_cluster_size: int = 2,
) -> List[List[Relation]]:
    """Group value-conditional rules into families ready for the
    generalization LLM call. Returns clusters of size at least
    ``min_cluster_size``; smaller groups are not worth a round trip.
    """
    by_key: Dict[Tuple[str, str, Tuple[str, ...]], List[Relation]] = {}
    for rel in rels:
        if not is_value_conditional(rel, digests.get(rel.source)):
            continue
        by_key.setdefault(_cluster_key(rel), []).append(rel)
    clusters: List[List[Relation]] = []
    for key, bucket in by_key.items():
        if len(bucket) >= min_cluster_size:
            clusters.append(bucket)
    return clusters


# ---------------------------------------------------------------------------
# Prompt config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneralizationPromptConfig:
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


def load_generalization_prompt(path: Path | str) -> GeneralizationPromptConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return GeneralizationPromptConfig(
        system_prompt=str(data["system"]).strip(),
        user_template=str(data["user_template"]),
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


@dataclass
class GeneralizationOutcome:
    """One LLM result for one cluster.

    ``replacement`` is the parametric Relation to emit; when ``None``
    the LLM declined and the originals are kept untouched.
    ``original_keys`` is the list of species_keys of the input cluster
    so the report can stitch originals back to the parametric card.
    """

    cluster: List[Relation]
    replacement: Optional[Relation]
    reason: str = ""
    error: Optional[str] = None
    raw_text: str = ""

    @property
    def original_keys(self) -> List[str]:
        return [r.species_key() for r in self.cluster]


def _coerce_replacement(
    payload: Any,
    template: Relation,
) -> Tuple[Optional[Relation], str]:
    """Lift the LLM's ``{"generalized": {...}, "reason": "..."}``
    response into a Relation, or return (None, reason) on decline.
    """
    if not isinstance(payload, dict):
        return None, "LLM response was not a JSON object"
    reason = str(payload.get("reason") or "")
    gen = payload.get("generalized")
    if gen in (None, {}, []):
        return None, reason or "LLM declined"
    if not isinstance(gen, dict):
        return None, f"`generalized` was not an object: {type(gen).__name__}"
    cols = gen.get("columns") or list(template.columns)
    if not isinstance(cols, list) or not cols:
        return None, "`columns` missing or empty"
    cols_t = tuple(str(c) for c in cols)
    # Reject columns the LLM invented.
    for c in cols_t:
        if c not in template.columns:
            return None, f"parametric rule referenced unknown column {c!r}"
    lhs_raw = gen.get("lhs") or []
    rhs_raw = gen.get("rhs") or []
    lhs = tuple(str(c) for c in lhs_raw) if isinstance(lhs_raw, list) else ()
    rhs = tuple(str(c) for c in rhs_raw) if isinstance(rhs_raw, list) else ()
    rel = Relation(
        source=template.source,
        error_type=template.error_type,
        kind=str(gen.get("kind") or template.kind),
        columns=cols_t,
        predicate=str(gen.get("predicate") or "").strip(),
        rationale=str(gen.get("rationale") or "").strip(),
        confidence=str(gen.get("confidence") or "medium").strip(),
        lhs=lhs,
        rhs=rhs,
        executable=(
            str(gen.get("executable")).strip()
            if gen.get("executable")
            else None
        ),
        raw=dict(gen),
    )
    if not rel.predicate:
        return None, "parametric rule had empty predicate"
    return rel, reason


def _generalize_one(
    cluster: List[Relation],
    *,
    digest: TableDigest,
    df: pd.DataFrame,
    prompt_cfg: GeneralizationPromptConfig,
    llm: LLMClient,
    model: str,
    reasoning_effort: str,
    reasoning_summary: str,
    n_row_samples: int,
) -> GeneralizationOutcome:
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
        return GeneralizationOutcome(
            cluster=cluster, replacement=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    if resp.error:
        return GeneralizationOutcome(
            cluster=cluster, replacement=None,
            error=resp.error, raw_text=resp.text,
        )
    # Parse the LLM's full JSON object (we want the wrapper with
    # ``generalized`` and ``reason``, not just an inner relations list).
    try:
        obj = json.loads(resp.text)
    except json.JSONDecodeError:
        # Try to extract the outermost {...} block (LLM may have
        # added prose before / after).
        start = resp.text.find("{")
        end = resp.text.rfind("}")
        obj = None
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(resp.text[start: end + 1])
            except json.JSONDecodeError:
                obj = None
    if obj is None:
        return GeneralizationOutcome(
            cluster=cluster, replacement=None,
            error="could not parse JSON",
            raw_text=resp.text,
        )
    replacement, reason = _coerce_replacement(obj, cluster[0])
    return GeneralizationOutcome(
        cluster=cluster,
        replacement=replacement,
        reason=reason,
        raw_text=resp.text,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@dataclass
class GeneralizationResult:
    """Aggregate outcome of a generalization pass."""

    outcomes: List[GeneralizationOutcome] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return len(self.outcomes)

    @property
    def n_generalized(self) -> int:
        return sum(1 for o in self.outcomes if o.replacement is not None)

    @property
    def n_declined(self) -> int:
        return sum(
            1 for o in self.outcomes
            if o.replacement is None and not o.error
        )

    @property
    def n_errored(self) -> int:
        return sum(1 for o in self.outcomes if o.error)

    @property
    def n_originals_collapsed(self) -> int:
        return sum(
            len(o.cluster) for o in self.outcomes if o.replacement is not None
        )


def generalize_relations(
    relations: Sequence[Relation],
    *,
    digests: Mapping[str, TableDigest],
    source_frames: Mapping[str, pd.DataFrame],
    prompt_cfg: GeneralizationPromptConfig,
    llm: LLMClient,
    model: str,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "detailed",
    n_row_samples: int = 6,
    min_cluster_size: int = 2,
    concurrency: int = 4,
) -> GeneralizationResult:
    """Cluster value-conditional rules and ask the LLM to generalize
    each cluster. See module docstring."""
    clusters = cluster_value_conditional(
        relations, digests, min_cluster_size=min_cluster_size,
    )
    if not clusters:
        logger.info("Generalization: no clusters found.")
        return GeneralizationResult(outcomes=[])
    logger.info(
        "Generalization: %d cluster(s) covering %d original relation(s).",
        len(clusters), sum(len(c) for c in clusters),
    )
    outcomes: List[GeneralizationOutcome] = []
    if concurrency <= 1 or len(clusters) == 1:
        for cluster in clusters:
            src = cluster[0].source
            outcomes.append(_generalize_one(
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
                    _generalize_one,
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
    return GeneralizationResult(outcomes=outcomes)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_generalized_jsonl(
    result: GeneralizationResult,
    path: Path,
) -> int:
    """Persist successful generalizations to ``relations_generalized.jsonl``.

    Each record is the parametric ``Relation.as_dict()`` plus a
    ``generalized_from`` list of original species_keys and an
    ``n_originals`` count for quick UI display. Declined / errored
    clusters are NOT written here - originals stay authoritative in
    ``relations.jsonl``.
    """
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for outcome in result.outcomes:
            if outcome.replacement is None:
                continue
            rec = outcome.replacement.as_dict()
            rec["generalized_from"] = outcome.original_keys
            rec["n_originals"] = len(outcome.cluster)
            rec["generalization_reason"] = outcome.reason
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_generalization_report_json(
    result: GeneralizationResult, path: Path,
) -> None:
    """Persist a per-cluster diagnostic so reasons / errors are
    auditable even when no replacement was emitted."""
    rows: List[Dict[str, Any]] = []
    for outcome in result.outcomes:
        rows.append({
            "source": outcome.cluster[0].source,
            "error_type": outcome.cluster[0].error_type,
            "columns": list(outcome.cluster[0].columns),
            "n_originals": len(outcome.cluster),
            "original_keys": outcome.original_keys,
            "replaced": outcome.replacement is not None,
            "reason": outcome.reason,
            "error": outcome.error,
        })
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
