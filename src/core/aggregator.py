"""Phase-A clustering and Phase-B LLM synthesis for per-type detections.

The per-type detector emits one issue per ``(row, error_type)`` verdict. The
16 error types are *not* independent: a single underlying defect often
triggers several of them on the same cell (a blank Actual Completion Date
in a "Completed" row is legitimately flagged as ``missing_values``,
``violated_attribute_dependencies``, ``rule_violations``, ``illegal_values``,
and ``scsr`` at the same time). Hand-reviewing that raw firehose is
impractical, so this module reconciles it into cluster records that each
represent one underlying defect.

Two phases:

Phase A  (:func:`cluster_issues`) - deterministic, free
    Connected-component clustering of issues whose ``row`` matches and whose
    ``columns`` sets overlap. All bookkeeping (evidence_count,
    row_values, distinct error types/categories) is derived locally.

Phase B  (:class:`ClusterSynthesiser`) - one LLM call per multi-type cluster
    Single-call reconciliation: the model picks a ``primary_error_type``,
    routes the rest into ``supporting_error_types``, and emits a canonical
    explanation + fix + confidence + false-positive flag. Single-detection
    clusters skip Phase B (there is nothing to reconcile).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .error_taxonomy import ErrorTaxonomy
from .llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


CONFIDENCE_VALUES = {"high", "medium", "low", "unknown"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClusterDetection:
    """One Phase-A detection attached to a cluster."""

    issue_id: str
    error_type: str
    error_type_name: str
    error_type_category: str
    observed: str
    explanation: str
    suggested_fix: Optional[str]
    columns: List[str]
    # Sibling rows the detector claimed are part of the same logical
    # incident (e.g. members of a row_duplication group). Preserved
    # verbatim from the flat issue so the labelling UI can bucket
    # clusters that share a duplicate-group key. Empty for
    # cell-/row-local error types.
    affected_rows: List[int] = field(default_factory=list)
    # Tri-state verdict for row_duplication detections. ``"duplicate"``
    # when the detector is confident the rows are interchangeable;
    # ``"near_duplicate"`` when they likely refer to the same entity
    # but a human should review. Empty / unset for non-duplication
    # detections (the UI ignores it for those).
    verdict: str = ""
    # Which clusterer surfaced this detection (e.g. ``"key_collision"``,
    # ``"blocking"``, ``"minhash_lsh"``). Empty for non-duplication
    # detections.
    discovery_strategy: str = ""
    # True when the per-source clustering budget was hit while
    # producing this detection's source - i.e. some duplicate
    # candidates may have been silently dropped. The UI surfaces a
    # warning so the operator can rerun with a larger cap.
    cluster_budget_exhausted: bool = False
    # The cap that was hit, copied through for UI display.
    max_groups_per_source: int = 0


@dataclass
class ClusterSynthesis:
    """Phase-B LLM reconciliation for a multi-type cluster."""

    primary_error_type: str = ""
    supporting_error_types: List[str] = field(default_factory=list)
    root_cause: str = ""
    canonical_explanation: str = ""
    canonical_fix: Optional[str] = None
    confidence: str = "unknown"
    is_likely_false_positive: bool = False
    reviewer_note: str = ""
    # Provenance
    endpoint: str = ""
    attempts: int = 0
    raw_text: str = ""
    error: Optional[str] = None
    parse_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.error and not self.parse_error


@dataclass
class Cluster:
    """One Phase-A cluster plus an optional Phase-B synthesis."""

    cluster_id: str
    row: Optional[int]
    source: str
    columns: List[str]
    row_values: Dict[str, str]
    error_types: List[str]
    error_type_categories: List[str]
    evidence_count: int
    detections: List[ClusterDetection]
    synthesis: Optional[ClusterSynthesis] = None
    # ``inferred=True`` marks clusters that only exist because a sibling
    # detection listed this row in ``affected_rows`` - the LLM never
    # filed a primary detection for this exact row. We expose them so
    # the operator can label every row the LLM claimed (essential for
    # FN-aware benchmark building) while still letting the UI render
    # them with a softer treatment.
    inferred: bool = False

    @property
    def is_multi_type(self) -> bool:
        return len(self.error_types) >= 2

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "cluster_id": self.cluster_id,
            "row": self.row,
            "source": self.source,
            "columns": list(self.columns),
            "row_values": dict(self.row_values),
            "error_types": list(self.error_types),
            "error_type_categories": list(self.error_type_categories),
            "evidence_count": self.evidence_count,
            "detections": [asdict(d) for d in self.detections],
            "inferred": self.inferred,
        }
        payload["synthesis"] = asdict(self.synthesis) if self.synthesis else None
        return payload


# ---------------------------------------------------------------------------
# Phase A - deterministic clustering
# ---------------------------------------------------------------------------


def _dedup_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


class _UnionFind:
    """Tiny union-find with path compression and union-by-rank."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def _cluster_row_group(
    row: Optional[int],
    entries: Sequence[Tuple[Mapping[str, Any], bool]],
) -> List[Tuple[Optional[int], List[Tuple[Mapping[str, Any], bool]]]]:
    """Split entries sharing the same row into column-overlap components.

    ``entries`` is a list of ``(issue, is_primary)`` pairs. ``is_primary``
    flags whether the issue *natively* targets this row (vs only
    listing it in ``affected_rows``). The mask is preserved through the
    union-find grouping so :func:`_build_cluster` can decide which
    contributors should supply ``row_values``.
    """
    issues = [e[0] for e in entries]
    n = len(issues)
    uf = _UnionFind(n)
    col_sets = [frozenset((i.get("columns") or [])) for i in issues]
    no_col_bucket: List[int] = []
    for i in range(n):
        if not col_sets[i]:
            no_col_bucket.append(i)
        for j in range(i + 1, n):
            if col_sets[i] and col_sets[j] and col_sets[i] & col_sets[j]:
                uf.union(i, j)
    # Issues with no column information on the same row all cluster together
    # (they mostly come from dataset-wide flags like SCMR/duplicates where
    # the detector reported rows but not columns; treating them as one
    # evidence group for that row is the right default).
    for i, j in zip(no_col_bucket, no_col_bucket[1:]):
        uf.union(i, j)

    groups: Dict[int, List[Tuple[Mapping[str, Any], bool]]] = {}
    for idx, entry in enumerate(entries):
        groups.setdefault(uf.find(idx), []).append(entry)

    # Deterministic ordering: larger clusters first, ties broken by first
    # issue id in the group.
    def _key(members: List[Tuple[Mapping[str, Any], bool]]) -> Tuple[int, str]:
        return (-len(members), str(members[0][0].get("id", "")))

    ordered = sorted(groups.values(), key=_key)
    return [(row, members) for members in ordered]


def _build_cluster(
    cluster_id: str,
    row: Optional[int],
    issues: Sequence[Mapping[str, Any]],
    *,
    primary_mask: Optional[Sequence[bool]] = None,
) -> Cluster:
    """Build one cluster from a bucket of issues.

    ``primary_mask`` (one bool per issue, same length) says which issues
    *natively* target this bucket's row (issue.row == row). Inferred
    contributors - issues that only mention this row via
    ``affected_rows`` - have ``False``. ``row_values`` is sourced only
    from primary contributors so an inferred-only cluster does not
    inherit a sibling row's snapshot. When every contributor is
    inferred, the cluster is marked ``inferred=True`` and ``row_values``
    is left empty (the labelling UI fetches the actual CSV row).
    """
    if not issues:
        raise ValueError("_build_cluster requires at least one issue")

    if primary_mask is None:
        # Back-compat default: every issue is treated as primary.
        primary_mask = [True] * len(issues)
    if len(primary_mask) != len(issues):
        raise ValueError("primary_mask length must match issues length")

    first = issues[0]
    source = str(first.get("source", "") or "")

    merged_columns = _dedup_preserve_order(
        col for issue in issues for col in (issue.get("columns") or [])
    )

    # Row values - only attribute to primary contributors so an
    # affected-rows-only cluster does not inherit row 6's snapshot
    # when its own row is 14.
    row_values: Dict[str, str] = {}
    for issue, is_primary in zip(issues, primary_mask):
        if not is_primary:
            continue
        rv = issue.get("full_row_values") or issue.get("row_values") or {}
        if not isinstance(rv, Mapping):
            continue
        for k, v in rv.items():
            if k not in row_values or (row_values[k] == "" and v):
                row_values[k] = "" if v is None else str(v)

    error_types = _dedup_preserve_order(
        str(issue.get("error_type", "unknown")) for issue in issues
    )
    error_type_categories = _dedup_preserve_order(
        str(issue.get("error_type_category", "unknown")) for issue in issues
    )

    detections: List[ClusterDetection] = []
    for issue in issues:
        ar_raw = issue.get("affected_rows") or ()
        ar_clean: List[int] = []
        for r in ar_raw:
            try:
                ar_clean.append(int(r))
            except (TypeError, ValueError):
                continue
        detections.append(
            ClusterDetection(
                issue_id=str(issue.get("id", "")),
                error_type=str(issue.get("error_type", "unknown")),
                error_type_name=str(
                    issue.get("error_type_name") or issue.get("error_type") or "unknown"
                ),
                error_type_category=str(
                    issue.get("error_type_category", "unknown")
                ),
                observed=str(issue.get("observed", "") or ""),
                explanation=str(issue.get("explanation", "") or ""),
                suggested_fix=issue.get("suggested_fix"),
                columns=list(issue.get("columns") or []),
                affected_rows=ar_clean,
                verdict=str(issue.get("verdict", "") or ""),
                discovery_strategy=str(issue.get("discovery_strategy", "") or ""),
                cluster_budget_exhausted=bool(
                    issue.get("cluster_budget_exhausted", False)
                ),
                max_groups_per_source=int(
                    issue.get("max_groups_per_source") or 0
                ),
            )
        )

    return Cluster(
        cluster_id=cluster_id,
        row=row,
        source=source,
        columns=merged_columns,
        row_values=row_values,
        error_types=error_types,
        error_type_categories=error_type_categories,
        evidence_count=len(detections),
        detections=detections,
        inferred=not any(primary_mask),
    )


_CLUSTER_SOURCE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _cluster_source_slug(source: str) -> str:
    """Stable, file-system safe slug used inside cluster ids."""
    cleaned = _CLUSTER_SOURCE_RE.sub("_", source).strip("_")
    return cleaned or "src"


def _representative_detection(
    detections: Sequence[ClusterDetection],
) -> ClusterDetection:
    """Pick the most informative detection from a single-type cluster.

    Priority is the detection whose textual evidence is richest:
    longest non-empty ``explanation`` first, then longest non-empty
    ``suggested_fix``, then longest ``observed``. Stable tie-break by
    ``issue_id`` so the choice is deterministic across runs.
    """
    def _key(d: ClusterDetection) -> tuple:
        return (
            -len((d.explanation or "").strip()),
            -len((d.suggested_fix or "").strip()),
            -len((d.observed or "").strip()),
            d.issue_id,
        )

    return sorted(detections, key=_key)[0]


def _auto_synthesise_single_type(cluster: Cluster) -> None:
    """Populate ``cluster.synthesis`` for a single-type, multi-evidence
    cluster without calling the LLM.

    Phase B is reserved for reconciling DIFFERENT error types on the same
    cell - there is nothing to reconcile when every detection agrees on
    the type. But downstream consumers (markdown report, HTML dashboard,
    labeling UI) still want a canonical (explanation, fix) pair instead
    of falling back to ``detections[0]`` blindly. This helper picks the
    richest detection deterministically and stamps it as the cluster's
    synthesis, marked with confidence derived from how much corroboration
    the cluster received across detections.

    Called only when ``len(cluster.error_types) == 1`` and
    ``cluster.evidence_count >= 2``. No-op otherwise.
    """
    if len(cluster.error_types) != 1 or cluster.evidence_count < 2:
        return
    rep = _representative_detection(cluster.detections)
    # Confidence ladder is intentionally simple: more independent
    # detections of the same defect = higher confidence. Tunable; the
    # exact thresholds are a documentation choice, not a correctness one.
    if cluster.evidence_count >= 4:
        confidence = "high"
    elif cluster.evidence_count >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    cluster.synthesis = ClusterSynthesis(
        primary_error_type=rep.error_type,
        supporting_error_types=[],
        root_cause="",
        canonical_explanation=rep.explanation,
        canonical_fix=rep.suggested_fix,
        confidence=confidence,
        is_likely_false_positive=False,
        reviewer_note=(
            f"auto-synthesised from {cluster.evidence_count} corroborating "
            f"detection(s) of {rep.error_type}; no LLM call"
        ),
    )


def cluster_issues(flat_issues: Sequence[Mapping[str, Any]]) -> List[Cluster]:
    """Group Phase-A flat issues into overlap-based clusters.

    An issue lands in the bucket for its primary ``row`` and *also* in
    the bucket for each row it lists in ``affected_rows``. The latter
    are tagged ``is_primary=False`` so the resulting cluster knows it
    is an *inferred* match (the LLM only mentioned this row by
    reference, not by filing a primary detection there). This keeps
    every LLM-claimed cell labellable - essential for FN-aware
    benchmark building - without conflating it with detections the
    LLM volunteered as primary examples.

    Within a bucket, two entries cluster iff their ``columns`` sets
    intersect (or both have no columns). Source-awareness matters for
    multi-CSV runs: row 5 of file A and row 5 of file B refer to
    unrelated entities and must never cluster together.
    """
    by_source_row: Dict[Tuple[str, int], List[Tuple[Mapping[str, Any], bool]]] = {}
    no_row: List[Mapping[str, Any]] = []
    for issue in flat_issues:
        primary_row = issue.get("row")
        source = str(issue.get("source", "") or "")
        rows: Dict[int, bool] = {}  # row -> is_primary
        if isinstance(primary_row, int):
            rows[primary_row] = True
        for r in issue.get("affected_rows") or ():
            if isinstance(r, int):
                # Primary wins if this row is also the issue's primary row.
                rows.setdefault(r, False)
        if rows:
            for r, is_primary in rows.items():
                by_source_row.setdefault((source, r), []).append((issue, is_primary))
        else:
            no_row.append(issue)

    clusters: List[Cluster] = []
    # Stable iteration order: by source, then row.
    for (source, row) in sorted(by_source_row, key=lambda k: (k[0], k[1])):
        slug = _cluster_source_slug(source)
        components = _cluster_row_group(row, by_source_row[(source, row)])
        for idx, (_, members) in enumerate(components):
            cluster_id = f"{slug}__r{row:05d}-c{idx:03d}"
            issues = [m[0] for m in members]
            primary_mask = [m[1] for m in members]
            clusters.append(_build_cluster(
                cluster_id, row, issues, primary_mask=primary_mask,
            ))

    for idx, issue in enumerate(no_row):
        slug = _cluster_source_slug(str(issue.get("source", "") or ""))
        cluster_id = f"{slug}__rNA-c{idx:04d}"
        clusters.append(_build_cluster(cluster_id, None, [issue]))

    # Auto-synthesise single-type clusters with multiple corroborating
    # detections so downstream consumers see a canonical explanation/fix
    # instead of blindly using ``detections[0]``. Multi-type clusters are
    # left untouched here - Phase B (LLM) handles those.
    for cluster in clusters:
        _auto_synthesise_single_type(cluster)

    distinct_sources = len({c.source for c in clusters if c.source})
    logger.info(
        "Phase A: %d issues -> %d clusters across %d source(s) "
        "(%d multi-type, %d single-type)",
        len(flat_issues),
        len(clusters),
        distinct_sources,
        sum(1 for c in clusters if c.is_multi_type),
        sum(1 for c in clusters if not c.is_multi_type),
    )
    return clusters


# ---------------------------------------------------------------------------
# Phase B - LLM synthesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthesisPrompt:
    """Shared system prompt + user template for Phase B."""

    system: str
    user_template: str
    source_path: Optional[Path] = None


def load_synthesis_prompt(path: Path) -> SynthesisPrompt:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Synthesis prompt not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top level must be a mapping")
    system = str(raw.get("system", "")).strip()
    user_template = str(raw.get("user_template", ""))
    if not system or not user_template:
        raise ValueError(f"{path}: both 'system' and 'user_template' are required")
    logger.info("Loaded synthesis prompt from %s", path)
    return SynthesisPrompt(system=system, user_template=user_template, source_path=path)


def _format_row_values(row_values: Mapping[str, str]) -> str:
    if not row_values:
        return "  (no row values captured)"
    lines: List[str] = []
    for key in sorted(row_values):
        value = row_values[key]
        display = value if value not in (None, "") else "(empty)"
        lines.append(f"  - {key}: {display}")
    return "\n".join(lines)


def _format_detections(detections: Sequence[ClusterDetection]) -> str:
    blocks: List[str] = []
    for idx, det in enumerate(detections, 1):
        cols = ", ".join(det.columns) if det.columns else "(none)"
        fix = det.suggested_fix if det.suggested_fix not in (None, "") else "null"
        blocks.append(
            f"  [{idx}] error_type: {det.error_type}\n"
            f"      name:          {det.error_type_name}\n"
            f"      category:      {det.error_type_category}\n"
            f"      columns:       {cols}\n"
            f"      observed:      {det.observed or '(not reported)'}\n"
            f"      explanation:   {det.explanation or '(not reported)'}\n"
            f"      suggested_fix: {fix}"
        )
    return "\n\n".join(blocks)


def _format_taxonomy_refs(
    taxonomy: ErrorTaxonomy, error_types: Sequence[str]
) -> str:
    blocks: List[str] = []
    for key in _dedup_preserve_order(error_types):
        try:
            et = taxonomy.get(key)
        except KeyError:
            blocks.append(f"  * {key}: (not in taxonomy; treat carefully)")
            continue
        notes = et.notes.strip() or "(no additional notes)"
        blocks.append(
            f"  * {et.key} [{et.category} / {et.name}]\n"
            f"      definition:    {et.description.strip()}\n"
            f"      applicability: {notes}"
        )
    return "\n\n".join(blocks) if blocks else "  (no taxonomy references)"


def render_synthesis_prompt(
    template: str,
    cluster: Cluster,
    taxonomy: ErrorTaxonomy,
) -> str:
    try:
        return template.format(
            cluster_id=cluster.cluster_id,
            row="n/a" if cluster.row is None else cluster.row,
            detection_count=cluster.evidence_count,
            available_keys=", ".join(_dedup_preserve_order(cluster.error_types))
            or "(none)",
            row_values=_format_row_values(cluster.row_values),
            detections=_format_detections(cluster.detections),
            taxonomy_refs=_format_taxonomy_refs(taxonomy, cluster.error_types),
        )
    except KeyError as exc:
        raise KeyError(
            f"synthesise_cluster.yaml user_template has an undefined placeholder: {exc}"
        ) from exc


from .jsonl_io import JSON_FENCE_RE as _JSON_FENCE_RE, extract_json_payloads


def _parse_synthesis_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return None, "empty response"
    candidates: List[str] = []
    for m in _JSON_FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    # Bracket-balanced fallback handles concatenated objects /
    # objects with prose around them in the same response.
    candidates.extend(extract_json_payloads(text))

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, None
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0], "response was a list; used first element"
    return None, "failed to parse a JSON object from response"


def _apply_synthesis_payload(
    synthesis: ClusterSynthesis,
    payload: Mapping[str, Any],
) -> None:
    synthesis.primary_error_type = str(payload.get("primary_error_type", "")).strip()
    raw_support = payload.get("supporting_error_types") or []
    if not isinstance(raw_support, list):
        raw_support = [raw_support]
    synthesis.supporting_error_types = [
        str(s).strip() for s in raw_support if str(s).strip()
    ]
    synthesis.root_cause = str(payload.get("root_cause", "") or "").strip()
    synthesis.canonical_explanation = str(
        payload.get("canonical_explanation", "") or ""
    ).strip()
    canon_fix = payload.get("canonical_fix")
    if canon_fix in (None, "", "null", "None"):
        synthesis.canonical_fix = None
    else:
        synthesis.canonical_fix = str(canon_fix).strip() or None
    confidence = str(payload.get("confidence", "unknown") or "unknown").strip().lower()
    synthesis.confidence = confidence if confidence in CONFIDENCE_VALUES else "unknown"
    fp = payload.get("is_likely_false_positive", False)
    if isinstance(fp, str):
        fp = fp.strip().lower() in {"true", "yes", "1"}
    synthesis.is_likely_false_positive = bool(fp)
    synthesis.reviewer_note = str(payload.get("reviewer_note", "") or "").strip()


class ClusterSynthesiser:
    """Run Phase-B LLM reconciliation over multi-type clusters in parallel."""

    def __init__(
        self,
        llm: LLMClient,
        taxonomy: ErrorTaxonomy,
        prompt: SynthesisPrompt,
        model: str,
        *,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        concurrency: Optional[int] = None,
    ) -> None:
        self.llm = llm
        self.taxonomy = taxonomy
        self.prompt = prompt
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        default_concurrency = min(8, max(1, len(self.llm.endpoints)))
        self.concurrency = (
            concurrency if concurrency and concurrency > 0 else default_concurrency
        )

    # ------------------------------------------------------------------

    def _synthesise_one(self, cluster: Cluster) -> Cluster:
        rendered = render_synthesis_prompt(
            self.prompt.user_template, cluster, self.taxonomy
        )
        response: LLMResponse = self.llm.query(
            self.model,
            rendered,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            system_prompt=self.prompt.system,
        )
        synthesis = ClusterSynthesis(
            endpoint=response.endpoint,
            attempts=response.attempts,
            raw_text=response.text,
            error=response.error,
        )
        if response.ok:
            payload, parse_err = _parse_synthesis_json(response.text)
            synthesis.parse_error = parse_err
            if payload:
                _apply_synthesis_payload(synthesis, payload)
        cluster.synthesis = synthesis
        return cluster

    # ------------------------------------------------------------------

    def run(self, clusters: Sequence[Cluster]) -> List[Cluster]:
        """Attach ``ClusterSynthesis`` to every multi-type cluster in place.

        Single-type clusters (``evidence_count == 1``) are left untouched -
        there is nothing to reconcile. The input list order is preserved.
        """
        clusters = list(clusters)
        jobs = [c for c in clusters if c.is_multi_type]
        total = len(jobs)
        logger.info(
            "Phase B: %d multi-type clusters to synthesise (concurrency=%d)",
            total,
            self.concurrency,
        )
        if total == 0:
            return clusters

        progress = tqdm(
            total=total,
            desc="Synth",
            unit="cluster",
            dynamic_ncols=True,
            leave=True,
        )
        lock = threading.Lock()
        error_count = {"n": 0}

        def _job(cluster: Cluster) -> Cluster:
            self._synthesise_one(cluster)
            with lock:
                syn = cluster.synthesis
                if syn and syn.error:
                    error_count["n"] += 1
                    logger.warning(
                        "Cluster synthesis failed: cluster=%s row=%s types=%d error=%s",
                        cluster.cluster_id,
                        cluster.row,
                        cluster.evidence_count,
                        syn.error,
                    )
                elif syn and syn.parse_error:
                    logger.warning(
                        "Cluster synthesis parse error: cluster=%s row=%s parse_error=%s",
                        cluster.cluster_id,
                        cluster.row,
                        syn.parse_error,
                    )
                progress.set_postfix(errors=error_count["n"], refresh=False)
                progress.update(1)
            return cluster

        try:
            with logging_redirect_tqdm():
                if self.concurrency == 1:
                    for cluster in jobs:
                        _job(cluster)
                else:
                    with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                        futures = [pool.submit(_job, c) for c in jobs]
                        for fut in as_completed(futures):
                            fut.result()  # surface exceptions
        finally:
            progress.close()

        if error_count["n"]:
            logger.warning(
                "Phase B finished with %d/%d failed synthesis call(s).",
                error_count["n"],
                total,
            )

        return clusters


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------


def summarise_clusters(clusters: Sequence[Cluster]) -> Dict[str, Any]:
    """Produce cluster-level aggregate counters for summary.json."""
    total = len(clusters)
    multi = [c for c in clusters if c.is_multi_type]
    singles = [c for c in clusters if not c.is_multi_type]

    clusters_by_primary: Dict[str, int] = {}
    clusters_by_confidence: Dict[str, int] = {}
    false_positive = 0
    synthesised = 0           # Phase-B (LLM) only - multi-type clusters
    auto_synthesised = 0      # Single-type, multi-evidence auto-synth (no LLM)
    synthesis_failed = 0
    for cluster in clusters:
        syn = cluster.synthesis
        if syn is None:
            # Single-evidence single-type clusters have no synthesis at
            # all. Count their lone error_type into the primary breakdown
            # so the dashboard shows a complete picture.
            if cluster.error_types:
                key = cluster.error_types[0]
                clusters_by_primary[key] = clusters_by_primary.get(key, 0) + 1
            continue
        # Distinguish LLM-driven synthesis from the deterministic auto
        # path used for single-type / multi-evidence clusters.
        if cluster.is_multi_type:
            synthesised += 1
        else:
            auto_synthesised += 1
        if syn.error or syn.parse_error:
            synthesis_failed += 1
            continue
        if syn.is_likely_false_positive:
            false_positive += 1
        if syn.primary_error_type:
            clusters_by_primary[syn.primary_error_type] = (
                clusters_by_primary.get(syn.primary_error_type, 0) + 1
            )
        clusters_by_confidence[syn.confidence or "unknown"] = (
            clusters_by_confidence.get(syn.confidence or "unknown", 0) + 1
        )

    return {
        "clusters_total": total,
        "clusters_multi_type": len(multi),
        "clusters_single_type": len(singles),
        "clusters_synthesised": synthesised,
        "clusters_auto_synthesised": auto_synthesised,
        "clusters_synthesis_failed": synthesis_failed,
        "clusters_likely_false_positives": false_positive,
        "clusters_by_primary_error_type": clusters_by_primary,
        "clusters_by_confidence": clusters_by_confidence,
    }
