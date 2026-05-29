"""Per-value correction suggester (LLM K-shot self-consistency).

Shared between two callers:

* The offline benchmark exporter
  (:mod:`experiments.export_inconsistency_benchmark_with_llm`).
* The in-loop labelling UI: the per-cluster ``suggest-corrections``
  endpoint (sync) and the ``correction-jobs`` batch runner (async).

What this module owns:

* The system + user prompt (``SYSTEM_PROMPT`` + ``_USER_TEMPLATE``)
  and the prompt-version pin ``PROMPT_VERSION``.
* Lightweight dataclasses (``ClusterCtx``, ``CellGroup``,
  ``ValueSuggestion``).
* Prompt construction (``build_prompt``), LLM invocation
  (``call_llm``), JSON parsing (``parse_llm_list``), and the
  resumable JSONL cache (``cache_key`` / ``load_cache`` /
  ``append_cache``).
* The K-shot + retry self-consistency aggregation
  (``aggregate_samples``) and a high-level wrapper
  (``suggest_corrections_for_clusters``) that returns one
  :class:`ValueSuggestion` per (source, column, value) inside the
  requested cluster targets.

What this module does NOT own:

* The export-specific cluster-rescue / label-from-involved logic
  and the BenchmarkCell emission - those stay in
  :mod:`experiments.export_inconsistency_benchmark_with_llm`.
* The label persistence path (the live endpoint writes back through
  the existing ``LabelStore`` via the ``post_label`` route).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import pandas as pd

from src.core.data_loader import load_csv

from src.core.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Prompt + heuristic constants
# ---------------------------------------------------------------------

# Bumped whenever the system/user prompt changes in a way that would
# invalidate cached LLM answers. Mixed into ``cache_key`` so a cache
# from an older prompt is not silently reused.
PROMPT_VERSION = 3

SYSTEM_PROMPT = (
    "You are a data-quality assistant. Given a CELL VALUE in a CSV column "
    "and one or more CLUSTERS that group it with other values (typo variants, "
    "formatting variants, etc.), output the COMPLETE set of canonical surface "
    "forms this cell could plausibly be normalized to.\n\n"
    "Decision rules:\n"
    "1. If the cell's CURRENT value is ALREADY a valid canonical form for "
    "this column, INCLUDE it in the list - even when it is the only entry. "
    "This includes blank/empty cells when blank is itself an acceptable "
    "canonical (e.g. the column legitimately has missing values).\n"
    "2. If the column legitimately mixes MULTIPLE canonical forms that are "
    "all acceptable (e.g. 'NY' AND 'New York' both appear as valid "
    "abbreviations), include ALL of them in the list - this is the "
    "bi-directional case and should be common when the cluster's verdict "
    "is 'unsure'.\n"
    "3. If the cluster looks like typos / malformations of a single "
    "underlying value, return just that one canonical.\n"
    "4. Prefer surface forms that already appear in the column context. "
    "Only invent a new form if no acceptable form is present.\n"
    "5. Use the cluster-kind hint and the SAMPLE ROWS to disambiguate: row "
    "neighbors (date, related identifiers) often tell you whether a value "
    "is a typo of its cluster-mates or a legitimate variant.\n"
    "6. If you genuinely cannot decide, output an empty list.\n\n"
    "Output ONLY a JSON object: "
    '{"possible_correct_values": [...]}.'
)


_USER_TEMPLATE = """File:    {source_json}
Column:  {column_json}
Cell value (verbatim): {value_json}

CLUSTERS THIS CELL BELONGS TO
(verdict legend: tp = annotator-confirmed inconsistency; unsure = annotator
uncertain; fp = annotator rejected the cluster; partial = annotator agrees
the cluster is wrong but description/fix is off; no verdict = not reviewed):

{cluster_blocks}

CLUSTER-KIND HINT:
{kind_hint}
{row_context_section}
ALL DISTINCT VALUES IN COLUMN {column_json} ({distinct_count} total{truncated_hint}):
{column_context}

Task: output the complete JSON list of canonical surface forms this cell could
plausibly be normalized to.
  * If the cell value is already canonical for this column, INCLUDE it in
    the list.
  * If multiple canonical surface forms coexist in this column, list ALL
    of them (bi-directional case).
  * Prefer values that already appear in the column above.
  * If you cannot decide, output an empty list.

Output JSON exactly in this shape, nothing else:
{{"possible_correct_values": ["...", "..."]}}
"""


# Effective cap on distinct values shown in the prompt. Our CSVs in
# this corpus top out at ~100 distinct values per column, so 500 means
# "show the entire column" in practice. The cap is still here as a
# safety net for unusually-wide columns elsewhere.
_COLUMN_CONTEXT_LIMIT = 500

# Cluster kinds that strongly imply a SINGLE canonical (formatting
# clusters where the majority-style form is the obvious target).
_SINGLE_CANONICAL_KINDS = frozenset({
    "numerical_format", "casing", "punctuation",
})
# Cluster kinds where multi-canonical is plausible (semantic content
# variations can legitimately coexist).
_MULTI_CANONICAL_KINDS = frozenset({
    "pattern", "text", "untagged",
})

# Verdicts considered "interesting" enough to materialise into the
# benchmark (TP-wins union rule). Re-exported for callers (the export
# script and the live batch endpoint both need this).
INTERESTING_STATUSES: Tuple[str, ...] = ("tp", "unsure")


# ---------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------


@dataclass
class ClusterCtx:
    """Frozen description of a column-cluster.

    Carries everything we need both for per-cell aggregation
    (value membership, kind, verdict) and for the LLM prompt
    (target_style, llm_canonical, confirmed_canonical).
    """

    source: str
    column: str
    local_id: str           # "c1", "c2", ...
    target_id: str          # workbench colclust\x00.. id
    global_id: str          # source::column::c{idx}
    kind: str
    values: List[str]
    value_set: FrozenSet[str]
    llm_canonical: Optional[str]
    target_style: str
    status: Optional[str]              # tp / unsure / fp / partial / skip / None
    confirmed_canonical: Optional[str] # only meaningful for tp / partial
    note: str
    annotator: Optional[str]
    # Operator-saved per-value corrections (only meaningful for
    # tp / unsure verdicts). Maps original value -> single canonical
    # (str) for one-directional fixes, or list of acceptable values
    # for bi-/multi-canonical cases. Absent keys = "drop the value
    # from the exported benchmark". When non-empty, the export
    # pipeline treats this as the operator's final word and skips the
    # LLM round-trip for any group whose value appears here.
    value_corrections: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CellGroup:
    """All cells (CSV rows) sharing one (source, column, value) tuple.

    The LLM is invoked once per group (per sample); the resulting
    consensus list is materialised across every row in ``rows``.
    """

    source: str
    column: str
    value: str
    rows: List[int] = field(default_factory=list)
    # Clusters in the same (source, column) that contain ``value`` --
    # all verdicts; the LLM sees this list, downstream label-union
    # logic only looks at TP / unsure entries.
    involved: List[ClusterCtx] = field(default_factory=list)


@dataclass
class ValueSuggestion:
    """One per-value LLM result (one per CellGroup)."""

    source: str
    column: str
    cluster_local_id: str       # cluster the suggestion was requested for
    value: str
    rows: List[int]
    suggestion: Optional[List[str]]   # None = no LLM consensus produced
    agreement: float                  # 0..1
    n_valid_samples: int
    n_total_samples: int
    samples: List[Optional[List[str]]]
    # True iff the consensus is exactly ``[value]`` -- the export
    # rule would normally drop this row from the benchmark (subject
    # to the cluster-rescue safety net).
    would_drop_per_export_rule: bool


# ---------------------------------------------------------------------
# target_id <-> tuple helpers
# ---------------------------------------------------------------------


def parse_colclust_target_id(tid: str) -> Optional[Tuple[str, str, str]]:
    """Parse a ``colclust\\x00<source>\\x00<column>\\x00<local_id>`` id.

    Returns ``(source, column, local_id)`` or ``None`` for anything
    that doesn't match the column-cluster shape.
    """

    parts = tid.split("\x00")
    if len(parts) != 4 or parts[0] != "colclust":
        return None
    return parts[1], parts[2], parts[3]


# ---------------------------------------------------------------------
# Drafts + labels loading (latest-wins per key)
# ---------------------------------------------------------------------


def load_labels(labels_path: Path) -> Dict[str, dict]:
    """Return ``{target_id: latest_record}`` for column-cluster verdicts.

    Latest-line-wins per ``target_id`` (mirrors :class:`LabelStore`).
    """

    latest: Dict[str, dict] = {}
    if not labels_path.exists():
        return latest
    with labels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("target_id")
            if isinstance(tid, str) and tid.startswith("colclust"):
                latest[tid] = rec
    return latest


def load_drafts(drafts_path: Path) -> Dict[Tuple[str, str], dict]:
    """Return ``{(source, column): latest_draft_doc}``.

    File format is one JSON line per (source, column). We keep the
    last occurrence on the rare chance the same key was written
    twice (e.g. concatenated logs).
    """

    out: Dict[Tuple[str, str], dict] = {}
    if not drafts_path.exists():
        return out
    with drafts_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = doc.get("source")
            c = doc.get("column")
            if not s or not c:
                continue
            out[(s, c)] = doc
    return out


def build_cluster_index(
    drafts: Dict[Tuple[str, str], dict],
    labels: Dict[str, dict],
) -> Dict[Tuple[str, str], List[ClusterCtx]]:
    """Materialise every draft cluster (any verdict) with its verdict
    looked up.

    Returns ``{(source, column): [ClusterCtx, ...]}`` ordered by the
    cluster's positional index in the draft document so ``local_id``
    matches ``c{idx}``.
    """

    out: Dict[Tuple[str, str], List[ClusterCtx]] = {}
    for (s, c), doc in drafts.items():
        clusters_raw = doc.get("draft_clusters") or []
        ctxs: List[ClusterCtx] = []
        for idx, cl in enumerate(clusters_raw, start=1):
            local_id = f"c{idx}"
            target_id = f"colclust\x00{s}\x00{c}\x00{local_id}"
            global_id = f"{s}::{c}::{local_id}"
            # Values may be a list of strings or a list of
            # ``{"value": ..., "count": ...}`` dicts; normalise.
            vals: List[str] = []
            for v in cl.get("values") or []:
                if isinstance(v, dict):
                    vv = v.get("value")
                else:
                    vv = v
                if isinstance(vv, str) and vv:
                    vals.append(vv)
            # Skip degenerate clusters.
            if len(vals) < 2:
                continue
            label_rec = labels.get(target_id)
            status = label_rec.get("status") if label_rec else None
            ctxs.append(ClusterCtx(
                source=s,
                column=c,
                local_id=local_id,
                target_id=target_id,
                global_id=global_id,
                kind=str(cl.get("kind") or "untagged"),
                values=vals,
                value_set=frozenset(vals),
                llm_canonical=(
                    cl.get("canonical") or cl.get("canonical_value")
                ),
                target_style=str(cl.get("target_style") or ""),
                status=status,
                confirmed_canonical=(
                    (label_rec or {}).get("confirmed_canonical")
                ),
                note=str((label_rec or {}).get("note") or ""),
                annotator=(
                    (label_rec or {}).get("annotator")
                ),
                value_corrections=dict(
                    (label_rec or {}).get("value_corrections") or {}
                ),
            ))
        if ctxs:
            out[(s, c)] = ctxs
    return out


def resolve_csv_map(run_dir: Path) -> Dict[str, Path]:
    """Read ``summary.json`` to map ``source`` (CSV basename) to its
    absolute path on disk.
    """

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = summary.get("config", {}).get("csv_paths") or []
    out: Dict[str, Path] = {}
    for p in paths:
        pp = Path(p)
        out[pp.name] = pp
    return out


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------


def format_cluster_block(ctx: ClusterCtx) -> str:
    verdict = ctx.status or "(unreviewed)"
    canon = ctx.confirmed_canonical or ctx.llm_canonical
    canon_line = (
        f"  canonical hint: {json.dumps(canon, ensure_ascii=False)}"
        if canon else ""
    )
    style_line = (
        f"  style: {ctx.target_style}" if ctx.target_style else ""
    )
    values_repr = json.dumps(sorted(ctx.values), ensure_ascii=False)
    lines = [
        f"- Cluster {ctx.local_id} "
        f"(kind={ctx.kind}, verdict={verdict})",
        f"  values: {values_repr}",
    ]
    if canon_line:
        lines.append(canon_line)
    if style_line:
        lines.append(style_line)
    return "\n".join(lines)


def cluster_kind_hint(involved: List[ClusterCtx]) -> str:
    """Return a short paragraph hinting the LLM at the expected
    cardinality of canonicals based on the involved cluster kinds.
    """

    kinds = sorted({(ctx.kind or "untagged") for ctx in involved})
    kinds_set = set(kinds)
    only_single = bool(kinds_set) and kinds_set.issubset(_SINGLE_CANONICAL_KINDS)
    has_multi = bool(kinds_set & _MULTI_CANONICAL_KINDS)
    if only_single:
        return (
            "Involved cluster kind(s): " + ", ".join(kinds) + ". These are "
            "FORMATTING clusters (number/casing/punctuation variants). "
            "Strongly prefer a SINGLE canonical form (the majority-style "
            "representation of the same underlying value). Only return "
            "multiple canonicals if you are highly confident the column "
            "legitimately mixes more than one acceptable format."
        )
    if has_multi:
        return (
            "Involved cluster kind(s): " + ", ".join(kinds) + ". These "
            "clusters can carry SEMANTIC content (e.g. 'Bradford' vs "
            "'Bradford (Forfeit)'). Inspect the column context and any "
            "row samples carefully: if the variants encode genuinely "
            "different meanings or both forms appear consistently, "
            "return MULTIPLE canonicals (bi-directional). If they look "
            "like a single underlying value rendered differently, "
            "return a single canonical."
        )
    return (
        "Involved cluster kind(s): " + ", ".join(kinds) + ". "
        "Use the column context and sample rows to decide between a "
        "single canonical and a multi-canonical list."
    )


def format_row_context(
    df: pd.DataFrame,
    group: CellGroup,
    *,
    max_rows: int,
    max_col_value_chars: int = 60,
) -> str:
    """Format up to ``max_rows`` sample CSV rows containing this cell
    value, showing all sibling columns truncated. Returns "" if
    ``max_rows`` is 0 or there is nothing to show.
    """

    if max_rows <= 0 or not group.rows:
        return ""
    sample_rows = group.rows[:max_rows]
    lines: List[str] = []
    for r in sample_rows:
        try:
            rec = df.iloc[r]
        except (IndexError, KeyError):
            continue
        parts: List[str] = []
        for col, val in rec.items():
            sval = "" if val is None else str(val)
            if len(sval) > max_col_value_chars:
                sval = sval[: max_col_value_chars - 3] + "..."
            parts.append(
                f"{col}={json.dumps(sval, ensure_ascii=False)}"
            )
        lines.append(f"  row {int(r)}: " + ", ".join(parts))
    if len(group.rows) > max_rows:
        lines.append(
            f"  (... {len(group.rows) - max_rows} additional matching "
            f"row(s) omitted; total = {len(group.rows)})"
        )
    return "\n".join(lines)


def format_column_context(
    df: pd.DataFrame, column: str, *, limit: int = _COLUMN_CONTEXT_LIMIT,
) -> Tuple[str, int, bool]:
    """Return (formatted-block, total-distinct, truncated)."""

    series = df[column].astype(str)
    vc = series.value_counts(dropna=False)
    total = len(vc)
    truncated = total > limit
    rows = []
    for v, n in vc.head(limit).items():
        rows.append(f"  - {json.dumps(v, ensure_ascii=False)} (count={int(n)})")
    block = "\n".join(rows) if rows else "  (column is empty)"
    return block, total, truncated


def build_prompt(
    group: CellGroup,
    df: pd.DataFrame,
    *,
    row_context_rows: int,
) -> str:
    blocks = "\n".join(
        format_cluster_block(c) for c in group.involved
    )
    col_ctx, distinct_count, truncated = format_column_context(df, group.column)
    kind_hint = cluster_kind_hint(group.involved)
    row_ctx = format_row_context(
        df, group, max_rows=row_context_rows,
    )
    row_context_section = (
        f"\nSAMPLE ROWS CONTAINING THIS CELL VALUE "
        f"(showing up to {row_context_rows} of {len(group.rows)} matching row(s)):\n"
        f"{row_ctx}\n"
        if row_ctx else ""
    )
    return _USER_TEMPLATE.format(
        source_json=json.dumps(group.source),
        column_json=json.dumps(group.column),
        value_json=json.dumps(group.value),
        cluster_blocks=blocks,
        kind_hint=kind_hint,
        row_context_section=row_context_section,
        distinct_count=distinct_count,
        truncated_hint=(
            f", showing top {_COLUMN_CONTEXT_LIMIT} by frequency"
            if truncated else ""
        ),
        column_context=col_ctx,
    )


# ---------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_list(text: str) -> Optional[List[str]]:
    if not text:
        return None
    t = text.strip()
    fence = _CODE_FENCE_RE.search(t)
    if fence:
        t = fence.group(1).strip()
    if not t.startswith("{"):
        m = _JSON_OBJ_RE.search(t)
        if m:
            t = m.group(0)
    try:
        payload = json.loads(t)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("possible_correct_values")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    out: List[str] = []
    seen: set = set()
    for x in raw:
        if isinstance(x, str):
            x = x.strip()
            if x and x not in seen:
                seen.add(x)
                out.append(x)
    return out


# ---------------------------------------------------------------------
# Cache + LLM call
# ---------------------------------------------------------------------


def cache_key(
    *,
    prompt: str,
    system_prompt: str,
    model: str,
    reasoning_effort: str,
    sample_idx: int,
) -> str:
    """Stable cache key. Different prompts, system messages, models,
    reasoning settings, or sample indices => different cache keys.

    Hashing the actually-rendered prompt (not just the structured
    cluster payload) means any future change to ``_USER_TEMPLATE``,
    the column-context format, row-context inclusion, or kind-hint
    logic invalidates the cache automatically without needing to
    bump :data:`PROMPT_VERSION`. ``PROMPT_VERSION`` is still mixed
    in as a manual override / human-readable version pin.
    """

    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "sample_idx": sample_idx,
        "system": system_prompt,
        "prompt": prompt,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("key")
            if isinstance(key, str):
                out[key] = rec
    return out


def append_cache(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False))
        fh.write("\n")


def call_llm(
    llm: LLMClient,
    *,
    model: str,
    prompt: str,
    reasoning_effort: str,
) -> Tuple[Optional[List[str]], str, Optional[str]]:
    """Returns (possible_values_or_None, raw_text, error_or_None)."""

    try:
        resp = llm.query(
            model,
            prompt,
            reasoning_effort=reasoning_effort,
            reasoning_summary="auto",
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001
        return None, "", f"call_failed: {exc}"
    raw = resp.text or ""
    return parse_llm_list(raw), raw, None


# ---------------------------------------------------------------------
# Sample aggregation (K-shot self-consistency)
# ---------------------------------------------------------------------


def aggregate_samples(
    samples: List[Optional[List[str]]],
    *,
    threshold_frac: float,
) -> Tuple[Optional[List[str]], float, int]:
    """Vote across LLM samples to derive a consensus canonical list.

    Returns ``(consensus_or_none, agreement_score, n_valid_samples)``.

    Strategy:
      * Drop ``None`` samples (LLM error / parse failure).
      * For each canonical surface form, count how many of the
        remaining samples included it.
      * Keep only those with vote count >= ceil(threshold_frac *
        n_valid_samples), preserving order by frequency-then-value.
      * If the threshold rejects every candidate, fall back to the
        FIRST non-empty sample.
      * ``agreement_score`` = fraction of valid samples whose
        canonical set equals the final consensus set.
    """

    valid = [s for s in samples if s is not None]
    n_valid = len(valid)
    if n_valid == 0:
        return None, 0.0, 0
    threshold = max(1, math.ceil(threshold_frac * n_valid))
    votes: Counter = Counter()
    for sample in valid:
        for v in set(sample):
            votes[v] += 1
    consensus = [
        v for v, _ in sorted(
            votes.items(), key=lambda kv: (-kv[1], kv[0])
        )
        if votes[v] >= threshold
    ]
    if not consensus:
        # Fall back to first non-empty sample so we don't silently
        # drop information when samples disagree completely.
        for s in valid:
            if s:
                consensus = sorted(set(s))
                break
    consensus_set = frozenset(consensus)
    matches = sum(1 for s in valid if frozenset(s) == consensus_set)
    agreement = matches / n_valid if n_valid else 0.0
    return (consensus if consensus else None), agreement, n_valid


# ---------------------------------------------------------------------
# Group dispatch (K + retry, cache-aware)
# ---------------------------------------------------------------------


def dispatch_groups(
    *,
    groups: List[CellGroup],
    csv_cache: Dict[str, pd.DataFrame],
    llm_client: Optional[LLMClient],
    cache_path: Path,
    cache: Dict[str, Dict[str, Any]],
    model: str,
    reasoning_effort: str,
    samples: int,
    samples_threshold: float,
    retry_samples: int,
    row_context_rows: int,
    concurrency: int,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
    stats: Optional[Counter] = None,
    dry_run: bool = False,
) -> List[List[Optional[List[str]]]]:
    """Run K + retry self-consistency for ``groups``.

    Returns ``results_by_idx``: a list parallel to ``groups`` whose
    i-th entry is the per-sample parsed canonical list (length
    ``samples`` initially, extended to ``samples + retry_samples``
    for groups that consensus-agreed with the original value on the
    first pass).

    The cache is consulted first (and updated as new samples arrive)
    so the same call resumes safely across runs and processes.

    ``progress_cb(phase_label, done, total)`` is invoked whenever
    progress changes; pass it from the live endpoint to surface
    "5 of 23 groups done" hints to the UI. ``None`` = silent.
    """

    if stats is None:
        stats = Counter()

    n_samples = max(1, int(samples))
    n_retry = max(0, int(retry_samples))

    # Pre-build prompts so the worker doesn't need to touch the
    # dataframe (we keep the ThreadPoolExecutor out of pandas).
    prompts: List[Optional[str]] = []
    for grp in groups:
        df = csv_cache.get(grp.source)
        if df is None:
            prompts.append(None)
            continue
        prompts.append(build_prompt(
            grp, df, row_context_rows=row_context_rows,
        ))

    results_by_idx: List[List[Optional[List[str]]]] = [
        [None] * n_samples for _ in groups
    ]

    def _build_pending(
        s_start: int, s_end: int,
        *, group_indices: Optional[List[int]] = None,
    ) -> List[Tuple[int, int, CellGroup, str, str]]:
        out: List[Tuple[int, int, CellGroup, str, str]] = []
        idx_iter = (
            group_indices
            if group_indices is not None
            else range(len(groups))
        )
        for i in idx_iter:
            prompt = prompts[i]
            if prompt is None:
                continue
            grp = groups[i]
            for s_idx in range(s_start, s_end):
                key = cache_key(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    sample_idx=s_idx,
                )
                if key in cache:
                    cached = cache[key].get("possible_correct_values")
                    if isinstance(cached, list):
                        results_by_idx[i][s_idx] = [
                            x for x in cached if isinstance(x, str)
                        ]
                    else:
                        results_by_idx[i][s_idx] = None
                    stats["cache_hit"] += 1
                    continue
                if dry_run:
                    results_by_idx[i][s_idx] = None
                    continue
                out.append((i, s_idx, grp, prompt, key))
        return out

    def _dispatch(
        pending: List[Tuple[int, int, CellGroup, str, str]],
        *, phase_label: str,
    ) -> None:
        if not pending or llm_client is None:
            return

        def _work(item: Tuple[int, int, CellGroup, str, str]):
            i, s_idx, grp, prompt, key = item
            values, raw, err = call_llm(
                llm_client, model=model, prompt=prompt,
                reasoning_effort=reasoning_effort,
            )
            return i, s_idx, grp, key, values, raw, err

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_work, it) for it in pending]
            done_n = 0
            total = len(pending)
            for fut in as_completed(futures):
                i, s_idx, grp, key, values, raw, err = fut.result()
                done_n += 1
                if err:
                    logger.warning(
                        "[%s %d/%d] LLM failed for %s/%s value=%r "
                        "sample=%d: %s",
                        phase_label, done_n, total,
                        grp.source, grp.column, grp.value, s_idx, err,
                    )
                    stats["llm_error"] += 1
                    results_by_idx[i][s_idx] = None
                else:
                    if values is None:
                        stats["llm_parse_failed"] += 1
                    else:
                        stats["llm_ok"] += 1
                    results_by_idx[i][s_idx] = values
                    append_cache(cache_path, {
                        "key": key,
                        "source": grp.source,
                        "column": grp.column,
                        "value": grp.value,
                        "sample_idx": s_idx,
                        "possible_correct_values": values,
                        "raw_response": raw,
                        "model": model,
                        "ts": _dt.datetime.now(
                            _dt.timezone.utc
                        ).isoformat(),
                    })
                if progress_cb is not None:
                    try:
                        progress_cb(phase_label, done_n, total)
                    except Exception:  # noqa: BLE001
                        pass
                if done_n % 20 == 0 or done_n == total:
                    logger.info(
                        "[%s %d/%d] LLM progress "
                        "(ok=%d parse_fail=%d err=%d)",
                        phase_label, done_n, total,
                        stats["llm_ok"], stats["llm_parse_failed"],
                        stats["llm_error"],
                    )

    # --- Phase A: initial K samples ----------------------------------
    pending = _build_pending(0, n_samples)
    logger.info(
        "[phase A] dispatching %d LLM call(s) across %d group(s) x %d "
        "sample(s); %d served from cache; dry_run=%s",
        len(pending), len(groups), n_samples,
        stats["cache_hit"], dry_run,
    )
    _dispatch(pending, phase_label="phase A")

    # --- Phase B: retry for groups whose consensus is [original] -----
    if n_retry > 0:
        retry_indices: List[int] = []
        for i in range(len(groups)):
            prompt = prompts[i]
            if prompt is None:
                continue
            sample_lists = results_by_idx[i]
            consensus, _agreement, n_valid = aggregate_samples(
                sample_lists, threshold_frac=samples_threshold,
            )
            if (
                n_valid > 0
                and consensus is not None
                and len(consensus) == 1
                and consensus[0] == groups[i].value
            ):
                retry_indices.append(i)
        if retry_indices:
            for i in retry_indices:
                results_by_idx[i].extend([None] * n_retry)
            retry_pending = _build_pending(
                n_samples, n_samples + n_retry,
                group_indices=retry_indices,
            )
            logger.info(
                "[phase B] %d group(s) flagged for retry; dispatching "
                "%d more LLM call(s) (%d additional sample(s) each)",
                len(retry_indices), len(retry_pending), n_retry,
            )
            _dispatch(retry_pending, phase_label="phase B")
            stats["groups_retried"] = len(retry_indices)
        else:
            logger.info(
                "[phase B] no groups needed retry "
                "(consensus already proposes a non-original canonical "
                "for every group with valid samples)"
            )

    return results_by_idx


# ---------------------------------------------------------------------
# CellGroup builder (cluster-scoped variant for the live endpoint)
# ---------------------------------------------------------------------


def build_cell_groups_for_clusters(
    *,
    cluster_index: Dict[Tuple[str, str], List[ClusterCtx]],
    csv_map: Dict[str, Path],
    csv_cache: Dict[str, pd.DataFrame],
    targets: List[Tuple[str, str, str]],
    stats: Optional[Counter] = None,
) -> List[CellGroup]:
    """Build CellGroups restricted to the values in the requested
    cluster targets.

    For each ``(source, column, cluster_local_id)`` in ``targets`` we
    enumerate the cluster's values, find their CSV rows, and emit one
    :class:`CellGroup` per (source, column, value). The group's
    ``involved`` list is the FULL set of clusters in the same
    (source, column) that contain the value -- so the prompt sees all
    overlapping cluster context, not just the requested cluster.
    """

    if stats is None:
        stats = Counter()
    by_sc: Dict[Tuple[str, str], List[str]] = {}
    for s, c, lid in targets:
        by_sc.setdefault((s, c), []).append(lid)

    groups: List[CellGroup] = []
    for (s, c), local_ids in sorted(by_sc.items()):
        ctxs = cluster_index.get((s, c), [])
        wanted_ids = set(local_ids)
        wanted_clusters = [ctx for ctx in ctxs if ctx.local_id in wanted_ids]
        if not wanted_clusters:
            stats["skip_cluster_not_found"] += len(local_ids)
            continue
        path = csv_map.get(s)
        if path is None:
            logger.warning("No CSV path for source %s; skipping.", s)
            stats["skip_missing_csv"] += 1
            continue
        if s not in csv_cache:
            try:
                df = load_csv(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load CSV %s: %s", path, exc)
                stats["skip_csv_load_error"] += 1
                continue
            csv_cache[s] = df
        df = csv_cache[s]
        if c not in df.columns:
            logger.warning("Column %r not in %s; skipping.", c, s)
            stats["skip_missing_column"] += 1
            continue
        col_series = df[c].astype(str)
        # Union of values across the requested clusters.
        wanted_values: set = set()
        for ctx in wanted_clusters:
            wanted_values.update(ctx.values)
        for v in sorted(wanted_values):
            mask = col_series == v
            row_idx = [int(i) for i in col_series.index[mask]]
            if not row_idx:
                stats["empty_value_match"] += 1
                continue
            # The prompt sees ALL clusters in (s, c) that contain v,
            # not only the requested ones, so context stays
            # consistent with the export-script path.
            involved = [ctx for ctx in ctxs if v in ctx.value_set]
            if not involved:
                stats["empty_involved"] += 1
                continue
            groups.append(CellGroup(
                source=s,
                column=c,
                value=v,
                rows=row_idx,
                involved=involved,
            ))
            stats["cell_groups"] += 1
            stats["matched_rows"] += len(row_idx)
    return groups


# ---------------------------------------------------------------------
# High-level wrapper for the live UI
# ---------------------------------------------------------------------


def suggest_corrections_for_clusters(
    run_dir: Path,
    *,
    targets: List[Tuple[str, str, str]],
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-5.2",
    reasoning_effort: str = "medium",
    samples: int = 3,
    samples_threshold: float = 0.5,
    retry_samples: int = 4,
    row_context_rows: int = 3,
    cache_path: Optional[Path] = None,
    csv_cache: Optional[Dict[str, pd.DataFrame]] = None,
    concurrency: int = 8,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
    dry_run: bool = False,
) -> Dict[Tuple[str, str, str], List[ValueSuggestion]]:
    """Run K + retry self-consistency for the requested cluster targets.

    Returns ``{(source, column, cluster_local_id): [ValueSuggestion, ...]}``.
    Every requested target is present in the result; targets that
    couldn't be matched (no draft cluster, missing CSV, etc.) map to
    an empty list.

    The cache file (default ``<run_dir>/inconsistency_llm_cache.jsonl``)
    is shared with the offline export script, so the live endpoint
    reuses anything already paid for.

    Pass ``llm_client=None`` together with ``dry_run=True`` to dry-run
    (returns whatever the cache holds; missing entries become
    ``suggestion=None``).
    """

    cache_path = cache_path or (run_dir / "inconsistency_llm_cache.jsonl")
    if csv_cache is None:
        csv_cache = {}

    labels = load_labels(run_dir / "labels.jsonl")
    drafts = load_drafts(run_dir / "column_cluster_drafts.jsonl")
    cluster_index = build_cluster_index(drafts, labels)
    csv_map = resolve_csv_map(run_dir)

    stats: Counter = Counter()
    groups = build_cell_groups_for_clusters(
        cluster_index=cluster_index,
        csv_map=csv_map,
        csv_cache=csv_cache,
        targets=targets,
        stats=stats,
    )
    cache = load_cache(cache_path)
    results_by_idx = dispatch_groups(
        groups=groups,
        csv_cache=csv_cache,
        llm_client=llm_client,
        cache_path=cache_path,
        cache=cache,
        model=model,
        reasoning_effort=reasoning_effort,
        samples=samples,
        samples_threshold=samples_threshold,
        retry_samples=retry_samples,
        row_context_rows=row_context_rows,
        concurrency=concurrency,
        progress_cb=progress_cb,
        stats=stats,
        dry_run=dry_run,
    )

    # Assemble per-target suggestion lists.
    out: Dict[Tuple[str, str, str], List[ValueSuggestion]] = {
        t: [] for t in targets
    }
    # Every group can map to multiple targets (a value can sit in
    # several requested clusters). The export script's per-cluster
    # rescue logic doesn't apply here -- the live endpoint surfaces
    # raw suggestions and lets the operator decide.
    target_lookup_by_sc: Dict[Tuple[str, str], List[str]] = {}
    for s, c, lid in targets:
        target_lookup_by_sc.setdefault((s, c), []).append(lid)

    for i, grp in enumerate(groups):
        sample_lists = results_by_idx[i]
        consensus, agreement, n_valid = aggregate_samples(
            sample_lists, threshold_frac=samples_threshold,
        )
        n_total = len(sample_lists)
        would_drop = bool(
            consensus is not None
            and len(consensus) == 1
            and consensus[0] == grp.value
        )
        # Fan out to every requested cluster in this (source, column)
        # whose value-set contains this group's value.
        wanted_lids = target_lookup_by_sc.get((grp.source, grp.column), [])
        for ctx in grp.involved:
            if ctx.local_id not in wanted_lids:
                continue
            key = (grp.source, grp.column, ctx.local_id)
            sugg = ValueSuggestion(
                source=grp.source,
                column=grp.column,
                cluster_local_id=ctx.local_id,
                value=grp.value,
                rows=list(grp.rows),
                suggestion=consensus,
                agreement=agreement,
                n_valid_samples=n_valid,
                n_total_samples=n_total,
                samples=list(sample_lists),
                would_drop_per_export_rule=would_drop,
            )
            out[key].append(sugg)

    return out


# Re-export private names that the export script still references by
# their leading-underscore aliases. Keeps the export script's hash key
# byte-identical to historical caches without having to rename a long
# list of imports inside that file.
_PROMPT_VERSION = PROMPT_VERSION
_SYSTEM_PROMPT = SYSTEM_PROMPT
_INTERESTING_STATUSES = INTERESTING_STATUSES


def suggestions_to_payload(
    suggestions: List["ValueSuggestion"],
) -> List[Dict[str, Any]]:
    """Trim ``ValueSuggestion`` records to the shape the labelling UI wants.

    Hides the per-sample raw lists (heavy, diagnostic-only) and keeps the
    consensus + agreement + provenance hints needed to render the table.
    Shared by the clusters router and the correction-job worker.
    """
    out: List[Dict[str, Any]] = []
    for s in suggestions:
        out.append({
            "value": s.value,
            "suggestion": s.suggestion,
            "agreement": s.agreement,
            "n_valid_samples": s.n_valid_samples,
            "n_total_samples": s.n_total_samples,
            "would_drop_per_export_rule": s.would_drop_per_export_rule,
            "rows": s.rows,
        })
    return out
