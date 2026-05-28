"""LLM-based pass-rate estimator for relations the code-verifier refused.

The code-path validator (:mod:`.relation_validator`) lets the LLM decline
to write ``check(df)`` when the rule depends on world-knowledge, free-text
semantics, or anything else that does not have a clean deterministic
predicate. Those rules carry ``verifiable=False`` and previously had no
pass-rate signal at all -- the downstream filter gate could not tell a
"false rule" (majority violate) from a "real rule" (a handful violate).

This module is the LLM fallback for that gap. For each ``verifiable=False``
relation it walks the **entire** source frame in fixed-size chunks, asking
the LLM "how many rows satisfy / violate the rule" on each chunk, and
aggregates the counts into a single pass-rate. The result is wrapped in
the same :class:`~.relation_validator.RelationVerification` record shape
the code path produces -- with ``verifier_kind="llm_sample"`` retained
for back-compat (callers branch on this label, see
:mod:`.relation_violations` and :mod:`.relation_llm_detector`) -- so the
filter gate in :mod:`.relation_violations` can apply uniform pass-rate /
no-violation criteria across both verification methods.

Previously this estimator drew one ~30-row random sample and asked the
LLM about that. Sampling produced unreliable pass-rates on tables with
rare anomalies (a 2-row typo in a 1000-row table is invisible to a 30-row
sample), so we now cover the full frame in chunks of ``sample_size``
rows. Cost scales linearly with table height but only for the (typically
small) fraction of relations the code-verifier refused.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import yaml

from ..jsonl_io import extract_json_payloads as _extract_json_payloads
from ..llm_client import LLMClient
from .validator import RelationVerification

logger = logging.getLogger(__name__)


def load_verifier_prompt(path: Path) -> Dict[str, Any]:
    """Load ``prompts/verify_relation.yaml``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"verify-relation prompt yaml must be a mapping: {path}")
    return raw


def _render_table_md(df: pd.DataFrame, *, max_rows: int) -> str:
    """Render a sampled frame as a Markdown table with a leading
    ``__row__`` column carrying the original integer index.

    Falls back to CSV when ``tabulate`` is not installed -- same
    convention as :mod:`.relation_llm_detector`.
    """
    if df.empty:
        return "(empty table)"
    capped = df.head(max_rows)
    work = capped.astype(object).copy()
    work.insert(0, "__row__", list(work.index))
    try:
        return work.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        return work.to_csv(index=False)


def _iter_chunks(df: pd.DataFrame, *, chunk_size: int):
    """Yield contiguous row chunks of size ``chunk_size`` covering the
    full frame in order. Preserves original row indices via ``df.iloc``
    slicing so the LLM sees the same ``__row__`` labels we use downstream.
    """
    if df.empty or chunk_size <= 0:
        return
    n = len(df)
    for start in range(0, n, chunk_size):
        yield df.iloc[start:start + chunk_size]


def _build_user_prompt(
    template: str,
    *,
    relation: Mapping[str, Any],
    refusal_reason: str,
    table_md: str,
    n_rows: int,
) -> str:
    return template.format(
        source=relation.get("source", "") or "",
        error_type=relation.get("error_type", "") or "",
        kind=relation.get("kind", "") or "",
        columns=", ".join(relation.get("columns") or []) or "(unspecified)",
        predicate=relation.get("predicate", "") or "(no predicate)",
        rationale=relation.get("rationale", "") or "(no rationale)",
        confidence=relation.get("confidence", "") or "unknown",
        refusal_reason=refusal_reason or "(none recorded)",
        table_md=table_md,
        n_rows=n_rows,
    )


def _parse_verifier_response(text: str) -> Dict[str, Any]:
    """Parse the LLM response into ``{applicable, n_evaluated, n_violating, rationale}``.

    Tolerates surrounding markdown / fences -- same scan logic as
    :mod:`.relation_validator._parse_llm_decision`.
    """
    if not text:
        raise ValueError("empty LLM response")
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        nl = body.find("\n")
        if nl > 0 and body[:nl].strip().isalpha():
            body = body[nl + 1:]
    obj: Optional[Dict[str, Any]] = None
    for candidate in (body.strip(), *_extract_json_payloads(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            obj = parsed
            break
    if obj is None:
        raise ValueError("no JSON object found in LLM response")
    applicable = bool(obj.get("applicable", True))
    try:
        n_evaluated = int(obj.get("n_evaluated") or 0)
    except (TypeError, ValueError):
        raise ValueError("n_evaluated is not an integer")
    try:
        n_violating = int(obj.get("n_violating") or 0)
    except (TypeError, ValueError):
        raise ValueError("n_violating is not an integer")
    if n_violating < 0 or n_evaluated < 0:
        raise ValueError("counts must be non-negative")
    if n_violating > n_evaluated:
        raise ValueError(
            f"n_violating={n_violating} > n_evaluated={n_evaluated}"
        )
    return {
        "applicable": applicable,
        "n_evaluated": n_evaluated,
        "n_violating": n_violating,
        "rationale": str(obj.get("rationale") or "").strip(),
    }


def verify_relation_llm(
    relation: Mapping[str, Any],
    df: pd.DataFrame,
    *,
    refusal_reason: str,
    llm: LLMClient,
    model: str,
    prompt_cfg: Mapping[str, Any],
    sample_size: int = 30,
    sample_seed: int = 1009,  # accepted for back-compat; unused.
    max_chunks: int = 100,
) -> RelationVerification:
    """Estimate the pass-rate of ``relation`` over the FULL frame by
    chunking ``df`` into ``sample_size``-row blocks and asking the LLM
    to count satisfiers / violators on each chunk.

    Replaces an older single-sample estimator; the chunked version eliminates
    the dependence on which 30 rows happened to be drawn, which mattered
    for tables with rare anomalies (e.g. two typo rows in a thousand-row
    file). Aggregates ``n_evaluated`` and ``n_violating`` over all chunks;
    ``pass_rate = sum(n_passing) / sum(n_evaluated)``.

    Returns a :class:`RelationVerification` with ``verifier_kind="llm_sample"``
    (label retained for back-compat -- downstream callers in
    :mod:`.relation_violations` and :mod:`.relation_llm_detector` branch
    on it). On any unrecoverable failure (no referenced columns,
    first-chunk LLM error, first-chunk "not applicable", every chunk
    returns n_evaluated=0) the record carries ``pass_rate=None`` and an
    ``error`` string; callers treat those as "no signal" and skip the
    rule rather than fabricating a pass-rate.

    Cost: O(ceil(len(df) / sample_size)) LLM calls per relation, capped
    at ``max_chunks`` (a soft brake against runaway cost on very large
    tables; remaining rows are simply not evaluated and the resulting
    pass_rate is computed over what was covered).
    """
    rec = RelationVerification(
        species_key=str(relation.get("species_key") or ""),
        source=str(relation.get("source") or ""),
        error_type=str(relation.get("error_type") or ""),
        kind=str(relation.get("kind") or ""),
        verifiable=False,
        reason=refusal_reason or "",
        verifier_kind="llm_sample",
    )

    columns = tuple(relation.get("columns") or [])
    if columns and not any(c in df.columns for c in columns):
        rec.error = "no referenced columns present in source"
        return rec
    if df.empty:
        rec.error = "source frame is empty"
        return rec

    system_prompt = prompt_cfg.get("system_prompt") or ""
    reasoning_effort = prompt_cfg.get("reasoning_effort") or "medium"
    user_template = prompt_cfg.get("user_template") or ""

    total_evaluated = 0
    total_violating = 0
    total_attempts = 0
    rationales: List[str] = []
    chunks_run = 0
    chunks_errored = 0
    first_chunk_applicable: Optional[bool] = None

    chunks = list(_iter_chunks(df, chunk_size=int(sample_size)))
    n_chunks_total = len(chunks)
    chunks = chunks[: int(max_chunks)]

    for idx, chunk in enumerate(chunks):
        table_md = _render_table_md(chunk, max_rows=int(sample_size))
        user_prompt = _build_user_prompt(
            user_template,
            relation=relation,
            refusal_reason=refusal_reason,
            table_md=table_md,
            n_rows=int(len(chunk)),
        )
        try:
            resp = llm.query(
                model=model,
                prompt=user_prompt,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            chunks_errored += 1
            if idx == 0:
                rec.error = f"llm-error (first chunk): {type(exc).__name__}: {exc}"
                return rec
            # Mid-run failure: skip this chunk but keep the totals from
            # the rest of the table.
            logger.warning(
                "verifier chunk %d/%d failed (relation=%s): %s",
                idx + 1, len(chunks),
                relation.get("species_key", "?"), exc,
            )
            continue
        total_attempts += getattr(resp, "attempts", 0) or 0

        try:
            decision = _parse_verifier_response(resp.text)
        except ValueError as exc:
            chunks_errored += 1
            if idx == 0:
                rec.error = f"parse-error (first chunk): {exc}"
                return rec
            logger.warning(
                "verifier chunk %d/%d parse failure (relation=%s): %s",
                idx + 1, len(chunks),
                relation.get("species_key", "?"), exc,
            )
            continue

        applicable = bool(decision.get("applicable"))
        if idx == 0:
            first_chunk_applicable = applicable
            if not applicable:
                # Hard stop: if the first chunk says the rule doesn't
                # engage with this table at all, we trust that signal
                # rather than burning LLM calls on the rest.
                rec.error = (
                    f"not-applicable: {decision.get('rationale') or 'unspecified'}"
                )
                return rec

        chunks_run += 1
        n_eval = int(decision.get("n_evaluated") or 0)
        n_vio = int(decision.get("n_violating") or 0)
        total_evaluated += n_eval
        total_violating += n_vio
        rat = (decision.get("rationale") or "").strip()
        if rat and len(rationales) < 3:
            rationales.append(rat)

    rec.llm_attempts = total_attempts

    if total_evaluated == 0:
        rec.error = (
            f"n_evaluated=0 across {chunks_run} chunk(s) "
            f"(rule never engaged on this table)"
        )
        return rec

    rec.verifiable = True
    rec.n_evaluated = total_evaluated
    rec.n_passing = total_evaluated - total_violating
    rec.n_violating = total_violating
    rec.pass_rate = rec.n_passing / total_evaluated
    coverage_note = (
        f"covered {chunks_run}/{n_chunks_total} chunks"
        if chunks_run < n_chunks_total or chunks_errored
        else f"covered all {chunks_run} chunks"
    )
    rec.reason = (
        f"{refusal_reason or 'code-verifier refused'}; "
        f"llm_sample (full-table, {coverage_note}): "
        f"{' | '.join(rationales) or '(no rationale)'}"
    )
    return rec


def verify_relations_llm(
    relations: Sequence[Mapping[str, Any]],
    verifications: Mapping[str, Mapping[str, Any]],
    source_frames: Mapping[str, pd.DataFrame],
    *,
    llm: LLMClient,
    model: str,
    prompt_cfg: Mapping[str, Any],
    sample_size: int = 30,
    sample_seed: int = 1009,
) -> List[RelationVerification]:
    """Walk every relation whose code-verification carries ``verifiable=False``
    and emit an LLM-based pass-rate estimate.

    Skips:
      * relations the code-verifier never reached (no verification record)
      * relations the code-verifier marked as a hard error (the LLM
        produced malformed code; we have no reason to re-trust it on
        the row-count task either)
      * multi-table relations (handled by ``multi_table`` pipeline)
      * relations whose ``source`` we have no DataFrame for
    """
    out: List[RelationVerification] = []
    for rel in relations:
        species_key = str(rel.get("species_key") or "")
        v = verifications.get(species_key)
        if not v:
            continue
        if v.get("verifiable"):
            # Code path already produced a pass_rate; do not overwrite.
            continue
        if v.get("error"):
            continue
        if rel.get("tables"):
            continue
        source = str(rel.get("source") or "")
        df = source_frames.get(source)
        if df is None:
            continue
        rec = verify_relation_llm(
            rel, df,
            refusal_reason=str(v.get("reason") or ""),
            llm=llm,
            model=model,
            prompt_cfg=prompt_cfg,
            sample_size=sample_size,
            sample_seed=sample_seed,
        )
        out.append(rec)
    logger.info(
        "LLM verifier produced %d pass-rate estimate(s) across %d non-code-verifiable rule(s)",
        sum(1 for r in out if r.pass_rate is not None),
        sum(1 for r in relations if not verifications.get(r.get("species_key", ""), {}).get("verifiable")),
    )
    return out
