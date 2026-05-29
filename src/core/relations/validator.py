"""LLM-driven empirical validation of discovered relations.

For each relation, ask the LLM to either:

  * declare the rule **not code-verifiable** (with a reason) - e.g.
    rules that depend on external knowledge ("matches the company's
    Q3 fiscal calendar"), free-text semantic agreement, or anything
    that cannot be expressed as a deterministic check over the table
    columns; OR
  * write a Python function ``check(df: pd.DataFrame) -> pd.Series``
    that returns a boolean Series (one entry per row, ``True`` =
    row satisfies the rule, ``False`` = row violates it).

The function is then run in a restricted sandbox (whitelisted module
namespace, no imports, no builtins beyond a small safe set, wall-clock
timeout) against the source DataFrame; the validator records pass-rate,
sample violations, and any sandbox/runtime error.

Pass-rate is the *observed* fraction of rows for which ``check`` is
``True``. A reviewer reading the report can use it as a sanity signal:
a pass-rate of 0.0 means the rule is wrong (or the code is); a
pass-rate near 1.0 means the rule actually holds in the data and is
worth reviewing; the bucket in between (e.g. 0.85-0.97) is exactly
where data-quality issues hide.

Sandbox notes
-------------
The AST validator rejects:

  * any ``Import`` / ``ImportFrom`` node
  * any attribute access whose name starts with ``__`` or ends with
    ``__`` (blocks ``__class__``, ``__subclasses__``, etc.)
  * any direct reference to ``eval``, ``exec``, ``compile``, ``open``,
    ``__import__``, ``input``, ``breakpoint``
  * ``Global`` / ``Nonlocal`` (not needed for a one-function check)

The runtime ``__builtins__`` is replaced with a whitelist of pure,
deterministic helpers (``len``, ``range``, ``min``, ``max``, ``sum``,
``abs``, ``int``, ``float``, ``str``, ``bool``, ``isinstance``, etc.).
``pd``, ``np``, and ``re`` are pre-bound; nothing else is available.
"""

from __future__ import annotations

import concurrent.futures as _cf
import dataclasses
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from ..llm_client import LLMClient
from ._check_runner import run_check_subprocess
from ._sandbox import CodeValidationError, _validate_code_ast
# Re-exported so existing tests can keep importing the sandbox primitives
# from this module (their definitions moved to ._sandbox; the killable
# child in ._check_runner is what actually runs check()).
from ._sandbox import (  # noqa: F401
    _SAFE_BUILTINS,
    _coerce_to_bool_series,
    _exec_check_code,
)

logger = logging.getLogger("relation_validator")


# The sandbox primitives -- AST guard, restricted-builtins namespace,
# and return-value coercion -- now live in ._sandbox (imported and
# re-exported above). The LLM-emitted check() runs in a killable,
# resource-capped subprocess via ._check_runner.run_check_subprocess,
# replacing the old in-process exec on an uncancellable daemon thread.


# ---------------------------------------------------------------------------
# Public verification record
# ---------------------------------------------------------------------------


@dataclass
class RelationVerification:
    """Outcome of validating one relation against its source DataFrame."""

    species_key: str
    source: str
    error_type: str
    kind: str
    verifiable: bool
    reason: str = ""                       # filled when verifiable=False
    code: Optional[str] = None             # filled when verifiable=True
    pass_rate: Optional[float] = None      # in [0,1]; None on error
    n_evaluated: int = 0
    n_passing: int = 0
    n_violating: int = 0
    sample_violations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None            # sandbox / runtime error
    llm_attempts: int = 0
    layer: str = "originals"               # which layer this rule came from
    # How ``pass_rate`` was estimated. ``"code"`` means an LLM-written
    # ``check(df)`` ran in the sandbox (the legacy path). ``"llm_sample"``
    # means the LLM was asked to count violators on a row sample, used
    # for rules the code-verifier refused. ``"none"`` covers records
    # that never reached a verifier (parse error, no signal).
    verifier_kind: str = "code"
    # Internal: full ``(sheet, row)`` keys for every violating row, not
    # just the truncated preview in ``sample_violations``. Populated by
    # ``validate_one`` when the check ran successfully. Used by the
    # adaptive ensemble (:func:`validate_one_adaptive`) to feed
    # Good-Turing the complete violator species set; otherwise GT
    # would see at most ``max_violation_samples`` species per run and
    # systematically under-estimate the missing-mass. Not persisted to
    # JSON.
    violator_keys: List[Any] = field(default_factory=list, repr=False)

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        # Strip internal-only fields from the persisted view.
        d.pop("violator_keys", None)
        return d


# ---------------------------------------------------------------------------
# Prompt loading + parsing
# ---------------------------------------------------------------------------


def load_validation_prompt(path: Path) -> Dict[str, Any]:
    """Load the validation prompt YAML.

    Expected shape::

        model: gpt-5.2
        reasoning_effort: medium
        system_prompt: |
          ...
        user_template: |
          ...
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"validation prompt yaml must be a mapping: {path}")
    return raw


from ..jsonl_io import extract_json_payloads as _extract_json_payloads


def _parse_llm_decision(text: str) -> Dict[str, Any]:
    """Parse the LLM response. Tolerates surrounding markdown / fences.

    Returns a dict with at least ``verifiable: bool`` and one of
    ``reason`` / ``code``. Raises ``ValueError`` on malformed input.
    """
    if not text:
        raise ValueError("empty LLM response")
    body = text.strip()
    # Strip markdown fence if present.
    if body.startswith("```"):
        body = body.strip("`")
        # Drop the language tag line if any.
        first_newline = body.find("\n")
        if first_newline > 0 and body[:first_newline].strip().isalpha():
            body = body[first_newline + 1:]
    obj: Optional[Dict[str, Any]] = None
    # Try the cleaned body first (the common happy path: the agent
    # returned a single JSON object). Then fall back to a
    # bracket-balanced scan over the original text in case the agent
    # wrapped the JSON in prose.
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
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    if "verifiable" not in obj:
        raise ValueError("LLM response missing 'verifiable' field")
    return obj


def _build_user_prompt(
    template: str,
    *,
    relation: Mapping[str, Any],
    headers: Sequence[str],
    dtypes: Mapping[str, str],
    column_samples: Mapping[str, Sequence[Any]],
    sample_csv: str,
    n_rows: int,
    multi_table: bool = False,
    table_schemas: Optional[Mapping[str, Mapping[str, str]]] = None,
    table_samples: Optional[Mapping[str, Mapping[str, Sequence[Any]]]] = None,
    table_csvs: Optional[Mapping[str, str]] = None,
    table_row_counts: Optional[Mapping[str, int]] = None,
) -> str:
    if multi_table:
        # Render per-sheet schema + sample blocks. The aggregated
        # ``schema`` / ``column_samples`` / ``sample_csv`` placeholders
        # carry the same content but namespaced by sheet so the LLM
        # writes ``check(dfs)`` against the right frames.
        schema_lines: List[str] = []
        sample_lines: List[str] = []
        csv_blocks: List[str] = []
        meta_lines: List[str] = []
        for sheet, sheet_dtypes in (table_schemas or {}).items():
            schema_lines.append(f"  Sheet {sheet!r}:")
            for h, dt in sheet_dtypes.items():
                schema_lines.append(f"    - {h!r}: {dt}")
            samples_for_sheet = (table_samples or {}).get(sheet, {})
            if samples_for_sheet:
                sample_lines.append(f"  Sheet {sheet!r}:")
                for c, vals in samples_for_sheet.items():
                    sample_lines.append(f"    - {c!r}: {list(vals)!r}")
            sheet_csv = (table_csvs or {}).get(sheet, "")
            if sheet_csv:
                csv_blocks.append(f"=== sheet {sheet!r} ===\n{sheet_csv}")
            n = (table_row_counts or {}).get(sheet, 0)
            meta_lines.append(f"  - sheet {sheet!r}: {n} rows")
        schema_block = "\n".join(schema_lines) if schema_lines else "  (no sheets)"
        samples_block = "\n".join(sample_lines) if sample_lines else "  (no relevant columns)"
        sample_csv_block = "\n\n".join(csv_blocks) if csv_blocks else "(no sample rows)"
        metadata_block = "\n".join(meta_lines) if meta_lines else f"total rows: {n_rows}"
    else:
        # Pretty-print "header (dtype)" for every column so the LLM can
        # see whether ``Year`` is int64, ``Date`` is object/datetime, etc.
        schema_lines = [
            f"  - {h!r}: {dtypes.get(h, 'unknown')}"
            for h in headers
        ]
        schema_block = "\n".join(schema_lines) if schema_lines else "  (no columns)"

        # For the columns this relation actually references, show a few
        # raw cell values so the LLM sees their concrete shape (e.g.
        # "  'Year': [2018, 2019, 2020]  vs  'Year': ['2018', '2019']").
        rel_cols: List[str] = []
        for key in ("columns", "lhs", "rhs"):
            for c in relation.get(key) or []:
                if c not in rel_cols:
                    rel_cols.append(str(c))
        sample_lines = []
        for c in rel_cols:
            vals = column_samples.get(c)
            if vals is None:
                continue
            sample_lines.append(f"  - {c!r}: {list(vals)!r}")
        samples_block = (
            "\n".join(sample_lines) if sample_lines else "  (no relevant columns)"
        )
        sample_csv_block = sample_csv
        metadata_block = f"total rows: {n_rows}"

    return template.format(
        kind=relation.get("kind", ""),
        columns=json.dumps(list(relation.get("columns") or []), ensure_ascii=False),
        lhs=json.dumps(list(relation.get("lhs") or []), ensure_ascii=False),
        rhs=json.dumps(list(relation.get("rhs") or []), ensure_ascii=False),
        tables=json.dumps(list(relation.get("tables") or []), ensure_ascii=False),
        predicate=relation.get("predicate", ""),
        executable=relation.get("executable", "") or "(none)",
        rationale=relation.get("rationale", "") or "",
        source=relation.get("source", ""),
        error_type=relation.get("error_type", ""),
        headers=json.dumps(list(headers), ensure_ascii=False),
        schema=schema_block,
        column_samples=samples_block,
        sample_csv=sample_csv_block,
        n_rows=n_rows,
        table_metadata=metadata_block,
    )


# ---------------------------------------------------------------------------
# Sample-violation extraction
# ---------------------------------------------------------------------------


def _render_sample_csv(
    df: pd.DataFrame, *, head_n: int, sample_seed: Optional[int],
) -> str:
    """Render a CSV sample of ``df`` for the LLM prompt.

    ``sample_seed=None`` returns the deterministic ``df.head(head_n)``
    slice (the historical behaviour). An integer seed permutes the
    row order first so the adaptive ensemble shows the LLM a different
    concrete view of the table on each pass.
    """
    if head_n <= 0 or df.empty:
        return ""
    if sample_seed is None:
        return df.head(head_n).to_csv(index=False)
    # Deterministic per-seed permutation; seeds modulo-folded so we
    # accept arbitrary integers without overflowing pandas' RNG.
    return (
        df.sample(
            n=min(head_n, len(df)),
            random_state=int(sample_seed) & 0x7FFFFFFF,
        )
        .to_csv(index=False)
    )


def _extract_violator_keys(
    df: Optional[pd.DataFrame],
    dfs: Mapping[str, pd.DataFrame],
    tables_order: Sequence[str],
    mask: pd.Series,
    *,
    is_multi: bool,
) -> List[Any]:
    """Return ``(sheet, row)`` keys for EVERY violating row, before
    truncation. The single-table form uses ``"<single>"`` as the
    sheet placeholder so callers see a uniform key shape across modes.
    """
    if mask.empty:
        return []
    fail = (~mask).fillna(True)
    keys: List[Any] = []
    if is_multi:
        cursor = 0
        for sheet in tables_order:
            if sheet not in dfs:
                continue
            df_s = dfs[sheet]
            n = len(df_s)
            if not df_s.index.is_unique:
                # Same defensive bail-out as _extract_multi_table_violations.
                cursor += n
                continue
            sub_mask = mask.iloc[cursor:cursor + n].reset_index(drop=True)
            sub_mask.index = df_s.index
            cursor += n
            sub_fail = (~sub_mask).fillna(True)
            for idx in df_s.index[sub_fail]:
                keys.append((sheet, int(idx)))
    else:
        if df is None:
            return []
        for idx in df.index[fail]:
            keys.append(("<single>", int(idx)))
    return keys


def _extract_sample_violations(
    df: pd.DataFrame,
    mask: pd.Series,
    columns: Sequence[str],
    *,
    max_rows: int = 5,
) -> List[Dict[str, Any]]:
    """Pull up to ``max_rows`` rows where ``mask`` is False.

    Only the columns the relation references are projected (when
    available) plus a ``__row__`` index so the reviewer can locate
    them in the source CSV.
    """
    if mask.empty:
        return []
    fail = (~mask).fillna(True)
    if not fail.any():
        return []
    # Project to columns that actually exist in df.
    keep = [c for c in columns if c in df.columns]
    sub = df.loc[fail, keep] if keep else df.loc[fail]
    sub = sub.head(max_rows)
    rows: List[Dict[str, Any]] = []
    for idx, row in sub.iterrows():
        rec: Dict[str, Any] = {"__row__": int(idx)}
        for c in sub.columns:
            v = row[c]
            if pd.isna(v):
                rec[c] = None
            elif isinstance(v, (bool, np.bool_)):
                rec[c] = bool(v)
            elif isinstance(v, (int, np.integer)):
                rec[c] = int(v)
            elif isinstance(v, (float, np.floating)):
                rec[c] = float(v)
            else:
                rec[c] = str(v)
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Per-relation validation entrypoint
# ---------------------------------------------------------------------------


def validate_one(
    *,
    relation: Mapping[str, Any],
    df: Optional[pd.DataFrame] = None,
    source_frames: Optional[Mapping[str, pd.DataFrame]] = None,
    llm: LLMClient,
    model: str,
    prompt_cfg: Mapping[str, Any],
    sample_rows: int = 30,
    timeout_seconds: float = 8.0,
    max_violation_samples: int = 5,
    layer: str = "originals",
    sample_seed: Optional[int] = None,
) -> RelationVerification:
    """Validate one relation against ``df`` (single-table) or against
    a ``source_frames`` dict (multi-table).

    The function is small on purpose: caller (``validate_relations``)
    handles concurrency, retries, and persistence.

    For multi-table relations (``relation['tables']`` non-empty), pass
    ``source_frames``; the relation's listed sheets are looked up
    there and bundled into a ``dfs`` dict for the LLM-written
    ``check(dfs)`` function. The ``df`` argument is ignored in that
    case.

    ``sample_seed`` controls which rows the LLM sees in the prompt's
    sample-CSV block. ``None`` (default) keeps the deterministic
    ``df.head(sample_rows)`` slice -- right for a one-shot validation
    -- while an integer seed permutes the row indices first so each
    adaptive pass sees a different concrete view of the table. The
    underlying check the LLM ends up writing is evaluated against the
    FULL frame either way; perturbation only varies which rows the
    LLM grounds its predicate in.
    """
    species_key = str(relation.get("species_key") or "")
    source = str(relation.get("source") or "")
    error_type = str(relation.get("error_type") or "")
    kind = str(relation.get("kind") or "")

    rec = RelationVerification(
        species_key=species_key,
        source=source,
        error_type=error_type,
        kind=kind,
        verifiable=False,
        layer=layer,
    )

    tables_list = [
        str(t) for t in (relation.get("tables") or []) if str(t).strip()
    ]
    is_multi = len(tables_list) >= 2

    # Resolve the data payload for the sandbox call.
    dfs: Dict[str, pd.DataFrame] = {}
    if is_multi:
        if source_frames is None:
            rec.error = "multi-table relation requires source_frames"
            return rec
        for t in tables_list:
            if t not in source_frames:
                rec.error = f"missing source frame for table {t!r}"
                return rec
            dfs[t] = source_frames[t]
        if not dfs:
            rec.error = "no source frames resolved for multi-table relation"
            return rec
    else:
        if df is None:
            rec.error = "single-table relation requires a df argument"
            return rec

    # Build the prompt. Schema and column samples differ by mode.
    if is_multi:
        # Per-sheet schema + sample dictionaries.
        table_schemas: Dict[str, Dict[str, str]] = {}
        table_samples: Dict[str, Dict[str, List[Any]]] = {}
        table_csvs: Dict[str, str] = {}
        table_row_counts: Dict[str, int] = {}
        rel_cols = [str(c) for c in (relation.get("columns") or [])]
        for sheet, sheet_df in dfs.items():
            table_schemas[sheet] = {
                str(c): str(sheet_df[c].dtype) for c in sheet_df.columns
            }
            samples_for_sheet: Dict[str, List[Any]] = {}
            for c in sheet_df.columns:
                if rel_cols and str(c) not in rel_cols:
                    continue
                try:
                    uniq = sheet_df[c].dropna().unique().tolist()[:5]
                except Exception:  # noqa: BLE001
                    uniq = []
                samples_for_sheet[str(c)] = uniq
            table_samples[sheet] = samples_for_sheet
            head_n = min(int(sample_rows), len(sheet_df))
            table_csvs[sheet] = _render_sample_csv(
                sheet_df, head_n=head_n, sample_seed=sample_seed,
            )
            table_row_counts[sheet] = int(len(sheet_df))
        # Aggregated headers / dtypes used by the row/group code path
        # are not consumed by the multi-table branch but kept for
        # uniform template substitution.
        headers: List[str] = []
        dtypes: Dict[str, str] = {}
        column_samples: Dict[str, List[Any]] = {}
        sample_csv = ""
        n_total_rows = sum(table_row_counts.values())
        user_prompt = _build_user_prompt(
            prompt_cfg.get("user_template") or "",
            relation=relation,
            headers=headers,
            dtypes=dtypes,
            column_samples=column_samples,
            sample_csv=sample_csv,
            n_rows=n_total_rows,
            multi_table=True,
            table_schemas=table_schemas,
            table_samples=table_samples,
            table_csvs=table_csvs,
            table_row_counts=table_row_counts,
        )
    else:
        assert df is not None  # narrowing; guarded above.
        headers = list(df.columns)
        dtypes = {str(c): str(df[c].dtype) for c in df.columns}
        # Per-column distinct-value preview (up to 5 non-null values
        # each) so the LLM sees the concrete shape of cells without
        # having to scan the full CSV sample.
        column_samples = {}
        for c in df.columns:
            try:
                uniq = df[c].dropna().unique().tolist()[:5]
            except Exception:  # noqa: BLE001
                uniq = []
            column_samples[str(c)] = uniq
        head_n = min(int(sample_rows), len(df))
        sample_csv = _render_sample_csv(
            df, head_n=head_n, sample_seed=sample_seed,
        )
        user_prompt = _build_user_prompt(
            prompt_cfg.get("user_template") or "",
            relation=relation,
            headers=headers,
            dtypes=dtypes,
            column_samples=column_samples,
            sample_csv=sample_csv,
            n_rows=int(len(df)),
        )

    system_prompt = prompt_cfg.get("system_prompt") or ""
    reasoning_effort = prompt_cfg.get("reasoning_effort") or "medium"

    try:
        resp = llm.query(
            model=model,
            prompt=user_prompt,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
        )
    except Exception as exc:  # noqa: BLE001
        rec.error = f"llm-error: {type(exc).__name__}: {exc}"
        return rec
    rec.llm_attempts = getattr(resp, "attempts", 0) or 0

    try:
        decision = _parse_llm_decision(resp.text)
    except ValueError as exc:
        rec.error = f"parse-error: {exc}"
        return rec

    rec.verifiable = bool(decision.get("verifiable"))
    if not rec.verifiable:
        rec.reason = str(decision.get("reason") or "").strip()
        return rec

    code = decision.get("code")
    if not isinstance(code, str) or not code.strip():
        rec.error = "verifiable=true but no code returned"
        return rec
    rec.code = code

    # AST-validate in-process (cheap; surfaces sandbox errors without
    # paying for a subprocess spawn). The actual exec + run happens in a
    # killable, resource-capped child so a runaway check() can't hang us.
    try:
        _validate_code_ast(code)
    except CodeValidationError as exc:
        rec.error = f"sandbox: {exc}"
        return rec

    # Run (out-of-process).
    payload: Any = dfs if is_multi else df
    try:
        mask = run_check_subprocess(code, payload, timeout_seconds)
    except TimeoutError as exc:
        rec.error = f"timeout: {exc}"
        return rec
    except CodeValidationError as exc:
        # Child re-validation / compile error (defence in depth).
        rec.error = f"sandbox: {exc}"
        return rec
    except Exception as exc:  # noqa: BLE001
        rec.error = f"runtime: {type(exc).__name__}: {exc}"
        return rec

    n = int(len(mask))
    n_pass = int(mask.sum())
    rec.n_evaluated = n
    rec.n_passing = n_pass
    rec.n_violating = n - n_pass
    rec.pass_rate = (n_pass / n) if n > 0 else None
    cols = (
        list(relation.get("columns") or [])
        + list(relation.get("lhs") or [])
        + list(relation.get("rhs") or [])
    )
    # Dedup while preserving order.
    seen = set()
    cols_unique = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            cols_unique.append(c)
    if is_multi:
        # For multi-table the mask is concatenated across sheets in
        # the order ``tables`` lists them. Map mask positions back to
        # (sheet, original_row) and sample violations from each
        # contributing sheet.
        rec.sample_violations = _extract_multi_table_violations(
            dfs, tables_list, mask, cols_unique,
            max_rows=max_violation_samples,
        )
    else:
        assert df is not None
        rec.sample_violations = _extract_sample_violations(
            df, mask, cols_unique, max_rows=max_violation_samples,
        )
    # Always capture the FULL violator (sheet, row) key set -- the
    # adaptive ensemble needs the unbounded species pool to drive
    # Good-Turing correctly. Truncating to ``max_violation_samples``
    # would hand GT at most 5 species per run and hide the missing
    # mass it is supposed to estimate.
    rec.violator_keys = _extract_violator_keys(
        df, dfs, tables_list, mask, is_multi=is_multi,
    )
    return rec


# ---------------------------------------------------------------------------
# Adaptive (K-sample ensemble) validation entrypoint (Gap E)
# ---------------------------------------------------------------------------


def validate_one_adaptive(
    *,
    relation: Mapping[str, Any],
    df: Optional[pd.DataFrame] = None,
    source_frames: Optional[Mapping[str, pd.DataFrame]] = None,
    llm: LLMClient,
    model: str,
    prompt_cfg: Mapping[str, Any],
    arm_config: Any,  # src.adaptive.scoped_gt.ArmConfig
    sample_rows: int = 30,
    timeout_seconds: float = 8.0,
    max_violation_samples: int = 5,
    layer: str = "originals",
    base_seed: int = 1009,
) -> RelationVerification:
    """K-sample ensemble wrapper around :func:`validate_one`.

    Runs validation up to ``arm_config.k_max`` times. Each run sees a
    different ``sample_seed`` (mixed from ``base_seed`` and the run
    index) which permutes the rows the LLM sees in the prompt's
    sample-CSV block, so the LLM grounds its ``check`` predicate on
    a different concrete view each pass. The mask the LLM-written
    function returns is always evaluated against the FULL frame, so
    perturbation only varies the prompt -- never the ground truth.

    Species for Good-Turing is the FULL ``(sheet, row_index)`` set
    of violators in each run -- not the truncated
    ``sample_violations`` preview, which would feed GT at most
    ``max_violation_samples`` species and badly bias the missing
    mass downward. The inner loop stops when each pass adds no new
    violators (the GT/Chao1 saturation predicate from
    ``arm_config``).

    The final :class:`RelationVerification` carries:

    * ``n_violating`` = size of the unioned violator set across runs.
    * ``violator_keys`` = the unioned key list (used by callers that
      want the full set; not persisted).
    * ``sample_violations`` = the first ``max_violation_samples``
      violators, annotated with which run indices flagged each one.
    * ``llm_attempts`` = total LLM attempts across all runs.

    On the first run with ``verifiable=False`` we short-circuit:
    the relation isn't code-checkable, so re-sampling won't help.
    """
    from ...adaptive.inner_loop import run_adaptive

    # Pre-compute one seed per run index so two adaptive runs with
    # the same ``base_seed`` are exactly reproducible. The mixer is
    # simple but stable across Python versions.
    def _seed_for(run_idx: int) -> int:
        return (int(base_seed) * 1_000_003 + int(run_idx) * 97 + 31) & 0x7FFFFFFF

    # Run 0 first (synchronous) to handle the not-verifiable case.
    # Run 0 keeps the deterministic head() sample so single-shot and
    # K=1 adaptive paths produce byte-identical prompts.
    first = validate_one(
        relation=relation, df=df, source_frames=source_frames,
        llm=llm, model=model, prompt_cfg=prompt_cfg,
        sample_rows=sample_rows, timeout_seconds=timeout_seconds,
        max_violation_samples=max_violation_samples, layer=layer,
        sample_seed=None,
    )
    if not first.verifiable or first.error:
        return first

    # If k_max <= 1 we are done; this preserves single-shot behavior.
    if arm_config.k_max is None or arm_config.k_max <= 1:
        return first

    # Track unioned violators by (sheet_or_'<single>', row).
    unioned: Dict[tuple, Dict[str, Any]] = {}

    def _ingest(rec: RelationVerification, run_idx: int) -> List[tuple]:
        # Build a quick lookup so we can attach the rich snapshot from
        # ``sample_violations`` (when present) to the corresponding
        # entry in the FULL ``violator_keys`` list.
        snapshot_by_key: Dict[tuple, Dict[str, Any]] = {}
        for v in rec.sample_violations or []:
            sheet = str(v.get("__sheet__", "<single>"))
            row = int(v.get("__row__", -1))
            snapshot_by_key[(sheet, row)] = v

        species: List[tuple] = []
        for key in rec.violator_keys or []:
            try:
                sheet, row = key
                key_tuple = (str(sheet), int(row))
            except (TypeError, ValueError):
                continue
            if key_tuple not in unioned:
                snap = snapshot_by_key.get(key_tuple)
                if snap is not None:
                    base = dict(snap)
                else:
                    base = {
                        "__sheet__": key_tuple[0],
                        "__row__": key_tuple[1],
                    }
                base["__discovery_runs__"] = [run_idx]
                unioned[key_tuple] = base
            else:
                runs = unioned[key_tuple].setdefault("__discovery_runs__", [])
                if run_idx not in runs:
                    runs.append(run_idx)
            species.append(key_tuple)
        return species

    species_per_run: Dict[int, List[tuple]] = {0: _ingest(first, 0)}
    counts = {"attempts": int(first.llm_attempts or 0)}
    last: List[RelationVerification] = [first]

    def sample_fn(run_idx: int):
        if run_idx == 0:
            return species_per_run[0]
        # Each non-baseline run gets its own deterministic
        # ``sample_seed`` so the LLM sees a different concrete row
        # sample. Two identical seeds (e.g. user replays the same
        # base_seed) collapse to the same prompt -- intentional, so
        # adaptive runs are reproducible.
        try:
            rec = validate_one(
                relation=relation, df=df, source_frames=source_frames,
                llm=llm, model=model, prompt_cfg=prompt_cfg,
                sample_rows=sample_rows, timeout_seconds=timeout_seconds,
                max_violation_samples=max_violation_samples, layer=layer,
                sample_seed=_seed_for(run_idx),
            )
        except Exception:  # noqa: BLE001
            logger.exception("validate_one failed inside adaptive loop")
            return []
        last[0] = rec
        counts["attempts"] += int(rec.llm_attempts or 0)
        return _ingest(rec, run_idx)

    species_key = str(relation.get("species_key") or "")
    arm_id = f"validate:{species_key}:{relation.get('source', '')}"
    result = run_adaptive(arm_id=arm_id, sample_fn=sample_fn, cfg=arm_config)

    # Build merged record. Take the latest verifiable run as the
    # base (so code/pass_rate fields reflect a successful run);
    # overlay unioned violators.
    merged = first
    for r in (last[0],):
        if r is not None and r.verifiable and not r.error:
            merged = r
            break
    # Persisted preview stays bounded at ``max_violation_samples``;
    # the dashboard / labelling UI doesn't need the full violator list.
    all_violators = list(unioned.values())
    merged.sample_violations = all_violators[:max_violation_samples]
    merged.n_violating = max(merged.n_violating, len(unioned))
    merged.violator_keys = list(unioned.keys())
    merged.llm_attempts = int(counts["attempts"])
    return merged


def _extract_multi_table_violations(
    dfs: Mapping[str, pd.DataFrame],
    tables_order: Sequence[str],
    mask: pd.Series,
    columns: Sequence[str],
    *,
    max_rows: int = 5,
) -> List[Dict[str, Any]]:
    """Pull up to ``max_rows`` violating rows across all sheets.

    The mask is the concatenation of per-sheet boolean arrays in the
    order listed in ``tables_order``. We re-split it and project each
    sheet's violators down to the relation's columns (those that
    actually exist in that sheet). Each emitted record carries an
    extra ``__sheet__`` key so the reviewer can tell which frame the
    violator came from.
    """
    out: List[Dict[str, Any]] = []
    cursor = 0
    for sheet in tables_order:
        if sheet not in dfs:
            continue
        df_s = dfs[sheet]
        n = len(df_s)
        # Defensive: the mask is a positional slice of a concatenated
        # bool array. Aligning back to the source frame requires its
        # index to be unique; with a duplicate index ``df_s.loc[fail]``
        # broadcasts ambiguously and emits the wrong row in
        # ``__row__``. Fail loud rather than silently mis-attribute.
        if not df_s.index.is_unique:
            raise ValueError(
                f"_extract_multi_table_violations: source frame "
                f"{sheet!r} has a non-unique index; reset_index(drop=True) "
                f"before passing it in."
            )
        sub_mask = mask.iloc[cursor:cursor + n].reset_index(drop=True)
        sub_mask.index = df_s.index
        cursor += n
        # Project to columns that exist in this sheet.
        keep = [c for c in columns if c in df_s.columns]
        fail = (~sub_mask).fillna(True)
        if not fail.any():
            continue
        violators = df_s.loc[fail, keep] if keep else df_s.loc[fail]
        room = max_rows - len(out)
        if room <= 0:
            break
        for idx, row in violators.head(room).iterrows():
            rec: Dict[str, Any] = {"__sheet__": sheet, "__row__": int(idx)}
            for c in violators.columns:
                v = row[c]
                if pd.isna(v):
                    rec[c] = None
                elif isinstance(v, (bool, np.bool_)):
                    rec[c] = bool(v)
                elif isinstance(v, (int, np.integer)):
                    rec[c] = int(v)
                elif isinstance(v, (float, np.floating)):
                    rec[c] = float(v)
                else:
                    rec[c] = str(v)
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Batch validation with LLM concurrency
# ---------------------------------------------------------------------------


def validate_relations(
    relations: Sequence[Mapping[str, Any]],
    *,
    source_frames: Mapping[str, pd.DataFrame],
    llm: LLMClient,
    model: str,
    prompt_cfg: Mapping[str, Any],
    concurrency: int = 4,
    sample_rows: int = 30,
    timeout_seconds: float = 8.0,
    max_violation_samples: int = 5,
    layer: str = "originals",
    progress_cb: Optional[Callable[[int, int], None]] = None,
    arm_config_resolver: Optional[Callable[[Mapping[str, Any]], Any]] = None,
) -> List[RelationVerification]:
    """Validate a batch of relations against their source DataFrames.

    Concurrency parallelises the LLM I/O; the actual ``check`` exec
    holds the GIL so heavy CPU-bound checks serialise on it.

    When ``arm_config_resolver`` is supplied, each relation is
    dispatched through :func:`validate_one_adaptive`, which runs up
    to ``cfg.k_max`` LLM samples and unions their violator sets
    (recall-first). The resolver receives the relation dict and
    returns an :class:`~src.adaptive.scoped_gt.ArmConfig`. Pass
    ``None`` (the default) to preserve single-shot behaviour.
    """
    out: List[RelationVerification] = [None] * len(relations)  # type: ignore[list-item]
    total = len(relations)
    done = 0
    lock = threading.Lock()

    def _work(idx: int, rel: Mapping[str, Any]) -> None:
        nonlocal done
        src = str(rel.get("source") or "")
        tables_list = [
            str(t) for t in (rel.get("tables") or []) if str(t).strip()
        ]
        is_multi = len(tables_list) >= 2

        # Resolve the per-relation arm config. When resolver returns
        # None (or itself is None), we fall through to single-shot.
        arm_cfg = None
        if arm_config_resolver is not None:
            try:
                arm_cfg = arm_config_resolver(rel)
            except Exception:  # noqa: BLE001
                logger.exception("arm_config_resolver failed; using single-shot")
                arm_cfg = None
        use_adaptive = (
            arm_cfg is not None
            and getattr(arm_cfg, "k_max", None) is not None
            and arm_cfg.k_max > 1
        )

        if is_multi:
            if use_adaptive:
                out[idx] = validate_one_adaptive(
                    relation=rel, df=None, source_frames=source_frames,
                    llm=llm, model=model, prompt_cfg=prompt_cfg,
                    arm_config=arm_cfg,
                    sample_rows=sample_rows,
                    timeout_seconds=timeout_seconds,
                    max_violation_samples=max_violation_samples,
                    layer=layer,
                )
            else:
                out[idx] = validate_one(
                    relation=rel,
                    df=None,
                    source_frames=source_frames,
                    llm=llm,
                    model=model,
                    prompt_cfg=prompt_cfg,
                    sample_rows=sample_rows,
                    timeout_seconds=timeout_seconds,
                    max_violation_samples=max_violation_samples,
                    layer=layer,
                )
        else:
            df = source_frames.get(src)
            if df is None:
                out[idx] = RelationVerification(
                    species_key=str(rel.get("species_key") or ""),
                    source=src,
                    error_type=str(rel.get("error_type") or ""),
                    kind=str(rel.get("kind") or ""),
                    verifiable=False,
                    error=f"source frame not loaded: {src!r}",
                    layer=layer,
                )
            elif use_adaptive:
                out[idx] = validate_one_adaptive(
                    relation=rel, df=df, source_frames=source_frames,
                    llm=llm, model=model, prompt_cfg=prompt_cfg,
                    arm_config=arm_cfg,
                    sample_rows=sample_rows,
                    timeout_seconds=timeout_seconds,
                    max_violation_samples=max_violation_samples,
                    layer=layer,
                )
            else:
                out[idx] = validate_one(
                    relation=rel,
                    df=df,
                    source_frames=source_frames,
                    llm=llm,
                    model=model,
                    prompt_cfg=prompt_cfg,
                    sample_rows=sample_rows,
                    timeout_seconds=timeout_seconds,
                    max_violation_samples=max_violation_samples,
                    layer=layer,
                )
        with lock:
            done += 1
            d = done
        if progress_cb is not None:
            try:
                progress_cb(d, total)
            except Exception:  # noqa: BLE001
                pass

    if concurrency <= 1:
        for i, rel in enumerate(relations):
            _work(i, rel)
    else:
        with _cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(_work, i, rel) for i, rel in enumerate(relations)
            ]
            for fut in _cf.as_completed(futures):
                fut.result()

    # Drop any None placeholders (shouldn't happen, but be defensive).
    return [r for r in out if r is not None]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_verifications_jsonl(
    records: Iterable[RelationVerification],
    path: Path,
) -> int:
    """Write one JSON line per verification record. Returns count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def load_verifications_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a verifications JSONL into plain dicts (for embedding in HTML)."""
    out: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        return out
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
