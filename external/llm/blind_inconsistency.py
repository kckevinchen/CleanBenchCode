"""Blind LLM baseline for the inconsistency benchmark.

The detector pipeline that produced
``benchmark/benchmark_inconsistency_<TS>_gpt5-family.jsonl`` is
specialised: it proposes column clusters first, then a human
labels each cluster, and only the TP/unsure clusters become
benchmark rows. That pipeline is the system under test. To
calibrate it, we need an INDEPENDENT baseline: hand the same model
a raw CSV column and ask it to find inconsistency errors, with no
detector hints, no cluster pre-groupings, no benchmark labels, no
knowledge of which sources the benchmark even covers.

The input is the RAW upstream corpus at
``data/source_inconsistency_bing21/`` -- a directory tree of
``<workbook_sha1>.xlsx/<short_id>.csv`` paths (192 workbooks, 218
sheets in the canonical corpus). The script enumerates these CSVs
DIRECTLY (no detector ``summary.json``, no benchmark sources zip)
so the detection step is fully decoupled from any prior
benchmark-side artefact. The ``source`` identifier emitted for
each prediction is the CSV basename (e.g. ``addf5bdd.csv``),
which matches the benchmark's ``source`` field for the scoring
step at the end.

Flow:

1. Walk ``--source-dir`` for ``*.xlsx/*.csv`` paths.
2. For each ``(source, column)``, build a prompt from the column's
   distinct values + counts and ask the LLM to surface
   normalisation groups (see ``external/llm/prompts/inconsistency.yaml``).
3. Expand every group into row-level predictions
   (``{source, row, column, original_value, predicted_correction,
   predicted_kind}``) by looking up which rows hold each value.
   Predictions are emitted only for rows whose value differs from
   the group's canonical (or for ALL rows in the group when the
   canonical is null), so the prediction set is directly
   comparable to the benchmark JSONL.
4. Score predictions against the benchmark JSONL. The benchmark is
   used ONLY for scoring -- it never influences detection inputs.
5. Write outputs under
   ``outputs/baseline_inconsistency_<TS>_gpt5-family/`` so reruns
   resume from cache.

CLI:

    python -m external.llm.blind_inconsistency \
        --source-dir data/source_inconsistency_bing21 \
        --benchmark benchmark/benchmark_inconsistency_20260513T090000Z_gpt5-family.jsonl \
        --max-sources 2 \
        --concurrency 4

Omit ``--max-sources`` to run the entire corpus.

The cache is sha256-keyed over the rendered prompt + model
settings, persisted as JSONL alongside the run. Reruns are cheap.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger("benchmark_llm_baseline")


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

PROMPT_VERSION = 2
COLUMN_CONTEXT_LIMIT = 500
# Skip columns with fewer than this many distinct values: a column
# with 1 distinct value has nothing to compare, and with 2 the
# detector already does a good job - this is a sanity cap, not a
# capability claim.
MIN_DISTINCT_VALUES = 2

VALID_KINDS = {
    "casing", "punctuation", "pattern",
    "numerical_format", "text", "untagged",
}


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------


@dataclass
class PromptCfg:
    system: str
    user_template: str
    model: str
    reasoning_effort: str


def load_prompt_cfg(path: Path) -> PromptCfg:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PromptCfg(
        system=str(doc.get("system") or "").strip(),
        user_template=str(doc.get("user_template") or ""),
        model=str(doc.get("model") or "gpt-5.2"),
        reasoning_effort=str(doc.get("reasoning_effort") or "medium"),
    )


def render_prompt(
    cfg: PromptCfg,
    *,
    source: str,
    column: str,
    df: pd.DataFrame,
) -> Tuple[str, int, bool]:
    """Return ``(prompt_text, distinct_count, truncated)``."""

    vc = df[column].astype(str).value_counts(dropna=False)
    total = int(len(vc))
    truncated = total > COLUMN_CONTEXT_LIMIT
    rows: List[str] = []
    for v, n in vc.head(COLUMN_CONTEXT_LIMIT).items():
        rows.append(f"  - {json.dumps(v, ensure_ascii=False)} (count={int(n)})")
    value_list = "\n".join(rows) if rows else "  (column is empty)"
    truncated_hint = (
        f", showing top {COLUMN_CONTEXT_LIMIT} by frequency"
        if truncated else ""
    )
    prompt = cfg.user_template.format(
        source=source,
        column=column,
        value_list=value_list,
        distinct_count=total,
        truncated_hint=truncated_hint,
    )
    return prompt, total, truncated


# ---------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_findings(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse ``{"findings": [...]}``; return ``None`` on shape failure."""

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
    raw = payload.get("findings")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    out: List[Dict[str, Any]] = []
    for f in raw:
        if not isinstance(f, dict):
            continue
        vals = f.get("values")
        if not isinstance(vals, list):
            continue
        vals_clean = [v for v in vals if isinstance(v, str)]
        if len(vals_clean) < 2:
            continue
        canon = f.get("canonical_value")
        if canon is not None and not isinstance(canon, str):
            canon = None
        kind = str(f.get("kind") or "untagged").lower()
        if kind not in VALID_KINDS:
            kind = "untagged"
        rationale = str(f.get("rationale") or "")
        out.append({
            "values": vals_clean,
            "canonical_value": canon,
            "kind": kind,
            "rationale": rationale,
        })
    return out


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------


def cache_key(
    *,
    system_prompt: str,
    prompt: str,
    model: str,
    reasoning_effort: str,
    sample_idx: int,
) -> str:
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


# ---------------------------------------------------------------------
# CSV resolution
# ---------------------------------------------------------------------


def resolve_csv_map(source_dir: Path) -> Dict[str, Path]:
    """Walk ``source_dir`` for ``*.xlsx/*.csv`` paths.

    Returns ``{csv_basename -> absolute path}``. The CSV basename is
    the canonical ``source`` identifier used by the benchmark JSONL,
    so predictions emitted with this key join cleanly into the
    scoring step.

    Raises ``ValueError`` if two CSVs share the same basename
    (would corrupt the (source, row, column) join).
    """

    if not source_dir.is_dir():
        raise FileNotFoundError(f"--source-dir does not exist: {source_dir}")
    out: Dict[str, Path] = {}
    # Match the curated bing21 layout: <wb_sha1>.xlsx/<sheet>.csv .
    # ``rglob`` so we also work on any flat layout that happens to
    # drop CSVs directly in ``source_dir``.
    for path in source_dir.rglob("*.csv"):
        if not path.is_file():
            continue
        name = path.name
        if name in out and out[name] != path:
            raise ValueError(
                f"Duplicate CSV basename {name!r} under {source_dir} "
                f"({out[name]} vs {path}); benchmark join is not unique."
            )
        out[name] = path
    return out


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Best-effort CSV load (sniff utf-8 vs latin-1)."""

    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path, dtype=str, keep_default_na=False, na_values=[],
                encoding="latin-1",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CSV load failed for %s: %s", path, exc)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("CSV load failed for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------


@dataclass
class BenchEntry:
    source: str
    row: int
    column: str
    original_value: str
    label: str                    # "error" | "ambiguous"
    # ``corrected_value`` is a string for single-canonical fixes and
    # a list of strings for bi-directional / multi-canonical cases
    # (both forms acceptable). ``None`` when the benchmark row has
    # no fix on file.
    corrected_value: Any
    cluster_kinds: List[str]


def load_benchmark(path: Path) -> Dict[Tuple[str, int, str], BenchEntry]:
    """Return ``{(source, row, column): BenchEntry}``.

    Latest-line-wins per key (mirrors the rest of the pipeline).
    """

    out: Dict[Tuple[str, int, str], BenchEntry] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = rec.get("source")
            row = rec.get("row")
            col = rec.get("column")
            if not isinstance(src, str) or not isinstance(row, int) or not isinstance(col, str):
                continue
            out[(src, row, col)] = BenchEntry(
                source=src,
                row=row,
                column=col,
                original_value=str(rec.get("original_value") or ""),
                label=str(rec.get("label") or ""),
                corrected_value=rec.get("corrected_value"),
                cluster_kinds=[
                    str(k) for k in (rec.get("cluster_kinds") or [])
                    if isinstance(k, str)
                ],
            )
    return out


# ---------------------------------------------------------------------
# Prediction emission
# ---------------------------------------------------------------------


def expand_findings_to_predictions(
    *,
    source: str,
    column: str,
    df: pd.DataFrame,
    findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert per-column findings into per-row predictions.

    When ``canonical_value`` is set, only rows whose value != canonical
    are emitted. When canonical is null, ALL rows in the group are
    emitted (with ``predicted_correction = null``).
    """

    if not findings:
        return []
    col_series = df[column].astype(str)
    out: List[Dict[str, Any]] = []
    for f in findings:
        values = f.get("values") or []
        canon = f.get("canonical_value")
        kind = f.get("kind") or "untagged"
        rationale = f.get("rationale") or ""
        # Build row index per value once.
        for v in values:
            if not isinstance(v, str):
                continue
            if canon is not None and v == canon:
                # Canonical rows are already correct; skip.
                continue
            mask = (col_series == v)
            rows = list(mask[mask].index.astype(int))
            for r in rows:
                out.append({
                    "source": source,
                    "row": int(r),
                    "column": column,
                    "original_value": v,
                    "predicted_correction": canon,
                    "predicted_kind": kind,
                    "group_values": list(values),
                    "rationale": rationale,
                })
    return out


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------


@dataclass
class ColumnJob:
    source: str
    column: str
    df: pd.DataFrame


@dataclass
class ColumnResult:
    source: str
    column: str
    distinct_count: int
    truncated: bool
    findings: Optional[List[Dict[str, Any]]]
    raw_text: str
    error: Optional[str]
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    cached: bool = False


def process_column(
    job: ColumnJob,
    *,
    cfg: PromptCfg,
    llm: LLMClient,
    cache: Dict[str, Dict[str, Any]],
    cache_path: Path,
    cache_lock,
) -> ColumnResult:
    t0 = time.time()
    prompt, distinct_count, truncated = render_prompt(
        cfg,
        source=job.source,
        column=job.column,
        df=job.df,
    )
    key = cache_key(
        system_prompt=cfg.system,
        prompt=prompt,
        model=cfg.model,
        reasoning_effort=cfg.reasoning_effort,
        sample_idx=0,
    )
    cached_rec = cache.get(key)
    if cached_rec is not None:
        raw_text = str(cached_rec.get("raw") or "")
        findings = parse_findings(raw_text)
        return ColumnResult(
            source=job.source,
            column=job.column,
            distinct_count=distinct_count,
            truncated=truncated,
            findings=findings,
            raw_text=raw_text,
            error=None,
            predictions=expand_findings_to_predictions(
                source=job.source,
                column=job.column,
                df=job.df,
                findings=findings or [],
            ),
            elapsed_s=time.time() - t0,
            cached=True,
        )

    try:
        resp = llm.query(
            cfg.model,
            prompt,
            reasoning_effort=cfg.reasoning_effort,
            reasoning_summary="auto",
            system_prompt=cfg.system,
        )
        raw_text = resp.text or ""
        err = resp.error
    except Exception as exc:  # noqa: BLE001
        raw_text = ""
        err = f"call_failed: {exc}"

    findings = parse_findings(raw_text) if raw_text else None

    # Persist cache (only on a real call).
    rec = {
        "key": key,
        "source": job.source,
        "column": job.column,
        "prompt_version": PROMPT_VERSION,
        "model": cfg.model,
        "reasoning_effort": cfg.reasoning_effort,
        "raw": raw_text,
        "error": err,
        "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with cache_lock:
        cache[key] = rec
        append_cache(cache_path, rec)

    predictions = expand_findings_to_predictions(
        source=job.source,
        column=job.column,
        df=job.df,
        findings=findings or [],
    )
    return ColumnResult(
        source=job.source,
        column=job.column,
        distinct_count=distinct_count,
        truncated=truncated,
        findings=findings,
        raw_text=raw_text,
        error=err,
        predictions=predictions,
        elapsed_s=time.time() - t0,
        cached=False,
    )


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------


def _normalize_value(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _correction_matches(
    *, bench_corrected: Any, predicted: Optional[str],
) -> bool:
    """Compare a predicted correction against the benchmark value.

    The benchmark stores ``corrected_value`` as either a string
    (single canonical) or a list of strings (multi-canonical /
    bi-directional case). In the list case any member counts as a
    match -- we don't penalise the model for picking ANY of the
    acceptable canonicals.
    """

    if predicted is None or bench_corrected is None:
        return False
    norm_pred = _normalize_value(predicted)
    if isinstance(bench_corrected, list):
        return any(
            _normalize_value(v) == norm_pred for v in bench_corrected
            if isinstance(v, str)
        )
    return _normalize_value(bench_corrected) == norm_pred


def score_predictions(
    *,
    predictions: List[Dict[str, Any]],
    bench: Dict[Tuple[str, int, str], BenchEntry],
    bench_sources: set,
) -> Dict[str, Any]:
    """Compute confusion-style metrics vs the benchmark.

    A prediction key is ``(source, row, column)``.

    * tp_strict   - prediction hits ``label=error``
    * tp_lenient  - prediction hits ``label in {error, ambiguous}``
    * fn_strict   - bench ``error`` row missed by predictions
    * fn_lenient  - bench ``error|ambiguous`` row missed
    * fp_in_bench_source     - prediction outside bench but inside a
                               source that has SOME bench rows
    * fp_outside_bench_source - prediction in a source the bench
                                never touches (label-filter dropped it)
    * correction_strict      - tp_strict where predicted_correction
                               matches bench.corrected_value
    * kind_strict            - tp_strict where predicted_kind matches
                               one of bench.cluster_kinds
    """

    pred_keys: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for p in predictions:
        k = (p["source"], int(p["row"]), p["column"])
        # Last-write-wins (multiple findings can target the same row).
        pred_keys[k] = p

    tp_strict = 0
    tp_lenient = 0
    correction_strict_match = 0
    correction_strict_eval = 0
    kind_strict_match = 0
    fp_in_bench_source = 0
    fp_outside_bench_source = 0

    for key, p in pred_keys.items():
        b = bench.get(key)
        if b is None:
            if key[0] in bench_sources:
                fp_in_bench_source += 1
            else:
                fp_outside_bench_source += 1
            continue
        if b.label == "error":
            tp_strict += 1
            tp_lenient += 1
            # Correction accuracy (compare normalised strings).
            if b.corrected_value is not None and p.get("predicted_correction"):
                correction_strict_eval += 1
                if _correction_matches(
                    bench_corrected=b.corrected_value,
                    predicted=p.get("predicted_correction"),
                ):
                    correction_strict_match += 1
            # Kind accuracy.
            pred_kind = (p.get("predicted_kind") or "").lower()
            if pred_kind and pred_kind in {k.lower() for k in b.cluster_kinds}:
                kind_strict_match += 1
        elif b.label == "ambiguous":
            tp_lenient += 1

    n_bench_error = sum(1 for b in bench.values() if b.label == "error")
    n_bench_ambiguous = sum(1 for b in bench.values() if b.label == "ambiguous")
    bench_error_keys = {k for k, b in bench.items() if b.label == "error"}
    bench_lenient_keys = {k for k, b in bench.items() if b.label in {"error", "ambiguous"}}
    fn_strict = len(bench_error_keys - pred_keys.keys())
    fn_lenient = len(bench_lenient_keys - pred_keys.keys())

    n_pred = len(pred_keys)
    return {
        "n_predictions": n_pred,
        "n_bench_error": n_bench_error,
        "n_bench_ambiguous": n_bench_ambiguous,
        "tp_strict": tp_strict,
        "tp_lenient": tp_lenient,
        "fn_strict": fn_strict,
        "fn_lenient": fn_lenient,
        "fp_in_bench_source": fp_in_bench_source,
        "fp_outside_bench_source": fp_outside_bench_source,
        "recall_strict": (tp_strict / n_bench_error) if n_bench_error else None,
        "recall_lenient": (tp_lenient / (n_bench_error + n_bench_ambiguous))
            if (n_bench_error + n_bench_ambiguous) else None,
        "precision_vs_bench": (tp_lenient / n_pred) if n_pred else None,
        "correction_accuracy_strict_eligible": correction_strict_eval,
        "correction_accuracy_strict_match": correction_strict_match,
        "correction_accuracy_strict_rate": (
            correction_strict_match / correction_strict_eval
            if correction_strict_eval else None
        ),
        "kind_accuracy_strict_match": kind_strict_match,
        "kind_accuracy_strict_rate": (
            kind_strict_match / tp_strict if tp_strict else None
        ),
    }


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------


def _utc_stamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blind LLM baseline for the inconsistency benchmark.",
    )
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("data/source_inconsistency_bing21"),
        help=(
            "Directory tree of <wb>.xlsx/<sheet>.csv files. "
            "Defaults to data/source_inconsistency_bing21."
        ),
    )
    parser.add_argument(
        "--benchmark", required=True, type=Path,
        help=(
            "Benchmark JSONL used ONLY for the post-hoc scoring step. "
            "Detection itself is blind to its contents."
        ),
    )
    parser.add_argument(
        "--prompt", type=Path,
        default=PROJECT_ROOT / "external" / "llm" / "prompts" / "inconsistency.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output dir. Defaults to outputs/baseline_inconsistency_<TS>_gpt5-family.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model. Defaults to YAML `model` field.",
    )
    parser.add_argument(
        "--reasoning-effort", default=None,
        help="Override reasoning_effort. Defaults to YAML field.",
    )
    parser.add_argument(
        "--max-sources", type=int, default=None,
        help="Cap sources processed (alphabetical). For dry runs.",
    )
    parser.add_argument(
        "--sources", default=None,
        help="Comma-separated source basenames; overrides --max-sources.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
    )
    parser.add_argument(
        "--log-level", default="INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load_prompt_cfg(args.prompt)
    if args.model:
        cfg.model = args.model
    if args.reasoning_effort:
        cfg.reasoning_effort = args.reasoning_effort

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path(f"outputs/baseline_inconsistency_{_utc_stamp()}_gpt5-family")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "cache.jsonl"
    responses_path = out_dir / "prompt_responses.jsonl"
    predictions_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    summary_path = out_dir / "summary.json"
    errors_path = out_dir / "errors.jsonl"

    # --- Resolve CSVs ---
    csv_map = resolve_csv_map(args.source_dir)
    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        csv_map = {k: v for k, v in csv_map.items() if k in wanted}
        missing = wanted - set(csv_map.keys())
        if missing:
            logger.warning("Sources not in run: %s", sorted(missing))
    elif args.max_sources is not None:
        keep = sorted(csv_map.keys())[: args.max_sources]
        csv_map = {k: csv_map[k] for k in keep}
    logger.info("Sources to process: %d", len(csv_map))

    # --- Build job list ---
    jobs: List[ColumnJob] = []
    skipped_columns = 0
    load_failures: List[Dict[str, str]] = []
    for src in sorted(csv_map.keys()):
        path = csv_map[src]
        df = load_csv(path)
        if df is None or df.empty:
            load_failures.append({"source": src, "reason": "load_failed_or_empty"})
            continue
        for col in df.columns:
            if not isinstance(col, str):
                continue
            try:
                nunique = df[col].astype(str).nunique(dropna=False)
            except Exception as exc:  # noqa: BLE001
                logger.debug("nunique failed for %s::%s: %s", src, col, exc)
                skipped_columns += 1
                continue
            if nunique < MIN_DISTINCT_VALUES:
                skipped_columns += 1
                continue
            jobs.append(ColumnJob(source=src, column=col, df=df))
    logger.info(
        "Jobs prepared: %d (across %d sources, %d columns skipped, %d load failures)",
        len(jobs), len(csv_map), skipped_columns, len(load_failures),
    )

    # --- Resume from cache ---
    cache = load_cache(cache_path)
    if cache:
        logger.info("Loaded %d cached responses from %s", len(cache), cache_path)

    # --- LLM client (single shared instance) ---
    llm = LLMClient()

    # --- Process ---
    import threading
    cache_lock = threading.Lock()
    responses_lock = threading.Lock()
    all_predictions: List[Dict[str, Any]] = []
    errors_list: List[Dict[str, Any]] = []
    completed = 0
    cached_count = 0
    t_start = time.time()

    # Write responses incrementally so a crash leaves a usable trail.
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with responses_path.open("w", encoding="utf-8") as resp_fh:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futures = {
                ex.submit(
                    process_column, j,
                    cfg=cfg, llm=llm, cache=cache,
                    cache_path=cache_path, cache_lock=cache_lock,
                ): j
                for j in jobs
            }
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Worker crashed for %s::%s", j.source, j.column)
                    errors_list.append({
                        "source": j.source, "column": j.column,
                        "error": f"worker_crash: {exc}",
                    })
                    continue
                completed += 1
                if res.cached:
                    cached_count += 1
                if res.error:
                    errors_list.append({
                        "source": res.source, "column": res.column,
                        "error": res.error,
                    })
                with responses_lock:
                    resp_fh.write(json.dumps({
                        "source": res.source,
                        "column": res.column,
                        "distinct_count": res.distinct_count,
                        "truncated": res.truncated,
                        "findings": res.findings,
                        "n_predictions": len(res.predictions),
                        "elapsed_s": round(res.elapsed_s, 3),
                        "cached": res.cached,
                        "error": res.error,
                    }, ensure_ascii=False) + "\n")
                    resp_fh.flush()
                all_predictions.extend(res.predictions)
                if completed % 25 == 0 or completed == len(jobs):
                    logger.info(
                        "Progress: %d/%d columns (cached=%d, preds=%d, errors=%d, elapsed=%.1fs)",
                        completed, len(jobs), cached_count,
                        len(all_predictions), len(errors_list),
                        time.time() - t_start,
                    )

    # --- Persist predictions ---
    with predictions_path.open("w", encoding="utf-8") as fh:
        for p in all_predictions:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    if errors_list:
        with errors_path.open("w", encoding="utf-8") as fh:
            for e in errors_list:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    # --- Score ---
    bench = load_benchmark(args.benchmark)
    bench_sources = {b.source for b in bench.values()}
    metrics = score_predictions(
        predictions=all_predictions,
        bench=bench,
        bench_sources=bench_sources,
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- Summary ---
    summary = {
        "config": {
            "source_dir": str(args.source_dir),
            "benchmark": str(args.benchmark),
            "prompt": str(args.prompt),
            "model": cfg.model,
            "reasoning_effort": cfg.reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "concurrency": args.concurrency,
            "max_sources": args.max_sources,
            "sources_override": args.sources,
        },
        "counts": {
            "sources_processed": len(csv_map),
            "csv_load_failures": len(load_failures),
            "columns_skipped": skipped_columns,
            "columns_processed": completed,
            "columns_cached": cached_count,
            "columns_errored": len(errors_list),
            "predictions_emitted": len(all_predictions),
            "bench_entries": len(bench),
            "bench_sources": len(bench_sources),
        },
        "metrics": metrics,
        "elapsed_s": round(time.time() - t_start, 2),
        "started_at": _dt.datetime.utcfromtimestamp(t_start).isoformat(timespec="seconds") + "Z",
        "finished_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", summary_path)
    print(json.dumps(summary["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
