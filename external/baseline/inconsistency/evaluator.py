"""Run the rulebook over a set of sources and score it (the per-iteration metric).

The scoring math (:func:`score_column`, :func:`aggregate`) is **pure** — it takes
parsed predictions + gold and returns TP/FP/FN — so it is unit-tested without any
network. The detector (what actually calls GPT-5.5) is **injected**, so the loop,
tests, and ``--dry-run`` can swap a stub in for the live model.

Scoring contract (vendored from ``RFT.inconsistency.v1.eval.run_baseline``):
unit = one distinct ``(column, value)``; gold-error values are positives;
``ambiguous`` values are FP-immune; clean columns of the split's sources are
audited so over-flagging is counted as FP.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from external.baseline.inconsistency._task import (
    MAX_DISTINCT,
    build_response_format,
    build_user_message,
    norm,
    parse_findings,
)

MIN_DISTINCT_VALUES = 2

# A detector turns (prompt, source, column, values) into raw model text (or None).
Detector = Callable[[str, str, str, list[tuple[str, int]]], "str | None"]


# ---------------------------------------------------------------------------
# Column value helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None


def column_values(df: pd.DataFrame, column: str) -> list[tuple[str, int]]:
    counts = df[column].value_counts()
    return [(str(v), int(c)) for v, c in counts.items() if str(v).strip() != ""][:MAX_DISTINCT]


def auditable_columns(df: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    out = {}
    for col in df.columns:
        vals = column_values(df, str(col))
        if len(vals) >= MIN_DISTINCT_VALUES:
            out[str(col)] = vals
    return out


# ---------------------------------------------------------------------------
# Pure scoring
# ---------------------------------------------------------------------------


def score_column(
    pred: dict[str, dict[str, Any]],
    gold: dict[str, Any],
    *,
    source: str,
    column: str,
    dataset: str,
    col_kind: str,
) -> dict[str, Any]:
    """Score one column's parsed predictions against its gold. Pure."""
    gerr: dict[str, set[str]] = gold["error"]
    gamb: set[str] = gold["amb"]
    tp = fp = corr_ok = 0
    fps: list[dict[str, Any]] = []
    corr_misses: list[dict[str, Any]] = []
    for nv, info in pred.items():
        pred_corr = info.get("corrected_value")
        if nv in gerr:
            tp += 1
            acc = gerr[nv]
            if pred_corr is not None and norm(pred_corr) in acc:
                corr_ok += 1
            elif acc:
                corr_misses.append({"value": nv, "pred": pred_corr, "acceptable": sorted(acc)})
        elif nv in gamb:
            pass  # FP-immune
        else:
            fp += 1
            fps.append(
                {
                    "source": source,
                    "column": column,
                    "dataset": dataset,
                    "value": nv,
                    "kind": info.get("kind"),
                    "rationale": info.get("rationale"),
                    "pred_correction": pred_corr,
                }
            )
    fns: list[dict[str, Any]] = []
    for nv, acc in gerr.items():
        if nv not in pred:
            fns.append(
                {
                    "source": source,
                    "column": column,
                    "dataset": dataset,
                    "value": nv,
                    "gold_correction": sorted(acc),
                }
            )
    return {
        "source": source,
        "column": column,
        "dataset": dataset,
        "col_kind": col_kind,
        "tp": tp,
        "fp": fp,
        "fn": len(fns),
        "corr_ok": corr_ok,
        "fps": fps,
        "fns": fns,
        "corr_misses": corr_misses,
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-column results into micro metrics (overall + per dataset)."""
    lab = [r for r in results if r["col_kind"] == "labeled"]
    cln = [r for r in results if r["col_kind"] == "clean"]
    tp = sum(r["tp"] for r in lab)
    fp_lab = sum(r["fp"] for r in lab)
    fp_clean = sum(r["fp"] for r in cln)
    fp = fp_lab + fp_clean
    fn = sum(r["fn"] for r in lab)
    corr_ok = sum(r["corr_ok"] for r in lab)
    p_lab, recall, f1_lab = _prf(tp, fp_lab, fn)
    p_all, _, f1_all = _prf(tp, fp, fn)

    by_dataset: dict[str, dict[str, Any]] = {}
    datasets = {r["dataset"] for r in results}
    for ds in sorted(datasets):
        d_lab = [r for r in lab if r["dataset"] == ds]
        d_cln = [r for r in cln if r["dataset"] == ds]
        d_tp = sum(r["tp"] for r in d_lab)
        d_fp = sum(r["fp"] for r in d_lab) + sum(r["fp"] for r in d_cln)
        d_fn = sum(r["fn"] for r in d_lab)
        d_p, d_r, d_f = _prf(d_tp, d_fp, d_fn)
        by_dataset[ds] = {
            "tp": d_tp,
            "fp": d_fp,
            "fn": d_fn,
            "precision": d_p,
            "recall": d_r,
            "f1": d_f,
        }
    return {
        "tp": tp,
        "fp": fp,
        "fp_labeled": fp_lab,
        "fp_clean": fp_clean,
        "fn": fn,
        "precision_labeled": p_lab,
        "precision_incl_clean": p_all,
        "recall": recall,
        "f1_labeled": f1_lab,
        "f1_incl_clean": f1_all,
        "correction_accuracy": (corr_ok / tp if tp else 0.0),
        "n_labeled_cols": len(lab),
        "n_clean_cols": len(cln),
        "by_dataset": by_dataset,
    }


# ---------------------------------------------------------------------------
# Orchestration (detector injected)
# ---------------------------------------------------------------------------


def _build_jobs(sources, cols, source_dataset, csv_map, *, include_clean, exclusions=None):
    """Yield (source, column, gold, values, col_kind, dataset).

    ``exclusions`` (a :class:`external.baseline.inconsistency.data.Exclusions`, optional)
    drops curator-ignored CSVs/columns entirely and folds skipped-cell values
    into each remaining column's FP-immune set.
    """
    labeled_keys = set(cols)
    jobs = []
    frames: dict[str, pd.DataFrame] = {}

    def _frame(source):
        if source not in frames:
            p = csv_map.get(source)
            frames[source] = _read_csv(p) if p else None
        return frames[source]

    def _gold_with_immunity(source, column, gold):
        """Add skipped-cell values to ``amb`` (FP-immune) without touching errors."""
        if not exclusions:
            return gold
        immune = exclusions.immune_values(source, column)
        if not immune:
            return gold
        amb = set(gold["amb"]) | {v for v in immune if v not in gold["error"]}
        return {"error": gold["error"], "amb": amb}

    for (source, column), gold in sorted(cols.items()):
        if source not in sources:
            continue
        if exclusions and exclusions.drops_column(source, column):
            continue
        df = _frame(source)
        if df is None or column not in df.columns:
            continue
        vals = column_values(df, column)
        if not vals:
            continue
        job_gold = _gold_with_immunity(source, column, gold)
        jobs.append((source, column, job_gold, vals, "labeled", source_dataset.get(source, "?")))

    if include_clean:
        for source in sorted(sources):
            df = _frame(source)
            if df is None:
                continue
            for column, vals in sorted(auditable_columns(df).items()):
                if (source, column) in labeled_keys:
                    continue
                if exclusions and exclusions.drops_column(source, column):
                    continue
                job_gold = _gold_with_immunity(source, column, {"error": {}, "amb": set()})
                jobs.append(
                    (source, column, job_gold, vals, "clean", source_dataset.get(source, "?"))
                )
    return jobs


def evaluate(
    prompt: str,
    sources: set[str],
    cols: dict[tuple[str, str], dict[str, Any]],
    source_dataset: dict[str, str],
    csv_map: dict[str, Path],
    detector: Detector,
    *,
    include_clean: bool = True,
    concurrency: int = 8,
    evidence_cap: int = 60,
    exclusions: Any = None,
) -> dict[str, Any]:
    """Run ``detector`` over ``sources`` with ``prompt`` and score the result."""
    jobs = _build_jobs(
        sources,
        cols,
        source_dataset,
        csv_map,
        include_clean=include_clean,
        exclusions=exclusions,
    )

    def _run(job):
        source, column, gold, vals, kind, ds = job
        raw = detector(prompt, source, column, vals)
        pred = parse_findings(raw)
        return score_column(pred, gold, source=source, column=column, dataset=ds, col_kind=kind)

    results: list[dict[str, Any]] = []
    if concurrency <= 1:
        results = [_run(j) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_run, j) for j in jobs]
            for f in as_completed(futs):
                results.append(f.result())

    metrics = aggregate(results)
    fp_examples = [x for r in results for x in r["fps"]]
    fn_examples = [x for r in results for x in r["fns"]]
    corr_misses = [
        {"source": r["source"], "column": r["column"], **x}
        for r in results
        for x in r["corr_misses"]
    ]
    return {
        "metrics": metrics,
        "evidence": {
            "fp": fp_examples[:evidence_cap],
            "fn": fn_examples[:evidence_cap],
            "corr_misses": corr_misses[:evidence_cap],
            "n_fp_total": len(fp_examples),
            "n_fn_total": len(fn_examples),
        },
        "n_jobs": len(jobs),
    }


# ---------------------------------------------------------------------------
# Live GPT-5.5 detector (cached); not imported by the pure-scoring tests
# ---------------------------------------------------------------------------


class Gpt55Detector:
    """Calls GPT-5.5 via chat.completions at ``effort``/``temperature``, cached.

    Cache key = sha(prompt | model | effort | temp | source | column), so editing
    the rulebook auto-invalidates and re-runs only what changed.
    """

    def __init__(
        self,
        clients: list[Any],
        *,
        model: str = "gpt-5.5",
        effort: str = "none",
        temperature: float | None = 0.0,
        cache_dir: Path | None = None,
        max_retries: int = 6,
    ) -> None:
        self.clients = clients
        self.model = model
        self.effort = effort
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.response_format = build_response_format()
        self.max_retries = max_retries
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _client_for(self, source: str, column: str):
        idx = int(hashlib.sha256(f"{source}|{column}".encode()).hexdigest(), 16) % len(self.clients)
        return self.clients[idx]

    def _key(self, prompt: str, source: str, column: str) -> str:
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        h.update(f"|{self.model}|{self.effort}|t={self.temperature}|{source}|{column}".encode())
        return h.hexdigest()[:16]

    def is_cached(self, prompt: str, source: str, column: str) -> bool:
        """True if a cached response exists for this (prompt, source, column)."""
        if not self.cache_dir:
            return False
        return (self.cache_dir / f"{self._key(prompt, source, column)}.json").exists()

    def __call__(self, prompt, source, column, values):
        cache_path = (
            self.cache_dir / f"{self._key(prompt, source, column)}.json" if self.cache_dir else None
        )
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text()).get("raw")
        messages = [
            {"role": "developer", "content": prompt},
            {
                "role": "user",
                "content": build_user_message(
                    column=column, source=source, values_with_counts=values
                ),
            },
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": self.response_format,
        }
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        client = self._client_for(source, column)
        raw = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content
                break
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if ("429" in msg or "RateLimit" in msg) and attempt < self.max_retries - 1:
                    time.sleep(min(60, 5 * 2**attempt))
                    continue
                break
        if cache_path and raw is not None:
            cache_path.write_text(json.dumps({"raw": raw}))
        return raw
