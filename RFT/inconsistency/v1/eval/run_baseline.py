"""Baseline / RFT eval + deterministic-grader test for the inconsistency rulebook.

Runs the v1 trainee prompt (the rulebook) over benchmark columns in the
whole-column framing, then scores its ``findings`` against the gold with
the deterministic grader, reporting precision / recall / F1 + correction
accuracy.

Scoring (the deterministic leg of the dual grader)
--------------------------------------------------
Unit = one distinct ``(column, value)``. Restricted to the benchmark's
labeled columns (no "outside bench source" confound); CLEAN columns of
the same sources are also audited so over-flagging is counted.

  G  gold-error values   = values with >=1 ``label=="error"`` record
  A  ambiguous values    = values with >=1 ``label=="ambiguous"`` record
                           (and not in G); FP-IMMUNE
  P  predicted values    = the ``value`` fields the trainee flagged

  TP = |P ∩ G|     FP = |P ∖ (G ∪ A)|     FN = |G ∖ P|
  precision = TP/(TP+FP)   recall = TP/(TP+FN)   F1 = 2PR/(P+R)

Use ``--model`` to point at a deployment (base ``o4-mini`` vs the RFT
deployment), and ``--sources`` to restrict to a held-out source set
(e.g. the eval split from ``data/manifest.json``).

Usage::

    python -m RFT.inconsistency.v1.eval.run_baseline --model o4-mini \
        --sources RFT/inconsistency/v1/data/manifest.json --tag base_eval
    python -m RFT.inconsistency.v1.eval.run_baseline --model o4-mini-inc-v1c \
        --sources RFT/inconsistency/v1/data/manifest.json --tag rft_eval
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common._rft_common import build_client  # noqa: E402
from RFT.inconsistency.v1 import (  # noqa: E402
    PROMPT_INCONSISTENCY,
    build_response_format,
    build_user_message,
)

API_VERSION = "2025-04-01-preview"
ENDPOINT = "https://dmxaz-test-aoai-eu2.openai.azure.com/"

BENCHMARK = (
    _PROJECT_ROOT / "benchmark"
    / "benchmark_inconsistency_20260515T010112Z_gpt5-family.jsonl"
)
SOURCES_DIR = _PROJECT_ROOT / "data" / "source_inconsistency_bing21"
OUT_DIR = _PROJECT_ROOT / "outputs" / "inconsistency_v1_baseline"

MIN_DISTINCT_VALUES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s) -> str:
    return " ".join(str(s).split())


def _acceptable_set(rec) -> set[str]:
    out: set[str] = set()
    cv = rec.get("corrected_value")
    items = cv if isinstance(cv, list) else [cv]
    for x in items:
        if isinstance(x, str) and x.strip():
            out.add(_norm(x))
    sv = rec.get("llm_suggested_value")
    if isinstance(sv, str) and sv.strip():
        out.add(_norm(sv))
    return out


def _prompt_hash(model: str, effort: str, temperature=None) -> str:
    h = hashlib.sha256()
    h.update(PROMPT_INCONSISTENCY.encode("utf-8"))
    h.update(f"|{model}|{effort}|t={temperature}|".encode("utf-8"))
    return h.hexdigest()[:12]


def _load_gold(benchmark_path=None):
    bench = Path(benchmark_path) if benchmark_path else BENCHMARK
    err = collections.defaultdict(dict)   # (s,c) -> {nvalue: set(acc)}
    amb = collections.defaultdict(set)     # (s,c) -> {nvalue}
    for line in bench.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["source"], r["column"])
        nv = _norm(r["original_value"])
        if r.get("label") == "error":
            err[key].setdefault(nv, set()).update(_acceptable_set(r))
        elif r.get("label") == "ambiguous":
            amb[key].add(nv)
    cols = {}
    for key in set(err) | set(amb):
        e = dict(err.get(key, {}))
        a = {v for v in amb.get(key, set()) if v not in e}
        cols[key] = {"error": e, "amb": a}
    return cols


def _resolve_sources(spec: str):
    """Return a set of source basenames to keep, or None for all.

    ``spec`` may be a comma-separated list of basenames, or a path to a
    JSON file -- either a manifest with ``sources.eval`` or a bare list.
    """
    if not spec:
        return None
    if spec.endswith(".json"):
        d = json.loads(Path(spec).read_text())
        if isinstance(d, dict) and isinstance(d.get("sources"), dict):
            return set(d["sources"].get("eval", []))
        if isinstance(d, list):
            return set(d)
        return set()
    return {x.strip() for x in spec.split(",") if x.strip()}


def _csv_lookup() -> dict[str, Path]:
    return {p.name: p for p in SOURCES_DIR.rglob("*.csv")}


def _column_values_from_df(df, column):
    counts = df[column].value_counts()
    return [(str(v), int(c)) for v, c in counts.items() if str(v).strip() != ""]


def _column_values(csv_path: Path, column: str):
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return None
    if column not in df.columns:
        return None
    return _column_values_from_df(df, column)


def _auditable_columns(csv_path: Path):
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for col in df.columns:
        vals = _column_values_from_df(df, col)
        if len(vals) >= MIN_DISTINCT_VALUES:
            out[str(col)] = vals
    return out


def _parse_findings(raw: str | None):
    if not raw:
        return {}
    s = raw.strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return {}
    out = {}
    for it in obj.get("findings", []) or []:
        v = it.get("value")
        if isinstance(v, str):
            out[_norm(v)] = it.get("corrected_value")
    return out


# ---------------------------------------------------------------------------
# Per-column run + score
# ---------------------------------------------------------------------------

def run_column(client, model, effort, response_format, source, column,
               gold, values, cache_dir, col_kind, temperature=None):
    cache_key = hashlib.sha256(
        f"{_prompt_hash(model, effort, temperature)}|{source}|{column}".encode("utf-8")
    ).hexdigest()[:16]
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        raw, err = cached.get("raw"), cached.get("error")
    else:
        messages = [
            {"role": "developer", "content": PROMPT_INCONSISTENCY},
            {"role": "user", "content": build_user_message(column, source, values)},
        ]
        raw, err = None, None
        kwargs = dict(model=model, messages=messages, response_format=response_format)
        if effort:
            kwargs["reasoning_effort"] = effort
        if temperature is not None:
            kwargs["temperature"] = temperature
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                if ("429" in err or "RateLimit" in err) and attempt < 5:
                    time.sleep(min(60, 5 * 2 ** attempt))  # 5,10,20,40,60,60
                    continue
                break
        # Only cache successful responses; cached errors would never retry.
        if raw is not None:
            cache_path.write_text(json.dumps({"raw": raw, "error": None}))

    pred = _parse_findings(raw)
    gerr, gamb = gold["error"], gold["amb"]

    tp = fp = corr_ok = 0
    fps, corr_misses = [], []
    for nv, pred_corr in pred.items():
        if nv in gerr:
            tp += 1
            acc = gerr[nv]
            if pred_corr is not None and _norm(pred_corr) in acc:
                corr_ok += 1
            else:
                corr_misses.append({"value": nv, "pred": pred_corr,
                                    "acceptable": sorted(acc)})
        elif nv in gamb:
            pass  # FP-immune
        else:
            fp += 1
            fps.append({"source": source, "column": column,
                        "value": nv, "pred_correction": pred_corr})
    fn = sum(1 for nv in gerr if nv not in pred)

    return {
        "source": source, "column": column, "error": err,
        "col_kind": col_kind,
        "n_pred": len(pred), "n_gold_err": len(gerr), "n_amb": len(gamb),
        "tp": tp, "fp": fp, "fn": fn, "corr_ok": corr_ok,
        "fps": fps, "corr_misses": corr_misses,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="o4-mini")
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--endpoints", default="",
                    help="Comma-separated endpoint pool for round-robin "
                         "(spreads calls to raise aggregate TPM; overrides "
                         "--endpoint when set).")
    ap.add_argument("--reasoning-effort", default="medium",
                    help="'' to omit (non-reasoning models).")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature; set 0 for deterministic, "
                         "comparable runs (gpt-5.5 is wildly stochastic at "
                         "the default temp=1).")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--labeled-only", action="store_true",
                    help="Skip clean (unlabeled) columns of benchmark sources.")
    ap.add_argument("--sources", default="",
                    help="Restrict to these sources: comma-separated basenames, "
                         "or a manifest.json path (uses sources.eval).")
    ap.add_argument("--benchmark", default="",
                    help="Override the benchmark JSONL (re-score cached "
                         "predictions against a different gold set).")
    args = ap.parse_args()
    include_clean = not args.labeled_only
    src_filter = _resolve_sources(args.sources)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)

    cols = _load_gold(args.benchmark or None)
    csv_map = _csv_lookup()
    response_format = build_response_format()
    eps = [e.strip() for e in (args.endpoints or args.endpoint).split(",") if e.strip()]
    clients = [build_client(ep, API_VERSION) for ep in eps]
    print(f"endpoint pool: {len(clients)}")

    EMPTY_GOLD = {"error": {}, "amb": set()}
    jobs, skipped = [], []      # (source, column, gold, values, col_kind)
    labeled_keys = set(cols)

    for (source, column), gold in sorted(cols.items()):
        if src_filter is not None and source not in src_filter:
            continue
        csv_path = csv_map.get(source)
        if csv_path is None:
            skipped.append((source, column, "no_csv"))
            continue
        values = _column_values(csv_path, column)
        if not values:
            skipped.append((source, column, "no_column"))
            continue
        jobs.append((source, column, gold, values, "labeled"))

    if include_clean:
        for source in sorted({s for s, _ in cols}):
            if src_filter is not None and source not in src_filter:
                continue
            csv_path = csv_map.get(source)
            if csv_path is None:
                continue
            for column, values in sorted(_auditable_columns(csv_path).items()):
                if (source, column) in labeled_keys:
                    continue
                jobs.append((source, column, EMPTY_GOLD, values, "clean"))

    if args.limit:
        jobs = jobs[: args.limit]

    ph = _prompt_hash(args.model, args.reasoning_effort, args.temperature)
    n_gold = sum(len(g["error"]) for _, _, g, _, _ in jobs)
    n_lab = sum(1 for j in jobs if j[4] == "labeled")
    n_clean = sum(1 for j in jobs if j[4] == "clean")
    print(f"prompt_hash={ph} model={args.model} effort={args.reasoning_effort!r} "
          f"labeled_cols={n_lab} clean_cols={n_clean} skipped={len(skipped)} "
          f"gold_err_values={n_gold} src_filter={'none' if src_filter is None else len(src_filter)}")

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        def _ep(s, c):  # stable column->endpoint map (paired across runs)
            return clients[int(hashlib.sha256(f"{s}|{c}".encode()).hexdigest(),
                               16) % len(clients)]
        futs = {
            ex.submit(run_column, _ep(s, c), args.model,
                      args.reasoning_effort, response_format, s, c, g, v,
                      cache_dir, k, args.temperature): (s, c)
            for (s, c, g, v, k) in jobs
        }
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} columns ({time.time()-t0:.0f}s)")

    lab = [r for r in results if r["col_kind"] == "labeled"]
    cln = [r for r in results if r["col_kind"] == "clean"]
    TP = sum(r["tp"] for r in lab)
    FP_lab = sum(r["fp"] for r in lab)
    FP_clean = sum(r["fp"] for r in cln)
    FP = FP_lab + FP_clean
    FN = sum(r["fn"] for r in lab)
    CORR = sum(r["corr_ok"] for r in lab)
    errs = [r for r in results if r["error"]]
    clean_cols_flagged = sum(1 for r in cln if r["fp"] > 0)

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    P_lab, R, F1_lab = _prf(TP, FP_lab, FN)
    P_all, _, F1_all = _prf(TP, FP, FN)
    corr_acc = CORR / TP if TP else 0.0

    print("\n" + "=" * 64)
    print("DETECTION (distinct-value, ambiguous-immune)")
    print(f"  TP={TP}  FN={FN}  recall={R:.3f}")
    print(f"  FP labeled-cols={FP_lab}  FP clean-cols={FP_clean}  FP total={FP}")
    print(f"  precision (labeled cols only) = {P_lab:.3f}   F1={F1_lab:.3f}")
    print(f"  precision (incl. clean cols)  = {P_all:.3f}   F1={F1_all:.3f}")
    print(f"  clean cols flagged >=1:        {clean_cols_flagged}/{len(cln)} "
          f"= {(clean_cols_flagged/len(cln) if cln else 0):.2f}  (false-alarm rate)")
    print(f"CORRECTION accuracy (TP w/ acceptable fix): {CORR}/{TP} = {corr_acc:.3f}")
    print(f"API errors (columns): {len(errs)}")
    print("=" * 64)

    tag = (args.tag + "_") if args.tag else ""
    (OUT_DIR / f"{tag}cols_{ph}.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in r.items()
                              if k not in ("fps", "corr_misses")})
                  for r in results) + "\n")
    with (OUT_DIR / f"{tag}fp_{ph}.jsonl").open("w") as f:
        for r in results:
            for x in r["fps"]:
                f.write(json.dumps(x) + "\n")
    with (OUT_DIR / f"{tag}corr_misses_{ph}.jsonl").open("w") as f:
        for r in results:
            for x in r["corr_misses"]:
                f.write(json.dumps({"source": r["source"], "column": r["column"],
                                    **x}) + "\n")
    summary = {
        "prompt_hash": ph, "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "sources_filter": (None if src_filter is None else sorted(src_filter)),
        "n_labeled_cols": len(lab), "n_clean_cols": len(cln),
        "skipped": skipped,
        "TP": TP, "FN": FN, "FP_labeled": FP_lab, "FP_clean": FP_clean,
        "FP_total": FP,
        "recall": R,
        "precision_labeled_only": P_lab, "f1_labeled_only": F1_lab,
        "precision_incl_clean": P_all, "f1_incl_clean": F1_all,
        "clean_cols_flagged": clean_cols_flagged,
        "correction_accuracy": corr_acc, "api_errors": len(errs),
    }
    (OUT_DIR / f"{tag}summary_{ph}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nartifacts -> {OUT_DIR}  (summary_{ph}.json, fp_{ph}.jsonl)")


if __name__ == "__main__":
    main()
