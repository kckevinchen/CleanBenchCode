"""Compare RFT v5 fine-tuned models against the unfine-tuned o4-mini.

For each row in ``RFT/typo/v5/data/eval.jsonl`` and for each candidate
model deployment we:

  1. Send the row's ``messages`` to the deployment via
     ``chat.completions.create`` (with the v5 response_format schema).
  2. Apply the SHIPPED Python gates locally:
     * ``trivial`` -- zero if any prediction is a CLEAN edit shape.
     * ``column_cap`` -- zero if the model proposed more than
       ``max(5, ceil(0.30 * num_distinct))`` typos.
  3. Score the response_with the ``score_model_extras`` leg using
     ``LLMClient`` (gpt-5 pool by default; stand-in for the
     gpt-4o-2024-08-06 grader Azure used at RFT time).
  4. Compose reward = ``5 * trivial * column_cap * extras``  (matches
     the RFT calculate_output).
  5. Compute cell-level Precision/Recall/F1 against ``gold_typos``,
     EXCLUDING ``ambiguous_values`` from FP penalty.

Outputs (under ``--out-dir``):

  * ``{model}.rollouts.jsonl`` -- per-row records
  * ``summary.csv``           -- per-model aggregates
  * ``summary.md``            -- markdown table + delta vs base
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common._rft_common import build_client  # noqa: E402
from RFT.typo.v5 import _task as t  # noqa: E402
from src.core.llm_client import LLMClient, configured_endpoint_pool  # noqa: E402

logger = logging.getLogger("compare_ft_vs_base")

API_VERSION = "2025-04-01-preview"

DEFAULT_EVAL = "RFT/typo/v5/data/eval.jsonl"


# ---------------------------------------------------------------------------
# Model registry: each "condition" is (label, endpoint, deployment_name)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    label: str
    endpoint: str
    deployment: str
    # "chat"   -> client.chat.completions.create (o4-mini & FT variants)
    # "responses" -> client.responses.create (gpt-5 family)
    kind: str = "chat"


def _configured_endpoint(index: int = 0) -> str:
    """Return a positional configured endpoint, falling back to the first."""
    endpoints = configured_endpoint_pool()
    if not endpoints:
        return ""
    return endpoints[index] if index < len(endpoints) else endpoints[0]


DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(
        label="base",
        endpoint=_configured_endpoint(0),
        deployment="o4-mini",
    ),
    ModelSpec(
        label="longer",
        endpoint=_configured_endpoint(0),
        deployment="o4-mini-v5-longer",
    ),
    ModelSpec(
        label="sharper",
        endpoint=_configured_endpoint(1),
        deployment="o4-mini-v5-sharper",
    ),
    # GPT-5 baselines (same prompt, Responses API).
    ModelSpec(
        label="gpt-5.2",
        endpoint=_configured_endpoint(0),
        deployment="gpt-5.2",
        kind="responses",
    ),
    ModelSpec(
        label="gpt-5.5",
        endpoint=_configured_endpoint(0),
        deployment="gpt-5.5",
        kind="responses",
    ),
]


# ---------------------------------------------------------------------------
# Local grader plumbing (executes the SHIPPED Python source, not a copy)
# ---------------------------------------------------------------------------


def _build_local_grader(source: str):
    ns: dict[str, Any] = {}
    exec(compile(source, "<v5-grader>", "exec"), ns)
    return ns["grade"]


_trivial_grade = _build_local_grader(t.TYPO_V5_PYTHON_TRIVIAL_GRADER_SOURCE)
_column_cap_grade = _build_local_grader(t.TYPO_V5_PYTHON_COLUMN_CAP_GRADER_SOURCE)


_PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")


def _render_template(template: str, mapping: dict[str, str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in mapping:
            raise KeyError(f"unresolved placeholder {{{{ {key} }}}}")
        return mapping[key]
    return _PLACEHOLDER_RE.sub(_sub, template)


_SCORE_LINE_RE = re.compile(
    r"^\s*score\s*[:=]\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_extras_score(text: str) -> float | None:
    m = _SCORE_LINE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def run_extras_grader(
    llm: LLMClient,
    grader_models: str,
    grader_spec: dict[str, Any],
    item: dict[str, Any],
    output_text: str,
) -> tuple[float, str | None]:
    """Score the score_model_extras grader via LLMClient (gpt-5 pool)."""
    msgs = grader_spec["input"]
    dev = next(m["content"] for m in msgs if m["role"] == "developer")
    usr_t = next(m["content"] for m in msgs if m["role"] == "user")
    mapping = {
        "item.column": item["column"],
        "item.source": item["source"],
        "item.gold_typos": item["gold_typos"],
        "item.ambiguous_values": item["ambiguous_values"],
        "item.column_values_json": item["column_values_json"],
        "sample.output_text": output_text,
    }
    user = _render_template(usr_t, mapping)
    resp = llm.query(
        model=grader_models,
        prompt=user,
        system_prompt=dev,
        reasoning_effort="low",
    )
    if not resp.ok:
        return 0.0, f"<extras grader failed: {resp.error}>"
    score = _parse_extras_score(resp.text)
    if score is None:
        return 0.0, resp.text
    return max(0.0, min(1.0, float(score))), resp.text


# ---------------------------------------------------------------------------
# Cell-level metric vs gold_typos (excluding ambiguous_values from FP)
# ---------------------------------------------------------------------------


def _cell_metrics(
    predicted: list[dict[str, Any]],
    gold_typos_json: str,
    ambiguous_values_json: str,
) -> dict[str, float | int]:
    try:
        gold_list = json.loads(gold_typos_json) if gold_typos_json else []
        amb_list = (
            json.loads(ambiguous_values_json) if ambiguous_values_json else []
        )
    except json.JSONDecodeError:
        gold_list, amb_list = [], []

    gold_values = {(g.get("value") or "").strip() for g in gold_list}
    amb_values = {(str(v) or "").strip() for v in amb_list}
    pred_values = {(p.get("value") or "").strip() for p in predicted}
    pred_values.discard("")

    tp = len(pred_values & gold_values)
    # FP excludes anything that's flagged as ambiguous (model couldn't
    # have known) and anything in gold.
    fp = len(pred_values - gold_values - amb_values)
    fn = len(gold_values - pred_values)
    p = tp / (tp + fp) if (tp + fp) else (1.0 if not gold_values else 0.0)
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _chat_response_format_to_responses_text_format(
    chat_rf: dict[str, Any],
) -> dict[str, Any]:
    """Convert chat-API ``response_format`` to Responses-API ``text.format``.

    Chat shape:    ``{type: 'json_schema', json_schema: {name, schema, strict}}``
    Responses:     ``{type: 'json_schema', name, schema, strict}``  (flat)
    """
    if chat_rf.get("type") != "json_schema":
        return chat_rf
    inner = chat_rf.get("json_schema", {})
    return {
        "type": "json_schema",
        "name": inner.get("name", "output"),
        "schema": inner.get("schema", {}),
        "strict": inner.get("strict", True),
    }


def _extract_responses_output_text(resp: Any) -> str:
    """Pull plain text out of a Responses-API result."""
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    # Fallback: walk .output -> message -> content -> text
    out = getattr(resp, "output", None) or []
    parts: list[str] = []
    for item in out:
        content = getattr(item, "content", None) or []
        for c in content:
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)


def call_deployment(
    client,
    spec: ModelSpec,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    reasoning_effort: str,
) -> tuple[str | None, str | None]:
    """Return (raw_text, error). Dispatches on ``spec.kind``."""
    if spec.kind == "responses":
        # Same prompt material as the chat path, just routed through
        # the Responses API (gpt-5 family doesn't expose chat.completions
        # with reasoning_effort + response_format in the same shape).
        instructions = ""
        user_input = ""
        for m in messages:
            role = m.get("role")
            if role in ("system", "developer"):
                instructions = m.get("content", "")
            elif role == "user":
                user_input = m.get("content", "")
        try:
            resp = client.responses.create(
                model=spec.deployment,
                instructions=instructions,
                input=user_input,
                text={"format": _chat_response_format_to_responses_text_format(response_format)},
                reasoning={"effort": reasoning_effort},
            )
            return _extract_responses_output_text(resp), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    try:
        resp = client.chat.completions.create(
            model=spec.deployment,
            messages=messages,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        return resp.choices[0].message.content, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Per-row, per-model evaluation
# ---------------------------------------------------------------------------


@dataclass
class RowResult:
    row_idx: int
    model: str
    raw_text: str | None
    error: str | None
    pred_count: int
    trivial: float
    column_cap: float
    extras: float
    reward: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    latency_s: float

    def asdict(self) -> dict[str, Any]:
        return self.__dict__


def evaluate_one(
    spec: ModelSpec,
    client,
    grader_spec: dict[str, Any],
    response_format: dict[str, Any],
    llm: LLMClient,
    grader_models: str,
    reasoning_effort: str,
    row_idx: int,
    row: dict[str, Any],
) -> RowResult:
    messages = row["messages"]
    item = {
        "column": row["column"],
        "source": row["source"],
        "gold_typos": row["gold_typos"],
        "ambiguous_values": row["ambiguous_values"],
        "column_values_json": row["column_values_json"],
    }
    t0 = time.time()
    raw_text, err = call_deployment(
        client, spec, messages, response_format, reasoning_effort,
    )
    latency = time.time() - t0
    parsed = _parse_json_obj(raw_text) if raw_text else None
    predicted = parsed.get("typos", []) if isinstance(parsed, dict) else []
    if not isinstance(predicted, list):
        predicted = []
    sample = {"output_text": raw_text or ""}

    # Local gates -- mirror Azure sandbox call signature.
    try:
        trivial = float(_trivial_grade(sample, item))
    except Exception:
        trivial = 0.0
    try:
        column_cap = float(_column_cap_grade(sample, item))
    except Exception:
        column_cap = 0.0

    # Extras leg: only call if both gates open; otherwise reward=0 anyway.
    extras = 0.0
    if raw_text and trivial > 0.0 and column_cap > 0.0:
        extras, _ = run_extras_grader(
            llm, grader_models, grader_spec, item, raw_text,
        )

    reward = 5.0 * trivial * column_cap * extras
    metrics = _cell_metrics(
        predicted, row["gold_typos"], row["ambiguous_values"],
    )
    return RowResult(
        row_idx=row_idx,
        model=spec.label,
        raw_text=raw_text,
        error=err,
        pred_count=len(predicted),
        trivial=trivial,
        column_cap=column_cap,
        extras=extras,
        reward=reward,
        tp=metrics["tp"], fp=metrics["fp"], fn=metrics["fn"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        latency_s=latency,
    )


# ---------------------------------------------------------------------------
# Aggregation + summary
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _bootstrap_ci(
    paired_deltas: list[float], n: int = 1000, seed: int = 0,
) -> tuple[float, float]:
    if not paired_deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means: list[float] = []
    N = len(paired_deltas)
    for _ in range(n):
        s = sum(paired_deltas[rng.randrange(N)] for _ in range(N))
        means.append(s / N)
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return (lo, hi)


def summarise(
    by_model: dict[str, list[RowResult]],
    base_label: str = "base",
) -> list[dict[str, Any]]:
    rows = []
    base = by_model.get(base_label, [])
    base_by_idx = {r.row_idx: r for r in base}
    for label, results in by_model.items():
        rewards = [r.reward for r in results]
        f1s = [r.f1 for r in results]
        ps = [r.precision for r in results]
        rs = [r.recall for r in results]
        trivial_fp = sum(1 for r in results if r.trivial == 0.0) / max(1, len(results))
        column_cap_hit = sum(1 for r in results if r.column_cap == 0.0) / max(1, len(results))
        empty_pred = sum(1 for r in results if r.pred_count == 0) / max(1, len(results))
        errors = sum(1 for r in results if r.error) / max(1, len(results))
        row = {
            "model": label,
            "n": len(results),
            "mean_reward": _mean(rewards),
            "mean_f1": _mean(f1s),
            "mean_precision": _mean(ps),
            "mean_recall": _mean(rs),
            "trivial_fp_rate": trivial_fp,
            "column_cap_hit_rate": column_cap_hit,
            "empty_prediction_rate": empty_pred,
            "error_rate": errors,
            "mean_latency_s": _mean([r.latency_s for r in results]),
        }
        if label != base_label and base:
            # Paired bootstrap on rewards + F1 deltas
            common = [i for i in base_by_idx if any(rr.row_idx == i for rr in results)]
            by_idx = {r.row_idx: r for r in results}
            d_reward = [by_idx[i].reward - base_by_idx[i].reward for i in common]
            d_f1 = [by_idx[i].f1 - base_by_idx[i].f1 for i in common]
            lo_r, hi_r = _bootstrap_ci(d_reward)
            lo_f, hi_f = _bootstrap_ci(d_f1)
            row["delta_reward_vs_base"] = _mean(d_reward)
            row["delta_reward_ci95"] = f"[{lo_r:+.3f}, {hi_r:+.3f}]"
            row["delta_f1_vs_base"] = _mean(d_f1)
            row["delta_f1_ci95"] = f"[{lo_f:+.3f}, {hi_f:+.3f}]"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def load_eval(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def run(
    models: list[ModelSpec],
    eval_path: Path,
    out_dir: Path,
    grader_models: str,
    reasoning_effort: str,
    limit: int | None,
    workers: int,
) -> None:
    if any(not model.endpoint for model in models):
        raise ValueError("configure AZURE_OPENAI_ENDPOINTS before running comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_eval(eval_path, limit=limit)
    logger.info("Loaded %d eval rows from %s", len(rows), eval_path)

    grader_spec = t.build_score_model_extras_grader()
    response_format = t.build_response_format()
    llm = LLMClient()

    # One AzureOpenAI client per unique endpoint
    clients = {ep: build_client(ep, API_VERSION) for ep in {m.endpoint for m in models}}

    by_model: dict[str, list[RowResult]] = {m.label: [] for m in models}

    for spec in models:
        logger.info("=== Evaluating %s (%s @ %s)", spec.label, spec.deployment, spec.endpoint)
        client = clients[spec.endpoint]
        results: list[RowResult] = [None] * len(rows)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    evaluate_one,
                    spec, client, grader_spec, response_format,
                    llm, grader_models, reasoning_effort,
                    i, r,
                ): i
                for i, r in enumerate(rows)
            }
            done = 0
            t0 = time.time()
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    results[idx] = RowResult(
                        row_idx=idx, model=spec.label, raw_text=None,
                        error=f"future_exception: {exc}", pred_count=0,
                        trivial=0.0, column_cap=0.0, extras=0.0, reward=0.0,
                        tp=0, fp=0, fn=0, precision=0.0, recall=0.0, f1=0.0,
                        latency_s=0.0,
                    )
                done += 1
                if done % 20 == 0 or done == len(rows):
                    rate = done / max(1e-6, time.time() - t0)
                    logger.info(
                        "  %s: %d/%d  rate=%.2f rows/s", spec.label, done, len(rows), rate,
                    )
        by_model[spec.label] = results

        # Write rollouts for this model
        out_path = out_dir / f"{spec.label}.rollouts.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.asdict(), ensure_ascii=False) + "\n")
        logger.info("  wrote %s", out_path)

    # Summary
    summary = summarise(by_model, base_label="base")
    summary_csv = out_dir / "summary.csv"
    fieldnames = sorted({k for row in summary for k in row.keys()})
    with summary_csv.open("w", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary:
            w.writerow(row)
    logger.info("Wrote %s", summary_csv)

    # Markdown
    md_lines = ["# v5 RFT comparison", "", f"Eval rows: {len(rows)}", ""]
    md_lines.append("| model | reward | F1 | P | R | trivFP% | capHit% | emptyPred% | Δreward (95% CI) | ΔF1 (95% CI) |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in summary:
        md_lines.append(
            f"| {row['model']} | {row['mean_reward']:.3f} | {row['mean_f1']:.3f} | "
            f"{row['mean_precision']:.3f} | {row['mean_recall']:.3f} | "
            f"{row['trivial_fp_rate']*100:.1f} | {row['column_cap_hit_rate']*100:.1f} | "
            f"{row['empty_prediction_rate']*100:.1f} | "
            f"{row.get('delta_reward_vs_base', 0):+.3f} {row.get('delta_reward_ci95', '')} | "
            f"{row.get('delta_f1_vs_base', 0):+.3f} {row.get('delta_f1_ci95', '')} |"
        )
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", md_path)
    print("\n".join(md_lines))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", default=DEFAULT_EVAL, type=Path)
    p.add_argument("--out-dir", default="outputs/v5_compare", type=Path)
    p.add_argument(
        "--grader-models",
        default="gpt-5.2,gpt-5.4,gpt-5.5",
        help="Comma-separated LLMClient pool models for the extras grader leg.",
    )
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of labels (e.g. base,longer).",
    )
    args = p.parse_args(argv)

    models = list(DEFAULT_MODELS)
    if args.only:
        keep = set(args.only.split(","))
        models = [m for m in models if m.label in keep]
        if "base" not in keep:
            logger.warning(
                "base excluded -- delta-vs-base columns will be omitted.",
            )

    run(
        models=models,
        eval_path=args.eval,
        out_dir=args.out_dir,
        grader_models=args.grader_models,
        reasoning_effort=args.reasoning_effort,
        limit=args.limit,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
