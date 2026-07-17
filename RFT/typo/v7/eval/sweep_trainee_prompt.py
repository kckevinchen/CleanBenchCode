"""Sweep candidate trainee prompts against the v7 eval set.

Goal: find a shorter, recall-leaning developer prompt for the trainee
(o4-mini) that matches or beats gpt-5.5's recall on the v7 eval split.

For each (prompt_id, model) we:
  1. Read RFT/typo/v7/data/eval.jsonl
  2. Replace the developer message with the candidate prompt; keep
     the existing user message (column header + value listing) intact.
  3. Call the deployment with the v5 strict JSON schema.
  4. Score cell-level Precision / Recall / F1 against gold_typos,
     excluding ambiguous_values from FP (same formula as
     RFT.typo.v5.eval.compare_ft_vs_base._cell_metrics).
  5. Dump per-row predictions + a summary CSV / md.

Prompts live in PROMPT_CANDIDATES below. Edit / extend that list and
rerun. The script caches per (prompt_id, model, row_idx) hashed by
prompt text + user content, so iterating only re-queries new rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
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
from RFT.typo.v5._task import build_response_format  # noqa: E402
from RFT.typo.v5.eval.compare_ft_vs_base import _configured_endpoint  # noqa: E402

logger = logging.getLogger("sweep_trainee_prompt")

API_VERSION = "2025-04-01-preview"
DEFAULT_EVAL = Path("RFT/typo/v7/data/eval.jsonl")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    endpoint: str
    deployment: str
    kind: str = "chat"  # "chat" | "responses"


MODELS: dict[str, ModelSpec] = {
    "o4-mini": ModelSpec(
        label="o4-mini",
        endpoint=_configured_endpoint(0),
        deployment="o4-mini",
        kind="chat",
    ),
    "gpt-5.5": ModelSpec(
        label="gpt-5.5",
        endpoint=_configured_endpoint(0),
        deployment="gpt-5.5",
        kind="responses",
    ),
}


# ---------------------------------------------------------------------------
# Candidate prompts. The key is the prompt id used on the CLI.
# Keep the v7 prompt verbatim under "v7" so we can A/B against it.
# ---------------------------------------------------------------------------

# The v7 trainee prompt (the full v4 rulebook) -- loaded from eval.jsonl
# on the fly so we always compare against the SHIPPED text, not a copy.
PROMPT_V7_PLACEHOLDER = "__V7_FROM_EVAL__"


# Candidate v8 prompts. They share the same JSON output schema as v7
# (which the response_format enforces anyway).
PROMPT_R1 = """You audit ONE CSV column for typos. You receive the column header and its
distinct values with row counts. Return EVERY value you suspect is a typo,
paired with its intended correction.

A typo is a value whose intended form is obvious from looking at the value
itself: a misspelled word (`Recieved` -> `Received`, `Adress` -> `Address`,
`Manger` -> `Manager`), a wrong / missing / doubled punctuation mark with one
obvious fix (`St.. Louis` -> `St. Louis`, `(555-) 123-4567` -> `(555) 123-4567`),
or a word-boundary spacing defect where the canonical form is unambiguous
(`web site` -> `website`, `e mail` -> `email`, `peanutbutter` -> `peanut butter`).
Adjacent duplicated words also qualify (`the the cat` -> `the cat`).

Propose generously. Borderline cases ARE in scope -- if a competent reader
could plausibly read the value as a slip of a real word, include it.

Do NOT flag differences that are ONLY:
  * padding whitespace (`New  Haven`, ` Apple`, `Apple `);
  * casing (`NEWCASTLE-UPON-TYNE` vs `Newcastle-upon-Tyne`);
  * a single comma / period / asterisk / dash that does not change a word
    (`Apples,Oranges`, `Sales-Marketing`, `Acme Holdings Inc`, `2024-25***`,
    `555 123-4567` vs `555-123-4567`);
  * pure numeric or date-shaped values (`30.01` vs `30`, `1919` vs `2019`);
  * fixed-length alphanumeric codes / IDs (ZIP, SKU, version, `XYZ1234`).

Output a single JSON object, no Markdown fences, no surrounding text:

{
  "typos": [
    {"value": "<verbatim>", "correction": "<canonical>",
     "flavour": "spelling | punctuation | spacing",
     "rationale": "<one short sentence>"}
  ]
}

Empty list only when nothing in the column looks suspicious."""


PROMPT_R2 = """You audit ONE CSV column for typos. You receive the column header and its
distinct values with row counts. Return EVERY value you suspect is a typo,
paired with its intended correction.

A typo is a value whose intended form is obvious from looking at the value
itself:
  * a misspelled word (`Recieved` -> `Received`, `Adress` -> `Address`,
    `Manger` -> `Manager`);
  * a wrong / missing / doubled punctuation mark with one obvious fix
    (`St.. Louis` -> `St. Louis`, `(555-) 123-4567` -> `(555) 123-4567`);
  * a word-boundary spacing defect where the canonical form is unambiguous
    (`web site` -> `website`, `e mail` -> `email`,
    `peanutbutter` -> `peanut butter`);
  * adjacent duplicated words (`the the cat` -> `the cat`);
  * a clear character-level slip in a NON-ENGLISH word, scientific name,
    or Latin binomial -- including names in parentheses inside a common-name
    string (e.g. `(Anarhichthys ocellatus)` -> `(Anarrhichthys ocellatus)`,
    `(Thamnophis s. surtalis)` -> `(Thamnophis s. sirtalis)`,
    `amblyrohynchos` -> `amblyrhynchos`). Flag whenever the slip yields an
    obvious near-neighbour word/name; treat French / Spanish / Latin with
    the same care as English.

Propose generously. Borderline cases ARE in scope -- if a competent reader
could plausibly read the value as a slip of a real word or name, include
it. Do NOT skip a value merely because it contains diacritics, is in a
foreign language, or sits inside parentheses.

Do NOT flag differences that are ONLY:
  * padding whitespace (`New  Haven`, ` Apple`, `Apple `);
  * casing (`NEWCASTLE-UPON-TYNE` vs `Newcastle-upon-Tyne`);
  * a single comma / period / asterisk / dash that does not change a word
    (`Apples,Oranges`, `Sales-Marketing`, `Acme Holdings Inc`, `2024-25***`,
    `555 123-4567` vs `555-123-4567`);
  * pure numeric or date-shaped values (`30.01` vs `30`, `1919` vs `2019`);
  * fixed-length alphanumeric codes / IDs (ZIP, SKU, version, `XYZ1234`).

Output a single JSON object, no Markdown fences, no surrounding text:

{
  "typos": [
    {"value": "<verbatim>", "correction": "<canonical>",
     "flavour": "spelling | punctuation | spacing",
     "rationale": "<one short sentence>"}
  ]
}

Empty list only when nothing in the column looks suspicious."""


PROMPT_R3 = """You audit ONE CSV column for typos. You receive the column header and its
distinct values with row counts. Return EVERY value you suspect is a typo,
paired with its intended correction.

A typo is a value whose intended form is obvious from looking at the value
itself:
  * a misspelled word (`Recieved` -> `Received`, `Adress` -> `Address`,
    `Manger` -> `Manager`);
  * a wrong / missing / doubled punctuation mark with one obvious fix
    (`St.. Louis` -> `St. Louis`, `(555-) 123-4567` -> `(555) 123-4567`);
  * a word-boundary spacing defect where the canonical form is unambiguous
    (`web site` -> `website`, `e mail` -> `email`,
    `peanutbutter` -> `peanut butter`);
  * adjacent duplicated words (`the the cat` -> `the cat`).

Audit EVERY token in the value with the same scrutiny, regardless of
language, register, or position. That includes foreign-language words,
personal and place names, technical or scientific terms, and any tokens
wrapped in parentheses, brackets, or quotes. If a token is one or two
characters off from a real word or name you recognise -- in any language --
flag it. Diacritics, hyphens, apostrophes, and grammatical inflections
(plural / singular / gender agreement) count as character-level evidence.

Propose generously. Borderline cases ARE in scope -- if a competent reader
could plausibly read the value as a slip of a real word or name, include
it. Do NOT skip a value just because it is not English, contains
diacritics, sits inside parentheses, or looks like jargon.

Do NOT flag differences that are ONLY:
  * padding whitespace (`New  Haven`, ` Apple`, `Apple `);
  * casing (`NEWCASTLE-UPON-TYNE` vs `Newcastle-upon-Tyne`);
  * a single comma / period / asterisk / dash that does not change a word
    (`Apples,Oranges`, `Sales-Marketing`, `Acme Holdings Inc`, `2024-25***`,
    `555 123-4567` vs `555-123-4567`);
  * pure numeric or date-shaped values (`30.01` vs `30`, `1919` vs `2019`);
  * fixed-length alphanumeric codes / IDs (ZIP, SKU, version, `XYZ1234`).

Output a single JSON object, no Markdown fences, no surrounding text:

{
  "typos": [
    {"value": "<verbatim>", "correction": "<canonical>",
     "flavour": "spelling | punctuation | spacing",
     "rationale": "<one short sentence>"}
  ]
}

Empty list only when nothing in the column looks suspicious."""


PROMPT_R4 = """You audit ONE CSV column for typos. You receive the column header and its
distinct values with row counts. Return EVERY value you suspect is a typo,
paired with its intended correction.

A typo is a value whose intended form is obvious from looking at the value
itself:
  * a misspelled word (`Recieved` -> `Received`, `Adress` -> `Address`,
    `Manger` -> `Manager`);
  * a wrong / missing / doubled punctuation mark with one obvious fix
    (`St.. Louis` -> `St. Louis`, `(555-) 123-4567` -> `(555) 123-4567`);
  * a word-boundary spacing defect where the canonical form is unambiguous
    (`web site` -> `website`, `e mail` -> `email`,
    `peanutbutter` -> `peanut butter`);
  * adjacent duplicated words (`the the cat` -> `the cat`,
    `Foo Bar Bar Baz` -> `Foo Bar Baz`).

Audit EVERY token with the same scrutiny regardless of language, register,
or position. A character-level slip in a foreign-language word, scientific
term, or proper name is still a typo:
  * `tres bien` -> `très bien`         (missing diacritic)
  * `corazon` -> `corazón`             (missing accent)
  * `los gato` -> `los gatos`          (number agreement)
  * `(Felis silvestrix)` -> `(Felis silvestris)`   (slip inside parentheses)

Before finalising, sweep the value list one more time looking for:
  * any two adjacent identical words (`X X` patterns anywhere inside a longer
    string -- a strong typo signal even if `X` is a real word);
  * any token inside parentheses, brackets, or quotes that differs by one
    or two characters from a name you recognise;
  * any accented, non-English, or jargon token whose canonical form you can
    name with confidence.

Propose generously. Borderline cases ARE in scope -- if a competent reader
could plausibly read the value as a slip of a real word or name, include
it. Do NOT skip a value just because it is not English, contains
diacritics, sits inside parentheses, or looks like jargon.

Do NOT flag differences that are ONLY:
  * padding whitespace (`New  Haven`, ` Apple`, `Apple `);
  * casing (`NEWCASTLE-UPON-TYNE` vs `Newcastle-upon-Tyne`);
  * a single comma / period / asterisk / dash that does not change a word
    (`Apples,Oranges`, `Sales-Marketing`, `Acme Holdings Inc`, `2024-25***`,
    `555 123-4567` vs `555-123-4567`);
  * pure numeric or date-shaped values (`30.01` vs `30`, `1919` vs `2019`);
  * fixed-length alphanumeric codes / IDs (ZIP, SKU, version, `XYZ1234`).

Output a single JSON object, no Markdown fences, no surrounding text:

{
  "typos": [
    {"value": "<verbatim>", "correction": "<canonical>",
     "flavour": "spelling | punctuation | spacing",
     "rationale": "<one short sentence>"}
  ]
}

Empty list only when nothing in the column looks suspicious."""


# Clean, spec-derived GPT-5.5 detector prompt (OOD-honest; no benchmark
# strings). Now the baseline "with-rulebook" prompt; loaded here so the sweep
# scores the SHIPPED text, not a copy.
from baseline.prompt import WITH_RULEBOOK_PROMPT as PROMPT_CLEAN_V9  # noqa: E402

PROMPT_CANDIDATES: dict[str, str] = {
    "v7": PROMPT_V7_PLACEHOLDER,
    "r1": PROMPT_R1,
    "r2": PROMPT_R2,
    "r3": PROMPT_R3,
    "r4": PROMPT_R4,
    "clean": PROMPT_CLEAN_V9,
}


# ---------------------------------------------------------------------------
# Metrics (copied from RFT.typo.v5.eval.compare_ft_vs_base for portability;
# stays in sync with that file by construction -- same formula).
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
    fp = len(pred_values - gold_values - amb_values)
    fn = len(gold_values - pred_values)
    p = tp / (tp + fp) if (tp + fp) else (1.0 if not gold_values else 0.0)
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
            "gold": sorted(gold_values), "pred": sorted(pred_values)}


# ---------------------------------------------------------------------------
# Deployment call
# ---------------------------------------------------------------------------


def _parse_json_obj(text: str | None) -> dict[str, Any] | None:
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


def _chat_rf_to_responses_text_format(rf: dict[str, Any]) -> dict[str, Any]:
    if rf.get("type") != "json_schema":
        return rf
    inner = rf.get("json_schema", {})
    return {
        "type": "json_schema",
        "name": inner.get("name", "output"),
        "schema": inner.get("schema", {}),
        "strict": inner.get("strict", True),
    }


def _extract_responses_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    out = getattr(resp, "output", None) or []
    parts: list[str] = []
    for item in out:
        content = getattr(item, "content", None) or []
        for c in content:
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)


def call_model(
    client,
    spec: ModelSpec,
    dev_prompt: str,
    user_prompt: str,
    response_format: dict[str, Any],
    reasoning_effort: str,
) -> tuple[str | None, str | None]:
    if spec.kind == "responses":
        try:
            resp = client.responses.create(
                model=spec.deployment,
                instructions=dev_prompt,
                input=user_prompt,
                text={"format": _chat_rf_to_responses_text_format(response_format)},
                reasoning={"effort": reasoning_effort},
            )
            return _extract_responses_text(resp), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
    try:
        resp = client.chat.completions.create(
            model=spec.deployment,
            messages=[
                {"role": "developer", "content": dev_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        return resp.choices[0].message.content, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _resolve_prompt(prompt_id: str, eval_rows: list[dict[str, Any]]) -> str:
    text = PROMPT_CANDIDATES[prompt_id]
    if text == PROMPT_V7_PLACEHOLDER:
        # use the developer message from eval.jsonl verbatim
        return eval_rows[0]["messages"][0]["content"]
    return text


def _cache_key(prompt_text: str, user_text: str, model_label: str) -> str:
    h = hashlib.sha256()
    h.update(prompt_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(user_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(model_label.encode("utf-8"))
    return h.hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = rec.get("cache_key")
            if k:
                out[k] = rec
    return out


def _append_cache(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run_one_combo(
    *,
    prompt_id: str,
    prompt_text: str,
    spec: ModelSpec,
    eval_rows: list[dict[str, Any]],
    out_dir: Path,
    reasoning_effort: str,
    workers: int,
    limit: int | None,
    skip_clean: bool,
) -> dict[str, Any]:
    response_format = build_response_format()
    cache_path = out_dir / "cache" / f"{prompt_id}__{spec.label}.jsonl"
    cache = _load_cache(cache_path)
    rows_to_run: list[tuple[int, dict[str, Any]]] = []
    for i, row in enumerate(eval_rows):
        if limit is not None and i >= limit:
            break
        if skip_clean:
            gold = json.loads(row.get("gold_typos") or "[]")
            if not gold:
                continue
        rows_to_run.append((i, row))

    client = build_client(spec.endpoint, API_VERSION)

    def _do(i: int, row: dict[str, Any]) -> dict[str, Any]:
        user_text = row["messages"][1]["content"]
        ck = _cache_key(prompt_text, user_text, spec.label)
        if ck in cache:
            return cache[ck]
        t0 = time.time()
        raw, err = call_model(
            client, spec, prompt_text, user_text,
            response_format, reasoning_effort,
        )
        latency = time.time() - t0
        parsed = _parse_json_obj(raw) if raw else None
        predicted = parsed.get("typos", []) if isinstance(parsed, dict) else []
        if not isinstance(predicted, list):
            predicted = []
        m = _cell_metrics(predicted, row["gold_typos"], row["ambiguous_values"])
        rec = {
            "cache_key": ck,
            "prompt_id": prompt_id,
            "model": spec.label,
            "row_idx": i,
            "source": row["source"],
            "column": row["column"],
            "raw_text": raw,
            "error": err,
            "pred_count": len(predicted),
            "pred_values": m["pred"],
            "gold_values": m["gold"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "latency_s": latency,
        }
        _append_cache(cache_path, rec)
        return rec

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_do, i, r): i for i, r in rows_to_run}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.exception("row failed: %s", exc)

    # Aggregate cell-pooled metrics
    gold_all: set[str] = set()
    amb_all: set[str] = set()
    pred_all: set[str] = set()
    for i, row in rows_to_run:
        gold_all.update(g.get("value", "") for g in json.loads(row.get("gold_typos") or "[]"))
        amb_all.update(str(v) for v in json.loads(row.get("ambiguous_values") or "[]"))
    # Use row-level pred sets from cache
    by_row = {r["row_idx"]: r for r in results}
    for i, row in rows_to_run:
        rec = by_row.get(i)
        if rec is None:
            continue
        pred_all.update(rec["pred_values"])
    gold_all.discard("")
    amb_all.discard("")
    pred_all.discard("")
    tp = len(pred_all & gold_all)
    fp = len(pred_all - gold_all - amb_all)
    fn = len(gold_all - pred_all)
    P = tp / (tp + fp) if (tp + fp) else (1.0 if not gold_all else 0.0)
    R = tp / (tp + fn) if (tp + fn) else 1.0
    F = (2 * P * R / (P + R)) if (P + R) else 0.0

    # Macro (mean over rows)
    macro_p = sum(r["precision"] for r in results) / max(1, len(results))
    macro_r = sum(r["recall"] for r in results) / max(1, len(results))
    macro_f = sum(r["f1"] for r in results) / max(1, len(results))

    n_errors = sum(1 for r in results if r["error"])
    summary = {
        "prompt_id": prompt_id,
        "model": spec.label,
        "rows": len(results),
        "errors": n_errors,
        "pool_tp": tp, "pool_fp": fp, "pool_fn": fn,
        "pool_precision": P, "pool_recall": R, "pool_f1": F,
        "macro_precision": macro_p, "macro_recall": macro_r,
        "macro_f1": macro_f,
        "predictions_total": sum(r["pred_count"] for r in results),
    }
    return summary


def write_summaries(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(summaries[0].keys()) if summaries else []
    with (out_dir / "summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    lines = [
        "| prompt | model | rows | err | pool P | pool R | pool F1 | macro R | preds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['prompt_id']} | {s['model']} | {s['rows']} | "
            f"{s['errors']} | {s['pool_precision']:.3f} | "
            f"{s['pool_recall']:.3f} | {s['pool_f1']:.3f} | "
            f"{s['macro_recall']:.3f} | {s['predictions_total']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", default=DEFAULT_EVAL, type=Path)
    p.add_argument("--out-dir", default=Path("outputs/v7_prompt_sweep"),
                   type=Path)
    p.add_argument("--prompts", default="v7,r1",
                   help="Comma list of prompt ids from PROMPT_CANDIDATES.")
    p.add_argument("--models", default="o4-mini",
                   help="Comma list of model ids: o4-mini,gpt-5.5")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["none", "minimal", "low", "medium", "high"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit to first N rows (for smoke runs).")
    p.add_argument("--skip-clean", action="store_true",
                   help="Only evaluate rows with non-empty gold (positive cols).")
    args = p.parse_args(argv)

    eval_rows = [json.loads(l) for l in args.eval.read_text().splitlines() if l.strip()]
    logger.info("eval rows: %d", len(eval_rows))

    prompt_ids = [s.strip() for s in args.prompts.split(",") if s.strip()]
    model_ids = [s.strip() for s in args.models.split(",") if s.strip()]
    for pid in prompt_ids:
        if pid not in PROMPT_CANDIDATES:
            print(f"Unknown prompt id: {pid}; available: {list(PROMPT_CANDIDATES)}")
            return 2
    for mid in model_ids:
        if mid not in MODELS:
            print(f"Unknown model id: {mid}; available: {list(MODELS)}")
            return 2

    summaries: list[dict[str, Any]] = []
    for pid in prompt_ids:
        ptext = _resolve_prompt(pid, eval_rows)
        for mid in model_ids:
            spec = MODELS[mid]
            logger.info("running prompt=%s model=%s", pid, mid)
            s = run_one_combo(
                prompt_id=pid, prompt_text=ptext, spec=spec,
                eval_rows=eval_rows, out_dir=args.out_dir,
                reasoning_effort=args.reasoning_effort,
                workers=args.workers, limit=args.limit,
                skip_clean=args.skip_clean,
            )
            summaries.append(s)
            logger.info("done %s/%s: pool R=%.3f P=%.3f preds=%d",
                        pid, mid, s["pool_recall"], s["pool_precision"],
                        s["predictions_total"])

    write_summaries(summaries, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
