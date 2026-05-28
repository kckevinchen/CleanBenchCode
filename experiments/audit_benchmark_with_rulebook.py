"""Audit the typo benchmark against the v4 rulebook using an LLM.

For each row in ``benchmark/benchmark_typo_*.jsonl`` we ask an LLM,
armed with the v4 rulebook, to decide whether the row's
``original_value`` is a TYPO, CLEAN, or AMBIGUOUS under the rulebook.
We then compare that verdict against the benchmark's existing
``label`` field and ``note`` (in particular the ``TRIVIAL`` marker).

Outputs
-------
``verdicts.jsonl``  -- one record per row, with both the benchmark
                      label and the LLM verdict, plus the raw LLM
                      reply for review.
``summary.json``    -- aggregate counts and a precomputed list of
                      conflict buckets that a human (or a second LLM
                      reviewer) should look at first.

Conflict buckets surfaced in ``summary.json``:

  - ``trivial_now_clean``    : rows tagged ``TRIVIAL`` in note that
                              the rulebook judges CLEAN. These are
                              the cases the user specifically asked
                              about: the benchmark treats them as
                              errors but the v4 rulebook does not.
  - ``error_now_clean``      : rows with ``label='error'`` (any note)
                              that the rulebook judges CLEAN.
  - ``clean_now_typo``       : rows with ``label='clean'`` that the
                              rulebook judges TYPO. Likely under-
                              labelling in the benchmark.
  - ``ambiguous_now_typo``   : rows with ``label='ambiguous'`` that
                              the rulebook judges a confident TYPO.
  - ``parse_failed``         : rows where the LLM reply did not
                              parse as the strict JSON schema.

Usage
-----
::

    python -m experiments.audit_benchmark_with_rulebook \\
        --benchmark benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl \\
        --rulebook RFT/typo_v4/prompts/rulebook.md \\
        --output-dir outputs/typo_v4/benchmark_audit \\
        --model gpt-5.2,gpt-5.4,gpt-5.5 \\
        --concurrency 8 \\
        --limit 200            # optional, for a smoke run

The script is idempotent: re-running with the same ``--output-dir``
reuses already-judged rows (keyed on ``(source,row,column)``) and
only queries the LLM for new ones. Pass ``--refresh`` to discard the
cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger("audit_benchmark_with_rulebook")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARK = Path("benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl")
DEFAULT_RULEBOOK = Path("RFT/typo_v4/prompts/rulebook.md")
DEFAULT_OUTPUT_DIR = Path("outputs/typo_v4/benchmark_audit")
DEFAULT_MODEL = "gpt-5.2,gpt-5.4,gpt-5.5"
DEFAULT_REASONING_EFFORT = "medium"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a careful benchmark auditor.

Below is the v4 typo rulebook. Apply it as written. The user will
hand you a single cell value (plus its column header, source table
id, and any extra context). You must decide whether that cell value
is a TYPO, CLEAN, or AMBIGUOUS UNDER THIS RULEBOOK -- independent of
how the benchmark currently labels it.

The benchmark label and note are ALSO supplied so you can comment on
agreement, but they MUST NOT influence your decision. Judge by the
rulebook.

Return STRICT JSON. No markdown fences, no prose outside JSON. Schema:

  {
    "verdict":          "typo" | "clean" | "ambiguous",
    "flavour":          "spelling" | "punctuation" | "spacing" | null,
    "correction":       string | null,
    "rulebook_section": string,      // e.g. "§1 spelling",
                                     //      "§2 EXCESSIVE whitespace",
                                     //      "§3 proper noun",
                                     //      "§4 alphanumeric code"
    "rationale":        string,      // one short sentence
    "agrees_with_benchmark": true | false,
    "disagreement_kind": string | null  // if !agrees, one of:
                                        //   "trivial_now_clean",
                                        //   "error_now_clean",
                                        //   "clean_now_typo",
                                        //   "ambiguous_now_typo",
                                        //   "other"
  }

Rules of thumb for ``agrees_with_benchmark``:

  - benchmark label='error' and verdict='typo' -> agrees.
  - benchmark label='clean' and verdict='clean' -> agrees.
  - benchmark label='ambiguous' and verdict='ambiguous' -> agrees.
  - benchmark note contains 'TRIVIAL' and verdict='clean' ->
    disagrees, kind='trivial_now_clean' (the v4 rulebook §2 says
    these are upstream-handled, not typos for the LLM).
  - benchmark label='error' (no TRIVIAL) and verdict='clean' ->
    disagrees, kind='error_now_clean'.
  - benchmark label='clean' and verdict='typo' -> disagrees,
    kind='clean_now_typo'.
  - benchmark label='ambiguous' and verdict='typo' -> disagrees,
    kind='ambiguous_now_typo'.
  - any other mismatch -> disagrees, kind='other'.

=========================== v4 RULEBOOK ===========================
{{RULEBOOK}}
========================== END RULEBOOK ===========================
"""


JUDGE_USER_TEMPLATE = """Audit this single cell:

source: {source}
column: {column}
original_value: {value_json}

Benchmark label:    {label}
Benchmark note:     {note_json}
Benchmark corrected_value (if any): {corrected_json}
Benchmark llm_suggested_value (if any): {suggested_json}

Reply with the JSON object specified in the system prompt."""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _row_key(row: dict[str, Any]) -> tuple[str, Any, str]:
    return (
        str(row.get("source", "")),
        row.get("row"),
        str(row.get("column", "")),
    )


# ---------------------------------------------------------------------------
# LLM reply parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.DOTALL)


def _parse_verdict(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


# ---------------------------------------------------------------------------
# Verdict record
# ---------------------------------------------------------------------------


@dataclass
class VerdictRecord:
    source: str
    row: Any
    column: str
    original_value: Any
    benchmark_label: str | None
    benchmark_note: str | None
    benchmark_corrected_value: Any
    benchmark_has_trivial_tag: bool
    llm_verdict: str | None
    llm_flavour: str | None
    llm_correction: str | None
    llm_rulebook_section: str | None
    llm_rationale: str | None
    llm_agrees: bool | None
    llm_disagreement_kind: str | None
    llm_raw_text: str
    llm_model: str
    llm_endpoint: str
    parse_ok: bool
    error: str | None


# ---------------------------------------------------------------------------
# Core query
# ---------------------------------------------------------------------------


def _judge_one(
    client: LLMClient,
    row: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    system_prompt: str,
) -> VerdictRecord:
    note = row.get("note") or ""
    has_trivial = "TRIVIAL" in note
    user_prompt = JUDGE_USER_TEMPLATE.format(
        source=row.get("source", ""),
        column=row.get("column", ""),
        value_json=json.dumps(row.get("original_value"), ensure_ascii=False),
        label=row.get("label", ""),
        note_json=json.dumps(note, ensure_ascii=False),
        corrected_json=json.dumps(row.get("corrected_value"), ensure_ascii=False),
        suggested_json=json.dumps(
            row.get("llm_suggested_value"), ensure_ascii=False
        ),
    )

    resp = client.query(
        model,
        user_prompt,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
    )

    parsed = _parse_verdict(resp.text)
    parse_ok = parsed is not None

    return VerdictRecord(
        source=str(row.get("source", "")),
        row=row.get("row"),
        column=str(row.get("column", "")),
        original_value=row.get("original_value"),
        benchmark_label=row.get("label"),
        benchmark_note=note,
        benchmark_corrected_value=row.get("corrected_value"),
        benchmark_has_trivial_tag=has_trivial,
        llm_verdict=(parsed or {}).get("verdict") if parse_ok else None,
        llm_flavour=(parsed or {}).get("flavour") if parse_ok else None,
        llm_correction=(parsed or {}).get("correction") if parse_ok else None,
        llm_rulebook_section=(parsed or {}).get("rulebook_section") if parse_ok else None,
        llm_rationale=(parsed or {}).get("rationale") if parse_ok else None,
        llm_agrees=(parsed or {}).get("agrees_with_benchmark") if parse_ok else None,
        llm_disagreement_kind=(parsed or {}).get("disagreement_kind") if parse_ok else None,
        llm_raw_text=resp.text or "",
        llm_model=resp.model or "",
        llm_endpoint=resp.endpoint or "",
        parse_ok=parse_ok,
        error=resp.error,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _summarise(records: list[VerdictRecord]) -> dict[str, Any]:
    by_label: Counter[tuple[str | None, str | None]] = Counter()
    by_kind: Counter[str] = Counter()
    trivial_now_clean: list[dict[str, Any]] = []
    error_now_clean: list[dict[str, Any]] = []
    clean_now_typo: list[dict[str, Any]] = []
    ambiguous_now_typo: list[dict[str, Any]] = []
    parse_failed: list[dict[str, Any]] = []

    for r in records:
        if not r.parse_ok:
            parse_failed.append(asdict(r))
            continue
        by_label[(r.benchmark_label, r.llm_verdict)] += 1
        if r.llm_agrees is False:
            kind = r.llm_disagreement_kind or "other"
            by_kind[kind] += 1
            entry = {
                "source": r.source,
                "row": r.row,
                "column": r.column,
                "original_value": r.original_value,
                "benchmark_label": r.benchmark_label,
                "benchmark_note": r.benchmark_note,
                "benchmark_corrected_value": r.benchmark_corrected_value,
                "llm_verdict": r.llm_verdict,
                "llm_flavour": r.llm_flavour,
                "llm_correction": r.llm_correction,
                "llm_rulebook_section": r.llm_rulebook_section,
                "llm_rationale": r.llm_rationale,
            }
            if kind == "trivial_now_clean":
                trivial_now_clean.append(entry)
            elif kind == "error_now_clean":
                error_now_clean.append(entry)
            elif kind == "clean_now_typo":
                clean_now_typo.append(entry)
            elif kind == "ambiguous_now_typo":
                ambiguous_now_typo.append(entry)

    return {
        "n_records": len(records),
        "n_parse_ok": sum(1 for r in records if r.parse_ok),
        "n_parse_failed": len(parse_failed),
        "n_agrees": sum(1 for r in records if r.llm_agrees is True),
        "n_disagrees": sum(1 for r in records if r.llm_agrees is False),
        "by_benchmark_x_llm_verdict": {
            f"{k[0]}__vs__{k[1]}": v for k, v in sorted(by_label.items())
        },
        "by_disagreement_kind": dict(by_kind),
        "conflicts": {
            "trivial_now_clean": trivial_now_clean,
            "error_now_clean": error_now_clean,
            "clean_now_typo": clean_now_typo,
            "ambiguous_now_typo": ambiguous_now_typo,
        },
        "parse_failed": parse_failed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--rulebook", type=Path, default=DEFAULT_RULEBOOK)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Audit at most this many rows (after filtering). 0 = no limit.",
    )
    p.add_argument(
        "--labels",
        default="error,clean,ambiguous",
        help=(
            "Comma list of benchmark labels to audit. Defaults to all. "
            "Pass 'error' for a fast trivial-only smoke run."
        ),
    )
    p.add_argument(
        "--only-trivial",
        action="store_true",
        help="Audit only rows whose note contains 'TRIVIAL'.",
    )
    p.add_argument(
        "--skip-trivial",
        action="store_true",
        help="Skip rows whose note contains 'TRIVIAL' (audit non-trivial only).",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Discard any existing verdicts.jsonl cache.",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    verdicts_path = args.output_dir / "verdicts.jsonl"
    summary_path = args.output_dir / "summary.json"

    rulebook = args.rulebook.read_text()
    system_prompt = JUDGE_SYSTEM_PROMPT.replace("{{RULEBOOK}}", rulebook)

    # Load benchmark.
    rows = _load_jsonl(args.benchmark)
    logger.info("loaded %d benchmark rows from %s", len(rows), args.benchmark)

    # Filter.
    wanted_labels = {s.strip() for s in args.labels.split(",") if s.strip()}
    filtered: list[dict[str, Any]] = []
    for r in rows:
        if r.get("label") not in wanted_labels:
            continue
        if args.only_trivial and "TRIVIAL" not in (r.get("note") or ""):
            continue
        if args.skip_trivial and "TRIVIAL" in (r.get("note") or ""):
            continue
        filtered.append(r)
    logger.info(
        "after filter (labels=%s only_trivial=%s): %d rows",
        sorted(wanted_labels),
        args.only_trivial,
        len(filtered),
    )

    if args.limit > 0:
        filtered = filtered[: args.limit]
        logger.info("after --limit %d: %d rows", args.limit, len(filtered))

    # Cache.
    cached: dict[tuple[str, Any, str], VerdictRecord] = {}
    if verdicts_path.exists() and not args.refresh:
        with verdicts_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rec = VerdictRecord(**d)
                cached[(rec.source, rec.row, rec.column)] = rec
        logger.info("reused %d cached verdicts from %s", len(cached), verdicts_path)
    elif args.refresh and verdicts_path.exists():
        verdicts_path.unlink()
        logger.info("--refresh: deleted existing %s", verdicts_path)

    to_query = [r for r in filtered if _row_key(r) not in cached]
    logger.info("will query LLM for %d new rows", len(to_query))

    # Query.
    client = LLMClient()
    new_records: list[VerdictRecord] = []
    if to_query:
        with verdicts_path.open("a") as out_fh, ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = {
                pool.submit(
                    _judge_one,
                    client,
                    row,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    system_prompt=system_prompt,
                ): row
                for row in to_query
            }
            t0 = time.time()
            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                new_records.append(rec)
                out_fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                out_fh.flush()
                done += 1
                if done % 50 == 0 or done == len(to_query):
                    rate = done / max(1e-6, time.time() - t0)
                    logger.info(
                        "judged %d/%d (%.1f rows/s)",
                        done,
                        len(to_query),
                        rate,
                    )

    # Combine cached + new.
    all_records: list[VerdictRecord] = list(cached.values()) + new_records
    # Re-key on filtered set so summary covers only the audited rows.
    wanted_keys = {_row_key(r) for r in filtered}
    audit_records = [
        r for r in all_records if (r.source, r.row, r.column) in wanted_keys
    ]

    summary = _summarise(audit_records)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("wrote %s", summary_path)

    # Console summary.
    print()
    print("=== audit summary ===")
    print(f"benchmark:   {args.benchmark}")
    print(f"rulebook:    {args.rulebook}")
    print(f"model:       {args.model}")
    print(f"n_records:   {summary['n_records']}")
    print(f"parse_ok:    {summary['n_parse_ok']}")
    print(f"parse_fail:  {summary['n_parse_failed']}")
    print(f"agrees:      {summary['n_agrees']}")
    print(f"disagrees:   {summary['n_disagrees']}")
    print()
    print("benchmark_label x llm_verdict:")
    for k, v in summary["by_benchmark_x_llm_verdict"].items():
        print(f"  {k:40s} {v}")
    print()
    print("by_disagreement_kind:")
    for k, v in summary["by_disagreement_kind"].items():
        print(f"  {k:40s} {v}")
    print()
    print(f"full conflict lists in {summary_path}")


if __name__ == "__main__":
    main()
