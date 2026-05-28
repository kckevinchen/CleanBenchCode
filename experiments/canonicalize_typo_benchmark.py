"""
Canonicalize ``corrected_value`` in the labeled typo benchmark.

The benchmark JSONL (e.g. ``label_result/benchmark_typo_*.jsonl``) was
hand-annotated, and many ``error`` rows carry an instruction-style correction
instead of a clean canonical value, e.g.:

    original_value:   "Murry Road"
    corrected_value:  "Review spelling; possible correction: Murray Road"

    original_value:   "Applesauce, Unsweetended, Canned"
    corrected_value:  "Replace 'Unsweetended' with 'Unsweetened'."

This script asks an LLM to rewrite each such ``corrected_value`` into the
exact canonical cell value the typo would resolve to ("Murray Road",
"Applesauce, Unsweetened, Canned"). When the instruction is ambiguous the
LLM is allowed to return ``null`` and we keep the row unchanged.

The original value is preserved as ``corrected_value_raw`` so the change is
reversible, and rows are written to a new output file (the input is never
mutated in place).

Usage:
    python -m experiments.canonicalize_typo_benchmark \
        --input  label_result/benchmark_typo_20260507T215150Z_gpt5-family.jsonl \
        --output label_result/benchmark_typo_20260507T215150Z_gpt5-family.canonical.jsonl \
        --model  gpt-5.2 \
        --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Make ``src`` importable when run as a script as well as a module.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger("canonicalize_typo_benchmark")


# ---------------------------------------------------------------------------
# Heuristics: does ``corrected_value`` already look canonical?
# ---------------------------------------------------------------------------

# An entry is "instruction-shaped" when it READS like a directive instead of a
# replacement string. We keep this deliberately tight - false-positives here
# trigger needless LLM calls, but worse, they invite the LLM to rewrite a
# value that was actually already canonical. Two complementary signals:
#
#   * starts with a known imperative verb (Replace / Change / Review / ...)
#   * contains a multi-word instructional phrase ("should be", "possible
#     correction:", "X -> Y", quoted "'foo' with 'bar'" idioms, "e.g.")
#
# Case-insensitive, applied to the trimmed string.
_INSTRUCTION_LEADING_VERBS = (
    "replace", "change", "correct", "fix", "review", "verify", "spell",
    "consider", "suggest", "rename", "update", "remove", "delete",
    "add", "rewrite", "use", "should", "if", "possibly",
    # Additional imperative verbs observed in real annotator notes:
    "set", "insert", "normalize", "normalise", "trim", "standardize",
    "standardise", "capitalize", "capitalise", "apply", "convert",
    "either", "confirm", "provide",
)
_INSTRUCTION_PHRASES = [
    r"\bshould be\b",
    r"\bpossible correction\b",
    r"\bpossibly\b",
    r"\bintended (?:value|suffix|spelling)\b",
    r"\baffected rows?\b",
    r"\b(?:e\.g\.|i\.e\.)\b",
    r"\b->\b",
    r"\b=>\b",
    # 'X' with 'Y' / "X" to "Y"
    r"['\"][^'\"]{1,80}['\"]\s+(?:to|with)\s+['\"]",
]
_INSTRUCTION_PHRASES_RE = re.compile("|".join(_INSTRUCTION_PHRASES), re.IGNORECASE)
_LEADING_VERB_RE = re.compile(
    r"^\s*(?:" + "|".join(_INSTRUCTION_LEADING_VERBS) + r")\b",
    re.IGNORECASE,
)


def looks_like_instruction(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    if _LEADING_VERB_RE.match(t):
        return True
    if _INSTRUCTION_PHRASES_RE.search(t):
        return True
    return False


def needs_canonicalization(row: dict) -> bool:
    if row.get("label") != "error":
        return False
    cv = (row.get("corrected_value") or "").strip()
    if not cv:
        return True  # blank correction on an error row -> ask the LLM
    return looks_like_instruction(cv)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You normalize annotator notes into a single canonical 'corrected cell value' "
    "for a data-quality typo benchmark. The annotator flagged a typo and proposed "
    "a correction; that correction is sometimes an instruction (e.g., \"Replace 'X' "
    "with 'Y'\") rather than the clean post-fix value. Your job is to output the "
    "exact string the cell would contain after applying the fix to the original "
    "value. Preserve everything else exactly: case, punctuation, spacing, "
    "abbreviations, numbers. If the instruction contains a single explicitly "
    "quoted candidate value (e.g. patterns like 'Verify ... (e.g., \"X\" if "
    "applicable)' or 'should be \"X\"'), extract that quoted value as the "
    "canonical value — hedging words like 'if applicable', 'possibly', or 'e.g.' "
    "do NOT make a single-candidate instruction ambiguous. Only return null when "
    "the instruction names no candidate or names multiple incompatible candidates. "
    "Output ONLY a JSON object."
)

_USER_TEMPLATE = """Original cell value (verbatim):
{original_value_json}

Annotator-proposed correction (may be an instruction or a clean value):
{corrected_value_json}

Annotator LLM-suggested value (may also be an instruction or a clean value, can be null):
{llm_suggested_value_json}

Column name: {column_json}
Annotator note: {note_json}

Output JSON exactly in this shape, nothing else:
{{"canonical_corrected_value": "<the cell value after applying the fix, or null if undetermined>"}}
"""


def build_prompt(row: dict) -> str:
    return _USER_TEMPLATE.format(
        original_value_json=json.dumps(row.get("original_value", "")),
        corrected_value_json=json.dumps(row.get("corrected_value", "")),
        llm_suggested_value_json=json.dumps(row.get("llm_suggested_value", None)),
        column_json=json.dumps(row.get("column", "")),
        note_json=json.dumps(row.get("note", "") or ""),
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> Optional[str]:
    """Return the canonical value, or None when the LLM couldn't decide / failed to parse.

    A return of ``None`` means "leave the row alone"; we never replace a
    real value with a parse failure.
    """
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
    val = payload.get("canonical_corrected_value")
    if val is None:
        return None
    if not isinstance(val, str):
        return None
    val = val.strip()
    return val or None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@dataclass
class CanonicalizationResult:
    canonical: Optional[str]
    raw_response: str
    error: Optional[str]


def canonicalize_one(
    llm: LLMClient,
    row: dict,
    *,
    model: str,
    reasoning_effort: str,
) -> CanonicalizationResult:
    try:
        resp = llm.query(
            model,
            build_prompt(row),
            reasoning_effort=reasoning_effort,
            reasoning_summary="auto",
            system_prompt=_SYSTEM_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001
        return CanonicalizationResult(None, "", f"call_failed: {exc}")
    canonical = parse_response(resp.text or "")
    return CanonicalizationResult(canonical, resp.text or "", None)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def _dedupe_key(row: dict) -> tuple:
    """Identical (original, correction-text, llm-suggested, column) cells share
    a single LLM call - the 8 'Sponsot E' rows in the benchmark would otherwise
    each pay for the same answer."""
    return (
        row.get("column") or "",
        row.get("original_value") or "",
        row.get("corrected_value") or "",
        row.get("llm_suggested_value") or "",
    )


def run(
    *,
    input_path: Path,
    output_path: Path,
    report_path: Optional[Path],
    model: str,
    reasoning_effort: str,
    concurrency: int,
    limit: Optional[int],
    force_all_errors: bool,
    dry_run: bool,
) -> dict:
    rows = load_jsonl(input_path)
    logger.info("Loaded %d rows from %s", len(rows), input_path)

    # Decide which rows to send.
    if force_all_errors:
        candidates = [(i, r) for i, r in enumerate(rows) if r.get("label") == "error"]
    else:
        candidates = [(i, r) for i, r in enumerate(rows) if needs_canonicalization(r)]
    if limit is not None:
        candidates = candidates[:limit]
    logger.info("Candidates for canonicalization: %d", len(candidates))

    if dry_run:
        for i, r in candidates:
            print(json.dumps({
                "index": i,
                "source": r.get("source"),
                "column": r.get("column"),
                "original_value": r.get("original_value"),
                "corrected_value": r.get("corrected_value"),
                "llm_suggested_value": r.get("llm_suggested_value"),
            }, ensure_ascii=False))
        return {"candidates": len(candidates), "called": 0, "updated": 0,
                "left_unchanged": len(candidates), "errors": 0}

    # Dedupe so identical (column, original, correction, suggestion) tuples make one call.
    seen: dict[tuple, list[int]] = {}
    representative_rows: dict[tuple, dict] = {}
    for i, r in candidates:
        k = _dedupe_key(r)
        seen.setdefault(k, []).append(i)
        representative_rows.setdefault(k, r)
    unique_keys = list(seen)
    logger.info("Unique LLM tasks after dedup: %d (saved %d redundant calls)",
                len(unique_keys), len(candidates) - len(unique_keys))

    llm = LLMClient()

    results: dict[tuple, CanonicalizationResult] = {}

    def submit(executor, key):
        rep = representative_rows[key]
        return executor.submit(
            canonicalize_one, llm, rep,
            model=model, reasoning_effort=reasoning_effort,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        future_to_key = {submit(ex, k): k for k in unique_keys}
        for n, fut in enumerate(as_completed(future_to_key), 1):
            key = future_to_key[fut]
            try:
                results[key] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[key] = CanonicalizationResult(None, "", f"future_failed: {exc}")
            if n % 10 == 0 or n == len(future_to_key):
                logger.info("  ... %d / %d unique tasks done", n, len(future_to_key))

    # Apply results back to rows.
    updated_indices: list[int] = []
    unchanged_indices: list[int] = []
    error_indices: list[int] = []
    report_rows: list[dict] = []

    for key, res in results.items():
        for i in seen[key]:
            row = rows[i]
            original_correction = row.get("corrected_value")
            entry = {
                "index": i,
                "source": row.get("source"),
                "row": row.get("row"),
                "column": row.get("column"),
                "original_value": row.get("original_value"),
                "previous_corrected_value": original_correction,
                "llm_suggested_value": row.get("llm_suggested_value"),
                "new_corrected_value": res.canonical,
                "raw_response": (res.raw_response or "")[:500],
                "error": res.error,
            }
            if res.error:
                error_indices.append(i)
                report_rows.append({**entry, "status": "error"})
                continue
            if res.canonical is None:
                unchanged_indices.append(i)
                report_rows.append({**entry, "status": "left_unchanged"})
                continue
            # Apply.
            row["corrected_value_raw"] = original_correction
            row["corrected_value"] = res.canonical
            row["corrected_value_canonicalized"] = True
            row["corrected_value_canonicalized_by"] = model
            updated_indices.append(i)
            report_rows.append({**entry, "status": "updated"})

    write_jsonl(output_path, rows)
    logger.info("Wrote canonicalized benchmark to %s", output_path)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            for r in report_rows:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")
        logger.info("Wrote change report to %s", report_path)

    return {
        "candidates": len(candidates),
        "unique_calls": len(unique_keys),
        "updated": len(updated_indices),
        "left_unchanged": len(unchanged_indices),
        "errors": len(error_indices),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Source benchmark JSONL.")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination JSONL (input is never modified in place).")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSONL log of every per-row decision.")
    p.add_argument("--model", default="gpt-5.2",
                   help="Azure OpenAI deployment to use (default: gpt-5.2). "
                        "Comma-separated list rotates round-robin per call.")
    p.add_argument("--reasoning-effort", default="low",
                   choices=["none", "low", "medium", "high", "xhigh"])
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N candidate rows (smoke test).")
    p.add_argument("--all-errors", action="store_true",
                   help="Send every error row, not just instruction-shaped ones.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the candidates without calling the LLM.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.output.resolve() == args.input.resolve():
        print("--output must differ from --input (we never overwrite the source).",
              file=sys.stderr)
        return 2

    if not args.dry_run:
        # Make a friendly failure if the operator hasn't run ``az login``.
        try:
            from azure.identity import AzureCliCredential
            AzureCliCredential().get_token("https://cognitiveservices.azure.com/.default")
        except Exception as exc:  # noqa: BLE001
            print(
                "Azure CLI credentials are required (run `az login`).\n"
                f"  underlying error: {exc}",
                file=sys.stderr,
            )
            return 3

    stats = run(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        concurrency=args.concurrency,
        limit=args.limit,
        force_all_errors=args.all_errors,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 40)
    print("canonicalize_typo_benchmark — summary")
    print("=" * 40)
    for k, v in stats.items():
        print(f"  {k:<20} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
