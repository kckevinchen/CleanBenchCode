"""Local end-to-end smoke test of the v5 multigrader using GPT-5.

Mimics what Azure RFT will do server-side, but runs locally so we can
trust the multigrader BEFORE burning RFT minutes on it:

1. ``typo_v5_python_gold`` -- runs the embedded Python source against
   a synthetic prediction. (Deterministic; this is the same code that
   ships in grader.json.)
2. ``typo_v5_score_model_extras`` -- substitutes the grader's
   ``{{item.*}}`` and ``{{sample.output_text}}`` placeholders, hits
   the GPT-5 endpoint pool via ``src.core.llm_client.LLMClient``,
   and parses the ``Score:`` line out of the reply. (Stand-in for
   the Azure-side gpt-4o grader -- the developer prompt is identical.)
3. Combines via the multigrader's ``calculate_output`` formula
   (``5 * (0.7*py + 0.3*sm)``) and prints expected-vs-actual scores.

Corner cases covered (per row, mix-and-matched by gold-set shape):
  * perfect_pred       -- pred == gold (py=1.0, sm=1.0 -> 5.0)
  * empty_pred         -- [] on non-empty gold (py=0.0, sm=1.0 -> 1.5)
  * empty_pred_clean   -- [] on empty gold     (py=1.0, sm=1.0 -> 5.0)
  * pure_fp            -- one spurious flag on a CLEAN value
  * wrong_correction   -- gold value with wrong fix (py=0.65*-ish)
  * trivial_padding    -- pred is pure padding/casing (must be DROPPED
                          by both legs; equivalent to empty_pred)
  * ambiguous_pred     -- pred on a value listed in ambiguous_values
                          (must be IGNORED by both legs)
  * malformed_json     -- output_text is not parseable JSON
  * spacing_typo       -- v5-specific flavour: 'DuPage' -> 'Du Page'
                          (substantive under v5 sec.1, not trivial)

Usage::

    python RFT/typo/v5/scripts/test_multigrader_local.py
    # or pin a single GPT-5 family member:
    python RFT/typo/v5/scripts/test_multigrader_local.py --model gpt-5.4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v5 import _task as t  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402


# ---------------------------------------------------------------------------
# Embedded-python-grader runner (executes the SHIPPED source, not a copy)
# ---------------------------------------------------------------------------


def _build_embedded_grade():
    ns: dict[str, Any] = {}
    exec(compile(
        t.TYPO_V5_PYTHON_GRADER_SOURCE, "<grader.json source>", "exec",
    ), ns)
    return ns["grade"]


# ---------------------------------------------------------------------------
# Score-model leg: substitute Azure {{...}} placeholders, call GPT-5
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")


def _render_template(template: str, mapping: dict[str, str]) -> str:
    """Resolve ``{{item.x}}`` / ``{{sample.output_text}}`` placeholders.

    Mirrors the (rough) semantics of Azure's grader template parser:
    it does NOT strip whitespace around the variable name -- the
    grader spec is already authored with no spaces so this is a
    no-op in practice, but we use a literal-key lookup for safety.
    """
    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in mapping:
            raise KeyError(f"unresolved placeholder {{{{ {key} }}}}")
        return mapping[key]
    return _PLACEHOLDER_RE.sub(_sub, template)


_SCORE_LINE_RE = re.compile(
    r"^\s*score\s*[:=]\s*([-+]?\d*\.?\d+)", re.IGNORECASE | re.MULTILINE,
)


def _parse_score(text: str) -> float | None:
    m = _SCORE_LINE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def run_score_model_leg(
    client: LLMClient,
    model: str,
    grader_spec: dict[str, Any],
    item: dict[str, Any],
    output_text: str,
) -> tuple[float, str]:
    """Call GPT-5 with the score_model sub-grader's developer+user prompts."""
    msgs = grader_spec["input"]
    dev = next(m["content"] for m in msgs if m["role"] == "developer")
    usr_template = next(m["content"] for m in msgs if m["role"] == "user")
    mapping = {
        "item.column": item["column"],
        "item.source": item["source"],
        "item.gold_typos": item["gold_typos"],
        "item.ambiguous_values": item["ambiguous_values"],
        "item.column_values_json": item["column_values_json"],
        "sample.output_text": output_text,
    }
    user = _render_template(usr_template, mapping)
    resp = client.query(
        model=model,
        prompt=user,
        system_prompt=dev,
        reasoning_effort="low",
    )
    if not resp.ok:
        return 0.0, f"<query failed: {resp.error}>"
    score = _parse_score(resp.text)
    if score is None:
        return 0.0, resp.text
    # Clamp to grader's declared range.
    lo, hi = grader_spec.get("range", [0.0, 1.0])
    return float(max(lo, min(hi, score))), resp.text


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


def _dump(typos: list[dict[str, Any]]) -> str:
    return json.dumps({"typos": typos}, ensure_ascii=False)


def _pick_rows() -> dict[str, dict[str, Any]]:
    rows = {"with_gold": None, "empty_gold": None, "with_amb": None}
    for line in (Path("RFT/typo/v5/data/eval.jsonl").read_text(
            encoding="utf-8")).splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = json.loads(r["gold_typos"])
        a = json.loads(r["ambiguous_values"])
        if g and rows["with_gold"] is None:
            rows["with_gold"] = r
        if not g and rows["empty_gold"] is None:
            rows["empty_gold"] = r
        if a and rows["with_amb"] is None:
            rows["with_amb"] = r
        if all(rows.values()):
            break
    return rows


def _build_cases(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return list of (label, item, output_text, expected_note) tuples."""
    cases: list[dict[str, Any]] = []

    g_row = rows["with_gold"]
    gold = json.loads(g_row["gold_typos"])
    column_values = json.loads(g_row["column_values_json"])
    # Pick a value the column DOES contain but that is NOT in gold,
    # so we can fabricate a pure FP.
    gold_values = {entry["value"] for entry in gold}
    fp_candidates = [
        v for v, _ in column_values
        if v not in gold_values and v.strip()
    ]
    fp_value = fp_candidates[0] if fp_candidates else "PLAUSIBLE_CLEAN_VALUE"

    cases.append({
        "label": "perfect_pred / non-empty gold",
        "item": g_row,
        "output_text": _dump([
            {**g, "rationale": "matches gold"} for g in gold
        ]),
        "expect_py": 1.0,
        "expect_sm": "high (no extras)",
        "expect_total": 5.0,
    })

    cases.append({
        "label": "empty_pred / non-empty gold (recall = 0)",
        "item": g_row,
        "output_text": _dump([]),
        "expect_py": 0.0,
        "expect_sm": "1.0 (no extras)",
        "expect_total": 1.5,
    })

    cases.append({
        "label": "wrong_correction on every gold value",
        "item": g_row,
        "output_text": _dump([
            {
                "value": g["value"],
                "correction": g["correction"] + "_WRONG",
                "flavour": g["flavour"],
                "rationale": "intentional wrong fix",
            }
            for g in gold
        ]),
        # raw = 0.3*G - 1.0*0 = 0.3G ; norm = (0.3G + G)/(2G) = 0.65.
        "expect_py": 0.65,
        "expect_sm": "1.0 (no extras)",
        "expect_total": 5 * (0.7 * 0.65 + 0.3 * 1.0),
    })

    cases.append({
        "label": "trivial_padding pred (must be dropped silently)",
        "item": g_row,
        "output_text": _dump([
            {
                "value": fp_value,
                "correction": "  " + fp_value + "  ",
                "flavour": "spelling",
                "rationale": "pure padding",
            }
        ]),
        # py = 0.0 (no gold caught, but trivial dropped so no spurious
        # entry either). sm should ALSO drop it -> 1.0.
        "expect_py": 0.0,
        "expect_sm": "1.0 (trivial dropped)",
        "expect_total": 1.5,
    })

    cases.append({
        "label": "malformed JSON output",
        "item": g_row,
        "output_text": "this is not json {{{",
        "expect_py": 0.0,
        "expect_sm": "0.0 (parse fail)",
        "expect_total": 0.0,
    })

    cases.append({
        "label": "spacing typo: DuPage -> Du Page (synthetic gold)",
        "item": {
            **g_row,
            "gold_typos": json.dumps([
                {
                    "value": "DuPage",
                    "correction": "Du Page",
                    "flavour": "spacing",
                }
            ]),
            "ambiguous_values": "[]",
            "column_values_json": json.dumps([
                ["DuPage", 1], ["Du Page", 47], ["Cook", 22],
            ]),
            "column": "County",
        },
        "output_text": _dump([
            {
                "value": "DuPage",
                "correction": "Du Page",
                "flavour": "spacing",
                "rationale": "matches majority spacing",
            }
        ]),
        "expect_py": 1.0,
        "expect_sm": "1.0 (no extras)",
        "expect_total": 5.0,
    })

    # Empty-gold cases.
    e_row = rows["empty_gold"]
    e_vals = json.loads(e_row["column_values_json"])
    sample_clean_value = (
        next((v for v, _ in e_vals if v.strip()), "ANY_VALUE")
    )
    cases.append({
        "label": "empty_pred / empty gold (clean column, correct silence)",
        "item": e_row,
        "output_text": _dump([]),
        "expect_py": 1.0,
        "expect_sm": "1.0 (no extras)",
        "expect_total": 5.0,
    })

    cases.append({
        "label": "pure_fp on empty gold (clean column, spurious flag)",
        "item": e_row,
        "output_text": _dump([
            {
                "value": sample_clean_value,
                "correction": sample_clean_value + "_X",
                "flavour": "spelling",
                "rationale": "(fabricated FP for grader test)",
            }
        ]),
        # py = 1.0 (G=0 short-circuit). sm should mark this fp -> 0.0
        # (or 0.5 if grader is uncertain). Total ~ 3.5.
        "expect_py": 1.0,
        "expect_sm": "~0.0 (clean column FP)",
        "expect_total": 5 * (0.7 * 1.0 + 0.3 * 0.0),
    })

    # Ambiguous-value case.
    a_row = rows["with_amb"]
    amb_values = json.loads(a_row["ambiguous_values"])
    amb = amb_values[0] if amb_values else "AMBIGUOUS"
    cases.append({
        "label": "pred on ambiguous value (must be neutral)",
        "item": a_row,
        "output_text": _dump([
            {
                "value": amb,
                "correction": amb + "_X",
                "flavour": "spelling",
                "rationale": "(should be neutralised)",
            }
        ]),
        # py drops ambiguous predictions; gold may or may not be empty
        # for this row. Compute expected py from the actual row.
        "expect_py": "row-dependent (ambiguous filter)",
        "expect_sm": "1.0 (extras list empty after filter)",
        "expect_total": "(0.7*py + 0.3*1.0) * 5",
    })

    return cases


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="gpt-5.2,gpt-5.4,gpt-5.5",
        help="GPT-5 deployment (or comma-separated pool).",
    )
    parser.add_argument(
        "--skip-sm", action="store_true",
        help="Skip the score-model leg; only run the python grader.",
    )
    parser.add_argument(
        "--report-path", default=None,
        help="Optional JSON path for full per-case results.",
    )
    args = parser.parse_args(argv)

    rows = _pick_rows()
    missing = [k for k, v in rows.items() if v is None]
    if missing:
        raise SystemExit(f"eval.jsonl missing required row types: {missing}")

    cases = _build_cases(rows)
    embedded_grade = _build_embedded_grade()
    grader_spec = t.build_grader()
    # The gold (py) sub-grader is currently NOT wired into build_grader();
    # this script keeps exercising the embedded gold-source via
    # `_build_embedded_grade()` for backwards compatibility / discrimination
    # checks, but the live grader_spec only carries the LLM + two gates.
    sm_sub = grader_spec["graders"]["typo_v5_score_model_extras"]
    gw = t.TYPO_V5_GOLD_WEIGHT
    ew = 1.0 - gw

    client = None if args.skip_sm else LLMClient()

    print(
        "=" * 78
        + f"\nv5 MULTIGRADER LOCAL SMOKE TEST  (model={args.model})\n"
        + "=" * 78
    )
    print(
        f"Formula: 5 * ({gw:.2f}*py + {ew:.2f}*sm)   "
        f"pass_threshold={grader_spec['pass_threshold']}\n"
    )

    results: list[dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        print(f"--- case {i}: {case['label']} ---")
        item = case["item"]
        out = case["output_text"]
        # Build the same sample dict shape Azure passes the python grader.
        sample = {"output_text": out}
        py_score = float(embedded_grade(sample, item))
        if args.skip_sm:
            sm_score, sm_reply = float("nan"), "(skipped)"
        else:
            sm_score, sm_reply = run_score_model_leg(
                client, args.model, sm_sub, item, out,
            )
        total = 5.0 * (gw * py_score + ew * sm_score) if not args.skip_sm else float("nan")
        print(f"  py    = {py_score:.3f}   (expected {case['expect_py']})")
        if args.skip_sm:
            print("  sm    = (skipped)")
        else:
            print(
                f"  sm    = {sm_score:.3f}   (expected {case['expect_sm']})"
            )
            print(
                f"  TOTAL = {total:.3f}   (expected {case['expect_total']})"
            )
            print(f"  sm-reply: {sm_reply.strip()[:160]!r}")
        print()
        results.append({
            "label": case["label"],
            "py": py_score,
            "sm": sm_score if not args.skip_sm else None,
            "total": total if not args.skip_sm else None,
            "expect_py": case["expect_py"],
            "expect_sm": case["expect_sm"],
            "expect_total": case["expect_total"],
            "sm_reply": sm_reply if not args.skip_sm else None,
        })

    if args.report_path:
        Path(args.report_path).write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote full report to {args.report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
