"""Thorough test of the v5 multigrader against the typo benchmark.

What this script does
---------------------
1. **Synthesis**: builds the exact per-column items the grader will
   see at RFT time. Reuses ``_typo_v5_task.build_record`` so the
   shape (messages + item.column / item.gold_typos /
   item.ambiguous_values / item.column_values_json / item.source)
   matches what the trained model + grader will consume on Azure.

2. **Filtering** (strict per user request): drops any benchmark
   column that contains AT LEAST ONE ambiguous cell OR AT LEAST ONE
   trivial-error cell (pure padding-whitespace / pure casing). The
   surviving columns are pure clean-and-non-trivial-error so the
   grader's behaviour on them is unambiguous.

3. **Rollouts**: for each surviving column, generates ONE trainee
   reply by calling the GPT-5 pool via ``LLMClient``. (No RFT
   training -- this is a black-box grader test.)

4. **Grading**: scores each rollout with the SHIPPED multigrader:
   - python leg: runs the embedded source from
     ``_typo_v5_task.TYPO_V5_PYTHON_GRADER_SOURCE`` (the same code
     that ships in grader.json).
   - score_model leg: substitutes ``{{item.*}}`` /
     ``{{sample.output_text}}`` placeholders and hits
     ``gpt-5.2,gpt-5.4,gpt-5.5`` via LLMClient (stand-in for
     Azure's gpt-4o grader).

5. **Reporting**: writes per-row JSON + a summary covering:
   - column-level: mean / std / per-G-bucket score distribution
   - cell-level: rolled-up precision / recall / F1 against the 974
     non-trivial-error + clean cells
   - dead-column count (single-rollout placeholder for variance
     analysis; a future K>1 run can tighten this).

Usage
-----
::

    python RFT/typo/v5/eval/test_grader_on_benchmark.py \\
        --benchmark benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl \\
        --csv-root data/source_inconsistency_bing21 \\
        --out-dir outputs/grader_benchmark_test \\
        --trainee-model gpt-5.2,gpt-5.4,gpt-5.5 \\
        --grader-model gpt-5.2,gpt-5.4,gpt-5.5 \\
        [--limit N] [--seed 0] [--skip-sm]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v5 import _task as t  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger("test_grader_on_benchmark")


DEFAULT_BENCHMARK = (
    "benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl"
)
DEFAULT_CSV_ROOT = "data/source_inconsistency_bing21"


# ---------------------------------------------------------------------------
# Step 1: load benchmark, group by column, apply strict filter
# ---------------------------------------------------------------------------


def _load_csv_distinct_values(
    csv_path: Path, column: str,
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    try:
        with csv_path.open(encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or column not in reader.fieldnames:
                return []
            for row in reader:
                val = row.get(column)
                if val is None:
                    continue
                val = str(val)
                if val.strip() == "":
                    continue
                counts[val] += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading %s: %s", csv_path, exc)
        return []
    return counts.most_common()


def _csv_index(csv_root: Path) -> dict[str, Path]:
    return {p.name: p for p in csv_root.rglob("*.csv")}


def load_kept_columns(
    benchmark_path: Path, csv_root: Path,
    *, drop_trivial_columns: bool = True,
    drop_ambiguous_columns: bool = True,
) -> list[dict[str, Any]]:
    """Return one dict per surviving column ready for ``build_record``.

    By default a column is dropped if it contains any ambiguous
    cell or any trivial-error cell (the old strict filter). With
    ``drop_trivial_columns=False`` columns with trivial-error cells
    are kept; the trivial cells themselves are filtered out of
    ``gold_typos`` so they do not become test cases, while the
    column's clean and non-trivial-error cells are preserved.
    Likewise, with ``drop_ambiguous_columns=False`` columns with
    ambiguous cells are kept; the ambiguous cells are excluded
    from the test pool (their values are surfaced via
    ``ambiguous_values`` so callers can skip them when enumerating
    test cases). A cell is considered trivial-error if either
    ``is_v5_trivial_correction`` is true on its (original,
    corrected) pair, or its ``note`` field equals ``"trivial"``.

    The ``values_with_counts`` field is sourced from the underlying
    CSV (not the benchmark) so unlabelled values in the column are
    still presented to the trainee.
    """
    cols: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {
            "clean": [], "error_nontrivial": [],
            "error_trivial": [], "ambiguous": [],
        }
    )
    with benchmark_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["source"], r["column"])
            lbl = r.get("label")
            if lbl == "clean":
                cols[key]["clean"].append(r)
            elif lbl == "ambiguous":
                cols[key]["ambiguous"].append(r)
            elif lbl == "error":
                o = r.get("original_value")
                c = r.get("corrected_value")
                note = (r.get("note") or "").strip().lower()
                is_trivial = (
                    note == "trivial"
                    or (
                        c is not None
                        and t.is_v5_trivial_correction(o, c)
                    )
                )
                if is_trivial:
                    cols[key]["error_trivial"].append(r)
                else:
                    cols[key]["error_nontrivial"].append(r)

    csv_paths = _csv_index(csv_root)

    items: list[dict[str, Any]] = []
    for (source, column), buckets in cols.items():
        if drop_ambiguous_columns and buckets["ambiguous"]:
            continue
        if drop_trivial_columns and buckets["error_trivial"]:
            continue
        csv_path = csv_paths.get(source)
        if csv_path is None:
            logger.warning("No CSV found for source=%s; skipping.", source)
            continue
        values_with_counts = _load_csv_distinct_values(csv_path, column)
        if not values_with_counts:
            logger.warning(
                "Empty CSV column source=%s col=%r; skipping.",
                source, column,
            )
            continue
        gold_typos = []
        for cell in buckets["error_nontrivial"]:
            o = cell.get("original_value")
            c = cell.get("corrected_value")
            if not isinstance(o, str) or not isinstance(c, str):
                continue
            gold_typos.append({
                "value": o, "correction": c, "flavour": "spelling",
            })
        trivial_values = [
            cell.get("original_value")
            for cell in buckets["error_trivial"]
            if isinstance(cell.get("original_value"), str)
        ]
        ambiguous_values = [
            cell.get("original_value")
            for cell in buckets["ambiguous"]
            if isinstance(cell.get("original_value"), str)
        ]
        items.append({
            "column": column,
            "source": source,
            "values_with_counts": values_with_counts,
            "gold_typos": gold_typos,
            "ambiguous_values": ambiguous_values,
            "trivial_values": trivial_values,
            "n_clean_cells": len(buckets["clean"]),
            "n_gold_cells": len(gold_typos),
            "n_trivial_cells": len(trivial_values),
            "n_ambiguous_cells": len(ambiguous_values),
        })
    return items


# ---------------------------------------------------------------------------
# Step 2: per-row grader-leg execution
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")


def _render(template: str, mapping: dict[str, str]) -> str:
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


def _build_embedded_grade():
    ns: dict[str, Any] = {}
    exec(compile(
        t.TYPO_V5_PYTHON_GRADER_SOURCE, "<grader.json source>", "exec",
    ), ns)
    return ns["grade"]


def run_score_model_leg(
    client: LLMClient, model: str,
    grader_spec: dict[str, Any],
    item: dict[str, Any], output_text: str,
) -> tuple[float, str, str]:
    """Returns (score, reasoning, full reply)."""
    dev = next(m["content"] for m in grader_spec["input"]
               if m["role"] == "developer")
    usr_template = next(m["content"] for m in grader_spec["input"]
                        if m["role"] == "user")
    mapping = {
        "item.column": item["column"],
        "item.source": item["source"],
        "item.gold_typos": item["gold_typos"],
        "item.ambiguous_values": item["ambiguous_values"],
        "item.column_values_json": item["column_values_json"],
        "sample.output_text": output_text,
    }
    user = _render(usr_template, mapping)
    resp = client.query(
        model=model, prompt=user, system_prompt=dev,
        reasoning_effort="low",
    )
    if not resp.ok:
        return 0.0, f"<query failed: {resp.error}>", ""
    score = _parse_score(resp.text)
    if score is None:
        return 0.0, "<score line missing>", resp.text
    lo, hi = grader_spec.get("range", [0.0, 1.0])
    return float(max(lo, min(hi, score))), resp.text, resp.text


# ---------------------------------------------------------------------------
# Step 3: trainee rollout (single model call per column)
# ---------------------------------------------------------------------------


def run_trainee_rollout(
    client: LLMClient, model: str, record: Mapping[str, Any],
) -> str:
    """Call the GPT-5 pool with the trainee messages; return raw text."""
    msgs = record["messages"]
    dev = next(m["content"] for m in msgs if m["role"] == "developer")
    user = next(m["content"] for m in msgs if m["role"] == "user")
    resp = client.query(
        model=model, prompt=user, system_prompt=dev,
        reasoning_effort="medium",
    )
    if not resp.ok:
        return json.dumps({"typos": []})
    return resp.text


# ---------------------------------------------------------------------------
# Step 4: per-cell metrics (rolled up across columns)
# ---------------------------------------------------------------------------


def _norm(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _parse_typos(output_text: str) -> list[dict[str, Any]]:
    try:
        p = json.loads(output_text)
    except Exception:
        return []
    if isinstance(p, dict):
        tl = p.get("typos")
        if isinstance(tl, list):
            return [e for e in tl if isinstance(e, dict)]
    return []


def per_cell_confusion(
    pred_typos: list[dict[str, Any]],
    gold_typos: list[dict[str, Any]],
) -> dict[str, int]:
    """Cell-level TP/FP/FN treating clean cells as implicit negatives.

    Only the values in gold are positives. Predictions on values
    outside gold count as FP (since columns containing ambiguous
    values are filtered out, there are no neutralised values to
    discount). Trivial padding/casing predictions are dropped.
    """
    gold_keys = {_norm(g.get("value")): _norm(g.get("correction"))
                 for g in gold_typos if isinstance(g, Mapping)}
    pred_keys: dict[str, str] = {}
    for e in pred_typos:
        v_raw = e.get("value")
        c_raw = e.get("correction")
        if t.is_v5_trivial_correction(v_raw, c_raw):
            continue
        v = _norm(v_raw)
        c = _norm(c_raw)
        if not v or not c:
            continue
        pred_keys[v] = c
    tp = sum(1 for k in pred_keys if k in gold_keys
             and pred_keys[k] == gold_keys[k])
    wrong_fix = sum(1 for k in pred_keys if k in gold_keys
                    and pred_keys[k] != gold_keys[k])
    fp = len(pred_keys) - tp - wrong_fix
    fn = len(gold_keys) - tp - wrong_fix
    return {
        "tp": tp, "wrong_fix": wrong_fix, "fp": fp, "fn": fn,
        "pred_count": len(pred_keys), "gold_count": len(gold_keys),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", default=DEFAULT_BENCHMARK,
        type=Path,
    )
    parser.add_argument(
        "--csv-root", default=DEFAULT_CSV_ROOT,
        type=Path,
    )
    parser.add_argument(
        "--out-dir", default=Path("outputs/grader_benchmark_test"),
        type=Path,
    )
    parser.add_argument(
        "--trainee-model", default="gpt-5.2,gpt-5.4,gpt-5.5",
    )
    parser.add_argument(
        "--grader-model", default="gpt-5.2,gpt-5.4,gpt-5.5",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N columns (after shuffling by seed). "
        "Useful for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-sm", action="store_true",
        help="Score with python leg only; skip the score_model call.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading + filtering benchmark...")
    items = load_kept_columns(args.benchmark, args.csv_root)
    logger.info("Kept %d columns after strict filter.", len(items))

    # Shuffle deterministically + optional limit.
    import random
    random.Random(args.seed).shuffle(items)
    if args.limit is not None:
        items = items[: args.limit]
        logger.info("Limited to first %d (post-shuffle).", len(items))

    # Build the v5 records (same shape as RFT train.jsonl).
    records: list[dict[str, Any]] = []
    for it in items:
        rec = t.build_record(it)
        # Stash counts for later reporting.
        rec["_n_clean_cells"] = it["n_clean_cells"]
        rec["_n_gold_cells"] = it["n_gold_cells"]
        records.append(rec)
    # Save the synthesised test set for reproducibility.
    test_set_path = args.out_dir / "grader_test_set.jsonl"
    with test_set_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote synthesised test set to %s (%d rows)",
                test_set_path, len(records))

    grader_spec = t.build_grader()
    py_grade = _build_embedded_grade()
    sm_sub = grader_spec["graders"]["typo_v5_score_model_extras"]
    gw = t.TYPO_V5_GOLD_WEIGHT
    ew = 1.0 - gw

    client = LLMClient()

    results: list[dict[str, Any]] = []
    confusion_total = {"tp": 0, "wrong_fix": 0, "fp": 0, "fn": 0,
                       "pred_count": 0, "gold_count": 0}
    by_g_bucket: dict[int, list[float]] = defaultdict(list)

    t_start = time.monotonic()
    for i, rec in enumerate(records, 1):
        col = rec["column"]
        src = rec["source"]
        G = rec["_n_gold_cells"]
        ncl = rec["_n_clean_cells"]
        logger.info(
            "[%d/%d] %s :: %s  (G=%d, clean=%d)",
            i, len(records), src, col, G, ncl,
        )

        # Trainee rollout.
        try:
            output_text = run_trainee_rollout(
                client, args.trainee_model, rec,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trainee rollout failed: %s", exc)
            output_text = json.dumps({"typos": []})

        # Python grader leg.
        py_score = float(py_grade({"output_text": output_text}, rec))

        # Score-model leg.
        if args.skip_sm:
            sm_score = float("nan")
            sm_reply = "(skipped)"
        else:
            try:
                sm_score, sm_reply, _ = run_score_model_leg(
                    client, args.grader_model, sm_sub, rec, output_text,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Grader call failed: %s", exc)
                sm_score, sm_reply = 0.0, f"<exception: {exc}>"

        total = (
            5.0 * (gw * py_score + ew * sm_score)
            if not args.skip_sm else float("nan")
        )

        # Per-cell confusion.
        pred_typos = _parse_typos(output_text)
        gold_typos = json.loads(rec["gold_typos"])
        conf = per_cell_confusion(pred_typos, gold_typos)
        for k, v in conf.items():
            confusion_total[k] += v

        by_g_bucket[G].append(total)

        results.append({
            "source": src,
            "column": col,
            "G": G,
            "n_clean_cells": ncl,
            "py_score": py_score,
            "sm_score": None if args.skip_sm else sm_score,
            "total": None if args.skip_sm else total,
            "sm_reply": sm_reply,
            "confusion": conf,
            "trainee_output_excerpt": output_text[:600],
            "gold_typos": gold_typos,
        })

        # Save incrementally so we can ctrl-C without losing data.
        if i % 5 == 0 or i == len(records):
            (args.out_dir / "per_row_results.jsonl").write_text(
                "\n".join(
                    json.dumps(r, ensure_ascii=False) for r in results
                ) + "\n",
                encoding="utf-8",
            )

    t_elapsed = time.monotonic() - t_start
    logger.info("Done in %.1fs.", t_elapsed)

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    tp = confusion_total["tp"]
    wf = confusion_total["wrong_fix"]
    fp = confusion_total["fp"]
    fn = confusion_total["fn"]
    pred_count = confusion_total["pred_count"]
    gold_count = confusion_total["gold_count"]
    precision = tp / max(tp + fp + wf, 1) if pred_count else 1.0
    recall = tp / max(gold_count, 1) if gold_count else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    totals = [r["total"] for r in results if r["total"] is not None]
    py_scores = [r["py_score"] for r in results]
    sm_scores = [r["sm_score"] for r in results
                 if r["sm_score"] is not None]

    summary: dict[str, Any] = {
        "n_columns": len(results),
        "elapsed_seconds": round(t_elapsed, 1),
        "models": {
            "trainee": args.trainee_model,
            "grader": args.grader_model,
            "skip_sm": args.skip_sm,
        },
        "column_level": {
            "mean_total": (statistics.mean(totals) if totals else None),
            "std_total": (statistics.pstdev(totals)
                          if len(totals) > 1 else None),
            "mean_py": statistics.mean(py_scores),
            "mean_sm": (statistics.mean(sm_scores) if sm_scores else None),
            "by_G_bucket": {
                str(g): {
                    "n": len(v),
                    "mean": (statistics.mean(v) if v
                             and not args.skip_sm else None),
                }
                for g, v in sorted(by_g_bucket.items())
            },
        },
        "cell_level": {
            "tp": tp, "wrong_fix": wf, "fp": fp, "fn": fn,
            "gold_total": gold_count,
            "pred_total": pred_count,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
    }

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nResults dir:", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
