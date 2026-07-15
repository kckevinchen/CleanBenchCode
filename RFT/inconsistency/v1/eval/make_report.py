"""Generate the GPT-5.5 inconsistency-exploration report (.docx).

Aggregate numbers are the verified temp=0 / single-endpoint (eu2) run;
example mistakes are pulled live from the mistakes JSONL so they're real.

    python -m RFT.inconsistency.v1.eval.make_report
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

OUT = Path("outputs/inconsistency_v1_baseline")
MISTAKES = OUT / "gpt55_mistakes_46b4d9166ec0.jsonl"
SUMMARY = OUT / "eu2_newbench_summary_46b4d9166ec0.json"   # re-scored on new bench
BENCH = Path("benchmark/benchmark_inconsistency_20260513T234733Z_gpt5-family.jsonl")
DOC_OUT = OUT / "GPT5.5_inconsistency_exploration.docx"


def _norm(s):
    return " ".join(str(s).split())


def recall_by_kind(miss_rows):
    """(kind, recall%) computed from the benchmark gold + the miss rows."""
    gold = collections.defaultdict(set)
    for line in open(BENCH):
        r = json.loads(line)
        if r["label"] == "error":
            k = (r.get("cluster_kinds") or ["?"])[0]
            gold[k].add((r["source"], r["column"], _norm(r["original_value"])))
    miss = collections.Counter(r["gold_subpattern"] for r in miss_rows)
    out = []
    for k in ("casing", "punctuation", "text", "number_format",
              "numerical_format", "pattern"):
        g = len(gold.get(k, ()))
        if not g:
            continue
        rec = (g - miss.get(k, 0)) / g
        out.append((k, g, rec))
    return out


def _first(x):
    return x[0] if isinstance(x, list) and x else (x if isinstance(x, str) else "")


# ---------------------------------------------------------------------------
# docx helpers
# ---------------------------------------------------------------------------

def add_table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold = True
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
    doc.add_paragraph()
    return t


def add_examples(doc, lines):
    for ln in lines:
        p = doc.add_paragraph()
        run = p.add_run(ln)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(2)


def add_bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load():
    rows = [json.loads(l) for l in open(MISTAKES)]
    miss = [r for r in rows if r["error_kind"] == "miss"]
    fp = [r for r in rows if r["error_kind"] == "false_positive"]
    return rows, miss, fp


def pick(items, pred, n):
    out = []
    for r in sorted(items, key=lambda r: (r["source"], r["column"], r["value"])):
        if pred(r):
            out.append(r)
        if len(out) >= n:
            break
    return out


def miss_line(r):
    return f"  [{r['column'][:22]:22}] {r['value'][:34]!r:36} -> gold {str(_first(r['gold_corrected_value']))[:26]!r}"


def fp_line(r):
    return f"  [{r['column'][:22]:22}] {r['value'][:30]!r:32} -> GPT5.5 {str(_first(r['gpt5_corrected_value']))[:24]!r}"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def main():
    rows, miss, fp = load()
    S = json.load(open(SUMMARY))
    rbk = recall_by_kind(miss)
    pat_recall = next((rec for k, g, rec in rbk if k == "pattern"), 0.0)
    doc = Document()

    doc.add_heading("GPT-5.5 on CSV Inconsistency Detection: Exploration Results", 0)
    sub = doc.add_paragraph()
    sub.add_run("Zero-shot evaluation, error analysis, and methodology findings. "
                "Benchmark: bing21 inconsistency set. Model: gpt-5.5-2026-04-24, "
                "reasoning_effort=none, temperature=0.").italic = True

    # 1. Executive summary
    doc.add_heading("1. Executive Summary", 1)
    add_bullets(doc, [
        f"GPT-5.5 (zero-shot, shared rulebook) detects single-column representational "
        f"inconsistencies with stable recall ~{S['recall']:.2f} and correction accuracy "
        f"~{S['correction_accuracy']:.2f} -- the strongest result of everything tried, "
        f"beating both base o4-mini and an RFT-tuned o4-mini.",
        f"Its one systematic weakness is structured-number formatting (e.g. phone-number "
        f"digit grouping): it judges each value as valid on its own and won't enforce the "
        f"column's shape convention (pattern recall only {pat_recall:.0%}).",
        "The reported precision is a LOWER BOUND, confounded by benchmark labeling: a large "
        "share of its 'false positives' are either correct-but-unlabeled, or are casing flags "
        "on proper nouns/emails that are not real defects.",
        "The two most reusable outputs are methodological: (a) GPT-5.5 is highly stochastic at "
        "the default temperature and must be evaluated at temperature=0; (b) the benchmark, not "
        "the model, is now the bottleneck.",
    ])

    # 2. Setup
    doc.add_heading("2. Setup", 1)
    add_bullets(doc, [
        "Task: audit ONE CSV column at a time; flag every value that departs from the column's "
        "dominant representational convention, with a canonical correction. Five sub-patterns: "
        "casing, punctuation, pattern (shape), text (wording), number_format.",
        "Benchmark: the REGENERATED inconsistency set (20260513T234733Z) -- 65 source CSVs, "
        "93 labeled columns, 201 error + 529 ambiguous gold cells. (A prior export had a "
        "label-system discrepancy that over-labeled 42 cells; see Section 7.)",
        "Prompt: a shared rulebook (whole-column framing); reasoning_effort=none.",
        "Eval: distinct-value unit, ambiguous cells are FP-immune; scored on labeled columns "
        "PLUS all clean columns of the same sources (to measure over-flagging). Single endpoint, "
        "temperature=0 (see Section 6 for why).",
    ])

    # 3. Headline results
    doc.add_heading("3. Headline Results (temp=0, single endpoint)", 1)
    n_err = S["TP"] + S["FN"]
    add_table(doc, ["Metric", "Value"], [
        ["Recall", f"{S['recall']:.2f}  ({S['TP']} / {n_err} distinct error values)"],
        ["Precision -- labeled columns only", f"{S['precision_labeled_only']:.2f}"],
        ["Precision -- including clean columns",
         f"{S['precision_incl_clean']:.2f}  (lower bound; see Section 7)"],
        ["F1 -- including clean columns", f"{S['f1_incl_clean']:.2f}"],
        ["Clean-column false-alarm rate",
         f"{S['clean_cols_flagged'] / S['n_clean_cols']:.0%}  "
         f"({S['clean_cols_flagged']} of {S['n_clean_cols']} clean columns)"],
        ["Correction accuracy (of true hits)", f"{S['correction_accuracy']:.2f}"],
    ])

    def _read(k, g, rec):
        if k == "pattern":
            return "THE weak spot"
        if g < 5:
            return "small N"
        return "strong" if rec >= 0.80 else ("moderate" if rec >= 0.50 else "weak")
    doc.add_paragraph("Recall by sub-pattern (full benchmark):")
    add_table(doc, ["Sub-pattern", "Gold N", "Recall", "Read"],
              [[k, str(g), f"{rec:.0%}", _read(k, g, rec)] for k, g, rec in rbk])

    doc.add_heading("3.1 Small-model comparison (held-out 20 sources)", 2)
    doc.add_paragraph(
        "On a held-out 20-source slice, GPT-5.5 beat both small models on both legs. "
        "(o4-mini numbers are single-endpoint; the GPT-5.5 slice was a noisier multi-endpoint "
        "run, so treat the exact value as approximate -- but the gap is large.)")
    add_table(doc, ["Model", "Recall", "Precision (incl. clean)"], [
        ["base o4-mini (reasoning=medium)", "0.40", "0.12"],
        ["RFT-tuned o4-mini", "0.27", "0.09"],
        ["GPT-5.5 (zero-shot)", "~0.46", "~0.41"],
    ])

    # 4. Error analysis
    doc.add_heading("4. Error Analysis", 1)
    doc.add_paragraph(
        f"From the per-mistake export ({len(rows)} rows: {len(miss)} misses, {len(fp)} false "
        "positives, 6 wrong-corrections). Every false positive carries GPT-5.5's own rationale.")

    doc.add_heading("4.1 What it misses (recall)", 2)
    add_bullets(doc, [
        "Dominant, systematic miss: STRUCTURED-NUMBER REGROUPING -- phone/ID numbers that break "
        "the column's grouping convention. GPT-5.5 reads each as a valid number and won't flag it.",
        "The rest are borderline: pluralisation, hyphen-spacing, abbreviation -- cases where the "
        "gold itself is debatable.",
    ])
    ex = pick(miss, lambda r: bool(re.search(r"\(\d{3}\)\s*\d{6,}", r["value"])), 3)
    ex += pick(miss, lambda r: r["gold_subpattern"] == "text", 2)
    add_examples(doc, [miss_line(r) for r in ex])

    doc.add_heading("4.2 What it over-flags (precision)", 2)
    add_bullets(doc, [
        "Casing on PROPER NOUNS / addresses (person, org, place names) -- incidental entry style, "
        "not a defect. This is the single biggest over-flag category.",
        "Casing on EMAIL / free-text columns -- case-insensitive or naturally varied.",
        "ZIP/code leading-zeros and hyphen-spacing in titles -- debatable.",
        "Whitespace collapse (double space, non-breaking space) -- trivial.",
    ])
    def _casing_diff(r):  # genuine case-only diff (excludes whitespace collapse)
        v = r["value"]; c = _first(r.get("gpt5_corrected_value"))
        return isinstance(c, str) and v.casefold() == c.casefold() and v != c
    fex = pick(fp, lambda r: _casing_diff(r) and any(
        k in r["column"].lower() for k in ("name", "agency", "site", "district",
                                           "applicant")), 3)
    fex += pick(fp, lambda r: "email" in r["column"].lower() and _casing_diff(r), 1)
    fex += pick(fp, lambda r: "zip" in r["column"].lower(), 1)
    add_examples(doc, [fp_line(r) for r in fex])

    # 5. Behaviour profile
    doc.add_heading("5. Model Behaviour Profile", 1)
    add_table(doc, ["Dimension", "Assessment"], [
        ["Casing (controlled vocab)", "Strong (86% recall)"],
        ["Text / wording variants", "Moderate"],
        ["Correction generation", "Strong (0.94 of hits get an acceptable fix)"],
        ["Structured-number shape", "WEAK -- the one real model gap"],
        ["Over-flag tendency", "Imposes casing on proper nouns / email / free-text"],
    ])

    # 6. Measurement pitfalls
    doc.add_heading("6. Measurement Pitfalls (key methodology findings)", 1)
    doc.add_heading("6.1 Stochasticity -- must use temperature=0", 2)
    doc.add_paragraph(
        "At the default temperature (1.0) GPT-5.5 is wildly non-deterministic. Probing one "
        "column (51 gold inconsistencies) four times:")
    add_examples(doc, [
        "  temperature = 1.0  ->  [59, 9, 8, 8]   findings   (a coin flip on recall)",
        "  temperature = 0.0  ->  [8, 8, 8, 8]     findings   (deterministic)",
    ])
    doc.add_paragraph(
        "Consequence: single-run deltas under ~0.1 F1 are noise. Every comparison must fix "
        "temperature=0 (and average runs or pin the endpoint for fine deltas).")
    doc.add_heading("6.2 Endpoints and reasoning level", 2)
    add_bullets(doc, [
        "All five endpoints serving 'gpt-5.5' are the SAME model version (gpt-5.5-2026-04-24) and "
        "genuinely honor reasoning_effort=none (reasoning_tokens=0). Reasoning level was ruled out "
        "as a cause of variance.",
        "Even so, temp=0 jitters run-to-run; a multi-endpoint round-robin (added for speed) "
        "silently confounded early prompt comparisons. Fix: pin a single endpoint, or a stable "
        "column->endpoint hash, for any comparison.",
    ])

    # 7. Benchmark quality
    doc.add_heading("7. Benchmark-Quality Findings", 1)
    doc.add_paragraph(
        f"Precision-incl-clean ({S['precision_incl_clean']:.2f}) is a LOWER BOUND -- it is "
        "confounded by labeling in both directions:")
    add_bullets(doc, [
        "GOLD GAPS: real inconsistencies the labelers missed (raises true precision).",
        "OVER-LABELING / scope confusion on casing: of 62 casing 'false positives' (prior export), "
        "56 are NOT real defects -- 34 are casing on proper nouns/addresses, 17 on email/free-text, "
        "5 on weak-convention columns. Only ~6 (controlled-vocab/code columns) are even debatable.",
        "PER-COLUMN INFERENCE violations: an automated check found ~79 of ~206 gold ERROR cells "
        "(prior benchmark) are over-labeled -- no dominant convention on the cell's axis (e.g. the "
        "PACK column mixes cans/ctn/ctns/pkgs), the value already conforms to the majority "
        "(STATE OF TENNESSEE in a 76% all-caps column), or the correction is empty/identical.",
        "The 'phone-regrouping' recall gap is itself partly an over-label: the phone column is "
        "57% hyphenated / 43% not -- a weak split, not a firm convention.",
    ])
    doc.add_paragraph(
        "Confirmation: the benchmark was REGENERATED to fix a label-system discrepancy, which "
        "removed/downgraded 42 error cells -- exactly these over-labels (typos like 'Balnk cell', "
        "casing on school names, no-convention cells). The numbers in this report are computed "
        "against that regenerated benchmark; re-scoring shifted recall UP (credit for correct "
        "abstentions) and precision DOWN (GPT-5.5 shared some of the over-flagging), confirming the "
        "casing-of-proper-nouns over-flag is genuine.")
    doc.add_paragraph(
        "Definition that resolves it: casing is a meaningful inconsistency ONLY for controlled-"
        "vocabulary / categorical / code columns -- never proper nouns, names, addresses, emails, "
        "or free text. More broadly: inconsistency is PER-COLUMN -- there is no universal correct "
        "format; the target is whatever the column's own majority establishes, and a column with "
        "no dominant form should yield no findings.")

    # 8. Conclusions
    doc.add_heading("8. Conclusions & Recommendations", 1)
    add_bullets(doc, [
        "GPT-5.5 zero-shot is a strong, usable inconsistency detector (~0.45 recall, 0.94 "
        "correction) and the best option measured -- RFT-tuning a small model did not beat it.",
        "The one model-side gap to close is structured-number recall; a cheap deterministic "
        "shape-blocker feeding candidates to GPT-5.5 would target exactly its blind spot.",
        "The bigger lever is the BENCHMARK: relabel gold gaps and fix casing scope. Until then, "
        "treat precision-incl-clean as a lower bound and report labeled-only precision alongside.",
        "Operational rules for any future GPT-5.5 eval: temperature=0, reasoning=none, single "
        "endpoint (or stable hashing); only trust deltas above the jitter floor.",
    ])

    # Appendix
    doc.add_heading("Appendix -- Reproducibility & Artifacts", 1)
    add_bullets(doc, [
        "Rulebook (shareable): docs/inconsistency_prompt_shareable.md",
        "Eval harness: RFT/inconsistency/v1/eval/run_baseline.py "
        "(--model gpt-5.5 --reasoning-effort none --temperature 0)",
        "Per-mistake export: outputs/.../gpt55_mistakes_<hash>.jsonl (this report uses it)",
        "Casing gold-gap worklist: outputs/.../gpt55_gold_gap_candidates_casing.jsonl",
        "Model: gpt-5.5-2026-04-24; endpoint: dmxaz-test-aoai-eu2; temp=0; reasoning=none.",
    ])

    doc.save(DOC_OUT)
    print(f"wrote {DOC_OUT}")
    print(f"sections: 8 + appendix; mistakes used: {len(rows)}; "
          f"examples embedded: {len(ex)} misses + {len(fex)} FPs")


if __name__ == "__main__":
    main()
