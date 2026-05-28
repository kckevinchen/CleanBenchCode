"""Consolidated error analysis for the v1 typo RFT model predictions.

Four angles (one per error bucket) on outputs/typo_rft/eval_v1_ft/predictions.jsonl:
  A. False positives  (gold=clean, pred=typo)
  B. False negatives + wrong corrections (gold=typo, pred=clean or pred=typo with diff correction)
  C. Ambiguous over-prediction (pred=ambiguous, gold!=ambiguous)
  D. Ambiguous under-prediction + parse failures
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PRED_FILE = Path("outputs/typo_rft/eval_v1_ft/predictions.jsonl")
OUT_DIR = Path("outputs/typo_rft/eval_v1_ft")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def lev(a: str, b: str) -> int:
    """Plain DP Levenshtein distance."""
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = cur
    return prev[-1]


def is_titlecase(s: str) -> bool:
    return bool(re.match(r"^[A-Z][a-z]+( [A-Z]([a-z]+|\.))*\.?$", s or ""))


def is_allcaps(s: str) -> bool:
    s = s or ""
    return len(s) >= 2 and s.isupper() and any(c.isalpha() for c in s)


def has_digit(s: str) -> bool:
    return any(c.isdigit() for c in (s or ""))


def has_accent(s: str) -> bool:
    return any(ord(c) > 127 for c in (s or ""))


def fmt_row(r: dict, max_orig: int = 28, max_corr: int = 28,
            max_rat: int = 90) -> str:
    src = Path(r.get("source", "")).name.replace(".csv", "")[:14]
    col = (r.get("column") or "")[:18]
    orig = (r.get("original_value") or "")[:max_orig]
    corr = (r.get("parsed_correction") or "") or (r.get("gold_correction") or "")
    corr = corr[:max_corr]
    rat = (r.get("parsed_rationale") or "")[:max_rat].replace("\n", " ")
    return f"  {src:14s} | {col:18s} | {orig:{max_orig}s} | {corr:{max_corr}s} | {rat}"


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------

rows = [json.loads(l) for l in PRED_FILE.read_text().splitlines() if l.strip()]
print(f"Loaded {len(rows)} predictions from {PRED_FILE}")

# Confusion matrix
cm = Counter((r["gold_label"], r["parsed_label"]) for r in rows)
print("\n=== Confusion matrix (gold rows x pred cols, None=parse_fail) ===")
preds = ["clean", "typo", "ambiguous", None]
print(f"  {'gold/pred':14s}", *[f"{(p or 'PARSE_FAIL'):>11s}" for p in preds])
for g in ["clean", "typo", "ambiguous"]:
    print(f"  {g:14s}", *[f"{cm.get((g, p), 0):>11d}" for p in preds])


# ----------------------------------------------------------------------------
# A. False positives: gold=clean, pred=typo  -- 318 rows
# ----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("A. FALSE POSITIVES  (gold=clean, pred=typo)")
print("=" * 78)
fps = [r for r in rows if r["gold_label"] == "clean" and r["parsed_label"] == "typo"]
print(f"n = {len(fps)}")

fp_buckets: dict[str, list[dict]] = defaultdict(list)
for r in fps:
    orig = r.get("original_value") or ""
    corr = r.get("parsed_correction") or ""
    d = lev(orig, corr)
    same_letters = orig.lower() == corr.lower() and orig != corr
    ws_only = (" ".join(orig.split()).lower()
               == " ".join(corr.split()).lower() and orig != corr)
    if (is_titlecase(orig) or is_allcaps(orig)) and d >= 2:
        b = "PROPER_NOUN"
    elif same_letters or ws_only:
        b = "CASE_OR_WHITESPACE"
    elif has_digit(orig) and len(orig.split()) == 1:
        b = "CODE_OR_ID"
    elif has_accent(orig):
        b = "ACCENTED"
    elif len(orig) <= 3:
        b = "SHORT_VALUE"
    elif d >= 3:
        b = "LARGE_EDIT"
    else:
        b = "OTHER_SMALL_EDIT"
    r["_bucket"] = b
    r["_edit_dist"] = d
    fp_buckets[b].append(r)

bucket_order_fp = ["PROPER_NOUN", "CASE_OR_WHITESPACE", "CODE_OR_ID",
                   "ACCENTED", "SHORT_VALUE", "LARGE_EDIT", "OTHER_SMALL_EDIT"]
print("\nBucket distribution:")
for b in bucket_order_fp:
    n = len(fp_buckets[b])
    print(f"  {b:22s}  {n:>4d}  ({100*n/len(fps):>5.1f}%)")

# Flavour cross-tab
print("\nflavour x bucket:")
for b in bucket_order_fp:
    flv = Counter(r.get("parsed_flavour") for r in fp_buckets[b])
    print(f"  {b:22s}  spelling={flv.get('spelling',0):>3d}  "
          f"punctuation={flv.get('punctuation',0):>3d}  None={flv.get(None,0):>3d}")

# Representative examples
for b in bucket_order_fp:
    if not fp_buckets[b]:
        continue
    n = 5 if b == "PROPER_NOUN" else 3
    print(f"\n  {b} examples (n={n}):")
    for r in fp_buckets[b][:n]:
        print(fmt_row(r))


# ----------------------------------------------------------------------------
# B. False negatives  (gold=typo, pred=clean)  +  wrong corrections
# ----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("B. FALSE NEGATIVES  (gold=typo, pred=clean)")
print("=" * 78)
fns = [r for r in rows if r["gold_label"] == "typo" and r["parsed_label"] == "clean"]
print(f"n = {len(fns)}")

fn_buckets: dict[str, list[dict]] = defaultdict(list)
for r in fns:
    orig = r.get("original_value") or ""
    gold = r.get("gold_correction") or ""
    d = lev(orig, gold)
    if orig.lower() == gold.lower():
        b = "CASE_ONLY_GOLD"
    elif d == 1:
        b = "SHORT_EDIT"
    elif d == 2 and sorted(orig.lower()) == sorted(gold.lower()):
        b = "TRANSPOSITION"
    elif d <= 2 and re.sub(r"[\W_]+", "", orig).lower() == re.sub(r"[\W_]+", "", gold).lower():
        b = "PUNCT_INSERTION"
    elif d >= 3:
        b = "LARGE_EDIT"
    else:
        b = "OTHER"
    r["_bucket"] = b
    r["_edit_dist"] = d
    fn_buckets[b].append(r)

bucket_order_fn = ["SHORT_EDIT", "TRANSPOSITION", "PUNCT_INSERTION",
                   "LARGE_EDIT", "CASE_ONLY_GOLD", "OTHER"]
print("\nBucket distribution:")
for b in bucket_order_fn:
    n = len(fn_buckets[b])
    print(f"  {b:22s}  {n:>4d}  ({100*n/len(fns):>5.1f}%)")

# Rationale keyword themes in FN
themes = Counter()
themes_keywords = {
    "no_typo_claim": ["not a typo", "no typo", "appears correct", "is correct",
                      "looks correct", "valid spelling"],
    "valid_variant": ["valid", "legitimate", "acceptable", "alternative spelling"],
    "named_entity": [" name", "entity", "proper noun", "brand", "place", "person"],
    "abbreviation": ["abbreviation", "shortened", "truncated", "contracted"],
    "domain_term": ["domain", "technical", "term", "jargon"],
    "ambiguous_call": ["could be", "may be", "possibly", "might be"],
}
for r in fns:
    rat = (r.get("parsed_rationale") or "").lower()
    for theme, kws in themes_keywords.items():
        if any(kw in rat for kw in kws):
            themes[theme] += 1
            break
    else:
        themes["other_rationale"] += 1
print("\nRationale themes:")
for k, v in sorted(themes.items(), key=lambda kv: -kv[1]):
    print(f"  {k:22s}  {v}")

# Examples per bucket
for b in bucket_order_fn:
    if not fn_buckets[b]:
        continue
    print(f"\n  {b} examples (lev_dist shown):")
    for r in fn_buckets[b][:3]:
        print(f"  [d={r['_edit_dist']}]" + fmt_row(r)[2:])

# Wrong corrections: gold=typo AND pred=typo AND gold_correction != parsed_correction
wrong_corr = [
    r for r in rows
    if r["gold_label"] == "typo" and r["parsed_label"] == "typo"
    and (r.get("gold_correction") or "").strip().lower()
        != (r.get("parsed_correction") or "").strip().lower()
]
print("\n" + "-" * 78)
print(f"WRONG-CORRECTION sub-bucket (label right, correction wrong): {len(wrong_corr)} rows")
wc_buckets: dict[str, list[dict]] = defaultdict(list)
for r in wrong_corr:
    orig = r.get("original_value") or ""
    gold = r.get("gold_correction") or ""
    pred = r.get("parsed_correction") or ""
    if pred.lower() == gold.lower():
        b = "CASE_VARIANT"
    elif re.sub(r"[\W_]+", "", pred).lower() == re.sub(r"[\W_]+", "", gold).lower():
        b = "PUNCT_VARIANT"
    else:
        # Common prefix share
        common = 0
        for c1, c2 in zip(pred.lower(), gold.lower()):
            if c1 == c2:
                common += 1
            else:
                break
        prefix_share = common / max(1, len(gold))
        # invented if very different
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, pred.lower(), gold.lower()).ratio()
        if prefix_share >= 0.6 and prefix_share < 1.0:
            b = "SAME_PREFIX_DIFF_SUFFIX"
        elif ratio < 0.5:
            b = "INVENTED"
        else:
            b = "OTHER"
    r["_bucket"] = b
    wc_buckets[b].append(r)

for b in ["CASE_VARIANT", "PUNCT_VARIANT", "SAME_PREFIX_DIFF_SUFFIX",
          "INVENTED", "OTHER"]:
    n = len(wc_buckets[b])
    print(f"  {b:22s}  {n:>4d}  ({100*n/max(1,len(wrong_corr)):>5.1f}%)")

for b in ["INVENTED", "SAME_PREFIX_DIFF_SUFFIX", "PUNCT_VARIANT", "CASE_VARIANT", "OTHER"]:
    if not wc_buckets[b]:
        continue
    print(f"\n  {b} examples (orig | gold_correction | parsed_correction):")
    for r in wc_buckets[b][:3]:
        src = Path(r.get("source", "")).name.replace(".csv", "")[:12]
        col = (r.get("column") or "")[:18]
        orig = (r.get("original_value") or "")[:22]
        gold = (r.get("gold_correction") or "")[:22]
        pred = (r.get("parsed_correction") or "")[:22]
        print(f"  {src:12s} | {col:18s} | {orig:22s} | {gold:22s} | {pred:22s}")


# ----------------------------------------------------------------------------
# C. Ambiguous over-prediction  (pred=ambiguous, gold != ambiguous)
# ----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("C. AMBIGUOUS OVER-PREDICTION  (pred=ambiguous, gold != ambiguous)")
print("=" * 78)
over = [r for r in rows if r["parsed_label"] == "ambiguous" and r["gold_label"] != "ambiguous"]
print(f"n = {len(over)}   (gold=clean: {sum(1 for r in over if r['gold_label']=='clean')}, "
      f"gold=typo: {sum(1 for r in over if r['gold_label']=='typo')})")

over_buckets: dict[str, list[dict]] = defaultdict(list)
theme_kws_over = [
    ("LACKS_CONTEXT", ["context", "without more info", "cannot determine",
                       "insufficient", "unclear"]),
    ("HEDGE", ["could be", "may be", "possibly", "might be"]),
    ("NAMED_ENTITY", [" name", "entity", "proper noun", "person", "place",
                     "brand", "surname"]),
    ("ABBREVIATION", ["abbreviation", "shortened", "truncated", "abbr"]),
    ("VALID_VARIANT", ["valid", "legitimate", "acceptable", "alternative"]),
    ("FORMAT", ["format", "convention", "style", "pattern"]),
]
for r in over:
    rat = (r.get("parsed_rationale") or "").lower()
    bucket = "OTHER"
    for b, kws in theme_kws_over:
        if any(kw in rat for kw in kws):
            bucket = b
            break
    r["_bucket"] = bucket
    over_buckets[bucket].append(r)

print("\nRationale theme distribution:")
for b, _ in theme_kws_over + [("OTHER", None)]:
    n = len(over_buckets[b])
    print(f"  {b:22s}  {n:>4d}  ({100*n/max(1,len(over)):>5.1f}%)")

for b, _ in theme_kws_over + [("OTHER", None)]:
    if not over_buckets[b]:
        continue
    print(f"\n  {b} examples:")
    for r in over_buckets[b][:3]:
        print(fmt_row(r))

# Bias check
clean_over = [r for r in over if r["gold_label"] == "clean"]
n_clean = len(clean_over)
counts = {
    "titlecase": sum(1 for r in clean_over if is_titlecase(r.get("original_value") or "")),
    "allcaps":   sum(1 for r in clean_over if is_allcaps(r.get("original_value") or "")),
    "has_digit": sum(1 for r in clean_over if has_digit(r.get("original_value") or "")),
    "has_accent": sum(1 for r in clean_over if has_accent(r.get("original_value") or "")),
    "short_<=3": sum(1 for r in clean_over if len(r.get("original_value") or "") <= 3),
}
print(f"\nBias check on (gold=clean, pred=ambiguous), n={n_clean}:")
for k, v in counts.items():
    print(f"  {k:14s}  {v:>4d}  ({100*v/max(1,n_clean):>5.1f}%)")

# The 16 gold=typo rows
typo_over = [r for r in over if r["gold_label"] == "typo"]
print(f"\nALL {len(typo_over)} rows where gold=typo but model deferred:")
for i, r in enumerate(typo_over, 1):
    print(f"  {i:2d}." + fmt_row(r)[1:])

# Rationale length compare
correct_typo_rats = [(r.get("parsed_rationale") or "") for r in rows
                     if r["gold_label"] == "typo" and r["parsed_label"] == "typo"]
over_rats = [(r.get("parsed_rationale") or "") for r in over]
print(f"\nMean rationale length: over-defer={sum(len(s) for s in over_rats)/max(1,len(over_rats)):.1f}  "
      f"correct-typo={sum(len(s) for s in correct_typo_rats)/max(1,len(correct_typo_rats)):.1f}")


# ----------------------------------------------------------------------------
# D. Ambiguous under-prediction + parse failures
# ----------------------------------------------------------------------------

print("\n" + "=" * 78)
print("D. AMBIGUOUS UNDER-PREDICTION  (gold=ambiguous, pred != ambiguous)")
print("=" * 78)
under = [r for r in rows if r["gold_label"] == "ambiguous"
         and r["parsed_label"] not in (None, "ambiguous")]
sub_a1 = [r for r in under if r["parsed_label"] == "clean"]
sub_a2 = [r for r in under if r["parsed_label"] == "typo"]
print(f"n = {len(under)}  (SUB-A1 pred=clean: {len(sub_a1)}; SUB-A2 pred=typo: {len(sub_a2)})")

# SUB-A2: anchored corrections?
plausible = 0
for r in sub_a2:
    orig = r.get("original_value") or ""
    pred = r.get("parsed_correction") or ""
    d = lev(orig, pred)
    if 1 <= d <= 2 and pred != orig:
        plausible += 1
print(f"\nSUB-A2 (n={len(sub_a2)}): plausible corrections (lev_dist 1-2): "
      f"{plausible} ({100*plausible/max(1,len(sub_a2)):.1f}%)")

# SUB-A1 heuristics
print(f"\nSUB-A1 (n={len(sub_a1)}) value-shape heuristics:")
for k, fn in [("titlecase", is_titlecase), ("allcaps", is_allcaps),
              ("has_digit", has_digit), ("has_accent", has_accent)]:
    c = sum(1 for r in sub_a1 if fn(r.get("original_value") or ""))
    print(f"  {k:14s}  {c:>4d}  ({100*c/max(1,len(sub_a1)):>5.1f}%)")

print("\nSUB-A1 examples (model said CLEAN, gold said ambiguous):")
for r in sub_a1[:5]:
    print(fmt_row(r))
print("\nSUB-A2 examples (model said TYPO, gold said ambiguous):")
for r in sub_a2[:5]:
    print(fmt_row(r))

# Parse failures
parse_fails = [r for r in rows if r["parsed_label"] is None]
print(f"\nPARSE FAILURES (n={len(parse_fails)}):")
for r in parse_fails:
    src = Path(r.get("source", "")).name.replace(".csv", "")[:14]
    col = (r.get("column") or "")[:18]
    orig = (r.get("original_value") or "")[:22]
    err = (r.get("error") or "")[:60]
    raw = (r.get("raw_text") or "")[:120].replace("\n", " ")
    print(f"  {src:14s} | {col:18s} | {orig:22s} | err={err}")
    print(f"    raw: {raw!r}")


# ----------------------------------------------------------------------------
# Persist
# ----------------------------------------------------------------------------

summary = {
    "n_rows": len(rows),
    "confusion_matrix": {f"{g}/{p}": c for (g, p), c in cm.items()},
    "fp": {
        "n": len(fps),
        "by_bucket": {b: len(fp_buckets[b]) for b in bucket_order_fp},
    },
    "fn": {
        "n": len(fns),
        "by_bucket": {b: len(fn_buckets[b]) for b in bucket_order_fn},
        "rationale_themes": dict(themes),
    },
    "wrong_correction": {
        "n": len(wrong_corr),
        "by_bucket": {k: len(v) for k, v in wc_buckets.items()},
    },
    "ambig_over": {
        "n": len(over),
        "gold_clean": sum(1 for r in over if r["gold_label"]=="clean"),
        "gold_typo": sum(1 for r in over if r["gold_label"]=="typo"),
        "by_bucket": {b: len(over_buckets[b]) for b, _ in theme_kws_over + [("OTHER", None)]},
        "clean_over_bias": counts,
    },
    "ambig_under": {
        "n": len(under),
        "sub_a1_clean": len(sub_a1),
        "sub_a2_typo": len(sub_a2),
        "sub_a2_plausible": plausible,
    },
    "parse_failures": {"n": len(parse_fails)},
}
(OUT_DIR / "error_analysis_v1.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
print(f"\nWrote summary -> {OUT_DIR/'error_analysis_v1.json'}")
