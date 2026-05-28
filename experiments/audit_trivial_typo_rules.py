"""Audit the typo benchmark with deterministic trivial-typo rules.

For each row in the benchmark, run the rule set from
``experiments._trivial_typo_rules.detect_trivial_typo``. Report:

  - Coverage: how many TRIVIAL-tagged errors are caught by some rule
    (and which rule). These should ideally be all caught.
  - Mismatches: errors NOT tagged TRIVIAL in the benchmark but flagged
    by a rule. These are likely mislabelled and should be reviewed.
  - Disagreements: TRIVIAL-tagged errors not caught by any rule. These
    suggest either an extra rule we are missing or a labelling
    inconsistency.
  - Auto-fix check: when a rule fires, does its proposed canonical
    form match the benchmark's ``corrected_value``?
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from experiments._trivial_typo_rules import detect_trivial_typo


def audit(path: Path, show: int = 10) -> None:
    by_label: collections.Counter[str] = collections.Counter()
    trivial_total = 0
    trivial_caught = 0
    nontrivial_error_caught = []
    nontrivial_error_total = 0
    trivial_uncaught = []
    correction_mismatches = []
    rule_hits: collections.Counter[str] = collections.Counter()

    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            label = d.get("label")
            note = d.get("note") or ""
            is_trivial_tag = "TRIVIAL" in note
            value = d.get("original_value")
            corrected = d.get("corrected_value")
            if not isinstance(value, str):
                continue
            by_label[(label, is_trivial_tag)] += 1
            hit = detect_trivial_typo(value)
            if label == "error" and is_trivial_tag:
                trivial_total += 1
                if hit is not None:
                    trivial_caught += 1
                    rule_hits[hit[0]] += 1
                    if corrected is not None and hit[1] != corrected:
                        correction_mismatches.append(
                            {
                                "source": d.get("source"),
                                "column": d.get("column"),
                                "value": value,
                                "rule": hit[0],
                                "rule_canonical": hit[1],
                                "benchmark_corrected": corrected,
                            }
                        )
                else:
                    trivial_uncaught.append(
                        {
                            "source": d.get("source"),
                            "column": d.get("column"),
                            "value": value,
                            "corrected": corrected,
                            "note": note,
                        }
                    )
            elif label == "error" and not is_trivial_tag:
                nontrivial_error_total += 1
                if hit is not None:
                    nontrivial_error_caught.append(
                        {
                            "source": d.get("source"),
                            "column": d.get("column"),
                            "value": value,
                            "corrected": corrected,
                            "rule": hit[0],
                            "rule_canonical": hit[1],
                            "note": note,
                        }
                    )

    print("=== label x TRIVIAL ===")
    for (label, is_triv), n in sorted(by_label.items()):
        print(f"  label={label:9s} trivial_tag={is_triv!s:5s}  n={n}")
    print()
    print("=== TRIVIAL errors caught by rules ===")
    print(f"  total TRIVIAL errors:        {trivial_total}")
    print(f"  caught by some rule:         {trivial_caught}")
    print(f"  not caught (review needed):  {len(trivial_uncaught)}")
    print(f"  correction mismatches:       {len(correction_mismatches)}")
    print()
    print("  per-rule hits:")
    for name, n in rule_hits.most_common():
        print(f"    {n:4d}  {name}")
    print()
    print("=== ERROR rows NOT tagged TRIVIAL but flagged by rules ===")
    print(f"  total non-TRIVIAL errors:    {nontrivial_error_total}")
    print(f"  flagged by some rule:        {len(nontrivial_error_caught)}")
    if nontrivial_error_caught:
        print(f"  first {min(show, len(nontrivial_error_caught))} examples:")
        for ex in nontrivial_error_caught[:show]:
            print(
                f"    rule={ex['rule']:30s} "
                f"src={ex['source']:14s} col={ex['column']!r}"
            )
            print(
                f"        val={ex['value']!r}  ->  "
                f"rule={ex['rule_canonical']!r}  bench={ex['corrected']!r}"
            )
    print()
    print("=== TRIVIAL errors NOT caught by any rule ===")
    if trivial_uncaught:
        print(f"  first {min(show, len(trivial_uncaught))} examples:")
        for ex in trivial_uncaught[:show]:
            print(
                f"    src={ex['source']:14s} col={ex['column']!r}\n"
                f"        val={ex['value']!r}  ->  bench={ex['corrected']!r}\n"
                f"        note={ex['note']!r}"
            )
    print()
    print("=== correction mismatches (rule fires but disagrees with gold) ===")
    if correction_mismatches:
        print(f"  first {min(show, len(correction_mismatches))} examples:")
        for ex in correction_mismatches[:show]:
            print(
                f"    rule={ex['rule']:30s} "
                f"src={ex['source']:14s} col={ex['column']!r}"
            )
            print(
                f"        val={ex['value']!r}\n"
                f"        rule={ex['rule_canonical']!r}\n"
                f"        bench={ex['benchmark_corrected']!r}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmark",
        default="benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl",
    )
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()
    audit(Path(args.benchmark), args.show)


if __name__ == "__main__":
    main()
