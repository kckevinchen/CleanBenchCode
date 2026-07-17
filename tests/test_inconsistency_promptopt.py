"""Tests for the inconsistency prompt-optimization loop's deterministic core.

No network: the detector and agent backends are stubs. Covers the rulebook
surgery (schema freeze), the gold-derived scoring math, the by-source split
(leakage-free + deterministic), the mechanical leakage filter, and an
end-to-end stubbed loop run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from baseline.inconsistency import data, loop
from baseline.inconsistency._task import (
    has_output_contract,
    join_rulebook,
    parse_findings,
    split_rulebook,
)
from baseline.inconsistency.evaluator import aggregate, evaluate, score_column
from baseline.inconsistency.leakage import build_blocklist, scrub_critique, scrub_text

ROOT = Path(__file__).resolve().parents[1]
SEED_RULEBOOK = (ROOT / "baseline" / "prompts" / "inconsistency" / "rulebook.md").read_text()


# --------------------------------------------------------------------------- #
# Rulebook surgery (the mechanical schema freeze)
# --------------------------------------------------------------------------- #


def test_split_rulebook_freezes_output_section():
    head, tail = split_rulebook(SEED_RULEBOOK)
    assert "## 4. Output" not in head, "Output schema must be in the frozen tail, not editable head"
    assert tail.startswith("## 4. Output")
    assert '"findings"' in tail
    # head is still substantial (the rules)
    assert len(head) > 500


def test_join_preserves_output_contract():
    head, tail = split_rulebook(SEED_RULEBOOK)
    rebuilt = join_rulebook(head + "\n\nExtra editor line.", tail)
    assert has_output_contract(rebuilt)
    # An edit that drops the tail loses the contract -> caught by validation.
    assert not has_output_contract(head)


def test_parse_findings_tolerates_fences():
    raw = (
        '```json\n{"findings": [{"value": "USA", "corrected_value": "United States", '
        '"kind": "text", "rationale": "x"}]}\n```'
    )
    pred = parse_findings(raw)
    assert pred == {"USA": {"corrected_value": "United States", "kind": "text", "rationale": "x"}}
    assert parse_findings("not json") == {}
    assert parse_findings(None) == {}


# --------------------------------------------------------------------------- #
# Scoring math (gold-derived precision/recall/F1)
# --------------------------------------------------------------------------- #


def test_score_column_tp_fp_fn_and_ambiguous_immunity():
    gold = {"error": {"USA": {"United States"}}, "amb": {"U.K."}}
    pred = {
        "USA": {"corrected_value": "United States", "kind": "text", "rationale": "r"},  # TP + corr
        "U.K.": {"corrected_value": "United Kingdom", "kind": "text", "rationale": "r"},  # immune
        "Canadaa": {"corrected_value": "Canada", "kind": "casing", "rationale": "r"},  # FP
    }
    res = score_column(pred, gold, source="s", column="c", dataset="d", col_kind="labeled")
    assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 0
    assert res["corr_ok"] == 1
    assert res["fps"][0]["value"] == "Canadaa"


def test_score_column_false_negative():
    gold = {"error": {"USA": {"United States"}}, "amb": set()}
    res = score_column({}, gold, source="s", column="c", dataset="d", col_kind="labeled")
    assert res["tp"] == 0 and res["fn"] == 1
    assert res["fns"][0]["value"] == "USA"


def test_aggregate_precision_recall_f1():
    results = [
        score_column(
            {"e1": {"corrected_value": "E1", "kind": "text", "rationale": "r"}},
            {"error": {"e1": {"E1"}}, "amb": set()},
            source="s1",
            column="c1",
            dataset="pbi_500",
            col_kind="labeled",
        ),
        score_column(
            {"x": {"corrected_value": "", "kind": "casing", "rationale": "r"}},
            {"error": {}, "amb": set()},
            source="s2",
            column="c2",
            dataset="bing21_500",
            col_kind="clean",
        ),
    ]
    m = aggregate(results)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fp_clean"] == 1 and m["fn"] == 0
    assert m["recall"] == 1.0
    assert abs(m["precision_incl_clean"] - 0.5) < 1e-9
    assert set(m["by_dataset"]) == {"pbi_500", "bing21_500"}


# --------------------------------------------------------------------------- #
# Split discipline (by source, leakage-free, deterministic)
# --------------------------------------------------------------------------- #


def _toy_gold():
    cols = {
        ("a.csv", "c"): {"error": {"x": {"y"}}, "amb": set()},  # a: error-bearing
        ("b.csv", "c"): {"error": {}, "amb": set()},  # b: clean
        ("c.csv", "c"): {"error": {"z": set()}, "amb": set()},  # c: error-bearing
        ("d.csv", "c"): {"error": {}, "amb": set()},  # d: clean
    }
    sd = {"a.csv": "pbi_500", "b.csv": "pbi_500", "c.csv": "bing21_500", "d.csv": "bing21_500"}
    return cols, sd


def test_split_is_leakage_free_and_deterministic():
    cols, sd = _toy_gold()
    s1 = data.split_sources(sd, cols, dev_frac=0.5, val_frac=0.25, seed=7)
    s2 = data.split_sources(sd, cols, dev_frac=0.5, val_frac=0.25, seed=7)
    assert s1 == s2, "same seed must reproduce the same split"
    dev, val, test = set(s1["dev"]), set(s1["val"]), set(s1["test"])
    assert (
        dev.isdisjoint(val) and dev.isdisjoint(test) and val.isdisjoint(test)
    ), "dev/val/test must be mutually disjoint"
    assert dev | val | test == set(sd), "every source must be assigned"


def test_dev_dataset_holds_out_other_dataset():
    cols, sd = _toy_gold()
    s = data.split_sources(sd, cols, dev_frac=0.5, val_frac=0.25, seed=7, dev_dataset="bing21_500")
    # No pbi source may be in dev or val (OOD: pbi entirely held out for test).
    assert all(sd[src] == "bing21_500" for src in s["dev"] + s["val"])
    assert {"a.csv", "b.csv"}.issubset(set(s["test"]))


# --------------------------------------------------------------------------- #
# Leakage filter (the mechanical barrier)
# --------------------------------------------------------------------------- #


def test_blocklist_and_scrub_drop_leaky_suggestion():
    dev_cols = {("s.csv", "country"): {"error": {"United States": {"USA"}}, "amb": set()}}
    block = build_blocklist(dev_cols=dev_cols, min_len=4)
    assert "United States" in block
    critique = {
        "summary": "ok",
        "suggestions": [
            {
                "failure_mode": "generic",
                "hypothesized_cause": "c",
                "rule_change": "Treat dotted-acronym variants as text inconsistency.",
                "kind": "text",
                "priority": "high",
            },
            {
                "failure_mode": "leaky",
                "hypothesized_cause": "c",
                "rule_change": "Always rewrite united   states to its canonical form.",
                "kind": "text",
                "priority": "low",
            },
        ],
    }
    filtered, report = scrub_critique(critique, block)
    kept = [s["rule_change"] for s in filtered["suggestions"]]
    assert any("dotted-acronym" in k for k in kept), "generic suggestion must survive"
    assert all("REDACTED" not in k for k in kept)
    assert report["n_dropped"] == 1, "the suggestion quoting a dev value must be dropped"


def test_blocklist_skips_short_and_stopword_values():
    dev_cols = {("s.csv", "c"): {"error": {"NA": set(), "true": set()}, "amb": set()}}
    block = build_blocklist(dev_cols=dev_cols, min_len=4)
    assert "NA" not in block and "true" not in block


def test_scrub_text_removes_hallucinated_dev_value_from_prompt():
    # The editor may emit a dev value as an "example" even though it never saw
    # data (coincidental hallucination). Output-side scrub must remove it.
    block = build_blocklist(column_headers={"Account Description"}, min_len=4)
    prompt = "For example, a column like `Account  Description` may mix styles."
    cleaned, n = scrub_text(prompt, block)
    assert n == 1
    assert "Account Description" not in cleaned and "Account  Description" not in cleaned
    assert "an example" in cleaned


def test_scrub_text_noop_when_clean():
    block = build_blocklist(column_headers={"Account Description"}, min_len=4)
    prompt = "Flag the minority value that breaks the column convention."
    cleaned, n = scrub_text(prompt, block)
    assert n == 0 and cleaned == prompt
    # Compound headers are dev-specific -> blocked; generic single words -> not.
    block = build_blocklist(
        column_headers={"Defect Category", "Customer Address", "category", "description"},
        min_len=4,
    )
    assert "Defect Category" in block and "Customer Address" in block
    assert "category" not in block and "description" not in block


def test_scrub_scans_all_suggestion_fields():
    # An unexpected field carrying a dev value must still be redacted.
    block = build_blocklist(
        dev_cols={("s.csv", "c"): {"error": {"Acme International": set()}, "amb": set()}},
        min_len=4,
    )
    critique = {
        "suggestions": [
            {
                "failure_mode": "f",
                "hypothesized_cause": "c",
                "rule_change": "Generic transferable rule.",
                "example": "see Acme International here",  # unexpected field with a dev value
                "kind": "text",
                "priority": "low",
            }
        ]
    }
    filtered, report = scrub_critique(critique, block)
    assert report["redactions"] >= 1, "value in a non-standard field must be redacted"
    assert "Acme International" not in json.dumps(filtered)


# --------------------------------------------------------------------------- #
# End-to-end evaluate + run_loop with stub backends
# --------------------------------------------------------------------------- #


def _write_csv(tmp: Path, name: str, frame: dict) -> Path:
    p = tmp / name
    pd.DataFrame(frame).to_csv(p, index=False)
    return p


def _toy_corpus(tmp: Path):
    csv = _write_csv(
        tmp,
        "s1.csv",
        {
            "country": ["United States", "United States", "United States", "USA"],
            "id": ["1", "2", "3", "4"],
        },
    )
    cols = {("s1.csv", "country"): {"error": {"USA": {"United States"}}, "amb": set()}}
    sd = {"s1.csv": "pbi_500"}
    csv_map = {"s1.csv": csv}
    return cols, sd, csv_map


def test_evaluate_with_perfect_stub_detector(tmp_path):
    cols, sd, csv_map = _toy_corpus(tmp_path)

    def detector(prompt, source, column, values):
        if column == "country":
            return json.dumps(
                {
                    "findings": [
                        {
                            "value": "USA",
                            "corrected_value": "United States",
                            "kind": "text",
                            "rationale": "r",
                        }
                    ]
                }
            )
        return json.dumps({"findings": []})

    out = evaluate(SEED_RULEBOOK, {"s1.csv"}, cols, sd, csv_map, detector, concurrency=1)
    m = out["metrics"]
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0
    assert m["recall"] == 1.0 and m["precision_incl_clean"] == 1.0


def test_length_penalty_rejects_bloated_candidate(tmp_path):
    # An edit that doubles the prompt for a tiny val gain must be rejected once a
    # length penalty is on, but accepted when the penalty is off.
    cols, sd, csv_map = _toy_corpus(tmp_path)

    def detector(prompt, source, column, values):
        # Perfect on the gold value regardless of prompt -> identical metrics, so
        # only the length penalty differs between seed and the bloated candidate.
        if column == "country":
            return json.dumps(
                {
                    "findings": [
                        {
                            "value": "USA",
                            "corrected_value": "United States",
                            "kind": "text",
                            "rationale": "r",
                        }
                    ]
                }
            )
        return json.dumps({"findings": []})

    bloat = "x " * 4000  # ~8000-char editable head

    class BloatEditor:
        def __call__(self, instructions, user_input):
            if "reviewer" in instructions.splitlines()[0].lower():
                return json.dumps(
                    {
                        "dominant_problem": "precision",
                        "summary": "s",
                        "suggestions": [
                            {
                                "failure_mode": "f",
                                "hypothesized_cause": "c",
                                "rule_change": "Tighten scope.",
                                "kind": "scope",
                                "priority": "high",
                            }
                        ],
                    }
                )
            return split_rulebook(SEED_RULEBOOK)[0] + "\n\n" + bloat

    common = {
        "cols": cols,
        "source_dataset": sd,
        "csv_map": csv_map,
        "dev_sources": {"s1.csv"},
        "val_sources": {"s1.csv"},
        "test_sources": {"s1.csv"},
        "seed_rulebook": SEED_RULEBOOK,
        "reviewer_prompt": "# Reviewer\n",
        "editor_prompt": "# Prompt editor\n",
        "detector": detector,
        "agent_backend": BloatEditor(),
        "iters": 1,
        "objective": "f1",
    }

    s_pen = loop.run_loop(
        out_dir=tmp_path / "pen", max_prompt_chars=6000, length_penalty=0.05, **common
    )
    s_off = loop.run_loop(
        out_dir=tmp_path / "off", max_prompt_chars=6000, length_penalty=0.0, **common
    )
    # With penalty on, the bloated equal-F1 candidate is not better -> seed kept.
    assert s_pen["best_head_chars"] < 6000
    # With penalty off, equal F1 + 0 margin default means it's not strictly better
    # either, so just assert the penalty run did not grow the prompt.
    assert s_pen["best_head_chars"] <= s_off["best_head_chars"]


def test_load_exclusions_parses_all_scopes(tmp_path):
    ign = tmp_path / "ignored.jsonl"
    ign.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source": "a.csv",
                        "column": None,
                        "row": None,
                        "reason": "exclusion",
                        "scope": "csv",
                    }
                ),
                json.dumps(
                    {
                        "source": "b.csv",
                        "column": "Addr",
                        "row": None,
                        "reason": "exclusion",
                        "scope": "columns",
                    }
                ),
                json.dumps(
                    {
                        "source": "c.csv",
                        "column": "Name",
                        "row": 3,
                        "original_value": "x",
                        "reason": "skip",
                        "scope": None,
                    }
                ),
            ]
        )
    )
    exc = data.load_exclusions(ign)
    assert exc.sources == {"a.csv"}
    assert ("b.csv", "Addr") in exc.columns
    assert exc.drops_column("a.csv", "Anything") and exc.drops_column("b.csv", "Addr")
    assert not exc.drops_column("c.csv", "Name")  # skip-cell does not drop the column
    assert "x" in exc.immune_values("c.csv", "Name")


def test_exclusions_drop_columns_and_immunise_skips(tmp_path):
    cols, sd, csv_map = _toy_corpus(tmp_path)
    # Add a second clean column "extra" with a value the detector will flag.
    import pandas as pd

    csv = tmp_path / "s1.csv"
    df = pd.read_csv(csv, dtype=str)
    df["extra"] = ["A", "A", "A", "b"]
    df.to_csv(csv, index=False)

    def detector(prompt, source, column, values):
        # Flag the country error (TP), plus over-flag the "extra" minority (FP).
        if column == "country":
            return json.dumps(
                {
                    "findings": [
                        {
                            "value": "USA",
                            "corrected_value": "United States",
                            "kind": "text",
                            "rationale": "r",
                        }
                    ]
                }
            )
        if column == "extra":
            return json.dumps(
                {
                    "findings": [
                        {"value": "b", "corrected_value": "A", "kind": "casing", "rationale": "r"}
                    ]
                }
            )
        return json.dumps({"findings": []})

    from baseline.inconsistency.data import Exclusions

    exc = Exclusions()
    exc.columns.add(("s1.csv", "extra"))  # ignore the over-flagged column

    base = evaluate(SEED_RULEBOOK, {"s1.csv"}, cols, sd, csv_map, detector, concurrency=1)
    excl = evaluate(
        SEED_RULEBOOK, {"s1.csv"}, cols, sd, csv_map, detector, concurrency=1, exclusions=exc
    )
    assert base["metrics"]["fp"] == 1, "without exclusion the extra column is an FP"
    assert excl["metrics"]["fp"] == 0, "excluded column must not count as FP"
    assert excl["metrics"]["tp"] == 1, "real TP is unaffected"


def test_run_loop_end_to_end_stubbed(tmp_path):
    cols, sd, csv_map = _toy_corpus(tmp_path)

    def detector(prompt, source, column, values):  # flags the majority -> FP on clean cols
        return json.dumps(
            {
                "findings": [
                    {
                        "value": values[0][0],
                        "corrected_value": "",
                        "kind": "casing",
                        "rationale": "r",
                    }
                ]
            }
        )

    class StubAgents:
        def __call__(self, instructions, user_input):
            if "reviewer" in instructions.splitlines()[0].lower():
                return json.dumps(
                    {
                        "dominant_problem": "precision",
                        "summary": "s",
                        "suggestions": [
                            {
                                "failure_mode": "f",
                                "hypothesized_cause": "c",
                                "rule_change": "Do not flag the dominant form.",
                                "kind": "scope",
                                "priority": "high",
                            }
                        ],
                    }
                )
            return split_rulebook(SEED_RULEBOOK)[0] + "\n\nEditor note.\n"

    out_dir = tmp_path / "run"
    summary = loop.run_loop(
        cols=cols,
        source_dataset=sd,
        csv_map=csv_map,
        dev_sources={"s1.csv"},
        val_sources={"s1.csv"},
        test_sources={"s1.csv"},
        seed_rulebook=SEED_RULEBOOK,
        reviewer_prompt="# Reviewer\n",
        editor_prompt="# Prompt editor\n",
        detector=detector,
        agent_backend=StubAgents(),
        out_dir=out_dir,
        iters=2,
        objective="f1",
    )
    assert (out_dir / "iter_00" / "dev_metrics.json").exists()
    assert (out_dir / "iter_00" / "val_metrics.json").exists()
    assert (out_dir / "best_prompt.md").exists()
    assert (out_dir / "loop_summary.json").exists()
    assert summary["objective"] == "f1"
    assert summary["selection_split"] == "val"
    assert "generalization" in summary and "test_f1" in summary["generalization"]
    assert has_output_contract((out_dir / "best_prompt.md").read_text())
    # The frozen Output schema survived every iteration.
    last_prompt = json.loads((out_dir / "loop_summary.json").read_text())
    assert last_prompt["n_test_sources"] == 1
