"""Typo-specialist baselines (SymSpell + NeuSpell) on ``benchmark/merged_v2``.

These are string spell/typo correctors repurposed as cell-level error *detectors*:
a cell is flagged when the corrector judges it misspelled (SymSpell: a token has a
closer dictionary word; NeuSpell: the contextual correction differs from the
original). To maximize coverage they run over **every textual cell** (any cell
containing a letter, all columns) -- not just typo columns -- and are scored
against the **full** benchmark error set, so we can see how much of each error
type (typo / inconsistency / semantic) a typo tool recovers.

This reuses the REIN harness so results are directly comparable to ``raha``:
* input    : the materialized ``external/rein/.../datasets/<ds>/dirty.csv``
* output   : ``external/rein/results/predictions_<tool>.jsonl`` (+ ``.ran.json``)
* scoring  : ``external/rein/score.py`` (cell-level micro P/R/F1 + per-type recall)

If two tools are run together, a ``union`` detector (SymSpell OR NeuSpell) is also
emitted to show the combined coverage ceiling.

Examples (from repo root, with external/typo_baselines/venv active):
    python external/typo_baselines/run_typo.py --tool symspell
    python external/typo_baselines/run_typo.py --tool symspell --tool neuspell --mode aggressive
    python external/typo_baselines/run_typo.py --tool symspell --limit 5   # smoke test
Then score:
    python external/rein/score.py --predictions external/rein/results/predictions_symspell.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Reuse the REIN harness helpers (paths, REIN-faithful CSV loader, slug).
_REIN_DIR = Path(__file__).resolve().parents[1] / "rein"
sys.path.insert(0, str(_REIN_DIR))
import common as C  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z']+")
HAS_ALPHA = re.compile(r"[A-Za-z]")
Cell = tuple[str, int, int]  # (dataset, row, col)


# --------------------------------------------------------------------------- #
# SymSpell
# --------------------------------------------------------------------------- #
def build_symspell(max_edit_distance: int):
    import importlib.resources as ir

    from symspellpy import SymSpell

    sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
    dict_path = ir.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
    sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
    return sym


def symspell_correct(value: str, sym, args):
    """Return ``(flagged, corrected)``.

    ``flagged`` matches ``symspell_flag`` exactly (detection is unchanged).
    ``corrected`` is the token-substituted string when a strictly-closer real word
    exists for at least one token, else ``None`` (flagged-but-uncorrectable, e.g.
    ``flag_unknown`` firing on an out-of-dictionary token with no near word).
    """
    from symspellpy import Verbosity

    if not HAS_ALPHA.search(value):
        return (False, None)
    flagged = False
    changed = False

    def _sub(m):
        nonlocal flagged, changed
        tok = m.group(0)
        if len(tok) < args.min_token_len:
            return tok
        if args.skip_allcaps and tok.isupper() and len(tok) > 1:
            return tok
        low = tok.lower()
        sugg = sym.lookup(low, Verbosity.CLOSEST, max_edit_distance=args.max_edit_distance,
                          include_unknown=False)
        if not sugg:
            if args.flag_unknown:
                flagged = True
            return tok
        if sugg[0].distance == 0:
            return tok
        flagged = True
        repl = sugg[0].term
        if tok.isupper():
            repl = repl.upper()
        elif tok[:1].isupper():
            repl = repl.capitalize()
        changed = True
        return repl

    corrected = TOKEN_RE.sub(_sub, value)
    return (flagged, corrected if changed else None)


def symspell_flag(value: str, sym, args) -> bool:
    """True if any token has a strictly-closer dictionary word (=> misspelled)."""
    return symspell_correct(value, sym, args)[0]


def run_symspell(cells: list[tuple[Cell, str]], args) -> dict[Cell, str | None]:
    sym = build_symspell(args.max_edit_distance)
    out: dict[Cell, str | None] = {}
    for (key, val) in cells:
        flagged, corrected = symspell_correct(val, sym, args)
        if flagged:
            out[key] = corrected
    return out


# --------------------------------------------------------------------------- #
# NeuSpell (contextual)
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def run_neuspell(cells: list[tuple[Cell, str]], args) -> dict[Cell, str | None]:
    import torch
    import neuspell

    # NeuSpell's checkpoints predate torch 2.6, whose torch.load default flipped to
    # weights_only=True and rejects them ("Unsupported operand"). Force the legacy
    # behavior for this trusted, just-downloaded checkpoint. Also load the BERT
    # state_dict non-strictly: old checkpoints carry an extra
    # ``embeddings.position_ids`` buffer that newer transformers no longer expect.
    _orig_load = torch.load
    _orig_load_sd = torch.nn.Module.load_state_dict

    def _compat_load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    def _compat_load_sd(self, state_dict, strict=True, **k):
        return _orig_load_sd(self, state_dict, strict=False, **k)

    torch.load = _compat_load
    torch.nn.Module.load_state_dict = _compat_load_sd
    try:
        checker_cls = getattr(neuspell, args.neuspell_checker)
        checker = checker_cls()
        checker.from_pretrained()
    finally:
        torch.load = _orig_load
        torch.nn.Module.load_state_dict = _orig_load_sd

    # Only score cells with alphabetic content; batch for speed.
    idx = [i for i, (_, v) in enumerate(cells) if HAS_ALPHA.search(v)]
    strings = [cells[i][1] for i in idx]
    flagged: dict[Cell, str | None] = {}
    B = args.neuspell_batch
    for start in range(0, len(strings), B):
        batch = strings[start:start + B]
        try:
            corrected = checker.correct_strings(batch)
        except Exception as e:  # noqa: BLE001
            print(f"[neuspell] batch {start} failed: {e}")
            continue
        for k, corr in enumerate(corrected):
            orig = batch[k]
            if _norm(corr) != _norm(orig):
                flagged[cells[idx[start + k]][0]] = corr
    return flagged


TOOLS = {"symspell": run_symspell, "neuspell": run_neuspell}


# --------------------------------------------------------------------------- #
def collect_cells(ds_root: Path, ds_names: list[str]) -> tuple[dict[str, list[tuple[Cell, str]]], list[str]]:
    """Return {dataset: [((ds,i,j), value), ...]} for every cell, and loaded ds."""
    per_ds: dict[str, list[tuple[Cell, str]]] = {}
    loaded: list[str] = []
    for ds in ds_names:
        try:
            df = C.load_rein_df(ds_root / ds / "dirty.csv")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {ds}: {e}")
            continue
        loaded.append(ds)
        items: list[tuple[Cell, str]] = []
        for j, col in enumerate(df.columns):
            colvals = df.iloc[:, j].tolist()
            for i, v in enumerate(colvals):
                items.append(((ds, i, j), "" if v is None else str(v)))
        per_ds[ds] = items
    return per_ds, loaded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", action="append", required=True, choices=list(TOOLS),
                    help="speller(s) to run (repeatable)")
    ap.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="default", choices=["default", "aggressive"],
                    help="aggressive lowers thresholds to maximize recall/coverage")
    ap.add_argument("--max-edit-distance", type=int, default=2)
    ap.add_argument("--min-token-len", type=int, default=4)
    ap.add_argument("--skip-allcaps", dest="skip_allcaps", action="store_true", default=True)
    ap.add_argument("--no-skip-allcaps", dest="skip_allcaps", action="store_false")
    ap.add_argument("--flag-unknown", action="store_true", default=False,
                    help="(SymSpell) also flag out-of-dictionary tokens with no near word")
    ap.add_argument("--neuspell-checker", default="BertChecker")
    ap.add_argument("--neuspell-batch", type=int, default=256)
    args = ap.parse_args()

    if args.mode == "aggressive":
        args.min_token_len = min(args.min_token_len, 3)
        args.skip_allcaps = False

    ds_root = Path(args.datasets_dir)
    manifest = ds_root / "_manifest.json"
    if not manifest.exists():
        raise SystemExit(f"Run external/rein/prepare_datasets.py first (missing {manifest}).")
    ds_names = sorted(json.loads(manifest.read_text())["datasets"])
    if args.limit:
        ds_names = ds_names[: args.limit]

    print(f"collecting cells from {len(ds_names)} datasets...")
    per_ds, loaded = collect_cells(ds_root, ds_names)
    flat = [item for ds in loaded for item in per_ds[ds]]
    n_textual = sum(1 for _, v in flat if HAS_ALPHA.search(v))
    print(f"  {len(flat)} cells total, {n_textual} textual (have a letter)")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    preds_by_tool: dict[str, dict[Cell, str | None]] = {}
    for tool in args.tool:
        t0 = time.time()
        preds_by_tool[tool] = TOOLS[tool](flat, args)
        print(f"  {tool}: {len(preds_by_tool[tool])} cells flagged in {time.time()-t0:.1f}s")

    tools_to_write: dict[str, dict[Cell, str | None]] = dict(preds_by_tool)
    if len(args.tool) > 1:
        union: dict[Cell, str | None] = {}
        for s in preds_by_tool.values():
            for cell in s:
                union.setdefault(cell, None)
        tools_to_write["union"] = union
        print(f"  union: {len(union)} cells flagged")

    for tool, cells in tools_to_write.items():
        tag = tool if tool == "union" else f"{tool}_{args.mode}"
        out_path = results_dir / f"predictions_{tag}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for (ds, i, j) in sorted(cells):
                rec = {"dataset": ds, "detector": tag, "row": i, "col": j}
                cv = cells.get((ds, i, j))
                if cv is not None:
                    rec["corrected_value"] = cv
                f.write(json.dumps(rec) + "\n")
        ran = {"per_detector": {tag: loaded}, "all_loaded": loaded}
        Path(str(out_path) + ".ran.json").write_text(json.dumps(ran))
        print(f"  -> {out_path}")

    print("\nScore with, e.g.:")
    for tool in tools_to_write:
        tag = tool if tool == "union" else f"{tool}_{args.mode}"
        print(f"  python external/rein/score.py --predictions {results_dir / f'predictions_{tag}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
