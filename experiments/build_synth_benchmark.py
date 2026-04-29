"""Build a synthetic-error benchmark from a clean CSV directory.

Workflow
--------
1. Walk the input directory and load every CSV.
2. For each ``--error-types`` injector, pick ``rate * n_cells`` cells
   uniformly at random (deterministic given ``--seed``), filtered to
   cells the injector says it can perturb (``Injector.applicable``).
3. Apply the injector. Skip cells where injection would be a no-op
   (e.g. value already in target casing). Record every accepted
   injection in ``injection_manifest.jsonl``.
4. Persist the perturbed CSVs to ``<output-dir>/injected/`` plus a
   ``benchmark_config.json`` describing the run.

The output directory is meant to be passed straight into
``experiments.run_cleaning_detection`` (LLM) or
``experiments.run_heuristic_detection`` (heuristic). Scoring is
done by ``experiments.score_synth_benchmark``.

Usage::

    python -m experiments.build_synth_benchmark \\
        --input data/curated_pbi_full/271841717 \\
        --output-dir outputs/synth_271841717 \\
        --rate 0.02 \\
        --error-types typo,casing,whitespace,punctuation,number_format,\\
                      pattern,numeric_outlier,missing_value \\
        --seed 17
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import load_csv  # noqa: E402
from src.synth import INJECTOR_REGISTRY, InjectionRecord  # noqa: E402

logger = logging.getLogger("build_synth_benchmark")


def _discover_csvs(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".csv")


def _cell_to_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _inject_dataframe(
    df: pd.DataFrame,
    *,
    source: str,
    injector_name: str,
    rate: float,
    rng: random.Random,
    max_attempts_multiplier: int = 3,
) -> tuple[pd.DataFrame, List[InjectionRecord]]:
    """Apply ``injector_name`` to ~rate fraction of cells in ``df``.

    Returns the perturbed DataFrame (a copy) and the list of accepted
    injection records. Cells already touched by a previous injector
    on the same DataFrame should not be passed in - the builder
    serialises injectors so we don't have to defensively dedup here.
    """
    cls = INJECTOR_REGISTRY[injector_name]
    inj = cls()
    rows, cols = df.shape
    n_cells = rows * cols
    target = max(1, int(round(rate * n_cells)))

    # Build the candidate cell list once: all (r, c) where the
    # injector says applicable. This avoids hammering applicable()
    # inside the random-pick loop.
    candidates: List[tuple[int, str]] = []
    str_view: Dict[tuple[int, str], str] = {}
    for c in df.columns:
        col_values = df[c].tolist()
        for r, raw in enumerate(col_values):
            s = _cell_to_str(raw)
            if inj.applicable(s):
                candidates.append((r, c))
                str_view[(r, c)] = s
    if not candidates:
        logger.info("[%s|%s] no applicable cells", source, injector_name)
        return df.copy(), []

    rng.shuffle(candidates)
    new_df = df.copy()
    records: List[InjectionRecord] = []
    attempts = 0
    max_attempts = target * max_attempts_multiplier
    for cell in candidates:
        if len(records) >= target or attempts >= max_attempts:
            break
        attempts += 1
        r, c = cell
        original = str_view[cell]
        result = inj.try_inject(original, rng=rng)
        if result is None:
            continue
        new_value, params = result
        new_df.iat[r, new_df.columns.get_loc(c)] = new_value
        records.append(
            InjectionRecord(
                source=source,
                row=r,
                column=str(c),
                error_type=inj.error_type,
                injector=inj.name,
                original_value=original,
                injected_value=new_value,
                params=params,
            )
        )
    logger.info(
        "[%s|%s] injected %d/%d (target %d, candidates %d)",
        source, injector_name, len(records), attempts, target, len(candidates),
    )
    return new_df, records


def build_benchmark(
    *,
    input_dir: Path,
    output_dir: Path,
    rate: float,
    injector_names: Sequence[str],
    seed: int,
) -> Dict[str, object]:
    csv_paths = _discover_csvs(input_dir)
    if not csv_paths:
        raise SystemExit(f"no CSVs found under {input_dir}")
    injected_dir = output_dir / "injected"
    injected_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "injection_manifest.jsonl"

    base_rng = random.Random(seed)
    all_records: List[InjectionRecord] = []
    per_source_counts: Dict[str, Dict[str, int]] = {}

    with manifest_path.open("w", encoding="utf-8") as mf:
        for path in csv_paths:
            source = path.stem
            df = load_csv(path)
            current = df
            per_source_counts[source] = {}
            # Serialise injectors so the same cell isn't perturbed
            # twice. Each injector gets its own deterministic sub-RNG
            # derived from (seed, source, name) for reproducibility.
            for name in injector_names:
                if name not in INJECTOR_REGISTRY:
                    raise SystemExit(
                        f"unknown injector {name!r}; "
                        f"available: {sorted(INJECTOR_REGISTRY)}"
                    )
                sub_rng = random.Random(
                    base_rng.random() if False else
                    hash((seed, source, name)) & 0xFFFFFFFF
                )
                current, records = _inject_dataframe(
                    current,
                    source=source,
                    injector_name=name,
                    rate=rate,
                    rng=sub_rng,
                )
                per_source_counts[source][name] = len(records)
                for rec in records:
                    mf.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                all_records.extend(records)
            out_path = injected_dir / path.name
            current.to_csv(out_path, index=False)

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rate": rate,
        "seed": seed,
        "injectors": list(injector_names),
        "n_sources": len(csv_paths),
        "n_injections_total": len(all_records),
        "per_source_counts": per_source_counts,
    }
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "wrote %d injections across %d CSVs to %s",
        len(all_records), len(csv_paths), output_dir,
    )
    return config


def main() -> None:
    p = argparse.ArgumentParser(description="Build synthetic-error benchmark")
    p.add_argument("--input", type=Path, required=True,
                   help="clean CSV file or directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rate", type=float, default=0.02,
                   help="fraction of cells to perturb per injector")
    p.add_argument("--error-types", type=str,
                   default="typo,casing,whitespace,punctuation,"
                           "number_format,pattern,numeric_outlier,missing_value",
                   help="comma-separated injector names; 'all' for every "
                        "registered injector")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--list-injectors", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.list_injectors:
        for name in sorted(INJECTOR_REGISTRY):
            cls = INJECTOR_REGISTRY[name]
            print(f"{name:20s} -> error_type={cls.error_type}")
        return

    spec = args.error_types.strip()
    if spec.lower() == "all":
        injector_names = sorted(INJECTOR_REGISTRY)
    else:
        injector_names = [s.strip() for s in spec.split(",") if s.strip()]

    build_benchmark(
        input_dir=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        rate=args.rate,
        injector_names=injector_names,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
