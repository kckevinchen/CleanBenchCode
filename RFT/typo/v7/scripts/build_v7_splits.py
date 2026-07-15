"""Build v7 train/valid/eval splits.

v7 reuses v6's exact valid/eval/demo splits (so cross-version metrics
are directly comparable) and resamples ONLY the train split by
upsampling positive (has_typo) columns ``--pos-upsample`` times.

Rationale: v6 train was 33 has_typo + 33 non-typo (1:1, balance_ratio=1.0).
v6 FT models still collapsed to abstention because the reward was
empty-preds-friendly (see RFT.typo.v7._grader_variants). v7 fixes the
reward AND further skews train toward positive columns so the gradient
on "engage" examples dominates.

Trivial typos remain excluded (v6's --keep-trivial=False is preserved).

Default: ``--pos-upsample 2`` -> 66 pos + 33 non-pos (67% positive).

Usage::

    python -m RFT.typo.v7.scripts.build_v7_splits \\
        --v6-data-dir RFT/typo/v6/data \\
        --output-dir RFT/typo/v7/data \\
        --pos-upsample 2 \\
        --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_v7_splits")


def _is_positive(rec: dict) -> bool:
    gold = rec.get("gold_typos", "[]")
    try:
        return bool(json.loads(gold) if isinstance(gold, str) else gold)
    except Exception:
        return False


def _resample_train(
    train_rows: list[dict],
    *,
    pos_upsample: int,
    seed: int,
) -> list[dict]:
    """Keep every row once; duplicate positive rows ``pos_upsample-1`` more times.

    The duplicated rows are NOT the same Python dict (we json round-trip)
    so Azure RFT treats them as independent training examples. Order is
    shuffled deterministically.
    """
    if pos_upsample < 1:
        raise ValueError("--pos-upsample must be >= 1")
    pos = [r for r in train_rows if _is_positive(r)]
    neg = [r for r in train_rows if not _is_positive(r)]
    out: list[dict] = []
    out.extend(neg)
    out.extend(pos)
    for _ in range(pos_upsample - 1):
        out.extend(json.loads(json.dumps(r)) for r in pos)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v6-data-dir", type=Path, default=Path("RFT/typo/v6/data"))
    p.add_argument("--output-dir", type=Path, default=Path("RFT/typo/v7/data"))
    p.add_argument("--pos-upsample", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resample train.
    train_src = args.v6_data_dir / "train.jsonl"
    train_rows = _read_jsonl(train_src)
    n_pos_in = sum(1 for r in train_rows if _is_positive(r))
    n_neg_in = len(train_rows) - n_pos_in
    new_train = _resample_train(
        train_rows, pos_upsample=args.pos_upsample, seed=args.seed,
    )
    n_pos_out = sum(1 for r in new_train if _is_positive(r))
    n_neg_out = len(new_train) - n_pos_out
    train_dst = args.output_dir / "train.jsonl"
    _write_jsonl(new_train, train_dst)
    logger.info(
        "train: in=%d (pos=%d neg=%d)  ->  out=%d (pos=%d neg=%d, %.1f%% pos)",
        len(train_rows), n_pos_in, n_neg_in,
        len(new_train), n_pos_out, n_neg_out,
        100.0 * n_pos_out / len(new_train),
    )

    # 2. Copy valid/eval/demo verbatim from v6.
    for fname in ("valid.jsonl", "eval.jsonl", "demo.jsonl"):
        src = args.v6_data_dir / fname
        if not src.exists():
            logger.warning("v6 %s missing -- skipped", fname)
            continue
        dst = args.output_dir / fname
        shutil.copyfile(src, dst)
        logger.info("copied %s -> %s (%d rows)", src, dst, len(_read_jsonl(dst)))

    # 3. Manifest for forensics.
    manifest = {
        "v6_data_dir": str(args.v6_data_dir),
        "pos_upsample": args.pos_upsample,
        "seed": args.seed,
        "train": {"in": len(train_rows), "out": len(new_train),
                  "pos_in": n_pos_in, "pos_out": n_pos_out},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
