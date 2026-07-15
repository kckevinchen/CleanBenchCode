"""Build v10 train/valid/eval/demo splits.

v10 = v9 with ONLY the trainee prompt retargeted. This builder reads the
already-built v9 splits (``RFT/typo/v9/data/``) and replaces each row's
``messages[0].content`` with ``RFT.typo.v10.PROMPT_V10``. Nothing else is
touched: gold cells, ambiguous_values, and every other field are copied
byte-for-byte, so the eval split stays disjoint and the gold reconciliation
decisions baked into v9 carry over unchanged.

Usage::

    python -m RFT.typo.v10.scripts.build_v10_splits
    python -m RFT.typo.v10.scripts.build_v10_splits --source-data-dir RFT/typo/v9/data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v10 import PROMPT_V10  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_v10_splits")

SPLITS = ("train.jsonl", "valid.jsonl", "eval.jsonl", "demo.jsonl")


def _retarget_prompt(rec: dict, new_prompt: str) -> dict:
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("row missing non-empty 'messages'")
    if msgs[0].get("role") != "developer":
        raise ValueError(
            f"messages[0] expected role=developer, got {msgs[0].get('role')!r}"
        )
    msgs = list(msgs)
    msgs[0] = {**msgs[0], "content": new_prompt}
    return {**rec, "messages": msgs}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source-data-dir", type=Path,
                   default=_PROJECT_ROOT / "RFT" / "typo" / "v9" / "data")
    p.add_argument("--output-dir", type=Path,
                   default=_PROJECT_ROOT / "RFT" / "typo" / "v10" / "data")
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_split_written = {}

    for fname in SPLITS:
        src = args.source_data_dir / fname
        if not src.exists():
            logger.warning("source %s missing -- skipped", fname)
            continue
        out_rows = []
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = _retarget_prompt(rec, PROMPT_V10)
            out_rows.append(rec)
        dst = args.output_dir / fname
        with dst.open("w", encoding="utf-8") as fh:
            for r in out_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_split_written[fname] = len(out_rows)
        logger.info("wrote %d rows -> %s", len(out_rows), dst)

    manifest = {
        "source_data_dir": str(args.source_data_dir),
        "prompt_module": "RFT.typo.v10._prompt.PROMPT_V10",
        "prompt_chars": len(PROMPT_V10),
        "forked_from": "RFT.typo.v9._prompt.PROMPT_V9 (recall-push wording removed)",
        "rows_written": per_split_written,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    logger.info("manifest -> %s", args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
