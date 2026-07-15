"""Build v8 train/valid/eval/demo splits.

v8 reuses v7's exact rows. The only difference is the trainee developer
message: every row's ``messages[0].content`` is replaced with
``RFT.typo.v8.PROMPT_V8``. Order, gold typos, ambiguous values,
column values, and the user message are byte-identical to v7.

Usage::

    python -m RFT.typo.v8.scripts.build_v8_splits \\
        --v7-data-dir RFT/typo/v7/data \\
        --output-dir RFT/typo/v8/data
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

from RFT.typo.v8 import PROMPT_V8  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_v8_splits")


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


def _process_file(src: Path, dst: Path) -> int:
    n = 0
    with src.open(encoding="utf-8") as in_fh, \
            dst.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = _retarget_prompt(rec, PROMPT_V8)
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v7-data-dir", type=Path, default=Path("RFT/typo/v7/data"))
    p.add_argument("--output-dir", type=Path, default=Path("RFT/typo/v8/data"))
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("train.jsonl", "valid.jsonl", "eval.jsonl", "demo.jsonl"):
        src = args.v7_data_dir / fname
        if not src.exists():
            logger.warning("v7 %s missing -- skipped", fname)
            continue
        dst = args.output_dir / fname
        n = _process_file(src, dst)
        logger.info("re-prompted %d rows: %s -> %s", n, src, dst)

    manifest = {
        "v7_data_dir": str(args.v7_data_dir),
        "prompt_module": "RFT.typo.v8._prompt.PROMPT_V8",
        "prompt_chars": len(PROMPT_V8),
        "prompt_words": len(PROMPT_V8.split()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
