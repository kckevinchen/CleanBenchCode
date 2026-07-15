"""Typo v8 RFT CLI for Azure OpenAI.

v8 reuses v7's grader, response schema, and base model. The only
differences from v7's launcher are:

* default ``--train-jsonl`` / ``--valid-jsonl`` point at
  ``RFT/typo/v8/data/``;
* default output dir is ``outputs/rft_typo_v8``;
* default suffix is ``typo-rft-v8``;
* default ``--n-epochs`` is **2** (up from v7's 1) -- with the higher
  starting recall (0.939 vs 0.758) the grader now has dense reward
  signal on engaged columns, so more iterations consolidate the
  precision filter without losing recall.

Three reward variants (same as v7):
    r1 -- additive    : 2.5 * T * C * (P + R)
    r2 -- harmonic    : 5   * T * C * 2PR / (P + R)
    r3 -- weighted    : 5   * T * C * (0.6P + 0.4R)

Usage::

    python -m RFT.typo.v8.scripts.typo_rft_v8 all \\
        --reward-variant r1 \\
        --output-dir outputs/rft_typo_v8_r1 \\
        --suffix v8-r1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse v7's command implementations verbatim. The only knobs we change
# are defaults wired through the argparse parser below.
from RFT.typo.v7.scripts import typo_rft_v7 as _v7  # noqa: E402
from RFT._common._rft_common import (  # noqa: E402
    add_cancel_args,
    add_common_args,
    add_models_args,
    add_status_args,
    cmd_cancel,
    cmd_models,
    cmd_status,
    configure_logging,
    resolve_endpoint_if_needed,
)

DEFAULT_OUTPUT_DIR = Path("outputs/rft_typo_v8")
DEFAULT_SUFFIX = "typo-rft-v8"
DEFAULT_N_EPOCHS = 2

DEFAULT_TRAIN_JSONL = "RFT/typo/v8/data/train.jsonl"
DEFAULT_VALID_JSONL = "RFT/typo/v8/data/valid.jsonl"


def _add_train_valid_jsonl_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    p.add_argument("--valid-jsonl", default=DEFAULT_VALID_JSONL)


def _add_submit_args(p: argparse.ArgumentParser) -> None:
    """Mirror v7._add_submit_args but with v8 defaults."""
    p.add_argument(
        "--reward-variant",
        choices=list(_v7.V7_VARIANTS),
        required=True,
        help="Which v7-style reward shape to use (r1=additive, r2=harmonic, "
             "r3=weighted). v8 reuses v7's grader unchanged.",
    )
    p.add_argument("--base-model", default=_v7.DEFAULT_BASE_MODEL)
    p.add_argument(
        "--grader-model", default=_v7.DEFAULT_GRADER_MODEL,
        choices=list(_v7._v5.TYPO_V5_GRADER_MODEL_WHITELIST),
    )
    p.add_argument("--suffix", default=DEFAULT_SUFFIX)
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--eval-samples", type=int, default=2)
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"], default="medium",
    )
    p.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    p.add_argument("--compute-multiplier", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--skip-model-check", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_train_valid_jsonl_args(sp)
    sp.set_defaults(func=_v7.cmd_prepare)

    sp = sub.add_parser("upload")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    sp.set_defaults(func=_v7.cmd_upload)

    sp = sub.add_parser("submit")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_submit_args(sp)
    sp.set_defaults(func=_v7.cmd_submit)

    sp = sub.add_parser("status")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_status_args(sp)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("cancel")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_cancel_args(sp)
    sp.set_defaults(func=cmd_cancel)

    sp = sub.add_parser("models")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_models_args(sp)
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("all")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_train_valid_jsonl_args(sp)
    _add_submit_args(sp)
    sp.add_argument("--poll-interval", type=int, default=30)
    sp.set_defaults(func=_v7.cmd_all)

    return p


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_endpoint_if_needed(args, network_command=args.command != "prepare")
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
