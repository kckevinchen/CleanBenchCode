"""Scaffold a new RFT task from ``RFT/_template/``.

    python -m RFT.cli new semantic_incompatible --findings-key findings

Creates ``RFT/<error_type>/``, rewrites the package imports and spec identity,
and prints the one registry line to add.
After that, editing the prompt + grader + building data splits is all that's
left.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

_RFT_DIR = Path(__file__).resolve().parent
_TEMPLATE = _RFT_DIR / "_template"
_ET_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def scaffold_task(
    error_type: str,
    *,
    findings_key: str = "findings",
    dest_root: Path | None = None,
) -> Path:
    if not _ET_RE.match(error_type):
        raise SystemExit(f"error_type {error_type!r} must be lower_snake_case (e.g. typo).")
    name = error_type
    dest_root = dest_root or _RFT_DIR

    dst = dest_root / error_type
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- choose another task name or remove it.")
    shutil.copytree(
        _TEMPLATE, dst, ignore=shutil.ignore_patterns("__pycache__", "data", "*.pyc")
    )

    replacements = [
        ("RFT._template", f"RFT.{error_type}"),
        ("example_pr", f"{name}_pr"),
        ('name="example"', f'name="{name}"'),
        ('error_type="example"', f'error_type="{error_type}"'),
        ('default_suffix="example-rft"', f'default_suffix="{name.replace("_", "-")}-rft"'),
    ]
    if findings_key != "findings":
        replacements.append(('FINDINGS_KEY = "findings"', f'FINDINGS_KEY = "{findings_key}"'))
    for py in dst.rglob("*.py"):
        text = py.read_text()
        for old, new in replacements:
            text = text.replace(old, new)
        py.write_text(text)

    (dst / "data").mkdir(exist_ok=True)
    _print_next_steps(error_type, name)
    return dst


def _print_next_steps(error_type: str, name: str) -> None:
    const = name.upper()
    print(
        f"\nScaffolded RFT/{error_type}/.\n\n"
        f"Next:\n"
        f"  1. Edit  RFT/{error_type}/_prompt.py   (TRAINEE_PROMPT + schema)\n"
        f"  2. Edit  RFT/{error_type}/_grader.py   (set TASK_DESCRIPTION)\n"
        f"  3. Build RFT/{error_type}/data/*.jsonl\n"
        f"       python -m RFT.{error_type}.scripts.build_splits --sources <csv_dir>\n"
        f"  4. Register the spec in RFT/registry.py:\n"
        f"         from RFT.{error_type}.spec import SPEC as {const}\n"
        f"         _ALL = (..., {const})\n\n"
        f"  Then: python -m RFT.cli --task {name} all --reward-variant default --endpoint <ep>\n"
    )


def add_new_subparser(subparsers) -> None:
    sp = subparsers.add_parser("new", help="Scaffold a new RFT task from the template.")
    sp.add_argument("error_type", help="Error type, lower_snake_case, e.g. typo.")
    sp.add_argument("--findings-key", default="findings",
                    help="JSON array key the trainee emits (default 'findings').")


def run_new(args: argparse.Namespace) -> None:
    scaffold_task(args.error_type, findings_key=args.findings_key)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="RFT.scaffold", description=__doc__)
    add_new_subparser(p.add_subparsers(dest="_cmd", required=True))
    args = p.parse_args(argv)
    run_new(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
