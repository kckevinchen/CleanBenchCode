"""Build the canonical CleanBench artifact from current workbench labels.

The command refreshes ``benchmark/CleanBench`` in place and produces the cell,
row, ignored, neutral, and sanitized-source artifacts. Existing content-keyed
LLM caches are retained unless ``--fresh`` is passed.

Run::

    python -m scripts.export_cleanbench
    python -m scripts.export_cleanbench --no-llm
    python -m scripts.export_cleanbench --pools excel_500
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "CleanBench"
CACHE_FILES = ("llm_explanation_cache.jsonl", "llm_row_cache.jsonl")


def _run_module(
    module: str,
    args: list[str],
    log_path: Path,
    *,
    stream_stdout: bool = True,
) -> None:
    """Run one exporter and abort with its log tail on failure."""
    command = [sys.executable, "-m", module, *args]
    print(f"\n$ {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=None if stream_stdout else log,
            stderr=log,
            check=False,
        )
    if result.returncode == 0:
        return
    tail = "".join(log_path.read_text(encoding="utf-8").splitlines(keepends=True)[-25:])
    raise SystemExit(
        f"\n{module} failed (exit {result.returncode}). " f"Last log lines ({log_path}):\n{tail}"
    )


def _reset_output(*, fresh: bool) -> None:
    retained: dict[str, bytes] = {}
    readme = OUTPUT_DIR / "README.md"
    if readme.is_file():
        retained[readme.name] = readme.read_bytes()
    if not fresh:
        for name in CACHE_FILES:
            path = OUTPUT_DIR / name
            if path.is_file():
                retained[name] = path.read_bytes()

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)
    for name, content in retained.items():
        (OUTPUT_DIR / name).write_bytes(content)
    retained_caches = sorted(set(retained) & set(CACHE_FILES))
    print(f"retained caches: {retained_caches or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pools",
        nargs="*",
        default=None,
        help="restrict to registered source pools (default: all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="parallel LLM calls for cell and row explanations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap new LLM calls per exporter (0 = all)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="make no new LLM calls; retained cached explanations still apply",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="drop existing content-keyed LLM caches before rebuilding",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="skip sanitized source directory and ZIP generation",
    )
    args = parser.parse_args()

    _reset_output(fresh=args.fresh)
    logs = Path(tempfile.mkdtemp(prefix="cleanbench_"))
    pool_args = ["--pools", *args.pools] if args.pools else []
    llm_args = ["--concurrency", str(args.concurrency)]
    if args.limit:
        llm_args.extend(["--limit", str(args.limit)])
    if args.no_llm:
        llm_args.append("--no-llm")

    _run_module(
        "experiments.export_cell_benchmark",
        ["--out", str(OUTPUT_DIR), *pool_args, *llm_args],
        logs / "cell.log",
    )
    _run_module(
        "experiments.export_row_benchmark",
        ["--out", str(OUTPUT_DIR), *pool_args, *llm_args],
        logs / "row.log",
    )

    with tempfile.TemporaryDirectory(prefix="cleanbench_ignored_") as scratch:
        _run_module(
            "experiments.export_cleanbench",
            ["--out", scratch, *pool_args],
            logs / "ignored.log",
            stream_stdout=False,
        )
        ignored = Path(scratch) / "ignored.jsonl"
        if not ignored.is_file():
            raise SystemExit("experiments.export_cleanbench produced no ignored.jsonl")
        shutil.copy2(ignored, OUTPUT_DIR / "ignored.jsonl")

    _run_module(
        "experiments.export_neutral_cells",
        ["--out", str(OUTPUT_DIR / "neutral_cells.jsonl"), *pool_args],
        logs / "neutral.log",
    )
    if not args.no_sources:
        _run_module(
            "experiments.export_sanitized_sources",
            ["--out", str(OUTPUT_DIR / "sanitized_sources")],
            logs / "sources.log",
        )

    files = sorted(path.name for path in OUTPUT_DIR.iterdir())
    print(f"\ndone: built {OUTPUT_DIR}")
    print(f"  files: {files}")
    print(f"  logs : {logs}")


if __name__ == "__main__":
    main()
