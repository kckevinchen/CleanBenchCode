"""Out-of-process runner for the LLM-emitted ``check(df)`` predicate.

The relation validator (VERIFY) and detector (DETECT) used to ``exec`` the
LLM's ``check()`` in-process on a daemon thread whose timeout could not
actually stop a runaway predicate: an infinite loop or a huge allocation
kept burning CPU / RAM until the whole server process exited. This module
runs ``check()`` in a **killable, resource-capped child process** instead:

  * **parent** -- :func:`run_check_subprocess` pickles ``(code, payload)``
    to a fresh interpreter, enforces a hard wall-clock timeout (the child
    is SIGKILLed on expiry), and re-raises ``TimeoutError`` /
    ``CodeValidationError`` / ``RuntimeError`` so the call site's error
    handling is unchanged from the in-process contract.
  * **child** (:func:`main`, run via ``python -m ..._check_runner``) --
    applies the SAME AST guard (defence in depth), sets ``RLIMIT_CPU`` +
    ``RLIMIT_AS``, ``exec``s + runs ``check()``, coerces the result, and
    writes the boolean mask back to a file the parent reads.

The child imports only :mod:`._sandbox` (ast / re / numpy / pandas) -- never
the LLM client and its Azure/OpenAI SDKs -- so per-call spawn stays cheap
(the spawn is dwarfed by the LLM round-trip that precedes each check).
"""

from __future__ import annotations

import math
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from ._sandbox import (
    CodeValidationError,
    _coerce_to_bool_series,
    _exec_check_code,
    _validate_code_ast,
)

# src/core/relations/_check_runner.py -> project root is parents[3].
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Generous virtual-memory cap for the child: plenty for pandas on a
# sampled frame, but turns an allocation bomb into a catchable MemoryError
# long before it can OOM the host.
_MEM_LIMIT_BYTES = 4 * 1024**3  # 4 GiB

Payload = Union[pd.DataFrame, Dict[str, pd.DataFrame]]


def _expected_len(payload: Payload) -> int:
    if isinstance(payload, pd.DataFrame):
        return len(payload)
    return sum(len(df) for df in payload.values())


# --------------------------------------------------------------- parent side


def run_check_subprocess(
    code: str,
    payload: Payload,
    timeout_seconds: float,
) -> pd.Series:
    """Run ``check(payload)`` in a killable, resource-capped subprocess.

    ``payload`` is a single ``DataFrame`` (signature ``check(df)``) or a
    ``Dict[str, DataFrame]`` (multi-table, ``check(dfs)``). Returns the
    coerced boolean Series -- byte-for-byte what the in-process path
    produced, since the child uses the same coercion and pickling
    preserves the index.

    Raises:
      * ``TimeoutError`` if the child exceeds ``timeout_seconds`` (it is
        SIGKILLed);
      * ``CodeValidationError`` for AST / compile / sandbox failures;
      * ``Exception`` (e.g. ``RuntimeError``) for errors inside ``check()``.
    """
    request = pickle.dumps({
        "code": code,
        "payload": payload,
        "timeout_seconds": float(timeout_seconds),
    })
    with tempfile.TemporaryDirectory(prefix="relcheck_") as td:
        out_path = Path(td) / "out.pkl"
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m",
                    "src.core.relations._check_runner", str(out_path),
                ],
                input=request,
                capture_output=True,
                timeout=timeout_seconds,
                cwd=str(_PROJECT_ROOT),
                # Own session/group so the child can't catch our signals
                # and is cleanly reaped; it spawns nothing itself.
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run already SIGKILLed the child and drained pipes.
            raise TimeoutError(
                f"check() did not return within {timeout_seconds:.1f}s"
            ) from exc

        if not out_path.exists():
            # Child died without producing a result: RLIMIT_CPU backstop,
            # segfault, OOM-kill, import failure, ...
            stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
            tail = stderr.splitlines()[-1] if stderr else f"exit code {proc.returncode}"
            raise RuntimeError(f"check() subprocess produced no result: {tail}")

        result = pickle.loads(out_path.read_bytes())

    status = result.get("status")
    if status == "ok":
        return result["mask"]
    msg = result.get("error") or "unknown error"
    if status == "sandbox_error":
        raise CodeValidationError(msg)
    raise RuntimeError(msg)


# ---------------------------------------------------------------- child side


def _set_resource_limits(timeout_seconds: float) -> None:
    """Best-effort RLIMIT_CPU + RLIMIT_AS for the current (child) process.

    ``RLIMIT_CPU`` is set *above* the parent's wall-clock timeout so a CPU
    spin is killed by the parent first (reported as a timeout); this only
    fires if the parent's SIGKILL somehow doesn't land. Limit-setting
    never aborts the run -- non-POSIX or a refused limit just means we
    fall back to the parent's wall-clock kill.
    """
    try:
        import resource
    except ImportError:
        return  # non-POSIX (e.g. Windows): wall-clock kill is the only cap
    cpu = int(math.ceil(timeout_seconds)) + 5
    for res_id, limit in (
        (getattr(resource, "RLIMIT_CPU", None), cpu),
        (getattr(resource, "RLIMIT_AS", None), _MEM_LIMIT_BYTES),
    ):
        if res_id is None:
            continue
        try:
            _soft, hard = resource.getrlimit(res_id)
            new_hard = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
            resource.setrlimit(res_id, (min(limit, new_hard), new_hard))
        except (ValueError, OSError):
            pass


def main() -> int:
    out_path = Path(sys.argv[1])
    request = pickle.loads(sys.stdin.buffer.read())
    code: str = request["code"]
    payload: Payload = request["payload"]
    timeout_seconds: float = request["timeout_seconds"]

    _set_resource_limits(timeout_seconds)

    result: Dict[str, Any]
    try:
        # Defence in depth: the parent AST-validated too, but never trust
        # that the thing that reached us is the thing that was checked.
        _validate_code_ast(code)
        fn = _exec_check_code(code)
    except CodeValidationError as exc:
        result = {"status": "sandbox_error", "error": str(exc)}
    else:
        try:
            out = fn(payload)
            mask = _coerce_to_bool_series(out, _expected_len(payload))
            result = {"status": "ok", "mask": mask}
        except Exception as exc:  # noqa: BLE001 - report, never crash silently
            result = {
                "status": "runtime_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    out_path.write_bytes(pickle.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
