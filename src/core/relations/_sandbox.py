"""Pure, dependency-light sandbox primitives for LLM-emitted ``check()`` code.

Extracted from :mod:`src.core.relations.validator` so the out-of-process
check runner (:mod:`src.core.relations._check_runner`) can import the AST
guard, the restricted-builtins namespace, and the return-value coercion
*without* dragging in the LLM client (which imports the heavy Azure /
OpenAI SDKs). ``validator`` re-exports every symbol here, so existing
imports such as ``from ...validator import _exec_check_code`` keep working.

Threat model (the AST validator rejects):

  * any ``Import`` / ``ImportFrom`` node
  * any attribute access whose name starts/ends with ``__``
    (blocks ``__class__``, ``__subclasses__``, ...)
  * direct reference to ``eval``, ``exec``, ``compile``, ``open``,
    ``__import__``, ``input``, ``breakpoint``, and the dynamic-attribute
    primitives (``getattr``/``setattr``/``vars``/``type``/...)
  * ``Global`` / ``Nonlocal``
  * the ``str.format`` / ``format_map`` / ``%``-on-literal class-graph-walk
    vectors, and any ``__`` dunder smuggled inside a string literal

The runtime ``__builtins__`` is replaced with a whitelist of pure,
deterministic helpers; ``pd``, ``np``, and ``re`` are pre-bound, nothing
else. This is defence in depth, not the only line of defence: the runner
also executes in a killable, resource-capped child process.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd


_FORBIDDEN_NAMES = frozenset({
    # Hard-block direct references to escape vectors AND the dynamic-
    # attribute-access primitives that can be used to bypass the AST
    # dunder-attribute filter via runtime string concatenation
    # (``getattr(int, "_" + "_subclasses__")``). ``type``/``vars`` are
    # also blocked because both expose ``__bases__`` / ``__dict__``
    # routes to walk the class graph. Without these names the
    # taxonomy of escapes from a Python sandbox shrinks dramatically.
    #
    # ``hasattr`` is intentionally NOT blocked: it returns a bool
    # only and does not expose the attribute object, so it cannot be
    # used as an escape vector. The LLM occasionally uses it for
    # defensive checks (``hasattr(s, 'name')``) and rejecting it
    # forces an unnecessary "verifier failed: forbidden name" path.
    #
    # ``object`` is also intentionally NOT blocked: with ``getattr``
    # / ``setattr`` / ``__*__`` access all forbidden, bare ``object``
    # has no escape route, and the LLM legitimately uses it as a
    # pandas dtype marker (``pd.Series(..., dtype=object)``,
    # ``df.astype(object)``). The remaining ``type`` block is kept
    # because ``type(x).__bases__`` is the most common runtime class-
    # graph walk, but ``object`` alone yields no exploitable handle.
    "eval", "exec", "compile", "open", "__import__",
    "input", "breakpoint",
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars",
    "type",
})


_SAFE_BUILTINS: Dict[str, Any] = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)  # type: ignore[index]
    for name in (
        # Pure values + simple containers + math/iter helpers.
        # NOTE: introspection / dynamic-attribute primitives
        # (``getattr``, ``vars``, ``type``, ``globals``, ...) are
        # intentionally OMITTED. The AST validator below also rejects
        # any reference to them, so the LLM can rely on a small,
        # auditable namespace and we get defense-in-depth against
        # the runtime-string-attr escape pattern.
        "abs", "all", "any", "bin", "bool", "callable", "chr",
        "complex", "dict", "divmod", "enumerate", "filter", "float",
        # NOTE: ``format`` is intentionally omitted -- str.format /
        # format() drive attribute/index traversal from a format-spec
        # string the AST never inspects (a sandbox-escape vector).
        "frozenset", "hash", "hasattr", "hex",
        "id", "int", "isinstance", "issubclass", "iter", "len",
        "list", "map", "max", "min", "next", "object",
        "oct", "ord", "pow",
        "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip",
        "True", "False", "None", "Exception", "ValueError",
        "TypeError", "KeyError", "IndexError", "ArithmeticError",
        "ZeroDivisionError", "OverflowError", "FloatingPointError",
    ) if (name in __builtins__ if isinstance(__builtins__, dict)
          else hasattr(__builtins__, name))
}


class CodeValidationError(ValueError):
    """Raised when an LLM-emitted check() function fails AST validation."""


def _validate_code_ast(code: str) -> None:
    """Reject code that imports, uses dunders, or names builtins we ban.

    The check is intentionally strict; legitimate ``check`` functions
    only need ``df`` (DataFrame), ``pd``, ``np``, ``re`` and a small
    set of pure builtins.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CodeValidationError(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeValidationError("imports are not allowed in check()")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise CodeValidationError("global/nonlocal not allowed in check()")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr.endswith("__"):
                raise CodeValidationError(
                    f"dunder attribute access not allowed: {node.attr!r}"
                )
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise CodeValidationError(
                f"reference to forbidden name: {node.id!r}"
            )
        # The dunder-attribute check above only sees attribute access
        # written as syntax (``x.__class__``). The class-graph-walk
        # escape instead smuggles dunder names through a *string* that is
        # resolved at runtime by str.format / format_map / %-formatting
        # (the field/key spec the AST sees only as a Constant). Close
        # that by (a) rejecting those formatting calls and (b) rejecting
        # any string literal carrying a ``__`` dunder name.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                raise CodeValidationError(
                    "double-underscore in a string literal is not allowed "
                    "in check() (format-string / getattr-by-name escape)"
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("format", "format_map"):
                raise CodeValidationError(
                    f"str.{node.func.attr}() is not allowed in check() "
                    "(format-spec attribute-traversal escape)"
                )
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
        ):
            raise CodeValidationError(
                "%-formatting on a string literal is not allowed in check()"
            )


def _exec_check_code(code: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Compile + load ``code`` in a restricted namespace, return ``check``.

    ``code`` must define a top-level function named ``check`` that
    accepts a single ``DataFrame`` argument. The returned callable is
    bound to the sandboxed namespace so subsequent attribute accesses
    inside it cannot reach module-level state.
    """
    _validate_code_ast(code)
    namespace: Dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "re": re,
    }
    try:
        exec(compile(code, "<llm-check>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        raise CodeValidationError(f"compile/exec error: {exc}") from exc
    fn = namespace.get("check")
    if not callable(fn):
        raise CodeValidationError("code does not define a callable 'check'")
    return fn


def _coerce_to_bool_series(
    out: Any,
    expected_len: int,
) -> pd.Series:
    """Normalise the return value of ``check`` to a boolean Series.

    ``check`` is allowed to return:
      * pd.Series  (preferred)
      * np.ndarray (1-D)
      * list / tuple
      * a scalar bool (broadcast to ``expected_len`` rows)

    Any other type is an error.
    """
    if isinstance(out, pd.Series):
        s = out
    elif isinstance(out, np.ndarray):
        s = pd.Series(out)
    elif isinstance(out, (list, tuple)):
        s = pd.Series(out)
    elif isinstance(out, (bool, np.bool_)):
        s = pd.Series([bool(out)] * expected_len)
    else:
        raise TypeError(
            f"check() must return a Series/array/list/bool; got "
            f"{type(out).__name__}"
        )
    if len(s) != expected_len:
        raise ValueError(
            f"check() returned {len(s)} values but DataFrame has "
            f"{expected_len} rows"
        )
    # NaN-aware bool cast: treat NaN as False (rule unverified for that
    # row). Coerce non-bool truthy/falsy values to bool.
    s = s.fillna(False).astype(bool)
    return s
