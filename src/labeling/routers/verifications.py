"""Web + column verification listing endpoints.

Read-only views over the verification artifacts written by
``experiments.run_web_verify`` (``web_verifications.json``) and
``experiments.run_web_verify_column`` (per-column verdict JSONL). The
verifier itself runs offline; these endpoints just surface its output.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, Query

from .._helpers import _validate_source_name
from ._context import RouterContext

logger = logging.getLogger(__name__)


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache

    @app.get("/api/runs/{run_id}/web-verifications")
    def list_web_verifications(run_id: str) -> Dict[str, Any]:
        """Aggregate counts + the full list. The list is small enough
        (one row per gate-eligible cluster) that the labeling UI can
        cache it client-side and decide whether to render the
        verification block per cluster without a second round-trip."""
        ctx = cache.get(run_id)
        ctx._reload_verifications()  # noqa: SLF001  (intentional)
        records = list(ctx._verifications.values())  # noqa: SLF001
        counts = {"confirmed_error": 0, "false_positive": 0, "ambiguous": 0, "errored": 0}
        for r in records:
            if r.get("error"):
                counts["errored"] += 1
                continue
            v = str(r.get("verdict") or "ambiguous")
            counts[v] = counts.get(v, 0) + 1
        return {
            "available": ctx._verifications_path().exists(),  # noqa: SLF001
            "counts": counts,
            "items": records,
        }

    # -------- column-wide web verification (column-audit) ----------
    #
    # Auxiliary artifacts written by ``experiments.run_web_verify_column``.
    # One JSONL per audited (source, column) under
    # ``outputs/<run_id>/column_verifications/`` plus a ``_index.jsonl``
    # manifest. The endpoints below are read-only; the audit itself is
    # launched via POST /api/launcher/jobs (kind="column_verify") so it
    # benefits from the existing job-runner / log-tail UI.

    def _column_verify_dir(rd: Path) -> Path:
        return rd / "column_verifications"

    def _column_verify_path(rd: Path, source: str, column: str) -> Path:
        # File-name slug must match ``run_web_verify_column._slug`` so
        # the UI can predict the on-disk path before issuing the read.
        from experiments.run_web_verify_column import (  # noqa: WPS433
            column_artifact_path,
        )
        return column_artifact_path(rd, source, column)

    def _read_column_verify_jsonl(path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not path.exists():
            return out
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        out.append(rec)
        except OSError:
            return []
        return out

    @app.get("/api/runs/{run_id}/column-verifications")
    def list_column_verifications(run_id: str) -> Dict[str, Any]:
        """Manifest of every (source, column) audited by the column-wide
        verifier.

        Reads ``outputs/<run_id>/column_verifications/_index.jsonl``
        (rebuilt on every audit run by ``run_web_verify_column``).
        Falls back to scanning the directory when the manifest is
        missing (e.g. someone moved files in by hand).

        Response::

            {
              "available": bool,
              "items": [
                {"source", "column", "path", "counts": {...}, "ts"},
                ...
              ]
            }
        """
        rd = _run_dir_or_404(run_id)
        base = _column_verify_dir(rd)
        if not base.is_dir():
            return {"available": False, "items": []}
        index_path = base / "_index.jsonl"
        items: List[Dict[str, Any]] = []
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(rec, dict):
                            items.append(rec)
            except OSError as exc:
                logger.warning(
                    "column-verifications index unreadable for %s: %s",
                    run_id, exc,
                )
        if not items:
            # Fallback: synthesise the manifest on the fly. Cheap when
            # there are only a handful of files; the per-file read is
            # the same the dedicated endpoint does.
            for p in sorted(base.glob("*.jsonl")):
                if p.name.startswith("_"):
                    continue
                recs = _read_column_verify_jsonl(p)
                if not recs:
                    continue
                first = recs[0]
                counts = {
                    "confirmed_error": 0,
                    "false_positive": 0,
                    "ambiguous": 0,
                    "errored": 0,
                    "fn_candidates": 0,
                    "total": len(recs),
                }
                for r in recs:
                    if r.get("error"):
                        counts["errored"] += 1
                        continue
                    v = str(r.get("verdict") or "ambiguous")
                    counts[v] = counts.get(v, 0) + 1
                    if v == "confirmed_error" and not r.get("was_flagged_by_llm"):
                        counts["fn_candidates"] += 1
                items.append({
                    "source": first.get("source") or "",
                    "column": first.get("column") or "",
                    "path": str(p.relative_to(rd)),
                    "counts": counts,
                    "ts": max(
                        (str(r.get("ts") or "") for r in recs),
                        default="",
                    ),
                })
        return {"available": True, "items": items}

    @app.get("/api/runs/{run_id}/column-verifications/detail")
    def get_column_verification(
        run_id: str,
        source: str = Query(..., min_length=1),
        column: str = Query(..., min_length=1),
    ) -> Dict[str, Any]:
        """Return every audit record for one (source, column).

        ``source`` + ``column`` are query params (not path segments)
        because column headers may legitimately contain ``/`` and other
        characters URL-decoded path templating doesn't handle cleanly.

        Response::

            {
              "available": bool,
              "source": str,
              "column": str,
              "counts": {confirmed_error, false_positive, ambiguous,
                         errored, fn_candidates, total},
              "items": [<column-audit record>, ...]
            }
        """
        _validate_source_name(source)
        rd = _run_dir_or_404(run_id)
        path = _column_verify_path(rd, source, column)
        if not path.exists():
            return {
                "available": False,
                "source": source,
                "column": column,
                "counts": {
                    "confirmed_error": 0, "false_positive": 0,
                    "ambiguous": 0, "errored": 0, "fn_candidates": 0,
                    "total": 0,
                },
                "items": [],
            }
        items = _read_column_verify_jsonl(path)
        counts = {
            "confirmed_error": 0, "false_positive": 0, "ambiguous": 0,
            "errored": 0, "fn_candidates": 0, "total": len(items),
        }
        for r in items:
            if r.get("error"):
                counts["errored"] += 1
                continue
            v = str(r.get("verdict") or "ambiguous")
            counts[v] = counts.get(v, 0) + 1
            if v == "confirmed_error" and not r.get("was_flagged_by_llm"):
                counts["fn_candidates"] += 1
        # Sort: FN candidates first (highest-priority for the
        # operator), then confirmed_error, then ambiguous, then
        # false_positive. Within a bucket, by first-seen row.
        verdict_rank = {"confirmed_error": 0, "ambiguous": 1,
                        "false_positive": 2, "errored": 3}

        def _row0(r: Dict[str, Any]) -> int:
            rs = r.get("rows") or []
            if isinstance(rs, list) and rs:
                try:
                    return int(rs[0])
                except (TypeError, ValueError):
                    return 1 << 30
            return 1 << 30

        def _key(r: Dict[str, Any]):
            v = "errored" if r.get("error") else str(r.get("verdict") or "ambiguous")
            is_fn = v == "confirmed_error" and not r.get("was_flagged_by_llm")
            return (
                0 if is_fn else 1,
                verdict_rank.get(v, 99),
                _row0(r),
                str(r.get("value") or ""),
            )

        items.sort(key=_key)
        return {
            "available": True,
            "source": source,
            "column": column,
            "counts": counts,
            "items": items,
        }

    @app.get("/api/runs/{run_id}/column-verifications/progress")
    def get_column_verification_progress(run_id: str) -> Dict[str, Any]:
        """Live progress JSON for an in-flight column-audit run.

        Returns ``{"available": False}`` when no audit has ever run
        against this run; otherwise the parsed contents of
        ``column_verifications/_progress.json`` (which the script
        atomically rewrites on every column transition + every
        value-batch checkpoint). The labelling UI polls this endpoint
        while a column-audit job is running to drive a progress bar.
        """
        rd = _run_dir_or_404(run_id)
        path = _column_verify_dir(rd) / "_progress.json"
        if not path.exists():
            return {"available": False}
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            # Best-effort: if the file is being rewritten when we
            # read it we may catch an empty file or a half-written
            # one. Surface "available but no payload" so the UI keeps
            # polling rather than giving up.
            return {"available": True, "stale": True}
        if not isinstance(payload, dict):
            return {"available": True, "stale": True}
        payload["available"] = True
        return payload

