"""Run the web-grounded verifier over a finished detection run.

Reads ``outputs/<run_id>/clusters.json`` + ``summary.json``, applies the
entity-gate (:mod:`src.verifier.gate`), then runs every eligible
cluster through :class:`src.verifier.verify.Verifier` and writes the
results to ``outputs/<run_id>/web_verifications.json``.

The verifier delegates web search to a Microsoft Foundry agent that
has the Web Search tool attached. Auth uses ``AzureCliCredential`` -
run ``az login`` once and set ``AZURE_AGENTS_ENDPOINT`` +
``AZURE_AGENTS_VERIFIER_AGENT_ID`` (or pass them via flags).

Usage::

    # default (env vars provide endpoint + agent id):
    python -m experiments.run_web_verify --run-id 20260428T120000Z_gpt-5.2

    # explicit overrides:
    python -m experiments.run_web_verify --run-id <id> \
        --agent-endpoint https://<rsrc>.services.ai.azure.com/api/projects/<proj> \
        --agent-id asst_xxxxxxxxxxxxxx

    # dry-run gate only (prints which clusters would be verified):
    python -m experiments.run_web_verify --run-id <id> --dry-run

    # rebuild from scratch (ignore on-disk cache + previous output):
    python -m experiments.run_web_verify --run-id <id> --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import load_csv  # noqa: E402
from src.verifier import (  # noqa: E402
    AzureAgentBackend,
    GateConfig,
    Verifier,
    VerifierConfig,
    classify_cluster,
    cluster_shape,
    should_verify,
)
from src.verifier.verify import (  # noqa: E402
    load_verifications,
    write_verifications,
)
from src.verifier.web_cache import WebCache  # noqa: E402

logger = logging.getLogger("run_web_verify")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="Run directory under outputs/")
    p.add_argument(
        "--outputs-dir", type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Root outputs directory (default: outputs/).",
    )
    p.add_argument(
        "--agent-endpoint", default=None,
        help="Foundry project endpoint URL. Falls back to "
             "AZURE_AGENTS_ENDPOINT.",
    )
    p.add_argument(
        "--agent-id", default=None,
        help="Foundry agent id (asst_...). Falls back to "
             "AZURE_AGENTS_VERIFIER_AGENT_ID.",
    )
    p.add_argument("--model", default="gpt-5.2",
                   help="Stamped on each Verification record for traceability; "
                        "the actual model is configured on the Foundry agent.")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-clusters", type=int, default=None,
                   help="Cap on clusters to verify (after gating). Useful for smoke tests.")
    p.add_argument("--error-types", default=None,
                   help="Comma-separated error_type whitelist. When set, only "
                        "clusters whose error_types intersect this list are "
                        "considered (gate still applies on top). Example: "
                        "--error-types typo,inconsistency.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print gate verdict and exit without making web/agent calls.")
    p.add_argument("--force", action="store_true",
                   help="Ignore on-disk web cache + existing web_verifications.json.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _parse_error_types(raw: Optional[str]) -> Optional[frozenset[str]]:
    """Parse the --error-types CLI value into a frozenset, or None."""
    if not raw:
        return None
    items = [t.strip() for t in raw.split(",") if t.strip()]
    return frozenset(items) if items else None


# ---------------------------------------------------------------------------


def _frame_loader(csv_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """Load every CSV path once, key by basename. We accept a missing
    file silently (corpora occasionally get pruned between runs) -
    affected clusters will get ``error: suspect cell value unavailable``.
    """
    out: Dict[str, pd.DataFrame] = {}
    for p in csv_paths:
        path = Path(p)
        if not path.is_file():
            logger.warning("CSV missing, skipping: %s", path)
            continue
        try:
            out[path.name] = load_csv(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load %s: %s", path, exc)
    return out


def _build_lookups(frames: Dict[str, pd.DataFrame]):
    """Return (cell_lookup, column_samples, row_lookup) closures over ``frames``."""

    def cell_lookup(source: str, row: int, column: str) -> Optional[str]:
        f = frames.get(source)
        if f is None or column not in f.columns or not (0 <= row < len(f)):
            return None
        v = f.iat[row, f.columns.get_loc(column)]
        if v is None:
            return ""
        return str(v)

    def column_samples(source: str, column: str, n: int) -> List[str]:
        f = frames.get(source)
        if f is None or column not in f.columns:
            return []
        # Distinct, non-empty, in dataset order.
        seen: set[str] = set()
        out: List[str] = []
        for v in f[column].astype(str).tolist():
            v = v.strip()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
            if len(out) >= n:
                break
        return out

    def row_lookup(source: str, row: int) -> Dict[str, Any]:
        """Return the full row as ``{column: value}`` (str values).

        Empty / NaN cells are kept as empty strings so the LLM can
        distinguish "field present but blank" from "field absent".
        Used by Verifier.row_lookup to feed the FULL ROW prompt block.
        """
        f = frames.get(source)
        if f is None or not (0 <= row < len(f)):
            return {}
        out: Dict[str, Any] = {}
        for col in f.columns:
            v = f.iat[row, f.columns.get_loc(col)]
            if v is None:
                out[str(col)] = ""
                continue
            try:
                # pandas NaN -> empty.
                import pandas as _pd
                if isinstance(v, float) and _pd.isna(v):
                    out[str(col)] = ""
                    continue
            except Exception:
                pass
            out[str(col)] = str(v)
        return out

    return cell_lookup, column_samples, row_lookup


def _summarise(verifications: List, dry_run: bool) -> Dict[str, int]:
    counts = {"confirmed_error": 0, "false_positive": 0, "ambiguous": 0, "errored": 0}
    for v in verifications:
        if v.error:
            counts["errored"] += 1
            continue
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return counts


# ---------------------------------------------------------------------------


class VerifyError(Exception):
    """Raised by :func:`verify_single_cluster` when the cluster cannot
    be verified for a deterministic reason (gate-rejected, missing
    cell value, already verified, etc.). Callers - particularly the
    HTTP endpoint - map ``code`` to the right status. Transient
    failures (agent 5xx, Foundry timeouts) are NOT raised here; they
    end up as a normal :class:`Verification` record with ``error`` set,
    just like in batch mode.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code  # e.g. "not_found" / "already_verified" / "gate_rejected"
        self.message = message


def verify_single_cluster(
    run_dir: Path,
    cluster_id: str,
    *,
    agent_endpoint: Optional[str] = None,
    agent_id: Optional[str] = None,
    model: str = "gpt-5.2",
    reasoning_effort: str = "medium",
    include_person: bool = True,
    retry_errored: bool = False,
) -> "Verification":
    """Verify ONE cluster on demand and append the result to
    ``run_dir/web_verifications.json`` atomically.

    Used by ``POST /api/runs/{run_id}/clusters/{cid}/web-verification``
    so labelers can ask for evidence on a cluster the batch verifier
    didn't pick up. Reuses the same gate / prompt / parsing path as
    :func:`verify_run` so verdicts are identical to a full re-run.
    Raises :class:`VerifyError` for non-transient problems; transient
    agent errors surface as a :class:`Verification` whose ``.error``
    is set (exactly as in batch mode).

    ``retry_errored`` is the UI's "retry" affordance: when set, an
    existing record whose ``error`` is non-null (e.g. an Azure agent
    timeout / DecodeError) is treated as if it didn't exist - we
    re-run the agent and overwrite the prior record. A non-errored
    prior record is still refused (refuse-to-clobber).
    """
    from src.verifier.verify import (  # noqa: WPS433  (local import)
        Verification,
        load_verifications,
        write_verifications,
    )

    clusters_path = run_dir / "clusters.json"
    summary_path = run_dir / "summary.json"
    if not clusters_path.exists():
        raise VerifyError("not_found", f"clusters.json missing: {clusters_path}")

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    cluster = next(
        (c for c in clusters
         if isinstance(c, Mapping) and c.get("cluster_id") == cluster_id),
        None,
    )
    if cluster is None:
        raise VerifyError("not_found", f"cluster not found: {cluster_id}")

    # Refuse to clobber an existing verification - the UI shouldn't
    # offer the button in that case, but this guards stale clients.
    # Exception: when ``retry_errored`` is set AND the prior record
    # is itself errored, allow the retry (the operator explicitly
    # asked for it).
    prior_path = run_dir / "web_verifications.json"
    prior = load_verifications(prior_path)
    prior_match = next(
        (v for v in prior if v.cluster_id == cluster_id), None,
    )
    if prior_match is not None:
        prior_errored = bool(getattr(prior_match, "error", None))
        if not (retry_errored and prior_errored):
            raise VerifyError(
                "already_verified",
                f"cluster {cluster_id!r} already has a verification",
            )

    # CSV access (frame loader) so we can resolve the cell + row.
    summary: Mapping[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8")) or {}
    csv_paths = (summary.get("config") or {}).get("csv_paths") or []
    frames = _frame_loader(csv_paths)
    cell_lookup, column_samples, row_lookup = _build_lookups(frames)

    # Run the same gate the batch flow uses so behaviour is consistent.
    gate_cfg = GateConfig(verify_person=include_person)
    shape = cluster_shape(cluster)
    pattern = shape["pattern"]
    cols = shape["columns"]
    col = cols[0] if cols else ""
    row = cluster.get("row")
    val: Optional[str] = None
    # Cell value is only mandatory for single_cell clusters; multi-
    # row / multi-column clusters use row_lookup instead.
    if pattern == "single_cell" and isinstance(row, int) and col:
        val = cell_lookup(cluster.get("source", ""), row, col)
    if not should_verify(cluster, cell_value=val, cfg=gate_cfg):
        # Distinguish "no value at all" (label gate-rejected, but
        # tell the caller why) from a generic gate reject. The
        # "cell value unavailable" branch only applies to single-
        # cell clusters; multi-row / multi-column clusters fail
        # the gate for structural reasons (too few rows / columns,
        # non-web-verifiable error type, etc.).
        if pattern == "single_cell" and (val is None or not str(val).strip()):
            raise VerifyError(
                "gate_rejected",
                "Cell value unavailable - cannot verify on the web.",
            )
        raise VerifyError(
            "gate_rejected",
            "Cluster does not match the entity-verification gate "
            "(non-entity column / unsuitable error type / value too "
            "long-or-numeric / too few rows or columns for its "
            "pattern).",
        )

    # Cache + agent backend. Same on-disk cache as the batch path so a
    # manual verify benefits from any previous searches.
    cache = WebCache(run_dir / "web_cache")
    agent = AzureAgentBackend(
        endpoint=agent_endpoint or "",
        agent_id=agent_id or "",
        cache=cache,
    )
    vcfg = VerifierConfig(
        model=model,
        reasoning_effort=reasoning_effort,
        concurrency=1,
        gate=gate_cfg,
        force=False,
    )
    verifier = Verifier(
        agent=agent,
        cell_lookup=cell_lookup,
        column_samples=column_samples,
        row_lookup=row_lookup,
        cfg=vcfg,
    )
    fresh: List[Verification] = verifier.run([cluster])
    if not fresh:
        raise VerifyError("internal", "verifier returned no records")
    record = fresh[0]

    # Re-read the file before writing so we don't clobber a parallel
    # write from another operator clicking at the same time.
    final = load_verifications(prior_path)
    existing = next(
        (v for v in final if v.cluster_id == cluster_id), None,
    )
    if existing is not None:
        existing_errored = bool(getattr(existing, "error", None))
        if not (retry_errored and existing_errored):
            # Someone else got there first with a non-errored record.
            # Treat their record as canon.
            return existing
        # Replace the prior errored record in-place. Sorting at the
        # end keeps the file deterministic.
        final = [v for v in final if v.cluster_id != cluster_id]
    final.append(record)
    final.sort(key=lambda v: v.cluster_id)
    write_verifications(prior_path, final)
    return record


# ---------------------------------------------------------------------------


def verify_run(
    run_dir: Path,
    *,
    agent_endpoint: Optional[str] = None,
    agent_id: Optional[str] = None,
    model: str = "gpt-5.2",
    reasoning_effort: str = "medium",
    concurrency: int = 4,
    max_clusters: Optional[int] = None,
    include_person: bool = True,
    force: bool = False,
    error_types: Optional[frozenset[str]] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the verifier over a finished detection run.

    Programmatic entry point used both by the CLI in :func:`main` and
    by ``run_cleaning_detection.py`` when ``--verify-web`` is passed.
    Returns a dict with ``{counts, total, eligible, output_path}`` so
    callers can log a one-liner without re-parsing the output file.
    """
    clusters_path = run_dir / "clusters.json"
    summary_path = run_dir / "summary.json"
    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters.json missing: {clusters_path}")

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    summary: Mapping[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8")) or {}
    cfg_section = summary.get("config") or {}
    csv_paths = cfg_section.get("csv_paths") or []

    frames = _frame_loader(csv_paths)
    cell_lookup, column_samples, row_lookup = _build_lookups(frames)

    gate_cfg = GateConfig(verify_person=include_person)

    # Pre-filter by --error-types whitelist before applying the gate,
    # so a caller asking for e.g. "typo only" doesn't pay for any
    # other web-verifiable types in the same run.
    pool: List[Mapping[str, Any]] = list(clusters)
    if error_types:
        pool = [
            c for c in pool
            if isinstance(c, Mapping)
            and any(t in error_types for t in (c.get("error_types") or []))
        ]
        logger.info(
            "error_types filter %s: %d / %d clusters retained",
            sorted(error_types), len(pool), len(clusters),
        )

    # Gate.
    eligible: List[Mapping[str, Any]] = []
    for c in pool:
        if not isinstance(c, Mapping):
            continue
        shape = cluster_shape(c)
        cols_c = shape["columns"]
        col = cols_c[0] if cols_c else ""
        row = c.get("row")
        val = None
        if shape["pattern"] == "single_cell" and isinstance(row, int) and col:
            val = cell_lookup(c.get("source", ""), row, col)
        if should_verify(c, cell_value=val, cfg=gate_cfg):
            eligible.append(c)

    if max_clusters is not None:
        eligible = eligible[:max_clusters]

    logger.info(
        "gate: %d / %d clusters eligible for verification",
        len(eligible), len(clusters),
    )

    cache_dir = run_dir / "web_cache"
    if force and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    cache = WebCache(cache_dir)

    prior_path = run_dir / "web_verifications.json"
    prior = [] if force else load_verifications(prior_path)
    prior_by_id = {v.cluster_id: v for v in prior}
    todo = [c for c in eligible if c.get("cluster_id") not in prior_by_id]
    logger.info(
        "skipping %d clusters with prior verifications; %d to run",
        len(eligible) - len(todo), len(todo),
    )

    agent = AzureAgentBackend(
        endpoint=agent_endpoint or "",
        agent_id=agent_id or "",
        cache=cache,
    )

    vcfg = VerifierConfig(
        model=model,
        reasoning_effort=reasoning_effort,
        concurrency=max(1, concurrency),
        gate=gate_cfg,
        force=force,
    )
    verifier = Verifier(
        agent=agent,
        cell_lookup=cell_lookup,
        column_samples=column_samples,
        row_lookup=row_lookup,
        cfg=vcfg,
    )

    if progress is None:
        def progress(done: int, total: int) -> None:  # noqa: ARG001
            if done % 5 == 0 or done == total:
                logger.info("verifier progress: %d / %d", done, total)

    fresh = verifier.run(todo, progress=progress)
    merged = list(prior_by_id.values()) + fresh
    merged.sort(key=lambda v: v.cluster_id)

    write_verifications(prior_path, merged)
    counts = _summarise(merged, dry_run=False)
    return {
        "counts": counts,
        "total_clusters": len(clusters),
        "eligible": len(eligible),
        "verified_now": len(fresh),
        "output_path": prior_path,
    }


# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    run_dir: Path = args.outputs_dir / args.run_id
    if not run_dir.is_dir():
        logger.error("run dir not found: %s", run_dir)
        return 2

    clusters_path = run_dir / "clusters.json"
    if not clusters_path.exists():
        logger.error("clusters.json missing: %s", clusters_path)
        return 2

    if args.dry_run:
        # Dry-run path stays inline because it skips the LLM/web work
        # entirely and just prints the gate verdict.
        clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
        summary_path = run_dir / "summary.json"
        summary: Mapping[str, Any] = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8")) or {}
        csv_paths = (summary.get("config") or {}).get("csv_paths") or []
        frames = _frame_loader(csv_paths)
        cell_lookup, _, _ = _build_lookups(frames)
        gate_cfg = GateConfig()
        et_filter = _parse_error_types(args.error_types)
        pool = list(clusters)
        if et_filter:
            pool = [
                c for c in pool
                if isinstance(c, Mapping)
                and any(t in et_filter for t in (c.get("error_types") or []))
            ]
            logger.info(
                "error_types filter %s: %d / %d clusters retained",
                sorted(et_filter), len(pool), len(clusters),
            )
        eligible: List[Mapping[str, Any]] = []
        for c in pool:
            if not isinstance(c, Mapping):
                continue
            cols = c.get("columns") or []
            col = cols[0] if isinstance(cols, list) and cols else ""
            row = c.get("row")
            val = None
            if isinstance(row, int) and col:
                val = cell_lookup(c.get("source", ""), row, col)
            if should_verify(c, cell_value=val, cfg=gate_cfg):
                eligible.append(c)
        if args.max_clusters is not None:
            eligible = eligible[: args.max_clusters]
        logger.info(
            "gate: %d / %d clusters eligible for verification",
            len(eligible), len(clusters),
        )
        for c in eligible[:50]:
            kind = classify_cluster(c, cfg=gate_cfg).value
            cols = c.get("columns") or []
            col = cols[0] if isinstance(cols, list) and cols else ""
            print(
                f"{c.get('cluster_id'):40s}  {kind:7s}  "
                f"{c.get('source', '')[:30]:30s}  "
                f"row={c.get('row')!s:>4s}  col={col[:30]}"
            )
        if len(eligible) > 50:
            print(f"... and {len(eligible) - 50} more")
        return 0

    result = verify_run(
        run_dir,
        agent_endpoint=args.agent_endpoint,
        agent_id=args.agent_id,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        concurrency=args.concurrency,
        max_clusters=args.max_clusters,
        force=args.force,
        error_types=_parse_error_types(args.error_types),
    )
    out_path = result["output_path"]
    rel = (out_path.relative_to(PROJECT_ROOT)
           if out_path.is_relative_to(PROJECT_ROOT) else out_path)
    logger.info("verifier done: %s -> %s", rel, result["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
