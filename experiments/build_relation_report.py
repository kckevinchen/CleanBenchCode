"""Render the relation-discovery dashboard from a run directory.

A discovery run (``experiments.run_relation_discovery``) writes
``relations.jsonl``, ``discovery_reports.json``, ``summary.json``, and
- for adaptive runs - ``gt_summary.json`` into a timestamped output
directory. This script bundles those artifacts into the JSON payload
expected by ``frontend/relations_template.html`` and writes
``report.html`` next to them.

Usage::

    python -m experiments.build_relation_report \\
        outputs/20260430T002157Z_gpt-5.2_relations

    # different template / output
    python -m experiments.build_relation_report <run_dir> \\
        --template frontend/relations_template.html \\
        --output /tmp/relations.html

The renderer is offline and never re-invokes the LLM, so dashboards
can be regenerated against a refreshed template without rerunning
discovery.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("build_relation_report")

DATA_PLACEHOLDER = "/*__RELATION_DATA_JSON__*/"


from src.core.jsonl_io import read_jsonl as _read_jsonl  # noqa: E402


def _load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("could not parse %s; ignoring", path)
        return None


def _build_payload(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    relations_path = run_dir / "relations.jsonl"
    multi_groups_path = run_dir / "multi_table_groups.jsonl"
    multi_violations_path = run_dir / "multi_table_violations.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing artifact: {summary_path}")
    # ``relations.jsonl`` is the legacy single-table artifact; a
    # multi-table-only run (--error-types inter_table_inconsistency)
    # produces no relations but does write multi_table_groups.jsonl
    # and multi_table_violations.jsonl. Tolerate either being absent
    # so long as at least one is present.
    has_relations = relations_path.is_file()
    has_multi = multi_groups_path.is_file() or multi_violations_path.is_file()
    if not (has_relations or has_multi):
        raise FileNotFoundError(f"missing artifact: {relations_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    relations = _read_jsonl(relations_path) if has_relations else []
    gt_summary = _load_optional_json(run_dir / "gt_summary.json")

    # Multi-table cross-sheet pass artifacts (optional). Each line of
    # multi_table_groups.jsonl is a ResolvedWorkbook record from
    # ``src.core.multi_table_resolver``: workbook name, candidate
    # groups (union/join), and the LLM's confirm/reject decision +
    # column mapping + primary key. multi_table_violations.jsonl is
    # the flat list of MultiTableViolation rows produced by the
    # deterministic compiler in ``src.core.multi_table_violations``.
    # multi_table_verifications.jsonl is one Verification record per
    # violation_id from the step-4 verifier in
    # ``src.core.multi_table_verifier``.
    multi_table_groups: List[Dict[str, Any]] = (
        _read_jsonl(multi_groups_path) if multi_groups_path.is_file() else []
    )
    multi_table_violations: List[Dict[str, Any]] = (
        _read_jsonl(multi_violations_path)
        if multi_violations_path.is_file() else []
    )
    multi_verifications_path = run_dir / "multi_table_verifications.jsonl"
    multi_table_verifications: List[Dict[str, Any]] = (
        _read_jsonl(multi_verifications_path)
        if multi_verifications_path.is_file() else []
    )

    # Generalization output (optional). When the runner ran the
    # parametric-rewrite pass, ``relations_generalized.jsonl`` holds
    # one parametric rule per collapsed family with a
    # ``generalized_from`` list pointing back to the originals'
    # species_keys. The report uses this to surface the parametric
    # rule as the primary card and tuck the originals under expand,
    # without ever discarding the LLM's raw output.
    generalized_path = run_dir / "relations_generalized.jsonl"
    generalized: List[Dict[str, Any]] = (
        _read_jsonl(generalized_path) if generalized_path.is_file() else []
    )

    # Aggregation output (optional). When the runner ran the
    # paraphrase-merging pass, ``relations_aggregated.jsonl`` holds
    # one canonical rule per collapsed paraphrase cluster, with an
    # ``aggregated_from`` list pointing back to the originals'
    # species_keys. The report stacks aggregation BEFORE
    # generalization in the UI: an aggregated canonical rule may
    # itself be the originals-side of a generalized parametric rule.
    aggregated_path = run_dir / "relations_aggregated.jsonl"
    aggregated: List[Dict[str, Any]] = (
        _read_jsonl(aggregated_path) if aggregated_path.is_file() else []
    )

    # Per-source context: the runner persists a sources.json whose
    # entries are TableDigest.to_dict() values (shape, headers,
    # per-column dtype/distinct/top values, candidate FDs,
    # identity-like columns, total-row indices). The report template
    # uses this to render a context strip for each CSV so users can
    # tell what they're looking at without leaving the dashboard.
    sources_payload: List[Dict[str, Any]] = []
    sources_path = run_dir / "sources.json"
    if sources_path.is_file():
        try:
            loaded = json.loads(sources_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                sources_payload = [s for s in loaded if isinstance(s, dict)]
        except json.JSONDecodeError:
            logger.warning("could not parse %s; ignoring", sources_path)

    # Fallback name lists when sources.json is absent (older runs).
    sources_in_payload = {
        s.get("source") for s in sources_payload if isinstance(s.get("source"), str)
    }
    extra_names: List[str] = []
    reports_path = run_dir / "discovery_reports.json"
    if reports_path.is_file():
        try:
            reports = json.loads(reports_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reports = []
        seen: Dict[str, None] = {}
        for rep in reports if isinstance(reports, list) else []:
            src = rep.get("source")
            if isinstance(src, str) and src not in sources_in_payload:
                seen.setdefault(src, None)
        extra_names.extend(seen.keys())
    for r in relations:
        s = r.get("source")
        if (
            isinstance(s, str)
            and s not in sources_in_payload
            and s not in extra_names
        ):
            extra_names.append(s)
    for name in extra_names:
        # Stub entries so the source still appears in the tabs.
        sources_payload.append({"source": name})
    sources_payload.sort(key=lambda d: d.get("source") or "")

    # Backwards compat: keep the bare-name list under ``sources`` so
    # older renderers don't crash; the new richer view reads from
    # ``sources_detail``.
    sources = [
        s["source"] for s in sources_payload
        if isinstance(s.get("source"), str)
    ]

    run_metadata = {
        "run_id": run_dir.name,
        "model": summary.get("model"),
        "generated_at": run_dir.name.split("_", 1)[0] if "_" in run_dir.name else "",
    }

    # Empirical-validation output (optional). When
    # ``experiments.run_relation_validation`` has been run on this
    # directory, ``relation_verifications.jsonl`` holds one record
    # per relation with ``species_key``, ``verifiable``, ``pass_rate``,
    # ``code``, and a few sample violations. Indexing by species_key
    # at build time lets the report renderer attach the badge to the
    # right card without scanning the list per render.
    verifications_path = run_dir / "relation_verifications.jsonl"
    verifications: Dict[str, Dict[str, Any]] = {}
    if verifications_path.is_file():
        for vrec in _read_jsonl(verifications_path):
            key = vrec.get("species_key")
            if isinstance(key, str) and key:
                verifications[key] = vrec

    return {
        "run_metadata": run_metadata,
        "summary": summary,
        "relations": relations,
        "sources": sources,
        "sources_detail": sources_payload,
        "gt_summary": gt_summary,
        "generalized": generalized,
        "aggregated": aggregated,
        "verifications": verifications,
        "multi_table_groups": multi_table_groups,
        "multi_table_violations": multi_table_violations,
        "multi_table_verifications": multi_table_verifications,
    }


def _embed_payload(template: str, payload: Dict[str, Any]) -> str:
    if DATA_PLACEHOLDER not in template:
        raise ValueError(
            "template does not contain placeholder " + DATA_PLACEHOLDER
        )
    # ``</script>`` inside the JSON would break the surrounding script
    # element; escape forward-slashes following ``<`` to be safe.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("</", "<\\/")
    return template.replace(DATA_PLACEHOLDER, encoded)


def build_relation_report(
    run_dir: Path,
    *,
    template_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    template = template_path.read_text(encoding="utf-8")
    payload = _build_payload(run_dir)
    html = _embed_payload(template, payload)
    out = output_path or (run_dir / "report.html")
    out.write_text(html, encoding="utf-8")
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render report.html from a saved relation-discovery run.",
    )
    p.add_argument(
        "run_dir", type=Path,
        help="Run directory produced by experiments.run_relation_discovery.",
    )
    p.add_argument(
        "--template", type=Path,
        default=PROJECT_ROOT / "frontend" / "relations_template.html",
        help="Path to the discovery dashboard template.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the rendered HTML. Defaults to <run_dir>/report.html.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        logger.error("run directory not found: %s", run_dir)
        return 2
    if not args.template.is_file():
        logger.error("template not found: %s", args.template)
        return 2
    out = build_relation_report(
        run_dir,
        template_path=args.template,
        output_path=args.output,
    )
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
