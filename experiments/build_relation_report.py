"""Render the schema-discovery dashboard from a run directory.

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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


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
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing artifact: {summary_path}")
    if not relations_path.is_file():
        raise FileNotFoundError(f"missing artifact: {relations_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    relations = _read_jsonl(relations_path)
    gt_summary = _load_optional_json(run_dir / "gt_summary.json")

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

    return {
        "run_metadata": run_metadata,
        "summary": summary,
        "relations": relations,
        "sources": sources,
        "sources_detail": sources_payload,
        "gt_summary": gt_summary,
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
        description="Render report.html from a saved schema-discovery run.",
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
