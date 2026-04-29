"""Per-cluster web-grounded verifier.

Pulls eligible clusters through a 3-step loop:

1. **Build queries** from the suspect value (verbatim) and the
   detector's suggested fix (if any), biased by entity kind.
2. **Run web search** (cached) and dedupe results by URL.
3. **One LLM call** asks the model to emit a structured verdict using
   only the retrieved snippets + sibling values from the same column.

The output is a list of :class:`Verification` records ready to dump
to JSON. We never overwrite a verification record from a previous
run unless the operator explicitly asks for it (the CLI flag
``--force`` clears the cache).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

from src.core.llm_client import LLMClient, LLMResponse

from .gate import EntityKind, GateConfig, classify_cluster, should_verify
from .web_tool import SearchHit, SearchResult, WebSearchTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class EvidenceCitation:
    url: str
    title: str
    quote: str


@dataclass
class Verification:
    """One verifier output row, persisted as JSON."""

    cluster_id: str
    source: str
    row: Optional[int]
    column: str
    suspect_value: str
    entity_kind: str
    verdict: str = "ambiguous"
    canonical_value: Optional[str] = None
    confidence: float = 0.0
    evidence: List[EvidenceCitation] = field(default_factory=list)
    reasoning: str = ""
    queries: List[str] = field(default_factory=list)
    n_results: int = 0
    n_unique_urls: int = 0
    error: Optional[str] = None
    parse_error: Optional[str] = None
    llm_attempts: int = 0
    llm_endpoint: str = ""
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict produces nested dicts for evidence already.
        return d


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VerifierConfig:
    """Tunables for the verifier loop."""

    model: str = "gpt-5.2"
    reasoning_effort: str = "medium"
    reasoning_summary: str = "concise"
    concurrency: int = 4
    # Per-cluster search budget. Two queries (raw value + suggested
    # fix) is the sweet spot - more rarely helps and burns rate limit.
    max_queries_per_cluster: int = 2
    # Per-query result cap.
    max_results_per_query: int = 5
    # Per-cluster cap on snippets fed to the LLM (deduped by URL).
    max_snippets: int = 6
    # Hard cap on snippet length (chars) before we truncate.
    snippet_chars: int = 600
    # Top-N distinct sibling values to include for in-corpus context.
    sibling_samples: int = 8
    # Path to the YAML prompt file.
    prompt_path: Path = Path("prompts/verify_entity.yaml")
    # Gate config (filters which clusters to run).
    gate: GateConfig = field(default_factory=GateConfig)
    # When True, even cached verifications get recomputed.
    force: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_object(text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract a JSON object from possibly-fenced LLM output."""
    if not text:
        return None, "empty response"
    candidates: List[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text.strip())
    m = _OBJECT_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        try:
            payload = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, None
    return None, "failed to parse a JSON object"


def _query_for_kind(value: str, kind: str) -> str:
    """Bias the search query to the entity kind."""
    base = f'"{value}"'
    if kind == EntityKind.ORG.value:
        return f"{base} official site"
    if kind == EntityKind.ADDRESS.value:
        return f"{base} address postal"
    if kind == EntityKind.PHONE.value:
        return f"{base} phone"
    if kind == EntityKind.URL.value:
        return value  # don't quote URLs
    return base


def _build_queries(
    suspect_value: str,
    suggested_fix: Optional[str],
    kind: str,
    cap: int,
) -> List[str]:
    queries: List[str] = []

    def _push(q: str) -> None:
        q = (q or "").strip()
        if not q or q in queries or len(queries) >= cap:
            return
        queries.append(q)

    _push(_query_for_kind(suspect_value, kind))
    if suggested_fix and suggested_fix.strip() and suggested_fix.strip() != suspect_value.strip():
        _push(_query_for_kind(suggested_fix, kind))
    return queries


def _format_sibling_values(values: Sequence[str]) -> str:
    if not values:
        return "  (none)"
    return "\n".join(f"  - {v}" for v in values)


def _format_row_context(
    row: Mapping[str, Any],
    *,
    suspect_column: str,
    cap_value_chars: int = 120,
    max_columns: int = 24,
) -> str:
    """Render the full row as ``column: value`` lines, with the
    flagged column marked. Long values are truncated; the suspect
    column always appears first regardless of column order.

    Returns ``"  (unavailable)"`` when ``row`` is empty so the prompt
    template still has a deterministic block.
    """
    if not row:
        return "  (unavailable)"

    def _fmt(col: str, val: Any) -> str:
        s = "" if val is None else str(val)
        s = s.replace("\n", " ").replace("\r", " ").strip()
        if len(s) > cap_value_chars:
            s = s[:cap_value_chars].rstrip() + "\u2026"
        marker = " *" if col == suspect_column else ""
        return f"  - {col}{marker}: {s}"

    items = [(k, v) for k, v in row.items()]
    # Move the suspect column to the top so the LLM sees it first.
    head = [(k, v) for k, v in items if k == suspect_column]
    tail = [(k, v) for k, v in items if k != suspect_column]
    ordered = head + tail
    rendered = [_fmt(k, v) for k, v in ordered[:max_columns]]
    if len(ordered) > max_columns:
        rendered.append(f"  - ... and {len(ordered) - max_columns} more columns")
    return "\n".join(rendered)


def _format_evidence(snippets: Sequence[SearchHit], cap_chars: int) -> str:
    if not snippets:
        return "  (no web results)"
    lines: List[str] = []
    for i, h in enumerate(snippets, 1):
        title = (h.title or "").strip() or "(no title)"
        body = (h.snippet or "").strip()
        if len(body) > cap_chars:
            body = body[:cap_chars].rstrip() + "..."
        lines.append(f"[{i}] {title} - {h.url}\n    {body}")
    return "\n\n".join(lines)


def _dedup_hits(hits: Sequence[SearchHit], limit: int) -> List[SearchHit]:
    seen: set[str] = set()
    out: List[SearchHit] = []
    for h in hits:
        url = (h.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(h)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


@dataclass
class Verifier:
    """Run :class:`Verification` for each gate-eligible cluster.

    Inputs are kept narrow on purpose: the verifier never opens
    DataFrames itself - the caller passes a ``cell_lookup`` callback
    that resolves ``(source, row, column) -> str`` and a
    ``column_samples`` callback that returns up to N distinct sibling
    values for a column.
    """

    llm: LLMClient
    web: WebSearchTool
    cell_lookup: Callable[[str, int, str], Optional[str]]
    column_samples: Callable[[str, str, int], List[str]]
    # Optional: full-row context (column -> value). Defaults to a
    # no-op so existing tests / callers that don't supply it still
    # work; when provided, the row is included in the LLM prompt.
    row_lookup: Optional[Callable[[str, int], Mapping[str, Any]]] = None
    cfg: VerifierConfig = field(default_factory=VerifierConfig)
    _prompt: Mapping[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._prompt = self._load_prompt(self.cfg.prompt_path)

    # ---- prompt loading ------------------------------------------

    @staticmethod
    def _load_prompt(path: Path) -> Mapping[str, str]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"verify prompt not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: top level must be a mapping")
        system = str(raw.get("system", "")).strip()
        template = str(raw.get("user_template", ""))
        if not system or not template:
            raise ValueError(f"{path}: 'system' and 'user_template' required")
        return {"system": system, "user_template": template}

    # ---- single-cluster pipeline ---------------------------------

    def _verify_one(self, cluster: Mapping[str, Any]) -> Verification:
        cid = str(cluster.get("cluster_id") or "")
        source = str(cluster.get("source") or "")
        row = cluster.get("row")
        cols = cluster.get("columns") or []
        column = str(cols[0]) if isinstance(cols, list) and cols else ""
        kind = classify_cluster(cluster, cfg=self.cfg.gate)

        # Resolve cell value via callback. We tolerate missing values:
        # the gate already vetted eligibility, but if we can't load the
        # cell now (e.g. CSV moved), record the error and bail.
        suspect_value = ""
        try:
            if isinstance(row, int) and column:
                suspect_value = self.cell_lookup(source, row, column) or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("cell_lookup failed for %s: %s", cid, exc)

        ver = Verification(
            cluster_id=cid,
            source=source,
            row=row if isinstance(row, int) else None,
            column=column,
            suspect_value=str(suspect_value),
            entity_kind=kind.value,
        )

        if not suspect_value:
            ver.error = "suspect cell value unavailable"
            return ver

        # Pull suggested_fix out of the synthesis block if present.
        syn = cluster.get("synthesis") or {}
        if not isinstance(syn, Mapping):
            syn = {}
        suggested_fix = (
            (syn.get("canonical_fix") or "")
            or (cluster.get("suggested_fix") or "")
        )

        queries = _build_queries(
            suspect_value, suggested_fix or None, kind.value,
            cap=self.cfg.max_queries_per_cluster,
        )
        ver.queries = queries

        # Search every query, merge.
        all_hits: List[SearchHit] = []
        n_results = 0
        for q in queries:
            res: SearchResult = self.web.search(
                q, max_results=self.cfg.max_results_per_query,
                include_raw_content=False,
            )
            if res.error:
                logger.warning(
                    "search error for cluster=%s query=%r: %s",
                    cid, q, res.error,
                )
                continue
            n_results += len(res.hits)
            all_hits.extend(res.hits)
        snippets = _dedup_hits(all_hits, self.cfg.max_snippets)
        ver.n_results = n_results
        ver.n_unique_urls = len(snippets)

        # Sibling values for in-corpus context.
        try:
            siblings = self.column_samples(
                source, column, self.cfg.sibling_samples,
            ) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("column_samples failed for %s: %s", cid, exc)
            siblings = []
        # Drop the suspect value itself and dedupe.
        siblings = [
            s for s in (str(s).strip() for s in siblings)
            if s and s != suspect_value.strip()
        ]
        # Stable dedupe.
        seen_sib: set[str] = set()
        siblings = [
            s for s in siblings
            if not (s in seen_sib or seen_sib.add(s))  # type: ignore[func-returns-value]
        ][: self.cfg.sibling_samples]

        # Full-row context (column -> value mapping). Provides the LLM
        # the same fields a human reviewer would see when judging the
        # cell - e.g. "if District=Marin County Office of Education then
        # County=Marin is consistent". Skipped silently when no
        # row_lookup callback is supplied (back-compat).
        row_dict: Mapping[str, Any] = {}
        if self.row_lookup is not None and isinstance(row, int):
            try:
                row_dict = self.row_lookup(source, row) or {}
            except Exception as exc:  # noqa: BLE001
                logger.warning("row_lookup failed for %s: %s", cid, exc)
                row_dict = {}
        row_context_str = (
            _format_row_context(row_dict, suspect_column=column)
            if row_dict else "  (unavailable)"
        )

        prompt = self._prompt["user_template"].format(
            source=source,
            column=column or "(unknown)",
            row=row if isinstance(row, int) else "n/a",
            entity_kind=kind.value,
            error_types=", ".join(
                str(t) for t in (cluster.get("error_types") or [])
            ) or "(none)",
            suspect_value=suspect_value,
            suggested_fix=suggested_fix or "(none)",
            sibling_values=_format_sibling_values(siblings),
            row_context=row_context_str,
            evidence_block=_format_evidence(snippets, self.cfg.snippet_chars),
        )

        resp: LLMResponse = self.llm.query(
            self.cfg.model,
            prompt,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            system_prompt=self._prompt["system"],
        )
        ver.llm_attempts = resp.attempts
        ver.llm_endpoint = resp.endpoint
        if not resp.ok:
            ver.error = resp.error or "llm returned no text"
            return ver

        payload, parse_err = _parse_json_object(resp.text)
        ver.parse_error = parse_err
        if not payload:
            return ver

        verdict = str(payload.get("verdict") or "").strip().lower()
        if verdict not in ("confirmed_error", "false_positive", "ambiguous"):
            ver.parse_error = f"unknown verdict: {verdict!r}"
            return ver
        ver.verdict = verdict

        canon = payload.get("canonical_value")
        if canon in (None, "", "null", "None"):
            ver.canonical_value = None
        else:
            ver.canonical_value = str(canon).strip() or None

        try:
            ver.confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            ver.confidence = 0.0
        ver.confidence = max(0.0, min(1.0, ver.confidence))
        ver.reasoning = str(payload.get("reasoning") or "").strip()

        snippet_urls = {h.url for h in snippets if h.url}
        for ev in payload.get("evidence") or []:
            if not isinstance(ev, Mapping):
                continue
            url = str(ev.get("url") or "").strip()
            if not url:
                continue
            # Anti-hallucination: only keep cites whose URL we actually
            # gave the model. This is the cheapest filter that catches
            # the vast majority of fabricated citations.
            if url not in snippet_urls:
                continue
            ver.evidence.append(EvidenceCitation(
                url=url,
                title=str(ev.get("title") or "").strip(),
                quote=str(ev.get("quote") or "").strip()[:600],
            ))

        # Tighten the verdict if it claims confirmed_error without
        # surviving evidence: downgrade to ambiguous.
        if ver.verdict == "confirmed_error" and not ver.evidence:
            ver.verdict = "ambiguous"
            ver.canonical_value = None
            ver.parse_error = (ver.parse_error or "") + (
                "; downgraded to ambiguous: no evidence URLs survived"
            ).lstrip("; ")

        return ver

    # ---- batch -----------------------------------------------------

    def run(
        self,
        clusters: Sequence[Mapping[str, Any]],
        *,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Verification]:
        """Run the verifier over ``clusters``. Eligibility filtering
        is the caller's job (use :func:`src.verifier.gate.filter_eligible`).
        """
        clusters = list(clusters)
        total = len(clusters)
        out: List[Verification] = []
        if total == 0:
            return out

        lock = threading.Lock()
        done = {"n": 0}

        def _job(c: Mapping[str, Any]) -> Verification:
            v = self._verify_one(c)
            with lock:
                done["n"] += 1
                if progress is not None:
                    progress(done["n"], total)
            return v

        if self.cfg.concurrency <= 1:
            for c in clusters:
                out.append(_job(c))
        else:
            with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
                futures = [pool.submit(_job, c) for c in clusters]
                for fut in as_completed(futures):
                    out.append(fut.result())

        # Stable order by cluster_id so re-runs diff cleanly.
        out.sort(key=lambda v: v.cluster_id)
        return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_verifications(path: Path, records: Sequence[Verification]) -> None:
    """Write ``records`` as a single JSON array under ``path``.

    Uses an atomic ``os.replace`` so partial files never appear if the
    process is killed mid-write.
    """
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [v.to_dict() for v in records]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_verifications(path: Path) -> List[Verification]:
    """Read a previous run's verifications, or ``[]`` if missing."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: List[Verification] = []
    for r in raw:
        if not isinstance(r, Mapping):
            continue
        evidence = [
            EvidenceCitation(
                url=str(e.get("url") or ""),
                title=str(e.get("title") or ""),
                quote=str(e.get("quote") or ""),
            )
            for e in (r.get("evidence") or [])
            if isinstance(e, Mapping)
        ]
        out.append(Verification(
            cluster_id=str(r.get("cluster_id") or ""),
            source=str(r.get("source") or ""),
            row=r.get("row") if isinstance(r.get("row"), int) else None,
            column=str(r.get("column") or ""),
            suspect_value=str(r.get("suspect_value") or ""),
            entity_kind=str(r.get("entity_kind") or "other"),
            verdict=str(r.get("verdict") or "ambiguous"),
            canonical_value=r.get("canonical_value") if r.get("canonical_value") not in ("", None) else None,
            confidence=float(r.get("confidence") or 0.0),
            evidence=evidence,
            reasoning=str(r.get("reasoning") or ""),
            queries=[str(q) for q in (r.get("queries") or [])],
            n_results=int(r.get("n_results") or 0),
            n_unique_urls=int(r.get("n_unique_urls") or 0),
            error=r.get("error") or None,
            parse_error=r.get("parse_error") or None,
            llm_attempts=int(r.get("llm_attempts") or 0),
            llm_endpoint=str(r.get("llm_endpoint") or ""),
            ts=str(r.get("ts") or ""),
        ))
    return out


__all__ = [
    "EvidenceCitation",
    "Verification",
    "VerifierConfig",
    "Verifier",
    "write_verifications",
    "load_verifications",
]
