"""Per-cluster web-grounded verifier.

Runs every gate-eligible cluster through a single Foundry Agents call:

1. Build the user prompt from the suspect cell, sibling values, and
   full row context.
2. Hand it to an :class:`AgentBackend` (production: the Web-Search-
   equipped Foundry agent; tests: :class:`StubAgentBackend`).
3. Parse the agent's JSON reply into a structured :class:`Verification`,
   filtering out citation URLs the runtime did not actually record as
   Web Search sources (anti-hallucination).

The agent owns query formulation and result fetching; we never see raw
snippet text. We DO see the URL citations the agent emitted via
``url_citation`` annotations, plus the Bing search query URLs from the
run steps - both of which feed into the labeling UI.

The output is a list of :class:`Verification` records ready to dump
to JSON. We never overwrite a verification record from a previous
run unless the operator explicitly asks for it (the CLI flag
``--force`` clears the cache).
"""

from __future__ import annotations

import hashlib
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

from .agent_backend import AgentBackend, AgentVerdict, Citation
from .gate import EntityKind, GateConfig, classify_cluster, cluster_shape

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
    # Bing search query URLs (api host swapped for www.bing.com per
    # display requirements) the agent fired during this run.
    queries: List[str] = field(default_factory=list)
    # Total Web Search tool calls the agent made for this cluster.
    n_results: int = 0
    # Unique citation URLs the agent recorded in url_citation annotations.
    n_unique_urls: int = 0
    error: Optional[str] = None
    parse_error: Optional[str] = None
    # Carried over from the LLM era: how many backend calls + which
    # endpoint produced this verdict. We keep both for backwards-
    # compat with on-disk records and the labeling UI.
    llm_attempts: int = 0
    llm_endpoint: str = ""
    # ---- Multi-row / multi-column extensions (additive) -----------
    # Spatial layout the verifier processed for this cluster.
    # ``pattern`` is one of ``"single_cell"`` | ``"multi_row"`` |
    # ``"multi_column"`` | ``"composite"``. ``rows`` / ``columns_all``
    # are the full row + column lists the cluster spans (``row`` and
    # ``column`` remain the primary anchors for the UI).
    pattern: str = "single_cell"
    rows: List[int] = field(default_factory=list)
    columns_all: List[str] = field(default_factory=list)
    # One projected snapshot per record actually shown to the agent
    # (column -> str value). For single_cell this is the single
    # suspect cell wrapped in a 1-element list; for multi_row it
    # holds one entry per affected row, projected to the cluster's
    # columns when set or to the full row otherwise.
    record_snapshots: List[Dict[str, str]] = field(default_factory=list)
    # Optional structured fix returned by the agent for multi-column
    # patterns: ``{column_name: corrected_value}``. ``canonical_value``
    # remains the human-readable summary; this field is consumed by
    # the labeling UI when applying a multi-column correction.
    canonical_record: Optional[Dict[str, str]] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VerifierConfig:
    """Tunables for the verifier loop."""

    # Carried for output-schema compatibility (``Verification.llm_endpoint``);
    # the actual model is configured on the Foundry agent in the portal.
    model: str = "gpt-5.2"
    reasoning_effort: str = "medium"
    reasoning_summary: str = "concise"
    concurrency: int = 4
    # Per-cluster cap on snippets / citations we surface to the LLM.
    # The agent decides on its own how many searches to run; this is
    # only a hard cap on what we copy back into the Verification record.
    max_snippets: int = 6
    # Hard cap on snippet length (chars) before we truncate.
    snippet_chars: int = 600
    # Top-N distinct sibling values to include for in-corpus context.
    sibling_samples: int = 8
    # Path to the YAML prompt file. Resolved against the project root
    # (``<repo>/prompts/verify_entity.yaml``) by default so the
    # verifier doesn't depend on the caller's CWD.
    prompt_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "prompts" / "verify_entity.yaml"
    )
    # Gate config (filters which clusters to run).
    gate: GateConfig = field(default_factory=GateConfig)
    # When True, even cached verifications get recomputed (caller
    # clears the on-disk cache; the verifier itself only honours this
    # flag for diagnostics).
    force: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_object(text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract a JSON object from possibly-fenced agent output."""
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


def _format_sibling_values(values: Sequence[str]) -> str:
    if not values:
        return "  (none)"
    return "\n".join(f"  - {v}" for v in values)


def _cache_key(cluster_id: str, prompt: str) -> str:
    """Stable per-cluster cache key. Includes a hash of the prompt so
    a prompt-template change invalidates the cache automatically.
    """
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{cluster_id}::{h}"


def _format_records_under_review(
    records: Sequence[Mapping[str, str]],
    rows: Sequence[int],
    cluster_columns: Sequence[str],
    *,
    suspect_column: str = "",
    cap_value_chars: int = 120,
    max_columns_per_record: int = 24,
) -> str:
    """Render the ``RECORDS UNDER REVIEW`` block of the prompt.

    Each record is one row's projected snapshot. Columns that are
    part of the cluster (i.e. flagged for review) are marked with
    ``" *"`` so the agent knows which fields the detector
    suspects. For single-cell clusters ``cluster_columns`` is empty
    and ``suspect_column`` carries the one flagged column.
    """
    if not records:
        return "  (no records available)"

    flagged: set[str] = {c for c in cluster_columns if c}
    if suspect_column:
        flagged.add(suspect_column)

    def _fmt_value(val: Any) -> str:
        s = "" if val is None else str(val)
        s = s.replace("\n", " ").replace("\r", " ").strip()
        if len(s) > cap_value_chars:
            s = s[: cap_value_chars].rstrip() + "\u2026"
        return s

    lines: List[str] = []
    for idx, rec in enumerate(records):
        row_idx = rows[idx] if idx < len(rows) else None
        header = (
            f"  Record {idx + 1} (row {row_idx}):"
            if isinstance(row_idx, int)
            else f"  Record {idx + 1}:"
        )
        lines.append(header)
        # Order: flagged columns first (preserving cluster_columns
        # order), then other columns in insertion order. This lets
        # the agent see the columns under review at the top of each
        # record block.
        items = list(rec.items())
        head = [(k, v) for k, v in items if k in flagged]
        # Preserve cluster_columns order for the head section.
        if cluster_columns:
            order = {c: i for i, c in enumerate(cluster_columns)}
            head.sort(key=lambda kv: order.get(kv[0], 1_000_000))
        tail = [(k, v) for k, v in items if k not in flagged]
        ordered = head + tail
        rendered = 0
        for col, val in ordered:
            mark = " *" if col in flagged else ""
            lines.append(f"    - {col}{mark}: {_fmt_value(val)}")
            rendered += 1
            if rendered >= max_columns_per_record:
                remaining = len(ordered) - rendered
                if remaining > 0:
                    lines.append(
                        f"    - ... and {remaining} more columns"
                    )
                break
    return "\n".join(lines)


def _format_task_block(pattern: str, columns: Sequence[str], n_rows: int) -> str:
    """Pattern-specific TASK instructions slotted into the prompt.

    The verifier composes this in code (rather than in the YAML)
    because each pattern asks the agent a fundamentally different
    question about the records under review.
    """
    cols = [c for c in columns if c]
    cols_str = ", ".join(cols) if cols else ""

    if pattern == "single_cell":
        return (
            "Use the Web Search tool to find authoritative evidence about "
            "the suspect cell value (and, if useful, the suggested fix). "
            "Decide which verdict best fits using ONLY the evidence the "
            "tool surfaced.\n"
            "  * \"confirmed_error\" - the suspect value is wrong and a "
            "single canonical alternative is supported by at least one "
            "quoted snippet. Provide it as ``canonical_value``.\n"
            "  * \"false_positive\" - the suspect value is in fact "
            "correct. Set ``canonical_value`` to null.\n"
            "  * \"ambiguous\" - evidence is missing or contradictory. "
            "Set ``canonical_value`` to null."
        )
    if pattern == "multi_row":
        return (
            f"Use the Web Search tool to determine whether the {n_rows} "
            "records under review describe the SAME real-world entity. "
            "The detector suspects they may be duplicate records of one "
            "entity that has been written down in different ways.\n"
            "  * \"confirmed_error\" - the records ARE the same entity "
            "and at least one record has a wrong / non-canonical "
            "representation. Set ``canonical_value`` to the canonical "
            "name (or a short description like \"merge as <Name>\"). "
            "Cite at least one URL whose page text contains the "
            "canonical name verbatim.\n"
            "  * \"false_positive\" - the records refer to DISTINCT "
            "real-world entities (i.e. they are NOT duplicates). Set "
            "``canonical_value`` to null and explain in ``reasoning`` "
            "which differentiating fact the web confirmed.\n"
            "  * \"ambiguous\" - the web does not have enough evidence "
            "to decide."
        )
    if pattern == "multi_column":
        return (
            "Use the Web Search tool to verify whether the columns "
            f"under review ({cols_str}) hold values that are mutually "
            "consistent for THIS record - i.e. they all describe the "
            "same real-world entity according to authoritative sources.\n"
            "  * \"confirmed_error\" - at least one column's value is "
            "inconsistent with the others according to the web. Set "
            "``canonical_value`` to a short human-readable summary of "
            "the corrected record AND populate ``canonical_record`` "
            "with ``{col: corrected_value}`` for every column you are "
            "correcting (omit columns whose value is already correct).\n"
            "  * \"false_positive\" - the columns are mutually "
            "consistent. Set ``canonical_value`` and ``canonical_record`` "
            "to null.\n"
            "  * \"ambiguous\" - the web does not adjudicate the "
            "relationship clearly."
        )
    # composite (multi-row + multi-column)
    return (
        f"Use the Web Search tool to verify whether the {n_rows} "
        f"records share consistent values across the columns "
        f"({cols_str}). The detector suspects either a functional-"
        "dependency violation (different rows disagree on what the "
        "value of a column should be for a shared key) or a duplicate "
        "group with inconsistent attributes.\n"
        "  * \"confirmed_error\" - the records refer to the same "
        "entity / key but disagree on at least one column's value, "
        "and the web supports a single canonical value. Use "
        "``canonical_record`` to map ``{col: corrected_value}``.\n"
        "  * \"false_positive\" - the records are mutually consistent "
        "or refer to distinct entities. Set ``canonical_value`` to null.\n"
        "  * \"ambiguous\" - insufficient web evidence."
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


@dataclass
class Verifier:
    """Run :class:`Verification` for each gate-eligible cluster.

    The verifier is intentionally narrow: it never opens DataFrames
    itself - the caller passes a ``cell_lookup`` callback that
    resolves ``(source, row, column) -> str`` and a
    ``column_samples`` callback that returns up to N distinct sibling
    values for a column. ``row_lookup`` is optional and provides the
    full row dictionary for cross-column context.
    """

    agent: AgentBackend
    cell_lookup: Callable[[str, int, str], Optional[str]]
    column_samples: Callable[[str, str, int], List[str]]
    row_lookup: Optional[Callable[[str, int], Mapping[str, Any]]] = None
    cfg: VerifierConfig = field(default_factory=VerifierConfig)
    _prompt: Mapping[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._prompt = self._load_prompt(self.cfg.prompt_path)
        # Plumb the system prompt onto the backend so it doesn't have
        # to be re-passed per call. The backend uses it as the run's
        # ``instructions`` override.
        if hasattr(self.agent, "system_prompt"):
            try:
                setattr(self.agent, "system_prompt", self._prompt["system"])
            except Exception:  # pragma: no cover (immutable backend)
                pass

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

        shape = cluster_shape(cluster)
        pattern = shape["pattern"]
        all_rows: List[int] = list(shape["rows"])
        all_cols: List[str] = list(shape["columns"])

        # Apply caps from the gate config so we never blast a
        # 200-row dup group into one prompt.
        capped_rows = all_rows[: max(1, self.cfg.gate.multi_row_cap)] if all_rows else []
        capped_cols = all_cols[: max(1, self.cfg.gate.multi_col_cap)] if all_cols else []

        primary_row = capped_rows[0] if capped_rows else (
            cluster.get("row") if isinstance(cluster.get("row"), int) else None
        )
        primary_col = capped_cols[0] if capped_cols else ""

        kind = classify_cluster(cluster, cfg=self.cfg.gate)

        # ---- build per-record snapshots --------------------------
        records: List[Dict[str, str]] = []
        if pattern == "single_cell":
            # Existing behavior: a single cell, with the full row
            # populating the snapshot for cross-column hints. The
            # agent sees the row marked with ``*`` on the suspect
            # column.
            cell_val = ""
            try:
                if isinstance(primary_row, int) and primary_col:
                    cell_val = self.cell_lookup(source, primary_row, primary_col) or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("cell_lookup failed for %s: %s", cid, exc)
            row_dict: Mapping[str, Any] = {}
            if (
                self.row_lookup is not None
                and isinstance(primary_row, int)
            ):
                try:
                    row_dict = self.row_lookup(source, primary_row) or {}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("row_lookup failed for %s: %s", cid, exc)
                    row_dict = {}
            if row_dict:
                snap = {str(k): ("" if v is None else str(v)) for k, v in row_dict.items()}
                # If primary_col is missing from the row, splice in
                # the explicitly-fetched cell so the prompt is
                # never silently empty.
                if primary_col and primary_col not in snap:
                    snap[primary_col] = cell_val
            else:
                snap = {primary_col: cell_val} if primary_col else {}
            records.append(snap)
            suspect_value_for_record = cell_val
        else:
            # Multi-row, multi-column, or composite: one snapshot
            # per affected row.
            for r in capped_rows:
                row_dict = {}
                if self.row_lookup is not None:
                    try:
                        row_dict = self.row_lookup(source, r) or {}
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "row_lookup failed for %s row=%s: %s", cid, r, exc,
                        )
                        row_dict = {}
                if capped_cols:
                    # Project to the cluster's columns first; for
                    # multi-row clusters with no cluster.columns the
                    # projection is the full row.
                    snap = {
                        c: ("" if row_dict.get(c) is None else str(row_dict.get(c, "")))
                        for c in capped_cols
                    }
                    # Also keep a few high-signal identifier columns
                    # (e.g. license #, ID, name) so the agent can
                    # ground its searches.
                    for c, v in row_dict.items():
                        if c in snap:
                            continue
                        snap[str(c)] = "" if v is None else str(v)
                else:
                    snap = {
                        str(c): ("" if v is None else str(v))
                        for c, v in row_dict.items()
                    }
                records.append(snap)
            suspect_value_for_record = ""

        # ---- build the Verification skeleton ---------------------
        # ``suspect_value`` is the textual value of the primary
        # suspect cell for single_cell, or a short JSON summary for
        # multi-record patterns (kept short so it fits cleanly in
        # the labeling UI).
        if pattern == "single_cell":
            display_suspect = suspect_value_for_record
        else:
            try:
                display_suspect = json.dumps(records, ensure_ascii=False)[:1000]
            except (TypeError, ValueError):
                display_suspect = ""

        ver = Verification(
            cluster_id=cid,
            source=source,
            row=primary_row if isinstance(primary_row, int) else None,
            column=primary_col,
            suspect_value=str(display_suspect),
            entity_kind=kind.value,
            pattern=pattern,
            rows=list(all_rows),
            columns_all=list(all_cols),
            record_snapshots=[dict(r) for r in records],
        )

        # ---- early-exit: missing data ----------------------------
        if pattern == "single_cell":
            if not suspect_value_for_record:
                ver.error = "suspect cell value unavailable"
                return ver
        else:
            if not records or not any(records):
                ver.error = "row context unavailable for multi-record cluster"
                return ver

        # ---- suggested fix + sibling values ----------------------
        syn = cluster.get("synthesis") or {}
        if not isinstance(syn, Mapping):
            syn = {}
        suggested_fix = (
            (syn.get("canonical_fix") or "")
            or (cluster.get("suggested_fix") or "")
        )

        siblings: List[str] = []
        if pattern == "single_cell" and primary_col:
            try:
                raw_sib = self.column_samples(
                    source, primary_col, self.cfg.sibling_samples,
                ) or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("column_samples failed for %s: %s", cid, exc)
                raw_sib = []
            cleaned = [
                s for s in (str(s).strip() for s in raw_sib)
                if s and s != suspect_value_for_record.strip()
            ]
            seen_sib: set[str] = set()
            siblings = [
                s for s in cleaned
                if not (s in seen_sib or seen_sib.add(s))  # type: ignore[func-returns-value]
            ][: self.cfg.sibling_samples]

        # ---- prompt construction ---------------------------------
        records_block = _format_records_under_review(
            records=records,
            rows=capped_rows or ([primary_row] if isinstance(primary_row, int) else []),
            cluster_columns=capped_cols,
            suspect_column=primary_col if pattern == "single_cell" else "",
        )
        task_block = _format_task_block(
            pattern=pattern,
            columns=capped_cols,
            n_rows=max(len(capped_rows), 1),
        )

        prompt = self._prompt["user_template"].format(
            source=source,
            pattern=pattern,
            column=primary_col or "(unknown)",
            columns=", ".join(capped_cols) if capped_cols else "(none)",
            row=primary_row if isinstance(primary_row, int) else "n/a",
            rows=", ".join(str(r) for r in capped_rows) if capped_rows else "n/a",
            n_records=len(records),
            entity_kind=kind.value,
            error_types=", ".join(
                str(t) for t in (cluster.get("error_types") or [])
            ) or "(none)",
            suspect_value=(
                suspect_value_for_record
                if pattern == "single_cell"
                else "(see RECORDS UNDER REVIEW)"
            ),
            suggested_fix=suggested_fix or "(none)",
            sibling_values=_format_sibling_values(siblings),
            records_block=records_block,
            task_block=task_block,
        )

        verdict: AgentVerdict = self.agent.verify(
            prompt, cache_key=_cache_key(cid, prompt),
        )
        ver.llm_attempts = 1
        ver.llm_endpoint = getattr(self.agent, "endpoint", "") or self.agent.name
        ver.queries = list(verdict.bing_query_urls)
        ver.n_results = int(verdict.n_search_calls)
        ver.n_unique_urls = len(verdict.citations)

        if verdict.error:
            ver.error = verdict.error
            return ver
        if not verdict.text:
            ver.error = "agent returned empty response"
            return ver

        payload, parse_err = _parse_json_object(verdict.text)
        ver.parse_error = parse_err
        if not payload:
            return ver

        verdict_str = str(payload.get("verdict") or "").strip().lower()
        if verdict_str not in ("confirmed_error", "false_positive", "ambiguous"):
            ver.parse_error = f"unknown verdict: {verdict_str!r}"
            return ver
        ver.verdict = verdict_str

        canon = payload.get("canonical_value")
        if canon in (None, "", "null", "None"):
            ver.canonical_value = None
        else:
            ver.canonical_value = str(canon).strip() or None

        # Optional structured per-column fix for multi-column /
        # composite patterns. We accept any mapping the agent emits
        # and filter to columns we asked about (ignoring keys for
        # columns the agent invented).
        canon_record_raw = payload.get("canonical_record")
        if isinstance(canon_record_raw, Mapping) and capped_cols:
            allowed_cols = {c for c in capped_cols if c}
            cleaned_record: Dict[str, str] = {}
            for k, v in canon_record_raw.items():
                key = str(k).strip()
                if not key or key not in allowed_cols:
                    continue
                if v in (None, "", "null", "None"):
                    continue
                cleaned_record[key] = str(v).strip()
            ver.canonical_record = cleaned_record or None

        try:
            ver.confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            ver.confidence = 0.0
        ver.confidence = max(0.0, min(1.0, ver.confidence))
        ver.reasoning = str(payload.get("reasoning") or "").strip()

        # Anti-hallucination filter: only keep cites whose URL the
        # agent runtime actually recorded as a Web Search source
        # (i.e. it appeared as a ``url_citation`` annotation).
        allowed_urls = {c.url for c in verdict.citations if c.url}
        url_to_title = {c.url: c.title for c in verdict.citations}
        for ev in payload.get("evidence") or []:
            if not isinstance(ev, Mapping):
                continue
            url = str(ev.get("url") or "").strip()
            if not url or url not in allowed_urls:
                continue
            ver.evidence.append(EvidenceCitation(
                url=url,
                title=str(ev.get("title") or url_to_title.get(url, "")).strip(),
                quote=str(ev.get("quote") or "").strip()[: self.cfg.snippet_chars],
            ))
            if len(ver.evidence) >= self.cfg.max_snippets:
                break

        # Tighten the verdict if it claims confirmed_error without
        # surviving evidence: downgrade to ambiguous.
        if ver.verdict == "confirmed_error" and not ver.evidence:
            ver.verdict = "ambiguous"
            ver.canonical_value = None
            ver.canonical_record = None
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
        if not isinstance(r, dict):
            continue
        evidence = []
        for ev in r.get("evidence") or []:
            if isinstance(ev, dict):
                evidence.append(EvidenceCitation(
                    url=str(ev.get("url") or ""),
                    title=str(ev.get("title") or ""),
                    quote=str(ev.get("quote") or ""),
                ))
        # Multi-row / multi-column extension fields. Older records
        # written before the schema bump simply default these to
        # empty / None, which preserves the single_cell semantics.
        rows_raw = r.get("rows") or []
        rows: List[int] = []
        if isinstance(rows_raw, list):
            for x in rows_raw:
                try:
                    rows.append(int(x))
                except (TypeError, ValueError):
                    continue
        columns_all_raw = r.get("columns_all") or []
        columns_all = [str(c) for c in columns_all_raw] if isinstance(columns_all_raw, list) else []
        snapshots_raw = r.get("record_snapshots") or []
        record_snapshots: List[Dict[str, str]] = []
        if isinstance(snapshots_raw, list):
            for s in snapshots_raw:
                if isinstance(s, dict):
                    record_snapshots.append({str(k): "" if v is None else str(v) for k, v in s.items()})
        canon_rec_raw = r.get("canonical_record")
        canon_rec: Optional[Dict[str, str]] = None
        if isinstance(canon_rec_raw, dict) and canon_rec_raw:
            canon_rec = {str(k): "" if v is None else str(v) for k, v in canon_rec_raw.items()}

        out.append(Verification(
            cluster_id=str(r.get("cluster_id") or ""),
            source=str(r.get("source") or ""),
            row=r.get("row") if isinstance(r.get("row"), int) else None,
            column=str(r.get("column") or ""),
            suspect_value=str(r.get("suspect_value") or ""),
            entity_kind=str(r.get("entity_kind") or "other"),
            verdict=str(r.get("verdict") or "ambiguous"),
            canonical_value=(
                str(r["canonical_value"])
                if r.get("canonical_value") not in (None, "", "null") else None
            ),
            confidence=float(r.get("confidence") or 0.0),
            evidence=evidence,
            reasoning=str(r.get("reasoning") or ""),
            queries=[str(q) for q in (r.get("queries") or [])],
            n_results=int(r.get("n_results") or 0),
            n_unique_urls=int(r.get("n_unique_urls") or 0),
            error=r.get("error") if r.get("error") else None,
            parse_error=r.get("parse_error") if r.get("parse_error") else None,
            llm_attempts=int(r.get("llm_attempts") or 0),
            llm_endpoint=str(r.get("llm_endpoint") or ""),
            pattern=str(r.get("pattern") or "single_cell"),
            rows=rows,
            columns_all=columns_all,
            record_snapshots=record_snapshots,
            canonical_record=canon_rec,
            ts=str(r.get("ts") or ""),
        ))
    return out
