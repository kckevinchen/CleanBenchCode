"""Schema-discovery: enumerate multi-column relations per error type.

This is "Phase D", a pre-pass that runs ONCE per (source, error_type)
before per-row detection. The LLM sees only the compact whole-table
digest plus a small stratified row sample, and is asked to enumerate
every relation / dependency / constraint that should hold over the
table for a single error type. The resulting relations are persisted
and later fed into per-type detection prompts as a closed
"RELATIONS TO CHECK" list, so detection no longer has to re-discover
the rules from scratch on every row chunk.

Cost shape: one LLM call per (source, error_type) per pass. A small
constant — the multi-column / multi-row error types are a handful of
keys, not the full taxonomy — so the discovery cost is independent
of table height.

Canonicalization (see :func:`Relation.species_key`) folds two
phrasings of the same rule into the same species key. That key is
the unit consumed by the adaptive Good-Turing driver when we
graduate this from a single-pass tool to a multi-pass coverage
estimator (planned Phase D follow-up).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import pandas as pd
import yaml

from .columns import canonicalize_columns
from .error_taxonomy import ErrorTaxonomy, ErrorType
from .llm_client import LLMClient, LLMResponse
from .table_digest import TableDigest

logger = logging.getLogger(__name__)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_WS_RE = re.compile(r"\s+")
# Keep word chars + comparison/arithmetic operators that materially
# change a predicate's meaning ("a <= b" vs "a < b" vs "a >= b").
# Everything else (periods, commas, parentheses, articles spelled out
# with punctuation) is dropped so two phrasings of the same rule
# collapse to the same fold.
_KEEP_RE = re.compile(r"[\w+\-*/=<>]+")


# ---------------------------------------------------------------------------
# Relation dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Relation:
    """One multi-column rule discovered for a (source, error_type).

    The free-form ``predicate`` is the canonical natural-language form
    the LLM emits; ``executable`` is an optional machine-evaluable
    expression for kinds that admit it (calculations, aggregations,
    ordered comparisons). ``species_key`` is the stable hash used by
    the adaptive coverage estimator to deduplicate paraphrases of the
    same underlying rule across passes.
    """

    source: str
    error_type: str
    kind: str
    columns: Tuple[str, ...]
    predicate: str
    rationale: str = ""
    confidence: str = "medium"
    lhs: Tuple[str, ...] = ()
    rhs: Tuple[str, ...] = ()
    executable: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Tuples are not JSON-serialisable as themselves but ``asdict``
        # already converted them - just normalise to lists for stable
        # JSON output.
        d["columns"] = list(self.columns)
        d["lhs"] = list(self.lhs)
        d["rhs"] = list(self.rhs)
        d["raw"] = dict(self.raw)
        d["species_key"] = self.species_key()
        return d

    def species_key(self) -> str:
        """Stable canonical hash for adaptive Good-Turing accounting.

        Two relations collapse to the same species when:
          * the error type matches, AND
          * the kind matches, AND
          * the column set matches (sorted; lhs/rhs preserved separately
            for directional kinds), AND
          * a fold of the predicate text matches (lower, whitespace
            collapsed, punctuation stripped beyond a small allowed set).

        Bijection collapse: when the kind is ``fd`` and BOTH ``lhs``
        and ``rhs`` are single-column, ``A -> B`` and ``B -> A`` fold
        to the same species. The two directions encode the same
        functional constraint (the columns are bijective in this
        slice) and double-counting them inflates the species pool
        without giving the LLM more signal.

        Conservative dedup is fine: over-counting just makes the
        adaptive driver run an extra pass; under-counting (false
        merges) silently hides relations.
        """
        cols = ",".join(sorted(self.columns))
        if (
            self.kind == "fd"
            and len(self.lhs) == 1
            and len(self.rhs) == 1
        ):
            # Symmetric pair: sort the two columns so direction is
            # erased. Predicate is also dropped from the key because
            # the two directions naturally phrase the rule
            # differently (e.g. "A determines B" vs "B determines A")
            # and we explicitly want them to collapse.
            pair = ",".join(sorted([self.lhs[0], self.rhs[0]]))
            payload = f"{self.error_type}|{self.kind}|{cols}|bijection|{pair}"
        else:
            lhs = ",".join(sorted(self.lhs))
            rhs = ",".join(sorted(self.rhs))
            pred = _fold_predicate(self.predicate)
            payload = f"{self.error_type}|{self.kind}|{cols}|{lhs}|{rhs}|{pred}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _fold_predicate(text: str) -> str:
    s = (text or "").lower()
    tokens = _KEEP_RE.findall(s)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Per-error-type prompt config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryTypeSpec:
    """Per-error-type entry in ``prompts/discover_relations.yaml``."""

    key: str
    kinds: Tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryPromptConfig:
    """Loaded ``prompts/discover_relations.yaml`` (system + template + per-type)."""

    system_prompt: str
    user_template: str
    types: Mapping[str, DiscoveryTypeSpec]
    source_path: Optional[Path] = None

    def render(
        self,
        *,
        error_type: ErrorType,
        digest: TableDigest,
        row_sample_md: str,
    ) -> str:
        spec = self.types.get(error_type.key)
        if spec is None:
            raise KeyError(
                f"discover_relations.yaml has no entry for "
                f"error_type '{error_type.key}'"
            )
        kinds_block = "\n".join(f"  - {k}" for k in spec.kinds)
        return self.user_template.format(
            type_key=error_type.key,
            type_name=error_type.name,
            type_category=error_type.category,
            type_description=error_type.description.strip(),
            type_example=error_type.example.strip()
            or "(no reference example provided)",
            type_notes=error_type.notes.strip() or "(no additional guidance)",
            table_overview=digest.to_prompt_block(),
            row_sample=row_sample_md,
            error_type_kinds=kinds_block,
        )


def load_discovery_prompt(path: Path | str) -> DiscoveryPromptConfig:
    """Parse ``prompts/discover_relations.yaml`` into a config object."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Discovery prompt not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{p}: top level must be a mapping")
    base = raw.get("base") or {}
    if not isinstance(base, Mapping):
        raise ValueError(f"{p}: 'base' must be a mapping")
    system_prompt = str(base.get("system", "")).strip()
    user_template = str(base.get("user_template", ""))
    if not system_prompt or not user_template:
        raise ValueError(
            f"{p}: 'base.system' and 'base.user_template' are required"
        )
    types_raw = raw.get("types") or {}
    if not isinstance(types_raw, Mapping):
        raise ValueError(f"{p}: 'types' must be a mapping")
    types: Dict[str, DiscoveryTypeSpec] = {}
    for key, body in types_raw.items():
        if not isinstance(body, Mapping):
            raise ValueError(f"{p}: types.{key} must be a mapping")
        kinds = body.get("kinds") or []
        if not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds):
            raise ValueError(f"{p}: types.{key}.kinds must be a list of strings")
        types[str(key)] = DiscoveryTypeSpec(key=str(key), kinds=tuple(kinds))
    return DiscoveryPromptConfig(
        system_prompt=system_prompt,
        user_template=user_template,
        types=types,
        source_path=p,
    )


# ---------------------------------------------------------------------------
# Stratified row sampling
# ---------------------------------------------------------------------------


def sample_rows_for_prompt(
    df: pd.DataFrame,
    *,
    n_samples: int = 5,
    total_row_indices: Sequence[int] = (),
    max_cell_chars: int = 80,
    seed: Optional[int] = None,
) -> str:
    """Return a Markdown table with a stratified row sample.

    When ``seed`` is ``None`` (default), picks ``n_samples`` row
    indices spread evenly across the table (first, last, evenly-spaced
    interior rows) - deterministic, reproducible, no per-pass
    variation. This is the right choice for a one-shot run.

    When ``seed`` is an integer, draws ``n_samples`` indices
    pseudo-randomly without replacement using ``random.Random(seed)``.
    Use this in adaptive multi-pass mode to surface different rows
    each pass and let the LLM see different evidence; the seed is
    deterministic so a pass can be replayed exactly.

    In both cases rows the digest has flagged as total / summary rows
    are EXCLUDED - those distort grounding for non-aggregation rules.
    Falls back to "every row" when the table is shorter than the
    requested sample.
    """
    if df.empty:
        return "(empty table - no row sample available)"

    n_rows = len(df)
    excluded = set(int(i) for i in total_row_indices)
    candidates = [i for i in range(n_rows) if i not in excluded]
    if not candidates:
        candidates = list(range(n_rows))

    if len(candidates) <= n_samples:
        picks = candidates
    elif seed is None:
        # Evenly-spaced indices into the candidate list, including
        # both endpoints.
        step = (len(candidates) - 1) / (n_samples - 1)
        picks = sorted({candidates[int(round(i * step))] for i in range(n_samples)})
    else:
        rng = random.Random(int(seed))
        picks = sorted(rng.sample(candidates, n_samples))

    sample = df.iloc[picks].copy()

    def _truncate(v: object) -> str:
        s = "" if pd.isna(v) else str(v)
        if len(s) > max_cell_chars:
            return s[: max_cell_chars - 1] + "\u2026"
        return s

    sample = sample.astype(object).map(_truncate)
    sample.insert(0, "__row__", picks)
    return sample.to_markdown(index=False)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _coerce_relations_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("relations", "rules", "constraints"):
            v = payload.get(key)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        return [payload]
    return []


def _parse_relations_json(text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not text:
        return [], "empty response"
    candidates: List[str] = []
    for m in _JSON_BLOCK_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text.strip())
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            payload = json.loads(cand)
        except json.JSONDecodeError:
            continue
        return _coerce_relations_payload(payload), None
    return [], "failed to parse JSON from response"


def _normalize_columns(value: Any, frame_headers: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    resolved = canonicalize_columns(value, frame_headers)
    return tuple(resolved)


def _list_len(value: Any) -> int:
    """Number of column tokens the LLM declared. Used to spot
    hallucinated columns: if canonicalization drops any, the
    relation references something not in the schema."""
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, (list, tuple)):
        return sum(1 for v in value if isinstance(v, str) and v.strip())
    return 0


# ---------------------------------------------------------------------------
# Per-(source, error_type) discovery report
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryReport:
    """Result of one (source, error_type) discovery LLM call."""

    source: str
    error_type: str
    error_type_name: str
    error_type_category: str
    model: str
    prompt: str
    relations: List[Relation] = field(default_factory=list)
    raw_text: str = ""
    reasoning_summary: str = ""
    endpoint: str = ""
    attempts: int = 0
    error: Optional[str] = None
    parse_error: Optional[str] = None

    def as_dict(self, *, include_raw: bool = True) -> Dict[str, Any]:
        d = {
            "source": self.source,
            "error_type": self.error_type,
            "error_type_name": self.error_type_name,
            "error_type_category": self.error_type_category,
            "model": self.model,
            "endpoint": self.endpoint,
            "attempts": self.attempts,
            "error": self.error,
            "parse_error": self.parse_error,
            "n_relations": len(self.relations),
            "relations": [r.as_dict() for r in self.relations],
        }
        if include_raw:
            d["prompt"] = self.prompt
            d["raw_text"] = self.raw_text
            d["reasoning_summary"] = self.reasoning_summary
        return d


# ---------------------------------------------------------------------------
# Discoverer
# ---------------------------------------------------------------------------


class SchemaDiscoverer:
    """Run schema-discovery LLM calls over (source, error_type) pairs.

    One call per (source, error_type) pair. Calls are dispatched
    through a thread pool so multiple sources / error types run
    concurrently across the LLM endpoint pool, mirroring
    :class:`PerTypeCleaningDetector`.

    The discoverer is intentionally STATELESS w.r.t. perturbation:
    callers wanting multi-pass adaptive discovery rebuild the
    discoverer per pass with whatever variant prompt-config / digest
    sample they want. This keeps the single-pass code path trivial
    while leaving room to graft the adaptive driver on top.
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_config: DiscoveryPromptConfig,
        taxonomy: ErrorTaxonomy,
        model: str,
        *,
        digests: Mapping[str, TableDigest],
        source_frames: Mapping[str, pd.DataFrame],
        error_types: Optional[Sequence[ErrorType]] = None,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        concurrency: Optional[int] = None,
        n_row_samples: int = 5,
    ) -> None:
        self.llm = llm
        self.prompt_config = prompt_config
        self.taxonomy = taxonomy
        self.model = model
        self.digests = dict(digests)
        self.source_frames = dict(source_frames)
        # Default: every taxonomy type that has a discovery entry. The
        # discovery YAML deliberately omits the single-column types,
        # so this filter naturally restricts to multi-column / multi-
        # row / multi-table types.
        all_types = list(error_types) if error_types is not None else list(
            taxonomy.types
        )
        self.error_types = [
            et for et in all_types if et.key in self.prompt_config.types
        ]
        if not self.error_types:
            raise ValueError(
                "SchemaDiscoverer: no error types match discover_relations.yaml; "
                "check the taxonomy / discovery YAML pairing."
            )
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        default_concurrency = min(8, max(1, len(self.llm.endpoints)))
        self.concurrency = concurrency if concurrency and concurrency > 0 else default_concurrency
        self.n_row_samples = max(1, int(n_row_samples))

        # Pre-render row samples once per source. Sampling is
        # deterministic, so this is cheap and avoids duplicating the
        # work across error types.
        self._row_samples: Dict[str, str] = {}
        for src, frame in self.source_frames.items():
            digest = self.digests.get(src)
            totals = digest.total_row_indices if digest else ()
            self._row_samples[src] = sample_rows_for_prompt(
                frame,
                n_samples=self.n_row_samples,
                total_row_indices=totals,
            )

    # ------------------------------------------------------------------
    # Single-call execution
    # ------------------------------------------------------------------

    def _discover(self, source: str, error_type: ErrorType, *, seed: Optional[int] = None) -> DiscoveryReport:
        digest = self.digests.get(source)
        if digest is None:
            raise KeyError(f"No digest available for source {source!r}")
        if seed is None:
            row_sample_md = self._row_samples.get(source, "(no row sample)")
        else:
            frame = self.source_frames[source]
            totals = digest.total_row_indices if digest else ()
            row_sample_md = sample_rows_for_prompt(
                frame,
                n_samples=self.n_row_samples,
                total_row_indices=totals,
                seed=seed,
            )
        prompt = self.prompt_config.render(
            error_type=error_type,
            digest=digest,
            row_sample_md=row_sample_md,
        )
        report = DiscoveryReport(
            source=source,
            error_type=error_type.key,
            error_type_name=error_type.name,
            error_type_category=error_type.category,
            model=self.model,
            prompt=prompt,
        )
        response: LLMResponse = self.llm.query(
            self.model,
            prompt,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            system_prompt=self.prompt_config.system_prompt,
        )
        report.endpoint = response.endpoint
        report.attempts = response.attempts
        report.raw_text = response.text
        report.reasoning_summary = response.reasoning_summary
        report.error = response.error

        if not response.ok:
            return report

        items, parse_err = _parse_relations_json(response.text)
        report.parse_error = parse_err

        frame = self.source_frames[source]
        headers = list(frame.columns)
        seen_keys: set[str] = set()
        relations: List[Relation] = []
        for raw in items:
            cols = _normalize_columns(raw.get("columns"), headers)
            cols_declared = _list_len(raw.get("columns"))
            if cols_declared and len(cols) != cols_declared:
                # The LLM named a column we cannot resolve - reject
                # the whole relation rather than silently truncating
                # it to a different (possibly meaningless) shape.
                logger.debug(
                    "Dropping relation with unresolved columns: %r", raw,
                )
                continue
            lhs = _normalize_columns(raw.get("lhs"), headers)
            rhs = _normalize_columns(raw.get("rhs"), headers)
            if (
                _list_len(raw.get("lhs")) != len(lhs)
                or _list_len(raw.get("rhs")) != len(rhs)
            ):
                logger.debug(
                    "Dropping relation with unresolved lhs/rhs: %r", raw,
                )
                continue
            if not cols:
                # Fall back to lhs ∪ rhs if `columns` was omitted.
                cols = tuple(dict.fromkeys(lhs + rhs))
                if not cols:
                    continue
            kind = str(raw.get("kind") or "").strip()
            if not kind:
                continue
            predicate = str(raw.get("predicate") or "").strip()
            if not predicate:
                continue
            executable_raw = raw.get("executable")
            executable = (
                str(executable_raw).strip()
                if isinstance(executable_raw, str) and executable_raw.strip()
                else None
            )
            relation = Relation(
                source=source,
                error_type=error_type.key,
                kind=kind,
                columns=cols,
                predicate=predicate,
                rationale=str(raw.get("rationale") or "").strip(),
                confidence=str(raw.get("confidence") or "medium").strip().lower(),
                lhs=lhs,
                rhs=rhs,
                executable=executable,
                raw=dict(raw),
            )
            key = relation.species_key()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            relations.append(relation)
        report.relations = relations
        return report

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        seed: Optional[int] = None,
        skip: Optional[Any] = None,
        desc: str = "Discover",
    ) -> List[DiscoveryReport]:
        """Run one discovery pass.

        ``seed`` controls per-pass row sampling: ``None`` uses the
        deterministic evenly-spaced sample (single-pass mode);
        passing an integer reseeds the row sample so adaptive
        passes see different rows.

        ``skip`` is an optional callable ``(source, error_type_key)
        -> bool``; arms for which it returns truthy are skipped this
        pass. Used by the adaptive driver to retire saturated arms.

        ``desc`` is the tqdm progress-bar label; the adaptive driver
        overrides it with a per-pass title.
        """
        skip_fn = skip if callable(skip) else (lambda _s, _e: False)
        jobs: List[Tuple[str, ErrorType]] = [
            (src, et)
            for src in self.source_frames
            for et in self.error_types
            if src in self.digests and not skip_fn(src, et.key)
        ]
        total = len(jobs)
        # Demoted to debug: the tqdm bar already shows job count and
        # progress. The pass-summary log lines above the bar were the
        # main source of noise in the workbench job tail.
        logger.debug(
            "Schema discovery pass: %d job(s) (concurrency=%d, seed=%s)",
            total, self.concurrency, seed,
        )
        if total == 0:
            return []

        results: List[Optional[DiscoveryReport]] = [None] * total
        progress = tqdm(
            total=total,
            desc=desc,
            unit="call",
            dynamic_ncols=True,
            leave=True,
        )
        progress_lock = threading.Lock()
        counters = {"errors": 0, "parse": 0, "relations": 0}

        def _on_done(report: DiscoveryReport, src: str, et: ErrorType) -> None:
            with progress_lock:
                if report.error:
                    counters["errors"] += 1
                    logger.warning(
                        "discover failed: source=%s type=%s error=%s",
                        src, et.key, report.error,
                    )
                if report.parse_error:
                    counters["parse"] += 1
                counters["relations"] += len(report.relations)
                progress.set_postfix(
                    rels=counters["relations"],
                    err=counters["errors"],
                    parse=counters["parse"],
                    refresh=False,
                )
                progress.update(1)

        try:
            with logging_redirect_tqdm():
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = {
                        pool.submit(self._discover, src, et, seed=seed): (idx, src, et)
                        for idx, (src, et) in enumerate(jobs)
                    }
                    for fut in as_completed(futures):
                        idx, src, et = futures[fut]
                        try:
                            report = fut.result()
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.exception(
                                "Schema discovery failed for source=%s error_type=%s",
                                src, et.key,
                            )
                            report = DiscoveryReport(
                                source=src,
                                error_type=et.key,
                                error_type_name=et.name,
                                error_type_category=et.category,
                                model=self.model,
                                prompt="",
                                error=f"unhandled: {exc}",
                            )
                        results[idx] = report
                        _on_done(report, src, et)
        finally:
            progress.close()
        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def write_relations_jsonl(reports: Iterable[DiscoveryReport], path: Path) -> int:
    """Write one JSON line per (source, error_type, relation).

    Returns the total number of relation lines written. Reports that
    surface zero relations contribute nothing - the file's purpose is
    to be the closed list of relations downstream detection consumes.
    """
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rep in reports:
            for rel in rep.relations:
                fh.write(json.dumps(rel.as_dict(), ensure_ascii=False) + "\n")
                n += 1
    return n


def write_discovery_reports_json(
    reports: Iterable[DiscoveryReport],
    path: Path,
    *,
    include_raw: bool = False,
) -> None:
    """Write the full per-call reports (slim by default; raw=True keeps
    the full LLM response text for debugging)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [rep.as_dict(include_raw=include_raw) for rep in reports]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
