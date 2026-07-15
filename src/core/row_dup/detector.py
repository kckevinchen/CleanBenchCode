"""Row-duplication detector that bypasses chunk-level scanning.

The default per-chunk LLM scan misses duplicate pairs that straddle a
chunk boundary on tables larger than ``chunk_size`` rows - exactly the
case where ``row_duplication`` matters most. This module replaces that
scan with a three-stage pipeline:

1. **Deterministic key discovery.** We run the existing
   :func:`src.core.multi_row_priors.find_duplicate_subsets` (and,
   optionally, :func:`find_duplicate_subsets_normalized`) over the
   *whole* source frame. The output is a small set of candidate
   identity-key subsets where collisions actually occur.

2. **Collision-group enumeration.** For every candidate key, we list
   ALL groups (not just samples) of >= 2 rows that share the key
   value, via :func:`enumerate_collision_groups`. Cross-group
   deduplication ensures the same row-pair is not adjudicated twice
   under nested keys.

3. **Per-group LLM adjudication.** For each remaining group we send
   one LLM call asking "are these rows duplicates?". Each call is
   independent of the table size; concurrency is delegated to a
   thread pool. The verdict is parsed into a single
   :class:`~src.core.detector.ChunkReport` per group, in the same
   shape that :class:`~src.core.detector.PerTypeCleaningDetector`
   emits, so the existing aggregation, adaptive estimator, and
   reporting layers consume the output unchanged.

Configuration is read from the ``row_duplication`` block in
``config/experiment.yaml``; see :class:`AdjudicatorSettings` for the
plumbed-through fields.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ...adaptive.inner_loop import (
    InnerLoopResult,
    merge_partitions,
    run_adaptive,
)
from ...adaptive.scoped_gt import ArmConfig
from ..detector import ChunkReport, parse_llm_json
from ..error_taxonomy import ErrorTaxonomy
from ..llm_client import LLMClient, LLMResponse
from .priors import (
    enumerate_collision_groups,
    find_duplicate_subsets,
    find_duplicate_subsets_normalized,
    normalize_for_dedup,
)
from .clusterers import (
    BlockingClusterer,
    Cluster,
    Clusterer,
    ClusterBatch,
    EmbeddingClusterer,
    KeyCollisionClusterer,
    MinHashLSHClusterer,
    merge_clusters,
    rank_and_cap,
)
from ..table_digest import TableDigest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdjudicatorSettings:
    """Settings consumed by :class:`RowDuplicationDetector`.

    Mirrors the ``row_duplication`` block in ``config/experiment.yaml``.
    """

    detection_mode: str = "collision_adjudication"
    max_key_subsets: int = 4
    max_key_size: int = 2
    min_dup_rate: float = 0.0
    max_groups_per_source: int = 200
    normalize_keys: bool = True
    adjudicate_groups: bool = True
    # Cap on rows passed to the LLM in ``whole_table`` mode (the
    # zero-clustering PoC mode that hands the entire table to one
    # LLM call). Sources with more than this many rows are skipped
    # for ``whole_table`` mode (the operator should switch to
    # ``collision_adjudication`` for them). Default is conservative
    # so the prompt fits comfortably in a 128k-context call.
    whole_table_max_rows: int = 200
    # Per-column LLM canonicalisation knobs. When enabled, before the
    # collision scan we ask the LLM to bucket equivalent surface forms
    # within each candidate key column ("US"/"USA"/"United States" ->
    # one canonical key). The resulting value->canonical map is used
    # as the normaliser inside ``enumerate_collision_groups`` so
    # semantic duplicates collide deterministically.
    canonicalize_keys: bool = False
    canonicalize_max_cardinality: int = 200
    canonicalize_prompt_path: str = "prompts/canonicalize_values.yaml"
    # Token-signature blocking (off by default; recall-first
    # candidate generator that catches abbreviations / variants
    # without an exact-key match).
    blocking_enabled: bool = False
    blocking_signature: str = "first_token"
    blocking_min_block_size: int = 2
    blocking_max_block_size: int = 50
    # MinHash LSH near-duplicate clusterer (off by default; needs
    # ``datasketch`` installed). Catches row-level near-duplicates
    # at a tunable Jaccard threshold.
    minhash_enabled: bool = False
    minhash_threshold: float = 0.7
    minhash_num_perm: int = 128
    minhash_shingle_size: int = 1
    minhash_min_tokens: int = 3
    minhash_max_cluster_size: int = 50
    # Sentence-embedding near-duplicate clusterer (off by default;
    # needs ``sentence-transformers`` installed). Catches paraphrase /
    # semantic near-duplicates that token-level methods miss.
    embedding_enabled: bool = False
    embedding_threshold: float = 0.85
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_min_chars: int = 8
    embedding_max_cluster_size: int = 50
    # NOTE: "adaptive" is not a separate toggle. Whenever
    # ``adjudicate_groups`` or ``canonicalize_keys`` is on, the
    # detector drives the LLM call through the shared adaptive inner
    # loop. The stop rule (``stop_when`` / ``epsilon`` /
    # ``epsilon_chao1`` / ``k_min`` / ``k_max`` /
    # ``require_two_consecutive``) is read from
    # ``good_turing.per_arm.defaults`` and applied uniformly. With
    # ``k_max == 1`` the loop runs exactly once, which matches the
    # old single-shot behaviour.

    def __post_init__(self) -> None:
        # Range / sanity checks. We fail loudly here so a malformed
        # ``config/experiment.yaml`` is caught at startup instead of
        # silently producing zero candidates mid-run.
        if self.max_key_subsets <= 0:
            raise ValueError(
                f"max_key_subsets must be > 0 (got {self.max_key_subsets})"
            )
        if self.max_key_size <= 0:
            raise ValueError(
                f"max_key_size must be > 0 (got {self.max_key_size})"
            )
        if not (0.0 <= self.min_dup_rate <= 1.0):
            raise ValueError(
                f"min_dup_rate must be in [0,1] (got {self.min_dup_rate})"
            )
        if self.max_groups_per_source <= 0:
            raise ValueError(
                "max_groups_per_source must be > 0 "
                f"(got {self.max_groups_per_source})"
            )
        if self.whole_table_max_rows <= 0:
            raise ValueError(
                "whole_table_max_rows must be > 0 "
                f"(got {self.whole_table_max_rows})"
            )
        if self.canonicalize_max_cardinality <= 0:
            raise ValueError(
                "canonicalize_max_cardinality must be > 0 "
                f"(got {self.canonicalize_max_cardinality})"
            )
        if not (0.0 <= self.minhash_threshold <= 1.0):
            raise ValueError(
                f"minhash_threshold must be in [0,1] (got {self.minhash_threshold})"
            )
        if not (0.0 <= self.embedding_threshold <= 1.0):
            raise ValueError(
                f"embedding_threshold must be in [0,1] "
                f"(got {self.embedding_threshold})"
            )
        if self.minhash_num_perm <= 0:
            raise ValueError(
                f"minhash_num_perm must be > 0 (got {self.minhash_num_perm})"
            )

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "AdjudicatorSettings":
        if not cfg:
            return cls()
        return cls(
            detection_mode=str(
                cfg.get("detection_mode", cls.detection_mode)
            ).strip(),
            max_key_subsets=int(cfg.get("max_key_subsets", cls.max_key_subsets)),
            max_key_size=int(cfg.get("max_key_size", cls.max_key_size)),
            min_dup_rate=float(cfg.get("min_dup_rate", cls.min_dup_rate)),
            max_groups_per_source=int(
                cfg.get("max_groups_per_source", cls.max_groups_per_source)
            ),
            normalize_keys=bool(cfg.get("normalize_keys", cls.normalize_keys)),
            adjudicate_groups=bool(
                cfg.get("adjudicate_groups", cls.adjudicate_groups)
            ),
            whole_table_max_rows=int(
                cfg.get("whole_table_max_rows", cls.whole_table_max_rows)
            ),
            canonicalize_keys=bool(
                cfg.get("canonicalize_keys", cls.canonicalize_keys)
            ),
            canonicalize_max_cardinality=int(
                cfg.get(
                    "canonicalize_max_cardinality",
                    cls.canonicalize_max_cardinality,
                )
            ),
            canonicalize_prompt_path=str(
                cfg.get(
                    "canonicalize_prompt_path", cls.canonicalize_prompt_path
                )
            ),
            blocking_enabled=bool(
                cfg.get("blocking_enabled", cls.blocking_enabled)
            ),
            blocking_signature=str(
                cfg.get("blocking_signature", cls.blocking_signature)
            ),
            blocking_min_block_size=int(
                cfg.get("blocking_min_block_size", cls.blocking_min_block_size)
            ),
            blocking_max_block_size=int(
                cfg.get("blocking_max_block_size", cls.blocking_max_block_size)
            ),
            minhash_enabled=bool(
                cfg.get("minhash_enabled", cls.minhash_enabled)
            ),
            minhash_threshold=float(
                cfg.get("minhash_threshold", cls.minhash_threshold)
            ),
            minhash_num_perm=int(
                cfg.get("minhash_num_perm", cls.minhash_num_perm)
            ),
            minhash_shingle_size=int(
                cfg.get("minhash_shingle_size", cls.minhash_shingle_size)
            ),
            minhash_min_tokens=int(
                cfg.get("minhash_min_tokens", cls.minhash_min_tokens)
            ),
            minhash_max_cluster_size=int(
                cfg.get(
                    "minhash_max_cluster_size", cls.minhash_max_cluster_size
                )
            ),
            embedding_enabled=bool(
                cfg.get("embedding_enabled", cls.embedding_enabled)
            ),
            embedding_threshold=float(
                cfg.get("embedding_threshold", cls.embedding_threshold)
            ),
            embedding_model=str(
                cfg.get("embedding_model", cls.embedding_model)
            ),
            embedding_min_chars=int(
                cfg.get("embedding_min_chars", cls.embedding_min_chars)
            ),
            embedding_max_cluster_size=int(
                cfg.get(
                    "embedding_max_cluster_size",
                    cls.embedding_max_cluster_size,
                )
            ),
        )


# ---------------------------------------------------------------------------
# Group-job dataclass
# ---------------------------------------------------------------------------


# ``_CollisionGroup`` is an internal alias for :class:`Cluster` from
# :mod:`row_dup_clusterers`. Kept as a type alias so older code paths
# that referenced the previous local dataclass keep working without
# rewriting every helper signature.
_CollisionGroup = Cluster


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


_ROW_DUPLICATION_KEY = "row_duplication"


# ---------------------------------------------------------------------------
# Per-column key canonicaliser (LLM-driven)
# ---------------------------------------------------------------------------


class KeyCanonicalizer:
    """LLM-driven equivalence-class builder for a single column's values.

    For each candidate key column, takes the distinct non-null values
    and asks the LLM to bucket synonymous surface forms ("US",
    "U.S.A.", "United States" -> one canonical key). The returned
    map is a ``value -> canonical`` dict that the duplicate scan
    feeds into :func:`enumerate_collision_groups` as a normaliser.

    Each (source, column) is driven through the shared adaptive
    inner loop. With ``arm_config.k_max == 1`` that is a single LLM
    call; with k_max > 1 the call is repeated with shuffled value
    order and the resulting partitions are merged via union-find
    (recall-first). Skipped silently when the column's distinct
    cardinality exceeds ``max_cardinality`` (the canonicalisation
    prompt would be too long and a single call would not be
    reliable).
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        model: str,
        prompt_path: Path,
        arm_config: ArmConfig,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        max_cardinality: int = 200,
    ) -> None:
        self.llm = llm
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.max_cardinality = max(0, int(max_cardinality))
        # ``canonicalize_column`` always drives the LLM call through
        # the shared adaptive inner loop. With ``k_max == 1`` it is a
        # single shot; with k_max > 1 it shuffles value order between
        # runs and merges the resulting partitions via union-find
        # (recall-first). The config is shared with the outer adaptive
        # loop so the same FN-control intent applies uniformly.
        self.arm_config = arm_config
        self._prompt = self._load_prompt(prompt_path)

    @staticmethod
    def _load_prompt(path: Path) -> Mapping[str, str]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"canonicalize prompt not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: top level must be a mapping")
        system = str(raw.get("system", "")).strip()
        template = str(raw.get("user_template", ""))
        if not system or not template:
            raise ValueError(f"{path}: 'system' and 'user_template' required")
        return {"system": system, "user_template": template}

    def canonicalize_column(
        self, *, source: str, column: str, values: Sequence[str]
    ) -> Dict[str, str]:
        """Return a ``value -> canonical`` map for ``values``.

        On any failure (cardinality too high, parse error, model
        returns malformed groups) returns an *identity* map so the
        caller can fall back to lexical matching without crashing.

        Always drives the call through the adaptive inner loop using
        ``self.arm_config``. With ``k_max == 1`` that is a single LLM
        call; with k_max > 1 partitions across runs are merged via
        union-find.
        """
        distinct = sorted({v for v in values if v != ""})
        identity = {v: v for v in distinct}
        if not distinct:
            return identity
        if (
            self.max_cardinality > 0
            and len(distinct) > self.max_cardinality
        ):
            logger.info(
                "canonicalize_keys: skipping %s.%s (cardinality %d > cap %d)",
                source,
                column,
                len(distinct),
                self.max_cardinality,
            )
            return identity
        if len(distinct) < 2:
            return identity

        partitions = self._adaptive_partitions(
            source=source, column=column, distinct=distinct,
        )
        if not partitions:
            return identity

        # Merge all partitions via union-find. A single run that
        # bucketed two values together is enough to merge them in the
        # final output - mirrors the recall-first design of the rest
        # of the row-duplication pipeline.
        merged_classes = merge_partitions(partitions)

        out: Dict[str, str] = {}
        for cls in merged_classes:
            members = [m for m in cls if m in distinct]
            if not members:
                continue
            canonical = max(members, key=len)
            for m in members:
                out[m] = canonical
        # Anything unmapped: identity. Guarantees every original value
        # is still in the resulting map so downstream lookups don't
        # KeyError.
        for v in distinct:
            out.setdefault(v, v)
        merged_count = sum(1 for v, c in out.items() if v != c)
        if merged_count:
            logger.info(
                "canonicalize_keys: %s.%s collapsed %d/%d distinct value(s) "
                "into canonical forms (%d run%s)",
                source, column, merged_count, len(distinct),
                len(partitions), "" if len(partitions) == 1 else "s",
            )
        return out

    def _adaptive_partitions(
        self, *, source: str, column: str, distinct: Sequence[str],
    ) -> List[List[frozenset]]:
        """Drive the inner adaptive loop, returning per-run partitions."""
        partitions: List[List[frozenset]] = []
        # Stable per-(source, column) RNG so repeat invocations on
        # the same input are reproducible.
        rng = _stable_rng("canonicalize", str(source), str(column))

        def sample_fn(run_idx: int):
            order = list(distinct)
            if run_idx > 0 and len(order) > 1:
                rng.shuffle(order)
            partition = self._canonicalize_once(
                source=source, column=column, value_order=order,
            )
            if partition is None:
                partitions.append([])
                return frozenset()
            partitions.append(partition)
            # Species = the partition itself, encoded as a tuple of
            # sorted member-tuples so two runs producing the same
            # buckets collide as one species.
            return [tuple(sorted(c)) for c in partition]

        run_adaptive(
            arm_id=f"canonicalize:{source}:{column}",
            sample_fn=sample_fn,
            cfg=self.arm_config,
        )
        return partitions

    def _canonicalize_once(
        self, *, source: str, column: str, value_order: Sequence[str],
    ) -> Optional[List[frozenset]]:
        """One LLM call. Returns a list of frozenset partitions, or None on failure.

        ``value_order`` controls the order in which values appear in
        the prompt; shuffling between runs perturbs the LLM's output
        without changing the underlying value set.
        """
        distinct = set(value_order)
        value_list = "\n".join(f'- "{v}"' for v in value_order)
        try:
            prompt = self._prompt["user_template"].format(
                source=source, column=column, value_list=value_list
            )
        except KeyError as exc:
            logger.warning(
                "canonicalize_keys: prompt template missing placeholder %s; "
                "using identity map", exc,
            )
            return None

        try:
            response = self.llm.query(
                self.model,
                prompt,
                reasoning_effort=self.reasoning_effort,
                reasoning_summary=self.reasoning_summary,
                system_prompt=self._prompt["system"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "canonicalize_keys: LLM call failed for %s.%s: %s; "
                "using identity map",
                source, column, exc,
            )
            return None

        groups, parse_err = _parse_canonical_groups(response.text)
        if parse_err is not None:
            logger.warning(
                "canonicalize_keys: parse failed for %s.%s: %s; "
                "using identity map",
                source, column, parse_err,
            )
            return None

        partition: List[frozenset] = []
        for g in groups:
            members = g.get("members") or []
            if not isinstance(members, list):
                continue
            kept = [
                m for m in members
                if isinstance(m, str) and m in distinct
            ]
            if not kept:
                continue
            partition.append(frozenset(kept))
        return partition


def _parse_canonical_groups(
    text: str,
) -> Tuple[List[Mapping[str, Any]], Optional[str]]:
    """Pull a ``{"groups": [...]}`` payload out of an LLM response."""
    issues, err = parse_llm_json(text)
    # ``parse_llm_json`` returns the issues list when the top-level
    # has an "issues" key; for canonicalisation we want "groups".
    # Fall back to manual JSON extraction when it didn't return
    # anything useful.
    if isinstance(issues, list) and issues and "members" in (issues[0] or {}):
        return list(issues), None  # type: ignore[arg-type]
    # Manual scan for the JSON object.
    stripped = (text or "").strip()
    # Strip markdown fences if present.
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return [], f"JSON decode error: {exc}"
    if not isinstance(obj, Mapping):
        return [], "top-level JSON is not an object"
    groups = obj.get("groups")
    if not isinstance(groups, list):
        return [], "missing 'groups' list"
    return list(groups), None


class RowDuplicationDetector:
    """Whole-table row-duplication detector.

    Parameters
    ----------
    llm
        Shared, thread-safe LLM client.
    taxonomy
        Loaded error-type catalogue. The detector reads the
        ``row_duplication`` :class:`ErrorType` for its prompt
        description / example / notes.
    model
        Deployment name passed to :meth:`LLMClient.query`.
    settings
        Detector configuration.
    digests
        Optional whole-table digests keyed by ``str(source_path)``.
        Used to seed candidate identity columns.
    reasoning_effort, reasoning_summary
        Forwarded to :meth:`LLMClient.query`.
    concurrency
        Maximum number of concurrent group adjudications. Defaults
        to ``min(8, len(llm.endpoints))``.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        taxonomy: ErrorTaxonomy,
        model: str,
        settings: AdjudicatorSettings,
        digests: Optional[Mapping[str, TableDigest]] = None,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        concurrency: Optional[int] = None,
        arm_config: Optional[ArmConfig] = None,
    ) -> None:
        self.llm = llm
        self.taxonomy = taxonomy
        self.model = model
        self.settings = settings
        # Per-CSV Good-Turing arms collected during a ``whole_table`` run
        # (one InnerLoopResult per source). The caller drains this after
        # ``run()`` and writes it to good_turing/per_arm_estimates.json so
        # whole_table row_duplication reports per-CSV saturation like the
        # other adaptive types. Stays empty for collision_adjudication.
        self.whole_table_arms: List[Dict[str, Any]] = []
        self.digests = dict(digests or {})
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        # Resolve the adaptive-loop ArmConfig once. Shared with the
        # outer Good-Turing per-arm loop -- one place to control all
        # LLM-call FN behaviour. Callers pass the resolved ArmConfig
        # built from ``good_turing.per_arm.defaults`` (+ CLI overrides);
        # we layer ArmConfig.default() under it for safety so any
        # missing field falls back to the canonical defaults.
        # When no config is supplied, we default to single-shot
        # (``k_max == 1``) so the detector behaves like the old
        # non-adaptive path. Operators who want repeated sampling
        # configure k_max via the shared Stopping rule.
        self.arm_config: ArmConfig = (
            arm_config
            if arm_config is not None
            else ArmConfig(k_max=1, k_min=1).merge_into(ArmConfig.default())
        )
        if (self.arm_config.k_max or 0) > 1:
            # Heads-up: each cluster adjudication and per-column
            # canonicalisation arm now repeats the LLM call up to
            # k_max times. With many clusters this can multiply the
            # detector's token cost; the trade is tighter
            # FN-control via the Good-Turing/Chao1 stop rule.
            logger.warning(
                "row_duplication: adaptive inner loop ON "
                "(k_max=%d, k_min=%d, stop_when=%s) -- each cluster "
                "adjudication and per-column canonicalisation may "
                "issue up to %dx more LLM calls than single-shot.",
                self.arm_config.k_max,
                self.arm_config.k_min,
                self.arm_config.stop_when,
                self.arm_config.k_max,
            )
        default_concurrency = min(8, max(1, len(self.llm.endpoints)))
        self.concurrency = (
            concurrency if concurrency and concurrency > 0 else default_concurrency
        )
        # The taxonomy may have been filtered to exclude row_duplication
        # by upstream code; access it directly from the index so we can
        # still operate.
        try:
            self._row_dup_type = self.taxonomy.get(_ROW_DUPLICATION_KEY)
        except KeyError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "RowDuplicationDetector requires the row_duplication "
                "error type in the taxonomy"
            ) from exc
        # Optional LLM-driven per-column canonicaliser. Built lazily so
        # the prompt file is only loaded when the feature is enabled.
        # Skipped entirely in ``whole_table`` mode (no clustering ->
        # no per-column canonicalisation needed).
        self._canonicalizer: Optional[KeyCanonicalizer] = None
        if (
            self.settings.canonicalize_keys
            and self.settings.detection_mode != "whole_table"
        ):
            try:
                prompt_path = Path(self.settings.canonicalize_prompt_path)
                if not prompt_path.is_absolute():
                    # Resolve relative to the project root
                    # (src/core/row_dup/detector.py -> repo/).
                    prompt_path = (
                        Path(__file__).resolve().parents[3]
                        / prompt_path
                    )
                self._canonicalizer = KeyCanonicalizer(
                    llm=self.llm,
                    model=self.model,
                    prompt_path=prompt_path,
                    reasoning_effort=self.reasoning_effort,
                    reasoning_summary=self.reasoning_summary,
                    max_cardinality=self.settings.canonicalize_max_cardinality,
                    arm_config=self.arm_config,
                )
                logger.info(
                    "row_duplication: per-column canonicalisation ON "
                    "(max_cardinality=%d, prompt=%s)",
                    self.settings.canonicalize_max_cardinality,
                    prompt_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "row_duplication: canonicaliser init failed (%s); "
                    "continuing with lexical normalisation only",
                    exc,
                )
                self._canonicalizer = None

        # Build the active clusterer registry. Phase 1 ships only
        # KeyCollisionClusterer (porting the existing exact-key logic);
        # broader strategies (blocking, MinHash LSH, embeddings) are
        # added in subsequent phases behind their own settings flags.
        # Skipped in ``whole_table`` mode -- that PoC mode hands the
        # full CSV to the LLM with zero blocking / clustering, so the
        # registry would only confuse the operator.
        self._clusterers: List[Clusterer] = []
        if self.settings.detection_mode == "whole_table":
            logger.info(
                "row_duplication: detection_mode=whole_table -- "
                "skipping clusterer registry; whole table will be "
                "fed to the LLM directly (cap=%d rows).",
                self.settings.whole_table_max_rows,
            )
            return
        self._clusterers = [
            KeyCollisionClusterer(
                max_key_subsets=self.settings.max_key_subsets,
                max_key_size=self.settings.max_key_size,
                min_dup_rate=self.settings.min_dup_rate,
                normalize_keys=self.settings.normalize_keys,
            ),
        ]
        if self.settings.blocking_enabled:
            self._clusterers.append(
                BlockingClusterer(
                    signature=self.settings.blocking_signature,
                    min_block_size=self.settings.blocking_min_block_size,
                    max_block_size=self.settings.blocking_max_block_size,
                )
            )
        if self.settings.minhash_enabled:
            self._clusterers.append(
                MinHashLSHClusterer(
                    threshold=self.settings.minhash_threshold,
                    num_perm=self.settings.minhash_num_perm,
                    shingle_size=self.settings.minhash_shingle_size,
                    min_tokens=self.settings.minhash_min_tokens,
                    max_cluster_size=self.settings.minhash_max_cluster_size,
                )
            )
        if self.settings.embedding_enabled:
            self._clusterers.append(
                EmbeddingClusterer(
                    threshold=self.settings.embedding_threshold,
                    model_name=self.settings.embedding_model,
                    min_chars=self.settings.embedding_min_chars,
                    max_cluster_size=self.settings.embedding_max_cluster_size,
                )
            )
        logger.info(
            "row_duplication: clusterer registry = [%s]",
            ", ".join(c.name for c in self._clusterers),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self, source_frames: Mapping[str, pd.DataFrame]
    ) -> List[ChunkReport]:
        """Run the detector against every (source, frame) pair.

        Returns one :class:`ChunkReport` per LLM-adjudicated collision
        group, plus zero-LLM ``deterministic`` reports when
        ``adjudicate_groups`` is disabled.
        """
        if self.settings.detection_mode == "whole_table":
            # PoC mode: skip blocking / clustering entirely. Hand the
            # whole table to one LLM call per source under the same
            # adaptive inner loop. One ChunkReport per source.
            reports: List[ChunkReport] = []
            sources = list(source_frames.items())
            progress = tqdm(
                total=len(sources),
                desc="RowDup[whole_table]",
                unit="src",
                dynamic_ncols=True,
                leave=True,
            )
            with logging_redirect_tqdm():
                for source, frame in sources:
                    progress.set_postfix_str(
                        f"{str(source)[-40:]} ({len(frame)} rows)"
                    )
                    reports.extend(self._run_whole_table(source, frame))
                    progress.update(1)
            progress.close()
            return reports
        return self._run_collision(source_frames)

    def _run_collision(
        self, source_frames: Mapping[str, pd.DataFrame]
    ) -> List[ChunkReport]:
        """Default collision-adjudication path (blocking + cluster
        enumeration + per-cluster LLM adjudication).
        """
        groups: List[_CollisionGroup] = []
        # Sources whose per-source cluster cap was hit. We attach a
        # ``cluster_budget_exhausted`` flag to every report from
        # those sources so the UI / aggregator can warn the operator.
        exhausted_sources: set = set()
        for source, frame in source_frames.items():
            digest = self.digests.get(str(source))
            try:
                batch = self._collect_clusters(source, frame, digest)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "row_duplication: group enumeration failed for %s: %s",
                    source, exc,
                )
                continue
            if batch.budget_exhausted:
                exhausted_sources.add(source)
                logger.warning(
                    "row_duplication: %s budget exhausted "
                    "(%d candidates discovered, capped to %d). "
                    "Some duplicate clusters may not be reviewed.",
                    source,
                    batch.total_discovered,
                    self.settings.max_groups_per_source,
                )
            if batch.clusters:
                logger.info(
                    "row_duplication: %s -> %d cluster(s) across "
                    "%d candidate key set(s) [strategies: %s]",
                    source,
                    len(batch.clusters),
                    len({c.key_columns for c in batch.clusters}),
                    ", ".join(sorted({
                        s for c in batch.clusters for s in c.all_strategies
                    })),
                )
            groups.extend(batch.clusters)

        if not groups:
            logger.info("row_duplication: no collision groups discovered")
            return []

        if not self.settings.adjudicate_groups:
            reports = [self._deterministic_report(g, source_frames) for g in groups]
        else:
            reports = self._adjudicate(groups, source_frames)
        # Stamp the budget-exhausted flag on every report from an
        # exhausted source so consumers don't need to reason about
        # batch state.
        if exhausted_sources:
            for r in reports:
                if r.source in exhausted_sources:
                    r.metadata["cluster_budget_exhausted"] = True
                    r.metadata["max_groups_per_source"] = (
                        self.settings.max_groups_per_source
                    )
                    # Also annotate every issue so the flag survives
                    # the flat-issues serialisation in the aggregator
                    # (where ChunkReport.metadata is dropped).
                    for issue in r.issues:
                        issue["cluster_budget_exhausted"] = True
                        issue["max_groups_per_source"] = (
                            self.settings.max_groups_per_source
                        )
        return reports

    # ------------------------------------------------------------------
    # Group enumeration
    # ------------------------------------------------------------------

    def _canonicalisation_candidate_columns(
        self,
        df: pd.DataFrame,
        digest: Optional[TableDigest],
    ) -> List[str]:
        """Pick which columns to canonicalise.

        Prefers the digest's identity-like columns; falls back to
        every object-dtype column. Columns whose distinct non-null
        cardinality exceeds
        ``settings.canonicalize_max_cardinality`` are skipped (the
        prompt would be too long for one reliable LLM call).
        """
        if digest is not None and digest.candidate_identity_columns:
            cols = [c for c in digest.candidate_identity_columns if c in df.columns]
        else:
            # Fallback: every non-numeric, non-bool, non-datetime
            # column. We can't rely on ``dtype == object`` alone
            # because modern pandas uses ``string`` /``str`` dtype
            # for text columns.
            cols = [
                c for c in df.columns
                if not (
                    pd.api.types.is_numeric_dtype(df[c])
                    or pd.api.types.is_bool_dtype(df[c])
                    or pd.api.types.is_datetime64_any_dtype(df[c])
                )
            ]
        cap = self.settings.canonicalize_max_cardinality
        out: List[str] = []
        for c in cols:
            distinct = df[c].dropna().nunique()
            if cap > 0 and distinct > cap:
                logger.info(
                    "row_duplication: skipping canonicalisation for "
                    "%s (cardinality=%d > cap=%d)",
                    c, distinct, cap,
                )
                continue
            if distinct < 2:
                continue
            out.append(c)
        return out

    def _collect_clusters(
        self,
        source: str,
        df: pd.DataFrame,
        digest: Optional[TableDigest],
    ) -> ClusterBatch:
        """Run every active clusterer, merge by row-set, apply cap."""
        if df.empty:
            return ClusterBatch(source=source)

        # Optional LLM-driven canonicalisation pre-pass. When enabled
        # we canonicalise over the digest's identity-like columns
        # BEFORE the subset-discovery phase, then build a canonicalised
        # view of the dataframe so candidate-key discovery sees the
        # folded values. Without this, columns whose only collisions
        # arise from canonicalisation (e.g. "US" / "United States")
        # would never surface as candidate keys.
        canonical_maps: Dict[str, Dict[str, str]] = {}
        df_for_subsets = df
        if self._canonicalizer is not None:
            candidate_cols = self._canonicalisation_candidate_columns(df, digest)
            for col in candidate_cols:
                raw_values = df[col].dropna().tolist()
                normalised = sorted({normalize_for_dedup(v) for v in raw_values})
                normalised = [v for v in normalised if v != ""]
                if not normalised:
                    continue
                cmap = self._canonicalizer.canonicalize_column(
                    source=source, column=col, values=normalised,
                )
                if any(v != c for v, c in cmap.items()):
                    canonical_maps[col] = cmap
            if canonical_maps:
                df_for_subsets = df.copy()
                for col, cmap in canonical_maps.items():
                    df_for_subsets[col] = df_for_subsets[col].map(
                        lambda v, m=cmap: (
                            m.get(normalize_for_dedup(v), normalize_for_dedup(v))
                            if pd.notna(v) else v
                        )
                    )

        # Run every active clusterer. ``self._clusterers`` is built in
        # ``__init__`` from settings; the registry currently contains
        # KeyCollisionClusterer, with broader strategies (blocking,
        # MinHash, embeddings) wired in later phases.
        all_clusters: List[Cluster] = []
        for clusterer in self._clusterers:
            # KeyCollisionClusterer needs the canonicalised view and
            # per-column maps; pass them via attribute mutation rather
            # than threading kwargs through the protocol so non-key
            # clusterers stay simple.
            if isinstance(clusterer, KeyCollisionClusterer):
                clusterer.canonical_frames = (
                    {source: df_for_subsets} if canonical_maps else {}
                )
                clusterer.canonical_maps = (
                    {source: canonical_maps} if canonical_maps else {}
                )
            try:
                produced = clusterer.cluster(
                    source=source, df=df, digest=digest
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "row_duplication: clusterer %s failed for %s: %s",
                    clusterer.name, source, exc,
                )
                continue
            all_clusters.extend(produced)

        # Merge across strategies (same row-set -> single cluster).
        merged = merge_clusters(all_clusters)
        kept, exhausted = rank_and_cap(
            merged, max_per_source=self.settings.max_groups_per_source
        )
        return ClusterBatch(
            source=source,
            clusters=kept,
            budget_exhausted=exhausted,
            total_discovered=len(merged),
        )

    def _collect_groups(
        self,
        source: str,
        df: pd.DataFrame,
        digest: Optional[TableDigest],
    ) -> List[_CollisionGroup]:
        """Backwards-compatible wrapper that returns just the cluster
        list (drops budget metadata). Kept for tests and external
        callers that pre-date the registry refactor."""
        return self._collect_clusters(source, df, digest).clusters

    # ------------------------------------------------------------------
    # Deterministic emit (no LLM)
    # ------------------------------------------------------------------

    def _deterministic_report(
        self,
        group: _CollisionGroup,
        source_frames: Mapping[str, pd.DataFrame],
    ) -> ChunkReport:
        identity = ", ".join(group.key_columns)
        rows = list(group.rows)
        normalized_tag = " (normalised)" if bool(group.evidence.get("normalized", False)) else ""
        issue: Dict[str, Any] = {
            "row": int(rows[0]),
            "columns": [],
            "affected_rows": [int(r) for r in rows],
            "observed": f"{len(rows)} duplicate rows",
            "explanation": (
                f"row_duplication: identity={identity}{normalized_tag}; "
                "deterministic key collision (no LLM verification)."
            ),
            "confidence": "medium",
            "verdict": "duplicate",
            "discovery_strategy": group.source_strategy,
            "discovery_strategies": list(group.all_strategies),
            "error_type": _ROW_DUPLICATION_KEY,
            "error_type_name": self._row_dup_type.name,
            "error_type_category": self._row_dup_type.category,
            "category": self._row_dup_type.category.lower(),
        }
        return ChunkReport(
            source=group.source,
            start_row=group.start_row,
            end_row=group.end_row,
            model="(deterministic)",
            prompt_name="row_duplication.deterministic",
            issues=[issue],
            error_type=_ROW_DUPLICATION_KEY,
            error_type_name=self._row_dup_type.name,
            error_type_category=self._row_dup_type.category,
        )

    # ------------------------------------------------------------------
    # LLM adjudication
    # ------------------------------------------------------------------

    def _adjudicate(
        self,
        groups: Sequence[_CollisionGroup],
        source_frames: Mapping[str, pd.DataFrame],
    ) -> List[ChunkReport]:
        results: List[Optional[ChunkReport]] = [None] * len(groups)
        progress = tqdm(
            total=len(groups),
            desc="RowDup",
            unit="group",
            dynamic_ncols=True,
            leave=True,
        )
        lock = threading.Lock()

        def _job(idx: int, group: _CollisionGroup) -> None:
            try:
                df = source_frames[group.source]
                report = self._adjudicate_group(group, df)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "row_duplication adjudication failed for group %s",
                    group,
                )
                report = self._error_report(group, str(exc))
            results[idx] = report
            with lock:
                progress.update(1)

        with logging_redirect_tqdm():
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = [
                    pool.submit(_job, i, g) for i, g in enumerate(groups)
                ]
                for fut in as_completed(futures):
                    fut.result()
        progress.close()
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # whole_table mode (PoC: zero clustering, full table -> LLM)
    # ------------------------------------------------------------------

    def _run_whole_table(
        self, source: str, df: pd.DataFrame,
    ) -> List[ChunkReport]:
        """One ChunkReport per source, built by handing the entire
        frame to the LLM and asking it to return all duplicate
        clusters in one shot.

        Driven through the same adaptive inner loop as collision
        adjudication: each run shuffles row order, species are
        ``(verdict, frozenset(affected_rows))``, and the recall-first
        union of species across runs becomes the final issue list.
        With ``arm_config.k_max == 1`` this is a single LLM call.
        """
        n_rows = int(len(df))
        max_rows = max(0, int(self.settings.whole_table_max_rows))
        if n_rows == 0:
            logger.info(
                "row_duplication[whole_table]: %s is empty; skipping", source,
            )
            return []
        if max_rows and n_rows > max_rows:
            # Honest fail: refuse to silently truncate the table.
            # Operators can either raise the cap or fall back to
            # collision_adjudication for large sources.
            logger.warning(
                "row_duplication[whole_table]: %s has %d rows > "
                "whole_table_max_rows=%d; skipping. Either raise the "
                "cap or use detection_mode=collision_adjudication.",
                source, n_rows, max_rows,
            )
            report = ChunkReport(
                source=str(source),
                start_row=0,
                end_row=n_rows,
                model=self.model,
                prompt_name="row_duplication.whole_table",
                error_type=_ROW_DUPLICATION_KEY,
                error_type_name=self._row_dup_type.name,
                error_type_category=self._row_dup_type.category,
                error=(
                    f"whole_table: {n_rows} rows exceeds cap "
                    f"{max_rows}"
                ),
            )
            report.metadata["whole_table_skipped"] = True
            return [report]

        all_rows = list(range(n_rows))
        prompt_renderer = self._make_whole_table_prompt_renderer(source, df)
        rng = _stable_rng("whole_table", str(source))
        species_to_issue: Dict[Tuple[str, frozenset], Dict[str, Any]] = {}
        last_report: List[Optional[ChunkReport]] = [None]

        def sample_fn(run_idx: int):
            order = list(all_rows)
            if run_idx > 0 and len(order) > 1:
                rng.shuffle(order)
            report, issues = self._whole_table_once(
                source=str(source),
                df=df,
                rows=order,
                prompt_renderer=prompt_renderer,
            )
            last_report[0] = report
            species: List[Tuple[str, frozenset]] = []
            for issue in issues:
                sp = (issue["verdict"], frozenset(issue["affected_rows"]))
                species.append(sp)
                species_to_issue.setdefault(sp, issue)
            return species

        result = run_adaptive(
            arm_id=f"whole_table:{source}",
            sample_fn=sample_fn,
            cfg=self.arm_config,
        )

        report = last_report[0]
        if report is None:  # pragma: no cover - defensive
            report = ChunkReport(
                source=str(source),
                start_row=0,
                end_row=n_rows,
                model=self.model,
                prompt_name="row_duplication.whole_table",
                error_type=_ROW_DUPLICATION_KEY,
                error_type_name=self._row_dup_type.name,
                error_type_category=self._row_dup_type.category,
                error="whole_table: adaptive loop produced no result",
            )
            return [report]

        # One issue per unique species, annotated with k-of-K stats.
        merged: List[Dict[str, Any]] = []
        for sp, runs in result.species_to_runs.items():
            base = dict(species_to_issue[sp])
            base["discovery_runs"] = sorted(runs)
            base["discovery_runs_total"] = result.runs_completed
            base["adaptive_stopped_reason"] = result.stopped_reason
            merged.append(base)
        report.issues = merged
        # Record this source's Good-Turing saturation as one per-CSV arm.
        # ``status`` mirrors the per-chunk / column-clustering schema:
        # the loop stopped either because a saturation criterion fired
        # (gt / chao1 / both / two_consecutive) -> "saturated", or because
        # it spent its budget ("k_max") -> "exhausted".
        if result.runs_completed > 0:
            n_species = len(result.species_to_runs)
            self.whole_table_arms.append({
                "scope_kind": "row_duplication",
                "scope_id": str(source),
                "error_type": _ROW_DUPLICATION_KEY,
                "runs_completed": result.runs_completed,
                "num_runs": result.runs_completed,
                "stopped_reason": result.stopped_reason,
                "status": (
                    "exhausted" if result.stopped_reason == "k_max"
                    else "saturated"
                ),
                "n_observed": n_species,
                "n_species_observed": n_species,
                "good_turing_missing_mass": result.gt_missing_mass,
                "chao1_estimated_missed": result.chao1_missed,
            })
        return [report]

    def _whole_table_once(
        self,
        *,
        source: str,
        df: pd.DataFrame,
        rows: Sequence[int],
        prompt_renderer: "_PromptRenderer",
    ) -> Tuple[ChunkReport, List[Dict[str, Any]]]:
        """Single LLM whole-table pass with the given row order."""
        n_rows = int(len(df))
        all_rows = list(range(n_rows))
        prompt = prompt_renderer(rows)
        report = ChunkReport(
            source=str(source),
            start_row=0,
            end_row=n_rows,
            model=self.model,
            prompt_name="row_duplication.whole_table",
            error_type=_ROW_DUPLICATION_KEY,
            error_type_name=self._row_dup_type.name,
            error_type_category=self._row_dup_type.category,
        )
        response: LLMResponse = self.llm.query(
            self.model,
            prompt,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            system_prompt=self.taxonomy.system_prompt,
        )
        report.endpoint = response.endpoint
        report.attempts = response.attempts
        report.raw_text = response.text
        report.reasoning_summary = response.reasoning_summary
        report.error = response.error
        if not response.ok:
            return report, []

        parsed, parse_err = _parse_verdicts(response.text)
        report.parse_error = parse_err
        if not parsed:
            return report, []

        emitted: List[Dict[str, Any]] = []
        for verdict in parsed:
            label = str(verdict.get("verdict", "")).strip().lower()
            if label not in {"duplicate", "near_duplicate"}:
                continue
            rows_subset = _extract_verdict_rows(verdict, all_rows)
            if not rows_subset:
                continue
            observed = verdict.get("observed") or (
                f"{len(rows_subset)} {label.replace('_', ' ')} rows"
            )
            explanation = verdict.get("explanation") or (
                "row_duplication: whole-table LLM pass"
            )
            confidence = verdict.get("confidence") or (
                "high" if label == "duplicate" else "medium"
            )
            issue: Dict[str, Any] = {
                "row": rows_subset[0],
                "columns": [],
                "affected_rows": rows_subset,
                "observed": str(observed),
                "explanation": str(explanation),
                "confidence": str(confidence),
                "verdict": label,
                "discovery_strategy": "whole_table",
                "discovery_strategies": ["whole_table"],
                "error_type": _ROW_DUPLICATION_KEY,
                "error_type_name": self._row_dup_type.name,
                "error_type_category": self._row_dup_type.category,
                "category": self._row_dup_type.category.lower(),
            }
            emitted.append(issue)
        report.issues = emitted
        return report, emitted

    def _make_whole_table_prompt_renderer(
        self, source: str, df: pd.DataFrame,
    ) -> "_PromptRenderer":
        """Return a renderer that maps row order -> full prompt for
        the whole-table mode. The static scaffold (description,
        digest, schema) is rendered once; only the row markdown is
        re-rendered between adaptive runs.
        """
        et = self._row_dup_type
        n_rows = int(len(df))
        digest = self.digests.get(str(source))
        digest_block = (
            digest.to_prompt_block() if digest is not None else "(unavailable)"
        )
        prefix = (
            f"# {et.name} - WHOLE-TABLE PASS\n\n"
            f"You are reviewing the ENTIRE table from source `{source}`\n"
            f"({n_rows} rows). Find every duplicate / near-duplicate\n"
            f"cluster *without* any pre-clustering hints. Each cluster\n"
            f"is a set of rows that describe the same real-world\n"
            f"entity.\n\n"
            f"## Description\n{et.description.strip()}\n\n"
            f"## Reference example\n"
            f"{(et.example.strip() or '(none)')}\n\n"
            f"## Detection rules\n{et.notes.strip()}\n\n"
            f"## Table overview\n{digest_block}\n\n"
            f"## Full table ({n_rows} rows)\n"
        )
        suffix = (
            "\n\n## Your task\n"
            "Return EVERY duplicate / near-duplicate cluster as a\n"
            "separate issue. Rows that have no duplicate elsewhere in\n"
            "the table must be omitted from `affected_rows`. If the\n"
            "table contains NO duplicates at all, return\n"
            '`{"issues": []}`.\n\n'
            "Schema:\n\n"
            "```json\n"
            "{\n"
            '  "issues": [\n'
            "    {\n"
            '      "verdict": "duplicate" | "near_duplicate",\n'
            '      "affected_rows": [<row indices in the table>],\n'
            '      "confidence": "high" | "medium" | "low",\n'
            '      "observed": "<short human-readable summary>",\n'
            '      "explanation": "<why - cite columns or values>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "Verdict semantics:\n"
            "- `duplicate`: rows are interchangeable - same entity,\n"
            "  same facts, no meaningful difference.\n"
            "- `near_duplicate`: rows describe the same entity but\n"
            "  differ on minor fields (formatting, casing,\n"
            "  abbreviations, fresher information). When in doubt,\n"
            "  prefer `near_duplicate` so a human reviewer sees it.\n\n"
            "Recall is more important than precision: it is better to\n"
            "flag two rows as `near_duplicate` and let a human dismiss\n"
            "them than to silently drop a real duplicate.\n"
        )

        def render(rows: Optional[Sequence[int]]) -> str:
            order = list(rows) if rows is not None else list(range(n_rows))
            rows_md = _render_rows_markdown(df, order)
            return f"{prefix}{rows_md}{suffix}"

        return render

    def _adjudicate_group(
        self, group: _CollisionGroup, df: pd.DataFrame
    ) -> ChunkReport:
        """Adjudicate one cluster via the shared adaptive inner loop.

        The cluster is sampled up to ``self.arm_config.k_max`` times
        with shuffled row order; species
        ``(verdict, frozenset(affected_rows))`` are unioned across
        runs (recall-first). With ``k_max == 1`` this is a single LLM
        call, identical to the old non-adaptive path.
        """
        # Stable per-cluster RNG so repeated detector runs on the same
        # data are reproducible.
        rng = _stable_rng(
            "adjudicate",
            str(group.source),
            int(group.start_row),
            tuple(group.key_columns),
        )
        # Render the row-independent scaffold once; ``sample_fn``
        # only re-renders the row markdown per shuffled run.
        prompt_renderer = self._make_prompt_renderer(group, df)
        species_to_issue: Dict[Tuple[str, frozenset], Dict[str, Any]] = {}
        last_report: List[Optional[ChunkReport]] = [None]

        def sample_fn(run_idx: int):
            rows = list(group.rows)
            if run_idx > 0 and len(rows) > 1:
                rng.shuffle(rows)
            report, issues = self._adjudicate_group_once(
                group, df, rows, prompt_renderer=prompt_renderer,
            )
            last_report[0] = report
            species: List[Tuple[str, frozenset]] = []
            for issue in issues:
                sp = (issue["verdict"], frozenset(issue["affected_rows"]))
                species.append(sp)
                # Keep the FIRST observation of each species so its
                # explanation/observed text is from a real LLM call.
                species_to_issue.setdefault(sp, issue)
            return species

        arm_id = f"{group.source}:{group.start_row}:{','.join(group.key_columns)}"
        result = run_adaptive(
            arm_id=arm_id, sample_fn=sample_fn, cfg=self.arm_config,
        )

        report = last_report[0]
        if report is None:  # pragma: no cover - defensive
            report = self._error_report(group, "adaptive loop produced no result")
            return report

        # Build the merged issue list. Each species emits one issue,
        # annotated with which runs surfaced it and the total runs
        # completed - so consumers can compute "k of K agreement".
        merged: List[Dict[str, Any]] = []
        for sp, runs in result.species_to_runs.items():
            base = dict(species_to_issue[sp])
            base["discovery_runs"] = sorted(runs)
            base["discovery_runs_total"] = result.runs_completed
            base["adaptive_stopped_reason"] = result.stopped_reason
            merged.append(base)
        report.issues = merged
        return report

    def _adjudicate_group_once(
        self,
        group: _CollisionGroup,
        df: pd.DataFrame,
        rows: Sequence[int],
        *,
        prompt_renderer: Optional["_PromptRenderer"] = None,
    ) -> Tuple[ChunkReport, List[Dict[str, Any]]]:
        """Single LLM adjudication pass over ``group`` with the given row order.

        ``prompt_renderer``, if provided, is reused across runs in the
        adaptive inner loop so the row-independent prompt scaffold
        (digest, evidence, schema description) is rendered once.

        Returns the populated :class:`ChunkReport` and the parsed
        issue list (also assigned onto ``report.issues``).
        """
        if prompt_renderer is None:
            prompt_renderer = self._make_prompt_renderer(group, df)
        prompt = prompt_renderer(rows)
        report = ChunkReport(
            source=group.source,
            start_row=group.start_row,
            end_row=group.end_row,
            model=self.model,
            prompt_name=f"row_duplication.adjudicate.{','.join(group.key_columns)}",
            error_type=_ROW_DUPLICATION_KEY,
            error_type_name=self._row_dup_type.name,
            error_type_category=self._row_dup_type.category,
        )
        response: LLMResponse = self.llm.query(
            self.model,
            prompt,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            system_prompt=self.taxonomy.system_prompt,
        )
        report.endpoint = response.endpoint
        report.attempts = response.attempts
        report.raw_text = response.text
        report.reasoning_summary = response.reasoning_summary
        report.error = response.error

        if not response.ok:
            return report, []

        # Multi-issue parser: a single cluster can split into several
        # verdict groups (e.g. rows {0,1,2} = duplicate, rows {3,4} =
        # near_duplicate, others dropped). When the LLM returns just
        # one verdict object we still emit it as one issue.
        parsed, parse_err = _parse_verdicts(response.text)
        report.parse_error = parse_err
        if not parsed:
            return report, []

        identity = ", ".join(group.key_columns)
        normalized_tag = (
            " (normalised)"
            if bool(group.evidence.get("normalized", False)) else ""
        )
        all_rows = [int(r) for r in group.rows]
        emitted: List[Dict[str, Any]] = []
        for verdict in parsed:
            label = str(verdict.get("verdict", "")).strip().lower()
            if label not in {"duplicate", "near_duplicate"}:
                # "distinct" -> drop. Anything we don't recognise -> drop.
                continue
            rows_subset = _extract_verdict_rows(verdict, all_rows)
            observed = verdict.get("observed") or (
                f"{len(rows_subset)} {label.replace('_', ' ')} rows"
            )
            explanation = verdict.get("explanation") or (
                f"row_duplication: identity={identity}{normalized_tag}"
            )
            confidence = verdict.get("confidence") or (
                "high" if label == "duplicate" else "medium"
            )
            issue: Dict[str, Any] = {
                "row": rows_subset[0],
                "columns": [],
                "affected_rows": rows_subset,
                "observed": str(observed),
                "explanation": str(explanation),
                "confidence": str(confidence),
                "verdict": label,
                "discovery_strategy": group.source_strategy,
                "discovery_strategies": list(group.all_strategies),
                "error_type": _ROW_DUPLICATION_KEY,
                "error_type_name": self._row_dup_type.name,
                "error_type_category": self._row_dup_type.category,
                "category": self._row_dup_type.category.lower(),
            }
            emitted.append(issue)
        report.issues = emitted
        return report, emitted

    def _error_report(self, group: _CollisionGroup, error: str) -> ChunkReport:
        return ChunkReport(
            source=group.source,
            start_row=group.start_row,
            end_row=group.end_row,
            model=self.model,
            prompt_name=f"row_duplication.adjudicate.{','.join(group.key_columns)}",
            error_type=_ROW_DUPLICATION_KEY,
            error_type_name=self._row_dup_type.name,
            error_type_category=self._row_dup_type.category,
            error=error,
        )

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _make_prompt_renderer(
        self, group: _CollisionGroup, df: pd.DataFrame,
    ) -> "_PromptRenderer":
        """Return a renderer that maps row order -> full prompt string.

        All row-independent parts (description, evidence, digest,
        schema, instructions) are rendered once and captured in a
        closure so the adaptive inner loop only re-renders the row
        markdown between runs.
        """
        et = self._row_dup_type
        identity = ", ".join(group.key_columns)
        normalized_tag = (
            " (normalised match)"
            if bool(group.evidence.get("normalized", False)) else ""
        )
        digest = self.digests.get(str(group.source))
        digest_block = (
            digest.to_prompt_block() if digest is not None else "(unavailable)"
        )
        # Render the discovery strategy + evidence so the LLM knows
        # how this cluster was surfaced. High-confidence strategies
        # (key_collision) deserve more trust; broad ones (blocking,
        # minhash_lsh, embedding) carry more false positives and the
        # LLM should be willing to mark a subset as `distinct`.
        strategies = ", ".join(group.all_strategies) or group.source_strategy
        evidence_lines = [
            f"- **{k}**: {v}" for k, v in sorted(group.evidence.items())
        ]
        evidence_block = (
            "\n".join(evidence_lines) if evidence_lines else "(none)"
        )
        prefix = (
            f"# {et.name} - CLUSTER ADJUDICATION\n\n"
            f"You are reviewing ONE candidate duplicate cluster from source\n"
            f"`{group.source}`. The cluster was surfaced by **{strategies}**\n"
            f"with key columns **{identity}**{normalized_tag}.\n\n"
            f"## Description\n{et.description.strip()}\n\n"
            f"## Reference example\n"
            f"{(et.example.strip() or '(none)')}\n\n"
            f"## Detection rules\n{et.notes.strip()}\n\n"
            f"## Discovery evidence\n{evidence_block}\n\n"
            f"## Table overview\n{digest_block}\n\n"
            f"## Candidate cluster ({len(group.rows)} rows)\n"
            f"Row indices: {list(group.rows)}\n\n"
        )
        suffix = (
            "\n\n## Your task\n"
            "Decide which rows in this cluster are duplicates of the\n"
            "same real-world entity. The cluster may be **mixed**:\n"
            "some rows duplicate one entity, others duplicate another,\n"
            "and some may not be duplicates at all. Split as needed.\n\n"
            "Return EXACTLY one JSON object with this schema:\n\n"
            "```json\n"
            "{\n"
            '  "issues": [\n'
            "    {\n"
            '      "verdict": "duplicate" | "near_duplicate",\n'
            '      "affected_rows": [<row indices, subset of the cluster>],\n'
            '      "confidence": "high" | "medium" | "low",\n'
            '      "observed": "<short human-readable summary>",\n'
            '      "explanation": "<why - cite columns or values>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "Verdict semantics:\n"
            "- `duplicate`: the rows are interchangeable - same entity,\n"
            "  same facts, no meaningful difference.\n"
            "- `near_duplicate`: the rows describe the same entity but\n"
            "  differ on minor fields (formatting, casing, abbreviations,\n"
            "  or one row has slightly more / fresher information).\n"
            "  When in doubt between duplicate and near_duplicate, prefer\n"
            "  `near_duplicate` so a human reviewer sees it.\n"
            "- Rows that are NOT duplicates of any other row in the\n"
            "  cluster: simply omit them from every `affected_rows`\n"
            "  list. If NONE of the rows are duplicates, return\n"
            '   `{"issues": []}`.\n\n'
            "Recall is more important than precision: it is better to\n"
            "flag two rows as `near_duplicate` and let a human dismiss\n"
            "them than to silently drop a real duplicate.\n"
        )

        def render(rows: Optional[Sequence[int]]) -> str:
            rows_for_render = (
                list(rows) if rows is not None else list(group.rows)
            )
            rows_md = _render_rows_markdown(df, rows_for_render)
            return f"{prefix}{rows_md}{suffix}"

        return render

    def _render_prompt(
        self,
        group: _CollisionGroup,
        df: pd.DataFrame,
        *,
        rows: Optional[Sequence[int]] = None,
    ) -> str:
        """Single-shot prompt rendering. Equivalent to building a\n        renderer and immediately invoking it; kept for callers that\n        only need one prompt.\n        """
        return self._make_prompt_renderer(group, df)(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A renderer takes an optional row ordering and returns the full
# prompt. ``None`` falls back to the cluster's natural row order.
_PromptRenderer = Callable[[Optional[Sequence[int]]], str]


def _stable_rng(*keys: object) -> random.Random:
    """Return a ``random.Random`` seeded from ``keys``.

    Used to derive reproducible per-arm RNGs (one per cluster, one
    per (source, column)) so repeated detector runs on the same data
    shuffle inputs identically.
    """
    return random.Random(hash(tuple(keys)))


def _extract_verdict_rows(
    verdict: Mapping[str, Any], cluster_rows: Sequence[int],
) -> List[int]:
    """Resolve the row subset a verdict applies to.

    The LLM may supply ``affected_rows`` (or legacy ``rows``) as a
    subset of the cluster; we intersect with ``cluster_rows`` so we
    never invent indices the cluster did not contain. On any miss
    (missing key, empty list, no overlap, malformed entries) we fall
    back to the entire cluster.
    """
    cluster_set = {int(r) for r in cluster_rows}
    llm_rows = verdict.get("affected_rows") or verdict.get("rows")
    if isinstance(llm_rows, list) and llm_rows:
        rows_subset = sorted(
            {int(r) for r in llm_rows if isinstance(r, (int, float))}
            & cluster_set
        )
        if rows_subset:
            return rows_subset
    return sorted(cluster_set)


_VERDICT_KEYS = {"verdict", "duplicate", "label", "decision"}


def _parse_verdict(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Best-effort *single-verdict* parser (back-compat).

    Returns the first parsed verdict object. New code should use
    :func:`_parse_verdicts`, which returns the full list and supports
    mixed clusters where the LLM splits a cluster into multiple
    verdict groups.
    """
    verdicts, err = _parse_verdicts(text)
    if not verdicts:
        return None, err
    return verdicts[0], None


def _parse_verdicts(
    text: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse a (possibly multi-issue) verdict payload.

    Accepts either:

    1. ``{"issues": [{"verdict": ..., "affected_rows": ...}, ...]}``
       - the new multi-issue schema. Each entry becomes one verdict
       and supports an explicit ``affected_rows`` subset.
    2. ``{"verdict": ..., "confidence": ..., ...}`` - the legacy
       single-verdict schema. Treated as a one-element list.
    3. ``[{"verdict": ...}, ...]`` - a bare JSON array of verdicts.
    """
    issues, err = parse_llm_json(text)
    if not issues:
        return [], err or "no JSON object found"
    out: List[Dict[str, Any]] = []
    for payload in issues:
        if not isinstance(payload, Mapping):
            continue
        norm: Dict[str, Any] = {}
        for k, v in payload.items():
            if k.lower() in _VERDICT_KEYS:
                norm["verdict"] = str(v).strip().lower()
            else:
                norm[k] = v
        if "verdict" in norm:
            out.append(norm)
    if not out:
        return [], "no verdict objects found"
    return out, None


def _render_rows_markdown(
    df: pd.DataFrame,
    rows: Sequence[int],
    *,
    max_cell_chars: int = 120,
    max_cols: int = 24,
) -> str:
    """Render the listed rows as a Markdown table with a leading
    ``__row__`` column carrying the original row numbers."""
    if not rows:
        return "(empty)"
    cols = list(df.columns)[:max_cols]
    sub = df.loc[list(rows), cols].copy()
    sub = sub.astype(object).map(
        lambda v: _truncate(v, max_cell_chars)
    )
    sub.insert(0, "__row__", list(rows))
    try:
        return sub.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        # Fallback if tabulate is unavailable.
        header = "| " + " | ".join(["__row__"] + list(cols)) + " |"
        sep = "|" + "|".join(["---"] * (1 + len(cols))) + "|"
        body = []
        for _, r in sub.iterrows():
            cells = [str(r["__row__"])] + [str(r[c]) for c in cols]
            body.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep, *body])


def _truncate(value: object, max_chars: int) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value)
    if len(s) > max_chars:
        return s[: max_chars - 1] + "\u2026"
    return s
