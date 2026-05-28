"""Clusterer protocol + implementations for row-duplication detection.

The row-duplication pipeline (see :mod:`src.core.row_duplication_detector`)
is split in two: a *clustering* phase that produces candidate row
groups likely to be duplicates, and an *adjudication* phase where the
LLM rules on each cluster.

Different clustering strategies have different precision/recall
trade-offs:

============================  =======  ======  ===========================
Strategy                       Recall   Cost    Catches
============================  =======  ======  ===========================
``KeyCollisionClusterer``      low      free    exact / canonicalised key
                                               collisions on identity
                                               columns. High precision -
                                               always on.
``BlockingClusterer``          high     free    rows sharing token
                                               signatures on key
                                               columns. Classic ER
                                               blocking.
``MinHashLSHClusterer``        high     low     row-level Jaccard
                                               near-duplicates via LSH.
``EmbeddingClusterer``         highest  $$      semantic / paraphrase
                                               near-duplicates via
                                               sentence-transformers.
============================  =======  ======  ===========================

Each implementation conforms to :class:`Clusterer`. A registry returns
the active clusterers based on :class:`AdjudicatorSettings`.

Output: a list of :class:`Cluster` records, deduplicated across
strategies via :func:`merge_clusters` so the same row-set discovered
by two strategies is adjudicated once (with all evidence pinned).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

import pandas as pd

from .priors import (
    enumerate_collision_groups,
    find_duplicate_subsets,
    find_duplicate_subsets_normalized,
    normalize_for_dedup,
)
from ..table_digest import TableDigest

logger = logging.getLogger(__name__)


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cluster:
    """One candidate row-cluster for adjudication.

    ``source_strategy`` is the originating clusterer's ``name`` (e.g.
    ``"key_collision"``, ``"blocking"``). When the same row-set is
    surfaced by multiple strategies the merged cluster keeps every
    contributing strategy in :attr:`source_strategies`.
    """

    source: str
    rows: Tuple[int, ...]
    key_columns: Tuple[str, ...]
    source_strategy: str
    # Supplementary strategies that independently produced this cluster.
    source_strategies: Tuple[str, ...] = ()
    # Free-form metadata, intended for the adjudicator prompt and the
    # reviewer UI (e.g. ``{"normalized": True, "block_signature":
    # "acme|nyc"}``).
    evidence: Mapping[str, object] = field(default_factory=dict)
    # Confidence label assigned by the clusterer. ``high`` is the
    # default for exact-key collisions; ``medium``/``low`` mark
    # broader candidate strategies.
    confidence: str = "medium"

    @property
    def fingerprint(self) -> Tuple[int, ...]:
        """Stable identity key (sorted row indices)."""
        return tuple(sorted(self.rows))

    @property
    def start_row(self) -> int:
        return min(self.rows)

    @property
    def end_row(self) -> int:
        # Mirror ChunkReport's exclusive-end convention.
        return max(self.rows) + 1

    @property
    def all_strategies(self) -> Tuple[str, ...]:
        """Primary + supplementary strategies as one tuple."""
        return (self.source_strategy, *self.source_strategies)


@dataclass
class ClusterBatch:
    """Per-source clustering output, including budget metadata."""

    source: str
    clusters: List[Cluster] = field(default_factory=list)
    # True when the per-source ``max_groups_per_source`` cap was hit
    # before all candidate clusters could be queued. The UI uses this
    # to flag sources where the operator may have missed duplicates.
    budget_exhausted: bool = False
    # Total number of clusters discovered before the cap was applied.
    total_discovered: int = 0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Clusterer(Protocol):
    """A row-clustering strategy.

    Implementations should be **stateless** with respect to per-call
    inputs; configuration is captured at construction time.
    """

    name: str

    def cluster(
        self,
        *,
        source: str,
        df: pd.DataFrame,
        digest: Optional[TableDigest],
    ) -> List[Cluster]:
        ...


# ---------------------------------------------------------------------------
# Merging across strategies
# ---------------------------------------------------------------------------


def merge_clusters(clusters: Sequence[Cluster]) -> List[Cluster]:
    """Deduplicate clusters by row-set fingerprint.

    Two clusters with the same sorted row indices are collapsed into
    one. The first occurrence wins on identity; supplementary
    strategies are accumulated into ``source_strategies`` and
    evidence dicts are merged. Confidence escalates: if any
    contributing clusterer reports ``high`` the merged cluster is
    ``high``.
    """
    by_fp: Dict[Tuple[int, ...], Cluster] = {}
    for c in clusters:
        fp = c.fingerprint
        existing = by_fp.get(fp)
        if existing is None:
            by_fp[fp] = c
            continue
        # Merge: union evidence, append strategy, escalate confidence.
        merged_evidence = dict(existing.evidence)
        for k, v in c.evidence.items():
            merged_evidence.setdefault(k, v)
        new_strategies = tuple(
            s for s in c.all_strategies if s not in existing.all_strategies
        )
        confidence = _max_confidence(existing.confidence, c.confidence)
        # Prefer the longer / more specific key-column tuple.
        key_columns = (
            existing.key_columns
            if len(existing.key_columns) >= len(c.key_columns)
            else c.key_columns
        )
        by_fp[fp] = Cluster(
            source=existing.source,
            rows=existing.rows,
            key_columns=key_columns,
            source_strategy=existing.source_strategy,
            source_strategies=existing.source_strategies + new_strategies,
            evidence=merged_evidence,
            confidence=confidence,
        )
    return list(by_fp.values())


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _max_confidence(a: str, b: str) -> str:
    return a if _CONFIDENCE_RANK.get(a, 0) >= _CONFIDENCE_RANK.get(b, 0) else b


# ---------------------------------------------------------------------------
# Cluster ranking + budget
# ---------------------------------------------------------------------------


def rank_and_cap(
    clusters: Sequence[Cluster], *, max_per_source: int
) -> Tuple[List[Cluster], bool]:
    """Sort clusters and apply the per-source cap.

    Sort order: confidence DESC, group size DESC, smallest row index
    ASC, then key columns. Returns ``(kept, exhausted)`` where
    ``exhausted`` is True when the cap removed at least one cluster.
    """
    ranked = sorted(
        clusters,
        key=lambda c: (
            -_CONFIDENCE_RANK.get(c.confidence, 0),
            -len(c.rows),
            c.start_row,
            c.key_columns,
        ),
    )
    exhausted = False
    if max_per_source > 0 and len(ranked) > max_per_source:
        exhausted = True
        ranked = ranked[:max_per_source]
    return ranked, exhausted


# ---------------------------------------------------------------------------
# KeyCollisionClusterer (current behaviour, ported)
# ---------------------------------------------------------------------------


@dataclass
class KeyCollisionClusterer:
    """Exact / canonicalised key-collision clusterer.

    Runs :func:`find_duplicate_subsets` (and, when ``normalize_keys``
    is set, :func:`find_duplicate_subsets_normalized`) over the input
    frame, then enumerates every collision group via
    :func:`enumerate_collision_groups`. This is the high-precision
    backbone of the pipeline: clusters from this strategy carry
    ``confidence="high"`` and are always emitted.

    The optional ``canonical_maps`` argument lets the caller plug in
    LLM-built ``value -> canonical`` maps from
    :class:`KeyCanonicalizer` so semantic variants ("US"/"USA") are
    folded before the collision scan.
    """

    name: str = "key_collision"
    # Configuration mirroring AdjudicatorSettings.
    max_key_subsets: int = 4
    max_key_size: int = 2
    min_dup_rate: float = 0.0
    normalize_keys: bool = True
    # Optional per-source canonical maps: ``{source -> {col -> {val -> canon}}}``.
    canonical_maps: Mapping[str, Mapping[str, Mapping[str, str]]] = field(
        default_factory=dict
    )
    # Optional canonicalised dataframe override per source. When the
    # caller has already applied a ``df_for_subsets`` transformation
    # (because canonicalisation requires it), pass it here so subset
    # discovery sees the canonicalised column values.
    canonical_frames: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    def cluster(
        self,
        *,
        source: str,
        df: pd.DataFrame,
        digest: Optional[TableDigest],
    ) -> List[Cluster]:
        if df.empty:
            return []
        df_for_subsets = self.canonical_frames.get(source, df)
        col_canon = self.canonical_maps.get(source, {})

        subsets = list(
            find_duplicate_subsets(
                df_for_subsets,
                digest=digest,
                max_subsets=self.max_key_subsets,
                max_subset_size=self.max_key_size,
            )
        )
        if self.normalize_keys:
            extra = find_duplicate_subsets_normalized(
                df_for_subsets,
                digest=digest,
                max_subsets=self.max_key_subsets,
                max_subset_size=self.max_key_size,
            )
            seen_cols = {s.columns for s in subsets}
            for s in extra:
                if s.columns not in seen_cols:
                    subsets.append(s)
        subsets = [s for s in subsets if s.dup_rate >= self.min_dup_rate]
        if not subsets:
            return []

        seen_groups: Set[Tuple[int, ...]] = set()
        out: List[Cluster] = []
        for subset in subsets:
            modes = (False, True) if self.normalize_keys else (False,)
            for normalize in modes:
                cmaps = col_canon if (normalize and col_canon) else None
                rows_list = enumerate_collision_groups(
                    df,
                    list(subset.columns),
                    min_group_size=2,
                    normalize=normalize,
                    canonical_maps=cmaps,
                )
                for rows in rows_list:
                    fp = tuple(sorted(rows))
                    if fp in seen_groups:
                        continue
                    seen_groups.add(fp)
                    evidence = {
                        "normalized": normalize,
                        "key_columns": list(subset.columns),
                        "dup_rate": float(subset.dup_rate),
                    }
                    if cmaps:
                        evidence["canonicalised"] = sorted(
                            c for c in subset.columns if c in cmaps
                        )
                    out.append(
                        Cluster(
                            source=source,
                            rows=tuple(rows),
                            key_columns=tuple(subset.columns),
                            source_strategy=self.name,
                            evidence=evidence,
                            confidence="high",
                        )
                    )
        return out


# ---------------------------------------------------------------------------
# Helpers for token / signature-based clusterers
# ---------------------------------------------------------------------------


def _row_text(df: pd.DataFrame, row_idx: int, columns: Sequence[str]) -> str:
    """Concatenate the row's column values into a normalised text blob."""
    parts: List[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        v = df.iat[row_idx, df.columns.get_loc(col)]
        if pd.isna(v):
            continue
        parts.append(normalize_for_dedup(v))
    return " ".join(p for p in parts if p)


def _tokenise(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _candidate_text_columns(
    df: pd.DataFrame, digest: Optional[TableDigest]
) -> List[str]:
    """Pick text-like columns to feed signature/embedding clusterers.

    Prefers the digest's identity-like columns; falls back to every
    non-numeric, non-bool, non-datetime column.
    """
    if digest is not None and digest.candidate_identity_columns:
        cols = [c for c in digest.candidate_identity_columns if c in df.columns]
        if cols:
            return cols
    return [
        c for c in df.columns
        if not (
            pd.api.types.is_numeric_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])
            or pd.api.types.is_datetime64_any_dtype(df[c])
        )
    ]


# ---------------------------------------------------------------------------
# BlockingClusterer (token-signature blocking, no LLM)
# ---------------------------------------------------------------------------


@dataclass
class BlockingClusterer:
    """Classic ER blocking: cluster rows that share a *signature*.

    A signature is a deterministic, low-cardinality string derived
    from a row's key columns. Two rows with the same signature land
    in the same block. The default signature is ``"<first_token>"``
    of each candidate-key column joined with ``"|"`` -- enough to
    catch ``"Acme Corp"`` / ``"Acme Corporation"`` / ``"ACME Inc."``
    on a single column without exploding cardinality.

    Parameters
    ----------
    name
        Strategy name written into ``Cluster.source_strategy``.
    min_block_size
        Skip blocks smaller than this. Default ``2`` (one duplicate
        pair per block).
    max_block_size
        Skip blocks larger than this. Very large blocks are usually
        the result of a generic token (e.g. ``"the"``) and produce
        adjudication noise.
    signature
        ``"first_token"`` (default) or ``"first_char"``. The latter
        is much coarser - reserved for tiny tables.
    """

    name: str = "blocking"
    min_block_size: int = 2
    max_block_size: int = 50
    signature: str = "first_token"

    def cluster(
        self,
        *,
        source: str,
        df: pd.DataFrame,
        digest: Optional[TableDigest],
    ) -> List[Cluster]:
        if df.empty:
            return []
        cols = _candidate_text_columns(df, digest)
        if not cols:
            return []
        signatures: Dict[Tuple[str, ...], List[int]] = {}
        for ridx in range(len(df)):
            sig: List[str] = []
            for col in cols:
                v = df.iat[ridx, df.columns.get_loc(col)]
                if pd.isna(v):
                    sig.append("")
                    continue
                norm = normalize_for_dedup(v)
                if not norm:
                    sig.append("")
                    continue
                if self.signature == "first_char":
                    sig.append(norm[:1])
                else:  # first_token
                    toks = _tokenise(norm)
                    sig.append(toks[0] if toks else "")
            sig_tup = tuple(sig)
            # Drop signatures that are entirely empty / single-empty -
            # those land everything in one degenerate block.
            if not any(sig_tup):
                continue
            signatures.setdefault(sig_tup, []).append(ridx)

        out: List[Cluster] = []
        for sig_tup, rows in signatures.items():
            n = len(rows)
            if n < self.min_block_size or n > self.max_block_size:
                continue
            evidence = {
                "block_signature": "|".join(sig_tup),
                "block_columns": list(cols),
                "block_size": n,
            }
            out.append(
                Cluster(
                    source=source,
                    rows=tuple(rows),
                    key_columns=tuple(cols),
                    source_strategy=self.name,
                    evidence=evidence,
                    confidence="low",
                )
            )
        return out


# ---------------------------------------------------------------------------
# MinHashLSHClusterer (datasketch)
# ---------------------------------------------------------------------------


@dataclass
class MinHashLSHClusterer:
    """Row-level Jaccard near-duplicate clusterer using MinHash + LSH.

    Each row is tokenised over its candidate text columns; a MinHash
    sketch is computed and inserted into a Locality-Sensitive Hashing
    index. Pairs of rows hashing into the same bucket form candidate
    clusters; we then expand each seed row's neighbours into a
    transitive cluster via union-find.

    Parameters
    ----------
    name
        Strategy name.
    threshold
        Jaccard threshold. ``0.7`` is a reasonable default for
        textual rows; lower for higher recall, higher for higher
        precision.
    num_perm
        MinHash permutations. ``128`` is the standard datasketch
        default; tradeoff between accuracy and CPU.
    shingle_size
        Token n-gram size for the sketch input. ``1`` (unigrams)
        is good for short rows; ``2`` for prose-like content.
    min_tokens
        Skip rows with fewer than this many tokens (sketch is
        unreliable).
    max_cluster_size
        Cap the size of any one returned cluster (very large
        clusters are usually false-positive snowballs).
    """

    name: str = "minhash_lsh"
    threshold: float = 0.7
    num_perm: int = 128
    shingle_size: int = 1
    min_tokens: int = 3
    max_cluster_size: int = 50

    def cluster(
        self,
        *,
        source: str,
        df: pd.DataFrame,
        digest: Optional[TableDigest],
    ) -> List[Cluster]:
        if df.empty:
            return []
        try:
            from datasketch import MinHash, MinHashLSH  # type: ignore
        except ImportError:
            logger.warning(
                "row_duplication: datasketch not installed; skipping "
                "MinHashLSHClusterer for %s. Install with "
                "`pip install datasketch`.",
                source,
            )
            return []

        cols = _candidate_text_columns(df, digest)
        if not cols:
            return []
        # Build sketches.
        sketches: Dict[int, "MinHash"] = {}
        token_sets: Dict[int, Set[str]] = {}
        for ridx in range(len(df)):
            text = _row_text(df, ridx, cols)
            tokens = _tokenise(text)
            if len(tokens) < self.min_tokens:
                continue
            if self.shingle_size > 1:
                shingles = {
                    " ".join(tokens[i : i + self.shingle_size])
                    for i in range(len(tokens) - self.shingle_size + 1)
                }
            else:
                shingles = set(tokens)
            if not shingles:
                continue
            mh = MinHash(num_perm=self.num_perm)
            for s in shingles:
                mh.update(s.encode("utf-8"))
            sketches[ridx] = mh
            token_sets[ridx] = shingles
        if len(sketches) < 2:
            return []
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for ridx, mh in sketches.items():
            lsh.insert(str(ridx), mh)

        # Union-find: every row shares a cluster with each of its LSH
        # neighbours. Transitive closure means A~B and B~C implies
        # {A,B,C}.
        parent = {r: r for r in sketches}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for ridx, mh in sketches.items():
            neighbours = lsh.query(mh)
            for n_str in neighbours:
                n = int(n_str)
                if n == ridx:
                    continue
                union(ridx, n)

        # Materialise clusters.
        groups: Dict[int, List[int]] = {}
        for r in sketches:
            groups.setdefault(find(r), []).append(r)
        out: List[Cluster] = []
        for root, rows in groups.items():
            if len(rows) < 2 or len(rows) > self.max_cluster_size:
                continue
            rows_sorted = sorted(rows)
            # Compute average pairwise Jaccard for evidence.
            jacs: List[float] = []
            for i in range(len(rows_sorted)):
                for j in range(i + 1, len(rows_sorted)):
                    a = token_sets[rows_sorted[i]]
                    b = token_sets[rows_sorted[j]]
                    if not a or not b:
                        continue
                    jac = len(a & b) / max(1, len(a | b))
                    jacs.append(jac)
            avg_jac = sum(jacs) / len(jacs) if jacs else 0.0
            evidence = {
                "minhash_threshold": float(self.threshold),
                "avg_jaccard": round(avg_jac, 4),
                "shingle_size": int(self.shingle_size),
                "block_columns": list(cols),
            }
            out.append(
                Cluster(
                    source=source,
                    rows=tuple(rows_sorted),
                    key_columns=tuple(cols),
                    source_strategy=self.name,
                    evidence=evidence,
                    # MinHash is more discriminating than blocking but
                    # less than exact-key collision -> medium.
                    confidence="medium",
                )
            )
        return out


# ---------------------------------------------------------------------------
# EmbeddingClusterer (sentence-transformers, optional)
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingClusterer:
    """Sentence-embedding near-duplicate clusterer.

    Each row is rendered to text (its candidate-key columns,
    concatenated) and embedded. Pairs with cosine similarity above
    ``threshold`` are connected; transitive closure forms clusters.

    The default embedder is ``sentence-transformers/all-MiniLM-L6-v2``
    (~80MB, fast on CPU). For higher quality at ~3x cost, pass
    ``model_name="sentence-transformers/all-mpnet-base-v2"``.

    For testing or pluggability you can pass an ``embedder`` callable
    of signature ``(List[str]) -> np.ndarray`` (shape
    ``(n_rows, dim)``); when provided, ``model_name`` is ignored and
    the ``sentence_transformers`` import is bypassed.

    Parameters
    ----------
    name
        Strategy name.
    threshold
        Cosine-similarity threshold in ``[0, 1]``. ``0.85`` is a
        reasonable default for short rows; lower for higher recall.
    model_name
        sentence-transformers model identifier. Ignored when
        ``embedder`` is supplied.
    min_chars
        Skip rows whose rendered text is shorter than this (the
        embedding would be too generic to be useful).
    max_cluster_size
        Cap the size of any one returned cluster.
    embedder
        Optional callable returning a 2-D numpy array of embeddings.
        When ``None`` the clusterer lazily instantiates a
        ``SentenceTransformer`` on first call.
    """

    name: str = "embedding"
    threshold: float = 0.85
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_chars: int = 8
    max_cluster_size: int = 50
    embedder: Optional[object] = None  # callable[[List[str]], np.ndarray]

    # Internal: lazy-loaded SentenceTransformer + encode callable. We
    # cache them on the instance so a multi-CSV detector run does NOT
    # reload the ~80MB model once per source. Set non-init so the
    # dataclass equality / repr stays simple.
    _cached_encode: Optional[object] = field(default=None, init=False, repr=False)
    _cached_load_failed: bool = field(default=False, init=False, repr=False)

    def cluster(
        self,
        *,
        source: str,
        df: pd.DataFrame,
        digest: Optional[TableDigest],
    ) -> List[Cluster]:
        if df.empty:
            return []
        cols = _candidate_text_columns(df, digest)
        if not cols:
            return []
        # Render row texts, keep only rows with enough content.
        row_texts: List[str] = []
        kept_rows: List[int] = []
        for ridx in range(len(df)):
            text = _row_text(df, ridx, cols)
            if len(text) < self.min_chars:
                continue
            row_texts.append(text)
            kept_rows.append(ridx)
        if len(row_texts) < 2:
            return []
        try:
            import numpy as np  # type: ignore
        except ImportError:
            logger.warning(
                "row_duplication: numpy not installed; skipping "
                "EmbeddingClusterer for %s.",
                source,
            )
            return []
        # Resolve the embedder. When the caller didn't pre-build one,
        # lazily import sentence-transformers so the dependency is
        # only required when this clusterer actually runs. The loaded
        # model is cached on the instance (``_cached_encode``) so
        # subsequent ``cluster()`` calls (one per source CSV in
        # multi-CSV mode) reuse it instead of reloading ~80MB from
        # disk every time.
        embedder = self.embedder
        if embedder is None:
            if self._cached_load_failed:
                # Earlier source already failed to load the model;
                # bail without retrying so we don't spam warnings.
                return []
            if self._cached_encode is None:
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                except ImportError:
                    logger.warning(
                        "row_duplication: sentence-transformers not "
                        "installed; skipping EmbeddingClusterer for %s. "
                        "Install with `pip install sentence-transformers`.",
                        source,
                    )
                    self._cached_load_failed = True
                    return []
                try:
                    model = SentenceTransformer(self.model_name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "row_duplication: failed to load embedding model "
                        "%s for %s: %s",
                        self.model_name, source, exc,
                    )
                    self._cached_load_failed = True
                    return []

                def _encode(texts: List[str], _model=model):
                    return _model.encode(
                        texts, convert_to_numpy=True, normalize_embeddings=True
                    )
                self._cached_encode = _encode
            embedder = self._cached_encode

        try:
            embeddings = embedder(row_texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "row_duplication: embedder failed for %s: %s",
                source, exc,
            )
            return []
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(row_texts):
            logger.warning(
                "row_duplication: embedder returned unexpected shape "
                "%s for %s",
                getattr(embeddings, "shape", None), source,
            )
            return []
        # L2-normalise so cosine sim is just a dot product.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms
        # Pairwise cosine similarity.
        sim = normed @ normed.T
        np.fill_diagonal(sim, 0.0)
        # Union-find over the kept rows.
        parent = {r: r for r in kept_rows}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        n = len(kept_rows)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= self.threshold:
                    union(kept_rows[i], kept_rows[j])

        groups: Dict[int, List[int]] = {}
        for r in kept_rows:
            groups.setdefault(find(r), []).append(r)
        out: List[Cluster] = []
        for rows in groups.values():
            if len(rows) < 2 or len(rows) > self.max_cluster_size:
                continue
            rows_sorted = sorted(rows)
            # Average pairwise similarity within the cluster, for the
            # adjudicator's "discovery evidence" block.
            idx_lookup = {r: i for i, r in enumerate(kept_rows)}
            sims: List[float] = []
            for i in range(len(rows_sorted)):
                for j in range(i + 1, len(rows_sorted)):
                    sims.append(
                        float(sim[idx_lookup[rows_sorted[i]],
                                  idx_lookup[rows_sorted[j]]])
                    )
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            evidence = {
                "embedding_threshold": float(self.threshold),
                "avg_cosine_similarity": round(avg_sim, 4),
                "embedding_model": self.model_name if self.embedder is None else "(custom)",
                "block_columns": list(cols),
            }
            out.append(
                Cluster(
                    source=source,
                    rows=tuple(rows_sorted),
                    key_columns=tuple(cols),
                    source_strategy=self.name,
                    evidence=evidence,
                    # Embedding clusters carry semantic meaning but are
                    # the broadest strategy -> medium confidence.
                    confidence="medium",
                )
            )
        return out
