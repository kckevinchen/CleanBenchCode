"""Unit tests for ``src.core.row_dup_clusterers``.

Covers:
- ``merge_clusters`` deduplication + evidence union + confidence escalation.
- ``rank_and_cap`` ordering + budget exhaustion flag.
- ``KeyCollisionClusterer`` (smoke; behaviour exercised by
  ``tests/test_row_duplication_detector.py`` end-to-end).
- ``BlockingClusterer`` token-signature grouping.
- ``MinHashLSHClusterer`` Jaccard near-duplicate grouping
  (skipped if ``datasketch`` is not installed).
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from src.core.row_dup_clusterers import (
    BlockingClusterer,
    Cluster,
    KeyCollisionClusterer,
    merge_clusters,
    rank_and_cap,
)


# ---------------------------------------------------------------------------
# merge_clusters
# ---------------------------------------------------------------------------


class TestMergeClusters:
    def test_dedup_by_fingerprint(self) -> None:
        a = Cluster(
            source="t.csv", rows=(0, 1, 2), key_columns=("k",),
            source_strategy="key_collision",
            evidence={"key_columns": ["k"]}, confidence="high",
        )
        b = Cluster(
            source="t.csv", rows=(2, 1, 0), key_columns=("k",),
            source_strategy="blocking",
            evidence={"block_signature": "acme|nyc"}, confidence="low",
        )
        merged = merge_clusters([a, b])
        assert len(merged) == 1
        m = merged[0]
        # Confidence escalates to the highest contributor.
        assert m.confidence == "high"
        # Both strategies pinned.
        assert "key_collision" in m.all_strategies
        assert "blocking" in m.all_strategies
        # Evidence merged.
        assert m.evidence["key_columns"] == ["k"]
        assert m.evidence["block_signature"] == "acme|nyc"

    def test_distinct_rowsets_preserved(self) -> None:
        a = Cluster(
            source="t.csv", rows=(0, 1), key_columns=("k",),
            source_strategy="key_collision",
        )
        b = Cluster(
            source="t.csv", rows=(2, 3), key_columns=("k",),
            source_strategy="key_collision",
        )
        assert len(merge_clusters([a, b])) == 2


# ---------------------------------------------------------------------------
# rank_and_cap
# ---------------------------------------------------------------------------


class TestRankAndCap:
    def test_high_confidence_first(self) -> None:
        low = Cluster(
            source="t", rows=(5, 6, 7, 8), key_columns=("a",),
            source_strategy="blocking", confidence="low",
        )
        high = Cluster(
            source="t", rows=(0, 1), key_columns=("a",),
            source_strategy="key_collision", confidence="high",
        )
        kept, exhausted = rank_and_cap([low, high], max_per_source=10)
        assert kept[0] is high
        assert exhausted is False

    def test_budget_exhausted(self) -> None:
        clusters = [
            Cluster(
                source="t", rows=(i, i + 100), key_columns=("a",),
                source_strategy="key_collision", confidence="medium",
            )
            for i in range(5)
        ]
        kept, exhausted = rank_and_cap(clusters, max_per_source=3)
        assert len(kept) == 3
        assert exhausted is True


# ---------------------------------------------------------------------------
# KeyCollisionClusterer (smoke)
# ---------------------------------------------------------------------------


class TestKeyCollisionClusterer:
    def test_emits_high_confidence(self) -> None:
        df = pd.DataFrame({"id": [1, 1, 2, 3], "name": ["a", "a", "b", "c"]})
        cl = KeyCollisionClusterer(max_key_size=1, normalize_keys=False)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        assert len(out) >= 1
        assert all(c.confidence == "high" for c in out)
        # Rows 0 and 1 share id=1 / name=a -> same cluster.
        assert any(set(c.rows) == {0, 1} for c in out)


# ---------------------------------------------------------------------------
# BlockingClusterer
# ---------------------------------------------------------------------------


class TestBlockingClusterer:
    def test_token_signature_groups_abbreviations(self) -> None:
        # All three rows share the first-token signature "acme" on
        # the company column. With *only* the company column the
        # block should cover all three. (When city is also a
        # signature column, rows split by city signature.)
        df = pd.DataFrame(
            {"company": ["Acme Corp", "Acme Corporation", "ACME Inc"]}
        )
        cl = BlockingClusterer(min_block_size=2)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        # At least one cluster covers all three rows.
        assert any(set(c.rows) == {0, 1, 2} for c in out)
        # All have low confidence by design.
        assert all(c.confidence == "low" for c in out)

    def test_signature_combines_all_key_columns(self) -> None:
        # When multiple candidate columns exist the signature is the
        # tuple of first-tokens. Rows must agree on every column.
        df = pd.DataFrame(
            {
                "company": ["Acme Corp", "Acme Corp", "ACME Inc"],
                "city": ["NYC", "NYC", "Boston"],
            }
        )
        cl = BlockingClusterer(min_block_size=2)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        # Rows 0 and 1 share (acme, nyc); row 2 differs on city.
        assert any(set(c.rows) == {0, 1} for c in out)
        assert not any(set(c.rows) == {0, 1, 2} for c in out)

    def test_block_size_filter(self) -> None:
        df = pd.DataFrame({"x": ["foo", "foo", "bar"]})
        cl = BlockingClusterer(min_block_size=3)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        # Block of 2 ('foo','foo') is below min; result empty.
        assert out == []

    def test_max_block_size_filter(self) -> None:
        df = pd.DataFrame({"x": ["the same"] * 100})
        cl = BlockingClusterer(min_block_size=2, max_block_size=10)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        # 100-row block exceeds cap; dropped.
        assert out == []

    def test_empty_signatures_skipped(self) -> None:
        df = pd.DataFrame({"x": [None, None, "real"]})
        cl = BlockingClusterer()
        out = cl.cluster(source="t.csv", df=df, digest=None)
        assert out == []


# ---------------------------------------------------------------------------
# MinHashLSHClusterer (datasketch optional)
# ---------------------------------------------------------------------------


_HAS_DATASKETCH = importlib.util.find_spec("datasketch") is not None


@pytest.mark.skipif(not _HAS_DATASKETCH, reason="datasketch not installed")
class TestMinHashLSHClusterer:
    def test_near_duplicates_grouped(self) -> None:
        from src.core.row_dup_clusterers import MinHashLSHClusterer
        df = pd.DataFrame(
            {
                "review": [
                    "the quick brown fox jumped over the lazy dog",
                    "the quick brown fox jumps over the lazy dog",  # near-dup
                    "completely different text about cats and yarn",
                    "cats and yarn are completely different topics",
                ]
            }
        )
        cl = MinHashLSHClusterer(threshold=0.5, min_tokens=3)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        # The two fox sentences should land in the same cluster.
        fox_cluster = next(
            (c for c in out if {0, 1}.issubset(set(c.rows))), None
        )
        assert fox_cluster is not None, f"no fox cluster: {out}"
        assert fox_cluster.confidence == "medium"
        assert fox_cluster.evidence["avg_jaccard"] > 0

    def test_unrelated_rows_not_grouped(self) -> None:
        from src.core.row_dup_clusterers import MinHashLSHClusterer
        df = pd.DataFrame(
            {
                "x": [
                    "alpha beta gamma delta epsilon",
                    "one two three four five",
                    "red green blue yellow purple",
                ]
            }
        )
        cl = MinHashLSHClusterer(threshold=0.7, min_tokens=3)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        assert out == []

    def test_short_rows_skipped(self) -> None:
        from src.core.row_dup_clusterers import MinHashLSHClusterer
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        cl = MinHashLSHClusterer(min_tokens=3)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        assert out == []


# ---------------------------------------------------------------------------
# EmbeddingClusterer (stub embedder; sentence-transformers not required)
# ---------------------------------------------------------------------------


_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
class TestEmbeddingClusterer:
    def _stub_embedder(self, mapping):
        """Return a callable mapping each input text to a fixed vector."""
        import numpy as np

        def _emb(texts):
            vecs = []
            for t in texts:
                # Find the closest registered text by exact match;
                # fall back to a unit vector along axis 0.
                if t in mapping:
                    vecs.append(mapping[t])
                else:
                    base = [0.0] * len(next(iter(mapping.values())))
                    base[0] = 1.0
                    vecs.append(base)
            return np.asarray(vecs, dtype=np.float32)

        return _emb

    def test_semantic_near_duplicates_clustered(self) -> None:
        from src.core.row_dup_clusterers import EmbeddingClusterer
        # Three rows: two semantically similar (vectors [1,0,0] and
        # [0.95,0.1,0]), one unrelated ([0,1,0]).
        texts = [
            "the leading widget vendor",
            "widget vendor leading",
            "completely unrelated banana",
        ]
        embedder = self._stub_embedder({
            texts[0]: [1.0, 0.0, 0.0],
            texts[1]: [0.95, 0.1, 0.0],
            texts[2]: [0.0, 1.0, 0.0],
        })
        df = pd.DataFrame({"x": texts})
        cl = EmbeddingClusterer(threshold=0.9, min_chars=3, embedder=embedder)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        # Rows 0 and 1 cluster together; row 2 is excluded.
        assert any(set(c.rows) == {0, 1} for c in out)
        assert not any(2 in c.rows for c in out)

    def test_threshold_separates_clusters(self) -> None:
        from src.core.row_dup_clusterers import EmbeddingClusterer
        texts = ["a", "b", "c"]
        embedder = self._stub_embedder({
            texts[0]: [1.0, 0.0],
            texts[1]: [0.0, 1.0],
            texts[2]: [-1.0, 0.0],
        })
        df = pd.DataFrame({"x": ["alpha words", "beta words", "gamma words"]})
        # Map the rendered text to vectors. The clusterer renders rows
        # via normalize_for_dedup -> " ".join(parts), so reconstruct
        # the same texts here.
        from src.core.multi_row_priors import normalize_for_dedup
        rendered = [normalize_for_dedup(s) for s in df["x"]]
        embedder = self._stub_embedder({
            rendered[0]: [1.0, 0.0],
            rendered[1]: [0.0, 1.0],
            rendered[2]: [-1.0, 0.0],
        })
        cl = EmbeddingClusterer(threshold=0.99, min_chars=3, embedder=embedder)
        out = cl.cluster(source="t.csv", df=df, digest=None)
        # No pairs above 0.99 -> empty.
        assert out == []

    def test_empty_when_no_text_columns(self) -> None:
        from src.core.row_dup_clusterers import EmbeddingClusterer
        df = pd.DataFrame({"n": [1, 2, 3]})
        cl = EmbeddingClusterer(embedder=lambda texts: [[1.0]] * len(texts))
        out = cl.cluster(source="t.csv", df=df, digest=None)
        assert out == []
