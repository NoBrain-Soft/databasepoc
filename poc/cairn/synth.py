"""Synthetic corpora for the PoC benchmarks.

Two regimes are generated on purpose:

``clustered``  -- what real embedding corpora look like: low intrinsic
                  dimension, topically coherent documents.  Block summaries are
                  tight and pruning works.
``uniform``    -- vectors drawn uniformly on the sphere.  Concentration of
                  measure makes every block's angular radius approach pi/2, the
                  dense bound goes slack, and branch-and-bound degenerates to a
                  scan.  This is the honest worst case for the design and the
                  benchmarks report it rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .segment import RowBatch, _normalize
from .summary import QueryBounds

LANGS = 4
TENANTS = 32
KINDS = 3
DAY = 86400.0


@dataclass
class Corpus:
    batch: RowBatch
    centers: np.ndarray
    cluster_of_row: np.ndarray
    term_pool: List[np.ndarray]
    vocab: int
    now: float


def make_corpus(
    n: int = 100_000,
    d: int = 96,
    clusters: int = 200,
    intra_cos: float = 0.85,
    vocab: int = 20_000,
    terms_per_row: int = 18,
    units_per_doc: int = 8,
    regime: str = "clustered",
    seed: int = 7,
    with_text: bool = False,
) -> Corpus:
    rng = np.random.default_rng(seed)
    centers = _normalize(rng.normal(size=(clusters, d)).astype(np.float32))
    # Per-dimension noise scale that yields the requested expected cosine
    # between a row and its cluster centre: cos ~= 1 / sqrt(1 + d * sigma^2).
    sigma = float(np.sqrt(max(1e-9, (1.0 / intra_cos ** 2 - 1.0)) / d))

    if regime == "uniform":
        vecs = _normalize(rng.normal(size=(n, d)).astype(np.float32))
        cluster_of_row = rng.integers(0, clusters, size=n)
    else:
        # Documents are topically coherent: all units of a doc share a cluster.
        n_docs = max(1, n // units_per_doc)
        doc_cluster = rng.integers(0, clusters, size=n_docs)
        cluster_of_row = np.repeat(doc_cluster, units_per_doc)[:n]
        if len(cluster_of_row) < n:
            cluster_of_row = np.concatenate(
                [cluster_of_row, rng.integers(0, clusters, size=n - len(cluster_of_row))]
            )
        vecs = _normalize(
            centers[cluster_of_row] + sigma * rng.normal(size=(n, d)).astype(np.float32)
        )

    doc_id = np.arange(n, dtype=np.int64) // units_per_doc
    ordinal = (np.arange(n, dtype=np.int64) % units_per_doc).astype(np.int32)
    tokens = np.clip(rng.normal(90, 22, size=n), 24, 220).astype(np.int32)
    now = 1_800_000_000.0
    ts = now - rng.gamma(2.0, 90.0, size=n) * DAY

    categorical = {
        "lang": rng.choice(LANGS, size=n, p=_skew(LANGS, rng)).astype(np.int32),
        "tenant": (doc_id % TENANTS).astype(np.int32),
        "kind": rng.integers(0, KINDS, size=n).astype(np.int32),
    }
    numeric = {"quality": rng.beta(5, 2, size=n).astype(np.float64), "ts": ts}

    # Each cluster draws its terms from an overlapping Zipf-shaped pool, so
    # lexical and semantic signals correlate the way they do in real corpora.
    zipf = 1.0 / np.arange(1, vocab + 1) ** 0.9
    zipf /= zipf.sum()
    term_pool = [rng.choice(vocab, size=200, replace=False, p=zipf) for _ in range(clusters)]

    indices = np.zeros(n * terms_per_row, dtype=np.int32)
    data = np.zeros(n * terms_per_row, dtype=np.float32)
    for i in range(n):
        pool = term_pool[cluster_of_row[i]]
        picks = rng.choice(pool, size=terms_per_row, replace=False)
        a = i * terms_per_row
        indices[a:a + terms_per_row] = picks
        data[a:a + terms_per_row] = rng.gamma(2.0, 0.35, size=terms_per_row).astype(np.float32)
    indptr = np.arange(n + 1, dtype=np.int64) * terms_per_row

    text = None
    if with_text:
        text = [
            f"[doc {doc_id[i]} unit {ordinal[i]}] "
            + " ".join(f"t{t}" for t in indices[i * terms_per_row:(i + 1) * terms_per_row][:8])
            for i in range(n)
        ]

    batch = RowBatch(
        vecs=vecs,
        ts=ts,
        tokens=tokens,
        doc_id=doc_id,
        ordinal=ordinal,
        row_id=np.arange(n, dtype=np.int64),
        numeric=numeric,
        categorical=categorical,
        sp_indptr=indptr,
        sp_indices=indices,
        sp_data=data,
        text=text,
    )
    return Corpus(batch, centers, cluster_of_row, term_pool, vocab, now)


def _skew(k: int, rng) -> np.ndarray:
    p = 1.0 / np.arange(1, k + 1) ** 1.4
    return p / p.sum()


def make_queries(
    corpus: Corpus,
    count: int = 100,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    query_cos: float = 0.80,
    query_terms: int = 6,
    seed: int = 11,
) -> List[QueryBounds]:
    rng = np.random.default_rng(seed)
    d = corpus.batch.vecs.shape[1]
    sigma = float(np.sqrt(max(1e-9, (1.0 / query_cos ** 2 - 1.0)) / d))
    out = []
    for _ in range(count):
        c = rng.integers(0, len(corpus.centers))
        v = _normalize(corpus.centers[c] + sigma * rng.normal(size=d).astype(np.float32))
        sparse: Dict[int, float] = {}
        if beta:
            picks = rng.choice(corpus.term_pool[c], size=query_terms, replace=False)
            for t in picks:
                sparse[int(t)] = float(rng.gamma(2.0, 0.4))
        out.append(
            QueryBounds(
                dense=v.astype(np.float32),
                sparse=sparse,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                now=corpus.now,
            )
        )
    return out


def brute_force(batch: RowBatch, qb: QueryBounds, k: int, predicates=(), live=None):
    """Exact top-k by scanning every row -- the ground truth for the benchmarks."""
    n = len(batch)
    scores = np.zeros(n, dtype=np.float64)
    if qb.alpha and qb.dense is not None:
        scores += qb.alpha * (batch.vecs @ qb.dense)
    if qb.beta and qb.sparse:
        vocab = int(batch.sp_indices.max()) + 1
        qv = np.zeros(vocab, dtype=np.float32)
        for t, w in qb.sparse.items():
            if 0 <= t < vocab:
                qv[t] = w
        contrib = qv[batch.sp_indices] * batch.sp_data
        counts = np.diff(batch.sp_indptr)
        rowids = np.repeat(np.arange(n), counts)
        scores += qb.beta * np.bincount(rowids, weights=contrib, minlength=n)
    if qb.gamma:
        scores += qb.gamma * qb.recency_vector(batch.ts)

    mask = np.ones(n, dtype=bool) if live is None else live.copy()
    cols = dict(batch.numeric)
    cols.update(batch.categorical)
    for p in predicates:
        mask &= p.mask(cols, slice(0, n))
    scores = np.where(mask, scores, -np.inf)
    order = np.argsort(-scores)[:k]
    return [(int(batch.row_id[i]), float(scores[i])) for i in order if np.isfinite(scores[i])]
