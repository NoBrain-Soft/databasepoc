"""The cairn: a stack of mergeable summaries, and the traversal that uses it.

Search is a single branch-and-bound walk over one tree of bounds.  Dense
similarity, sparse/lexical impact, scalar predicates, and the recency prior are
*not* separate indexes that get fused after the fact -- they are components of
one bound that is evaluated per node.  A node is opened only if its bound can
still beat the current k-th best score, so the engine reads exactly the blocks
that could change the answer.

Because the bounds are sound, the default mode returns the **exact** top-k
under the declared scoring function.  Approximate modes trade work for a
quantified risk: every answer carries a *certificate* recording the largest
bound left unexplored, so the caller learns how much a missed row could
possibly have beaten the result.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .quantize import Quantizer
from .segment import RowBatch, plan_blocks, summarize, _normalize
from .summary import (
    DEFAULT_SPARSE_LAMBDA,
    QueryBounds,
    Summary,
    TriState,
)


@dataclass
class Node:
    summary: Summary
    block: int = -1                       # >= 0 for leaves
    children: List["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.block >= 0


@dataclass
class SearchStats:
    nodes_visited: int = 0
    blocks_opened: int = 0
    rows_scored: int = 0
    rows_total: int = 0
    pruned_by_predicate: int = 0
    pruned_by_bound: int = 0

    def add(self, other: "SearchStats") -> "SearchStats":
        return SearchStats(
            self.nodes_visited + other.nodes_visited,
            self.blocks_opened + other.blocks_opened,
            self.rows_scored + other.rows_scored,
            self.rows_total + other.rows_total,
            self.pruned_by_predicate + other.pruned_by_predicate,
            self.pruned_by_bound + other.pruned_by_bound,
        )


@dataclass
class Certificate:
    """What the engine can prove about the answer it returned."""

    kth_score: float
    max_remaining_bound: float
    exact: bool

    @property
    def ratio(self) -> float:
        """Worst-case factor by which an unretrieved row could beat the k-th.

        1.0 means "nothing left could have beaten it" -- the answer is exact.
        """
        if self.kth_score <= 0:
            return math.inf if self.max_remaining_bound > 0 else 1.0
        return max(1.0, self.max_remaining_bound / self.kth_score)


@dataclass
class Hit:
    row: int                # row offset within the segment (physical order)
    row_id: int
    score: float
    segment: int = 0


class Segment:
    """An immutable, block-laid-out slice of the corpus with its cairn tree."""

    def __init__(
        self,
        batch: RowBatch,
        block_size: int = 256,
        max_radius: Optional[float] = None,
        fanout: int = 16,
        facet_columns: Sequence[str] = (),
        partition_by: Optional[str] = None,
        quantizer: Optional[Quantizer] = None,
        lam: int = DEFAULT_SPARSE_LAMBDA,
        term_filter: bool = True,
        seed: int = 0,
        segment_id: int = 0,
    ):
        self.segment_id = segment_id
        self.facet_columns = tuple(facet_columns)
        part = batch.categorical.get(partition_by) if partition_by else None
        groups = plan_blocks(
            batch.vecs, block_size, partition_by=part, max_radius=max_radius, seed=seed
        )
        order = np.concatenate(groups) if groups else np.zeros(0, dtype=np.int64)
        self.batch = batch.take(order)
        sizes = [len(g) for g in groups]
        self.bounds = np.zeros(len(sizes) + 1, dtype=np.int64)
        np.cumsum(sizes, out=self.bounds[1:])
        self.live = np.ones(len(self.batch), dtype=bool)

        self.leaf_summaries = [
            summarize(self.batch, int(self.bounds[i]), int(self.bounds[i + 1]),
                      facet_columns=self.facet_columns, lam=lam,
                      term_filter=term_filter)
            for i in range(len(sizes))
        ]
        self.root = _build_tree(self.leaf_summaries, fanout, lam, seed)

        self.quantizer = quantizer
        self.codes = self.scale = None
        if quantizer is not None:
            self.codes, self.scale = quantizer.encode(self.batch.vecs)

        self.vocab = int(self.batch.sp_indices.max()) + 1 if (
            self.batch.sp_indices is not None and len(self.batch.sp_indices)
        ) else 1
        self._qvec_cache: Tuple[int, Optional[np.ndarray]] = (-1, None)

        self._columns: Dict[str, np.ndarray] = {}
        self._columns.update(self.batch.numeric)
        self._columns.update(self.batch.categorical)

    # ------------------------------------------------------------------ misc

    def __len__(self) -> int:
        return len(self.batch)

    @property
    def num_blocks(self) -> int:
        return len(self.leaf_summaries)

    def live_count(self) -> int:
        return int(self.live.sum())

    def mark_deleted(self, row_ids: Sequence[int]) -> int:
        """Tombstone rows.  Bounds stay sound: removing rows can only lower the
        true maximum, so no summary needs to be rebuilt."""
        if not len(row_ids):
            return 0
        hit = np.isin(self.batch.row_id, np.asarray(list(row_ids), dtype=np.int64))
        n = int((hit & self.live).sum())
        self.live &= ~hit
        return n

    def summary_bytes(self) -> int:
        return sum(s.nbytes() for s in self.leaf_summaries) + _tree_bytes(self.root)

    # ---------------------------------------------------------------- search

    def search(
        self,
        qb: QueryBounds,
        k: int = 10,
        predicates: Sequence[object] = (),
        epsilon: float = 0.0,
        block_budget: Optional[int] = None,
        heap: Optional[List[Tuple[float, int, int, int]]] = None,
    ) -> Tuple[List[Hit], Certificate, SearchStats]:
        """Branch-and-bound top-k over this segment.

        ``heap`` may carry results from other segments so that a good threshold
        found elsewhere prunes here too (cross-segment bound sharing).
        """
        stats = SearchStats(rows_total=len(self.batch))
        results: List[Tuple[float, int, int, int]] = list(heap) if heap else []
        heapq.heapify(results)

        eps_row = self.quantizer.eps * qb.alpha if self.quantizer is not None else 0.0
        # With quantised row scores the top-k heap alone is not safe: a true
        # top-k row can be evicted by an over-estimated rival.  Every row whose
        # estimate is within 2*eps of the k-th estimate is therefore retained
        # for exact re-ranking, which restores the guarantee.
        cands: Optional[List[Tuple[float, int, int]]] = (
            [] if self.quantizer is not None else None
        )
        self._cand_cap = max(4096, k * 64)
        counter = itertools.count()
        pq: List[Tuple[float, int, Node, Summary]] = []

        if self.root.summary.evaluate(predicates) != TriState.NONE:
            heapq.heappush(
                pq,
                (-self.root.summary.filtered_upper_bound(qb, predicates),
                 next(counter), self.root, self.root.summary),
            )

        max_remaining: Optional[float] = None
        exhausted = True
        while pq:
            neg_ub, _, node, node_sum = heapq.heappop(pq)
            ub = -neg_ub
            threshold = _threshold(results, k)
            # A candidate is only safe to discard if it cannot beat the k-th
            # best even after the most favourable quantisation error.
            stop_at = threshold * (1.0 + epsilon) - eps_row if threshold > -math.inf else -math.inf
            if ub <= stop_at:
                max_remaining = ub
                exhausted = False
                stats.pruned_by_bound += 1
                break
            if block_budget is not None and stats.blocks_opened >= block_budget:
                max_remaining = ub
                exhausted = False
                break

            stats.nodes_visited += 1
            if node.is_leaf:
                self._score_block(node.block, qb, predicates, results, k, stats,
                                  cands, eps_row)
                if cands is not None and len(cands) > self._cand_cap:
                    cands = _trim(cands, _threshold(results, k) - 2 * eps_row,
                                  self._cand_cap)
                continue
            for child in node.children:
                cs = child.summary
                if cs.evaluate(predicates) == TriState.NONE:
                    stats.pruned_by_predicate += 1
                    continue
                if cs.refine(predicates).evaluate(predicates) == TriState.NONE:
                    stats.pruned_by_predicate += 1
                    continue
                heapq.heappush(
                    pq, (-cs.filtered_upper_bound(qb, predicates), next(counter), child, cs)
                )

        if max_remaining is None:
            max_remaining = -math.inf if exhausted else (-pq[0][0] if pq else -math.inf)

        # Re-rank the retained candidates exactly when scoring used quantised
        # codes, then fold them back into the result heap.
        if self.quantizer is not None and cands:
            cands = _trim(cands, _threshold(results, k) - 2 * eps_row, self._cand_cap)
            results = self._rerank_candidates(results, cands, qb, k)

        best = sorted(results, key=lambda r: -r[0])[:k]
        hits = [Hit(row=r[1], row_id=r[2], score=r[0], segment=r[3]) for r in best]
        kth = best[-1][0] if len(best) >= k else (best[-1][0] if best else -math.inf)
        remaining = max_remaining + eps_row
        cert = Certificate(
            kth_score=kth,
            max_remaining_bound=remaining,
            exact=exhausted or remaining <= kth,
        )
        return hits, cert, stats

    # --------------------------------------------------------------- scoring

    def _score_block(self, block: int, qb: QueryBounds, predicates, results, k, stats,
                     cands=None, eps_row: float = 0.0):
        lo, hi = int(self.bounds[block]), int(self.bounds[block + 1])
        stats.blocks_opened += 1

        mask = self.live[lo:hi]
        summ = self.leaf_summaries[block]
        if summ.evaluate(predicates) == TriState.SOME:
            for p in predicates:
                mask = mask & p.mask(self._columns, slice(lo, hi))
        if not mask.any():
            return

        idx = np.nonzero(mask)[0]
        scores = np.zeros(len(idx), dtype=np.float32)
        if qb.alpha and qb.dense is not None:
            if self.quantizer is not None:
                est = self.quantizer.score(
                    self.codes[lo:hi][idx], self.scale[lo:hi][idx], qb.dense[None, :]
                )[0]
                scores += qb.alpha * est
            else:
                scores += qb.alpha * (self.batch.vecs[lo:hi][idx] @ qb.dense)
        if qb.beta and qb.sparse:
            scores += qb.beta * self._sparse_scores(lo, hi, idx, self._query_vector(qb))
        if qb.gamma:
            scores += qb.gamma * qb.recency_vector(self.batch.ts[lo:hi][idx])

        stats.rows_scored += len(idx)
        row_ids = self.batch.row_id[lo:hi][idx]
        for j, s in enumerate(scores):
            s = float(s)
            if len(results) < k:
                heapq.heappush(results, (s, int(lo + idx[j]), int(row_ids[j]), self.segment_id))
            elif s > results[0][0]:
                heapq.heapreplace(results, (s, int(lo + idx[j]), int(row_ids[j]), self.segment_id))
        if cands is not None:
            keep = _threshold(results, k) - 2 * eps_row
            for j, s in enumerate(scores):
                if float(s) >= keep:
                    cands.append((float(s), int(lo + idx[j]), int(row_ids[j])))

    def _sparse_scores(self, lo: int, hi: int, idx: np.ndarray, qvec: np.ndarray) -> np.ndarray:
        if self.batch.sp_indptr is None:
            return np.zeros(len(idx), dtype=np.float32)
        indptr, indices, data = self.batch.sp_indptr, self.batch.sp_indices, self.batch.sp_data
        a, b = int(indptr[lo]), int(indptr[hi])
        if b == a:
            return np.zeros(len(idx), dtype=np.float32)
        sub_idx, sub_dat = indices[a:b], data[a:b]
        weights = np.where(sub_idx < len(qvec), qvec[np.minimum(sub_idx, len(qvec) - 1)], 0.0)
        contrib = weights * sub_dat
        counts = np.diff(indptr[lo:hi + 1])
        rowids = np.repeat(np.arange(hi - lo), counts)
        sums = np.bincount(rowids, weights=contrib, minlength=hi - lo)
        return sums[idx].astype(np.float32)

    def _query_vector(self, qb: QueryBounds) -> np.ndarray:
        """Densify the sparse query once per search, not once per block."""
        key = id(qb)
        if self._qvec_cache[0] == key and self._qvec_cache[1] is not None:
            return self._qvec_cache[1]
        v = np.zeros(self.vocab, dtype=np.float32)
        for t, w in qb.sparse.items():
            if 0 <= t < self.vocab:
                v[t] = w
        self._qvec_cache = (key, v)
        return v

    def _rerank_candidates(self, results, cands, qb: QueryBounds, k: int):
        """Replace the quantised dense term with the full-precision one."""
        if qb.dense is None:
            return results
        rows = np.array([c[1] for c in cands], dtype=np.int64)
        exact = qb.alpha * (self.batch.vecs[rows] @ qb.dense)
        approx = qb.alpha * self.quantizer.score(
            self.codes[rows], self.scale[rows], qb.dense[None, :]
        )[0]
        foreign = [r for r in results if r[3] != self.segment_id]
        rescored = [
            (float(c[0] - approx[i] + exact[i]), c[1], c[2], self.segment_id)
            for i, c in enumerate(cands)
        ]
        merged = foreign + rescored
        merged.sort(key=lambda r: -r[0])
        return merged[:max(k, 1)]


# --------------------------------------------------------------------------
# Tree construction
# --------------------------------------------------------------------------


def _build_tree(summaries: List[Summary], fanout: int, lam: int, seed: int) -> Node:
    nodes = [Node(summary=s, block=i) for i, s in enumerate(summaries)]
    if not nodes:
        raise ValueError("cannot build a cairn over zero blocks")
    rng = np.random.default_rng(seed)
    while len(nodes) > 1:
        centroids = np.vstack([n.summary.centroid for n in nodes])
        groups = plan_blocks(centroids, fanout, seed=int(rng.integers(1 << 30)))
        parents = []
        for g in groups:
            members = [nodes[i] for i in g]
            merged = members[0].summary
            for m in members[1:]:
                merged = merged.merge(m.summary, lam)
            parents.append(Node(summary=merged, children=members))
        if len(parents) >= len(nodes):      # no progress; collapse in one step
            merged = nodes[0].summary
            for m in nodes[1:]:
                merged = merged.merge(m.summary, lam)
            return Node(summary=merged, children=nodes)
        nodes = parents
    return nodes[0]


def _tree_bytes(node: Node) -> int:
    if node.is_leaf:
        return 0
    return node.summary.nbytes() + sum(_tree_bytes(c) for c in node.children)


def _trim(cands, keep: float, cap: int):
    """Drop candidates that can no longer reach the top-k, cheapest first."""
    if keep == -math.inf:
        return cands[-cap:] if len(cands) > cap else cands
    out = [c for c in cands if c[0] >= keep]
    if len(out) > cap:
        out.sort(key=lambda c: -c[0])
        out = out[:cap]
    return out


def _threshold(results, k: int) -> float:
    if len(results) < k:
        return -math.inf
    return results[0][0]
