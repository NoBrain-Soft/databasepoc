"""CairnDB: the semantic LSM engine tying storage, bounds and assembly together.

Write path
    Rows land in an in-memory **memtable** that is searched by brute force.
    Flushing seals the memtable into an immutable, block-laid-out **segment**
    with its own cairn of summaries.  Compaction merges segments, drops
    tombstoned rows, and *re-lays out* the survivors -- which is where bound
    quality is regained after churn.

Why the write path looks like this
    Summaries are sound upper bounds.  Deleting a row can only lower the true
    maximum of a set, so tombstones never invalidate a bound (they only loosen
    it).  Inserting a row *can* raise the maximum, so an insert into a sealed
    block would break soundness.  The LSM shape is therefore not a performance
    convenience here, it is what keeps the pruning invariant true.

Read path
    Segments are visited best-first by their root bound, and the k-th best score
    found so far is threaded through every segment, so a strong result in one
    segment prunes the others (cross-segment bound sharing).
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .assemble import ContextPack, Unit, assemble
from .index import Certificate, Hit, SearchStats, Segment
from .quantize import Quantizer
from .segment import RowBatch, _normalize
from .summary import QueryBounds, TriState


@dataclass
class Config:
    block_size: int = 256
    max_radius: Optional[float] = 0.7
    fanout: int = 16
    facet_columns: Tuple[str, ...] = ()
    partition_by: Optional[str] = None
    memtable_rows: int = 20_000
    compact_threshold: int = 4
    quantize: bool = False
    lam: int = 96
    term_filter: bool = True
    seed: int = 0


@dataclass
class QueryResult:
    hits: List[Hit]
    certificate: Certificate
    stats: SearchStats
    elapsed_ms: float
    context: Optional[ContextPack] = None


class CairnDB:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.segments: List[Segment] = []
        self._mem: Optional[RowBatch] = None
        self._mem_live: Optional[np.ndarray] = None
        self._next_segment_id = 0
        self.quantizer: Optional[Quantizer] = None

    # ----------------------------------------------------------- write path

    def insert(self, batch: RowBatch) -> None:
        self._mem = batch if self._mem is None else self._mem.concat(batch)
        self._mem_live = np.ones(len(self._mem), dtype=bool)
        if len(self._mem) >= self.config.memtable_rows:
            self.flush()

    def flush(self) -> None:
        if self._mem is None or len(self._mem) == 0:
            return
        keep = np.nonzero(self._mem_live)[0]
        batch = self._mem.take(keep) if len(keep) < len(self._mem) else self._mem
        if self.config.quantize and self.quantizer is None:
            self.quantizer = Quantizer.fit(batch.vecs, seed=self.config.seed)
        self.segments.append(self._build_segment(batch))
        self._mem, self._mem_live = None, None
        if len(self.segments) > self.config.compact_threshold:
            self.compact()

    def _build_segment(self, batch: RowBatch) -> Segment:
        seg = Segment(
            batch,
            block_size=self.config.block_size,
            max_radius=self.config.max_radius,
            fanout=self.config.fanout,
            facet_columns=self.config.facet_columns,
            partition_by=self.config.partition_by,
            quantizer=self.quantizer if self.config.quantize else None,
            lam=self.config.lam,
            term_filter=self.config.term_filter,
            seed=self.config.seed,
            segment_id=self._next_segment_id,
        )
        self._next_segment_id += 1
        return seg

    def delete(self, row_ids: Sequence[int]) -> int:
        n = 0
        for seg in self.segments:
            n += seg.mark_deleted(row_ids)
        if self._mem is not None:
            hit = np.isin(self._mem.row_id, np.asarray(list(row_ids), dtype=np.int64))
            n += int((hit & self._mem_live).sum())
            self._mem_live &= ~hit
        return n

    def compact(self) -> None:
        """Merge every sealed segment, dropping tombstones and re-laying out."""
        if not self.segments:
            return
        batches = []
        for seg in self.segments:
            keep = np.nonzero(seg.live)[0]
            if len(keep):
                batches.append(seg.batch.take(keep))
        if not batches:
            self.segments = []
            return
        merged = batches[0]
        for b in batches[1:]:
            merged = merged.concat(b)
        self.segments = [self._build_segment(merged)]

    # ------------------------------------------------------------ read path

    def search(
        self,
        qb: QueryBounds,
        k: int = 10,
        predicates: Sequence[object] = (),
        epsilon: float = 0.0,
        block_budget: Optional[int] = None,
    ) -> QueryResult:
        t0 = time.perf_counter()
        stats = SearchStats()
        heap: List[Tuple[float, int, int, int]] = []
        exact = True
        remaining = -math.inf

        if self._mem is not None and len(self._mem):
            hits, st = self._scan_memtable(qb, k, predicates)
            heap = [(h.score, h.row, h.row_id, -1) for h in hits]
            heapq.heapify(heap)
            stats = stats.add(st)

        order = sorted(
            self.segments,
            key=lambda s: -s.root.summary.filtered_upper_bound(qb, predicates),
        )
        for seg in order:
            if seg.root.summary.evaluate(predicates) == TriState.NONE:
                continue
            hits, cert, st = seg.search(
                qb, k=k, predicates=predicates, epsilon=epsilon,
                block_budget=block_budget, heap=heap,
            )
            stats = stats.add(st)
            heap = [(h.score, h.row, h.row_id, h.segment) for h in hits]
            heapq.heapify(heap)
            exact = exact and cert.exact
            remaining = max(remaining, cert.max_remaining_bound)

        best = sorted(heap, key=lambda r: -r[0])[:k]
        out = [Hit(row=r[1], row_id=r[2], score=r[0], segment=r[3]) for r in best]
        kth = best[-1][0] if best else -math.inf
        cert = Certificate(kth_score=kth, max_remaining_bound=remaining,
                           exact=exact or remaining <= kth)
        return QueryResult(out, cert, stats, (time.perf_counter() - t0) * 1e3)

    def _scan_memtable(self, qb: QueryBounds, k: int, predicates):
        batch, live = self._mem, self._mem_live
        n = len(batch)
        scores = np.zeros(n, dtype=np.float64)
        if qb.alpha and qb.dense is not None:
            scores += qb.alpha * (batch.vecs @ qb.dense)
        if qb.beta and qb.sparse and batch.sp_indices is not None and len(batch.sp_indices):
            vocab = int(batch.sp_indices.max()) + 1
            qv = np.zeros(vocab, dtype=np.float32)
            for t, w in qb.sparse.items():
                if 0 <= t < vocab:
                    qv[t] = w
            contrib = qv[batch.sp_indices] * batch.sp_data
            counts = np.diff(batch.sp_indptr)
            scores += qb.beta * np.bincount(
                np.repeat(np.arange(n), counts), weights=contrib, minlength=n
            )
        if qb.gamma:
            scores += qb.gamma * qb.recency_vector(batch.ts)

        mask = live.copy()
        cols = dict(batch.numeric)
        cols.update(batch.categorical)
        for p in predicates:
            mask &= p.mask(cols, slice(0, n))
        scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        hits = [
            Hit(row=int(i), row_id=int(batch.row_id[i]), score=float(scores[i]), segment=-1)
            for i in top
            if np.isfinite(scores[i])
        ]
        return hits, SearchStats(rows_scored=int(mask.sum()), rows_total=n)

    # ---------------------------------------------------- context retrieval

    def context(
        self,
        qb: QueryBounds,
        budget_tokens: int = 2000,
        pool: int = 64,
        predicates: Sequence[object] = (),
        epsilon: float = 0.0,
        bridge_gap: int = 1,
    ) -> QueryResult:
        """Retrieve a budgeted, de-duplicated, provenance-carrying context."""
        res = self.search(qb, k=pool, predicates=predicates, epsilon=epsilon)
        units = [self._unit(h) for h in res.hits]
        neighbours: Dict[Tuple[int, int], Unit] = {}
        for u in units:
            neighbours[(u.doc_id, u.ordinal)] = u
        for u in list(units):                      # make bridging possible
            for o in (u.ordinal - 1, u.ordinal + 1):
                nb = self._lookup(u.doc_id, o)
                if nb is not None:
                    neighbours.setdefault((u.doc_id, o), nb)
        res.context = assemble(units, budget_tokens, bridge_gap=bridge_gap,
                               neighbours=neighbours)
        return res

    def _batch_of(self, segment_id: int) -> RowBatch:
        if segment_id == -1:
            return self._mem
        for s in self.segments:
            if s.segment_id == segment_id:
                return s.batch
        raise KeyError(segment_id)

    def _unit(self, hit: Hit) -> Unit:
        b = self._batch_of(hit.segment)
        i = hit.row
        return Unit(
            row_id=int(b.row_id[i]),
            doc_id=int(b.doc_id[i]),
            ordinal=int(b.ordinal[i]),
            tokens=int(b.tokens[i]),
            score=float(hit.score),
            vec=b.vecs[i],
            text=b.text[i] if b.text is not None else "",
        )

    def _lookup(self, doc_id: int, ordinal: int) -> Optional[Unit]:
        if ordinal < 0:
            return None
        for b in ([self._mem] if self._mem is not None else []) + [s.batch for s in self.segments]:
            m = np.nonzero((b.doc_id == doc_id) & (b.ordinal == ordinal))[0]
            if len(m):
                i = int(m[0])
                return Unit(
                    row_id=int(b.row_id[i]), doc_id=doc_id, ordinal=ordinal,
                    tokens=int(b.tokens[i]), score=0.0, vec=b.vecs[i],
                    text=b.text[i] if b.text is not None else "",
                )
        return None

    # -------------------------------------------------------------- CairnQL

    def execute(self, sql: str, params: Optional[Dict[str, object]] = None,
                now: Optional[float] = None) -> QueryResult:
        """Parse and run a CairnQL statement."""
        from .ql import parse

        q = parse(sql)
        now = now if now is not None else time.time()
        qb = q.bind(params or {}, now=now)
        if q.mode == "context":
            return self.context(
                qb, budget_tokens=q.budget_tokens, pool=max(q.k, 64),
                predicates=q.predicates, epsilon=q.epsilon,
            )
        return self.search(
            qb, k=q.k, predicates=q.predicates,
            epsilon=q.epsilon, block_budget=q.block_budget,
        )

    # ------------------------------------------------------------ introspect

    def stats(self) -> Dict[str, float]:
        rows = sum(s.live_count() for s in self.segments)
        rows += 0 if self._mem is None else int(self._mem_live.sum())
        blocks = sum(s.num_blocks for s in self.segments)
        summary_bytes = sum(s.summary_bytes() for s in self.segments)
        return {
            "rows": rows,
            "segments": len(self.segments),
            "blocks": blocks,
            "memtable_rows": 0 if self._mem is None else len(self._mem),
            "summary_bytes": summary_bytes,
            "summary_bytes_per_row": summary_bytes / max(1, rows),
        }
