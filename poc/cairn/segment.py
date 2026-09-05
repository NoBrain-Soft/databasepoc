"""Sealed segments: the immutable, block-laid-out unit of storage.

A segment holds rows in a *physical order chosen by the layout algorithm* so
that every block (a contiguous range of rows) is simultaneously

  * geometrically cohesive in embedding space -- small angular radius, hence a
    tight dense bound, and
  * predicate-cohesive -- narrow zone maps, hence many blocks answer NONE/ALL
    instead of SOME.

This is the semantic generalisation of the classic block-skipping layout
problem (min/max zone maps, qd-trees): the layout is optimised for the number
of blocks a query must open, except the query now has a vector component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .summary import (
    CategoryStats,
    NumericStats,
    SparseSummary,
    Summary,
    DEFAULT_SPARSE_LAMBDA,
    MAX_TRACKED_CATEGORIES,
)


@dataclass
class RowBatch:
    """Column-major row payload.  Vectors are L2-normalised on ingest."""

    vecs: np.ndarray                       # (n, d) float32, unit norm
    ts: np.ndarray                         # (n,) float64
    tokens: np.ndarray                     # (n,) int32 -- length of the unit in tokens
    doc_id: np.ndarray                     # (n,) int64
    ordinal: np.ndarray                    # (n,) int32 -- position of the unit within its doc
    row_id: np.ndarray                     # (n,) int64 -- globally unique, monotone
    numeric: Dict[str, np.ndarray] = field(default_factory=dict)
    categorical: Dict[str, np.ndarray] = field(default_factory=dict)
    # Sparse (learned-sparse / lexical) side, CSR-style.
    sp_indptr: Optional[np.ndarray] = None
    sp_indices: Optional[np.ndarray] = None
    sp_data: Optional[np.ndarray] = None
    text: Optional[List[str]] = None

    def __len__(self) -> int:
        return int(self.vecs.shape[0])

    def take(self, order: np.ndarray) -> "RowBatch":
        """Materialise the rows in a new physical order."""
        sp_indptr = sp_indices = sp_data = None
        if self.sp_indptr is not None:
            counts = np.diff(self.sp_indptr)[order]
            sp_indptr = np.zeros(len(order) + 1, dtype=np.int64)
            np.cumsum(counts, out=sp_indptr[1:])
            idx_parts, dat_parts = [], []
            for r in order:
                a, b = self.sp_indptr[r], self.sp_indptr[r + 1]
                idx_parts.append(self.sp_indices[a:b])
                dat_parts.append(self.sp_data[a:b])
            sp_indices = (
                np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int32)
            )
            sp_data = (
                np.concatenate(dat_parts) if dat_parts else np.zeros(0, dtype=np.float32)
            )
        return RowBatch(
            vecs=self.vecs[order],
            ts=self.ts[order],
            tokens=self.tokens[order],
            doc_id=self.doc_id[order],
            ordinal=self.ordinal[order],
            row_id=self.row_id[order],
            numeric={k: v[order] for k, v in self.numeric.items()},
            categorical={k: v[order] for k, v in self.categorical.items()},
            sp_indptr=sp_indptr,
            sp_indices=sp_indices,
            sp_data=sp_data,
            text=[self.text[r] for r in order] if self.text is not None else None,
        )

    def concat(self, other: "RowBatch") -> "RowBatch":
        sp_indptr = sp_indices = sp_data = None
        if self.sp_indptr is not None and other.sp_indptr is not None:
            sp_indptr = np.concatenate(
                [self.sp_indptr, other.sp_indptr[1:] + self.sp_indptr[-1]]
            )
            sp_indices = np.concatenate([self.sp_indices, other.sp_indices])
            sp_data = np.concatenate([self.sp_data, other.sp_data])
        text = None
        if self.text is not None and other.text is not None:
            text = self.text + other.text
        return RowBatch(
            vecs=np.concatenate([self.vecs, other.vecs]),
            ts=np.concatenate([self.ts, other.ts]),
            tokens=np.concatenate([self.tokens, other.tokens]),
            doc_id=np.concatenate([self.doc_id, other.doc_id]),
            ordinal=np.concatenate([self.ordinal, other.ordinal]),
            row_id=np.concatenate([self.row_id, other.row_id]),
            numeric={k: np.concatenate([v, other.numeric[k]]) for k, v in self.numeric.items()},
            categorical={
                k: np.concatenate([v, other.categorical[k]]) for k, v in self.categorical.items()
            },
            sp_indptr=sp_indptr,
            sp_indices=sp_indices,
            sp_data=sp_data,
            text=text,
        )


# --------------------------------------------------------------------------
# Layout: bisecting spherical k-means
# --------------------------------------------------------------------------


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def bisect(vecs: np.ndarray, idx: np.ndarray, rng: np.random.Generator, iters: int = 8):
    """Split ``idx`` into two spherically-coherent halves."""
    if len(idx) < 2:
        return idx, idx[:0]
    sub = vecs[idx]
    a, b = rng.choice(len(idx), size=2, replace=False)
    ca, cb = sub[a].copy(), sub[b].copy()
    assign = np.zeros(len(idx), dtype=bool)
    for _ in range(iters):
        assign = (sub @ ca) >= (sub @ cb)
        if assign.all() or (~assign).any() is False:
            break
        na, nb = int(assign.sum()), int((~assign).sum())
        if na == 0 or nb == 0:
            break
        ca = _normalize(sub[assign].mean(axis=0))
        cb = _normalize(sub[~assign].mean(axis=0))
    left, right = idx[assign], idx[~assign]
    if len(left) == 0 or len(right) == 0:  # degenerate: fall back to an even cut
        half = len(idx) // 2
        left, right = idx[:half], idx[half:]
    return left, right


def angular_radius(vecs: np.ndarray, idx: np.ndarray) -> float:
    """Angle from the group centroid to its furthest member."""
    sub = vecs[idx]
    c = _normalize(sub.mean(axis=0))
    return float(np.arccos(np.clip(sub @ c, -1.0, 1.0).min()))


def plan_blocks(
    vecs: np.ndarray,
    block_size: int,
    partition_by: Optional[np.ndarray] = None,
    max_radius: Optional[float] = None,
    min_block: int = 16,
    seed: int = 0,
) -> List[np.ndarray]:
    """Return a list of row-index groups, each of at most ``block_size`` rows.

    ``partition_by`` (a categorical column) is honoured first: rows with
    different values never share a block.  This is the cheap, high-value half
    of query-conditioned layout -- it makes a filter on that column prune whole
    blocks instead of merely masking rows inside them.

    ``max_radius`` makes the layout target *bound quality* rather than block
    size alone: a group keeps splitting while its angular radius exceeds the
    target, because a single outlier is enough to make a block's dense bound
    vacuous (the bound is set by the worst member, not the average one).
    Blocks therefore come out variable-sized, which is what the bound algebra
    wants and what a real columnar writer can emit anyway.
    """
    rng = np.random.default_rng(seed)
    groups: List[np.ndarray] = []
    if partition_by is not None:
        roots = [np.where(partition_by == v)[0] for v in np.unique(partition_by)]
    else:
        roots = [np.arange(len(vecs))]

    for root in roots:
        stack = [root]
        while stack:
            cur = stack.pop()
            too_big = len(cur) > block_size
            too_wide = (
                max_radius is not None
                and len(cur) > max(min_block, 2)
                and angular_radius(vecs, cur) > max_radius
            )
            if not (too_big or too_wide):
                if len(cur):
                    groups.append(cur)
                continue
            left, right = bisect(vecs, cur, rng)
            if len(left) == 0 or len(right) == 0:
                groups.append(cur)
                continue
            stack.append(left)
            stack.append(right)
    return groups


# --------------------------------------------------------------------------
# Summary construction
# --------------------------------------------------------------------------


def summarize(
    batch: RowBatch,
    lo: int,
    hi: int,
    facet_columns: Sequence[str] = (),
    lam: int = DEFAULT_SPARSE_LAMBDA,
    term_filter: bool = True,
    _depth: int = 0,
) -> Summary:
    """Build the summary of rows ``[lo, hi)`` of ``batch``."""
    vecs = batch.vecs[lo:hi]
    centroid = _normalize(vecs.mean(axis=0))
    cos = np.clip(vecs @ centroid, -1.0, 1.0)
    ang_radius = float(np.arccos(cos.min()))

    maxvec: Dict[int, float] = {}
    max_row_norm = 0.0
    if batch.sp_indptr is not None:
        a, b = int(batch.sp_indptr[lo]), int(batch.sp_indptr[hi])
        idx, dat = batch.sp_indices[a:b], batch.sp_data[a:b]
        if len(idx):
            order = np.argsort(idx, kind="stable")
            idx_s, dat_s = idx[order], dat[order]
            uniq, starts = np.unique(idx_s, return_index=True)
            maxima = np.maximum.reduceat(dat_s, starts)
            maxvec = {int(t): float(w) for t, w in zip(uniq, maxima)}
            counts = np.diff(batch.sp_indptr[lo:hi + 1])
            rowids = np.repeat(np.arange(hi - lo), counts)
            sq = np.bincount(rowids, weights=dat.astype(np.float64) ** 2,
                             minlength=hi - lo)
            max_row_norm = float(np.sqrt(sq.max())) if len(sq) else 0.0

    numeric = {k: NumericStats(float(v[lo:hi].min()), float(v[lo:hi].max()))
               for k, v in batch.numeric.items()}
    categorical = {}
    for k, v in batch.categorical.items():
        vals = np.unique(v[lo:hi])
        categorical[k] = CategoryStats(
            frozenset(int(x) for x in vals) if len(vals) <= MAX_TRACKED_CATEGORIES else None
        )

    facets: Dict[Tuple[str, int], Summary] = {}
    facet_cols = set()
    if _depth == 0:
        for col in facet_columns:
            v = batch.categorical.get(col)
            if v is None:
                continue
            vals = np.unique(v[lo:hi])
            if len(vals) > MAX_TRACKED_CATEGORIES:
                continue
            # Every value present gets a facet, singletons included: facets have
            # to partition the block completely or they cannot survive a merge
            # (see Summary.merge).
            for val in vals:
                sel = np.where(v[lo:hi] == val)[0] + lo
                facets[(col, int(val))] = _summarize_rows(
                    batch, sel, lam, term_filter, _depth=1
                )
            facet_cols.add(col)

    return Summary(
        count=hi - lo,
        centroid=centroid,
        ang_radius=ang_radius,
        sparse=SparseSummary.from_max_vector(maxvec, lam, max_row_norm, term_filter),
        max_ts=float(batch.ts[lo:hi].max()),
        numeric=numeric,
        categorical=categorical,
        facets=facets,
        facet_cols=frozenset(facet_cols),
    )


def _summarize_rows(batch: RowBatch, rows: np.ndarray, lam: int,
                    term_filter: bool, _depth: int) -> Summary:
    """Summarise an arbitrary (non-contiguous) row set -- used for facets."""
    vecs = batch.vecs[rows]
    centroid = _normalize(vecs.mean(axis=0))
    cos = np.clip(vecs @ centroid, -1.0, 1.0)
    maxvec: Dict[int, float] = {}
    max_row_norm = 0.0
    if batch.sp_indptr is not None:
        for r in rows:
            a, b = int(batch.sp_indptr[r]), int(batch.sp_indptr[r + 1])
            dat = batch.sp_data[a:b]
            max_row_norm = max(max_row_norm, float(np.sqrt((dat.astype(np.float64) ** 2).sum())))
            for t, w in zip(batch.sp_indices[a:b], dat):
                t = int(t)
                if w > maxvec.get(t, 0.0):
                    maxvec[t] = float(w)
    return Summary(
        count=len(rows),
        centroid=centroid,
        ang_radius=float(np.arccos(cos.min())),
        sparse=SparseSummary.from_max_vector(maxvec, lam, max_row_norm, term_filter),
        max_ts=float(batch.ts[rows].max()),
        numeric={k: NumericStats(float(v[rows].min()), float(v[rows].max()))
                 for k, v in batch.numeric.items()},
        categorical={
            k: CategoryStats(frozenset(int(x) for x in np.unique(v[rows]))
                             if len(np.unique(v[rows])) <= MAX_TRACKED_CATEGORIES else None)
            for k, v in batch.categorical.items()
        },
    )
