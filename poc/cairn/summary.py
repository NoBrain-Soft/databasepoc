"""Mergeable sound summaries -- the bound algebra at the heart of Cairn.

A ``Summary`` describes a *set of rows* (a block, a sub-tree of blocks, or a
facet-restricted subset of a block) with enough information to compute a
provable **upper bound** on the fused relevance score of any row in that set,
without touching the rows themselves.

Two properties make the whole engine work:

1.  **Soundness.**  ``S.upper_bound(query) >= max(score(query, row))`` for every
    row covered by ``S``.  Pruning a node whose bound falls below the current
    k-th best score therefore cannot change the answer.
2.  **Mergeability.**  Summaries form a commutative monoid under ``merge``:
    ``merge(A, B)`` covers exactly the rows covered by A and B, and its bound
    dominates both.  That is what lets us stack summaries into a tree (the
    "cairn"), rebuild them during compaction, and combine them across segments,
    all while preserving property 1.

Deletion is a *free* operation for soundness: removing rows from a set can only
lower the true maximum, so an existing bound stays valid (it merely loosens).
Insertion is the only operation that can invalidate a bound, which is precisely
why new rows land in a mutable memtable instead of a sealed segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# A block/node that carries more than this many distinct values for a
# categorical column stops tracking them individually and degrades to "ANY",
# which is still sound (it just answers UNKNOWN more often).
MAX_TRACKED_CATEGORIES = 64

# Number of terms retained in a sparse max-impact summary.  Terms outside the
# retained set are represented by a single residual bound, which keeps the
# summary small *without* giving up soundness (unlike pure static pruning).
DEFAULT_SPARSE_LAMBDA = 48

# Bloom parameters for the per-summary term filter.  256 bytes per block holds
# a few hundred terms at well under 1% false-positive rate.
BLOOM_BITS = 2048
BLOOM_K = 4


class TermFilter:
    """Bloom filter over the term ids present in a summary's rows.

    It is used only to answer "could this term appear here?".  A false positive
    charges the residual for a term that is absent, which *over*-estimates the
    bound -- still sound.  There are no false negatives, so the bound can never
    be under-estimated.  Merging is a bitwise OR, which keeps summaries a
    monoid; at internal nodes the filter saturates and the bound degrades
    gracefully back to charging every unmatched term.
    """

    __slots__ = ("bits",)

    def __init__(self, bits: Optional[np.ndarray] = None):
        self.bits = np.zeros(BLOOM_BITS // 8, dtype=np.uint8) if bits is None else bits

    @staticmethod
    def build(terms: Iterable[int]) -> "TermFilter":
        f = TermFilter()
        for t in terms:
            for h in _hashes(int(t)):
                f.bits[h >> 3] |= np.uint8(1 << (h & 7))
        return f

    def maybe_contains(self, t: int) -> bool:
        for h in _hashes(int(t)):
            if not (self.bits[h >> 3] >> (h & 7)) & 1:
                return False
        return True

    def merge(self, other: "TermFilter") -> "TermFilter":
        return TermFilter(np.bitwise_or(self.bits, other.bits))

    def saturation(self) -> float:
        return float(np.unpackbits(self.bits).mean())

    def nbytes(self) -> int:
        return int(self.bits.nbytes)


def _hashes(t: int):
    # Kirsch-Mitzenmacher double hashing: two independent hashes generate k.
    h1 = (t * 0x9E3779B1) & 0xFFFFFFFF
    h2 = ((t ^ 0x85EBCA6B) * 0xC2B2AE35) & 0xFFFFFFFF
    for i in range(BLOOM_K):
        yield (h1 + i * h2) % BLOOM_BITS


class TriState:
    """Result of evaluating a predicate against a summary."""

    NONE = 0   # provably no row in this set matches
    SOME = 1   # unknown; rows must be checked individually
    ALL = 2    # provably every row in this set matches


@dataclass
class NumericStats:
    lo: float
    hi: float

    def merge(self, other: "NumericStats") -> "NumericStats":
        return NumericStats(min(self.lo, other.lo), max(self.hi, other.hi))


@dataclass
class CategoryStats:
    """Distinct values, or ``None`` meaning "too many to track" (ANY)."""

    values: Optional[frozenset]

    def merge(self, other: "CategoryStats") -> "CategoryStats":
        if self.values is None or other.values is None:
            return CategoryStats(None)
        union = self.values | other.values
        if len(union) > MAX_TRACKED_CATEGORIES:
            return CategoryStats(None)
        return CategoryStats(frozenset(union))


@dataclass
class SparseSummary:
    """Block-max impact vector with a *sound* residual for pruned terms.

    ``kept`` maps term id -> max weight of that term over the covered rows.
    ``residual`` is an upper bound on the weight of any term *not* in ``kept``.
    A query term missing from ``kept`` is therefore charged ``residual`` rather
    than being silently dropped, so the bound stays an upper bound even though
    the summary is aggressively pruned.
    """

    kept: Dict[int, float]
    residual: float
    max_row_norm: float = 0.0        # max L2 norm of any covered row's sparse vector
    terms: Optional[TermFilter] = None

    @staticmethod
    def from_max_vector(
        maxvec: Dict[int, float],
        lam: int = DEFAULT_SPARSE_LAMBDA,
        max_row_norm: float = 0.0,
        with_filter: bool = True,
    ) -> "SparseSummary":
        filt = (TermFilter.build(maxvec.keys()) if maxvec else TermFilter()) if with_filter else None
        if len(maxvec) <= lam:
            return SparseSummary(dict(maxvec), 0.0, max_row_norm, filt)
        ordered = sorted(maxvec.items(), key=lambda kv: kv[1], reverse=True)
        kept = dict(ordered[:lam])
        residual = ordered[lam][1] if len(ordered) > lam else 0.0
        return SparseSummary(kept, float(residual), max_row_norm, filt)

    def merge(self, other: "SparseSummary", lam: int = DEFAULT_SPARSE_LAMBDA) -> "SparseSummary":
        combined: Dict[int, float] = dict(self.kept)
        for t, w in other.kept.items():
            if w > combined.get(t, 0.0):
                combined[t] = w
        residual = max(self.residual, other.residual)
        norm = max(self.max_row_norm, other.max_row_norm)
        filt = self.terms.merge(other.terms) if (self.terms and other.terms) else (
            self.terms or other.terms
        )
        if len(combined) <= lam:
            return SparseSummary(combined, residual, norm, filt)
        ordered = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        kept = dict(ordered[:lam])
        # Demoted terms must be absorbed into the residual to stay sound.
        residual = max(residual, ordered[lam][1])
        return SparseSummary(kept, float(residual), norm, filt)

    def upper_bound(self, query: Dict[int, float], query_norm: float = 0.0) -> float:
        """Tightest of two sound bounds.

        *Impact bound*: sum of per-term block maxima, the classic block-max
        (WAND/BMW) bound, plus the residual charge for pruned terms.  It is
        loose because each term's maximum may be attained by a different row.

        *Cauchy-Schwarz bound*: ||q|| * max_row ||w_row||.  It costs four bytes
        per block and is far tighter for queries with many terms, where the
        impact bound accumulates one slack term per query term.
        """
        if not query:
            return 0.0
        total = 0.0
        rest_mass = 0.0
        for t, qw in query.items():
            w = self.kept.get(t)
            if w is None:
                # Only terms that might actually be present cost the residual.
                if self.residual > 0.0 and (self.terms is None or self.terms.maybe_contains(t)):
                    rest_mass += qw
            else:
                total += qw * w
        impact = total + rest_mass * self.residual
        if self.max_row_norm > 0.0 and query_norm > 0.0:
            return min(impact, query_norm * self.max_row_norm)
        return impact


@dataclass
class Summary:
    """Everything needed to bound a set of rows without reading them."""

    count: int
    centroid: np.ndarray            # unit vector, float32
    ang_radius: float               # max angle (radians) from centroid to any member
    sparse: SparseSummary
    max_ts: float
    numeric: Dict[str, NumericStats] = field(default_factory=dict)
    categorical: Dict[str, CategoryStats] = field(default_factory=dict)
    # Optional per-facet-value refinements of this summary (see design doc,
    # "faceted zone maps").  Keyed by (column, value).
    facets: Dict[Tuple[str, int], "Summary"] = field(default_factory=dict)
    # Columns whose facets *completely* partition the rows this summary covers.
    # A facet may only be used to bound a filtered query when its column is
    # complete here: an incomplete facet would silently omit rows, which is the
    # one way a sound bound can be made unsound by merging.
    facet_cols: frozenset = frozenset()

    # ---------------------------------------------------------------- bounds

    def dense_upper_bound(self, q: np.ndarray) -> float:
        """Max possible cosine between ``q`` and any member vector.

        Members lie inside a spherical cap of angular radius ``ang_radius``
        around ``centroid``.  By the triangle inequality for the geodesic
        metric on the unit sphere, the angle between ``q`` and any member is at
        least ``theta - ang_radius`` where ``theta = angle(q, centroid)``.
        ``cos`` is decreasing on ``[0, pi]``, so the bound follows and is tight
        whenever a member sits on the geodesic from the centroid toward ``q``.
        """
        dot = float(np.dot(q, self.centroid))
        theta = math.acos(min(1.0, max(-1.0, dot)))
        return math.cos(max(0.0, theta - self.ang_radius))

    def upper_bound(self, query: "QueryBounds") -> float:
        ub = 0.0
        if query.alpha and query.dense is not None:
            ub += query.alpha * self.dense_upper_bound(query.dense)
        if query.beta and query.sparse:
            ub += query.beta * self.sparse.upper_bound(query.sparse, query.sparse_norm)
        if query.gamma:
            ub += query.gamma * query.recency(self.max_ts)
        return ub

    # ------------------------------------------------------------ predicates

    def filtered_upper_bound(self, query: "QueryBounds", preds: Sequence["Predicate"]) -> float:
        """Best (smallest) sound bound available for a filtered query.

        A facet sub-summary covers only the rows that can survive the filter, so
        its bound is sound for the filtered query -- but it is not *always*
        tighter than the block's own bound (a narrow subset can sit closer to
        the query than the block centroid does, and the cap bound is not
        monotone under subsetting).  Both bounds are sound, so take the min.
        """
        ub = self.upper_bound(query)
        ref = self.refine(preds)
        if ref is not self:
            ub = min(ub, ref.upper_bound(query))
        return ub

    def evaluate(self, preds: Sequence["Predicate"]) -> int:
        """Three-valued evaluation of a conjunction of predicates."""
        verdict = TriState.ALL
        for p in preds:
            v = p.evaluate(self)
            if v == TriState.NONE:
                return TriState.NONE
            if v == TriState.SOME:
                verdict = TriState.SOME
        return verdict

    def refine(self, preds: Sequence["Predicate"]) -> "Summary":
        """Return the tightest facet summary implied by ``preds``, else self.

        When a query pins a facet column to a single value and this summary
        tracks that value, the facet sub-summary bounds only the rows that can
        actually survive the filter -- which is what makes filtered search fast
        instead of merely correct.
        """
        best = self
        for p in preds:
            key = p.facet_key()
            if key is not None and key[0] in self.facet_cols and key in self.facets:
                sub = self.facets[key]
                if sub.count < best.count:
                    best = sub
        return best

    # ----------------------------------------------------------------- merge

    def merge(self, other: "Summary", lam: int = DEFAULT_SPARSE_LAMBDA) -> "Summary":
        n = self.count + other.count
        # Weighted mean of the two centroids, renormalised.  Any unit vector
        # would keep the result sound as long as the radius is recomputed
        # against it; the weighted mean simply keeps the radius small.
        mean = self.centroid * self.count + other.centroid * other.count
        norm = float(np.linalg.norm(mean))
        centroid = (mean / norm).astype(np.float32) if norm > 1e-12 else self.centroid
        radius = max(
            _angle(centroid, self.centroid) + self.ang_radius,
            _angle(centroid, other.centroid) + other.ang_radius,
        )
        radius = min(radius, math.pi)

        numeric = {}
        for k in set(self.numeric) | set(other.numeric):
            a, b = self.numeric.get(k), other.numeric.get(k)
            numeric[k] = a.merge(b) if a and b else (a or b)
        categorical = {}
        for k in set(self.categorical) | set(other.categorical):
            a, b = self.categorical.get(k), other.categorical.get(k)
            categorical[k] = a.merge(b) if a and b else (a or b)

        # A facet key missing from one side means that side has *no* rows with
        # that value -- but only if the column is complete on both sides.  If it
        # is not, the merged facet would omit rows and the bound would stop
        # being an upper bound, so the column is dropped instead.
        facet_cols = self.facet_cols & other.facet_cols
        facets: Dict[Tuple[str, int], Summary] = {}
        for key in set(self.facets) | set(other.facets):
            if key[0] not in facet_cols:
                continue
            a, b = self.facets.get(key), other.facets.get(key)
            facets[key] = a.merge(b, lam) if (a and b) else (a or b)
        for col in list(facet_cols):
            if sum(1 for k in facets if k[0] == col) > MAX_TRACKED_CATEGORIES:
                facet_cols = facet_cols - {col}
                facets = {k: v for k, v in facets.items() if k[0] != col}

        return Summary(
            count=n,
            centroid=centroid,
            ang_radius=radius,
            sparse=self.sparse.merge(other.sparse, lam),
            max_ts=max(self.max_ts, other.max_ts),
            numeric=numeric,
            categorical=categorical,
            facets=facets,
            facet_cols=facet_cols,
        )

    def nbytes(self) -> int:
        """Approximate on-disk footprint, for the sizing model in the docs."""
        n = self.centroid.nbytes + 8 + 8 + 8
        n += len(self.sparse.kept) * 6 + 4
        n += self.sparse.terms.nbytes() if self.sparse.terms is not None else 0
        n += len(self.numeric) * 16 + len(self.categorical) * 8
        for sub in self.facets.values():
            n += sub.nbytes()
        return n


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    return math.acos(min(1.0, max(-1.0, dot)))


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------


@dataclass
class Range:
    column: str
    lo: float = -math.inf
    hi: float = math.inf

    def evaluate(self, s: Summary) -> int:
        st = s.numeric.get(self.column)
        if st is None:
            return TriState.SOME
        if st.hi < self.lo or st.lo > self.hi:
            return TriState.NONE
        if st.lo >= self.lo and st.hi <= self.hi:
            return TriState.ALL
        return TriState.SOME

    def mask(self, cols: Dict[str, np.ndarray], rows: slice) -> np.ndarray:
        v = cols[self.column][rows]
        return (v >= self.lo) & (v <= self.hi)

    def facet_key(self):
        return None


@dataclass
class InSet:
    column: str
    values: frozenset

    def evaluate(self, s: Summary) -> int:
        st = s.categorical.get(self.column)
        if st is None or st.values is None:
            return TriState.SOME
        if not (st.values & self.values):
            return TriState.NONE
        if st.values <= self.values:
            return TriState.ALL
        return TriState.SOME

    def mask(self, cols: Dict[str, np.ndarray], rows: slice) -> np.ndarray:
        v = cols[self.column][rows]
        return np.isin(v, np.fromiter(self.values, dtype=v.dtype, count=len(self.values)))

    def facet_key(self):
        if len(self.values) == 1:
            return (self.column, next(iter(self.values)))
        return None


Predicate = object  # structural: .evaluate(Summary), .mask(cols, slice), .facet_key()


# --------------------------------------------------------------------------
# Query-side bundle
# --------------------------------------------------------------------------


@dataclass
class QueryBounds:
    """The fused scoring function, in the form the bound algebra consumes.

    Fusion is a *linear* combination on purpose.  Rank-fusion schemes such as
    RRF are not prunable: a document's contribution depends on its rank in each
    list, which is unknown until every list is fully materialised.  A calibrated
    linear combination decomposes into per-component bounds that sum, which is
    exactly what a single branch-and-bound traversal needs.
    """

    dense: Optional[np.ndarray] = None
    sparse: Dict[int, float] = field(default_factory=dict)
    _norm: Optional[float] = field(default=None, repr=False)
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    now: float = 0.0
    half_life: float = 86400.0 * 30

    @property
    def sparse_norm(self) -> float:
        if not self.sparse:
            return 0.0
        if self._norm is None:
            self._norm = math.sqrt(sum(w * w for w in self.sparse.values()))
        return self._norm

    def recency(self, ts: float) -> float:
        """Monotone non-decreasing in ``ts`` -- so ``max_ts`` bounds it."""
        age = max(0.0, self.now - ts)
        return 0.5 ** (age / self.half_life)

    def recency_vector(self, ts: np.ndarray) -> np.ndarray:
        age = np.maximum(0.0, self.now - ts)
        return np.power(0.5, age / self.half_life)
