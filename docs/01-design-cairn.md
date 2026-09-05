# Cairn: a context-searchable database

> A *cairn* is a stack of stones that marks a path. In this system it is a stack
> of summaries: each one bounds everything beneath it, so a query can walk
> straight to the data that matters and prove that it skipped nothing.

Cairn is a design for a database whose primary access path is **"give me the
context relevant to this, under these constraints, within this budget"** over
corpora too large to scan. It exists because the six constraints derived in the
[research survey](00-research-survey.md) are not jointly satisfied by any
published system.

**The thesis in one sentence:** if every physical block carries a *mergeable,
sound summary* that can upper-bound a fused relevance score under a predicate,
then hybrid search, filtering, freshness and provable answers all collapse into
a single branch-and-bound traversal over one index.

Everything else in this document follows from that sentence.

---

## 1. Goals and non-goals

**Goals**

1. One index, one traversal, for dense + lexical + scalar + temporal predicates.
2. Exact top-k by default; approximation as an explicit, *certified* choice.
3. Filters that prune I/O rather than merely masking rows.
4. Streaming ingest and deletes with no index rebuild and no recall cliff.
5. The answer unit is an assembled, budgeted, de-duplicated **context**.
6. Petabyte-scale economics: summaries in RAM, codes on NVMe, payload in object
   storage.

**Non-goals**

- General-purpose OLTP. Cairn has transactions, but row-level update throughput
  is not what it optimises.
- Beating a tuned HNSW on unfiltered, static, single-vector recall@10 benchmarks.
  It will not; that is a solved problem and a graph index will win on raw work
  per query. Cairn trades some of that for guarantees, filters and freshness.
- Training embeddings. Embedding is the caller's concern; Cairn stores and
  indexes whatever vectors it is given.

---

## 2. Data model

The unit of storage is a **unit**: a short, contiguous, provenance-carrying span
of a document — typically a sentence or a short passage.

```
unit {
  row_id      u64            -- globally unique, monotone with commit order
  doc_id      u64            -- the document this unit belongs to
  ordinal     u32            -- position of the unit within the document
  span        (u32, u32)     -- byte offsets into the document
  vec         f32[d]         -- dense embedding, L2-normalised
  sparse      {term -> w}    -- learned-sparse or BM25 impacts, non-negative
  tokens      u16            -- length in model tokens, for budgeting
  valid_time  (t0, t1)       -- when the fact was true in the world
  system_time (s0, s1)       -- when the database believed it
  columns     {name -> scalar}
}
```

Three deliberate choices:

**Units, not chunks.** Chunk boundaries are the single most consequential
retrieval decision, and chunking at ingest makes it before the query is known.
Cairn indexes small units and assembles spans at query time (§9). The document
remains the unit of *provenance*; the unit is only the unit of *scoring*.

**Bitemporality is not optional.** Agent memory is continuously revised; "what
did we believe on Tuesday" and "what was true on Tuesday" are different
questions and both get asked. Both time axes are ordinary indexed columns, so
they participate in the same pruning path as any other predicate.

**Sparse weights are non-negative.** This is a real restriction and it buys the
monotonicity that every sparse bound in §4 depends on. Learned sparse models
(SPLADE-family) and BM25 impacts satisfy it.

---

## 3. The scoring function

```
score(q, x) = α·⟨q_dense, x.vec⟩ + β·Σ_t q_sparse[t]·x.sparse[t] + γ·g(x.time)
```

with `α, β, γ ≥ 0` and `g` any monotone non-decreasing function of time (the
implementation uses exponential decay with a declared half-life).

This is a **calibrated linear fusion**, and the choice is forced. Reciprocal
Rank Fusion — the industry default — scores a document by its *rank* in each
retriever's list, and a rank cannot be bounded before the list is materialised.
RRF is therefore structurally incompatible with pruning: it requires running
both retrievers to completion. Linear fusion decomposes:

> **Lemma 0.** If `s = Σ_j w_j·s_j` with `w_j ≥ 0` and `U_j ≥ max_x s_j(x)`,
> then `Σ_j w_j·U_j ≥ max_x s(x)`.

The cost is calibration: component scores must be on comparable scales, which is
a per-corpus fitting step (§13), not a per-query one. The benefit is that a
single traversal answers the whole query.

---

## 4. The bound algebra

A **summary** describes a set of rows and can upper-bound their score without
reading them.

```
Summary {
  count        u32
  centroid     unit vector c
  ang_radius   ρ = max_{x∈S} arccos(⟨c, x.vec⟩)
  sparse       { top-λ term maxima m_t ; residual r ; ‖·‖ cap ; Bloom filter F }
  max_time     max_{x∈S} x.time
  zone_maps    per numeric column (lo, hi); per categorical column, the distinct
               set if small, else ANY
  facets       optional sub-summaries keyed by (column, value)
}
```

### 4.1 The dense bound

> **Theorem 1.** For unit vectors, let `θ = arccos(⟨q, c⟩)`. Then for every
> `x ∈ S`, `⟨q, x⟩ ≤ cos(max(0, θ − ρ))`.
>
> *Proof.* Geodesic distance `d(a,b) = arccos(⟨a,b⟩)` is a metric on the sphere,
> so `d(q,x) ≥ d(q,c) − d(c,x) ≥ θ − ρ`. Also `d(q,x) ≥ 0`. Since `cos` is
> decreasing on `[0, π]`, `⟨q,x⟩ = cos(d(q,x)) ≤ cos(max(0, θ − ρ))`. ∎

The bound is tight: it is attained when a member lies on the geodesic from `c`
toward `q`. This is classical metric-tree pruning (M-tree, ball trees), and its
classical objection applies — see §14.

For un-normalised inner product, store `max‖x‖` and multiply; MIPS reduces to
this case.

### 4.2 The sparse bound

Two sound bounds, and the engine takes the smaller.

> **Theorem 2a (filtered impact bound).** Let `K` be the retained terms with
> maxima `m_t`, `r` an upper bound on the weight of any non-retained term, and
> `F` a Bloom filter over the terms occurring in `S`. Then for every `x ∈ S`
> with non-negative weights,
>
> `⟨q, x.sparse⟩ ≤ Σ_{t∈K} q_t·m_t + r·Σ_{t∉K, F(t)} q_t`.
>
> *Proof.* Split the sum at `K`. Retained terms are bounded by their maxima.
> A non-retained term is bounded by `r` if it occurs in `S`, and is exactly zero
> otherwise; `F` has no false negatives, so `¬F(t)` implies absence. False
> positives merely add slack. ∎

The Bloom filter is the difference between a usable bound and a useless one.
Without it, every query term the summary did not retain costs a full residual
charge, so the bound grows with query length and swamps the dense component.
The [benchmarks](03-results.md) measure this directly: on hybrid queries the
term filter takes rows scored from 91% to 13.5% with no change in the answer.

> **Theorem 2b (norm bound).** `⟨q, x.sparse⟩ ≤ ‖q‖₂ · max_{x∈S} ‖x.sparse‖₂`.
> *Proof.* Cauchy-Schwarz, then maximise over `S`. ∎

Four bytes per summary. It dominates 2a on long queries, where the impact bound
accumulates one slack term per query term.

### 4.3 Predicates: three-valued, at block granularity

Zone maps answer `ALL`, `NONE` or `SOME` for each predicate:

- `NONE` — the block is skipped without reading or scoring anything.
- `ALL` — the block is scored with no per-row predicate evaluation at all.
- `SOME` — the mask is computed for the block's rows.

A conjunction is `NONE` if any conjunct is `NONE`, `ALL` if all are `ALL`, else
`SOME`. This is what makes filters *prune* rather than *mask*, and it is the
same mechanism analytical engines have used for scalar data since Small
Materialized Aggregates — extended here to sit inside the vector traversal.

**Faceted zone maps.** A block whose top-scoring rows all fail the predicate
still advertises a high bound, and the traversal opens it for nothing — the
classic filtered-ANN pathology. Cairn therefore stores, for workload-designated
facet columns, a sub-summary per value: centroid, radius, sparse maxima and
counts computed over *only* the rows with that value. A query pinning that facet
gets a bound describing only rows that can survive the filter.

Note that a facet bound is not automatically tighter than its parent's — a
narrow subset can sit closer to the query than the block centroid does, and the
cap bound is not monotone under subsetting. Both are sound, so the engine uses
`min(parent, facet)`.

### 4.4 Composition and mergeability

> **Theorem 3 (fusion).** `UB(S) = α·UB_dense + β·UB_sparse + γ·g(max_time)`
> upper-bounds `max_{x∈S} score(q,x)`. *Proof.* Lemma 0 with Theorems 1, 2 and
> monotonicity of `g`. ∎

The slack is real: the three maxima may be attained by three different rows.
This is the price of a single traversal, and it is why block cohesion (§6)
matters so much.

> **Theorem 4 (mergeability).** Summaries form a commutative monoid under `⊕`,
> where `⊕` takes the count-weighted centroid `c'`, radius
> `ρ' = max_i (arccos⟨c', c_i⟩ + ρ_i)`, elementwise-max sparse maxima, OR-ed
> Bloom filters, unioned zone maps and `max` of times. `S₁ ⊕ S₂` is sound for
> `S₁ ∪ S₂`, and `UB(S₁ ⊕ S₂) ≥ max(UB(S₁), UB(S₂))`.
>
> *Proof.* For `x` in child `i`: `d(c', x) ≤ d(c', c_i) + d(c_i, x) ≤ ρ'`, so
> Theorem 1 applies to the merged summary; the other components are maxima and
> compose trivially. ∎

Mergeability is the load-bearing property. It is what lets summaries stack into
a tree, be rebuilt during compaction, be combined across segments and shards,
and be computed in parallel — all without ever re-reading the rows.

### 4.5 Why the write path is an LSM

> **Theorem 5 (deletion is free, insertion is not).** Removing rows from `S`
> preserves soundness of any summary of `S`. Adding rows does not.
>
> *Proof.* Every bound is a maximum over `S`; maxima are monotone under
> `⊆`. ∎

So deletes can be tombstones and touch no summary, while an insert into a sealed
block would break the invariant. The only sound options for an insert are to
recompute the summary (which invalidates every ancestor) or to write elsewhere.
Cairn writes elsewhere. **The LSM shape is a consequence of the bound algebra,
not a performance preference** — the same conclusion FreshDiskANN and SPFresh
reached empirically for graph indexes.

---

## 5. Search: one branch-and-bound traversal

```
open a max-heap of nodes, keyed by UB(node)
push the root
while the heap is non-empty:
    pop the node with the largest bound
    if UB(node) ≤ stop_threshold:  break        -- nothing left can win
    if node is a leaf:  score its rows, update the top-k heap
    else:               push children whose predicate verdict ≠ NONE
```

> **Theorem 6 (exactness).** With `stop_threshold = s_k` (the k-th best score
> found so far) the traversal returns the exact top-k.
>
> *Proof.* At the break, the popped node had the largest bound in the heap, so
> every unexpanded node `n` has `UB(n) ≤ s_k`. Every unscored row lies beneath
> some unexpanded node, and by soundness its score is at most that node's bound,
> hence at most `s_k`. No unscored row can enter the top-k. ∎

> **Theorem 7 (the certificate).** Let `U_rem` be the largest bound left
> unexplored when the traversal stops for any reason. Every row not scored has
> score ≤ `U_rem`. Hence the returned k-th result is within a factor
> `U_rem / s_k` of the true k-th, and `U_rem ≤ s_k` proves exactness. ∎

This is the part no deployed vector database offers. Every Cairn answer carries
`(s_k, U_rem, exact)`. A caller can:

- demand `EXACT` and get a proof;
- demand `EPSILON e`, stopping at `s_k·(1+e)`, and get a certified
  approximation ratio;
- cap I/O with `BLOCKS n` and *learn from the certificate* how much the answer
  might be off — instead of discovering it in production.

`efSearch` and `nprobe` are corpus-level guesses. A certificate is a per-query
fact.

### 5.1 Quantised rows

Row payloads are stored as RaBitQ-style codes; summaries keep full precision
because there are orders of magnitude fewer of them. So *pruning arithmetic is
exact* and only row scoring carries error `|ŝ − s| ≤ ε`.

Two adjustments restore the guarantee:

1. **Traversal.** Stop at `s_k − ε` rather than `s_k`, since the true k-th score
   is at least the estimated k-th minus `ε`.
2. **Retention.** Keep every row with `ŝ ≥ ŝ_(k) − 2ε` as a re-ranking
   candidate, then re-score those exactly against full-precision vectors.

> **Theorem 8.** Rule 2 retains every true top-k row. *Proof.* Suppose
> `ŝ_(k) > s*_k + ε`. Then k rows have estimates above `s*_k + ε` and hence true
> scores above `s*_k`, contradicting that `s*_k` is the k-th largest true score.
> So `ŝ_(k) ≤ s*_k + ε`. Any true top-k row `r` has
> `ŝ(r) ≥ s(r) − ε ≥ s*_k − ε ≥ ŝ_(k) − 2ε`. ∎

The catch is that `ε` from a *deterministic* bound is only available for
quantisers that have one — which is precisely why RaBitQ matters and PQ does
not. The prototype calibrates `ε` empirically as a high quantile and labels the
resulting guarantee probabilistic; a production build should use the RaBitQ
bound and inherit a deterministic one.

Note also the trade-off the benchmarks make visible: 1-bit codes at `d=64` give
`ε ≈ 0.3`, which forces such a low stop threshold that the candidate set
explodes. Quantisation pays off at realistic embedding dimensions (`ε` shrinks
as `1/√d`), and the sound engine *exposes* that cost rather than silently
losing recall to it.

---

## 6. Physical layout

A segment is immutable and holds rows in an order chosen by the layout
algorithm; a block is a contiguous row range. Layout decides bound quality, and
bound quality decides everything else.

**Radius-targeted splitting.** Recursive bisecting spherical k-means, splitting
a group while `|group| > block_size` **or** `angular_radius(group) > ρ_max`.
The second condition is the important one: the dense bound is set by a block's
*worst* member, so one outlier makes a whole block's bound vacuous. Blocks come
out variable-sized, which a columnar writer can emit anyway. The
[measurements](03-results.md) show radius-targeted layout cutting rows scored
from 88.5% to 11.9% — a 7× reduction — against size-only 256-row blocks on the
same corpus, and still beating size-only blocking at a quarter the block size
(38.4%). The benefit plateaus once the radius target stops binding.

**Predicate partitioning.** A designated column can be partitioned first, so
rows with different values never share a block and a filter on it prunes whole
blocks. This is qd-tree-style instance optimisation, and it has the same
liability: specialising the layout for one query shape costs the others. The
benchmarks show partitioning by tenant helping selective tenant filters and
*hurting* broad queries, because it fragments semantic cohesion. Layout is a
workload decision, and Cairn treats it as one — it is a property of a segment,
so compaction can change it as the workload changes.

**The tree.** Leaf summaries are grouped by the same bisecting procedure and
merged with `⊕`, recursively, to a root. Fanout is a tuning knob trading bound
tightness against depth.

### 6.1 Storage tiers and a sizing model

Analytical projection for **10¹⁰ units** (≈1 trillion tokens), `d = 768`,
256 rows/block ⇒ ~3.9×10⁷ blocks, fanout 32 ⇒ 5 tree levels.

| Tier | Contents | Per unit | Total | Residency |
|---|---|---|---|---|
| Cairn tree above leaves | merged summaries, ~1.2×10⁶ nodes | — | ~2.8 GB | RAM |
| Leaf summaries | 1-bit centroid (96 B) + Bloom (256 B) + zone maps + λ maxima | ~2 B | ~20 GB | RAM |
| Row codes | RaBitQ 1-bit + correction scalar | 98 B | 980 GB | NVMe |
| Sparse postings | ~40 terms × 6 B | 240 B | 2.4 TB | NVMe |
| Full-precision vectors | fp32, re-ranking only | 3 KB | 30 TB | object store |
| Text and provenance | source spans | ~400 B | 4 TB | object store |

Per-query I/O at 500 opened blocks: `500 × 256 × 98 B ≈ 12 MB` of codes plus
~200 full-precision fetches for re-ranking (~0.6 MB). At 7 GB/s that is a few
milliseconds of device time — the design's bet is that *bounds are cheap and
RAM-resident, payload is expensive and cold*.

These are projections from the structure, not measurements. What the prototype
measures is the ratio that drives them: how many blocks a query has to open.

---

## 7. Cost model

For `M` blocks, fanout `f`, block size `b`, `V` visited nodes and `O` opened
blocks:

```
work ≈ V·c_bound + O·b·c_score + |candidates|·c_rerank
```

`c_bound` is a dot product against a quantised centroid plus a handful of
Bloom probes — cheap, vectorisable, RAM-resident. `c_score` touches NVMe.
The traversal's job is to keep `O` small at the cost of a slightly larger `V`,
which is exactly the trade radius-targeted layout makes.

`O` is not a tuning parameter: it is `|{blocks : UB(block) > s*_k}|`, a property
of the corpus, the query and the layout. That is the quantity to optimise, and
it is the quantity the benchmarks report.

---

## 8. Freshness: the semantic LSM

- **Memtable.** New rows land in memory and are searched by brute force. Small
  and always exact.
- **Flush.** Sealing runs the layout algorithm and builds the cairn.
- **Deletes.** Tombstones in a per-segment validity bitmap. Sound by Theorem 5,
  O(1), no summary touched.
- **Compaction.** Merges segments, drops tombstoned rows, and *re-lays out* the
  survivors. Churn does not break correctness; it loosens bounds, and compaction
  is what buys tightness back. The benchmarks show exactly this: recall stays at
  1.000 through a 33% delete, and compaction reduces rows scored afterwards.
- **Cross-segment bound sharing.** Segments are visited best-first by their root
  bound, and the running k-th score is threaded through all of them, so a strong
  hit in one segment prunes the others.

**The embedding staleness contract.** Embedding is asynchronous — a row is
lexically searchable at commit and semantically searchable when its vector
lands. Rather than hiding this, Cairn makes it a first-class, queryable state:
a row with a pending embedding is scored on its sparse component only, and the
answer's certificate accounts for the un-embedded population as an explicitly
unbounded set. A system that silently returns "no results" for a document
ingested ten seconds ago is worse than one that says which part of the corpus it
could not bound.

---

## 9. Elastic context assembly

Retrieval returns a *context*, not a ranked list. Given a candidate pool of
scored units and a token budget, Cairn maximises a facility-location objective

```
f(S) = Σ_u rel(u) · max_{s∈S} sim(u, s)
```

subject to `Σ_{s∈S} tokens(s) ≤ budget`. `f` is monotone and submodular, so
cost-benefit greedy (best marginal gain per token) is a constant-factor
approximation of the optimum and, more usefully, it *automatically* suppresses
near-duplicates: a second copy of an already-selected passage has near-zero
marginal gain.

Selected units adjacent in their source document are then merged into contiguous
spans, bridging small gaps when the budget allows, so the caller gets readable
passages with byte-level provenance rather than shuffled fragments.

Measured on a corpus with injected near-duplicates, at the same ~1480-token
budget, assembly returns 21.3 distinct documents versus 14.9 for score-ordered
top-k, with zero near-duplicate pairs per query versus one, at slightly higher
relevance coverage (0.924 vs 0.907). The budget is the same; what changes is
what it buys.

---

## 10. CairnQL

One statement carries everything the planner needs, so it compiles to one
traversal:

```sql
SELECT CONTEXT
  FROM corpus
 WHERE tenant = 7 AND lang IN (0, 1) AND ts BETWEEN 1780000000 AND 1800000000
  NEAR :question
 MATCH :terms
  FUSE dense 0.7, sparse 0.3, recency 0.1 HALFLIFE 30d
BUDGET 2000 TOKENS
GUARANTEE EXACT
```

`NEAR` and `MATCH` take bind parameters because embedding the query is the
caller's job — the embedding model is part of the schema contract, not the
engine. `GUARANTEE` is the knob nothing else exposes: `EXACT`, `EPSILON e`, or
`BLOCKS n`, with the certificate returned either way.

`SELECT TOP k` returns rows instead of an assembled context, for callers that
want to do their own assembly.

---

## 11. Consistency and multi-tenancy

- **MVCC over an append-only commit log.** A query pins a commit LSN; segments
  and tombstone bitmaps are versioned; readers never block writers.
- **Snapshot + as-of.** Because system time is an ordinary column, historical
  queries are ordinary predicates that prune through the same zone maps.
- **Tenancy.** `tenant` is the canonical partition column: partitioning by it
  makes cross-tenant blocks impossible, which is a physical-isolation argument
  rather than a filtering one. ACL predicates are ordinary predicates, evaluated
  *before* scoring, so a tenant's data is never scored for another tenant's
  query.

## 12. Distribution

Shards are independent Cairn instances; the coordinator holds each shard's root
summary, so shard selection is the same bound comparison as node selection — the
cairn extends across machines because `⊕` does. A query broadcasts with a
threshold hint from the best-bounded shard, and each shard prunes against the
global running k-th. Segments are immutable objects in the object store, so
compute is stateless and scales independently; summaries are the only state that
must be resident.

## 13. Calibration

Linear fusion needs comparable component scales. Cairn fits `α, β, γ` per corpus
from a sample of queries (score normalisation plus a small logistic or
least-squares fit against labelled or click data), stored as a named *fusion
profile* referenced by `FUSE`. This is an offline job, not a query-time cost,
and it is the one place where Cairn asks for something RRF does not — in
exchange for prunability.

---

## 14. Limitations and failure modes

Stated plainly, because they decide where this design should and should not be
used.

1. **Concentration of measure.** Cap bounds go vacuous when data is not
   clustered: on uniformly random vectors every block's angular radius
   approaches π/2, every bound approaches 1, and the traversal degenerates to a
   full scan. This is the classical objection to metric trees and it is real.
   The benchmarks reproduce it deliberately. Cairn's bet is that *learned
   embeddings of real corpora are strongly clustered* — the same bet HNSW,
   IVF and Seismic make — with the difference that when the bet fails, Cairn
   loses speed rather than recall.
2. **Fusion slack.** The three component maxima can be attained by three
   different rows, so hybrid bounds are looser than dense-only ones. Measured on
   the same corpus: dense-only queries score 5.9% of rows, hybrid 12-13.5%.
3. **Sparse bounds are the weak link.** Even with the term filter, block-max
   bounds are much looser than cap bounds. Long queries make it worse.
4. **Quantisation and pruning fight each other.** Smaller codes mean a larger
   `ε`, which lowers the safe stop threshold and inflates the candidate set.
   The sound engine surfaces the trade instead of hiding it.
5. **Layout specialisation is a trade, not a win.** Partitioning by a filter
   column helps queries with that filter and hurts the rest.
6. **Exactness is exactness *of the declared scoring function*.** The embedding
   is still lossy, and a certificate says nothing about whether the embedding
   captured meaning. It is a statement about search, not about truth.
7. **No learned component yet.** Layout is a heuristic (bisecting k-means), not
   the RL-optimised routing tree the qd-tree line of work shows is worth ~an
   order of magnitude.
8. **Exact mode gets relatively more expensive as a corpus densifies.** Measured
   at fixed cluster size, the *fraction* of rows scored rises with corpus size
   (2.9% at 25k rows, 5.9% at 100k, 13.9% at 400k). This is intrinsic rather
   than an index defect: exact top-k must score every row that could beat the
   k-th, and denser corpora have more near-ties. Loosening the guarantee is what
   restores the scaling — the same queries under `EPSILON 0.20` score 0.7%,
   0.2% and 0.2% — which is the argument for treating a certified epsilon,
   rather than `EXACT`, as the production default. Note that a *tight* epsilon
   does not help: ties still have to be resolved.
9. **Exactness is not free relative to approximate search.** On the same corpus,
   IVF reaches ~0.96 recall scoring 0.7% of rows; Cairn's exact mode scores
   12.3% to *prove* its answer. Cairn's certified budget mode is the fair
   comparison — competitive recall at matched cost, plus a certificate — and the
   exact mode should be understood as a mode you turn on when correctness
   matters more than latency, not as a faster ANN.
10. **The prototype is single-node, single-threaded, in Python.** Everything in
    §6.1, §11 and §12 is design, not implementation.

---

## 15. What is genuinely new here

Being precise about this matters, because most of the parts are old (§7 of the
[survey](00-research-survey.md) lists the prior art):

1. **A single mergeable summary that bounds a *fused* dense + sparse + scalar +
   temporal score.** Metric trees bound distances; Block-Max WAND bounds term
   impacts; zone maps bound scalars. Cairn puts them in one algebra so one
   traversal serves the whole query, and proves the composition sound.
2. **Sound sparse block bounds via a membership filter.** Static pruning
   (Seismic) achieves speed by discarding what it cannot bound. Charging the
   residual only to terms a Bloom filter admits keeps the bound sound *and*
   tight — the single highest-impact mechanism in the benchmarks (91% → 13.5%
   of rows scored on hybrid queries).
3. **Faceted zone maps for filtered vector search.** Per-facet-value
   sub-summaries give bounds conditioned on the predicate without a per-label
   index copy, and `min(parent, facet)` keeps them sound.
4. **Per-answer certificates.** Turning "how approximate is this?" from a
   corpus-level benchmark statistic into a per-query returned value.
5. **Deriving the LSM write path from the bound algebra** (Theorem 5) rather
   than adopting it by analogy.
6. **Radius-targeted layout**: optimising the physical layout for *bound
   quality* rather than block size, and measuring what it buys.
7. **Context assembly as the query's answer type**, budgeted and submodular,
   inside the engine where the provenance and token counts already live.

---

## 16. Roadmap

| Stage | Work |
|---|---|
| Near | Deterministic `ε` from RaBitQ instead of a calibrated quantile; SIMD/packed-bit bound evaluation; real corpora (MS MARCO, BEIR) with real embeddings |
| Mid | Learned layout (qd-tree-style RL over the observed query distribution); multi-vector/late-interaction units via MUVERA-style fixed dimensional encodings, which fit the algebra unchanged; adaptive λ and Bloom sizing per block |
| Later | Distributed coordinator with shard-level bounds; incremental compaction driven by measured bound degradation; graph-structured expansion (HippoRAG-style) as a second assembly phase over the retrieved pool |

The prototype in [`../poc`](../poc) implements §4, §5, §6, §8, §9 and §10, and
the [measured results](03-results.md) are what it produced.
