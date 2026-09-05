# What the literature actually says about context-searchable big data

A survey assembled to answer one question: *if you were designing a database
today whose primary access path is "find me the context relevant to this",
over a corpus too large to scan, what would the published work tell you to
build?*

Sources fall into six clusters. Each section ends with the constraint that the
design in [`01-design-cairn.md`](01-design-cairn.md) had to satisfy.

---

## 1. Vector search: the index is not the hard part any more

The graph-index line of work (HNSW, Malkov & Yashunin, TPAMI 2020; DiskANN,
Jayaram Subramanya et al., NeurIPS 2019; SPANN, Chen et al., NeurIPS 2021) has
essentially solved unfiltered nearest-neighbour search at billion scale. The
2024 VLDB Journal survey of vector database management systems and the 2026
follow-up analyses treat HNSW/IVF/DiskANN/CAGRA as commodity building blocks
and move the discussion to systems concerns: updates, filters, multi-tenancy,
cost.

What these indexes do *not* provide is any statement about the answer they just
returned. `efSearch` and `nprobe` are global knobs tuned against a benchmark;
they say nothing about whether *this* query's result is right. Recall is
reported as a corpus-level average, which is exactly the wrong statistic when a
single bad retrieval silently corrupts a downstream answer.

Quantisation is where the recent theoretical progress is. **RaBitQ** (Gao &
Long, SIGMOD 2024) quantises D-dimensional vectors to D bits with a *sharp
theoretical error bound*, unlike PQ and its variants, which have no bound and
can fail badly on adversarial data; extended RaBitQ generalises this to B bits
per dimension with an asymptotically optimal space/accuracy trade-off. This
matters far beyond compression: an error bound is the difference between an
estimate and a *bound*, and only bounds compose into guarantees.

> **Constraint 1.** Compression must come with an error term the query planner
> can add to a bound, not just an empirical recall number.

- [RaBitQ (SIGMOD 2024)](https://arxiv.org/abs/2405.12497) ·
  [ACM DL](https://dl.acm.org/doi/10.1145/3654970) ·
  [Extended RaBitQ](https://dev.to/gaoj0017/extended-rabitq-an-optimized-scalar-quantization-method-83m)
- [Survey of vector database systems, 2026](https://effoma.com/blog/vector-database-performance-benchmark-comparison-2026/)
- [Awesome ANN search papers](https://github.com/SimoneZeng/awesome-vector-ANN-search-papers)

---

## 2. Filtered search is the actual open problem

Every production retrieval query has predicates: tenant, language, ACL, time
window, document type. The literature is unambiguous that combining them with
vector search is unsolved in general. The VLDB 2025 survey *Filtered Vector
Search: State-of-the-art and Research Opportunities* frames the whole design
space; Filtered-DiskANN, ACORN, SIEVE (PVLDB 18, 2025), VecFlow (GPU,
arXiv 2506.00812), FAVOR and RACORN-1 (2026) all attack the same failure:

- **Pre-filtering** — resolve the predicate, then scan the survivors. Optimal
  at high selectivity, catastrophic at low.
- **Post-filtering** — run ANN, then discard non-matching results. Recall
  collapses when the predicate is selective, and no amount of over-fetching
  fixes it in the tail.

The 2025-26 work converges on *filter-aware indexes*: per-label graph copies,
selectivity-aware traversal, exclusion distances. These are effective but they
specialise the index to a predicate vocabulary known in advance, and they
multiply storage per label.

> **Constraint 2.** The predicate must participate in the *same* pruning
> decision as the similarity search, at block granularity, without materialising
> a separate index per filter value.

- [Filtered Vector Search: State-of-the-art and Research Opportunities (PVLDB 18)](https://www.vldb.org/pvldb/vol18/p5488-caminal.pdf)
- [VecFlow (arXiv 2506.00812)](https://arxiv.org/pdf/2506.00812)
- [FAVOR (arXiv 2605.07770)](https://arxiv.org/pdf/2605.07770) ·
  [RACORN-1 (arXiv 2607.00768)](https://arxiv.org/pdf/2607.00768)

---

## 3. Hybrid retrieval works, but the plumbing is wrong

Dense and lexical retrieval fail differently, and the empirical result is
consistent across the literature: combining them beats either. Reported gains
are large — hybrid late-interaction plus learned-sparse reaching 80.8%
Recall@10 on MS MARCO against 13.9% (dense) and 11.9% (BM25) in one 2026 study.
ColBERT-style late interaction (Khattab & Zaharia, SIGIR 2020) buys precision at
a heavy storage cost, which **PLAID** (Santhanam et al., SIGIR 2022) reduces by
progressive centroid-based pruning: 6.8× on GPU, 45× on CPU at equal quality.
**MUVERA** (NeurIPS 2024) goes further and reduces multi-vector similarity to
*single-vector* MIPS via fixed dimensional encodings with theoretical
guarantees — 10% better recall at 90% lower latency on BEIR.

On the sparse side, **Seismic** (Bruch et al., SIGIR 2024) reaches
sub-millisecond latency on learned sparse embeddings by organising inverted
lists into geometrically cohesive blocks with **summary vectors**, exploiting
the "concentration of importance" in learned sparse representations. It is one
to two orders of magnitude faster than inverted-index baselines — and it gets
there by *static pruning*, i.e. by giving up on returning what it dropped.

The plumbing problem is fusion. In practice the two retrievers are separate
systems combined by Reciprocal Rank Fusion. RRF depends on a document's *rank*
in each list, which is unknown until each list is fully materialised — so RRF
is structurally unprunable. You cannot bound a rank.

> **Constraint 3.** Fusion must be a calibrated *linear* combination of
> component scores, so that per-component bounds sum into a bound on the fused
> score, and one traversal can serve the whole query.

- [PLAID (arXiv 2205.09707)](https://arxiv.org/pdf/2205.09707) ·
  [reproducibility study](https://arxiv.org/pdf/2404.14989)
- [MUVERA (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/b71cfefae46909178603b5bc6c11d3ae-Paper-Conference.pdf)
- [Seismic (SIGIR 2024)](https://arxiv.org/pdf/2404.18812)
- [Hybrid sparse-dense retrieval study (2026)](https://www.researchgate.net/publication/399428523_Hybrid_Dense-Sparse_Retrieval_for_High-Recall_Information_Retrieval)

---

## 4. Freshness: immutable indexes versus a live corpus

Graph indexes are built, not maintained. **FreshDiskANN** (Singh et al., 2021)
answers this with an LSM-shaped design: inserts go to a small in-memory dynamic
graph plus a fresh SSD region, deletes are tombstones, and a background
*streaming merge* folds them into the long-term index at a cost proportional to
the change set, holding >95% recall. **SPFresh** (SOSP 2023) does in-place
incremental updates at billion scale; *In-Place Updates of a Graph Index for
Streaming ANN* (2025) continues the line. Deletions are processed lazily and
consolidated in batches, because eagerly repairing a graph around a deleted
vertex is prohibitively expensive.

The lesson generalises beyond graphs: **an immutable, summarised, background-merged
structure is the shape that survives churn**, which is exactly the LSM-tree
(O'Neil et al., 1996) rediscovered under a different workload.

> **Constraint 4.** Ingest must not require rebuilding anything, deletes must be
> O(1), and whatever the index promises must remain true between compactions.

- [FreshDiskANN (arXiv 2105.09613)](https://arxiv.org/pdf/2105.09613)
- [In-Place Updates of a Graph Index for Streaming ANN (arXiv 2502.13826)](https://www.alphaxiv.org/abs/2502.13826)
- [Big ANN NeurIPS'23 competition results](https://arxiv.org/pdf/2409.17424)
- [Vector Search for the Future: from memory-resident to cloud-native (arXiv 2601.01937)](https://arxiv.org/pdf/2601.01937)

---

## 5. Storage and layout: block skipping is the oldest trick that still works

Analytical engines have known since Small Materialized Aggregates (Moerkotte,
VLDB 1998) that the cheapest I/O is the block you never read. **Qd-tree**
(Yang et al., SIGMOD 2020) makes this an optimisation problem: partitioning by
arrival time or hash cannot minimise *blocks accessed*, and a query-data routing
tree learned from the workload gives more than an order of magnitude speedup,
within 2× of the data-skipping lower bound. **MTO** (SIGMOD 2021) extends
instance-optimised layouts across joins.

Meanwhile the storage format layer is being rewritten for AI workloads, which
mix full scans with point lookups. **Lance** (arXiv 2504.15247) shows Parquet
and Arrow leave NVMe random-access performance on the table and proposes
adaptive structural encodings; Vortex, Nimble and BtrBlocks are contemporaries.
**ByteHouse** (arXiv 2602.08226) describes a production warehouse serving OLAP
scans and feature-level point lookups over the same immutable columnar data.

> **Constraint 5.** Physical layout is a first-class, workload-tuned decision,
> and the same file must serve scan-shaped and lookup-shaped access.

- [Qd-tree (SIGMOD 2020)](https://arxiv.org/pdf/2004.10898) ·
  [Instance-Optimized Data Layouts (SIGMOD 2021)](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/04/msr-mto-sigmod.pdf)
- [Lance (arXiv 2504.15247)](https://arxiv.org/html/2504.15247v1)
- [ByteHouse (arXiv 2602.08226)](https://arxiv.org/pdf/2602.08226)

---

## 6. What the retrieval consumer actually needs

The consumer of a "context search" is usually a model with a token budget, and
the memory/RAG literature is converging on the finding that top-k chunk
retrieval is the wrong interface. A-MEM (2025), MemTree, D-Mem (2026) and the
2026 *Memory in the Age of AI Agents* survey describe retrieval as an
**evidence-completion** problem rather than a top-1 ranking problem: the
required support is old, scattered, or spread across turns. Graph-structured
retrieval (Microsoft GraphRAG, HippoRAG's Personalized PageRank over an entity
graph, LightRAG's dual-level indexing) exists because the evidence for a
question is rarely one chunk — with HippoRAG reporting up to 20% accuracy gains
on multi-hop QA at 10-20× lower cost than iterative retrieval. The systematic
evaluations are more sober: *RAG vs. GraphRAG* (2025) and GraphRAG-Bench find
the advantage is domain-dependent, and graph construction is expensive.

Two other properties recur in every 2026 memory-system paper: agents need a
**write path**, not just a retriever, and they need retrieval **as of** a point
in time, since memory is continuously revised.

Retrieved chunks also arrive redundant. Selecting a diverse, budget-constrained
subset is a solved problem in the abstract — maximal marginal relevance
(Carbonell & Goldstein, SIGIR 1998) and submodular summarisation (Lin & Bilmes,
2011, building on Nemhauser-Wolsey-Fisher's 1978 greedy bound) — but retrieval
engines return ranked lists and leave it to the caller.

> **Constraint 6.** The answer unit should be a budgeted, de-duplicated,
> provenance-carrying *context*, assembled at query time, not a list of chunks
> whose boundaries were fixed at ingest.

- [Memory in the Age of AI Agents (arXiv 2512.13564)](https://arxiv.org/pdf/2512.13564)
- [A-MEM (arXiv 2502.12110)](https://arxiv.org/pdf/2502.12110) ·
  [Are We Ready For An Agent-Native Memory System? (arXiv 2606.24775)](https://arxiv.org/html/2606.24775v1)
- [Towards Practical GraphRAG (arXiv 2507.03226)](https://arxiv.org/pdf/2507.03226) ·
  [RAG vs. GraphRAG (arXiv 2502.11371)](https://arxiv.org/html/2502.11371v3)
- [State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)

---

## 7. Prior art the design deliberately reuses

Nothing here is invented from nothing, and it is worth being explicit about what
is old:

| Idea | Where it comes from |
|---|---|
| Bounding a set of points by a centroid and a radius, pruning by triangle inequality | Metric/ball trees: R-tree (Guttman, 1984), M-tree (Ciaccia et al., VLDB 1997), cover trees |
| Upper-bounding a scored scan and stopping early | Threshold Algorithm (Fagin et al., 2001), WAND (Broder et al., CIKM 2003), Block-Max WAND (Ding & Suel, SIGIR 2011) |
| Per-block min/max statistics to skip I/O | Small Materialized Aggregates (Moerkotte, VLDB 1998), zone maps |
| Block summaries over cohesive posting blocks | Seismic (SIGIR 2024) |
| Immutable segments, tombstones, background merge | LSM-tree (O'Neil et al., 1996), FreshDiskANN, SPFresh |
| Approximate membership at a few bits per key | Bloom (1970), ribbon/cuckoo filters |
| Quantisation with a provable error bound | RaBitQ (SIGMOD 2024) |
| Budgeted diverse selection | MMR (SIGIR 1998), submodular summarisation (Lin & Bilmes, 2011) |
| Workload-driven physical layout | Qd-tree (SIGMOD 2020), MTO (SIGMOD 2021) |

Metric trees in particular were largely abandoned for high-dimensional search,
for good reason: under concentration of measure, a ball's radius approaches the
distance to everything, and the bound goes vacuous. That objection is real and
[the benchmarks reproduce it](03-results.md) on uniformly random vectors. What
has changed is the data: learned embeddings of real corpora have low intrinsic
dimension and are strongly clustered, which is exactly the regime where cap
bounds are tight — and a layout algorithm can *make* them tight by splitting on
radius rather than on row count.

---

## 8. The gap

Put the six constraints together and no published system satisfies them at once:

| | filters in the same pruning decision | prunable hybrid fusion | per-answer guarantee | freshness without rebuild | context as the answer unit |
|---|---|---|---|---|---|
| HNSW / IVF vector DBs | post-filter or per-label index | separate index + RRF | no | partial | no |
| Filtered-DiskANN / ACORN / SIEVE | yes, for known labels | no | no | partial | no |
| Seismic | no | sparse only | no (static pruning) | no | no |
| Lakehouse + zone maps | yes, scalar only | no vector path | exact by scanning | yes | no |
| Elasticsearch-style hybrid | yes, scalar only | RRF | no | yes | no |
| GraphRAG / agent memory | application level | application level | no | write path, yes | yes, but expensive |

The gap is a single index whose *summary* is rich enough to bound a fused
score under a predicate — and whose engine reports what it can prove about the
answer. That is what [Cairn](01-design-cairn.md) is.
