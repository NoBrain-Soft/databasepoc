# Cairn prototype

A working, single-node implementation of the design in
[`../docs/01-design-cairn.md`](../docs/01-design-cairn.md), written to answer one
question: **do the bounds actually prune anything on data that looks like a real
corpus, and do the guarantees survive contact with an implementation?**

It is a proof of concept, not a database. It is pure Python/NumPy, single
threaded, in memory, with no durability, no concurrency and no network. What it
does have is the full bound algebra, the traversal, the layout algorithm, the
LSM write path, context assembly, and a test suite that checks soundness by
brute force.

## Quick start

```bash
pip install -r requirements.txt          # numpy only

python3 -m unittest discover -s tests    # 21 tests, ~15s
python3 bench/run_bench.py --quick       # ~30s
python3 bench/run_bench.py --out ../docs/03-results.md   # ~4 min, regenerates the results doc
```

```python
from cairn import CairnDB, Config
from cairn.synth import make_corpus

corpus = make_corpus(n=50_000, d=96, with_text=True)
db = CairnDB(Config(block_size=128, max_radius=0.7, facet_columns=("lang",)))
db.insert(corpus.batch)
db.flush()

result = db.execute(
    """
    SELECT CONTEXT FROM corpus
     WHERE lang = 0 AND quality >= 0.3
      NEAR :question MATCH :terms
      FUSE dense 0.7, sparse 0.3, recency 0.05 HALFLIFE 60d
    BUDGET 1500 TOKENS
    GUARANTEE EXACT
    """,
    {"question": query_vector, "terms": {12: 0.8, 4471: 0.5}},
)

print(result.certificate)              # Certificate(kth_score=..., exact=True)
print(result.stats.rows_scored)        # how much of the corpus it had to touch
print(result.context.render())         # assembled spans with provenance
```

## Module map

| File | What it is |
|---|---|
| `cairn/summary.py` | The bound algebra: `Summary`, the sound dense/sparse/scalar/temporal bounds, the Bloom term filter, mergeability, three-valued predicates |
| `cairn/segment.py` | Row storage, radius-targeted block layout, summary construction |
| `cairn/index.py` | The cairn tree and the branch-and-bound traversal; certificates; quantised scoring with candidate retention |
| `cairn/quantize.py` | Rotation + sign quantisation (RaBitQ-lite) with a calibrated error term |
| `cairn/engine.py` | `CairnDB`: memtable, flush, tombstones, compaction, cross-segment bound sharing, context queries |
| `cairn/assemble.py` | Elastic context assembly: submodular selection under a token budget, span stitching |
| `cairn/ql.py` | CairnQL parser |
| `cairn/synth.py` | Synthetic corpora (clustered and uniform regimes) and the brute-force oracle |
| `bench/run_bench.py` | The nine experiments behind [`../docs/03-results.md`](../docs/03-results.md) |
| `tests/test_cairn.py` | Soundness, exactness, lifecycle, assembly and parser tests |

## What the tests actually check

The load-bearing tests are the soundness ones, because an unsound bound produces
*silently wrong answers* that no benchmark would catch:

- **Every node of the tree** is checked against brute force over the rows it
  covers, for dense, hybrid and hybrid+recency queries, on both clustered and
  uniformly random corpora.
- **Three-valued predicates**: a block that answers `ALL` must have every row
  match; a block that answers `NONE` must have none.
- **Exactness**: results and scores are compared element-wise with exhaustive
  scan, including under filters, after deletes, and after compaction.
- **The `EPSILON` guarantee** is checked against the true best score.
- **Regression**: `test_faceted_filters_stay_exact_across_merges` reproduces a
  real bug found during benchmarking, where a facet sub-summary present in one
  child and missing in another survived a merge as a partial summary and quietly
  stopped bounding all its rows. It showed up as recall 0.993 in a mode that
  claims exactness. That is precisely the failure the soundness tests exist to
  catch, and it is why the fix tracks facet *completeness* explicitly.

## Honest scope

Implemented and measured: the bound algebra, the traversal and its certificates,
radius-targeted and partitioned layout, faceted zone maps, the term filter,
quantisation with sound candidate retention, the LSM write path with tombstones
and compaction, context assembly, CairnQL.

Designed but **not** implemented: the on-disk format and tiering, distribution
and shard-level bounds, MVCC and bitemporal queries, learned layout, SIMD bound
evaluation, real embeddings. Section 14 of the design document lists the known
limitations; the benchmarks deliberately include the case where the whole
approach degenerates.

Absolute latencies from this prototype are meaningless — it is interpreted
Python. The transferable numbers are the work ratios: rows scored, blocks
opened, and the recall achieved against a brute-force oracle.
