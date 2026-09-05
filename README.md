# Cairn — a database design for context-searchable big data

> A *cairn* is a stack of stones that marks a path. Here it is a stack of
> summaries: each one bounds everything beneath it, so a query walks straight to
> the data that matters — and can prove it skipped nothing.

This repository contains a survey of the current database and retrieval
literature, a system design derived from its gaps, and a working prototype with
a benchmark harness that measures whether the design's claims hold.

| Document | What it is |
|---|---|
| [`docs/00-research-survey.md`](docs/00-research-survey.md) | What the published work says, and the six constraints it implies |
| [`docs/01-design-cairn.md`](docs/01-design-cairn.md) | The system design: data model, bound algebra with proofs, layout, execution, guarantees, sizing, limitations |
| [`docs/03-results.md`](docs/03-results.md) | Measured results from the prototype (generated) |
| [`poc/`](poc/) | The prototype: ~2,000 lines of Python, 21 tests, 9 experiments |

## The problem

A query over a large corpus today is really three queries. A vector index
answers "what is semantically close", an inverted index answers "what matches
these terms", a scan-oriented engine answers "what satisfies these predicates" —
and the results are stitched together afterwards by rank fusion. Nobody can
prune anybody else's work, filters are applied before or after the vector search
(both of which fail, in opposite regimes), and no engine will tell you whether
the answer it just returned is actually the right one.

## The idea

**Give every physical block a mergeable, sound summary that can upper-bound a
fused relevance score under a predicate. Then hybrid search, filtering,
freshness and provable answers all collapse into one branch-and-bound traversal
over one index.**

A summary holds a centroid and angular radius (bounding dense similarity by the
spherical triangle inequality), block-max term impacts plus a Bloom filter over
the terms present (bounding the lexical component *soundly*), min/max zone maps
(three-valued predicate evaluation: `ALL`/`SOME`/`NONE`), and a max timestamp
(bounding a recency prior). Since fusion is linear, the component bounds sum
into a bound on the fused score. Summaries merge, so they stack into a tree, are
rebuilt by compaction, and combine across segments and shards.

Three consequences fall out:

- **Search is exact by default.** The traversal stops when nothing unopened can
  beat the k-th result, and every answer carries a *certificate* — the largest
  bound left unexplored. Approximation becomes an explicit, quantified choice
  (`GUARANTEE EPSILON 0.05`) instead of a globally tuned guess (`efSearch`).
- **The write path must be an LSM**, and this is provable, not stylistic:
  removing rows preserves an upper bound, adding rows does not. So deletes are
  free tombstones and inserts go to a memtable.
- **Filters prune I/O** instead of masking rows, because the predicate is
  evaluated against the same summary as the similarity bound.

The answer unit is an assembled **context**: a budgeted, de-duplicated set of
spans with provenance, selected by submodular optimisation at query time — not
a list of chunks whose boundaries were fixed at ingest.

## Measured results

Full tables in [`docs/03-results.md`](docs/03-results.md). Synthetic corpora,
pure Python, single core — so read the *work ratios*, not the latencies. Recall
is measured against exhaustive brute force.

- **Exact hybrid retrieval scores 12–13% of rows** (dense-only: 5.9%), at
  recall 1.000 by construction.
- **The Bloom term filter is the difference between a usable bound and a useless
  one**: hybrid queries go from 91% of rows scored to 13.5%, same answers.
- **Radius-targeted layout beats size-only blocking 7×** (88.5% → 11.9% of rows
  scored), because a block's dense bound is set by its worst member.
- **At matched cost, the certified budget mode beats IVF**: 0.987 recall
  scoring 0.7% of rows, versus IVF's 0.957 at the same 0.7% — and Cairn reports
  what it could not rule out.
- **Proving exactness is not free**: it costs ~17× the work IVF needs to be
  approximately right, and exact mode's cost *fraction* grows with corpus
  density (2.9% → 5.9% → 13.9% of rows from 25k to 400k rows) because near-ties
  multiply. Under a certified `EPSILON 0.20` the same queries score 0.7%, 0.2%
  and 0.2%. That trade is the design's, and it states it rather than hiding it.
- **The approach degenerates gracefully.** On uniformly random vectors,
  concentration of measure makes every bound vacuous and the engine falls back
  to a full scan — losing speed, never recall. That experiment is in the
  benchmark on purpose.

## Quick start

```bash
cd poc
pip install -r requirements.txt          # numpy only
python3 -m unittest discover -s tests    # soundness + exactness tests
python3 bench/run_bench.py --quick       # ~30s
```

## Status

This is a design study with an executable core, not a database. The bound
algebra, traversal, certificates, layout, LSM write path, context assembly and
query language are implemented and measured. The on-disk format, tiering,
distribution, MVCC and learned layout are designed and specified but not built.
Section 14 of the design document lists the known limitations — including the
ones the benchmarks expose rather than hide.
