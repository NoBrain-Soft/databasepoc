"""Cairn benchmark harness.

Everything here is measured on synthetic corpora on a single CPU core in pure
Python/NumPy, so absolute latencies mean nothing.  The quantities that do carry
over to a real implementation are the *work ratios*: what fraction of rows the
engine had to score, how many blocks it opened, and what recall it achieved --
because those are properties of the bounds and the layout, not of the language.

Run:  python3 bench/run_bench.py [--quick] [--out ../docs/03-results.md]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cairn import CairnDB, Config, InSet, Range
from cairn.assemble import Unit, assemble
from cairn.index import Segment
from cairn.segment import plan_blocks, _normalize
from cairn.summary import QueryBounds
from cairn.synth import brute_force, make_corpus, make_queries


class Table:
    def __init__(self, title: str, columns: Sequence[str], note: str = ""):
        self.title, self.columns, self.note, self.rows = title, list(columns), note, []

    def add(self, *values):
        self.rows.append([str(v) for v in values])

    def render(self) -> str:
        out = [f"### {self.title}", ""]
        if self.note:
            out += [self.note, ""]
        out.append("| " + " | ".join(self.columns) + " |")
        out.append("|" + "|".join("---" for _ in self.columns) + "|")
        for r in self.rows:
            out.append("| " + " | ".join(r) + " |")
        out.append("")
        return "\n".join(out)


def build(corpus, **cfg) -> CairnDB:
    db = CairnDB(Config(memtable_rows=10 ** 9, **cfg))
    db.insert(corpus.batch)
    db.flush()
    return db


def evaluate(db, corpus, queries, k=10, predicates=(), epsilon=0.0, block_budget=None):
    """Run queries and compare against exact brute force over the same rows."""
    recalls, rows, blocks, nodes, ms, ratios, exacts = [], [], [], [], [], [], []
    n = len(corpus.batch)
    for qb in queries:
        r = db.search(qb, k=k, predicates=predicates, epsilon=epsilon,
                      block_budget=block_budget)
        want = set(x[0] for x in brute_force(corpus.batch, qb, k, predicates=predicates))
        got = set(h.row_id for h in r.hits)
        recalls.append(len(got & want) / max(1, len(want)))
        rows.append(r.stats.rows_scored / n)
        blocks.append(r.stats.blocks_opened)
        nodes.append(r.stats.nodes_visited)
        ms.append(r.elapsed_ms)
        ratios.append(r.certificate.ratio)
        exacts.append(1.0 if r.certificate.exact else 0.0)
    return {
        "recall": statistics.mean(recalls),
        "rows": statistics.mean(rows),
        "blocks": statistics.mean(blocks),
        "nodes": statistics.mean(nodes),
        "ms": statistics.mean(ms),
        "ratio": statistics.mean(ratios),
        "exact": statistics.mean(exacts),
    }


def pct(x):
    return f"{100 * x:.1f}%"


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def exp_cohesion(n, nq, d) -> Table:
    t = Table(
        "1. Pruning power vs corpus cohesion (exact top-10, dense only)",
        ["intra-cluster cos", "blocks", "rows scored", "blocks opened", "recall@10", "exact"],
        "How tightly the corpus clusters is *the* variable that decides whether "
        "bound-based pruning works.  `uniform` is vectors drawn uniformly on the "
        "sphere: concentration of measure makes every cap bound vacuous and the "
        "engine correctly degrades to a full scan rather than losing recall.",
    )
    for label, kwargs in [
        ("0.95", dict(intra_cos=0.95)),
        ("0.90", dict(intra_cos=0.90)),
        ("0.85", dict(intra_cos=0.85)),
        ("0.75", dict(intra_cos=0.75)),
        ("uniform", dict(regime="uniform")),
    ]:
        c = make_corpus(n=n, d=d, clusters=max(20, n // 500), **kwargs)
        db = build(c, block_size=128, max_radius=0.7)
        qs = make_queries(c, nq, alpha=1.0, query_cos=0.9)
        r = evaluate(db, c, qs)
        t.add(label, db.stats()["blocks"], pct(r["rows"]), f"{r['blocks']:.0f}",
              f"{r['recall']:.3f}", f"{r['exact']:.0%}")
    return t


def exp_layout(n, nq, d) -> Table:
    t = Table(
        "2. Layout ablation (exact top-10, dense only, intra-cluster cos 0.85)",
        ["layout", "blocks", "avg block", "rows scored", "blocks opened", "nodes visited"],
        "The dense bound is set by a block's *worst* member, so one outlier makes "
        "a whole block's bound vacuous.  Splitting on a radius target rather than "
        "a row count is worth far more than shrinking blocks uniformly: it beats "
        "size-only blocking even where size-only uses blocks a quarter the size.  "
        "The cost is more blocks to bound, which the summary tree keeps "
        "sub-linear, and the benefit plateaus once the radius target stops "
        "binding.",
    )
    c = make_corpus(n=n, d=d, clusters=max(20, n // 500), intra_cos=0.85)
    qs = make_queries(c, nq, alpha=1.0, query_cos=0.9)
    for label, cfg in [
        ("size only (256)", dict(block_size=256, max_radius=None)),
        ("size only (64)", dict(block_size=64, max_radius=None)),
        ("radius 0.9", dict(block_size=256, max_radius=0.9)),
        ("radius 0.7", dict(block_size=256, max_radius=0.7)),
        ("radius 0.5", dict(block_size=256, max_radius=0.5)),
    ]:
        db = build(c, **cfg)
        st = db.stats()
        r = evaluate(db, c, qs)
        t.add(label, st["blocks"], f"{st['rows'] / st['blocks']:.0f}", pct(r["rows"]),
              f"{r['blocks']:.0f}", f"{r['nodes']:.0f}")
    return t


def exp_sparse(n, nq, d) -> Table:
    t = Table(
        "3. Sound sparse bounds: the term filter is what makes hybrid queries prunable",
        ["query", "term filter", "rows scored", "recall@10", "exact"],
        "The block-max impact bound must charge a residual for every query term "
        "it did not retain, which swamps the bound on long queries.  A 256-byte "
        "Bloom filter per summary says which terms could be present at all; false "
        "positives only over-charge, so the bound stays soundly an upper bound.",
    )
    c = make_corpus(n=n, d=d, clusters=max(20, n // 500), intra_cos=0.90)
    for beta, label in [(0.0, "dense only"), (0.4, "hybrid 1.0/0.4"), (1.0, "hybrid 1.0/1.0")]:
        for tf in ([True] if beta == 0 else [False, True]):
            db = build(c, block_size=128, max_radius=0.7, lam=96, term_filter=tf)
            qs = make_queries(c, nq, alpha=1.0, beta=beta, query_cos=0.9)
            r = evaluate(db, c, qs)
            t.add(label, "on" if tf else "off", pct(r["rows"]),
                  f"{r['recall']:.3f}", f"{r['exact']:.0%}")
    return t


def exp_filters(n, nq, d) -> Table:
    t = Table(
        "4. Filtered search: predicate-aware layout and faceted zone maps",
        ["selectivity", "layout", "rows scored", "blocks opened", "recall@10"],
        "`plain` uses one layout for everything; `partitioned` keeps the filter "
        "column out of shared blocks; `+facets` additionally stores per-value "
        "sub-summaries so the bound describes only the rows that can survive the "
        "filter.  All three return the exact filtered top-10 -- the difference is "
        "purely how much data has to be touched to prove it.",
    )
    c = make_corpus(n=n, d=d, clusters=max(20, n // 500), intra_cos=0.90)
    qs = make_queries(c, nq, alpha=1.0, query_cos=0.9)
    variants = [
        ("plain", dict(block_size=128, max_radius=0.7)),
        ("partitioned", dict(block_size=128, max_radius=0.7, partition_by="tenant")),
        ("+facets", dict(block_size=128, max_radius=0.7, partition_by="tenant",
                         facet_columns=("lang",))),
    ]
    built = {label: build(c, **cfg) for label, cfg in variants}
    for sel_label, preds in [
        ("~3% (1 tenant)", [InSet("tenant", frozenset({5}))]),
        ("~1% (tenant+lang)", [InSet("tenant", frozenset({5})), InSet("lang", frozenset({1}))]),
        ("~25% (8 tenants)", [InSet("tenant", frozenset(range(8)))]),
        ("~50% (quality)", [Range("quality", lo=0.72)]),
    ]:
        for label, db in built.items():
            r = evaluate(db, c, qs, predicates=preds)
            t.add(sel_label, label, pct(r["rows"]), f"{r['blocks']:.0f}", f"{r['recall']:.3f}")
    return t


def exp_guarantees(n, nq, d) -> Table:
    t = Table(
        "5. Guarantee modes and the certificate",
        ["mode", "rows scored", "recall@10", "answers proved exact", "avg certificate ratio"],
        "Every answer carries the largest bound left unexplored, so an approximate "
        "answer states how much a missed row could have beaten it.  A ratio of "
        "1.00 means nothing unopened could have won -- the answer is exact even "
        "though the engine was allowed to stop early.",
    )
    c = make_corpus(n=n, d=d, clusters=max(20, n // 500), intra_cos=0.85)
    db = build(c, block_size=128, max_radius=0.7)
    qs = make_queries(c, nq, alpha=1.0, query_cos=0.9)
    for label, kw in [
        ("EXACT", dict()),
        ("EPSILON 0.02", dict(epsilon=0.02)),
        ("EPSILON 0.05", dict(epsilon=0.05)),
        ("EPSILON 0.20", dict(epsilon=0.20)),
        ("BLOCKS 32", dict(block_budget=32)),
        ("BLOCKS 8", dict(block_budget=8)),
    ]:
        r = evaluate(db, c, qs, **kw)
        t.add(label, pct(r["rows"]), f"{r['recall']:.3f}", f"{r['exact']:.0%}",
              f"{r['ratio']:.3f}")
    return t


def exp_baselines(n, nq, d) -> Table:
    t = Table(
        "6. Against the usual suspects (dense top-10, intra-cluster cos 0.85)",
        ["engine", "rows scored", "recall@10", "guarantee"],
        "IVF is the honest baseline for a bound-free system: the same clustering, "
        "without the bounds.  Read this table as two comparisons.  *Exactness "
        "costs*: proving the answer takes several times the work IVF needs to be "
        "approximately right, and that is the price of the proof.  *At matched "
        "cost*, Cairn's budget mode is competitive with IVF on recall and still "
        "reports what it could not rule out -- because it spends its budget where "
        "this query's bounds say it matters, instead of on a globally tuned "
        "nprobe.",
    )
    c = make_corpus(n=n, d=d, clusters=max(20, n // 500), intra_cos=0.85)
    qs = make_queries(c, nq, alpha=1.0, query_cos=0.9)
    db = build(c, block_size=128, max_radius=0.7)

    r = evaluate(db, c, qs)
    t.add("Cairn (EXACT)", pct(r["rows"]), f"{r['recall']:.3f}", "exact, certified")
    r = evaluate(db, c, qs, epsilon=0.05)
    t.add("Cairn (EPSILON 0.05)", pct(r["rows"]), f"{r['recall']:.3f}",
          f"within 5%, certified (avg ratio {r['ratio']:.3f})")
    r = evaluate(db, c, qs, epsilon=0.20)
    t.add("Cairn (EPSILON 0.20)", pct(r["rows"]), f"{r['recall']:.3f}",
          f"within 20%, certified (avg ratio {r['ratio']:.3f})")
    for budget in (32, 8):
        r = evaluate(db, c, qs, block_budget=budget)
        t.add(f"Cairn (BLOCKS {budget})", pct(r["rows"]), f"{r['recall']:.3f}",
              f"certified ratio {r['ratio']:.3f}")

    lists = plan_blocks(c.batch.vecs, max(64, n // 256))
    cents = np.vstack([_normalize(c.batch.vecs[g].mean(axis=0)) for g in lists])
    for nprobe in (1, 4, 16, 64):
        recalls, scanned = [], []
        for qb in qs:
            order = np.argsort(-(cents @ qb.dense))[:nprobe]
            rows = np.concatenate([lists[i] for i in order])
            sc = c.batch.vecs[rows] @ qb.dense
            top = rows[np.argsort(-sc)[:10]]
            want = set(x[0] for x in brute_force(c.batch, qb, 10))
            recalls.append(len(set(int(c.batch.row_id[i]) for i in top) & want) / 10)
            scanned.append(len(rows) / n)
        t.add(f"IVF-Flat (nprobe={nprobe})", pct(statistics.mean(scanned)),
              f"{statistics.mean(recalls):.3f}", "none")

    t.add("full scan", "100.0%", "1.000", "exact, trivially")
    return t


def exp_scale(nq, d, sizes) -> Table:
    t = Table(
        "7. Scaling (exact top-10, dense only, intra-cluster cos 0.90)",
        ["rows", "blocks", "rows scored (EXACT)", "rows scored (EPSILON 0.20)",
         "blocks opened", "nodes visited", "summary bytes/row", "ms/query"],
        "The honest result, and not the one hoped for: the *fraction* of rows "
        "scored **rises** with corpus size in exact mode.  This is intrinsic "
        "rather than an index defect -- exact top-k must score every row that "
        "could beat the k-th, and as a corpus densifies around a query the "
        "number of near-ties grows.  Loosening the guarantee is what buys back "
        "the scaling: a tight epsilon barely moves the number (ties still have "
        "to be resolved), which is why the epsilon column here is the loose 0.20 "
        "setting.  Absolute work still grows far more slowly than a scan, and "
        "latency is pure-Python, shown only for its shape.",
    )
    for n in sizes:
        c = make_corpus(n=n, d=d, clusters=max(20, n // 500), intra_cos=0.90)
        db = build(c, block_size=128, max_radius=0.7)
        qs = make_queries(c, nq, alpha=1.0, query_cos=0.9)
        r = evaluate(db, c, qs)
        re_ = evaluate(db, c, qs, epsilon=0.20)
        st = db.stats()
        t.add(f"{n:,}", f"{st['blocks']:.0f}", pct(r["rows"]), pct(re_["rows"]),
              f"{r['blocks']:.0f}", f"{r['nodes']:.0f}",
              f"{st['summary_bytes_per_row']:.1f}", f"{r['ms']:.1f}")
    return t


def exp_assembly(n, nq, d) -> Table:
    t = Table(
        "8. Elastic context assembly vs plain top-k, at equal token budget",
        ["strategy", "tokens used", "distinct docs", "near-dup pairs", "coverage"],
        "The corpus here contains injected near-duplicates, as real ones do.  "
        "`near-dup pairs` counts selected pairs with cosine > 0.98 -- budget spent "
        "saying the same thing twice.  `coverage` is the fraction of the candidate "
        "pool's relevance mass the selection represents.  Assembly is a budgeted "
        "submodular selection, so it drops the duplicates without dropping the top "
        "result.",
    )
    c = make_corpus(n=n, d=d, clusters=max(20, n // 500), intra_cos=0.90, with_text=True)
    # Real corpora are full of near-duplicates -- boilerplate, re-published
    # articles, the same passage quoted in ten places.  Inject them, because
    # de-duplication is exactly what assembly is supposed to buy.
    rng = np.random.default_rng(5)
    dup_src = rng.choice(n, size=n // 3, replace=False)
    b = c.batch
    noise = 0.02 * rng.normal(size=(len(dup_src), b.vecs.shape[1])).astype(np.float32)
    b.vecs[-len(dup_src):] = _normalize(b.vecs[dup_src] + noise)
    db = build(c, block_size=128, max_radius=0.7)
    qs = make_queries(c, nq, alpha=1.0, query_cos=0.9)
    budget = 1500

    def near_dup_pairs(vecs, thresh=0.98):
        if len(vecs) < 2:
            return 0.0
        s = np.clip(vecs @ vecs.T, -1, 1)
        iu = np.triu_indices(len(vecs), 1)
        return float((s[iu] > thresh).sum())

    topk_tokens, topk_docs, topk_red, topk_cov = [], [], [], []
    asm_tokens, asm_docs, asm_red, asm_cov = [], [], [], []
    for qb in qs:
        res = db.search(qb, k=64)
        units = [db._unit(h) for h in res.hits]
        # plain top-k: take best-scoring units until the budget is spent
        picked, spent = [], 0
        for u in sorted(units, key=lambda u: -u.score):
            if spent + u.tokens > budget:
                continue
            picked.append(u)
            spent += u.tokens
        vecs = np.vstack([u.vec for u in picked])
        rel = np.array([max(0.0, u.score) for u in units])
        cov = np.clip(np.vstack([u.vec for u in units]) @ vecs.T, 0, 1).max(axis=1)
        topk_tokens.append(spent)
        topk_docs.append(len(set(u.doc_id for u in picked)))
        topk_red.append(near_dup_pairs(vecs))
        topk_cov.append(float(cov.dot(rel) / rel.sum()))

        pack = assemble(units, budget=budget)
        ids = {r for s in pack.spans for r in s.row_ids}
        sel = [u for u in units if u.row_id in ids]
        asm_tokens.append(pack.tokens)
        asm_docs.append(len(pack.spans))
        asm_red.append(near_dup_pairs(np.vstack([u.vec for u in sel])) if sel else 0.0)
        asm_cov.append(pack.coverage)

    t.add("plain top-k (fill budget)", f"{statistics.mean(topk_tokens):.0f}",
          f"{statistics.mean(topk_docs):.1f}", f"{statistics.mean(topk_red):.1f}",
          f"{statistics.mean(topk_cov):.3f}")
    t.add("elastic assembly", f"{statistics.mean(asm_tokens):.0f}",
          f"{statistics.mean(asm_docs):.1f}", f"{statistics.mean(asm_red):.1f}",
          f"{statistics.mean(asm_cov):.3f}")
    return t


def exp_freshness(n, nq, d) -> Table:
    t = Table(
        "9. Semantic LSM: churn, tombstones and compaction",
        ["state", "segments", "live rows", "rows scored", "recall@10"],
        "Deletes are tombstones and never invalidate a bound (removing rows can "
        "only lower a maximum).  What churn *does* cost is bound quality, and "
        "compaction is what buys it back by re-laying out the survivors.",
    )
    c = make_corpus(n=n, d=d, clusters=max(20, n // 500), intra_cos=0.90)
    db = CairnDB(Config(block_size=128, max_radius=0.7, memtable_rows=n // 6,
                        compact_threshold=99))
    step = n // 6
    for i in range(0, n, step):
        db.insert(c.batch.take(np.arange(i, min(i + step, n))))
    db.flush()
    qs = make_queries(c, nq, alpha=1.0, query_cos=0.9)

    live = np.ones(n, dtype=bool)

    def measure(label):
        recalls, rows = [], []
        for qb in qs:
            r = db.search(qb, k=10)
            want = set(x[0] for x in brute_force(c.batch, qb, 10, live=live))
            recalls.append(len(set(h.row_id for h in r.hits) & want) / 10)
            rows.append(r.stats.rows_scored / n)
        t.add(label, len(db.segments), f"{db.stats()['rows']:,}",
              pct(statistics.mean(rows)), f"{statistics.mean(recalls):.3f}")

    measure("6 segments, no churn")
    victims = list(range(0, n, 3))
    db.delete(victims)
    live[victims] = False
    measure("after deleting 33%")
    db.compact()
    measure("after compaction")
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n = 20_000 if args.quick else 100_000
    nq = 10 if args.quick else 30
    d = 96
    sizes = [10_000, 25_000, 50_000] if args.quick else [25_000, 100_000, 400_000]

    t0 = time.time()
    tables: List[Table] = [
        exp_cohesion(n, nq, d),
        exp_layout(n, nq, d),
        exp_sparse(n, nq, d),
        exp_filters(n, nq, d),
        exp_guarantees(n, nq, d),
        exp_baselines(n, nq, d),
        exp_scale(nq, d, sizes),
        exp_assembly(n // 2, max(5, nq // 3), d),
        exp_freshness(n // 2, nq, d),
    ]
    elapsed = time.time() - t0

    header = [
        "# Cairn PoC: measured results",
        "",
        "_Generated by `poc/bench/run_bench.py`; regenerate with_",
        "`python3 bench/run_bench.py --out ../docs/03-results.md`.",
        "",
        f"Corpus: synthetic, {n:,} rows, {d} dimensions, {nq} queries per cell, "
        "single core, pure Python/NumPy.",
        "Absolute latencies are meaningless at this scale -- read the work ratios "
        "(rows scored, blocks opened) and the recall.",
        "Recall is measured against exhaustive brute force over the same rows and "
        "the same scoring function.",
        "",
    ]
    body = "\n".join(header + [t.render() for t in tables])
    body += f"\n_Benchmark wall time: {elapsed:.0f}s._\n"

    print(body)
    if args.out:
        path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(body)
        print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
