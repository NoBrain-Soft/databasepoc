"""Property tests for Cairn.

The tests that matter most are the soundness ones: if a summary ever reports an
upper bound below a score actually achievable by a row it covers, the engine
silently returns wrong answers and no amount of benchmarking would reveal it.
Those are checked by brute force against every node of the tree.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cairn import CairnDB, Config, InSet, Range, Segment
from cairn.assemble import Unit, assemble
from cairn.index import Node
from cairn.ql import ParseError, parse
from cairn.summary import TriState
from cairn.synth import brute_force, make_corpus, make_queries


def leaf_blocks(node: Node):
    if node.is_leaf:
        return [node.block]
    out = []
    for c in node.children:
        out.extend(leaf_blocks(c))
    return out


def row_scores(seg, rows, qb):
    """Exact fused score of the given physical rows."""
    b = seg.batch
    s = np.zeros(len(rows), dtype=np.float64)
    if qb.alpha and qb.dense is not None:
        s += qb.alpha * (b.vecs[rows] @ qb.dense)
    if qb.beta and qb.sparse:
        qv = np.zeros(seg.vocab, dtype=np.float32)
        for t, w in qb.sparse.items():
            if 0 <= t < seg.vocab:
                qv[t] = w
        for j, r in enumerate(rows):
            a, e = int(b.sp_indptr[r]), int(b.sp_indptr[r + 1])
            s[j] += qb.beta * float((qv[b.sp_indices[a:e]] * b.sp_data[a:e]).sum())
    if qb.gamma:
        s += qb.gamma * qb.recency_vector(b.ts[rows])
    return s


class BoundSoundness(unittest.TestCase):
    """Every node's bound must dominate every row it covers."""

    def _check(self, regime, alpha, beta, gamma, n=6000, seed=3):
        corpus = make_corpus(n=n, d=48, clusters=25, regime=regime, seed=seed)
        seg = Segment(corpus.batch, block_size=64, max_radius=0.7, fanout=8,
                      facet_columns=("lang",), lam=32)
        queries = make_queries(corpus, 12, alpha=alpha, beta=beta, gamma=gamma, seed=seed + 1)

        stack = [seg.root]
        nodes = []
        while stack:
            nd = stack.pop()
            nodes.append(nd)
            stack.extend(nd.children)

        for qb in queries:
            for nd in nodes:
                rows = np.concatenate([
                    np.arange(int(seg.bounds[b]), int(seg.bounds[b + 1]))
                    for b in leaf_blocks(nd)
                ])
                true_max = float(row_scores(seg, rows, qb).max())
                ub = nd.summary.upper_bound(qb)
                self.assertGreaterEqual(
                    ub + 1e-5, true_max,
                    f"unsound bound ({regime}): ub={ub} < true={true_max}",
                )

    def test_dense_clustered(self):
        self._check("clustered", 1.0, 0.0, 0.0)

    def test_dense_uniform(self):
        self._check("uniform", 1.0, 0.0, 0.0)

    def test_hybrid(self):
        self._check("clustered", 0.7, 0.3, 0.0)

    def test_hybrid_with_recency(self):
        self._check("clustered", 0.6, 0.3, 0.2)

    def test_facet_summaries_are_sound(self):
        corpus = make_corpus(n=4000, d=32, clusters=12, seed=5)
        seg = Segment(corpus.batch, block_size=64, facet_columns=("lang",), lam=32)
        qb = make_queries(corpus, 1, alpha=1.0, beta=0.4, seed=9)[0]
        for b, summ in enumerate(seg.leaf_summaries):
            lo, hi = int(seg.bounds[b]), int(seg.bounds[b + 1])
            for (col, val), sub in summ.facets.items():
                rows = np.arange(lo, hi)[seg.batch.categorical[col][lo:hi] == val]
                if not len(rows):
                    continue
                self.assertGreaterEqual(
                    sub.upper_bound(qb) + 1e-5, float(row_scores(seg, rows, qb).max())
                )
            # The engine takes the min of the block bound and the facet bound,
            # so the bound it actually uses is never worse than the block's.
            preds = [InSet("lang", frozenset({0}))]
            self.assertLessEqual(
                summ.filtered_upper_bound(qb, preds), summ.upper_bound(qb) + 1e-9
            )

    def test_merge_dominates_children(self):
        corpus = make_corpus(n=3000, d=32, clusters=10, seed=13)
        seg = Segment(corpus.batch, block_size=48, lam=32)
        qb = make_queries(corpus, 1, alpha=0.8, beta=0.2, gamma=0.1, seed=2)[0]
        a, b = seg.leaf_summaries[0], seg.leaf_summaries[1]
        m = a.merge(b)
        self.assertGreaterEqual(m.upper_bound(qb) + 1e-6, a.upper_bound(qb))
        self.assertGreaterEqual(m.upper_bound(qb) + 1e-6, b.upper_bound(qb))
        self.assertEqual(m.count, a.count + b.count)


class PredicateTriState(unittest.TestCase):
    def test_all_and_none_are_never_wrong(self):
        corpus = make_corpus(n=5000, d=32, clusters=15, seed=17)
        seg = Segment(corpus.batch, block_size=64, lam=32)
        preds = [InSet("lang", frozenset({0, 1})), Range("quality", lo=0.4, hi=0.9)]
        cols = dict(seg.batch.numeric)
        cols.update(seg.batch.categorical)
        for b, summ in enumerate(seg.leaf_summaries):
            lo, hi = int(seg.bounds[b]), int(seg.bounds[b + 1])
            mask = np.ones(hi - lo, dtype=bool)
            for p in preds:
                mask &= p.mask(cols, slice(lo, hi))
            verdict = summ.evaluate(preds)
            if verdict == TriState.NONE:
                self.assertFalse(mask.any())
            elif verdict == TriState.ALL:
                self.assertTrue(mask.all())


class Exactness(unittest.TestCase):
    def test_matches_brute_force(self):
        corpus = make_corpus(n=8000, d=48, clusters=20, seed=23)
        db = CairnDB(Config(block_size=64, max_radius=0.7, memtable_rows=10 ** 9, lam=64))
        db.insert(corpus.batch)
        db.flush()
        for qb in make_queries(corpus, 15, alpha=0.7, beta=0.3, gamma=0.1, seed=29):
            got = db.search(qb, k=10)
            want = brute_force(corpus.batch, qb, 10)
            self.assertTrue(got.certificate.exact)
            np.testing.assert_allclose(
                [h.score for h in got.hits], [w[1] for w in want], atol=1e-4
            )

    def test_filtered_search_matches_brute_force(self):
        corpus = make_corpus(n=8000, d=48, clusters=20, seed=31)
        db = CairnDB(Config(block_size=64, max_radius=0.7, memtable_rows=10 ** 9,
                            facet_columns=("lang",), lam=64))
        db.insert(corpus.batch)
        db.flush()
        preds = [InSet("lang", frozenset({1})), Range("quality", lo=0.6)]
        for qb in make_queries(corpus, 10, alpha=1.0, seed=37):
            got = db.search(qb, k=10, predicates=preds)
            want = brute_force(corpus.batch, qb, 10, predicates=preds)
            self.assertEqual([h.row_id for h in got.hits], [w[0] for w in want])

    def test_faceted_filters_stay_exact_across_merges(self):
        """Regression: a facet present in one child and absent in another must
        not survive a merge as a partial summary -- it would omit rows and stop
        being an upper bound.  Caught in benchmarking as recall 0.993."""
        corpus = make_corpus(n=20000, d=48, clusters=40, seed=101)
        db = CairnDB(Config(block_size=128, max_radius=0.7, memtable_rows=10 ** 9,
                            partition_by="tenant", facet_columns=("lang", "kind"),
                            lam=64))
        db.insert(corpus.batch)
        db.flush()
        queries = make_queries(corpus, 12, alpha=1.0, seed=103)
        for lang in range(3):
            preds = [InSet("tenant", frozenset({5})), InSet("lang", frozenset({lang}))]
            for qb in queries:
                got = db.search(qb, k=10, predicates=preds)
                want = brute_force(corpus.batch, qb, 10, predicates=preds)
                self.assertEqual(
                    [h.row_id for h in got.hits], [w[0] for w in want],
                    f"facet-filtered search lost rows (lang={lang})",
                )
                self.assertTrue(got.certificate.exact)

    def test_epsilon_mode_respects_its_guarantee(self):
        corpus = make_corpus(n=8000, d=48, clusters=20, seed=41)
        db = CairnDB(Config(block_size=64, max_radius=0.7, memtable_rows=10 ** 9, lam=64))
        db.insert(corpus.batch)
        db.flush()
        for qb in make_queries(corpus, 10, alpha=1.0, seed=43):
            got = db.search(qb, k=10, epsilon=0.1)
            want = brute_force(corpus.batch, qb, 10)
            best = want[0][1]
            # Anything missed is bounded by the certificate, by construction.
            self.assertLessEqual(best, got.certificate.max_remaining_bound + 1e-6
                                 if not got.certificate.exact else best + 1e9)
            self.assertGreaterEqual(got.hits[0].score * (1 + 0.1) + 1e-6, best)

    def test_block_budget_reports_inexactness(self):
        corpus = make_corpus(n=8000, d=48, clusters=20, seed=47)
        db = CairnDB(Config(block_size=64, max_radius=0.7, memtable_rows=10 ** 9))
        db.insert(corpus.batch)
        db.flush()
        qb = make_queries(corpus, 1, alpha=1.0, seed=53)[0]
        got = db.search(qb, k=10, block_budget=2)
        self.assertGreaterEqual(got.certificate.ratio, 1.0)
        self.assertLessEqual(got.stats.blocks_opened, 3)


class Lifecycle(unittest.TestCase):
    def test_memtable_segments_and_deletes(self):
        corpus = make_corpus(n=9000, d=32, clusters=15, seed=59)
        db = CairnDB(Config(block_size=64, max_radius=0.7, memtable_rows=2500,
                            compact_threshold=10, lam=64))
        for i in range(0, 9000, 1500):
            db.insert(corpus.batch.take(np.arange(i, min(i + 1500, 9000))))
        self.assertGreater(len(db.segments), 1)

        victims = set(int(x) for x in np.arange(0, 9000, 7))
        db.delete(sorted(victims))

        live = np.ones(9000, dtype=bool)
        live[sorted(victims)] = False
        for qb in make_queries(corpus, 8, alpha=1.0, beta=0.2, seed=61):
            got = db.search(qb, k=10)
            want = brute_force(corpus.batch, qb, 10, live=live)
            self.assertFalse(set(h.row_id for h in got.hits) & victims)
            np.testing.assert_allclose(
                [h.score for h in got.hits], [w[1] for w in want], atol=1e-4
            )

    def test_compaction_preserves_answers(self):
        corpus = make_corpus(n=6000, d=32, clusters=12, seed=67)
        db = CairnDB(Config(block_size=64, max_radius=0.7, memtable_rows=1500,
                            compact_threshold=99, lam=64))
        for i in range(0, 6000, 1500):
            db.insert(corpus.batch.take(np.arange(i, i + 1500)))
        db.flush()
        qbs = make_queries(corpus, 6, alpha=1.0, seed=71)
        before = [[h.row_id for h in db.search(q, k=10).hits] for q in qbs]
        db.delete([int(x) for x in range(0, 6000, 11)])
        db.compact()
        self.assertEqual(len(db.segments), 1)
        live = np.ones(6000, dtype=bool)
        live[list(range(0, 6000, 11))] = False
        for q, prev in zip(qbs, before):
            want = brute_force(corpus.batch, q, 10, live=live)
            got = [h.row_id for h in db.search(q, k=10).hits]
            self.assertEqual(got, [w[0] for w in want])


class Quantized(unittest.TestCase):
    def test_quantized_scoring_keeps_recall(self):
        corpus = make_corpus(n=8000, d=64, clusters=20, seed=73)
        db = CairnDB(Config(block_size=64, max_radius=0.7, memtable_rows=10 ** 9,
                            quantize=True, lam=64))
        db.insert(corpus.batch)
        db.flush()
        hit = 0
        for qb in make_queries(corpus, 10, alpha=1.0, seed=79):
            got = db.search(qb, k=10)
            want = set(w[0] for w in brute_force(corpus.batch, qb, 10))
            hit += len(set(h.row_id for h in got.hits) & want)
        self.assertGreaterEqual(hit / 100.0, 0.95)


class Assembly(unittest.TestCase):
    def _units(self, n=40, seed=83):
        rng = np.random.default_rng(seed)
        base = rng.normal(size=(4, 16)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        units = []
        for i in range(n):
            v = base[i % 4] + 0.05 * rng.normal(size=16).astype(np.float32)
            v /= np.linalg.norm(v)
            units.append(Unit(row_id=i, doc_id=i // 4, ordinal=i % 4,
                              tokens=int(rng.integers(40, 120)),
                              score=float(rng.uniform(0.2, 1.0)), vec=v,
                              text=f"u{i}"))
        return units

    def test_respects_budget_and_dedupes(self):
        units = self._units()
        pack = assemble(units, budget=400)
        self.assertLessEqual(pack.tokens, 400)
        ids = [r for s in pack.spans for r in s.row_ids]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(pack.coverage, 1.0 + 1e-6)

    def test_bigger_budget_never_hurts_coverage(self):
        units = self._units()
        small = assemble(units, budget=300)
        large = assemble(units, budget=1200)
        self.assertGreaterEqual(large.coverage + 1e-9, small.coverage)

    def test_spans_are_contiguous(self):
        units = self._units()
        pack = assemble(units, budget=2000, bridge_gap=1,
                        neighbours={(u.doc_id, u.ordinal): u for u in units})
        for s in pack.spans:
            self.assertEqual(len(s.row_ids), s.end_ordinal - s.start_ordinal + 1)


class Language(unittest.TestCase):
    def test_parses_full_statement(self):
        q = parse("SELECT CONTEXT FROM c WHERE tenant = 3 AND ts BETWEEN 1 AND 2 "
                  "NEAR :v MATCH :t FUSE dense 0.6, sparse 0.4, recency 0.2 HALFLIFE 7d "
                  "BUDGET 800 TOKENS GUARANTEE EPSILON 0.2")
        self.assertEqual((q.mode, q.alpha, q.beta, q.gamma), ("context", 0.6, 0.4, 0.2))
        self.assertEqual((q.half_life_days, q.budget_tokens, q.epsilon), (7.0, 800, 0.2))
        self.assertEqual(len(q.predicates), 2)

    def test_rejects_nonsense(self):
        with self.assertRaises(ParseError):
            parse("SELECT EVERYTHING FROM c")
        with self.assertRaises(ParseError):
            parse("SELECT TOP 5 FROM c NEAR 'not a bind'")

    def test_binding_normalises_the_query_vector(self):
        q = parse("SELECT TOP 5 FROM c NEAR :v")
        qb = q.bind({"v": np.array([3.0, 4.0], dtype=np.float32)})
        self.assertAlmostEqual(float(np.linalg.norm(qb.dense)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
