"""Rotation + sign quantisation with a calibrated error term (RaBitQ-lite).

Row payloads are stored as 1-bit-per-dimension codes; block and node summaries
keep full precision because there are orders of magnitude fewer of them.  That
split matters for soundness: the *pruning* arithmetic is exact, and only the
*scoring* of rows inside an opened block carries quantisation error, which is
handled by (a) inflating the running threshold by ``eps`` so no candidate is
discarded that could win, and (b) re-ranking the shortlist against full-precision
vectors before the answer is returned.

The production form of this is RaBitQ / extended RaBitQ, which comes with a
sharp theoretical error bound.  Here we implement the same construction --
random rotation, sign codes, per-vector norm-correction scalar -- and calibrate
``eps`` empirically as a high quantile of observed error, which is a
*probabilistic* bound, not a deterministic one.  The PoC labels it as such.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Quantizer:
    rotation: np.ndarray        # (d, d) float32, orthogonal
    eps: float                  # calibrated |error| bound on the inner product

    @staticmethod
    def fit(vecs: np.ndarray, seed: int = 0, sample: int = 2048, quantile: float = 0.999):
        d = vecs.shape[1]
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        rotation = q.astype(np.float32)
        qz = Quantizer(rotation, eps=0.0)

        n = min(sample, len(vecs))
        idx = rng.choice(len(vecs), size=n, replace=False)
        sub = vecs[idx]
        codes, scale = qz.encode(sub)
        qidx = rng.choice(len(vecs), size=min(256, len(vecs)), replace=False)
        queries = vecs[qidx]
        approx = qz.score(codes, scale, queries)          # (nq, n)
        exact = queries @ sub.T
        err = np.abs(approx - exact)
        qz.eps = float(np.quantile(err, quantile))
        return qz

    def encode(self, vecs: np.ndarray):
        """Return (codes int8 +/-1, scale float32) for unit-norm ``vecs``."""
        y = vecs @ self.rotation
        codes = np.where(y >= 0, 1, -1).astype(np.int8)
        d = vecs.shape[1]
        # <y, xbar> with xbar = codes / sqrt(d); equals L1(y)/sqrt(d).
        a = (np.abs(y).sum(axis=1) / math.sqrt(d)).astype(np.float32)
        scale = np.where(a > 1e-6, 1.0 / (math.sqrt(d) * a), 0.0).astype(np.float32)
        return codes, scale

    def score(self, codes: np.ndarray, scale: np.ndarray, queries: np.ndarray) -> np.ndarray:
        """Estimate <query, original vector> for every (query, code) pair."""
        qr = (queries @ self.rotation).astype(np.float32)
        raw = qr @ codes.T.astype(np.float32)
        return raw * scale[None, :]

    def nbytes_per_vector(self, d: int) -> int:
        return d // 8 + 2       # packed bits + fp16 correction scalar
