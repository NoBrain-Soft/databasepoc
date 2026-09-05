"""Elastic context assembly: return evidence, not fragments.

Classic retrieval hands back the top-k chunks and leaves re-assembly to the
caller, which is where most of the quality loss in retrieval pipelines lives:
the chunk boundary was chosen at ingest time, before anyone knew the query, and
the k results are usually near-duplicates of one another.

Cairn scores at a fine granularity (the *unit* -- a sentence or short passage)
and only decides what a "chunk" is at query time.  Assembly is a budgeted
maximisation of a facility-location objective

    f(S) = sum_u  rel(u) * max_{s in S} sim(u, s)

over the retrieved candidate pool, subject to a token budget.  ``f`` is monotone
and submodular, so cost-benefit greedy (take the unit with the best marginal
gain per token) is a constant-factor approximation of the optimum, and it
naturally suppresses the near-duplicate results that plague top-k retrieval.
Selected units that are adjacent in their source document are then merged into
contiguous spans, optionally bridging small gaps, so the caller receives
readable passages with full provenance instead of shuffled fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Unit:
    row_id: int
    doc_id: int
    ordinal: int
    tokens: int
    score: float
    vec: np.ndarray
    text: str = ""


@dataclass
class Span:
    doc_id: int
    start_ordinal: int
    end_ordinal: int
    tokens: int
    row_ids: List[int]
    score: float
    text: str = ""
    bridged: int = 0            # units pulled in purely for contiguity


@dataclass
class ContextPack:
    spans: List[Span]
    tokens: int
    budget: int
    utility: float
    coverage: float             # fraction of candidate relevance mass represented
    selected: int
    candidates: int

    def render(self) -> str:
        out = []
        for s in self.spans:
            out.append(
                f"[doc {s.doc_id} units {s.start_ordinal}-{s.end_ordinal} "
                f"({s.tokens} tok, score {s.score:.3f})]\n{s.text}"
            )
        return "\n\n".join(out)


def assemble(
    units: Sequence[Unit],
    budget: int,
    bridge_gap: int = 1,
    neighbours: Optional[Dict[Tuple[int, int], Unit]] = None,
    min_marginal: float = 1e-6,
) -> ContextPack:
    """Select and stitch units into a context pack under ``budget`` tokens."""
    if not units:
        return ContextPack([], 0, budget, 0.0, 0.0, 0, 0)

    vecs = np.vstack([u.vec for u in units]).astype(np.float32)
    rel = np.array([max(0.0, u.score) for u in units], dtype=np.float32)
    sim = np.clip(vecs @ vecs.T, 0.0, 1.0)
    total_mass = float((rel * 1.0).sum())

    covered = np.zeros(len(units), dtype=np.float32)   # max sim to chosen set
    chosen: List[int] = []
    spent = 0
    utility = 0.0

    # Lazy greedy with an upper-bound heap would be the production form; the
    # candidate pool here is small enough that a direct pass is clearer.
    while True:
        best_i, best_gain, best_ratio = -1, 0.0, 0.0
        for i, u in enumerate(units):
            if i in chosen or spent + u.tokens > budget:
                continue
            gain = float(np.maximum(covered, sim[i]).dot(rel) - covered.dot(rel))
            if gain <= min_marginal:
                continue
            ratio = gain / max(1, u.tokens)
            if ratio > best_ratio:
                best_i, best_gain, best_ratio = i, gain, ratio
        if best_i < 0:
            break
        chosen.append(best_i)
        covered = np.maximum(covered, sim[best_i])
        spent += units[best_i].tokens
        utility += best_gain

    spans = _stitch(
        [units[i] for i in chosen], budget - spent, bridge_gap, neighbours or {}
    )
    tokens = sum(s.tokens for s in spans)
    coverage = float(covered.dot(rel) / total_mass) if total_mass > 0 else 0.0
    return ContextPack(
        spans=spans,
        tokens=tokens,
        budget=budget,
        utility=utility,
        coverage=coverage,
        selected=len(chosen),
        candidates=len(units),
    )


def _stitch(
    picked: List[Unit],
    slack: int,
    bridge_gap: int,
    neighbours: Dict[Tuple[int, int], Unit],
) -> List[Span]:
    by_doc: Dict[int, List[Unit]] = {}
    for u in picked:
        by_doc.setdefault(u.doc_id, []).append(u)

    spans: List[Span] = []
    for doc, us in by_doc.items():
        us.sort(key=lambda u: u.ordinal)
        run: List[Unit] = [us[0]]
        bridged = 0
        for prev, cur in zip(us, us[1:]):
            gap = cur.ordinal - prev.ordinal - 1
            fill: List[Unit] = []
            if 0 < gap <= bridge_gap:
                fill = [
                    neighbours[(doc, o)]
                    for o in range(prev.ordinal + 1, cur.ordinal)
                    if (doc, o) in neighbours
                ]
                if len(fill) != gap or sum(f.tokens for f in fill) > slack:
                    fill = []
            if gap == 0 or fill:
                slack -= sum(f.tokens for f in fill)
                bridged += len(fill)
                run.extend(fill)
                run.append(cur)
            else:
                spans.append(_span(doc, run, bridged))
                run, bridged = [cur], 0
        spans.append(_span(doc, run, bridged))

    spans.sort(key=lambda s: -s.score)
    return spans


def _span(doc: int, run: List[Unit], bridged: int) -> Span:
    run = sorted(run, key=lambda u: u.ordinal)
    return Span(
        doc_id=doc,
        start_ordinal=run[0].ordinal,
        end_ordinal=run[-1].ordinal,
        tokens=int(sum(u.tokens for u in run)),
        row_ids=[u.row_id for u in run],
        score=float(max(u.score for u in run)),
        text=" ".join(u.text for u in run if u.text),
        bridged=bridged,
    )
