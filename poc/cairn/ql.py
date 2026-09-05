"""CairnQL: a small surface syntax for context retrieval.

The point of the language is that a *single* statement declares everything the
bound algebra needs -- filters, the semantic target, the lexical target, the
fusion weights, the answer size, and the guarantee the caller wants -- so the
planner can compile it into one traversal instead of three sub-queries and a
fusion step.

    SELECT CONTEXT
      FROM corpus
     WHERE tenant = 7 AND lang IN (0, 1) AND ts BETWEEN 1780000000 AND 1800000000
      NEAR :question
     MATCH :terms
      FUSE dense 0.7, sparse 0.3, recency 0.1 HALFLIFE 30d
    BUDGET 2000 TOKENS
    GUARANTEE EXACT

``NEAR`` and ``MATCH`` take bind parameters, because embedding the query text is
the caller's job (and its model choice is part of the schema, not the engine).
``GUARANTEE`` is the knob the rest of the industry does not expose: EXACT walks
until nothing unopened can beat the k-th result, EPSILON stops once the answer
is provably within a factor, and BLOCKS caps I/O and reports the resulting
certificate rather than pretending the answer is exact.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .summary import InSet, QueryBounds, Range

_SECTIONS = ["SELECT", "FROM", "WHERE", "NEAR", "MATCH", "FUSE", "BUDGET", "GUARANTEE"]
_SECTION_RE = re.compile(r"\b(" + "|".join(_SECTIONS) + r")\b", re.IGNORECASE)


class ParseError(ValueError):
    pass


@dataclass
class Query:
    mode: str = "top"                 # "top" | "context"
    k: int = 10
    table: str = "corpus"
    predicates: List[object] = field(default_factory=list)
    near: Optional[str] = None        # bind name for the dense query vector
    match: Optional[str] = None       # bind name for the sparse query vector
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    half_life_days: float = 30.0
    budget_tokens: int = 2000
    guarantee: str = "exact"          # "exact" | "epsilon" | "blocks"
    epsilon: float = 0.0
    block_budget: Optional[int] = None

    def bind(self, params: Dict[str, Any], now: float = 0.0) -> QueryBounds:
        dense = params.get(self.near) if self.near else None
        if dense is not None:
            dense = np.asarray(dense, dtype=np.float32)
            n = float(np.linalg.norm(dense))
            if n > 0:
                dense = dense / n
        sparse = dict(params.get(self.match) or {}) if self.match else {}
        return QueryBounds(
            dense=dense,
            sparse={int(t): float(w) for t, w in sparse.items()},
            alpha=self.alpha if dense is not None else 0.0,
            beta=self.beta if sparse else 0.0,
            gamma=self.gamma,
            now=now,
            half_life=self.half_life_days * 86400.0,
        )


def parse(sql: str) -> Query:
    text = " ".join(sql.split())
    parts = _split_sections(text)
    q = Query()

    head = parts.get("SELECT")
    if head is None:
        raise ParseError("statement must start with SELECT")
    if head.upper().startswith("CONTEXT"):
        q.mode = "context"
    elif head.upper().startswith("TOP"):
        q.mode = "top"
        m = re.match(r"TOP\s+(\d+)", head, re.IGNORECASE)
        if not m:
            raise ParseError("SELECT TOP needs a row count")
        q.k = int(m.group(1))
    else:
        raise ParseError("SELECT must be followed by CONTEXT or TOP <k>")

    if "FROM" in parts:
        q.table = parts["FROM"].strip()
    if "WHERE" in parts:
        q.predicates = _parse_predicates(parts["WHERE"])
    if "NEAR" in parts:
        q.near = _bind_name(parts["NEAR"])
    if "MATCH" in parts:
        q.match = _bind_name(parts["MATCH"])
    if "FUSE" in parts:
        _parse_fuse(parts["FUSE"], q)
    if "BUDGET" in parts:
        m = re.match(r"(\d+)\s*TOKENS?", parts["BUDGET"].strip(), re.IGNORECASE)
        if not m:
            raise ParseError("BUDGET must look like 'BUDGET 2000 TOKENS'")
        q.budget_tokens = int(m.group(1))
    if "GUARANTEE" in parts:
        g = parts["GUARANTEE"].strip()
        if re.match(r"EXACT$", g, re.IGNORECASE):
            q.guarantee = "exact"
        elif re.match(r"EPSILON", g, re.IGNORECASE):
            q.guarantee, q.epsilon = "epsilon", float(g.split()[1])
        elif re.match(r"BLOCKS", g, re.IGNORECASE):
            q.guarantee, q.block_budget = "blocks", int(g.split()[1])
        else:
            raise ParseError(f"unknown guarantee: {g!r}")
    return q


def _split_sections(text: str) -> Dict[str, str]:
    hits = list(_SECTION_RE.finditer(text))
    out: Dict[str, str] = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        key = m.group(1).upper()
        # BETWEEN ... AND ... may contain no section keywords, but a bare word
        # such as "MATCH" inside a string literal would; the grammar has no
        # string literals, so first-wins is unambiguous here.
        out.setdefault(key, text[m.end():end].strip())
    return out


def _bind_name(s: str) -> str:
    s = s.strip()
    if not s.startswith(":"):
        raise ParseError(f"NEAR/MATCH take a bind parameter, got {s!r}")
    return s[1:]


def _parse_fuse(s: str, q: Query) -> None:
    hl = re.search(r"HALFLIFE\s+([\d.]+)\s*d", s, re.IGNORECASE)
    if hl:
        q.half_life_days = float(hl.group(1))
        s = s[: hl.start()]
    q.alpha = q.beta = q.gamma = 0.0
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.match(r"(dense|sparse|recency)\s+([\d.]+)", token, re.IGNORECASE)
        if not m:
            raise ParseError(f"bad FUSE term: {token!r}")
        name, w = m.group(1).lower(), float(m.group(2))
        setattr(q, {"dense": "alpha", "sparse": "beta", "recency": "gamma"}[name], w)


_PRED_RE = re.compile(
    r"(?P<col>\w+)\s*(?:"
    r"IN\s*\((?P<in>[^)]*)\)"
    r"|BETWEEN\s+(?P<lo>-?[\d.eE+]+)\s+AND\s+(?P<hi>-?[\d.eE+]+)"
    r"|(?P<op><=|>=|=|<|>)\s*(?P<val>-?[\d.eE+]+)"
    r")",
    re.IGNORECASE,
)


def _parse_predicates(s: str) -> List[object]:
    preds: List[object] = []
    consumed = 0
    for m in _PRED_RE.finditer(s):
        consumed += 1
        col = m.group("col")
        if m.group("in") is not None:
            vals = frozenset(int(float(v)) for v in m.group("in").split(",") if v.strip())
            preds.append(InSet(col, vals))
        elif m.group("lo") is not None:
            preds.append(Range(col, float(m.group("lo")), float(m.group("hi"))))
        else:
            op, val = m.group("op"), float(m.group("val"))
            if op == "=":
                preds.append(InSet(col, frozenset({int(val)})))
            elif op == ">":
                preds.append(Range(col, lo=math.nextafter(val, math.inf)))
            elif op == ">=":
                preds.append(Range(col, lo=val))
            elif op == "<":
                preds.append(Range(col, hi=math.nextafter(val, -math.inf)))
            else:
                preds.append(Range(col, hi=val))
    if not consumed:
        raise ParseError(f"could not parse WHERE clause: {s!r}")
    return preds
