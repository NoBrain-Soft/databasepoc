"""Cairn -- a context-searchable database prototype.

See ``docs/01-design-cairn.md`` for the system design this code demonstrates.
"""

from .assemble import ContextPack, Span, Unit, assemble
from .engine import CairnDB, Config, QueryResult
from .index import Certificate, Hit, SearchStats, Segment
from .quantize import Quantizer
from .ql import Query, parse
from .segment import RowBatch, plan_blocks, summarize
from .summary import InSet, QueryBounds, Range, Summary, TriState

__all__ = [
    "CairnDB", "Config", "QueryResult", "Segment", "Certificate", "Hit",
    "SearchStats", "RowBatch", "Summary", "QueryBounds", "Range", "InSet",
    "TriState", "Quantizer", "assemble", "ContextPack", "Span", "Unit",
    "plan_blocks", "summarize", "parse", "Query",
]
