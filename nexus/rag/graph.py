"""Code graph: symbol nodes with `calls` and `tests` edges, built from chunk metadata (Neo4j in production).
Graph expansion adds callers, callees and tests of the seed hits so answers can explain who calls what."""
from __future__ import annotations

from collections import defaultdict

from lexora.prep.chunking import Chunk


class CodeGraph:
    def __init__(self, chunks: list[Chunk]):
        self.edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        by_name: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            if c.kind in ("function", "method", "class"):
                by_name[c.symbol.split(".")[-1]].append(c)
        for c in chunks:
            for name in c.calls:
                for tgt in by_name.get(name, []):
                    if tgt.id == c.id:
                        continue
                    kind = "tests" if "/test" in c.path or c.path.startswith("tests") else "calls"
                    self.edges[c.id].append((tgt.id, kind))
                    self.edges[tgt.id].append((c.id, "called_by" if kind == "calls" else "tested_by"))

    def expand(self, seeds: list[str], hops: int = 1, allowed: set[str] | None = None) -> list[tuple[str, str, str]]:
        """Returns [(chunk_id, via_seed, edge_kind)] for nodes reached within `hops`."""
        seen = set(seeds)
        frontier = list(seeds)
        out = []
        for _ in range(hops):
            nxt = []
            for s in frontier:
                for tgt, kind in self.edges.get(s, []):
                    if tgt in seen or (allowed is not None and tgt not in allowed):
                        continue
                    seen.add(tgt)
                    out.append((tgt, s, kind))
                    nxt.append(tgt)
            frontier = nxt
        return out

    def to_dict(self) -> dict:
        return {k: v for k, v in self.edges.items()}
