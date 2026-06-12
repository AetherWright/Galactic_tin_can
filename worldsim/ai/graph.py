"""Minimal directed graph used in NEAT and other modules."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

class DiGraph:
    def __init__(self) -> None:
        self.succ: Dict[int, Dict[int, Dict]] = {}
        self.pred: Dict[int, Dict[int, Dict]] = {}

    def add_node(self, n: int) -> None:
        self.succ.setdefault(n, {})
        self.pred.setdefault(n, {})

    def add_edge(self, u: int, v: int, **data) -> None:
        self.add_node(u)
        self.add_node(v)
        self.succ[u][v] = data
        self.pred[v][u] = data

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.succ.get(u, {})

    def nodes(self) -> Iterable[int]:
        return list(self.succ.keys())

    def edges(self, data: bool = False):
        for u, nbrs in self.succ.items():
            for v, d in nbrs.items():
                if data:
                    yield (u, v, d)
                else:
                    yield (u, v)

    def edge_data(self, u: int, v: int) -> Dict | None:
        return self.succ.get(u, {}).get(v)

    def copy(self) -> "DiGraph":
        g = DiGraph()
        for n in self.succ:
            g.add_node(n)
        for u, v, d in self.edges(data=True):
            g.add_edge(u, v, **d.copy())
        return g
