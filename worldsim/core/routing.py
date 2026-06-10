"""Lightweight graph utilities with optional Rust acceleration."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, Tuple
import heapq
import os
import shutil

_LIB = None
_DIR = Path(__file__).resolve().parent.parent / "native" / "rust_graph"
_LIB_PATH = _DIR / "target" / "release" / ("librust_graph.so")


def _rust_available() -> bool:
    return shutil.which("cargo") is not None


USE_RUST = True
env_flag = os.getenv("WORLDSIM_USE_RUST", "auto").lower()
if env_flag in {"0", "false", "no"}:
    USE_RUST = False
elif env_flag in {"1", "true", "yes"}:
    USE_RUST = _rust_available()
else:
    USE_RUST = _rust_available()


def _build_lib() -> None:
    if not USE_RUST:
        return
    if _LIB_PATH.exists():
        return
    if not (_DIR / "Cargo.toml").exists():
        return
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=_DIR, check=False)
    except Exception:
        return


def _load_lib() -> None:
    global _LIB
    if _LIB is not None:
        return
    if not USE_RUST:
        return
    _build_lib()
    if _LIB_PATH.exists():
        try:
            _LIB = ctypes.CDLL(str(_LIB_PATH))
            _LIB.rg_create.restype = ctypes.c_void_p
            _LIB.rg_free.argtypes = [ctypes.c_void_p]
            _LIB.rg_add_node.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _LIB.rg_add_edge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
            _LIB.rg_shortest_paths.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
            ]
        except OSError:
            _LIB = None


def set_use_rust(flag: bool) -> None:
    """Enable or disable Rust graph helpers at runtime."""
    global USE_RUST, _LIB
    USE_RUST = flag and _rust_available()
    if USE_RUST:
        _build_lib()
        if _LIB_PATH.exists():
            try:
                _LIB = ctypes.CDLL(str(_LIB_PATH))
            except OSError:
                _LIB = None
        else:
            _LIB = None
    else:
        _LIB = None
    

class _RustGraph:
    def __init__(self) -> None:
        _load_lib()
        if _LIB is None:
            self.ptr = None
        else:
            self.ptr = _LIB.rg_create()

    def add_node(self, nid: int) -> None:
        if self.ptr is not None:
            _LIB.rg_add_node(self.ptr, nid)

    def add_edge(self, a: int, b: int, w: float) -> None:
        if self.ptr is not None:
            _LIB.rg_add_edge(self.ptr, a, b, w)

    def shortest_paths(self, start: int, nodes: Iterable[int]) -> Dict[int, float]:
        if self.ptr is None:
            return {}
        arr_nodes = (ctypes.c_int * len(nodes))(*nodes)
        arr_out = (ctypes.c_double * len(nodes))()
        _LIB.rg_shortest_paths(self.ptr, start, arr_nodes, len(nodes), arr_out)
        return {n: arr_out[i] for i, n in enumerate(nodes)}

    def __del__(self) -> None:
        if self.ptr is not None:
            _LIB.rg_free(self.ptr)
            self.ptr = None


class RouteGraph:
    """Graph tracking routes between structures."""

    def __init__(self) -> None:
        self._rg = _RustGraph()
        self.node_ids: Dict[Tuple[int, int, str], int] = {}
        self.id_nodes: Dict[int, Tuple[int, int, str]] = {}
        self.adj: Dict[int, Dict[int, float]] = defaultdict(dict)
        self._version: int = 0

        # Cache shortest-path computations per start node with an LRU policy
        # to cap memory usage.
        self._shortest_cache = lru_cache(maxsize=256)(self._shortest_paths_from)
        # When True, cache clears and version bumps are deferred until
        # end_batch_update() is called, reducing overhead during bulk adds.
        self._batch_mode: bool = False


    def _ensure_node(self, node: Tuple[int, int, str]) -> int:
        if node not in self.node_ids:
            nid = len(self.node_ids)
            self.node_ids[node] = nid
            self.id_nodes[nid] = node
            self._rg.add_node(nid)
        return self.node_ids[node]

    def begin_batch_update(self) -> None:
        """Suppress per-operation cache clearing during bulk modifications."""
        self._batch_mode = True

    def end_batch_update(self) -> None:
        """Flush accumulated cache invalidation after bulk modifications."""
        self._batch_mode = False
        self._shortest_cache.cache_clear()
        self._version += 1

    def add_node(self, node: Tuple[int, int, str]) -> None:
        self._ensure_node(node)
        if not self._batch_mode:
            self._shortest_cache.cache_clear()
            self._version += 1

    def has_node(self, node: Tuple[int, int, str]) -> bool:
        return node in self.node_ids

    def nodes(self) -> Iterable[Tuple[int, int, str]]:
        return self.node_ids.keys()

    def add_edge(self, a: Tuple[int, int, str], b: Tuple[int, int, str], weight: float) -> None:
        na = self._ensure_node(a)
        nb = self._ensure_node(b)
        self.adj[na][nb] = weight
        self.adj[nb][na] = weight
        self._rg.add_edge(na, nb, weight)
        if not self._batch_mode:
            self._shortest_cache.cache_clear()
            self._version += 1
    def _shortest_paths_from(self, nid: int) -> Dict[int, float]:
        nodes = list(self.node_ids.values())
        if self._rg.ptr is not None:
            return self._rg.shortest_paths(nid, nodes)
        return _dijkstra(self.adj, nid)

    def shortest_paths(
        self, start: Tuple[int, int, str]
    ) -> Dict[Tuple[int, int, str], float]:
        """Return shortest distances from ``start`` to all other nodes."""
        nid = self._ensure_node(start)
        dists = self._shortest_cache(nid)
        return {self.id_nodes[n]: d for n, d in dists.items() if n != nid}

    def shortest_path(
        self, a: Tuple[int, int, str], b: Tuple[int, int, str]
    ) -> float:
        """Return shortest distance between nodes ``a`` and ``b``."""
        na = self._ensure_node(a)
        nb = self._ensure_node(b)
        return self._shortest_cache(na).get(nb, float("inf"))

    def all_pairs_shortest_paths(
        self,
    ) -> Dict[Tuple[int, int, str], Dict[Tuple[int, int, str], float]]:
        """Return all-pairs shortest paths without persistent caching."""
        return {node: self.shortest_paths(node) for node in self.nodes()}
    @property
    def version(self) -> int:
        """Return current modification counter for caching."""
        return self._version


def _dijkstra(adj: Dict[int, Dict[int, float]], start: int) -> Dict[int, float]:
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float('inf')):
            continue
        for nxt, w in adj.get(node, {}).items():
            nd = d + w
            if nd < dist.get(nxt, float('inf')):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return dist

