"""Minimal NEAT-inspired network using a lightweight graph implementation.

This module defines a small neural network representation built from a custom
``DiGraph`` class. It supports simple mutation operations so that reinforcement
learners can evolve their own topologies without relying on NetworkX.
Connections may be bidirectional which can introduce feedback loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List
import random
import math
from .graph import DiGraph


@dataclass
class Node:
    """Represents a neuron in the NEAT graph."""

    activation: str = "relu"
    bias: float = 0.0


class NEATNetwork:
    """Evolvable neural network built atop :class:`DiGraph`."""

    def __init__(self, n_inputs: int, n_outputs: int):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.graph = DiGraph()
        self.nodes: Dict[int, Node] = {}
        # create input and output nodes
        for i in range(n_inputs + n_outputs):
            self.nodes[i] = Node()
            self.graph.add_node(i)
        self.inputs = list(range(n_inputs))
        self.outputs = list(range(n_inputs, n_inputs + n_outputs))
        self.next_node = n_inputs + n_outputs
        # fully connect inputs to outputs initially
        for i in self.inputs:
            for j in self.outputs:
                self.graph.add_edge(i, j, weight=random.uniform(-1, 1), enabled=True)

    # activation helpers -------------------------------------------------
    @staticmethod
    def _activate(x: float, fn: str) -> float:
        if fn == "relu":
            return max(0.0, x)
        if fn == "leaky_relu":
            return x if x > 0 else 0.01 * x
        if fn == "tanh":
            return math.tanh(x)
        if fn == "gelu":
            return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
        if fn == "linear":
            return x
        raise ValueError(f"Unknown activation {fn}")

    def forward(self, inputs: List[float], steps: int = 2) -> List[float]:
        """Evaluate the network with optional recurrent iterations."""
        values = {i: inputs[idx] for idx, i in enumerate(self.inputs)}
        for n in self.graph.nodes:
            if n not in values:
                values[n] = self.nodes.get(n, Node()).bias
        for _ in range(steps):
            new_vals = values.copy()
            for n in self.graph.nodes:
                if n in self.inputs:
                    continue
                s = sum(
                    values[p] * data.get("weight", 0.0)
                    for p, data in self.graph.pred[n].items()
                    if data.get("enabled", True)
                ) + self.nodes.get(n, Node()).bias
                new_vals[n] = self._activate(s, self.nodes.get(n, Node()).activation)
            values = new_vals
        return [values[o] for o in self.outputs]

    # mutation operations ------------------------------------------------
    def mutate_weights(self, sigma: float = 0.1) -> None:
        for u, v, data in self.graph.edges(data=True):
            if not data.get("enabled", True):
                continue
            data["weight"] += random.gauss(0.0, sigma)

    def mutate_add_connection(self) -> None:
        a = random.choice(list(self.graph.nodes))
        b = random.choice(list(self.graph.nodes))
        if a == b:
            return
        if self.graph.has_edge(a, b):
            return
        self.graph.add_edge(a, b, weight=random.uniform(-1, 1), enabled=True)

    def mutate_add_node(self) -> None:
        edges = list(self.graph.edges())
        if not edges:
            return
        edge = random.choice(edges)
        data = self.graph.edge_data(*edge)
        if data is None or not data.get("enabled", True):
            return
        data["enabled"] = False
        new_id = self.next_node
        self.next_node += 1
        self.graph.add_node(new_id)
        self.nodes[new_id] = Node()
        u, v = edge
        weight = data.get("weight", 1.0) if data else 1.0
        self.graph.add_edge(u, new_id, weight=1.0, enabled=True)
        self.graph.add_edge(new_id, v, weight=weight, enabled=True)

    def clone(self) -> "NEATNetwork":
        other = NEATNetwork(self.n_inputs, self.n_outputs)
        other.graph = self.graph.copy()
        other.nodes = {n: Node(node.activation, node.bias) for n, node in self.nodes.items()}
        other.next_node = self.next_node
        return other

