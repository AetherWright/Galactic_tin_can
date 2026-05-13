from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random

@dataclass
class Genome:
    """Weight vector with an associated fitness value."""
    weights: List[float]
    fitness: float = 0.0


class RewardGA:
    """Very small genetic algorithm for evolving reward weights."""

    def __init__(self, length: int, population: int = 4) -> None:
        self.length = length
        self.population = [
            Genome([random.uniform(0.5, 1.5) for _ in range(length)])
            for _ in range(population)
        ]
        self.active = 0

    @property
    def weights(self) -> List[float]:
        return self.population[self.active].weights

    def step(self) -> None:
        """Increment fitness for the active genome."""
        self.population[self.active].fitness += 1

    def evolve(self) -> None:
        """Breed new genomes from the fittest two."""
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        parents = self.population[:2]
        for genome in self.population[2:]:
            merged = [(a + b) / 2 for a, b in zip(parents[0].weights, parents[1].weights)]
            genome.weights = [w + random.gauss(0, 0.1) for w in merged]
            genome.fitness = 0.0
        for g in parents:
            g.fitness = 0.0
        self.active = 0

    def mutate(self, sigma: float = 0.1) -> None:
        """Apply random noise to all genomes."""
        for genome in self.population:
            genome.weights = [w + random.gauss(0, sigma) for w in genome.weights]
            genome.fitness = 0.0

    def clone(self) -> "RewardGA":
        """Return a deep copy."""
        # Avoid expensive random initialization by creating an empty instance
        # and then copying over the existing genomes.  ``RewardGA.__init__``
        # populates random weights which are immediately overwritten during a
        # clone; for large simulations this extra work showed up as a major
        # bottleneck.  Initializing with ``population=0`` skips the random
        # generation step and lets us directly assign the copied genomes.
        # Use ``object.__new__`` to bypass ``__init__`` entirely which avoids
        # the overhead of constructing temporary random genomes.
        other = object.__new__(RewardGA)
        other.length = self.length
        other.population = [Genome(list(g.weights), g.fitness) for g in self.population]
        other.active = self.active
        return other
