from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from ..utils import logistic_growth, _np

__all__ = ["Colony", "process_colony_batch"]


@dataclass
class Colony:
    """Early settlement that can grow into a city."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    population: int = 1000

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def process_turn(self, plague_resist: float = 0.0) -> None:
        rate = 0.05
        cap = 5000
        self.population = int(logistic_growth(self.population, rate, cap))
        planet = PLANETS.get(self.planet)
        if planet and planet.plague_level > 0:
            effect = planet.plague_level * (1 - plague_resist)
            self.population = max(0, int(self.population * (1 - effect)))


from .registry import PLANETS


def process_colony_batch(
    colonies: List[Colony],
    plague_level: float,
    plague_resist: float,
) -> int:
    """Vectorized update for colony growth."""
    if not colonies:
        return 0
    if _np is not None:
        pops = _np.array([c.population for c in colonies], dtype=float)
        pops = logistic_growth(pops, 0.05, 5000)
        if plague_level > 0:
            effect = plague_level * (1 - plague_resist)
            pops = _np.maximum(0, pops * (1 - effect))
        total = int(pops.sum())
        for c, p in zip(colonies, pops.tolist()):
            c.population = int(p)
        return total
    total = 0
    for c in colonies:
        c.process_turn(plague_resist)
        total += c.population
    return total
