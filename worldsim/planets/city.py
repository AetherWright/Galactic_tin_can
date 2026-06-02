from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import random

from ..utils import logistic_growth, _np
from ..ideas import Idea
from .registry import PLANETS

__all__ = ['City', 'process_city_batch']


@dataclass
class City:
    x: int
    y: int
    planet: str
    infrastructure: int = 1
    owner: Optional[int] = None
    population: int = 5000
    industry: float = 1.0
    commerce: float = 1.0
    services: float = 1.0
    ideas: Dict[str, Idea] = field(default_factory=dict)

    def dominant_idea(self) -> str | None:
        """Return the name of the strongest idea in this city."""
        if not self.ideas:
            return None
        return max(self.ideas.values(), key=lambda i: i.followers).name

    def process_turn(self, plague_resist: float = 0.0) -> None:
        """Advance one turn adjusting for plague resistance."""
        rate = 0.03
        cap = self.infrastructure * 20000
        self.population = int(logistic_growth(self.population, rate, cap))
        planet = PLANETS.get(self.planet)
        if planet and planet.plague_level > 0:
            effect = planet.plague_level * (1 - plague_resist)
            self.population = max(0, int(self.population * (1 - effect)))
        if planet and planet.radiation_level > 0:
            rad = planet.radiation_level
            self.population = max(0, int(self.population * (1 - rad)))
            self.infrastructure = max(1, int(self.infrastructure * (1 - rad * 0.1)))

    @property
    def economic_output(self) -> float:
        planet = PLANETS.get(self.planet)
        richness = planet.resource_richness if planet else 1.0
        sector_bonus = (self.industry + self.commerce + self.services) / 3
        return self.population * self.infrastructure / 1000 * richness * sector_bonus

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    @property
    def world_coords(self) -> Tuple[float, float, float]:
        """Return absolute coordinates for this city including its planet."""
        planet = PLANETS.get(self.planet)
        if planet:
            return planet.x + self.x, planet.y + self.y, planet.z
        return float(self.x), float(self.y), 0.0

    def upgrade(
        self,
        level: int = 1,
        *,
        industry: bool = False,
        commerce: bool = False,
        services: bool = False,
    ) -> None:
        """Improve city attributes by ``level``."""
        if industry:
            self.industry += level * 0.1
        elif commerce:
            self.commerce += level * 0.1
        elif services:
            self.services += level * 0.1
        else:
            self.infrastructure += level


def process_city_batch(
    cities: List[City],
    plague_level: float,
    radiation_level: float,
    plague_resist: float,
    tech_bonus: float,
    growth_mult: float = 1.0,
    cap_mult: float = 1.0,
) -> Tuple[int, float]:
    """Vectorized update for a group of cities.

    The update now models simple migration when population exceeds
    housing capacity (``infrastructure * 20000``).  A small fraction
    of the excess population leaves the city each turn.

    ``growth_mult`` scales the logistic growth rate and ``cap_mult`` scales the
    housing carrying capacity — both default to ``1.0`` and are driven by
    biology tech signals (``pop_growth`` / ``pop_cap``) when supplied.
    """
    if not cities:
        return 0, 0.0

    if _np is not None:
        pops = _np.array([c.population for c in cities], dtype=float)
        infra = _np.array([c.infrastructure for c in cities], dtype=float)
        pops = logistic_growth(pops, 0.03 * growth_mult, infra * 20000 * cap_mult)
        if plague_level > 0:
            effect = plague_level * (1 - plague_resist)
            pops = _np.maximum(0, pops * (1 - effect))
        if radiation_level > 0:
            pops = _np.maximum(0, pops * (1 - radiation_level))
            infra = _np.maximum(1, infra * (1 - radiation_level * 0.1))
        capacity = infra * 20000 * cap_mult
        deficit = pops - capacity
        pops = pops - _np.maximum(deficit, 0) * 0.02
        rnd_up = _np.random.random(len(cities)) < 0.1
        infra = infra + rnd_up
        rnd_down = (_np.random.random(len(cities)) < 0.02) & (infra > 1)
        infra = infra - rnd_down
        total_pop = int(pops.sum())
        for c, p, inf in zip(cities, pops.tolist(), infra.tolist()):
            c.population = int(p)
            c.infrastructure = int(inf)
        # Build a per-planet richness map to avoid one PLANETS.get() per city.
        richness_map: Dict[str, float] = {}
        for c in cities:
            if c.planet not in richness_map:
                p = PLANETS.get(c.planet)
                richness_map[c.planet] = p.resource_richness if p else 1.0
        richness_arr = _np.array([richness_map[c.planet] for c in cities], dtype=float)
        sector_arr = _np.array(
            [(c.industry + c.commerce + c.services) / 3 for c in cities], dtype=float
        )
        pop_infra_arr = _np.array(
            [c.population * c.infrastructure / 1000 for c in cities], dtype=float
        )
        econ = pop_infra_arr * richness_arr * sector_arr
        total_econ = float(econ.sum() * tech_bonus)
        return total_pop, total_econ

    total_pop = 0
    total_econ = 0.0
    richness_map: Dict[str, float] = {}
    for city in cities:
        city.process_turn(plague_resist)
        capacity = city.infrastructure * 20000 * cap_mult
        if city.population > capacity:
            loss = int((city.population - capacity) * 0.02)
            city.population -= loss
        if random.random() < 0.1:
            city.infrastructure += 1
        if city.infrastructure > 1 and random.random() < 0.02:
            city.infrastructure -= 1
        total_pop += city.population
        pname = city.planet
        if pname not in richness_map:
            p = PLANETS.get(pname)
            richness_map[pname] = p.resource_richness if p else 1.0
        richness = richness_map[pname]
        sector_bonus = (city.industry + city.commerce + city.services) / 3
        total_econ += city.population * city.infrastructure / 1000 * richness * sector_bonus * tech_bonus
    return total_pop, total_econ
