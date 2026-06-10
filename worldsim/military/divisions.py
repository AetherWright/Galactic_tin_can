"""Ground divisions: templates, recruitment, movement and training."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from ..ai import SimplePerceptron
from ..planets import PLANETS
from .logistics import _supply_throughput

if TYPE_CHECKING:
    from ..nations.nation import Nation


# ---------------------------------------------------------------------------
# Doctrine → maximum manpower fraction of total population
# ---------------------------------------------------------------------------

_DOCTRINE_MANPOWER: Dict[str, float] = {
    "total_war":         0.05,   # extreme wartime mobilization
    "offensive":         0.008,  
    "defensive":         0.007,
    "strategic_reserve": 0.015,  # peacetime standing force
    "economic":          0.002,
}
_DEFAULT_MANPOWER: float = 0.08


ORDER_TO_LABEL = {
    "attack": [1.0, 0.0, 0.0],
    "defend": [0.0, 1.0, 0.0],
    "reserve": [0.0, 0.0, 1.0],
}


@dataclass(slots=True)
class DivisionTemplate:
    """Template for creating specialised divisions."""

    name: str
    base_size: int
    equipment_mod: float = 1.0
    experience_mod: float = 1.0


@dataclass(slots=True)
class Division:
    soldiers: int
    x: int
    y: int
    planet: str
    experience: float = 1.0
    equipment: float = 1.0
    template: str = "Infantry"
    doctrine: str = "Balanced"
    controller: SimplePerceptron = field(
        default_factory=lambda: SimplePerceptron(3, n_outputs=3)
    )
    movement_controller: SimplePerceptron = field(
        default_factory=lambda: SimplePerceptron(6, n_outputs=3)
    )
    posture: str = "reserve"
    order: str = "reserve"
    in_transit: bool = False

    def decide_posture(self, context=None) -> str:
        if context is None:
            context = [self.soldiers / 1000.0, self.experience, self.equipment]
        scores = self.controller.predict_prob(context)
        idx = scores.index(max(scores)) if isinstance(scores, list) else 0
        self.posture = ["attack", "defend", "reserve"][idx]
        return self.posture

    def decide_movement(self, context: List[float]) -> str:
        """Return 'stay', 'deploy', or 'recall'."""
        scores = self.movement_controller.predict_prob(context)
        idx = scores.index(max(scores)) if isinstance(scores, list) else 0
        return ["stay", "deploy", "recall"][idx]

    @property
    def power(self) -> float:
        return self.soldiers * self.experience * self.equipment

def move_division(
    div: Division,
    nation: "Nation",
    enemy: "Nation",
    nations: Dict[int, "Nation"],
) -> None:

    dist_enemy = min(
        ((div.x - c.x)**2 + (div.y - c.y)**2)**0.5
        for c in enemy.cities
    ) if enemy.cities else 999.0
    dist_spaceport = min(
        ((div.x - s.x)**2 + (div.y - s.y)**2)**0.5
        for s in nation.spaceports
    ) if nation.spaceports else 999.0
    on_enemy_planet = 1.0 if div.planet == enemy.planet else 0.0
    supply = _supply_throughput(nation, PLANETS.get(div.planet))
    pop_ratio = nation.population / max(1, enemy.population)
    context = [
        dist_enemy / 1000.0,
        dist_spaceport / 1000.0,
        supply,
        on_enemy_planet,
        float(div.order == "attack"),
        min(2.0, pop_ratio),
    ]
    decision = div.decide_movement(context)
    
    if decision == "deploy" and div.planet == nation.planet:
        # Find friendly spaceports on enemy planet first
        friendly_ports = [
            s for nid in nation.alliances
            if (ally := nations.get(nid))
            for s in ally.spaceports
            if s.planet == enemy.planet
        ]
        # Fall back to own spaceports on enemy planet (previous captures)
        friendly_ports += [s for s in nation.spaceports if s.planet == enemy.planet]
        
        if friendly_ports:
            target = random.choice(friendly_ports)
            div.planet = enemy.planet
            div.x = target.x
            div.y = target.y
        elif enemy.spaceports:
            # Deep strike — riskier, costs experience
            target = random.choice(enemy.spaceports)
            div.planet = enemy.planet
            div.x = target.x
            div.y = target.y
            div.experience = max(0.1, div.experience * 0.85)
            div.equipment = max(0.1, div.equipment * 0.80)

    elif decision == "recall" and div.planet != nation.planet:
        home_ports = [s for s in nation.spaceports if s.planet == nation.planet]
        if home_ports:
            target = random.choice(home_ports)
            div.planet = nation.planet
            div.x = target.x
            div.y = target.y


def reward_divisions(divisions: List[Division], success: float) -> None:
    """Train division controllers based on battle ``success``.

    ``success`` should be positive for victory and negative for defeat.
    Successful divisions are trained toward their assigned order while
    unsuccessful ones are nudged toward a neutral reserve stance.
    """

    for div in divisions:
        context = [div.soldiers / 1000.0, div.experience, div.equipment]
        if success > 0:
            label = ORDER_TO_LABEL.get(div.order, ORDER_TO_LABEL["reserve"])
        else:
            label = ORDER_TO_LABEL["reserve"]
        div.controller.train([context], [label])


# ---------------------------------------------------------------------------
# City population helpers
# ---------------------------------------------------------------------------

def _deduct_city_population(nation: "Nation", amount: int) -> None:
    """Remove *amount* people from the nation's cities, spread proportionally.

    This makes recruitment and combat casualties persistent across turns
    because city populations are the source of ``nation.population``.
    """
    if amount <= 0 or not nation.cities:
        return
    total = sum(c.population for c in nation.cities)
    if total <= 0:
        return
    remaining = amount
    for city in nation.cities:
        if remaining <= 0:
            break
        share = int(amount * city.population / max(1, total))
        share = min(share, city.population, remaining)
        city.population = max(0, city.population - share)
        remaining -= share
    # Any rounding remainder comes from the largest city
    if remaining > 0:
        largest = max(nation.cities, key=lambda c: c.population)
        largest.population = max(0, largest.population - remaining)


def _apply_city_casualties(nation: "Nation", soldier_deaths: int) -> None:
    """Translate battlefield deaths into persistent city population loss."""
    _deduct_city_population(nation, soldier_deaths)


def build_division(nation: "Nation", template_name: Optional[str] = None) -> None:
    """Create a new division for ``nation`` using ``template_name``.

    Recruits are drawn from the city populations so the cost persists across
    turns.  ``nation.population`` is also decremented within the current turn
    so downstream budget checks in the same fifth stay consistent.
    """

    if not nation.division_templates:
        return
    if template_name is None:
        template_name = next(iter(nation.division_templates))
    tmpl = nation.division_templates.get(template_name)
    if not tmpl:
        return
    size = tmpl.base_size
    if nation.population < size:
        return
    # Deduct from city populations (persistent) and the within-turn total
    _deduct_city_population(nation, size)
    nation.population = max(0, nation.population - size)
    anchor = max(nation.cities, key=lambda c: c.population) if nation.cities else None
    x      = anchor.x      if anchor else 0
    y      = anchor.y      if anchor else 0
    planet = anchor.planet if anchor else nation.planet
    nation.divisions.append(
        Division(
            size,
            x,
            y,
            planet,
            tmpl.experience_mod,
            tmpl.equipment_mod,
            template_name,
            nation.doctrine,
        )
    )

def logistic_value(population: float, k: float, manpower_frac: float) -> float:
    """Return the logistic curve value — soldiers supportable at this population."""
    # S-curve from 0 to k*manpower_frac
    midpoint = k * 0.1  # inflection point at 10% of carrying capacity
    steepness = 0.00001  # tune this
    return (k * manpower_frac) / (1 + math.exp(-steepness * (population - midpoint)))
