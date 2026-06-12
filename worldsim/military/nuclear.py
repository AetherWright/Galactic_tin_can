"""Nuclear weapons: production, strikes and first-strike doctrine."""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from ..core import wprint
from ..planets import PLANETS

if TYPE_CHECKING:
    from ..nations.nation import Nation


def produce_nuclear_weapons(nation: "Nation") -> None:
    """Convert uranium, metal and economy into nuclear stockpile."""
    if "Atomic Engineering" not in nation.tech_tree.unlocked:
        return
    uranium = nation.resources.get("uranium", 0.0)
    metal = nation.resources.get("metal", 0.0)
    base = int(min(uranium // 10, metal // 20, nation.economy_linear // 100))
    if base <= 0:
        return
    rate = sum(f.rate for f in nation.nuke_plants) if nation.nuke_plants else 1.0
    possible = int(
        min(
            base * rate,
            nation.resources.get("uranium", 0.0) // 10,
            nation.resources.get("metal", 0.0) // 20,
            nation.economy_linear // 100,
        )
    )
    if possible > 0:
        nation.nuclear_stockpile += possible
        nation.resources["uranium"] -= possible * 10
        nation.resources["metal"] -= possible * 20
        nation.economy_linear -= possible * 100


def launch_nuclear_strike(nation: "Nation", enemy: "Nation") -> None:
    """Inflict heavy losses on ``enemy`` using one warhead."""
    from ..planets import PLANETS

    nation.relations[enemy.id] = "enemy"
    enemy.relations[nation.id] = "enemy"
    nation.at_war.add(enemy.id)
    enemy.at_war.add(nation.id)

    nation.nuclear_stockpile -= 1

    tech_scale = 1.0 + nation.technology.military / 100.0
    defense = sum(d.strength for d in enemy.orbital_defenses)
    damp = max(0.0, 1.0 - defense)

    casualties = 0

    if enemy.cities:
        target = max(enemy.cities, key=lambda c: c.population)
        city_loss = int(target.population * 0.8 * tech_scale * damp)
        target.population = max(0, target.population - city_loss)
        infra_loss = max(1, int(target.infrastructure * 0.7 * damp))
        target.infrastructure = max(0, target.infrastructure - infra_loss)
        casualties += city_loss
    else:
        casualties += int(enemy.population * 0.2 * damp)

    enemy.population = max(0, enemy.population - casualties)
    enemy.economy_linear *= 1.0 - 0.7 * damp
    enemy.stability = max(0.0, enemy.stability - 40 * damp)
    nation.stability = max(0.0, nation.stability - 10)

    planet = PLANETS.get(enemy.planet)
    if planet:
        planet.radiation_level = min(1.0, planet.radiation_level + 0.4 * damp)

    wprint(nation.name, f"  {nation.name} launches a nuclear strike on {enemy.name}!")


def launch_first_strike(nation: "Nation", enemies: List["Nation"]) -> None:
    """Launch all available warheads across ``enemies`` in round-robin fashion."""
    enemies = [e for e in enemies if e is not nation]
    if nation.nuclear_stockpile <= 0 or not enemies:
        return

    idx = 0
    while nation.nuclear_stockpile > 0:
        enemy = enemies[idx % len(enemies)]
        launch_nuclear_strike(nation, enemy)
        if enemy.nuclear_stockpile > 0 and random.random() < 0.5:
            launch_nuclear_strike(enemy, nation)
        idx += 1


def consider_first_strike(nation: "Nation", nations: Dict[int, "Nation"]) -> None:
    """Allow the military AI to initiate a nuclear first strike."""
    if not nation.military_ai or nation.nuclear_stockpile <= 0:
        return
    potential = [
        n
        for n in nations.values()
        if n.id != nation.id
        and n.id not in nation.at_war
        and nation.relations.get(n.id, "neutral") != "ally"
    ]
    if not potential:
        return
    state = [
        float(nation.nuclear_stockpile),
        nation.military,
        nation.economy,
        nation.stability,
        float(sum(e.nuclear_stockpile for e in potential)),
    ]
    act = nation.military_ai.choose_action(state)
    if act == 1:
        target = max(potential, key=lambda n: n.military)
        launch_first_strike(nation, [target])
