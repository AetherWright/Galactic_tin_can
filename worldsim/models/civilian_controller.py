"""Civilian AI controller — standalone functions for the domestic policy loop.

All functions take a ``Nation`` instance as their first parameter, following
the same pattern used in ``war.py`` and ``diplomacy.py``.  The ``Nation``
class keeps thin wrapper methods so existing call sites are unaffected.
"""
from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from ..utils import distance_sq
from ..planets import PLANETS
from .war import build_division
from .government import PROJECT_CATALOG, PROJECT_NAMES, NationalProject
from .star import STARS

if TYPE_CHECKING:
    from .nation import Nation


def _civilian_state(nation: "Nation") -> List[float]:
    return [
        nation.economy,
        nation.technology.overall,
        nation.military,
        nation.infrastructure,
        nation.stability,
        float(len(nation.projects)),
        float(nation.star_count),
        nation._fleet_count(),
        float(len(nation.cities)),
        float(len(nation.divisions)),
        float(len(nation.mines)),
        float(len(nation.factories)),
        float(len(nation.schools)),
        float(len(nation.labs)),
        float(len(nation.hospitals)),
        nation.resources.get("metal", 0.0) / 100.0,
        nation.resources.get("uranium", 0.0) / 100.0,
        nation.resources.get("energy", 0.0) / 100.0,
        float(len(nation.at_war)),
        float(len(nation.alliances)),
    ]


def _execute_civilian_action(nation: "Nation", idx: int) -> None:
    actions = [
        nation.build_city,
        nation.build_base,
        nation.build_mine,
        nation.build_port,
        nation.build_factory,
        nation.build_hospital,
        nation.build_shipyard,
        nation.build_school,
        nation.build_power_plant,
        nation.build_spaceport,
        nation.build_lab,
        nation.build_nuke_facility,
        nation.build_orbital_defense,
        lambda: build_division(nation),
        lambda: nation.build_fleet_ship("Frigate"),
        lambda: nation.build_fleet_ship("Transport"),
        lambda: nation.build_fleet_ship("Cruiser"),
        lambda: nation.build_fleet_ship("Battleship"),
        nation.colonize_planet,
        nation.upgrade_assets,
        nation.start_project,
    ]
    if 0 <= idx < len(actions):
        actions[idx]()


def _valid_action_mask(nation: "Nation") -> List[bool]:
    has_nuke_tech = "Nuclear Weapons" in nation.tech_tree.unlocked
    has_shipyard = len(nation.shipyards) > 0
    has_spaceport = bool(nation.spaceports) if hasattr(nation, "spaceports") else False
    can_colonize = nation.star_count < len([s for s in STARS.values() if s.owner is None])

    return [
        True,           # build_city
        True,           # build_base
        True,           # build_mine
        True,           # build_port
        True,           # build_factory
        True,           # build_hospital
        True,           # build_shipyard
        True,           # build_school
        True,           # build_power_plant
        True,           # build_spaceport
        True,           # build_lab
        has_nuke_tech,  # build_nuke_facility
        True,           # build_orbital_defense
        True,           # build_division
        has_shipyard,   # frigate
        has_shipyard,   # transport
        has_shipyard,   # cruiser
        has_shipyard,   # battleship
        can_colonize,   # colonize_planet
        True,           # upgrade_assets
        True,           # start_project
    ]


def _apply_civilian_ai(nation: "Nation") -> None:
    state = _civilian_state(nation)

    valid_mask = _valid_action_mask(nation)
    idx = nation.civilian_ai.choose_action(state, valid_mask)

    nation.last_civilian_action = idx
    _execute_civilian_action(nation, idx)

    new_state = _civilian_state(nation)
    reward = nation.compute_reward("civilian", state, new_state)
    nation.civilian_ai.train(state, idx, reward, new_state)


def process_action_queue(nation: "Nation", limit: int = 2) -> None:
    for _ in range(min(limit, len(nation.action_queue))):
        idx = nation.action_queue.pop(0)
        state = _civilian_state(nation)
        _execute_civilian_action(nation, idx)
        new_state = _civilian_state(nation)
        reward = nation.compute_reward("civilian", state, new_state)
        nation.civilian_ai.train(state, idx, reward, new_state)


def _random_civilian_actions(
    nation: "Nation", nations: Dict[int, "Nation"]
) -> None:
    """Fallback actions using AI when available, otherwise random."""
    if random.random() < 0.05:
        nation.build_city()
    if nation.cities and nation.at_war:
        vulnerable = []
        for c in nation.cities:
            risk = (50 - nation.stability) / 100 if nation.stability < 50 else 0.0
            for eid in nation.at_war:
                enemy = nations.get(eid)
                if not enemy or enemy.planet != nation.planet:
                    continue
                dists_sq = [
                    distance_sq(c.coords, (d.x, d.y))
                    for d in enemy.divisions
                    if d.planet == nation.planet
                ]
                if enemy.cities:
                    dists_sq += [distance_sq(c.coords, ec.coords) for ec in enemy.cities]
                if dists_sq and min(dists_sq) < 75 * 75:
                    risk += 0.3
            if risk > 0 and random.random() < risk:
                vulnerable.append(c)
        if vulnerable:
            victim = min(vulnerable, key=lambda ct: ct.population)
            victim.owner = None
            if victim in nation.cities:
                nation.cities.remove(victim)

    if nation.civilian_ai is not None:
        state = _civilian_state(nation)
        idx = nation.civilian_ai.choose_action(state)
        _execute_civilian_action(nation, idx)
        new_state = _civilian_state(nation)
        reward = nation.compute_reward("civilian", state, new_state)
        nation.civilian_ai.train(state, idx, reward, new_state)
        return

    if random.random() < 0.03:
        nation.build_base()
    if random.random() < 0.02:
        nation.build_mine()
    if random.random() < 0.02:
        nation.build_port()
    if random.random() < 0.02:
        nation.build_factory()
    if random.random() < 0.01:
        nation.build_hospital()
    if random.random() < 0.01:
        nation.build_shipyard()
    if random.random() < 0.01:
        nation.build_school()
    if random.random() < 0.01:
        nation.build_power_plant()
    if random.random() < 0.01:
        nation.build_spaceport()
    if random.random() < 0.01:
        nation.build_nuke_facility()
    if random.random() < 0.01:
        nation.build_orbital_defense()
    if random.random() < 0.03:
        build_division(nation)
    if nation.spaceports and random.random() < 0.01:
        nation.colonize_planet()
    if random.random() < 0.01:
        nation.build_lab()
    if random.random() < 0.01:
        nation.start_project()
