"""Internal politics: collapses, succession states and civil wars."""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from ..core import vprint
from ..galaxy.genesis import get_next_nation_name
from ..galaxy.settling import assign_initial_cities
from ..nations import Nation
from ..planets import PLANETS, City
from ..society.culture import Culture


def _spawn_rebel_nation(parent: Nation, nations: Dict[int, Nation], year: int) -> None:
    """Create a breakaway nation from *parent* during a civil war."""

    new_id = max(nations) + 1
    new_name = get_next_nation_name()
    all_ids = list(parent.all_ids) + [new_id]
    rebel = Nation(
        new_name,
        new_id,
        parent.culture.copy(),
        all_ids,
        ai_table_dir=parent.ai_table_dir,
    )
    nations[new_id] = rebel
    rebel.year_born = year
    rebel.last_collapse = year

    for other in nations.values():
        if other.id == new_id:
            continue
        if new_id not in other.all_ids:
            other.all_ids.append(new_id)
            if other.military_ai:
                other.military_ai.set_allies_dimension(len(other.all_ids))
        other.relations[new_id] = "neutral"
        other.border_pressure[new_id] = 0.0
        rebel.relations[other.id] = "neutral"
        rebel.border_pressure[other.id] = 0.0

    if parent.cities:
        primary = parent.dominant_idea()
        taken: List[City] = []
        if primary is not None:
            for c in list(parent.cities):
                planet = PLANETS.get(c.planet)
                county = planet.get_county(c.x, c.y) if planet else None
                if county and county.dominant_idea() and county.dominant_idea() != primary:
                    taken.append(c)
        if not taken:
            rebel_share = max(1, len(parent.cities) // 2)
            taken = random.sample(parent.cities, rebel_share)
        for c in taken:
            c.owner = new_id
            parent.cities.remove(c)
            rebel.cities.append(c)
            rebel.planet = c.planet
            planet = PLANETS.get(c.planet)
            county = planet.get_county(c.x, c.y) if planet else None
            if county:
                county.owner = new_id

    ratio = 0.4
    rebel.military = parent.military * ratio
    parent.military *= 1 - ratio
    div_move = int(len(parent.divisions) * ratio)
    for _ in range(div_move):
        if not parent.divisions:
            break
        rebel.divisions.append(parent.divisions.pop())

    rebel.stability = 50.0
    parent.stability = max(parent.stability, 35.0)
    parent.at_war.add(new_id)
    rebel.at_war.add(parent.id)
    rebel.economy_linear = parent.economy_linear * ratio
    vprint(f"  {parent.name} erupts in civil war! {new_name} rebels.")


def _handle_internal_conflicts(nations: Dict[int, Nation], year: int) -> None:
    """Process collapses and potential civil wars."""

    collapsed = [n.id for n in nations.values() if n.population <= 0]
    for nid in collapsed:
        old = nations[nid]
        killer_alive = (
            old.killer_id is not None
            and old.killer_id in nations
            and nations[old.killer_id].population > 0
        )
        old.evolve_meta()
        for city in old.cities:
            city.owner = None
            city.population = 0
        old.cities.clear()

        if killer_alive:
            for other in nations.values():
                if other.id == nid:
                    continue
                other.alliances.discard(nid)
                other.trade_partners.discard(nid)
                other.at_war.discard(nid)
                other.relations.pop(nid, None)
                other.border_pressure.pop(nid, None)
            del nations[nid]
            vprint(f"  {old.name} has been eliminated!")
            continue

        new_name = get_next_nation_name()
        culture = Culture.random()
        new_nation = Nation(
            new_name,
            nid,
            culture,
            old.all_ids,
            ai_table_dir=old.ai_table_dir,
        )
        new_nation.year_born = year
        new_nation.last_collapse = year
        nations[nid] = new_nation
        assign_initial_cities({nid: new_nation})

        for other in nations.values():
            if other.id == nid:
                continue
            other.alliances.discard(nid)
            other.trade_partners.discard(nid)
            other.at_war.discard(nid)
            other.relations[nid] = "neutral"
        vprint(f"  {old.name} has collapsed! {new_name} rises.")

    for nation in list(nations.values()):
        if nation.population <= 0 or len(nation.cities) < 2:
            continue
        risk = 0.0
        if nation.stability < 50:
            risk = (50 - nation.stability) / 50
            risk += 0.001 * min(len(nation.cities), 5)
            risk -= min(0.3, nation.military / 50)
            risk = max(risk, 0.0)
        # Nations enjoy a 10-year grace period after collapsing
        years_since = year - nation.last_collapse
        if years_since < 10:
            risk *= years_since / 10
        if risk > 0 and random.random() < risk:
            _spawn_rebel_nation(nation, nations, year)
