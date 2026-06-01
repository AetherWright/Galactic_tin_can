"""Infrastructure construction and resource collection for nations."""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from ..planets import (
    PLANETS,
    City,
    MilitaryBase,
    Mine,
    Port,
    Factory,
    Hospital,
    Shipyard,
    Spaceport,
    School,
    PowerPlant,
    ResearchLab,
    NuclearFacility,
    OrbitalDefense,
)
from ..config import load_json

if TYPE_CHECKING:
    from .nation import Nation


# Resource costs for constructing various assets
RESOURCE_COSTS: Dict[str, Dict[str, float]] = load_json("resource_costs")


# ---------------------------------------------------------------------------
# Resource collection
# ---------------------------------------------------------------------------

def collect_resources(nation: "Nation") -> None:
    for mine in nation.mines:
        planet = PLANETS.get(mine.planet)
        if not planet:
            continue
        gained_m = planet.extract_resource("metal", mine.output)
        nation.add_resource("metal", gained_m)
        gained_u = planet.extract_resource("uranium", getattr(mine, "uranium", 0.0))
        nation.add_resource("uranium", gained_u)
    for plant in nation.power_plants:
        planet = PLANETS.get(plant.planet)
        if not planet:
            continue
        gained = planet.extract_resource("energy", plant.output)
        nation.add_resource("energy", gained)
    # biology: Synthetic Biology raises agricultural yield (food_output)
    food_mult = nation.tech_bonuses.get("food_output", 1.0)
    for _city in nation.cities:
        planet = PLANETS.get(_city.planet)
        if not planet:
            continue
        nation.add_resource("food", planet.extract_resource("food", 5.0 * food_mult))


# ---------------------------------------------------------------------------
# Build methods
# ---------------------------------------------------------------------------

def build_city(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet:
        return
    cost = RESOURCE_COSTS["city"]
    if not nation.has_resources(cost):
        return
    free = [co for co in planet.iter_colonies() if co.owner is None]
    for co in free:
        if nation._can_add_city(co):
            co.owner = nation.id
            planet.register_colony_usage(co)
            city = planet.upgrade_colony(co, nation.id)
            nation.cities.append(city)
            if co in nation.colonies:
                nation.colonies.remove(co)
            nation.infrastructure += 1
            nation.spend_resources(cost)
            nation._update_centroids()
            break


def build_base(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["base"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.bases:
        return
    base = MilitaryBase(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_base(base)
    nation.bases.append(base)
    nation.spend_resources(cost)


def build_mine(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["mine"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.mines:
        return
    mine = Mine(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_mine(mine)
    nation.mines.append(mine)
    nation.spend_resources(cost)


def build_port(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["port"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.ports:
        return
    port = Port(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_port(port)
    nation.ports.append(port)
    nation.spend_resources(cost)


def build_factory(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["factory"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.factories:
        return
    fac = Factory(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_factory(fac)
    nation.factories.append(fac)
    nation.spend_resources(cost)


def build_hospital(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["hospital"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.hospitals:
        return
    hos = Hospital(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_hospital(hos)
    nation.hospitals.append(hos)
    nation.spend_resources(cost)


def build_shipyard(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["shipyard"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.shipyards:
        return
    yard = Shipyard(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_shipyard(yard)
    nation.shipyards.append(yard)
    nation.spend_resources(cost)


def build_school(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["school"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.schools:
        return
    school = School(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_school(school)
    nation.schools.append(school)
    nation.spend_resources(cost)


def build_power_plant(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["power_plant"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.power_plants:
        return
    plant = PowerPlant(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_power_plant(plant)
    nation.power_plants.append(plant)
    nation.spend_resources(cost)


def build_lab(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["lab"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.labs:
        return
    lab = ResearchLab(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_lab(lab)
    nation.labs.append(lab)
    nation.spend_resources(cost)


def build_nuke_facility(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["nuke_facility"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.nuke_plants:
        return
    fac = NuclearFacility(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_nuke_facility(fac)
    nation.nuke_plants.append(fac)
    nation.spend_resources(cost)


def build_orbital_defense(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["orbital_defense"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.orbital_defenses:
        return
    od = OrbitalDefense(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_orbital_defense(od)
    nation.orbital_defenses.append(od)
    nation.spend_resources(cost)


def build_spaceport(nation: "Nation") -> None:
    planet = PLANETS.get(nation.planet)
    if not planet or not nation.cities:
        return
    cost = RESOURCE_COSTS["spaceport"]
    if not nation.has_resources(cost):
        return
    anchor = max(nation.cities, key=lambda c: c.population)
    if (anchor.x, anchor.y) in planet.spaceports:
        return
    port = Spaceport(anchor.x, anchor.y, nation.planet, owner=nation.id)
    planet.add_spaceport(port)
    nation.spaceports.append(port)
    nation.spend_resources(cost)


# ---------------------------------------------------------------------------
# Asset upgrades and colonisation
# ---------------------------------------------------------------------------

def upgrade_assets(nation: "Nation") -> None:
    """Invest economy into upgrading owned infrastructure."""
    if nation.economy_linear <= 0:
        return
    if nation.cities and nation.economy_linear >= 20:
        nation.cities[0].upgrade()
        nation.economy_linear -= 20
    if nation.bases and nation.economy_linear >= 15:
        nation.bases[0].upgrade()
        nation.economy_linear -= 15
    if nation.mines and nation.economy_linear >= 10:
        nation.mines[0].upgrade()
        nation.economy_linear -= 10
    if nation.ports and nation.economy_linear >= 15:
        nation.ports[0].upgrade()
        nation.economy_linear -= 15
    if nation.factories and nation.economy_linear >= 20:
        nation.factories[0].upgrade()
        nation.economy_linear -= 20
    if nation.hospitals and nation.economy_linear >= 10:
        nation.hospitals[0].upgrade()
        nation.economy_linear -= 10
    if nation.shipyards and nation.economy_linear >= 20:
        nation.shipyards[0].upgrade()
        nation.economy_linear -= 20
    if nation.schools and nation.economy_linear >= 10:
        nation.schools[0].upgrade()
        nation.economy_linear -= 10
    if nation.labs and nation.economy_linear >= 10:
        nation.labs[0].upgrade()
        nation.economy_linear -= 10
    if nation.power_plants and nation.economy_linear >= 15:
        nation.power_plants[0].upgrade()
        nation.economy_linear -= 15
    if nation.spaceports and nation.economy_linear >= 20:
        nation.spaceports[0].upgrade()
        nation.economy_linear -= 20


def colonize_planet(nation: "Nation") -> None:
    """Attempt to found a city on another planet if one is free."""
    candidates = []
    for planet in PLANETS.values():
        free = [c for c in planet.iter_colonies() if c.owner is None]
        if free:
            candidates.append((planet, free))
    if not candidates:
        return
    candidates.sort(key=lambda item: -len(item[1]))
    for target, free in candidates:
        for colony in free:
            if nation._can_add_city(colony):
                colony.owner = nation.id
                target.register_colony_usage(colony)
                city = target.upgrade_colony(colony, nation.id)
                nation.cities.append(city)
                if colony in nation.colonies:
                    nation.colonies.remove(colony)
                nation._update_centroids()
                return
