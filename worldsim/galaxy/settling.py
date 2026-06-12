"""Initial nation placement: scoring and claiming starting colonies."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, TYPE_CHECKING

from ..planets import PLANETS, Colony, Planet
from .star import STARS

if TYPE_CHECKING:  # pragma: no cover
    from ..nations import Nation


def _score_colony(colony: Colony, planet: Planet) -> float:
    """Score a colony for initial placement — higher is better."""
    from ..planets.terrain import BIOME_RESOURCES
    res = BIOME_RESOURCES.get(planet.biome, {})
    if not res:
        return 0.5
    values = [(rng[0] + rng[1]) / 2.0 for rng in res.values()]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    balance = 1.0 / (1.0 + variance / 1000.0)
    richness = getattr(planet, "resource_richness", 1.0)
    return balance * richness


def _colony_available(col: Colony, planet: Planet) -> bool:
    coord = (col.x, col.y)
    return all(
        coord not in getattr(planet, attr, {})
        for attr in (
            "factories", "hospitals", "mines", "farms", "ports",
            "power_plants", "labs", "schools", "bases",
            "shipyards", "spaceports", "nuke_plants", "orbital_defenses"
        )
    )


def _place_nation(
    nation: Nation,
    colony: Colony,
    starting_population: int,
) -> None:
    """Upgrade a colony into a city and assign it to nation."""
    planet = PLANETS[colony.planet]
    planet.register_colony_usage(colony)
    city = planet.upgrade_colony(colony, nation.id)
    city.population = starting_population
    county = planet.get_county(city.x, city.y)
    if county:
        county.owner = nation.id
        if city not in county.cities:
            county.cities.append(city)
    nation.cities.append(city)
    nation.planet = city.planet
    nation.population = starting_population


def assign_initial_cities(
    nations: Dict[int, Nation],
    *,
    starting_population: int = 5000,
    prefer_balanced: bool = True,
) -> None:
    """Give each nation a starting city on a balanced world."""

    if not nations:
        return

    used_ids: Set[int] = set()
    used_planets: Set[str] = set()
    # Build scored candidate list
    candidates: List[Tuple[float, Colony, Planet]] = []
    for planet in PLANETS.values():
        for colony in planet.iter_colonies():
            if colony.owner is not None or not _colony_available(colony, planet):
                continue
            score = _score_colony(colony, planet) if prefer_balanced else 1.0
            candidates.append((score, colony, planet))

    # Expand if not enough candidates
    while len(candidates) < len(nations):
        expanded = False
        for planet in PLANETS.values():
            if planet.promote_dense_region():
                expanded = True
                break
        if not expanded:
            break
        # Rebuild candidate list after expansion
        candidates = []
        for planet in PLANETS.values():
            for colony in planet.iter_colonies():
                if colony.owner is not None or not _colony_available(colony, planet):
                    continue
                score = _score_colony(colony, planet) if prefer_balanced else 1.0
                candidates.append((score, colony, planet))

    if not candidates:
        return

    # Sort by score descending so balanced worlds are preferred
    candidates.sort(key=lambda x: x[0], reverse=True)

    for nation in nations.values():
        colony = None

        # First preference — a colony on the nation's assigned star
        star = next((s for s in STARS.values() if s.owner == nation.id), None)
        if star:
            for pname in star.planet_names:
                if pname in used_planets:
                    continue  # skip planets already occupied
                planet = PLANETS.get(pname)
                if not planet:
                    continue
                for c in planet.iter_colonies():
                    if c.owner is None and _colony_available(c, planet) and id(c) not in used_ids:
                        colony = c
                        used_ids.add(id(c))
                        used_planets.add(pname)
                        break
                if colony:
                            break
        # Fall back to best available scored candidate
        if colony is None:
            for score, c, planet in candidates:
                if id(c) not in used_ids and c.owner is None and c.planet not in used_planets:
                    colony = c
                    used_ids.add(id(c))
                    used_planets.add(c.planet)
                    break
        if colony is None:
            # Last resort — allow planet sharing if no unique planets remain
            for score, c, planet in candidates:
                if id(c) not in used_ids and c.owner is None:
                    colony = c
                    used_ids.add(id(c))
                    break
        if colony is None:
            continue

        _place_nation(nation, colony, starting_population)

