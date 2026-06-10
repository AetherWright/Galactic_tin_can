"""Planet upkeep and star ownership resolution.

The two worker functions are module-level so they stay picklable for the
process pool.
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from ..core.memory import iter_memory_batches
from ..core.parallel import pooled_map
from ..galaxy.star import STARS
from ..planets import PLANETS

if TYPE_CHECKING:  # pragma: no cover
    from ..nations import Nation


def planet_turn_worker(state: tuple[str, float, float]) -> tuple[str, float, float]:
    name, plague, radiation = state
    if plague > 0:
        plague = max(0.0, plague - 0.01)
    if radiation > 0:
        radiation = max(0.0, radiation - 0.005)
    return name, plague, radiation


def star_owner_worker(
    data: tuple[str, list, dict[int, float], dict[int, float]]
) -> tuple[str, int | None]:
    """Return the most influential nation within a star system.

    Cities still determine base control. National economies and
    military strength act as modifiers so wealthy or heavily armed
    nations can claim contested systems. Fleets are approximated by
    the ``military`` score for now.
    """

    name, planets_data, economies, militaries = data
    scores: dict[int, float] = {}
    for city_list in planets_data:
        pop_by_owner: dict[int, int] = {}
        for owner, pop in city_list:
            if owner is not None:
                pop_by_owner[owner] = pop_by_owner.get(owner, 0) + pop
        if pop_by_owner:
            owner = max(pop_by_owner.items(), key=lambda kv: kv[1])[0]
            econ_bonus = economies.get(owner, 0.0) / 1000.0
            mil_bonus = militaries.get(owner, 0.0) / 100.0
            scores[owner] = scores.get(owner, 0.0) + 1 + econ_bonus + mil_bonus
    new_owner = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
    return name, new_owner


def process_planets() -> None:
    """Process planet turns in manageable chunks."""
    # TODO: incorporate seasonal effects and planetary climates
    planets = list(PLANETS.values())
    if not planets:
        return

    data = [(p.name, p.plague_level, p.radiation_level) for p in planets]
    results = pooled_map(planet_turn_worker, data, mode="process", chunksize=0)
    for name, plague, radiation in results:
        planet = PLANETS.get(name)
        if planet is not None:
            planet.plague_level = plague
            planet.radiation_level = radiation


def update_star_ownership(nations: Dict[int, "Nation"]) -> None:
    """Update star ownership in chunks."""
    # TODO: account for fleets when claiming stars
    stars = list(STARS.values())
    if not stars:
        return

    info = []
    economies = {nid: n.economy_linear for nid, n in nations.items()}
    militaries = {nid: n.military for nid, n in nations.items()}
    for star in stars:
        planets_data = []
        for pname in star.planet_names:
            planet = PLANETS.get(pname)
            if not planet:
                continue
            cities = [(c.owner, c.population) for c in planet.cities.values()]
            planets_data.append(cities)
        info.append((star.name, planets_data, economies, militaries))

    # Every task pickles the per-star city data plus both nation dicts, so
    # the submission footprint grows with stars × nations.  Chunk it against
    # the RAM budget; a single chunk covers the common case.
    item_nbytes = 256 + 128 * max(1, len(nations))
    for batch in iter_memory_batches(info, item_nbytes, fraction=0.05, lo=8):
        results = pooled_map(star_owner_worker, batch, mode="process", chunksize=0)
        for name, owner in results:
            star = STARS.get(name)
            if star is not None:
                star.owner = owner
