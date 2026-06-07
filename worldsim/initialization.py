"""World creation helpers for WorldSim."""

from __future__ import annotations

import importlib
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as _np

from .config import load_list
from .culture import Culture
from .mp_utils import pooled_map
from .models import Nation, Star, STARS
from .planets import PLANETS, Planet, Colony

_REGISTERED_PLANETS: List[Planet] = []


_cupy_spec = importlib.util.find_spec("cupy")
if _cupy_spec is not None:
    cupy = importlib.import_module("cupy")
else:
    cupy = None


__all__ = [
    "init_world",
    "assign_initial_stars",
    "assign_initial_cities",
    "add_star_cluster",
    "add_star_systems",
    "get_next_nation_name",
]


# Pool of names used when creating or replacing nations. These are loaded lazily
# from configuration files to avoid keeping multiple large lists in memory
# across modules.
NATION_NAMES: List[str] = load_list("nation_names")

# Name fragments used once predefined options are exhausted
NAME_PREFIXES: List[str] = load_list("name_prefixes")
NAME_SUFFIXES: List[str] = load_list("name_suffixes")

# Counter used to generate unique nation names when collapsed states are
# replaced
NEXT_NATION_NAME: List[int] = [0]


def _register_planet(planet: Planet) -> None:
    """Store ``planet`` in the global registry with a strong reference."""

    PLANETS[planet.name] = planet
    _REGISTERED_PLANETS.append(planet)


def get_next_nation_name() -> str:
    """Return a unique nation name."""

    idx = NEXT_NATION_NAME[0]
    NEXT_NATION_NAME[0] += 1
    if idx < len(NATION_NAMES):
        return NATION_NAMES[idx]
    return random.choice(NAME_PREFIXES) + random.choice(NAME_SUFFIXES)


def _int_dtype(xp_module) -> object:
    return getattr(xp_module, "int32", int)


def _get_backend(seed: int):
    if cupy is not None:
        return cupy, cupy.random.default_rng(seed)
    return _np, _np.random.default_rng(seed)


def _as_numpy(array):
    if cupy is not None and isinstance(array, cupy.ndarray):
        return cupy.asnumpy(array)
    return array


def _plan_cluster_tasks(num_stars: int) -> List[tuple[int, int, int, int]]:
    remaining = num_stars
    cluster_tasks: List[tuple[int, int, int, int]] = []
    cluster_id = 0
    start_index = 0
    while remaining > 0:
        size = min(remaining, random.randint(1, 3))
        seed = random.randint(0, 2**31 - 1)
        cluster_tasks.append((cluster_id, size, start_index, seed))
        cluster_id += 1
        start_index += size
        remaining -= size
    return cluster_tasks


def _build_star_cluster_cpu(state: tuple[int, int, int, int]):
    cluster_id, size, start_index, seed = state
    rng = random.Random(seed)
    cx, cy, cz = (rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-100, 100))
    radius = 30.0
    planets: list[dict] = []
    stars: list[dict] = []
    star_index = start_index
    for _ in range(size):
        sx = cx + rng.uniform(-radius, radius)
        sy = cy + rng.uniform(-radius, radius)
        sz = cz + rng.uniform(-radius, radius)
        star_name = f"Star {star_index}"
        star_index += 1
        planet_names: list[str] = []
        for j in range(rng.randint(1, 3)):
            pname = f"{star_name}-P{j}"
            planets.append(
                {
                    "name": pname,
                    "width": 100,
                    "height": 100,
                    "spacing": 10,
                    "x": sx + rng.uniform(-15, 15),
                    "y": sy + rng.uniform(-15, 15),
                    "z": sz + rng.uniform(-15, 15),
                }
            )
            planet_names.append(pname)
        stars.append(
            {
                "name": star_name,
                "planets": planet_names,
                "x": sx,
                "y": sy,
                "z": sz,
                "cluster": cluster_id,
            }
        )
    return cluster_id, planets, stars


def _build_star_cluster_gpu(state: tuple[int, int, int, int]):
    cluster_id, size, start_index, seed = state
    xp, rng = _get_backend(seed)
    center = rng.uniform(-100.0, 100.0, size=3, dtype=xp.float32)
    star_offsets = rng.uniform(-30.0, 30.0, size=(size, 3), dtype=xp.float32)
    star_coords = center + star_offsets
    planet_counts = rng.integers(1, 4, size=size, dtype=_int_dtype(xp))
    planet_counts_np = _as_numpy(planet_counts).astype(int, copy=False)
    star_coords_np = _as_numpy(star_coords)
    total_planets = int(planet_counts_np.sum())
    if total_planets:
        planet_offsets_np = _as_numpy(
            rng.uniform(-5.0, 5.0, size=(total_planets, 3), dtype=xp.float32)
        )
    else:
        planet_offsets_np = _np.empty((0, 3), dtype=_np.float32)

    planets: list[dict] = []
    stars: list[dict] = []
    offset_idx = 0
    for local_idx, (sx, sy, sz) in enumerate(star_coords_np):
        star_name = f"Star {start_index + local_idx}"
        planet_names: list[str] = []
        count = planet_counts_np[local_idx]
        if count:
            offsets = planet_offsets_np[offset_idx : offset_idx + count]
            offset_idx += count
            for j, (ox, oy, oz) in enumerate(offsets):
                pname = f"{star_name}-P{j}"
                planets.append(
                    {
                        "name": pname,
                        "width": 100,
                        "height": 100,
                        "spacing": 10,
                        "x": float(sx + ox),
                        "y": float(sy + oy),
                        "z": float(sz + oz),
                    }
                )
                planet_names.append(pname)
        stars.append(
            {
                "name": star_name,
                "planets": planet_names,
                "x": float(sx),
                "y": float(sy),
                "z": float(sz),
                "cluster": cluster_id,
            }
        )
    return cluster_id, planets, stars


def _build_clusters(tasks: List[tuple[int, int, int, int]]):
    if not tasks:
        return []
    if cupy is not None:
        return [_build_star_cluster_gpu(task) for task in tasks]
    return pooled_map(
        _build_star_cluster_cpu,
        tasks,
        mode="process",
        chunksize=0,
    )


def init_world(
    num_nations: int = 5,
    num_stars: int = 3,
    *,
    starting_population: int = 5000,
    use_java_planets: bool | None = None,
    ai_directory: str | Path | None = None,
) -> Dict[int, Nation]:
    """Create a new world with ``num_nations`` each starting at ``starting_population``."""

    if use_java_planets is not None:
        from .planets import set_use_java

        set_use_java(use_java_planets)

    ai_dir = Path(ai_directory).resolve() if ai_directory else None

    PLANETS.clear()
    _REGISTERED_PLANETS.clear()
    STARS.clear()

    cluster_tasks = _plan_cluster_tasks(num_stars)
    results = _build_clusters(cluster_tasks)

    for _, planet_data, star_data in results:
        for pdata in planet_data:
            planet = Planet(
                pdata["name"],
                pdata["width"],
                pdata["height"],
                pdata["spacing"],
                x=pdata["x"],
                y=pdata["y"],
                z=pdata["z"],
            )
            _register_planet(planet)
        for sdata in star_data:
            STARS[sdata["name"]] = Star(
                sdata["name"],
                sdata["planets"],
                x=sdata["x"],
                y=sdata["y"],
                z=sdata["z"],
                cluster=sdata["cluster"],
            )

    nations: Dict[int, Nation] = {}
    for i in range(num_nations):
        culture = Culture.random()
        name = get_next_nation_name()
        nations[i] = Nation(
            name,
            i,
            culture,
            list(range(num_nations)),
            ai_table_dir=ai_dir,
        )
        nations[i].year_born = 0
        nations[i].last_collapse = 0
    assign_initial_cities(nations, starting_population=starting_population)
    return nations

def _score_colony(colony: Colony, planet: Planet) -> float:
    """Score a colony for initial placement — higher is better."""
    from .planets.terrain import BIOME_RESOURCES
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


def add_star_cluster(size: int = 3) -> None:
    """Create a group of stars near the same coordinates."""

    cluster_id = max((s.cluster for s in STARS.values()), default=-1) + 1
    start_index = len(STARS)
    seed = random.randint(0, 2**31 - 1)
    tasks = [(cluster_id, size, start_index, seed)]
    results = _build_clusters(tasks)
    if not results:
        return
    _, planet_data, star_data = results[0]
    for pdata in planet_data:
        planet = Planet(
            pdata["name"],
            pdata["width"],
            pdata["height"],
            pdata["spacing"],
            x=pdata["x"],
            y=pdata["y"],
            z=pdata["z"],
        )
        _register_planet(planet)
    for sdata in star_data:
        STARS[sdata["name"]] = Star(
            sdata["name"],
            sdata["planets"],
            x=sdata["x"],
            y=sdata["y"],
            z=sdata["z"],
            cluster=sdata["cluster"],
        )


def add_star_systems(num: int = 1) -> None:
    """Create ``num`` additional star systems with random planets."""

    for _ in range(num):
        add_star_cluster(1)

