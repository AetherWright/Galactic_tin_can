"""Structured state representations used by lightweight AI models."""

from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

from .planets import PLANETS

if TYPE_CHECKING:  # pragma: no cover - imported for static analysis only
    from .models.nation import Nation


# ---------------------------------------------------------------------------
# Galaxy grid cache
# ---------------------------------------------------------------------------
# Stores (cache_key, star_cells, planet_cells) where star_cells / planet_cells
# map object names to (row, col) grid positions.  Recomputed only when the
# number of stars or planets changes or grid_size differs.
_galaxy_cell_cache: Tuple = ()


def _compute_galaxy_cells(
    grid_size: int,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
    """Return {name: (row, col)} maps for every star and planet."""
    from .models.nation import STARS  # local import to avoid circular dependency

    coords: List[Tuple[float, float]] = []
    for star in STARS.values():
        coords.append((star.x, star.y))
    for planet in PLANETS.values():
        coords.append((planet.x, planet.y))
    if not coords:
        return {}, {}

    min_x = min(x for x, _ in coords)
    max_x = max(x for x, _ in coords)
    min_y = min(y for _, y in coords)
    max_y = max(y for _, y in coords)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    gm1 = grid_size - 1

    def to_cell(x: float, y: float) -> Tuple[int, int]:
        col = int(round((x - min_x) / span_x * gm1))
        row = int(round((y - min_y) / span_y * gm1))
        return max(0, min(gm1, row)), max(0, min(gm1, col))

    star_cells = {s.name: to_cell(s.x, s.y) for s in STARS.values()}
    planet_cells = {p.name: to_cell(p.x, p.y) for p in PLANETS.values()}
    return star_cells, planet_cells


def build_alliance_matrix(
    nations: Dict[int, "Nation"]
) -> Tuple[List[List[int]], List[int], Dict[int, int]]:
    """Return a 2-D matrix indicating bilateral alliances between nations.

    The matrix is ordered by sorted nation ids and contains ``1`` when the
    column nation is an ally (including the nation itself) and ``0``
    otherwise. A companion list of ids and an index mapping are returned so
    callers can align the rows with specific nations.
    """

    ids = sorted(nations.keys())
    matrix: List[List[int]] = []
    for nid in ids:
        nation = nations[nid]
        row: List[int] = []
        for other_id in ids:
            if other_id == nid:
                row.append(1)
            else:
                row.append(1 if other_id in nation.alliances else 0)
        matrix.append(row)
    index = {nid: idx for idx, nid in enumerate(ids)}
    return matrix, ids, index


def _planet_owner(planet) -> int | None:
    pop_by_owner: Dict[int, int] = {}
    for city in planet.cities.values():
        if city.owner is None:
            continue
        pop_by_owner[city.owner] = pop_by_owner.get(city.owner, 0) + city.population
    if not pop_by_owner:
        return None
    return max(pop_by_owner.items(), key=lambda kv: kv[1])[0]


def galactic_control_layers(
    nation: "Nation",
    *,
    grid_size: int = 8,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Return friendly and hostile occupancy grids for the current galaxy.

    Star/planet cell positions are cached and only recomputed when the number
    of stars or planets changes or grid_size differs.
    """
    global _galaxy_cell_cache
    from .models.nation import STARS  # local import to avoid circular dependency

    grid_size = max(1, grid_size)
    friendly = [[0.0] * grid_size for _ in range(grid_size)]
    hostile = [[0.0] * grid_size for _ in range(grid_size)]

    cache_key = (len(STARS), len(PLANETS), grid_size)
    if not _galaxy_cell_cache or _galaxy_cell_cache[0] != cache_key:
        star_cells, planet_cells = _compute_galaxy_cells(grid_size)
        _galaxy_cell_cache = (cache_key, star_cells, planet_cells)
    else:
        _, star_cells, planet_cells = _galaxy_cell_cache

    if not star_cells and not planet_cells:
        return friendly, hostile

    ally_ids = set(nation.alliances)
    enemy_ids = set(nation.at_war)

    for star in STARS.values():
        cell = star_cells.get(star.name)
        if cell is None:
            continue
        row, col = cell
        owner = star.owner
        if owner is None:
            continue
        if owner == nation.id or owner in ally_ids:
            friendly[row][col] += 1.0
        elif owner in enemy_ids:
            hostile[row][col] += 1.0

    for planet in PLANETS.values():
        cell = planet_cells.get(planet.name)
        if cell is None:
            continue
        row, col = cell
        owner = _planet_owner(planet)
        if owner is None:
            continue
        if owner == nation.id or owner in ally_ids:
            friendly[row][col] += 0.3
        elif owner in enemy_ids:
            hostile[row][col] += 0.3

    return friendly, hostile
