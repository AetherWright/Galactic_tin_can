"""Planet data structures and global registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Tuple, TYPE_CHECKING, Any, Iterator
from collections import defaultdict
import random

from ..routes import RouteGraph
from .terrain import BIOMES, generate_resources

if TYPE_CHECKING:
    from .city import City
    from .infrastructure import (
        MilitaryBase,
        Mine,
        Farm,
        Port,
        Factory,
        Hospital,
        Shipyard,
        Spaceport,
        School,
        PowerPlant,
        ResearchLab,
        NuclearFacility,
    )

from .city import City
from .colony import Colony
from .county import County
from .infrastructure import (
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
    Farm,
)
from ..utils import distance, travel_time, APPROXIMATE


@dataclass
class _DetailState:
    """Track adaptive level-of-detail information for a planetary county."""

    level: str = "coarse"
    usage: int = 0
    generated: str = "none"


_DETAIL_LEVELS: Dict[str, int] = {"none": 0, "coarse": 1, "fine": 2}


@dataclass
class Planet:
    """A single planet containing cities and infrastructure."""

    name: str
    width: int
    height: int
    spacing: int = 10
    county_size: int = 50
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    plague_level: float = 0.0
    radiation_level: float = 0.0
    biome: str = field(default_factory=lambda: random.choice(BIOMES))
    resources: Dict[str, float] = field(default_factory=dict)
    resource_richness: float = field(default_factory=lambda: random.uniform(0.5, 1.5))
    colonies: Dict[Tuple[int, int], 'Colony'] = field(init=False, repr=False)
    cities: Dict[Tuple[int, int], 'City'] = field(default_factory=dict)
    bases: Dict[Tuple[int, int], 'MilitaryBase'] = field(default_factory=dict)
    mines: Dict[Tuple[int, int], 'Mine'] = field(default_factory=dict)
    ports: Dict[Tuple[int, int], 'Port'] = field(default_factory=dict)
    factories: Dict[Tuple[int, int], 'Factory'] = field(default_factory=dict)
    hospitals: Dict[Tuple[int, int], 'Hospital'] = field(default_factory=dict)
    shipyards: Dict[Tuple[int, int], 'Shipyard'] = field(default_factory=dict)
    spaceports: Dict[Tuple[int, int], 'Spaceport'] = field(default_factory=dict)
    schools: Dict[Tuple[int, int], 'School'] = field(default_factory=dict)
    power_plants: Dict[Tuple[int, int], 'PowerPlant'] = field(default_factory=dict)
    labs: Dict[Tuple[int, int], 'ResearchLab'] = field(default_factory=dict)
    nuke_plants: Dict[Tuple[int, int], 'NuclearFacility'] = field(default_factory=dict)
    orbital_defenses: Dict[Tuple[int, int], 'OrbitalDefense'] = field(default_factory=dict)
    farms: Dict[Tuple[int, int], 'Farm'] = field(default_factory=dict)
    counties: Dict[Tuple[int, int], 'County'] = field(default_factory=dict)
    routes: RouteGraph = field(default_factory=RouteGraph)
    _distance_cache: Any = field(default=None, init=False, repr=False)
    _routes_dirty: bool = field(default=False, init=False, repr=False)
    _cache_version: int = field(default=-1, init=False, repr=False)
    _detail_states: Dict[Tuple[int, int], _DetailState] = field(
        default_factory=dict, init=False, repr=False
    )
    _county_colonies: Dict[Tuple[int, int], set[Tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _detail_threshold: int = field(default=12, init=False, repr=False)
    _coarse_factor: int = field(default=3, init=False, repr=False)
    _coarse_step: int = field(default=0, init=False, repr=False)
    # Set during bulk colony generation to defer _mark_dirty() until the end.
    _batch_adding: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.resources:
            self.resources = generate_resources(self.biome)
        self._distance_cache = lru_cache(maxsize=2048)(self._compute_distance)
        self.colonies = {}
        self._county_colonies = defaultdict(set)
        self._detail_states = {}
        county_size = self.county_size
        counties = self.counties
        coarse_factor = max(1, self._coarse_factor)
        self._coarse_factor = coarse_factor
        self._coarse_step = self.spacing * coarse_factor
        self._detail_threshold = max(4, (county_size // self.spacing) * 2)
        for cx in range(0, self.width, county_size):
            for cy in range(0, self.height, county_size):
                idx = (cx // county_size, cy // county_size)
                name = f"{self.name}-C{idx[0]}-{idx[1]}"
                counties[idx] = County(
                    name,
                    cx,
                    cy,
                    county_size,
                    county_size,
                    rural_population=1000,
                )
                self._detail_states[idx] = _DetailState()

        randrange = random.randrange
        spacing = self.spacing
        mx = randrange(0, self.width, spacing)
        my = randrange(0, self.height, spacing)
        self.add_mine(Mine(mx, my, self.name))
        px = self.width // 2 + spacing // 2
        py = self.height // 2 + spacing // 2
        self.add_port(Port(px, py, self.name))
        fx = randrange(0, self.width, spacing)
        fy = randrange(0, self.height, spacing)
        self.add_factory(Factory(fx, fy, self.name))
        sx = randrange(0, self.width, spacing)
        sy = randrange(0, self.height, spacing)
        self.add_school(School(sx, sy, self.name))
        px2 = randrange(0, self.width, spacing)
        py2 = randrange(0, self.height, spacing)
        self.add_power_plant(PowerPlant(px2, py2, self.name))
        lx = randrange(0, self.width, spacing)
        ly = randrange(0, self.height, spacing)
        self.add_lab(ResearchLab(lx, ly, self.name))
        farmx = randrange(0, self.width, spacing)
        farmy = randrange(0, self.height, spacing)
        self.add_farm(Farm(farmx, farmy, self.name))
        self._cache_version = self.routes.version


    @property
    def coords(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    # --- Adaptive colony detail management -------------------------------------------------

    def _county_index(self, x: int, y: int) -> Tuple[int, int]:
        """Return the county grid index for coordinate ``(x, y)``."""

        return x // self.county_size, y // self.county_size

    def _ensure_county_detail(self, idx: Tuple[int, int], target: str) -> None:
        """Ensure ``idx`` has at least ``target`` level of colony detail."""

        state = self._detail_states[idx]
        current_level = _DETAIL_LEVELS[state.generated]
        desired_level = _DETAIL_LEVELS[target]
        if current_level >= desired_level:
            return
        if desired_level >= _DETAIL_LEVELS["coarse"] and current_level < _DETAIL_LEVELS["coarse"]:
            self._generate_coarse(idx)
            state.generated = "coarse"
            current_level = _DETAIL_LEVELS["coarse"]
        if desired_level >= _DETAIL_LEVELS["fine"] and current_level < _DETAIL_LEVELS["fine"]:
            self._generate_fine(idx)
            state.generated = "fine"
        if _DETAIL_LEVELS[state.level] < desired_level:
            state.level = target

    def _generate_coarse(self, idx: Tuple[int, int]) -> None:
        """Populate ``idx`` with a sparse grid of colonies."""

        county = self.counties[idx]
        step = max(self._coarse_step, self.spacing)
        x_end = min(self.width, county.x + county.width)
        y_end = min(self.height, county.y + county.height)
        self.routes.begin_batch_update()
        self._batch_adding = True
        try:
            for x in range(county.x, x_end, step):
                for y in range(county.y, y_end, step):
                    coord = (x, y)
                    if coord not in self._county_colonies[idx]:
                        self._county_colonies[idx].add(coord)
                    self._add_colony(x, y)
        finally:
            self._batch_adding = False
            self.routes.end_batch_update()
            self._mark_dirty()

    def _generate_fine(self, idx: Tuple[int, int]) -> None:
        """Populate ``idx`` with the full-resolution colony grid."""

        county = self.counties[idx]
        x_end = min(self.width, county.x + county.width)
        y_end = min(self.height, county.y + county.height)
        self.routes.begin_batch_update()
        self._batch_adding = True
        try:
            for x in range(county.x, x_end, self.spacing):
                for y in range(county.y, y_end, self.spacing):
                    coord = (x, y)
                    if coord not in self._county_colonies[idx]:
                        self._county_colonies[idx].add(coord)
                    self._add_colony(x, y)
        finally:
            self._batch_adding = False
            self.routes.end_batch_update()
            self._mark_dirty()

    def _add_colony(self, x: int, y: int) -> 'Colony':
        """Create a colony at ``(x, y)`` if missing and wire it into routes."""

        key = (x, y)
        colony = self.colonies.get(key)
        if colony is None:
            colony = Colony(x, y, self.name)
            self.colonies[key] = colony
            self.routes.add_node((x, y, "city"))
            self._connect_city_node(x, y)
            if not self._batch_adding:
                self._mark_dirty()
        return colony

    def _connect_city_node(self, x: int, y: int) -> None:
        """Connect city node ``(x, y)`` to nearby infrastructure."""

        node = (x, y, "city")
        spacing = float(self.spacing)
        neighbors = [
            (x - self.spacing, y),
            (x + self.spacing, y),
            (x, y - self.spacing),
            (x, y + self.spacing),
        ]
        coarse_step = max(self._coarse_step, self.spacing)
        if coarse_step != self.spacing:
            neighbors.extend(
                [
                    (x - coarse_step, y),
                    (x + coarse_step, y),
                    (x, y - coarse_step),
                    (x, y + coarse_step),
                ]
            )
        for nx, ny in neighbors:
            neighbor = (nx, ny, "city")
            if self.routes.has_node(neighbor):
                self.routes.add_edge(node, neighbor, distance((x, y), (nx, ny)))

        connectable = {
            "city",
            "base",
            "mine",
            "port",
            "factory",
            "hospital",
            "shipyard",
            "school",
            "power",
            "lab",
            "spaceport",
            "nuke",
            "orbital",
        }
        radius = float(self._coarse_step * 1.5)
        for existing in list(self.routes.nodes()):
            if existing == node:
                continue
            ex, ey, kind = existing
            if kind not in connectable:
                continue
            if kind == "city" and (abs(ex - x) <= self.spacing and abs(ey - y) <= self.spacing):
                # Already linked above using cardinal neighbors.
                continue
            dist = distance((x, y), (ex, ey))
            if dist <= radius:
                self.routes.add_edge(node, existing, dist)

    def _ensure_city_node(self, node: Tuple[int, int, str]) -> None:
        """Ensure a city node exists for ``node`` coordinates."""

        x, y, kind = node
        if kind != "city":
            return
        idx = self._county_index(x, y)
        state = self._detail_states.get(idx)
        if state is None:
            return
        self._ensure_county_detail(idx, state.level)
        if (x, y) not in self.colonies:
            # Promote to fine detail if the requested coordinate is not represented yet.
            if state.level != "fine":
                self._ensure_county_detail(idx, "fine")
                state.level = "fine"
            if (x, y) not in self.colonies:
                self._add_colony(x, y)

    def iter_colonies(self, detail: str = "auto") -> Iterator['Colony']:
        """Yield colonies honouring adaptive detail settings."""

        if detail not in {"auto", "coarse", "fine"}:
            raise ValueError("detail must be 'auto', 'coarse', or 'fine'")
        for idx, state in self._detail_states.items():
            target = state.level if detail == "auto" else detail
            self._ensure_county_detail(idx, target)
            for coord in sorted(self._county_colonies[idx]):
                colony = self.colonies.get(coord)
                if colony is not None:
                    yield colony

    def register_colony_usage(self, colony: 'Colony', weight: int = 1) -> None:
        """Record usage of ``colony`` and promote detail if it becomes a hotspot."""

        idx = self._county_index(colony.x, colony.y)
        state = self._detail_states.get(idx)
        if state is None:
            return
        state.usage += max(1, weight)
        if state.level != "fine" and state.usage >= self._detail_threshold:
            state.level = "fine"
            self._ensure_county_detail(idx, "fine")
            state.usage = max(state.usage, self._detail_threshold // 2)

    def set_detail_level(self, idx: Tuple[int, int], level: str) -> None:
        """Force county ``idx`` to use ``level`` detail going forward."""

        if level not in {"coarse", "fine"}:
            raise ValueError("level must be 'coarse' or 'fine'")
        state = self._detail_states[idx]
        state.level = level
        self._ensure_county_detail(idx, level)

    def set_global_detail(self, level: str) -> None:
        """Apply the same detail level to every county on the planet."""

        for idx in self._detail_states:
            self.set_detail_level(idx, level)

    def promote_dense_region(self) -> bool:
        """Increase detail for the busiest remaining county if possible."""

        # Promote the county with the highest recorded usage that is not yet fine.
        candidates = sorted(
            ((idx, state) for idx, state in self._detail_states.items() if state.level != "fine"),
            key=lambda item: (-item[1].usage, item[0]),
        )
        if not candidates:
            return False
        idx, state = candidates[0]
        state.level = "fine"
        self._ensure_county_detail(idx, "fine")
        state.usage = max(state.usage, self._detail_threshold // 2)
        return True

    def process_turn(self) -> None:
        if self.plague_level > 0:
            self.plague_level = max(0.0, self.plague_level - 0.01)
        if self.radiation_level > 0:
            self.radiation_level = max(0.0, self.radiation_level - 0.005)
        for state in self._detail_states.values():
            if state.usage > 0:
                state.usage -= 1
        
        # Regenerate resources up to biome-defined caps
        from .terrain import BIOME_RESOURCES
        biome_res = BIOME_RESOURCES.get(self.biome, {})
        for resource, rng in biome_res.items():
            cap = float(rng[1])
            current = self.resources.get(resource, 0.0)
            if current < cap:
                regen = cap * 0.05  # 5% of cap per turn
                self.resources[resource] = min(cap, current + regen)
    def _compute_distance(
        self, a: Tuple[int, int, str], b: Tuple[int, int, str]
    ) -> float:
        if APPROXIMATE:
            ax, ay, _ = a
            bx, by, _ = b
            return distance((ax, ay), (bx, by))
        return self.routes.shortest_path(a, b)

    def _mark_dirty(self) -> None:
        self._routes_dirty = True
        if self._distance_cache:
            self._distance_cache.cache_clear()

    def get_county(self, x: int, y: int) -> 'County | None':
        ix = x // self.county_size
        iy = y // self.county_size
        return self.counties.get((ix, iy))

    @property
    def rural_population(self) -> int:
        """Return total non-city population across all counties."""
        return sum(c.rural_population for c in self.counties.values())

    def get_route_distance(self, a, b) -> float:
        self._ensure_city_node(a)
        self._ensure_city_node(b)
        if a[2] == "city":
            colony = self.colonies.get((a[0], a[1]))
            if colony:
                self.register_colony_usage(colony)
        if b[2] == "city":
            colony = self.colonies.get((b[0], b[1]))
            if colony:
                self.register_colony_usage(colony)
        if self._routes_dirty or self._cache_version != self.routes.version:
            self._distance_cache.cache_clear()
            self._routes_dirty = False
            self._cache_version = self.routes.version
        return self._distance_cache(a, b)

    def get_travel_time(self, a, b) -> float:
        """Return surface travel time in days between nodes ``a`` and ``b``."""

        dist = self.get_route_distance(a, b)
        return travel_time(dist)

    def get_space_travel_time(self, other: 'Planet') -> float:
        """Return interplanetary travel time in days to ``other`` planet."""

        dist = distance(self.coords, other.coords)
        return travel_time(dist, space=True)

    def extract_resource(self, kind: str, amount: float) -> float:
        """Remove up to ``amount`` of ``kind`` from planetary stores.

        Parameters
        ----------
        kind:
            Resource identifier such as ``"metal"`` or ``"food"``.
        amount:
            Requested amount to extract.

        Returns
        -------
        float
            The quantity actually extracted.
        """

        available = self.resources.get(kind, 0.0)
        taken = min(available, amount)
        self.resources[kind] = available - taken
        return taken

    def add_base(self, base: 'MilitaryBase') -> None:
        self.bases[(base.x, base.y)] = base
        node = (base.x, base.y, "base")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((base.x - nx) ** 2 + (base.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_mine(self, mine: 'Mine') -> None:
        self.mines[(mine.x, mine.y)] = mine
        node = (mine.x, mine.y, "mine")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((mine.x - nx) ** 2 + (mine.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_farm(self, farm: 'Farm') -> None:
        self.farms[(farm.x, farm.y)] = farm
        node = (farm.x, farm.y, "farm")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((farm.x - nx) ** 2 + (farm.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_port(self, port: 'Port') -> None:
        self.ports[(port.x, port.y)] = port
        node = (port.x, port.y, "port")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((port.x - nx) ** 2 + (port.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_factory(self, fac: 'Factory') -> None:
        self.factories[(fac.x, fac.y)] = fac
        node = (fac.x, fac.y, "factory")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "factory", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((fac.x - nx) ** 2 + (fac.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_hospital(self, hos: 'Hospital') -> None:
        self.hospitals[(hos.x, hos.y)] = hos
        node = (hos.x, hos.y, "hospital")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "factory", "hospital", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((hos.x - nx) ** 2 + (hos.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_shipyard(self, yard: 'Shipyard') -> None:
        self.shipyards[(yard.x, yard.y)] = yard
        node = (yard.x, yard.y, "shipyard")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "factory", "hospital", "shipyard", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((yard.x - nx) ** 2 + (yard.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_school(self, school: 'School') -> None:
        self.schools[(school.x, school.y)] = school
        node = (school.x, school.y, "school")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "factory", "hospital", "shipyard", "school", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((school.x - nx) ** 2 + (school.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_power_plant(self, plant: 'PowerPlant') -> None:
        self.power_plants[(plant.x, plant.y)] = plant
        node = (plant.x, plant.y, "power")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "factory", "hospital", "shipyard", "school", "power", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((plant.x - nx) ** 2 + (plant.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_lab(self, lab: 'ResearchLab') -> None:
        self.labs[(lab.x, lab.y)] = lab
        node = (lab.x, lab.y, "lab")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "factory", "hospital", "shipyard", "school", "power", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((lab.x - nx) ** 2 + (lab.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_spaceport(self, port: 'Spaceport') -> None:
        """Add a spaceport enabling interplanetary travel."""
        self.spaceports[(port.x, port.y)] = port
        node = (port.x, port.y, "spaceport")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "factory", "hospital", "shipyard", "school", "power", "spaceport", "lab"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((port.x - nx) ** 2 + (port.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_nuke_facility(self, fac: 'NuclearFacility') -> None:
        """Add a nuclear weapon production site."""
        self.nuke_plants[(fac.x, fac.y)] = fac
        node = (fac.x, fac.y, "nuke")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "mine", "port", "factory", "power", "lab", "nuke"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((fac.x - nx) ** 2 + (fac.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def add_orbital_defense(self, od: 'OrbitalDefense') -> None:
        """Add an orbital defense installation."""
        self.orbital_defenses[(od.x, od.y)] = od
        node = (od.x, od.y, "orbital")
        self.routes.add_node(node)
        for (nx, ny, t) in list(self.routes.nodes()):
            if t not in {"city", "base", "orbital", "nuke"}:
                continue
            if (nx, ny, t) == node:
                continue
            dist = ((od.x - nx) ** 2 + (od.y - ny) ** 2) ** 0.5
            if dist <= self.spacing * 2:
                self.routes.add_edge(node, (nx, ny, t), dist)
        self._mark_dirty()

    def upgrade_colony(self, colony: 'Colony', owner: int) -> 'City':
        """Replace ``colony`` with a fully fledged city."""
        city = City(colony.x, colony.y, self.name, owner=owner, population=colony.population)
        self.cities[(colony.x, colony.y)] = city
        self.colonies.pop((colony.x, colony.y), None)
        county = self.get_county(colony.x, colony.y)
        if county is not None:
            county.cities.append(city)
            if county.owner is None:
                county.owner = owner
        return city
