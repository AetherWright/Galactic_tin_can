"""Star system model and global registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from ..utils import distance, travel_time
from ..planets import PLANETS

if TYPE_CHECKING:
    from .nation import Nation


@dataclass(slots=True)
class Star:
    """A star system linking several planets."""

    name: str
    planet_names: List[str]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    cluster: int = 0
    owner: Optional[int] = None

    def distance_to(self, other: "Star") -> float:
        """Return Euclidean distance to ``other`` star."""
        return distance((self.x, self.y, self.z), (other.x, other.y, other.z))

    def travel_time_to(self, other: "Star") -> float:
        """Return travel time in days to ``other`` star."""
        return travel_time(self.distance_to(other), space=True)

    def update_owner(self, nations: Dict[int, "Nation"]) -> None:
        counts: Dict[int, int] = {}
        for pname in self.planet_names:
            planet = PLANETS.get(pname)
            if not planet:
                continue
            pop_by_owner: Dict[int, int] = {}
            for city in planet.cities.values():
                if city.owner is not None:
                    pop_by_owner[city.owner] = pop_by_owner.get(city.owner, 0) + city.population
            if not pop_by_owner:
                continue
            owner = max(pop_by_owner.items(), key=lambda kv: kv[1])[0]
            counts[owner] = counts.get(owner, 0) + 1
        new_owner = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
        if new_owner != self.owner:
            self.owner = new_owner


STARS: Dict[str, Star] = {}
