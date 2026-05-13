from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .city import City
from ..ideas import Idea

@dataclass
class County:
    """Simple territorial division containing multiple cities.

    ``rural_population`` represents residents living outside city limits.
    ``solidarity`` approximates the cohesion of this rural group as a ratio of
    rural population to total population.
    """

    name: str
    x: int
    y: int
    width: int
    height: int
    owner: Optional[int] = None
    rural_population: int = 0
    cities: List['City'] = field(default_factory=list)
    ideas: Dict[str, Idea] = field(default_factory=dict)

    @property
    def population(self) -> int:
        return self.rural_population + sum(c.population for c in self.cities)

    @property
    def economy(self) -> float:
        return sum(c.economic_output for c in self.cities)

    @property
    def solidarity(self) -> float:
        """Return rural solidarity as the rural population share."""
        total = self.population
        if total == 0:
            return 0.0
        return self.rural_population / total

    def dominant_idea(self) -> str | None:
        """Return the name of the strongest idea in this county."""
        if not self.ideas:
            return None
        return max(self.ideas.values(), key=lambda i: i.followers).name
