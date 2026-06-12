from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .city import City
from ...society.ideas import Idea


# Rural demographic constants.  Rural areas have higher base fertility than
# settled cities (traditional high-birth agrarian communities) but no hospital
# access, so natural mortality is slightly higher than urban.
_RURAL_BASE_BIRTH: float = 0.025   # per turn; higher than settled urban (0.020)
_RURAL_BASE_DEATH: float = 0.013   # slightly higher natural mortality than cities (no hospitals)
_RURAL_STARVE_THRESH: float = 0.30 # food_ratio below which starvation sets in
_RURAL_STARVE_MAX: float   = 0.03  # maximum starvation mortality
_RURAL_INSTABILITY_SCALE: float = 0.005  # max extra mortality per turn at stability=0


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

    def process_rural_turn(
        self,
        food_ratio: float = 1.0,
        stability: float = 75.0,
    ) -> None:
        """Advance rural population by one simulation turn.

        Rural communities have lower birth rates than cities (less access to
        healthcare and economic opportunity) but are also more resilient to
        mild food shortages as subsistence farmers.

        Parameters
        ----------
        food_ratio:
            Planet food level normalised to 1.0 = fully adequate.  Below
            :data:`_RURAL_STARVE_THRESH` starvation mortality begins to climb.
        stability:
            Owner nation's stability (0–100).  Civil unrest suppresses births.
        """
        if self.rural_population <= 0:
            return

        stab_mod    = 0.5 + 0.5 * (stability / 100.0)
        food_birth  = 0.4 + 0.6 * food_ratio  # less sensitive than urban
        birth_rate  = _RURAL_BASE_BIRTH * stab_mod * food_birth

        starve_mort = max(
            0.0,
            (_RURAL_STARVE_THRESH - food_ratio) * _RURAL_STARVE_MAX / _RURAL_STARVE_THRESH,
        )
        instability_mort = max(0.0, (50.0 - stability) / 50.0 * _RURAL_INSTABILITY_SCALE)
        total_mort  = _RURAL_BASE_DEATH + starve_mort + instability_mort

        net = birth_rate - total_mort
        self.rural_population = max(0, int(self.rural_population * (1.0 + net)))
