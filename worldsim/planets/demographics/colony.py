from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from ...core.backend import _np

__all__ = ["Colony", "process_colony_batch"]


# ---------------------------------------------------------------------------
# Colony demographic constants
# ---------------------------------------------------------------------------
# Colonies are pioneer settlements: higher frontier birth bonus and higher
# carrying capacity relative to their infrastructure than established cities,
# but also higher base mortality (harsher living conditions, less healthcare).

_COL_BASE_BIRTH: float  = 0.030   # slightly higher fertility than established cities
_COL_FRONTIER:   float  = 0.020   # strong pioneer bonus at low density
_COL_BASE_DEATH: float  = 0.014   # higher baseline mortality (frontier conditions)
_COL_DISEASE_POW: float = 1.5     # same super-linear plague response as cities
_COL_DISEASE_SCALE: float = 0.15
_COL_STARVE_THRESH: float = 0.30  # colonies are more resilient to mild food stress
_COL_STARVE_MAX: float   = 0.04
_COL_RAD_MORT_SCALE: float = 0.08  # radiation mortality — same scale as cities
_COL_RAD_FERT_SCALE: float = 0.40  # radiation fertility penalty (floor: 50 %)
_COL_CAP: int            = 8_000  # colony population ceiling before city graduation
_COL_FOOD_ADEQUATE: float = 20.0  # food units = fully adequate (matches city constant)


@dataclass
class Colony:
    """Early settlement that can grow into a city."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    population: int = 1000

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def process_turn(self, plague_resist: float = 0.0, stability: float = 75.0) -> None:
        """Advance one turn using the pioneer-settlement birth/death model."""
        planet = PLANETS.get(self.planet)
        plague_level     = planet.plague_level     if planet else 0.0
        radiation_level  = planet.radiation_level  if planet else 0.0
        food_ratio       = min(1.0, planet.resources.get('food', _COL_FOOD_ADEQUATE) / _COL_FOOD_ADEQUATE) if planet else 1.0

        density     = self.population / _COL_CAP
        frontier    = max(0.0, _COL_FRONTIER * (1.0 - min(density, 1.0) * 1.5))
        stab_mod    = 0.5 + 0.5 * (stability / 100.0)
        food_birth  = 0.3 + 0.7 * food_ratio
        rad_fert    = max(0.5, 1.0 - radiation_level * _COL_RAD_FERT_SCALE)
        birth_rate  = (_COL_BASE_BIRTH + frontier) * stab_mod * food_birth * rad_fert

        disease_mort = (plague_level ** _COL_DISEASE_POW) * _COL_DISEASE_SCALE * (1.0 - plague_resist)
        rad_mort     = radiation_level * _COL_RAD_MORT_SCALE
        starve_mort  = max(0.0, (_COL_STARVE_THRESH - food_ratio) * _COL_STARVE_MAX / _COL_STARVE_THRESH)
        total_mort   = _COL_BASE_DEATH + disease_mort + rad_mort + starve_mort

        net = birth_rate - total_mort
        self.population = max(0, min(_COL_CAP, int(self.population * (1.0 + net))))


from ..registry import PLANETS


def process_colony_batch(
    colonies: List[Colony],
    plague_level: float,
    plague_resist: float,
    stability: float = 75.0,
    radiation_level: float = 0.0,
) -> int:
    """Vectorised birth/death update for pioneer colonies.

    Derives per-colony food ratios from each colony's host planet so that
    colonies on food-poor worlds (desert, volcanic) face harsher conditions
    than those on fertile continental or forest planets.

    Once a colony reaches the :data:`_COL_CAP` threshold it stops growing
    here; the civilian AI's ``build_city`` action is responsible for upgrading
    mature colonies to full cities.
    """
    if not colonies:
        return 0

    if _np is not None:
        pops = _np.array([c.population for c in colonies], dtype=float)

        # Per-colony food ratio from planet resources.
        def _col_food(planet_name: str) -> float:
            p = PLANETS.get(planet_name)
            return p.resources.get('food', _COL_FOOD_ADEQUATE) if p is not None else _COL_FOOD_ADEQUATE

        food_arr = _np.array(
            [min(1.0, _col_food(c.planet) / _COL_FOOD_ADEQUATE) for c in colonies],
            dtype=float,
        )

        density    = pops / _COL_CAP
        frontier   = _np.maximum(0.0, _COL_FRONTIER * (1.0 - _np.minimum(density, 1.0) * 1.5))
        stab_mod   = 0.5 + 0.5 * (stability / 100.0)
        food_birth = 0.3 + 0.7 * food_arr
        rad_fert   = max(0.5, 1.0 - radiation_level * _COL_RAD_FERT_SCALE)
        birth_rate = (_COL_BASE_BIRTH + frontier) * stab_mod * food_birth * rad_fert

        disease_mort = (plague_level ** _COL_DISEASE_POW) * _COL_DISEASE_SCALE * (1.0 - plague_resist)
        rad_mort     = radiation_level * _COL_RAD_MORT_SCALE
        starve_mort  = _np.maximum(0.0, (_COL_STARVE_THRESH - food_arr) * _COL_STARVE_MAX / _COL_STARVE_THRESH)
        total_mort   = _COL_BASE_DEATH + disease_mort + rad_mort + starve_mort

        net  = birth_rate - total_mort
        pops = _np.minimum(_COL_CAP, _np.maximum(0.0, pops * (1.0 + net)))

        for c, p in zip(colonies, pops.tolist()):
            c.population = int(p)
        return int(pops.sum())

    # Scalar fallback
    total = 0
    for c in colonies:
        c.process_turn(plague_resist, stability=stability)
        total += c.population
    return total
