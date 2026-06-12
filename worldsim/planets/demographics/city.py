from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

from ...core.backend import _np
from ...society.ideas import Idea
from ..registry import PLANETS
from ...core.memory import iter_memory_batches

__all__ = ['City', 'process_city_batch', 'process_rural_migration']


# ---------------------------------------------------------------------------
# Demographic model constants
# ---------------------------------------------------------------------------

# Per-turn birth/death rates (1 turn ≈ 20 simulation years).
_BASE_BIRTH: float = 0.020      # baseline birth contribution (settled urban; lower than rural)
_FRONTIER_BONUS: float = 0.015  # extra births at very low density (pioneer effect)
_BASE_DEATH: float = 0.010      # baseline natural mortality

# Hospital effect: each hospital reduces mortality by this fraction (stacks diminishingly).
_HOSP_MORT_SCALE: float = 0.15

# Plague mortality is super-linear so high plague levels are far more lethal
# than low ones.  Hospitals also reduce this via healthcare_factor.
_DISEASE_SCALE: float = 0.15
_DISEASE_POW: float = 1.5

# Radiation: infrastructure damage + mortality per unit radiation level.
_RAD_MORT_SCALE: float = 0.08
# Radiation also impairs fertility; reduces birth rate (floor: 50 % of normal).
_RAD_FERT_SCALE: float = 0.40

# Instability: civil unrest below stability=50 adds direct mortality (violence,
# supply-chain collapse, healthcare breakdown).
_INSTABILITY_THRESH: float = 50.0
_INSTABILITY_MORT_SCALE: float = 0.005  # max extra mortality per turn at stability=0

# Famine: below this food_ratio (0–1) starvation mortality kicks in.
_STARVE_THRESH: float = 0.35
_STARVE_MORT_MAX: float = 0.04  # max starvation mortality when food_ratio → 0

# Overcrowding: density above this threshold adds squalor/disease mortality.
_CROWD_THRESH: float = 0.90
_CROWD_MORT_SCALE: float = 0.04

# Infrastructure dynamics: target-based adjustment toward economic equilibrium.
_INFRA_MAX: float = 50.0        # hard ceiling — no mega-city sprawl
_INFRA_ECON_SCALE: float = 20.0 # target_infra = econ_health * ECON_SCALE
_INFRA_ADJUST_RATE: float = 0.03  # fraction of gap closed per turn

# Food adequacy reference: a planet with this many stored food units is
# considered fully adequate (food_ratio = 1.0).  Set to the mid-range of the
# harshest biomes (ice/volcanic floor at 10) so starting conditions are never
# below the starvation threshold.
_FOOD_ADEQUATE: float = 20.0


@dataclass
class City:
    x: int
    y: int
    planet: str
    infrastructure: float = 1.0
    owner: Optional[int] = None
    population: int = 5000
    industry: float = 1.0
    commerce: float = 1.0
    services: float = 1.0
    ideas: Dict[str, Idea] = field(default_factory=dict)

    def dominant_idea(self) -> str | None:
        """Return the name of the strongest idea in this city."""
        if not self.ideas:
            return None
        return max(self.ideas.values(), key=lambda i: i.followers).name

    def process_turn(
        self,
        plague_resist: float = 0.0,
        stability: float = 75.0,
        hospital_count: int = 0,
        econ_health: float = 0.5,
    ) -> None:
        """Advance one turn using the birth/death demographic model."""
        planet = PLANETS.get(self.planet)
        plague_level = planet.plague_level if planet else 0.0
        radiation_level = planet.radiation_level if planet else 0.0
        food_ratio = min(1.0, (planet.resources.get('food', _FOOD_ADEQUATE) / _FOOD_ADEQUATE)) if planet else 1.0

        capacity = self.infrastructure * 20_000.0
        density = self.population / max(capacity, 1.0)

        # Birth rate
        frontier = max(0.0, _FRONTIER_BONUS * (1.0 - min(density, 1.0) * 1.5))
        stab_mod = 0.5 + 0.5 * (stability / 100.0)
        food_birth = 0.3 + 0.7 * food_ratio
        rad_fert = max(0.5, 1.0 - radiation_level * _RAD_FERT_SCALE)
        birth_rate = (_BASE_BIRTH + frontier) * stab_mod * food_birth * rad_fert

        # Death rate
        hc_factor = 1.0 / (1.0 + hospital_count * _HOSP_MORT_SCALE)
        natural_mort = _BASE_DEATH * hc_factor
        disease_mort = (plague_level ** _DISEASE_POW) * _DISEASE_SCALE * (1.0 - plague_resist) * hc_factor
        rad_mort = radiation_level * _RAD_MORT_SCALE
        starve_mort = max(0.0, (_STARVE_THRESH - food_ratio) * _STARVE_MORT_MAX / _STARVE_THRESH)
        crowd_mort = max(0.0, (density - _CROWD_THRESH) * _CROWD_MORT_SCALE)
        instability_mort = max(0.0, (_INSTABILITY_THRESH - stability) / _INSTABILITY_THRESH * _INSTABILITY_MORT_SCALE)
        total_mort = natural_mort + disease_mort + rad_mort + starve_mort + crowd_mort + instability_mort

        net = birth_rate - total_mort
        self.population = max(0, int(self.population * (1.0 + net)))

        # Infrastructure: target-based adjustment
        if radiation_level > 0:
            self.infrastructure = max(1.0, self.infrastructure * (1.0 - radiation_level * 0.1))
        target = min(_INFRA_MAX, econ_health * _INFRA_ECON_SCALE)
        self.infrastructure = max(1.0, min(_INFRA_MAX, self.infrastructure + (target - self.infrastructure) * _INFRA_ADJUST_RATE))

    @property
    def economic_output(self) -> float:
        planet = PLANETS.get(self.planet)
        richness = planet.resource_richness if planet else 1.0
        sector_bonus = (self.industry + self.commerce + self.services) / 3
        return self.population * self.infrastructure / 1000 * richness * sector_bonus

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    @property
    def world_coords(self) -> Tuple[float, float, float]:
        """Return absolute coordinates for this city including its planet."""
        planet = PLANETS.get(self.planet)
        if planet:
            return planet.x + self.x, planet.y + self.y, planet.z
        return float(self.x), float(self.y), 0.0

    def upgrade(
        self,
        level: int = 1,
        *,
        industry: bool = False,
        commerce: bool = False,
        services: bool = False,
    ) -> None:
        """Improve city attributes by ``level``."""
        if industry:
            self.industry += level * 0.1
        elif commerce:
            self.commerce += level * 0.1
        elif services:
            self.services += level * 0.1
        else:
            self.infrastructure += float(level)


#: Rough per-city working-set during the vectorised update: ~20 float64
#: columns plus temporaries.
_CITY_ITEM_NBYTES: int = 2_048


def process_city_batch(
    cities: List[City],
    plague_level: float,
    radiation_level: float,
    plague_resist: float,
    tech_bonus: float,
    growth_mult: float = 1.0,
    cap_mult: float = 1.0,
    stability: float = 75.0,
    hospital_count: int = 0,
    econ_health: float = 0.5,
) -> Tuple[int, float]:
    """Memory-budgeted driver around :func:`_process_city_chunk`.

    Splits very large city lists into chunks sized from the available RAM so
    the vectorised update never allocates beyond its budget; results are
    summed across chunks.
    """
    total_pop = 0
    total_econ = 0.0
    for chunk in iter_memory_batches(
        cities, _CITY_ITEM_NBYTES, fraction=0.10, lo=512
    ):
        pop, econ = _process_city_chunk(
            chunk, plague_level, radiation_level, plague_resist, tech_bonus,
            growth_mult, cap_mult,
            stability=stability,
            hospital_count=hospital_count,
            econ_health=econ_health,
        )
        total_pop += pop
        total_econ += econ
    return total_pop, total_econ


def _process_city_chunk(
    cities: List[City],
    plague_level: float,
    radiation_level: float,
    plague_resist: float,
    tech_bonus: float,
    growth_mult: float = 1.0,
    cap_mult: float = 1.0,
    stability: float = 75.0,
    hospital_count: int = 0,
    econ_health: float = 0.5,
) -> Tuple[int, float]:
    """Vectorised demographic update for a group of cities.

    Uses a birth-rate / death-rate model in place of a single logistic growth
    rate, capturing the independent effects of:

    * **Food security** (derived per city from the planet's stored food).
    * **Healthcare** (``hospital_count`` reduces natural and plague mortality).
    * **Stability** (instability suppresses births).
    * **Overcrowding** (density above 90 % of carrying capacity adds squalor
      mortality).
    * **Infrastructure** (target-based economic adjustment replaces the old
      random growth/decay — wealthy nations invest; struggling nations see
      their cities slowly deteriorate).

    ``growth_mult`` amplifies the birth rate (biology tech: Pharmacology).
    ``cap_mult`` scales carrying capacity (biology tech: Genetic Engineering).
    ``econ_health`` (0–1) sets the infrastructure equilibrium target.
    """
    if not cities:
        return 0, 0.0

    if _np is not None:
        n = len(cities)
        pops  = _np.array([c.population     for c in cities], dtype=float)
        infra = _np.array([c.infrastructure  for c in cities], dtype=float)

        # ── Per-planet lookup (food level + resource richness) ─────────────
        _planet_cache: Dict[str, tuple] = {}
        for city in cities:
            if city.planet not in _planet_cache:
                p = PLANETS.get(city.planet)
                if p is not None:
                    _planet_cache[city.planet] = (
                        p.resources.get('food', _FOOD_ADEQUATE),
                        p.resource_richness,
                    )
                else:
                    _planet_cache[city.planet] = (_FOOD_ADEQUATE, 1.0)

        food_arr    = _np.array([min(1.0, _planet_cache[c.planet][0] / _FOOD_ADEQUATE) for c in cities], dtype=float)
        richness_arr = _np.array([_planet_cache[c.planet][1] for c in cities], dtype=float)
        sector_arr  = _np.array([(c.industry + c.commerce + c.services) / 3 for c in cities], dtype=float)

        # ── Carrying capacity ──────────────────────────────────────────────
        capacity = infra * 20_000.0 * cap_mult
        density  = pops / _np.maximum(capacity, 1.0)

        # ── Birth rate ─────────────────────────────────────────────────────
        # Frontier bonus: low-density settlements grow faster (pioneer effect).
        frontier   = _np.maximum(0.0, _FRONTIER_BONUS * (1.0 - _np.minimum(density, 1.0) * 1.5))
        # Stability: civil war / unrest suppresses births (0.5 floor).
        stab_mod   = 0.5 + 0.5 * (stability / 100.0)
        # Food: famine reduces birth rate toward a hard floor of 0.3.
        food_birth = 0.3 + 0.7 * food_arr
        # Radiation: impairs fertility (floor at 50 % of normal birth rate).
        rad_fert   = max(0.5, 1.0 - radiation_level * _RAD_FERT_SCALE)
        birth_rate = (_BASE_BIRTH + frontier) * stab_mod * food_birth * growth_mult * rad_fert

        # ── Death rate ─────────────────────────────────────────────────────
        # Hospitals reduce both natural mortality and plague mortality.
        hc_factor     = 1.0 / (1.0 + hospital_count * _HOSP_MORT_SCALE)
        natural_mort  = _BASE_DEATH * hc_factor
        # Plague: super-linear so high plague levels cause collapse, not just slow decline.
        disease_mort  = (plague_level ** _DISEASE_POW) * _DISEASE_SCALE * (1.0 - plague_resist) * hc_factor
        # Radiation: flat mortality independent of healthcare.
        rad_mort      = radiation_level * _RAD_MORT_SCALE
        # Starvation: kicks in below 50 % food adequacy, scales to max at 0 %.
        starve_mort   = _np.maximum(0.0, (_STARVE_THRESH - food_arr) * (_STARVE_MORT_MAX / _STARVE_THRESH))
        # Overcrowding: density above 90 % → disease, squalor.
        crowd_mort    = _np.maximum(0.0, (density - _CROWD_THRESH) * _CROWD_MORT_SCALE)
        # Instability: civil war / collapse raises mortality below stability=50.
        instability_mort = max(0.0, (_INSTABILITY_THRESH - stability) / _INSTABILITY_THRESH * _INSTABILITY_MORT_SCALE)

        total_mort = natural_mort + disease_mort + rad_mort + starve_mort + crowd_mort + instability_mort

        # ── Net population change ──────────────────────────────────────────
        net_rate = birth_rate - total_mort
        pops = _np.maximum(0.0, pops * (1.0 + net_rate))

        # ── Infrastructure dynamics ────────────────────────────────────────
        if radiation_level > 0:
            infra = _np.maximum(1.0, infra * (1.0 - radiation_level * 0.1))
        # Economic equilibrium: rich nations invest, poor ones see slow decay.
        # target_infra = econ_health * SCALE (capped at _INFRA_MAX).
        target_infra = min(_INFRA_MAX, econ_health * _INFRA_ECON_SCALE)
        infra = _np.maximum(1.0, _np.minimum(_INFRA_MAX, infra + (target_infra - infra) * _INFRA_ADJUST_RATE))

        # Write back updated state.
        total_pop = int(pops.sum())
        for c, p, inf in zip(cities, pops.tolist(), infra.tolist()):
            c.population    = int(p)
            c.infrastructure = float(inf)

        # ── Economic output ────────────────────────────────────────────────
        pop_infra = _np.array([c.population * c.infrastructure / 1000.0 for c in cities], dtype=float)
        econ      = pop_infra * richness_arr * sector_arr
        total_econ = float(econ.sum() * tech_bonus)
        return total_pop, total_econ

    # ── Scalar fallback (no NumPy / CuPy) ─────────────────────────────────
    total_pop  = 0
    total_econ = 0.0
    richness_cache: Dict[str, float] = {}
    for city in cities:
        city.process_turn(plague_resist, stability=stability,
                          hospital_count=hospital_count, econ_health=econ_health)
        total_pop += city.population
        pname = city.planet
        if pname not in richness_cache:
            p = PLANETS.get(pname)
            richness_cache[pname] = p.resource_richness if p else 1.0
        richness = richness_cache[pname]
        sector   = (city.industry + city.commerce + city.services) / 3
        total_econ += city.population * city.infrastructure / 1000.0 * richness * sector * tech_bonus
    return total_pop, total_econ


def process_rural_migration(cities: List[City]) -> None:
    """Migrate rural residents into cities that have spare capacity.

    For each city on its planet, the county that contains the city is
    consulted.  When the city is below 70 % of carrying capacity, rural
    residents migrate in at a rate proportional to the available headroom.
    When the city is over 95 % capacity, a small overflow moves back to
    the county's rural pool.

    Population is conserved: every person gained by a city is removed from
    its county's rural pool, and vice-versa.  The function modifies city and
    county objects in-place.
    """
    for city in cities:
        planet = PLANETS.get(city.planet)
        if planet is None:
            continue
        county = planet.get_county(city.x, city.y)
        if county is None or county.owner != city.owner:
            continue

        capacity = city.infrastructure * 20_000.0
        density  = city.population / max(capacity, 1.0)

        if density < 0.70 and county.rural_population > 50:
            # Pull factor: the emptier the city, the stronger the pull.
            pull_rate  = 0.025 * (1.0 - density)
            migrants   = max(1, int(county.rural_population * pull_rate))
            migrants   = min(migrants, county.rural_population)
            county.rural_population -= migrants
            city.population         += migrants

        elif density > 0.95 and city.population > 500:
            # Overcrowding: a small fraction spills back to rural areas.
            overflow = int(city.population * (density - 0.95) * 0.03)
            city.population         = max(0, city.population - overflow)
            county.rural_population += overflow
