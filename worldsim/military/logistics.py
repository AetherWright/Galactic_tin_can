"""Ground-combat logistics: supply, readiness, attrition and stacking."""

from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..nations.nation import Nation
    from ..planets.planet import Planet
    from .divisions import Division


_REFERENCE_PLANET_AREA = 100 * 100

_TERRAIN_CAPACITY_DENSITY: Dict[str, float] = {
    "forest": 0.8,
    "desert": 1.1,
    "oceanic": 0.95,
    "ice": 0.7,
    "volcanic": 0.75,
}

_TERRAIN_DIFFUSION_BASE: Dict[str, float] = {
    "forest": 0.18,
    "desert": 0.1,
    "oceanic": 0.12,
    "ice": 0.22,
    "volcanic": 0.2,
}

_DOCTRINE_MODIFIERS: Dict[str, Dict[str, float]] = {
    "Balanced": {"attack": 1.0, "defend": 1.0, "reserve": 1.0},
    "Offensive": {"attack": 1.1, "defend": 0.95, "reserve": 0.9},
    "Defensive": {"attack": 0.95, "defend": 1.1, "reserve": 1.05},
    "Mobile": {"attack": 1.05, "defend": 0.98, "reserve": 1.0},
    "Guerrilla": {"attack": 0.98, "defend": 1.05, "reserve": 1.1},
}


def _doctrine_bias(doctrine: str, order: str) -> float:
    """Return a multiplicative factor for ``doctrine`` executing ``order``."""

    table = _DOCTRINE_MODIFIERS.get(doctrine)
    if not table:
        return 1.0
    return table.get(order, 1.0)


def _supply_throughput(nation: "Nation | None", planet: "Planet | None") -> float:
    """Estimate surface logistics throughput for ``nation`` on ``planet``."""

    if nation is None:
        return 0.75
    base = 0.55 + nation.infrastructure / 200.0
    tech_mod = nation.tech_bonuses.get("logistics", 0.0)
    resilience = nation.resilience / 150.0
    base += tech_mod + resilience
    if planet is None:
        return max(0.3, min(2.5, base))

    pname = planet.name
    def _count(items: List[object]) -> int:
        return sum(1 for obj in items if getattr(obj, "planet", None) == pname)

    logistic_nodes = _count(getattr(nation, "bases", []))
    ports = _count(getattr(nation, "ports", []))
    spaceports = _count(getattr(nation, "spaceports", []))
    cities = _count(getattr(nation, "cities", []))
    throughput = base
    throughput += 0.12 * logistic_nodes
    throughput += 0.08 * ports
    throughput += 0.05 * spaceports
    throughput += 0.03 * cities
    return max(0.3, min(2.5, throughput))


def _combat_readiness(nation: "Nation | None", supply: float) -> float:
    """Return a readiness multiplier influenced by morale and logistics."""

    if nation is None:
        return 1.0
    morale = (nation.stability + nation.resilience) / 200.0
    command_bonus = nation.tech_bonuses.get("command", 0.0)
    training_bonus = min(0.3, len(getattr(nation, "bases", [])) * 0.02)
    readiness = 0.65 + morale + command_bonus + training_bonus
    readiness *= 0.7 + 0.3 * min(1.5, supply)
    return max(0.4, min(2.0, readiness))


def _attrition_multiplier(engaged: float, supply: float) -> float:
    """Return casualty scaling factor based on engagement intensity."""

    base = 1.0 + max(0.0, 1.0 - supply)
    mitigation = 0.35 * engaged
    return max(0.35, base - mitigation)


def _planet_combat_profile(planet: "Planet | None") -> Tuple[float, float]:
    """Return stacking cap and diffusion baseline for ``planet``."""

    if planet is None:
        return 5.0, 0.12
    area = max(1.0, float(planet.width * planet.height))
    size_scale = (area / _REFERENCE_PLANET_AREA) ** 0.5
    terrain_key = planet.biome.lower()
    capacity_density = _TERRAIN_CAPACITY_DENSITY.get(terrain_key, 1.0)
    cap = max(1.0, capacity_density * size_scale * 5.0)
    base_diffusion = _TERRAIN_DIFFUSION_BASE.get(terrain_key, 0.12)
    # Larger planets allow looser formations, reducing diffusion pressure.
    diffusion = base_diffusion / max(0.5, size_scale ** 0.5)
    diffusion = min(0.65, max(0.05, diffusion))
    return cap, diffusion


def _stacked_power(
    divisions: List["Division"],
    planet: "Planet | None",
    tech_bonus: float,
    supply_throughput: float,
) -> Tuple[float, float]:
    """Compute effective power and engagement ratio for ``divisions``.

    The calculation applies diminishing returns for each additional division
    based on the planet's stacking cap.  A diffusion factor also reduces the
    portion of each division that can actively fight when formations become
    crowded.
    """

    if not divisions:
        return 0.0, 0.0
    cap, base_diffusion = _planet_combat_profile(planet)
    contributions = sorted(
        [
            div.power
            * tech_bonus
            * _doctrine_bias(div.doctrine, getattr(div, "order", div.posture))
            for div in divisions
        ],
        reverse=True,
    )
    density = len(contributions) / cap if cap else float(len(contributions))
    diffusion_loss = min(0.7, base_diffusion * max(1.0, density))
    logistics_boost = 0.7 + 0.3 * min(1.6, supply_throughput)
    engaged_ratio = max(0.2, min(1.0, (1.0 - diffusion_loss) * logistics_boost))
    effective_power = 0.0
    for idx, value in enumerate(contributions):
        crowding = (idx / cap) if cap else idx
        stack_penalty = 1.0 / (1.0 + crowding ** 1.25)
        effective_power += value * stack_penalty * engaged_ratio
    return effective_power, engaged_ratio
