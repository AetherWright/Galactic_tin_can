"""Great-filter calamities, runaway-nation damping and zombie culling."""

from __future__ import annotations

import random
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..nations import Nation


# Rough ceilings used to detect runaway nations before numeric overflow.
# When a nation exceeds any threshold it is gently scaled back rather than
# waiting for the periodic great filter.
SOFT_FILTER_POP: int = 10_000_000_000
SOFT_FILTER_ECON: float = 1_000_000_000_000.0
SOFT_FILTER_MIL: float = 1_000_000.0


def _apply_soft_great_filter(nation: "Nation") -> None:
    """Gently rein in a nation that has grown too powerful."""

    nation.population = max(1000, int(nation.population * 0.85))
    nation.military *= 0.85
    nation.economy_linear *= 0.85
    nation.stability = max(10.0, nation.stability * 0.9)


def _maybe_apply_soft_great_filter(nation: "Nation") -> None:
    """Apply the soft great filter if *nation* exceeds overflow thresholds."""

    if (
        nation.population > SOFT_FILTER_POP
        or nation.economy_linear > SOFT_FILTER_ECON
        or nation.military > SOFT_FILTER_MIL
    ):
        _apply_soft_great_filter(nation)

def _cull_zombie_nations(nations: Dict[int, "Nation"], max_nations: int = 50) -> None:
    """Remove micro-states that are too small to be meaningful."""
    # First pass — instant kill genuinely dead nations
    for nid, n in list(nations.items()):
        if n.population < 2000 and len(n.cities) == 0:
            del nations[nid]
    
    # Second pass — if still over cap, absorb weakest into strongest neighbor
    if len(nations) <= max_nations:
        return
    
    sorted_nations = sorted(nations.values(), key=lambda n: n.population)
    while len(nations) > max_nations and sorted_nations:
        weakest = sorted_nations.pop(0)
        if weakest.id not in nations:
            continue
        # Find strongest neighbor by military
        absorber = max(
            (n for n in nations.values() if n.id != weakest.id),
            key=lambda n: n.military,
            default=None,
        )
        if absorber is None:
            break
        # Transfer cities and population
        for city in weakest.cities:
            city.owner = absorber.id
            absorber.cities.append(city)
        absorber.population += weakest.population
        # Clean up relations
        for n in nations.values():
            n.alliances.discard(weakest.id)
            n.trade_partners.discard(weakest.id)
            n.at_war.discard(weakest.id)
            n.relations.pop(weakest.id, None)
            n.border_pressure.pop(weakest.id, None)
        del nations[weakest.id]

def _apply_great_filter(nations: Dict[int, "Nation"]) -> None:
    """Severe calamity occurring every 20 centuries.

    Each nation suffers heavy losses and a chance of outright collapse.
    Nations can mitigate both via high stability or dedicated resilience
    programs, letting thoughtful play survive the filter.
    """

    for n in nations.values():
        resilience = getattr(n, "resilience", 0.0) / 100.0
        if n.population < 2000:
            n.population = 0
            continue
        stability = n.stability / 100.0
        collapse_chance = 0.1 * (1 - resilience) * (1 - stability)
        collapse_chance = max(0.01, collapse_chance)
        if random.random() < collapse_chance:
            n.population = 0
            continue
        keep = 0.5 + 0.3 * resilience + 0.1 * stability
        keep = min(0.95, keep)
        n.population = max(1000, int(n.population * keep))
        n.military *= keep
        n.economy_linear *= keep
        n.stability = max(5.0, n.stability * (0.7 + 0.2 * resilience))
