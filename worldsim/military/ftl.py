"""FTL drive registry and tech tree integration.

The entire FTL configuration lives in :data:`FTL_DRIVES`.  To add a new drive
tier, insert an entry there and (optionally) add its name to
:data:`FTL_TIER_ORDER`.  No other file needs to change.

Travel time for fleets is measured in simulation *fifths* (sub-turns within a
century).  A fleet's ``travel_turns_remaining`` counts down by 1 each fifth
until the fleet arrives.  The formula is::

    turns = max(1, ceil(distance_ly / speed_mult))

where ``distance_ly`` is the Euclidean distance between star coordinates
(treated as light-years) and ``speed_mult`` comes from the unlocked drive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..nations.nation import Nation
    from ..research.technology import TechnologyTree


# ---------------------------------------------------------------------------
# FTL drive specification
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FTLDrive:
    """Specification for one FTL technology tier.

    Parameters
    ----------
    name:
        Unique identifier — must match the key in :data:`FTL_DRIVES`.
    speed_mult:
        Divisor applied to raw light-year distance when computing fleet travel
        turns.  Higher values = faster travel.  ``1.0`` ≈ sub-light rockets.
    range_ly:
        Maximum jump / warp range per trip in light-years.  ``0.0`` means
        unlimited range (point-to-point across the galaxy).
    cooldown_turns:
        Fifths that must elapse between consecutive FTL jumps for this drive.
        ``0`` = no cooldown.
    prereqs:
        Names of :class:`~worldsim.research.technology.TechnologyNode` that must be
        unlocked *before* this drive can be researched.
    research_cost:
        Research-point cost for the tech-tree node that unlocks this drive.
    """

    name: str
    speed_mult: float = 1.0
    range_ly: float = 0.0
    cooldown_turns: int = 0
    prereqs: Set[str] = field(default_factory=set)
    research_cost: float = 100.0


# ---------------------------------------------------------------------------
# Registry — add new drives here only
# ---------------------------------------------------------------------------

FTL_DRIVES: Dict[str, FTLDrive] = {
    # Tier 0 — barely interplanetary; very slow, short-ranged
    "Chemical Rockets": FTLDrive(
        "Chemical Rockets",
        speed_mult=1.0,
        range_ly=0.5,
        cooldown_turns=3,
        prereqs={"Industrial Automation"},
        research_cost=80.0,
    ),
    # Tier 1 — efficient sub-light; reach nearby stars in reasonable time
    "Ion Drive": FTLDrive(
        "Ion Drive",
        speed_mult=5.0,
        range_ly=3.0,
        cooldown_turns=2,
        prereqs={"Chemical Rockets"},
        research_cost=120.0,
    ),
    # Tier 2 — first true FTL; warps space rather than brute-force thrust
    "Warp Drive": FTLDrive(
        "Warp Drive",
        speed_mult=20.0,
        range_ly=10.0,
        cooldown_turns=1,
        prereqs={"Ion Drive", "Atomic Engineering"},
        research_cost=200.0,
    ),
    # Tier 3 — mature FTL; routine interstellar commerce and warfare
    "Hyperdrive": FTLDrive(
        "Hyperdrive",
        speed_mult=80.0,
        range_ly=40.0,
        cooldown_turns=0,
        prereqs={"Warp Drive"},
        research_cost=300.0,
    ),
    # Tier 4 — instant jumps; range limited only by navigation precision
    "Jump Drive": FTLDrive(
        "Jump Drive",
        speed_mult=500.0,
        range_ly=0.0,       # unlimited
        cooldown_turns=0,
        prereqs={"Hyperdrive", "Nuclear Weapons"},
        research_cost=500.0,
    ),
}

# Ordered from slowest to fastest — used by best_ftl_drive()
FTL_TIER_ORDER: List[str] = [
    "Chemical Rockets",
    "Ion Drive",
    "Warp Drive",
    "Hyperdrive",
    "Jump Drive",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_ftl_drive(nation: "Nation") -> FTLDrive:
    """Return the highest-tier FTL drive ``nation`` has unlocked.

    Falls back to an implicit "no drive" baseline with ``speed_mult=0.5``
    (very slow interplanetary crawl) when nothing is researched yet.
    """
    unlocked = nation.tech_tree.unlocked
    best: Optional[FTLDrive] = None
    for name in FTL_TIER_ORDER:
        if name in unlocked:
            best = FTL_DRIVES[name]
    if best is not None:
        return best
    # No FTL at all — allow very slow sub-light travel within a single star
    return FTLDrive("None", speed_mult=0.5, range_ly=0.5, cooldown_turns=5)


def fleet_travel_turns(distance_ly: float, drive: FTLDrive) -> int:
    """Return the number of simulation fifths needed to cover ``distance_ly``.

    The result is always at least 1 so that the fleet ticks at least once
    before arriving (avoids instant zero-turn teleportation).
    """
    raw = distance_ly / max(0.01, drive.speed_mult)
    return max(1, math.ceil(raw))


def add_ftl_tech_nodes(tree: "TechnologyTree", nation: "Nation") -> None:
    """Inject all FTL drive tiers into ``tree`` as :class:`TechnologyNode` objects.

    Each node's *effect* sets ``nation.tech_bonuses["ftl_speed"]`` to the
    drive's ``speed_mult`` only if it is higher than what is already stored,
    so repeated unlocks never *reduce* travel speed.

    This function is idempotent — calling it multiple times is safe.
    """
    from ..research.technology import TechnologyNode

    for drive in FTL_DRIVES.values():
        if drive.name in tree.nodes:
            continue

        mult = drive.speed_mult

        def _effect(n: "Nation", m: float = mult) -> None:
            current = n.tech_bonuses.get("ftl_speed", 0.0)
            if m > current:
                n.tech_bonuses["ftl_speed"] = m

        node = TechnologyNode(
            name=drive.name,
            cost=drive.research_cost,
            prerequisites=drive.prereqs.copy(),
            effect=_effect,
        )
        tree.add_node(node)
